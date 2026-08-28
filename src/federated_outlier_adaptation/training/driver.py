"""
driver.py - Federated Training Driver

Purpose:
    Provides the unified driver that runs one federated experiment (a
    scenario/metadata/aggregation combination) for a given trainer.  This is the
    glue layer between the low-level simulation logic (BaseConcurrentRunner,
    BaseSequentialRunner, aggregation methods) and the high-level experiment
    scripts (``base_fl``, ``all_aggregations``, ``final_experiments``,
    ``extreme_cases``).

Capabilities:
    - Accepts configuration for trainer, scenario, metadata, and aggregation.
    - Optional per-trainer hyperparameter overrides (``trainer_kwargs``).
    - Optional run seed; when given, outputs move under a ``seed_<n>`` folder.
    - Spawns inner process pools for method-level fan-out.
    - Records timing, communication volume and a full provenance ``config``
      block next to every result file.

Outputs:
    results/<parent_name>_<trainer>_grid_search/[seed_<n>/]<scenario>_<metadata>/
        accuracies_<index>.json        (per-job results)
        summary_<index>.json           (all methods summary)
        accuracies_over_rounds_*.png   (per-method plots)
        overlay_accuracies.png         (overlay comparison plot)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Optional

from federated_outlier_adaptation.aggregation.selector import list_method_names, select_class
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.plotting.round_plots import (
    plot_accuracies_over_rounds,
    plot_overlay_accuracies_per_scenario,
)
from federated_outlier_adaptation.utils import provenance
from federated_outlier_adaptation.utils.seeding import seed_suffix


#: File name of the server model a run is asked to keep.  Absent unless
#: ``--save-final-model`` was given, so no existing run writes one.
FINAL_MODEL_NAME = "final_model.pt"


# ----------------- helpers -----------------
def _to_list(v: Any):
    """Make JSON-safe (handles numpy)"""
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_list(x) for x in v]
    return v


def _optional_float(value):
    """
    ``float(value)``, or ``None`` when there is no value.

    The two pooled-oracle reference accuracies are absent until
    ``foa combined-train`` has run in the results root.  They annotate a run;
    they are not part of it, so a run that cannot find them stores ``null`` and
    keeps its own numbers rather than failing after the training is done.
    """
    return None if value is None else float(value)


def _normalize_accuracies(accuracies_any):
    """
    Accepts:
      - numpy array shape [R, 2]
      - list of [a, b] / (a, b)
    Returns: list[(clients_acc, global_acc), ...] as floats

    Either entry may be ``None`` and stays ``None``: the in-sample number is
    absent on a round ``--insample-every`` skipped, and the source number is
    absent for a whole run that measures no source series.  Both are recorded as
    JSON ``null`` rather than coerced into a number that was never measured.
    """
    acc = _to_list(accuracies_any)
    return [
        (None if a is None else float(a), None if b is None else float(b))
        for (a, b) in acc
    ]


# --------------- param grid ----------------
def get_param_grid(
    scenario: str,
    metadata: str,
    agg_method_name: str | None = None,
    extended_aggregations: bool = False,
):
    """
    Return a list of aggregation method NAMES to run.
    Only pass names to keep items picklable for multiprocessing.

    ``extended_aggregations`` additionally includes the opt-in server-optimiser
    rules; the default reproduces the published method set.
    """
    cls = select_class(scenario, metadata)

    if agg_method_name is None:
        names = list_method_names(cls, extended=extended_aggregations)
        if not names:
            raise ValueError(f"No aggregation methods found on {cls.__name__}.")
        return names
    if not hasattr(cls, agg_method_name):
        raise ValueError(f"Aggregation method '{agg_method_name}' not found on {cls.__name__}.")
    return [agg_method_name]


# --------------- worker --------------------
def _limit_cpu_threads_for_child_process():
    """
    HARD guard against CPU over-subscription when many processes launch BLAS/OpenMP threads.
    Only affects the current process, so safe to call inside workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass


def run_base_train(
    trainer_name,
    scenario: str,
    metadata: str,
    agg_method_name: str,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    batch_size: int = 64,
    epochs: int = 100,
    max_round: int = 100,
    participation: float = 1.0,
    policy: str = "all",
    sampler_seed: Optional[int] = None,
    track_clients: bool = False,
    stop_when_global_below_clients: bool = False,
    global_name: str = "global",
    clients_per_round: Optional[int] = None,
    weighting: str = "proportional",
    server_eta: float = 1.0,
    server_kwargs: Optional[Mapping[str, Any]] = None,
    client_order: str = "fixed",
    seq_mix_alpha: Optional[float] = None,
    source_share: str = "off",
    source_share_cap: Optional[int] = None,
    init: str = "global",
    provider_name: str = "nist",
    pool_frac: Optional[float] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    save_model_dir: Optional[str] = None,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    old_folds: Any = "all",
    final_eval_batch_size: int = 256,
    source_share_multiplier: float = 1.0,
    val_blend_source: float = 0.0,
):
    """
    Worker entrypoint. Resolves the agg method by name inside the process.

    Returns:
        (msg, exp_name, accuracies, g_all_clients_acc, g_global_acc,
         result_path_str, agg_method_name, run_info)
    """
    # IMPORTANT: do this before heavy libs spin up threads:
    _limit_cpu_threads_for_child_process()

    # Per-process imports (avoid heavy imports in master)
    from federated_outlier_adaptation.aggregation.selector import select_class
    from federated_outlier_adaptation.logging_utils import NistLogger
    from federated_outlier_adaptation.providers import get_provider, provider_config_block
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
    from federated_outlier_adaptation.trainers import artefacts
    from federated_outlier_adaptation.trainers.registry import build_trainer
    from federated_outlier_adaptation.training.final_eval import evaluate_final_model

    provider = get_provider(provider_name)
    timer = provenance.RunTimer()

    trainer = build_trainer(
        trainer_name, trainer_kwargs, provider=provider, global_name=global_name
    )

    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    runner_kwargs = dict(
        trainer=trainer,
        single_outlier=single_outlier,
        provider=provider,
        seed=seed,
        outliers_file=Path(outliers_file) if outliers_file else None,
        participation=participation,
        policy=policy,
        sampler_seed=sampler_seed,
        track_clients=track_clients,
        stop_when_global_below_clients=stop_when_global_below_clients,
        clients_per_round=clients_per_round,
        pool_frac=pool_frac,
        init=init,
        eval_path=eval_path,
        insample_every=insample_every,
        source_share=source_share,
        source_share_cap=source_share_cap,
        # Where the per-round source series is measured, when the run has an
        # old-data book; see :mod:`..runners.source_series`.
        old_book=old_book,
        old_clients_file=old_clients_file,
        # Access to old data: how much of it joins a client's epoch, and how
        # much of it the early-stopping criterion is allowed to look at.
        source_share_multiplier=source_share_multiplier,
        val_blend_source=val_blend_source,
    )
    if scenario == "sequential":
        runner_cls = BaseSequentialRunner
        runner_kwargs["client_order"] = client_order
        runner_kwargs["seq_mix_alpha"] = seq_mix_alpha
    else:
        runner_cls = BaseConcurrentRunner
        runner_kwargs["weighting"] = weighting
        runner_kwargs["server_eta"] = server_eta
        runner_kwargs["server_kwargs"] = dict(server_kwargs or {})
    runner = runner_cls(**runner_kwargs)

    exp_name = f"base_agg_{agg_method_name}_{metadata}_{scenario}"
    if save_model_dir:
        # The worker process is the only place the trained server model still
        # exists - the assembly step runs after it has exited - so the
        # checkpoint is written here, into the run's own output folder, which
        # that step rebuilds from the same parts.
        runner.save_model_path = (
            Path(provider.results_dir) / save_model_dir / exp_name / FINAL_MODEL_NAME
        )
    NistLogger.info(f"[Parallel] {exp_name}")

    accuracies, g_all_clients_acc, g_global_acc, result_path_str = runner.simulate(
        exp_name=exp_name,
        global_name=global_name,
        aggregate_method=agg_method,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        grid_Search=False,
    )
    timer.stop()

    # The final server model exists only here - the assembly step runs after
    # this process has exited - so the numbers the paper reports are measured
    # now, off the books, on rows the model never trained on.
    final_evaluation = evaluate_final_model(
        provider,
        runner.global_model,
        clients=runner.selected_outliers,
        old_book=old_book,
        old_clients_file=old_clients_file,
        old_folds=old_folds,
        batch_size=final_eval_batch_size,
    )

    population = runner.population_info()

    run_info = {
        "instrumentation": runner.instrumentation,
        "population": population,
        "final_evaluation": final_evaluation,
        "config": provenance.build_run_config(
            trainer_name=trainer_name,
            trainer=trainer,
            trainer_kwargs=trainer_kwargs,
            scenario=scenario,
            metadata=metadata,
            agg_method_name=agg_method_name,
            batch_size=batch_size,
            epochs=epochs,
            max_round=max_round,
            seed=seed,
            outliers_file=runner.outliers_file or provider.outliers_file,
            selected_writers=runner.selected_outliers,
            single_outlier=single_outlier,
            global_model_path=artefacts.global_model_path(provider, global_name),
            fisher_dir=artefacts.fisher_dir(provider, global_name),
            started_at=timer.started_at,
            finished_at=timer.finished_at,
            extra={
                "wall_seconds": timer.seconds,
                "participants": list(runner.participants),
                "participation": participation,
                "policy": policy,
                "sampler_seed": sampler_seed,
                "track_clients": track_clients,
                "stop_when_global_below_clients": stop_when_global_below_clients,
                "global_name": global_name,
                "clients_per_round": clients_per_round,
                "weighting": weighting,
                "server_eta": server_eta,
                "client_order": client_order,
                "init": init,
                "provider": provider_name,
                "pool_frac": pool_frac,
                "eval_path": runner.eval_path,
                "insample_every": insample_every,
                "old_book": old_book,
                "old_clients_file": old_clients_file,
                "old_folds": old_folds,
                "source_share": source_share,
                "source_share_multiplier": source_share_multiplier,
                "source_share_cap": source_share_cap,
                "val_blend_source": val_blend_source,
                **provider_config_block(provider),
            },
        ),
    }

    return (
        f"Finished: {exp_name}",
        exp_name,
        accuracies,
        g_all_clients_acc,
        g_global_acc,
        result_path_str,
        agg_method_name,
        run_info,
    )


# --------------- driver --------------------
def final_output_dir(
    parent_name: str,
    trainer_name: str,
    scenario: str,
    metadata: str,
    seed: Optional[int] = None,
    results_dir: Optional[Path] = None,
) -> Path:
    """
    Output directory of one final-training job.

    Mirrors the layout ``final_training`` writes into, so the caller can tell
    whether a job has already produced its summary without running it.
    """
    from federated_outlier_adaptation import config

    root = Path(results_dir) if results_dir else Path(config.RESULTS_DIR)
    parent_dir = root / f"{parent_name}_{trainer_name}_grid_search"
    suffix = seed_suffix(seed)
    if suffix:
        parent_dir = parent_dir / suffix
    return parent_dir / f"{scenario}_{metadata}"


def final_training(
    trainer_name,
    scenario: str,
    metadata: str,
    index: int,
    agg_method_name: str | None = None,
    parent_name: str = "",
    inner_max_workers: int | None = None,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    batch_size: int = 64,
    epochs: int = 100,
    max_round: int = 100,
    participation: float = 1.0,
    policy: str = "all",
    sampler_seed: Optional[int] = None,
    track_clients: bool = False,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    extended_aggregations: bool = False,
    global_name: str = "global",
    clients_per_round: Optional[int] = None,
    weighting: str = "proportional",
    server_eta: float = 1.0,
    server_kwargs: Optional[Mapping[str, Any]] = None,
    client_order: str = "fixed",
    seq_mix_alpha: Optional[float] = None,
    source_share: str = "off",
    source_share_cap: Optional[int] = None,
    save_final_model: bool = False,
    init: str = "global",
    provider_name: str = "nist",
    pool_frac: Optional[float] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    old_folds: Any = "all",
    final_eval_batch_size: int = 256,
    source_share_multiplier: float = 1.0,
    val_blend_source: float = 0.0,
):
    """
    Run one job (defined by scenario/metadata/agg selection).

    If multiple aggregation methods resolve, fan out with a ProcessPoolExecutor
    (inner parallel).  Designed to be safely called from a threaded outer layer
    (uses mp_context='spawn').

    Args:
        seed: Optional run seed.  ``None`` keeps the historical behaviour and
            output paths; an integer seeds every RNG and adds a ``seed_<n>``
            component to the output directory.
        trainer_kwargs: Optional constructor overrides for the trainer, e.g.
            ``{"T": 4, "alpha": 0.95}``.  Defaults come from ``constants.py``.
        outliers_file: Optional alternative selected-client list.
        participation, policy, sampler_seed, track_clients: Client population
            options; the defaults reproduce the published every-client loop.
        stop_when_global_below_clients: Stop a run once the global accuracy
            falls below the clients accuracy or below 0.90.
        skip_existing: Return immediately when the job's ``summary_<index>``
            file already exists, so an interrupted job array can be resubmitted.
    """
    print(f"Running trainer: {trainer_name}")
    if agg_method_name == "none":
        agg_method_name = None

    expected_dir = final_output_dir(
        parent_name=parent_name,
        trainer_name=trainer_name,
        scenario=scenario,
        metadata=metadata,
        seed=seed,
    )
    if skip_existing and (expected_dir / f"summary_{index}.json").is_file():
        print(f"Skipping (summary exists): {expected_dir / f'summary_{index}.json'}")
        return {}

    param_grid = get_param_grid(
        scenario=scenario,
        metadata=metadata,
        agg_method_name=agg_method_name,
        extended_aggregations=extended_aggregations,
    )
    print(f"Aggregation methods: {param_grid}")
    if not param_grid:
        raise ValueError(f"No aggregation methods resolved for scenario={scenario}, metadata={metadata}")

    worker_args = dict(
        single_outlier=single_outlier,
        seed=seed,
        trainer_kwargs=dict(trainer_kwargs or {}),
        outliers_file=str(outliers_file) if outliers_file else None,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        participation=participation,
        policy=policy,
        sampler_seed=sampler_seed,
        track_clients=track_clients,
        stop_when_global_below_clients=stop_when_global_below_clients,
        global_name=global_name,
        clients_per_round=clients_per_round,
        weighting=weighting,
        server_eta=server_eta,
        server_kwargs=dict(server_kwargs or {}),
        client_order=client_order,
        seq_mix_alpha=seq_mix_alpha,
        source_share=source_share,
        source_share_cap=source_share_cap,
        # The folder every worker writes its checkpoint into: the same parts the
        # assembly step below rebuilds ``parent_dir`` from, so the model lands
        # next to the accuracies of the run that produced it.
        save_model_dir=(
            str(
                Path(f"{parent_name}_{trainer_name}_grid_search")
                / (seed_suffix(seed) or "")
                / f"{scenario}_{metadata}"
            )
            if save_final_model
            else None
        ),
        init=init,
        provider_name=provider_name,
        pool_frac=pool_frac,
        eval_path=eval_path,
        insample_every=insample_every,
        # Scored once, on the final server model, after the last aggregation.
        old_book=str(old_book) if old_book else None,
        old_clients_file=str(old_clients_file) if old_clients_file else None,
        old_folds=old_folds,
        final_eval_batch_size=final_eval_batch_size,
        source_share_multiplier=source_share_multiplier,
        val_blend_source=val_blend_source,
    )

    if len(param_grid) == 1:
        single_result = run_base_train(
            trainer_name, scenario, metadata, param_grid[0], **worker_args
        )
        results = [single_result]
        first_result_path_str = single_result[5]
    else:
        # ---- INNER PARALLEL (processes) ----
        max_workers = inner_max_workers or min(len(param_grid), (os.cpu_count() or 1))
        print(f"max_workers={max_workers}")
        # CRUCIAL: when launching processes from threads, use 'spawn'
        mp_ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as executor:
            results = list(
                executor.map(
                    _run_base_train_positional,
                    itertools.repeat(trainer_name),
                    itertools.repeat(scenario),
                    itertools.repeat(metadata),
                    param_grid,  # iterable of method names
                    itertools.repeat(worker_args),
                )
            )
        first_result_path_str = results[0][5]

    # ---- Output collation and plots ----
    parent_dir = Path(first_result_path_str) / f"{parent_name}_{trainer_name}_grid_search"
    suffix = seed_suffix(seed)
    if suffix:
        parent_dir = parent_dir / suffix
    parent_dir = parent_dir / f"{scenario}_{metadata}"
    parent_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    lines = []
    overlay_bucket: dict[str, list[tuple[float, float]]] = {}

    first_ref_clients_acc = _optional_float(results[0][3])
    first_ref_global_acc = _optional_float(results[0][4])

    for (
        msg,
        exp_name,
        accuracies_any,
        g_all_clients_acc,
        g_global_acc,
        result_path_str,
        agg_name,
        run_info,
    ) in results:
        lines.append(msg)
        out_dir = parent_dir / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)

        accuracies = _normalize_accuracies(accuracies_any)
        metrics = run_info.get("instrumentation", {})
        population = run_info.get("population", {})
        final_evaluation = run_info.get("final_evaluation")

        payload = {
            "scenario": scenario,
            "metadata": metadata,
            "agg_method_name": agg_name,
            "accuracies": accuracies,
            "global_clients_all_metrics_acc": _optional_float(g_all_clients_acc),
            "global_clients_metric_acc": _optional_float(g_global_acc),
            # --- additive instrumentation and provenance ---
            "round_seconds": metrics.get("round_seconds"),
            "client_seconds": metrics.get("client_seconds"),
            "comm_bytes_per_round": metrics.get("comm_bytes_per_round"),
            "param_count": metrics.get("param_count"),
            "device": metrics.get("device"),
            "seed": seed,
            # --- additive client-population metrics ---
            **population,
            # --- the reported numbers: the final model, on book test rows ---
            "final_evaluation": final_evaluation,
            "config": run_info.get("config"),
        }
        with open(out_dir / f"accuracies_{index}.json", "w") as f:
            json.dump(payload, f, indent=2)

        # The two reference lines come from the pooled-oracle run.  Without it
        # there is nothing to draw them at, so the plot is skipped and the run's
        # numbers are written all the same; `foa report` can stamp the lines on
        # later, once the oracle exists.
        plottable = all(
            clients is not None and source is not None for clients, source in accuracies
        )
        if g_all_clients_acc is None or g_global_acc is None or not plottable:
            NistLogger.warning(
                f"{exp_name}: no pooled-oracle reference in {parent_dir.parents[1]}, "
                "or a series with unmeasured rounds, so the per-run plot is "
                "skipped. The accuracies are stored."
            )
        else:
            plot_accuracies_over_rounds(
                accuracies=accuracies,
                global_clients_all_metrics_acc=float(g_all_clients_acc),
                global_clients_metric_acc=float(g_global_acc),
                results_path=out_dir,
                scenario=scenario,
                metadata=metadata,
                tag=agg_name,
            )

        if plottable:
            overlay_bucket[agg_name] = accuracies

        summary[exp_name] = {
            "agg_method_name": agg_name,
            "global_clients_all_metrics_acc": _optional_float(g_all_clients_acc),
            "global_clients_metric_acc": _optional_float(g_global_acc),
            "plot_path": str(
                out_dir / f"accuracies_over_rounds_{scenario}_{metadata}_{agg_name}.png"
            ),
            "json_path": str(out_dir / f"accuracies_{index}.json"),
            # --- additive instrumentation and provenance ---
            "round_seconds": metrics.get("round_seconds"),
            "client_seconds": metrics.get("client_seconds"),
            "comm_bytes_per_round": metrics.get("comm_bytes_per_round"),
            "param_count": metrics.get("param_count"),
            "device": metrics.get("device"),
            "seed": seed,
            # --- additive client-population metrics ---
            **population,
            # --- the reported numbers: the final model, on book test rows ---
            "final_evaluation": final_evaluation,
            "config": run_info.get("config"),
        }

    with open(parent_dir / f"summary_{index}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if overlay_bucket:
        plot_overlay_accuracies_per_scenario(
            results_by_method=overlay_bucket,
            results_path=parent_dir,
            scenario=scenario,
            metadata=metadata,
            ref_clients_acc=first_ref_clients_acc,
            ref_global_acc=first_ref_global_acc,
        )

    print("\n".join(lines))
    print(f"\nOutputs written under: {parent_dir}")

    return summary


def _run_base_train_positional(trainer_name, scenario, metadata, agg_method_name, worker_args):
    """Picklable adapter so ``executor.map`` can broadcast the keyword bundle."""
    return run_base_train(trainer_name, scenario, metadata, agg_method_name, **worker_args)


# --------------- CLI -----------------------
def main():
    parser = argparse.ArgumentParser(description="Run base aggregation sweeps.")
    parser.add_argument("--trainer", type=str, default="BaseTrainer", help="Trainer class to use.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="sequential",
        choices=["concurrent", "sequential"],
        help="Scenario to run.",
    )
    parser.add_argument(
        "--metadata", type=str, default="weights", choices=["weights", "delta"], help="Metadata type."
    )
    parser.add_argument("--index", type=int, default=0, help="Index suffix for output file names.")
    parser.add_argument(
        "--agg_method_name",
        type=str,
        default="none",
        help="Specific aggregation method name. Use 'none' to run all methods.",
    )
    parser.add_argument("--parent_name", type=str, default="", help="Parent folder name for results.")
    parser.add_argument(
        "--inner_max_workers", type=int, default=0, help="Max workers for inner ProcessPool (0 -> auto)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional run seed.")
    args = parser.parse_args()

    final_training(
        trainer_name=args.trainer,
        scenario=args.scenario,
        metadata=args.metadata,
        index=args.index,
        agg_method_name=args.agg_method_name,
        parent_name=args.parent_name,
        inner_max_workers=(args.inner_max_workers or None),
        single_outlier=None,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
