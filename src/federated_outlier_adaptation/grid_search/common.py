"""
Shared grid-search engine.

Every hyperparameter sweep in this package follows the same shape: build the
Cartesian product of a few hyperparameter ranges with a set of aggregation
methods, evaluate each configuration with the concurrent or sequential runner,
and merge the resulting accuracy traces into one JSON file per
(scenario, metadata) pair.  The sweeps differ only in the trainer, the
hyperparameter ranges, the experiment-name template and the output folder, so
those are expressed as a :class:`GridSpec` and the machinery lives here.

Execution model (unchanged from the original scripts):
    - outer level: one thread per (scenario, metadata, aggregation) task,
    - inner level: one process per hyperparameter configuration.

Settings shared by all sweeps: batch size 512, 50 local epochs, 50 rounds.
"""

from __future__ import annotations

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from federated_outlier_adaptation.aggregation.selector import (
    FEDAVG_CONCURRENT_FAMILY,
    FEDAVG_CONCURRENT_RULE,
    FEDAVG_SEQUENTIAL_FAMILY,
    FEDAVG_SEQUENTIAL_RULE,
    list_method_names,
    select_class,
)
from federated_outlier_adaptation.utils import provenance
from federated_outlier_adaptation.utils.seeding import seed_suffix

# Sweep settings shared by every grid search.
GRID_BATCH_SIZE = 512
GRID_EPOCHS = 50
GRID_MAX_ROUND = 50


def resolve_budget(
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
) -> tuple[int, int, int]:
    """
    Budget of one sweep configuration.

    ``None`` selects the published sweep setting (batch 512, 50 local epochs,
    50 rounds), so a sweep run without the overrides is unchanged.
    """
    return (
        GRID_BATCH_SIZE if batch_size is None else int(batch_size),
        GRID_EPOCHS if epochs is None else int(epochs),
        GRID_MAX_ROUND if max_round is None else int(max_round),
    )

# Aggregation methods swept per (scenario, metadata) pair.
#
# A sweep selects a hyperparameter and a final run then executes it, so the two
# have to use the *same* update or the selected value is not the one that was
# measured.  The default task list is therefore the FedAvg pair of
# ``aggregation.selector`` verbatim - the same two constants the
# ``--aggregation fedavg`` finals resolve to - and it is two tasks, not four,
# because the other two families' rules are algebraically identical to these.
WEIGHTED_TASKS = [
    (
        FEDAVG_CONCURRENT_FAMILY[0],
        FEDAVG_CONCURRENT_FAMILY[1],
        100,
        FEDAVG_CONCURRENT_RULE,
    ),
    (
        FEDAVG_SEQUENTIAL_FAMILY[0],
        FEDAVG_SEQUENTIAL_FAMILY[1],
        100,
        FEDAVG_SEQUENTIAL_RULE,
    ),
]

#: The capped-rule task list of the published knowledge-distillation sweep.  It
#: mixes ``*_capped_*`` rules into the sweep while the finals run FedAvg, which
#: is the mismatch the default above removes; it is kept so the published sweep
#: can still be reproduced, and is only reachable through an explicit opt-in
#: (``foa grid --capped-tasks``).
CAPPED_TASKS = [
    ("concurrent", "delta", 100, "con_delta_capped_cgd"),
    ("concurrent", "weights", 100, "con_capped_cgw"),
    ("sequential", "delta", 100, "seq_delta_fedavg_update"),
    ("sequential", "weights", 100, "seq_fedavg_update"),
]


def _to_list(v: Any):
    """Make JSON-safe (numpy/tuples/dicts)."""
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_to_list(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_list(val) for k, val in v.items()}
    return v


def _limit_cpu_threads_for_child_process():
    """
    Restrict thread usage inside child processes.

    Sets environment variables (OMP, MKL, OPENBLAS, NUMEXPR) and
    torch.set_num_threads(1) to prevent oversubscription.
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


@dataclass(frozen=True)
class GridSpec:
    """
    Declarative description of one hyperparameter sweep.

    Attributes:
        folder: Output directory name under the results root, e.g.
            ``"ewc_grid_search"``.
        trainer_name: Trainer class instantiated for every configuration.
        parameters: Ordered mapping of trainer keyword -> values to sweep.  The
            Cartesian product is taken in declaration order, which fixes the
            experiment ordering of the published result files.
        exp_prefix: Format template for the experiment name, using the
            parameter names, e.g. ``"distill_T{T}_a{alpha}"``.  The suffix
            ``_agg_<method>_<metadata>_<scenario>`` is appended automatically.
        tasks: (scenario, metadata, index, aggregation method) tuples.
        outer_max_workers / inner_max_workers: Default concurrency.
    """

    folder: str
    trainer_name: str
    parameters: Mapping[str, Sequence[Any]]
    exp_prefix: str
    tasks: Sequence[tuple] = field(default_factory=lambda: list(WEIGHTED_TASKS))
    outer_max_workers: int = 2
    inner_max_workers: int = 8

    def configurations(self, agg_names: Sequence[str]) -> list[tuple[dict, str]]:
        """Cartesian product of the hyperparameters with the aggregation names."""
        keys = list(self.parameters.keys())
        value_lists = [list(self.parameters[k]) for k in keys]
        configs = []
        for values in itertools.product(*value_lists):
            kwargs = dict(zip(keys, values))
            for name in agg_names:
                configs.append((kwargs, name))
        return configs

    def experiment_name(self, kwargs: Mapping[str, Any], agg_method_name: str,
                        metadata: str, scenario: str) -> str:
        prefix = self.exp_prefix.format(**kwargs)
        return f"{prefix}_agg_{agg_method_name}_{metadata}_{scenario}"


def resolve_agg_names(
    scenario: str, metadata: str, agg_method_name: Optional[str], extended: bool = False
) -> list[str]:
    """
    Resolve the aggregation methods a sweep should run.

    The opt-in server-optimiser rules are excluded unless ``extended`` is set,
    so every sweep keeps the published method set.
    """
    cls = select_class(scenario, metadata)
    if agg_method_name is None:
        agg_names = list_method_names(cls, extended=extended)
        if not agg_names:
            raise ValueError(f"No aggregation methods found on {cls.__name__}.")
        return agg_names
    if not hasattr(cls, agg_method_name):
        raise ValueError(f"Aggregation method '{agg_method_name}' not found on {cls.__name__}.")
    return [agg_method_name]


def grid_output_dir(
    folder: str,
    scenario: str,
    metadata: str,
    seed: Optional[int] = None,
    results_dir: Optional[Path] = None,
    provider_name: str = "nist",
) -> Path:
    """
    Directory a sweep writes its merged accuracy file into.

    Every dataset keeps its own results root, so the provider decides the root
    unless the caller passes one explicitly.
    """
    from federated_outlier_adaptation import config

    root = (
        Path(results_dir)
        if results_dir
        else Path(config.provider_results_dir(provider_name))
    )
    parent_dir = root / folder
    suffix = seed_suffix(seed)
    if suffix:
        parent_dir = parent_dir / suffix
    return parent_dir / f"{scenario}_{metadata}"


def run_grid_config(
    spec: GridSpec,
    scenario: str,
    metadata: str,
    kwargs: Mapping[str, Any],
    agg_method_name: str,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    global_name: str = "global",
    population: Optional[Mapping[str, Any]] = None,
    provider_name: str = "nist",
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    grid_info: Optional[Mapping[str, Any]] = None,
):
    """
    Run a single grid configuration inside a worker process.

    ``batch_size``/``epochs``/``max_round`` default to the published sweep
    setting (512 / 50 / 50) and are recorded in the provenance ``config`` block.

    Returns:
        tuple: (message, {experiment_name: accuracies}, result path,
                {experiment_name: provenance config})
    """
    _limit_cpu_threads_for_child_process()
    batch_size, epochs, max_round = resolve_budget(batch_size, epochs, max_round)

    # Per-process imports (keep the master process light)
    from federated_outlier_adaptation.aggregation.selector import select_class
    from federated_outlier_adaptation.logging_utils import NistLogger
    from federated_outlier_adaptation.providers import get_provider, provider_config_block
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
    from federated_outlier_adaptation.trainers import artefacts
    from federated_outlier_adaptation.trainers.registry import build_trainer

    provider = get_provider(provider_name)
    timer = provenance.RunTimer()

    agg_cls = select_class(scenario, metadata)
    agg_method = getattr(agg_cls, agg_method_name)

    all_kwargs = dict(kwargs)
    all_kwargs.update(trainer_kwargs or {})
    trainer = build_trainer(
        spec.trainer_name, all_kwargs, provider=provider, global_name=global_name
    )

    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(
        trainer=trainer,
        single_outlier=single_outlier,
        provider=provider,
        seed=seed,
        outliers_file=Path(outliers_file) if outliers_file else None,
        eval_path=eval_path,
        insample_every=insample_every,
        **dict(population or {}),
    )

    exp_name = spec.experiment_name(kwargs, agg_method_name, metadata, scenario)
    NistLogger.debug(f"[Parallel] {exp_name}")

    accs, result_path_str = runner.simulate(
        exp_name=exp_name,
        global_name=global_name,
        aggregate_method=agg_method,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        grid_Search=True,
    )
    timer.stop()

    config_block = provenance.build_run_config(
        trainer_name=spec.trainer_name,
        trainer=trainer,
        trainer_kwargs=all_kwargs,
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
        parent_name=spec.folder,
        started_at=timer.started_at,
        finished_at=timer.finished_at,
        extra={
            "wall_seconds": timer.seconds,
            "instrumentation": runner.instrumentation,
            "population": runner.population_info(),
            "global_name": global_name,
            "provider": provider_name,
            "eval_path": runner.eval_path,
            "insample_every": insample_every,
            "grid": dict(grid_info or {}),
            **provider_config_block(provider),
        },
    )

    return (
        f"Finished: {exp_name}",
        {exp_name: _to_list(accs)},
        result_path_str,
        {exp_name: config_block},
    )


def _run_grid_config_positional(args):
    """Picklable adapter for :func:`run_grid_config`."""
    return run_grid_config(*args[:5], **args[5])


def grid_search(
    spec: GridSpec,
    scenario: str,
    metadata: str,
    index: int,
    agg_method_name: Optional[str] = None,
    inner_max_workers: int = 8,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    folder: Optional[str] = None,
    skip_existing: bool = False,
    global_name: str = "global",
    population: Optional[Mapping[str, Any]] = None,
    provider_name: str = "nist",
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    grid_info: Optional[Mapping[str, Any]] = None,
):
    """
    Run one (scenario, metadata) sweep and write the merged accuracy JSON.

    Output:
        <results>/<folder>/[seed_<n>/]<scenario>_<metadata>/accuracies_points_<index>.json

    With ``skip_existing`` the sweep returns immediately when that file is
    already present, so an interrupted job array can be resubmitted unchanged.
    """
    if skip_existing:
        existing = grid_output_dir(
            folder or spec.folder,
            scenario,
            metadata,
            seed,
            provider_name=provider_name,
        ) / f"accuracies_points_{index}.json"
        if existing.is_file():
            print(f"Skipping (accuracies exist): {existing}")
            return {}

    agg_names = resolve_agg_names(scenario, metadata, agg_method_name)
    param_grid = spec.configurations(agg_names)
    if not param_grid:
        raise ValueError("Empty param grid.")

    worker_kwargs = dict(
        single_outlier=single_outlier,
        seed=seed,
        trainer_kwargs=dict(trainer_kwargs or {}),
        outliers_file=str(outliers_file) if outliers_file else None,
        global_name=global_name,
        population=dict(population) if population else None,
        provider_name=provider_name,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        eval_path=eval_path,
        insample_every=insample_every,
        grid_info=dict(grid_info) if grid_info else None,
    )

    mp_ctx = get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=inner_max_workers, mp_context=mp_ctx) as executor:
        futures = [
            executor.submit(
                _run_grid_config_positional,
                (spec, scenario, metadata, kwargs, name, worker_kwargs),
            )
            for kwargs, name in param_grid
        ]
        for future in as_completed(futures):
            results.append(future.result())

    msgs, acc_dicts, result_paths, config_dicts = zip(*results)
    merged = {}
    for entry in acc_dicts:
        merged.update(entry)
    configs = {}
    for entry in config_dicts:
        configs.update(entry)

    # The provider owns the results root; the runner's own path is only a
    # fallback for callers that do not go through a provider.
    from federated_outlier_adaptation import config as _config

    merge_root = Path(_config.provider_results_dir(provider_name) or result_paths[0])
    parent_dir = merge_root / (folder or spec.folder)
    suffix = seed_suffix(seed)
    if suffix:
        parent_dir = parent_dir / suffix
    out_dir = parent_dir / f"{scenario}_{metadata}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"accuracies_points_{index}.json"
    with open(out_json, "w") as handle:
        json.dump(merged, handle, indent=2)

    # Provenance lives next to the accuracies without changing that file's shape,
    # which downstream selectors and heatmap scripts parse verbatim.
    with open(out_dir / f"config_points_{index}.json", "w") as handle:
        json.dump({"config": configs}, handle, indent=2)

    print("\n".join(msgs))
    print(f"Saved: {out_json}")
    return merged


def run_all_parallel(
    spec: GridSpec,
    outer_max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = None,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    folder: Optional[str] = None,
    tasks: Optional[Sequence[tuple]] = None,
    skip_existing: bool = False,
    global_name: str = "global",
    population: Optional[Mapping[str, Any]] = None,
    provider_name: str = "nist",
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    grid_info: Optional[Mapping[str, Any]] = None,
):
    """
    Run every task of a sweep, threads outside and processes inside.

    ``batch_size``/``epochs``/``max_round`` are optional budget overrides;
    ``None`` keeps the published sweep setting of 512 / 50 / 50.
    """
    outer = outer_max_workers or spec.outer_max_workers
    inner = inner_max_workers or spec.inner_max_workers
    task_list = list(tasks if tasks is not None else spec.tasks)

    results = []
    with ThreadPoolExecutor(max_workers=outer) as pool:
        futures = [
            pool.submit(
                grid_search,
                spec,
                scenario,
                metadata,
                index,
                agg_method_name=name,
                inner_max_workers=inner,
                single_outlier=single_outlier,
                seed=seed,
                trainer_kwargs=trainer_kwargs,
                outliers_file=outliers_file,
                folder=folder,
                skip_existing=skip_existing,
                global_name=global_name,
                population=population,
                provider_name=provider_name,
                batch_size=batch_size,
                epochs=epochs,
                max_round=max_round,
                eval_path=eval_path,
                insample_every=insample_every,
                grid_info=grid_info,
            )
            for (scenario, metadata, index, name) in task_list
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results
