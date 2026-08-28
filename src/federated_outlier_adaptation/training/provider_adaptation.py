"""
Single adaptation run for an arbitrary dataset provider.

``training/driver.py`` orchestrates the published sweeps (thread and process
pools, plots, overlays) on the NIST results root.  This module is the minimal
counterpart used to exercise one adaptation configuration of a newly added
dataset end to end: it builds the trainer, runs one runner for a handful of
rounds and writes the numbers in the layout the driver uses -
``<results>/<parent>_<Trainer>_grid_search/<scenario>_<metadata>/base_agg_<agg>_<metadata>_<scenario>/accuracies_<i>.json``
- including the ``config`` provenance block, extended with the provider name
and the SHA-256 of the prepared dataset files.

It touches nothing the NIST pipeline uses; it only reads a provider.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.providers import get_provider, provider_config_block
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.registry import build_trainer
from federated_outlier_adaptation.utils import provenance
from federated_outlier_adaptation.utils.seeding import seed_suffix


def run_adaptation(
    provider,
    trainer_name: str = "BaseTrainer",
    scenario: str = "concurrent",
    metadata: str = "weights",
    agg_method_name: str = "con_weighted_cgw",
    batch_size: int = 64,
    epochs: int = 2,
    max_round: int = 5,
    k: int = 5,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    parent_name: str = "adaptation",
    index: int = 0,
) -> dict:
    """
    Run one adaptation configuration and persist its numbers.

    Args:
        provider: Dataset provider supplying loaders, model and artefacts.
        trainer_name: Name registered in ``trainers/registry.py``.
        scenario: ``"concurrent"`` or ``"sequential"``.
        metadata: ``"weights"`` or ``"delta"``.
        agg_method_name: Aggregation method of the selected family.
        batch_size, epochs, max_round: Federated schedule.
        k: Size of the participant list requested from the provider.
        seed: Optional run seed; adds a ``seed_<n>`` output component.
        trainer_kwargs: Trainer constructor overrides.
        parent_name: Output folder prefix.
        index: Result file index, as in the driver's layout.

    Returns:
        The payload that was written.
    """
    timer = provenance.RunTimer()
    trainer = build_trainer(trainer_name, trainer_kwargs, provider=provider)
    agg_method = getattr(select_class(scenario, metadata), agg_method_name)

    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider, seed=seed, k=k)

    exp_name = f"base_agg_{agg_method_name}_{metadata}_{scenario}"
    NistLogger.info(f"[{provider.name}] {exp_name}")

    accuracies, clients_reference, global_reference, _ = runner.simulate(
        exp_name=exp_name,
        global_name="global",
        aggregate_method=agg_method,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        grid_Search=False,
    )
    timer.stop()

    results_dir = Path(getattr(provider, "results_dir"))
    parent_dir = results_dir / f"{parent_name}_{trainer_name}_grid_search"
    suffix = seed_suffix(seed)
    if suffix:
        parent_dir = parent_dir / suffix
    out_dir = parent_dir / f"{scenario}_{metadata}" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = runner.instrumentation
    payload = {
        "scenario": scenario,
        "metadata": metadata,
        "agg_method_name": agg_method_name,
        "accuracies": [[float(a), float(b)] for a, b in accuracies],
        "global_clients_all_metrics_acc": float(clients_reference),
        "global_clients_metric_acc": float(global_reference),
        "round_seconds": metrics.get("round_seconds"),
        "client_seconds": metrics.get("client_seconds"),
        "comm_bytes_per_round": metrics.get("comm_bytes_per_round"),
        "param_count": metrics.get("param_count"),
        "device": metrics.get("device"),
        "seed": seed,
        "config": provenance.build_run_config(
            trainer_name=trainer_name,
            trainer=trainer,
            trainer_kwargs=trainer_kwargs,
            scenario=scenario,
            metadata=metadata,
            agg_method_name=agg_method_name,
            parent_name=parent_name,
            batch_size=batch_size,
            epochs=epochs,
            max_round=max_round,
            seed=seed,
            outliers_file=provider.outliers_file,
            selected_writers=runner.selected_outliers,
            global_model_path=provider.global_model_path,
            fisher_dir=provider.fisher_dir,
            started_at=timer.started_at,
            finished_at=timer.finished_at,
            extra={"wall_seconds": timer.seconds, **provider_config_block(provider)},
        ),
    }

    target = out_dir / f"accuracies_{index}.json"
    with open(target, "w") as handle:
        json.dump(payload, handle, indent=2)
    NistLogger.info(f"[{provider.name}] wrote {target}")

    with open(parent_dir / f"summary_{index}.json", "w") as handle:
        json.dump({exp_name: {**payload, "json_path": str(target)}}, handle, indent=2)

    payload["json_path"] = str(target)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one adaptation configuration of a dataset.")
    parser.add_argument("--provider", default="nist", help="nist | shakespeare | cifar10")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--trainer", default="BaseTrainer")
    parser.add_argument("--scenario", default="concurrent", choices=("concurrent", "sequential"))
    parser.add_argument("--metadata", default="weights", choices=("weights", "delta"))
    parser.add_argument("--agg", default="con_weighted_cgw")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parent-name", default="adaptation")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    provider = get_provider(args.provider, **kwargs)

    payload = run_adaptation(
        provider,
        trainer_name=args.trainer,
        scenario=args.scenario,
        metadata=args.metadata,
        agg_method_name=args.agg,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_round=args.rounds,
        k=args.k,
        seed=args.seed,
        parent_name=args.parent_name,
    )
    print(
        json.dumps(
            {
                "json_path": payload["json_path"],
                "accuracies": payload["accuracies"],
                "participants": payload["config"]["outlier_writers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
