"""
all_aggregations.py - baseline trainer against every aggregation method.

Purpose:
    Sweeps *all* aggregation variants (sequential/concurrent x weights/delta)
    with the plain :class:`BaseTrainer`, to benchmark and validate every
    aggregation implementation under one consistent baseline.

Execution model:
    - Outer (threads): one thread per (scenario, metadata) configuration.
    - Inner (processes): one process per aggregation method.

Outputs:
    ``<results>/all_aggs_fl_BaseTrainer_grid_search/[seed_<n>/]<scenario>_<metadata>/``
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Optional

from federated_outlier_adaptation.training.driver import final_training

PARENT_NAME = "all_aggs_fl"

JOBS = [
    dict(scenario="sequential", metadata="weights"),
    dict(scenario="sequential", metadata="delta"),
    dict(scenario="concurrent", metadata="weights"),
    dict(scenario="concurrent", metadata="delta"),
]


def run_all_parallel(
    outer_max_workers: int = 3,
    inner_max_workers: int = 12,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    skip_existing: bool = False,
    extended_aggregations: bool = False,
    global_name: str = "global",
    parent_name: str = PARENT_NAME,
    provider_name: str = "nist",
):
    """
    Launch every (scenario, metadata) sweep in parallel.

    Args:
        outer_max_workers: Threads for the outer pool (one job each).
        inner_max_workers: Processes used per job.
        seed: Optional run seed.
        trainer_kwargs: Optional trainer constructor overrides.
        outliers_file: Optional alternative selected-client list.
        skip_existing: Skip jobs whose summary file already exists.
    """
    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futures = [
            outer.submit(
                final_training,
                trainer_name="BaseTrainer",
                index=0,
                agg_method_name="none",
                parent_name=parent_name,
                provider_name=provider_name,
                inner_max_workers=inner_max_workers,
                seed=seed,
                trainer_kwargs=trainer_kwargs,
                outliers_file=outliers_file,
                skip_existing=skip_existing,
                extended_aggregations=extended_aggregations,
                global_name=global_name,
                **job,
            )
            for job in JOBS
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def all_aggregation_variants(
    outer_max_workers: int = 1,
    inner_max_workers: int = 20,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    skip_existing: bool = False,
    extended_aggregations: bool = False,
    global_name: str = "global",
    parent_name: str = PARENT_NAME,
    provider_name: str = "nist",
):
    """Entry point used by the CLI and the notebooks."""
    results = run_all_parallel(
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        skip_existing=skip_existing,
        extended_aggregations=extended_aggregations,
        global_name=global_name,
        parent_name=parent_name,
        provider_name=provider_name,
    )
    for result in results:
        print("Completed:", result)
    return results


if __name__ == "__main__":
    all_aggregation_variants()
