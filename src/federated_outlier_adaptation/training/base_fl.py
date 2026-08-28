"""
base_fl.py - baseline federated-learning runs (paper: "prove FL" phase).

Purpose:
    Trains the plain :class:`BaseTrainer` across the representative aggregation
    methods of each family, establishing the federated baseline the
    regularised trainers are compared against.

Jobs configured:
    - Sequential weights  (seq_fedavg_update)
    - Sequential delta    (seq_delta_fedavg_update)
    - Concurrent weights  (con_weighted_cgw, con_weighted_cw)
    - Concurrent delta    (con_delta_weighted_cgd)

Parallelism:
    - Outer (threads): one job per configuration.
    - Inner (processes): controlled by ``inner_max_workers``.

Outputs:
    ``<results>/prove_fl_BaseTrainer_grid_search/[seed_<n>/]<scenario>_<metadata>/``
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Optional

from federated_outlier_adaptation.training.driver import final_training

PARENT_NAME = "prove_fl"

JOBS = [
    dict(scenario="sequential", metadata="weights", agg_method_name="seq_fedavg_update"),
    dict(scenario="sequential", metadata="delta", agg_method_name="seq_delta_fedavg_update"),
    dict(scenario="concurrent", metadata="weights", agg_method_name="con_weighted_cgw"),
    dict(scenario="concurrent", metadata="weights", agg_method_name="con_weighted_cw"),
    dict(scenario="concurrent", metadata="delta", agg_method_name="con_delta_weighted_cgd"),
]


def run_all_parallel(
    outer_max_workers: int = 3,
    inner_max_workers: int = 12,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    parent_name: str = PARENT_NAME,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    global_name: str = "global",
    provider_name: str = "nist",
):
    """
    Launch all baseline federated jobs in parallel.

    Args:
        outer_max_workers: Number of jobs to run concurrently (thread pool).
        inner_max_workers: Workers for each job's inner process pool.
        seed: Optional run seed; adds a ``seed_<n>`` output component.
        trainer_kwargs: Optional trainer constructor overrides.
        outliers_file: Optional alternative selected-client list.
        parent_name: Output folder prefix; the default keeps the published one.
        stop_when_global_below_clients: Stop a run once the global accuracy
            falls below the clients accuracy or below 0.90.
        skip_existing: Skip jobs whose summary file already exists.

    Returns:
        list: Summary dict of every completed job.
    """
    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futures = [
            outer.submit(
                final_training,
                trainer_name="BaseTrainer",
                index=0,
                parent_name=parent_name,
                inner_max_workers=inner_max_workers,
                seed=seed,
                trainer_kwargs=trainer_kwargs,
                outliers_file=outliers_file,
                stop_when_global_below_clients=stop_when_global_below_clients,
                skip_existing=skip_existing,
                global_name=global_name,
                provider_name=provider_name,
                **job,
            )
            for job in JOBS
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def base_fl_training(
    outer_max_workers: int = 4,
    inner_max_workers: int = 1,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    parent_name: str = PARENT_NAME,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    global_name: str = "global",
    provider_name: str = "nist",
):
    """Entry point used by the CLI and the notebooks."""
    results = run_all_parallel(
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        parent_name=parent_name,
        stop_when_global_below_clients=stop_when_global_below_clients,
        skip_existing=skip_existing,
        global_name=global_name,
        provider_name=provider_name,
    )
    for result in results:
        print("Completed:", result)
    return results


if __name__ == "__main__":
    base_fl_training()
