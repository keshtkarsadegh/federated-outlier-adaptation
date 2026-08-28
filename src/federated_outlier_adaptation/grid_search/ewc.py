"""
EWC grid search.

Sweeps the elasticity coefficient of :class:`EWCTrainer` against the
representative aggregation method of each family.

    lambda in [0.2, 1, 4, 8, 10, 20, 100, 200]

Output: ``<results>/ewc_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
"""

from typing import Any, Mapping, Optional

from federated_outlier_adaptation.grid_search.common import (
    WEIGHTED_TASKS,
    GridSpec,
    grid_search,
    run_all_parallel,
)

SPEC = GridSpec(
    folder="ewc_grid_search",
    trainer_name="EWCTrainer",
    parameters={"ewc_lambda": [0.2, 1, 4, 8, 10, 20, 100, 200]},
    exp_prefix="ewc_{ewc_lambda}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=2,
    inner_max_workers=8,
)


def ewc_grid_search(
    outer_max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    skip_existing: bool = False,
    global_name: str = "global",
    population: Optional[Mapping[str, Any]] = None,
    provider_name: str = "nist",
    folder: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
):
    """Run the full EWC sweep."""
    return run_all_parallel(
        SPEC,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        skip_existing=skip_existing,
        global_name=global_name,
        population=population,
        provider_name=provider_name,
        folder=folder,
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        eval_path=eval_path,
        insample_every=insample_every,
    )


if __name__ == "__main__":
    ewc_grid_search()
