"""
Logit-consistency grid search.

Sweeps the consistency coefficient of :class:`CFLogitConsistencyTrainer`.

    lambda_consis in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]

Output:
``<results>/logit_consistency_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
"""

from typing import Any, Mapping, Optional

from federated_outlier_adaptation.grid_search.common import (
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)

SPEC = GridSpec(
    folder="logit_consistency_grid_search",
    trainer_name="CFLogitConsistencyTrainer",
    parameters={"lambda_consis": [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]},
    exp_prefix="logit_consistency_{lambda_consis}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=4,
    inner_max_workers=4,
)


def logit_consistency_grid_search(
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
    """Run the full logit-consistency sweep."""
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
    logit_consistency_grid_search()
