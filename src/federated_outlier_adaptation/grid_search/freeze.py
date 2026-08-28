"""
Layer-freezing grid search.

Sweeps the frozen scope and the learning rate of :class:`FreezeTrainer`:

    scope in ["conv", "body"]
    learning_rate in [1e-3, 1e-4]

Same four tasks and the same batch size / epochs / rounds as every other sweep.

Output:
``<results>/freeze_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
"""

from typing import Any, Mapping, Optional

from federated_outlier_adaptation.grid_search.common import (
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)

SPEC = GridSpec(
    folder="freeze_grid_search",
    trainer_name="FreezeTrainer",
    parameters={"scope": ["conv", "body"], "learning_rate": [1e-3, 1e-4]},
    exp_prefix="freeze_{scope}_lr{learning_rate}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=4,
    inner_max_workers=4,
)


def freeze_grid_search(
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
    """Run the full layer-freezing sweep."""
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
    freeze_grid_search()
