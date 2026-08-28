"""
Knowledge-distillation grid search.

Sweeps temperature and interpolation weight of :class:`DistillationTrainer`.

    T in [1, 2, 4, 8, 10, 50]
    alpha in [0.1, 0.5, 0.7, 0.9, 0.95]

The alpha range follows ``grid_ranges_from_literature.md`` section 3: Hinton's
own combinations and the empirical studies put the optimum in the 0.7-0.9 band,
which the published three points (0.1, 0.5, 0.95) stepped straight over.

Output:
``<results>/distillation_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
"""

from typing import Any, Mapping, Optional

from federated_outlier_adaptation.grid_search.common import (
    CAPPED_TASKS,
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)

#: Interpolation weights swept; see the module docstring.
KD_ALPHAS = [0.1, 0.5, 0.7, 0.9, 0.95]

#: Temperatures swept.
KD_TEMPERATURES = [1, 2, 4, 8, 10, 50]

SPEC = GridSpec(
    folder="distillation_grid_search",
    trainer_name="DistillationTrainer",
    parameters={"T": KD_TEMPERATURES, "alpha": KD_ALPHAS},
    exp_prefix="distill_T{T}_a{alpha}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=1,
    inner_max_workers=20,
)


def distillation_grid_search(
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
    capped_tasks: bool = False,
):
    """
    Run the full knowledge-distillation sweep.

    Args:
        capped_tasks: Sweep the published capped-rule task list instead of the
            FedAvg pair the finals use.  Off by default: the published list
            selects a hyperparameter under one rule and the final then runs it
            under another.
    """
    return run_all_parallel(
        SPEC,
        tasks=CAPPED_TASKS if capped_tasks else None,
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
    distillation_grid_search()
