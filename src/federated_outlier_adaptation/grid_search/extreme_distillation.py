"""
Extreme-case distillation grid search.

Same sweep as :mod:`federated_outlier_adaptation.grid_search.distillation`
but restricted to a hand-picked client set, and with a wider alpha range:

    T in [1, 2, 4, 8, 10, 50]
    alpha in [0.1, 0.2, 0.5, 0.7, 0.95, 0.98]

Cases (see ``constants.EXTREME_CASE_CLIENTS``):
    single  one outlier,
    double  two distinct low-accuracy writers,
    dual    the same outlier replicated into two participants.

Output:
``<results>/<tag>_distillation_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
with the tag taken from ``constants.EXTREME_CASE_TAGS``; ``single`` keeps the
published folder, the two changed cases get their own.
"""

from typing import Any, Mapping, Optional, Sequence

from federated_outlier_adaptation.constants import EXTREME_CASE_CLIENTS, EXTREME_CASE_TAGS
from federated_outlier_adaptation.grid_search.common import (
    CAPPED_TASKS,
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)

SPEC = GridSpec(
    folder="extreme_outlier_distillation_grid_search",
    trainer_name="DistillationTrainer",
    parameters={
        "T": [1, 2, 4, 8, 10, 50],
        "alpha": [0.1, 0.2, 0.5, 0.7, 0.95, 0.98],
    },
    exp_prefix="distill_T{T}_a{alpha}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=2,
    inner_max_workers=8,
)


def extreme_grid_folder(extreme_case: str) -> str:
    """Output folder of one extreme-case sweep."""
    return f"{EXTREME_CASE_TAGS[extreme_case]}_distillation_grid_search"


def extreme_cases_grid_search(
    single_outlier: Optional[Sequence[str]] = None,
    extreme_case: str = "single",
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
    Run the sweep for one extreme case.

    ``capped_tasks`` opts back into the published capped-rule task list; the
    default is the FedAvg pair the finals execute.

    Args:
        single_outlier: Client ids to restrict the run to.  Defaults to the
            configuration of ``extreme_case``.
        extreme_case: ``"single"``, ``"double"`` or ``"dual"``; also selects the
            output folder ``<case>_outlier_distillation_grid_search``.
    """
    if single_outlier is None:
        single_outlier = EXTREME_CASE_CLIENTS[extreme_case]
    return run_all_parallel(
        SPEC,
        tasks=CAPPED_TASKS if capped_tasks else None,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        single_outlier=list(single_outlier),
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        skip_existing=skip_existing,
        global_name=global_name,
        population=population,
        provider_name=provider_name,
        folder=folder or extreme_grid_folder(extreme_case),
        batch_size=batch_size,
        epochs=epochs,
        max_round=max_round,
        eval_path=eval_path,
        insample_every=insample_every,
    )


if __name__ == "__main__":
    extreme_cases_grid_search(extreme_case="single")
