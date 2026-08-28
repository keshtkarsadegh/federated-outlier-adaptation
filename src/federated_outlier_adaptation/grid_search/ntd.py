"""
Not-true distillation grid search.

Sweeps the two knobs of :class:`NTDTrainer` over the paper's own range
(``grid_ranges_from_literature.md`` section 3):

    beta in [0.0, 0.001, 0.01, 0.1, 0.3, 1.0]
    tau  in [1, 2, 4, 8]

Lee et al. sweep beta over {0.001 ... 1.0} and report 1.0 as the best value,
which the published three points did not cover; ``beta = 0`` is the explicit
"penalty off" arm, so a selection at the bottom of the range can be compared
against plain training instead of being read as a small penalty.

Same four tasks and the same batch size / epochs / rounds as every other sweep.

Output:
``<results>/ntd_grid_search/<scenario>_<metadata>/accuracies_points_100.json``
"""

from typing import Any, Mapping, Optional

from federated_outlier_adaptation.grid_search.common import (
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)

#: Penalty weights of the FedNTD paper, plus the explicit "off" arm.
NTD_BETAS = [0.0, 0.001, 0.01, 0.1, 0.3, 1.0]

#: Temperatures swept alongside them.
NTD_TAUS = [1, 2, 4, 8]

SPEC = GridSpec(
    folder="ntd_grid_search",
    trainer_name="NTDTrainer",
    parameters={"beta": NTD_BETAS, "tau": NTD_TAUS},
    exp_prefix="ntd_b{beta}_t{tau}",
    tasks=WEIGHTED_TASKS,
    outer_max_workers=4,
    inner_max_workers=4,
)


def ntd_grid_search(
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
    """Run the full not-true distillation sweep."""
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
    ntd_grid_search()
