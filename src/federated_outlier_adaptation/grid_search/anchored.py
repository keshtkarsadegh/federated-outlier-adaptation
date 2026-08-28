"""
Grid search over the unified regularisation family.

:class:`AnchoredTrainer` expresses every regularised objective as
``CE + lam * D_space(theta ; anchor)``, so one sweep shape covers them all:

    lam in [0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]   (8 points, 6 decades)
    T   in [1, 2, 4, 8]                              (kd / ntd spaces only)

``lam = 0`` is an explicit arm of every space, not a placeholder: the penalty
is then switched off and the objective is exactly the cross-entropy of plain
training.  It is what a selection at the bottom of the range has to be compared
against - an optimum at ``lam = 1e-3`` means "this space does not help" only if
``lam = 0`` is no worse, and that comparison needs the arm to have been run.

The batch size, epoch count and round count are the ones every other sweep
uses, so the results are directly comparable.  ``space`` and ``anchor`` are
fixed per sweep and appear in the output folder, which keeps one sweep per
combination instead of one enormous product:

``<results>/anchored_<space>_<anchor>_grid_search/<scenario>_<metadata>/``

Both ranges can be narrowed per sweep (``foa grid --method anchored --lams ...
--temperatures ...``), which is what the lean plans do; the defaults are the
published ranges above and the ranges actually swept are written into the
provenance block of every result file.
"""

from typing import Any, Mapping, Optional, Sequence

from federated_outlier_adaptation.grid_search.common import (
    WEIGHTED_TASKS,
    GridSpec,
    run_all_parallel,
)
from federated_outlier_adaptation.trainers.anchored_trainer import (
    ANCHOR_SPACES,
    ANCHORS,
)

#: Log-spaced penalty weights shared by every space, with the "off" arm first.
ANCHOR_LAMS = [0.0, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

#: Temperatures swept for the softmax-based spaces.
ANCHOR_TEMPERATURES = [1, 2, 4, 8]

#: Spaces that have a temperature.
TEMPERATURE_SPACES = ("kd", "ntd", "kd+fisher")


def folder_tag(space: str, anchor: str) -> str:
    """Filesystem-safe tag of a (space, anchor) combination."""
    return f"{space.replace('+', '_')}_{anchor}"


def resolve_grid(
    space: str,
    lams: Optional[Sequence[float]] = None,
    temperatures: Optional[Sequence[float]] = None,
) -> dict[str, list]:
    """
    The penalty weights and temperatures one sweep actually covers.

    ``None`` keeps the published ranges, so an existing command line produces
    exactly the sweep it produced before.  A short list is what the lean plans
    pass, and the resolved ranges are written into the provenance block, so a
    stored result always says which grid produced it.
    """
    if space not in ANCHOR_SPACES:
        raise ValueError(f"Unknown space {space!r}; expected one of {ANCHOR_SPACES}")
    grid: dict[str, list] = {
        "lam": list(ANCHOR_LAMS if lams is None else lams),
    }
    if not grid["lam"]:
        raise ValueError("The lambda grid of an anchored sweep must not be empty.")
    if space in TEMPERATURE_SPACES:
        grid["T"] = list(ANCHOR_TEMPERATURES if temperatures is None else temperatures)
        if not grid["T"]:
            raise ValueError("The temperature grid of an anchored sweep must not be empty.")
    return grid


def build_spec(
    space: str = "kd",
    anchor: str = "frozen",
    lams: Optional[Sequence[float]] = None,
    temperatures: Optional[Sequence[float]] = None,
) -> GridSpec:
    """Sweep definition for one (space, anchor) combination."""
    if anchor not in ANCHORS:
        raise ValueError(f"Unknown anchor {anchor!r}; expected one of {ANCHORS}")

    parameters = resolve_grid(space, lams=lams, temperatures=temperatures)
    exp_prefix = "anchored_lam{lam}_T{T}" if "T" in parameters else "anchored_lam{lam}"

    return GridSpec(
        folder=f"anchored_{folder_tag(space, anchor)}_grid_search",
        trainer_name="AnchoredTrainer",
        parameters=parameters,
        exp_prefix=exp_prefix,
        tasks=WEIGHTED_TASKS,
        outer_max_workers=4,
        inner_max_workers=4,
    )


def anchored_grid_search(
    space: str = "kd",
    anchor: str = "frozen",
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
    lams: Optional[Sequence[float]] = None,
    temperatures: Optional[Sequence[float]] = None,
    index: Optional[int] = None,
):
    """
    Run the anchored sweep for one distance space and anchor.

    ``index`` overrides the file index of every task, i.e. the ``<n>`` of
    ``accuracies_points_<n>.json``.  A sweep is read back by globbing those
    files, so a second invocation with a different index adds its points to the
    *same* sweep rather than overwriting the first - which is how the
    "penalty off" arm joins a space's grid as one extra point instead of being
    repeated once per temperature or living in a folder of its own where the
    selection could never compare it with anything.

    ``lams``/``temperatures`` narrow the swept ranges; ``None`` keeps the
    published ones.  Whatever is used is recorded in the provenance block.
    """
    spec = build_spec(space=space, anchor=anchor, lams=lams, temperatures=temperatures)
    tasks = (
        None
        if index is None
        else [
            (scenario, metadata, int(index), rule)
            for scenario, metadata, _, rule in spec.tasks
        ]
    )
    kwargs = dict(trainer_kwargs or {})
    kwargs.setdefault("space", space)
    kwargs.setdefault("anchor", anchor)
    return run_all_parallel(
        spec,
        tasks=tasks,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        seed=seed,
        trainer_kwargs=kwargs,
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
        grid_info={
            "space": space,
            "anchor": anchor,
            **{key: list(values) for key, values in spec.parameters.items()},
        },
    )


if __name__ == "__main__":
    anchored_grid_search()
