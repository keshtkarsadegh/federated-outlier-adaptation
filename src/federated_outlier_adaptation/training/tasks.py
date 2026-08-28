"""
Reading a task line back: expected outputs and expected cost.

A plan produced by :mod:`federated_outlier_adaptation.training.matrix` is a
text file of ``foa ...`` command lines.  Two questions about such a file come up
constantly once a job array has run:

*Which lines still have work to do?*
    Every generated task carries ``--skip-existing``, so resubmitting the whole
    array is already safe - but a 600-line array whose 40 timed-out lines are
    scattered through it wastes 560 queue slots on tasks that return in seconds.
    :func:`missing_tasks` keeps only the lines whose expected output files are
    absent, which is what ``foa matrix --plan resume --from tasks.txt`` emits.

*What will it cost?*
    A task is not one federated run: a final task runs one job per aggregation
    family, a sweep task runs the whole hyperparameter product against every
    aggregation rule.  :func:`task_runs` counts the runs a line expands into and
    :func:`task_rounds` reads its round budget, so a plan's GPU-hour estimate can
    be built from a *measured* seconds-per-round instead of a flat guess.

Both answers come from parsing the line with the real command-line parser, so
they cannot drift from what the command would actually do.  A command that
neither writes a result file nor trains (``select-outliers``, ``select``,
``signals``, ``global-train-fl``) has no expected output and no cost model; such
lines are always kept by :func:`missing_tasks` and contribute nothing to the
estimate.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

#: Commands whose outputs this module can predict.
RESUMABLE = ("final", "extreme", "grid")

#: Published round budgets, mirrored from the drivers and the grid engine.
FINAL_ROUNDS = 100
GRID_ROUNDS = 50


def parse_task(line: str):
    """
    The parsed arguments of one task line, or ``None`` when it is not a task.

    Blank lines, comments and anything the parser rejects return ``None``.
    """
    from federated_outlier_adaptation.cli import build_parser

    text = line.strip()
    if not text or text.startswith("#"):
        return None
    parts = shlex.split(text)
    if parts and parts[0] in ("foa", "fal"):
        parts = parts[1:]
    if not parts:
        return None
    parser = build_parser()
    try:
        return parser.parse_args(parts)
    except SystemExit:  # pragma: no cover - a hand-edited line
        return None


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def _final_jobs(args) -> list[tuple[str, str, Optional[str]]]:
    """``(scenario, metadata, rule)`` of every job a final/extreme line runs."""
    from federated_outlier_adaptation.aggregation.selector import (
        SKIP_FAMILY,
        resolve_aggregation,
    )
    from federated_outlier_adaptation.training.final_experiments import JOBS

    jobs = []
    for job in JOBS:
        resolved = resolve_aggregation(
            job["scenario"],
            job["metadata"],
            getattr(args, "aggregation", None),
            extended=getattr(args, "extended_aggregations", False),
        )
        if resolved == SKIP_FAMILY:
            continue
        jobs.append((job["scenario"], job["metadata"], resolved))
    return jobs


def _parent_name(args) -> str:
    if args.command == "extreme":
        from federated_outlier_adaptation.training.extreme_cases import extreme_case_parent

        return args.parent or extreme_case_parent(args.case)
    return args.parent


def _trainer_name(args) -> str:
    return getattr(args, "trainer", None) or "DistillationTrainer"


def _grid_spec(args):
    """The :class:`GridSpec` a ``foa grid`` line runs, or ``None``."""
    import importlib

    from federated_outlier_adaptation.cli import GRID_METHODS

    module_name, _ = GRID_METHODS[args.method]
    module = importlib.import_module(module_name)
    if args.method == "anchored":
        return module.build_spec(
            space=args.space,
            anchor=args.anchor,
            lams=args.lams,
            temperatures=args.temperatures,
        )
    return getattr(module, "SPEC", None)


def _grid_folder(args, spec) -> str:
    if args.parent:
        return args.parent
    if args.method == "extreme":
        from federated_outlier_adaptation.grid_search.extreme_distillation import (
            extreme_grid_folder,
        )

        return extreme_grid_folder(args.case)
    return spec.folder


def expected_outputs(line: str, results_dir: Optional[Path] = None) -> list[Path]:
    """
    Files a task line writes and ``--skip-existing`` checks for.

    An empty list means "this line has no predictable output", which the resume
    helper treats as "always run it again".
    """
    args = parse_task(line)
    if args is None or args.command not in RESUMABLE:
        return []
    root = _provider_root(args, results_dir)

    if args.command in ("final", "extreme"):
        from federated_outlier_adaptation.training.driver import final_output_dir

        return [
            final_output_dir(
                parent_name=_parent_name(args),
                trainer_name=_trainer_name(args),
                scenario=scenario,
                metadata=metadata,
                seed=args.seed,
                results_dir=root,
            )
            / "summary_0.json"
            for scenario, metadata, _rule in _final_jobs(args)
        ]

    from federated_outlier_adaptation.grid_search.common import grid_output_dir

    spec = _grid_spec(args)
    if spec is None:  # pragma: no cover - a sweep without a static spec
        return []
    folder = _grid_folder(args, spec)
    return [
        grid_output_dir(
            folder,
            scenario,
            metadata,
            args.seed,
            results_dir=root,
            provider_name=getattr(args, "provider", "nist"),
        )
        / f"accuracies_points_{index}.json"
        for scenario, metadata, index, _name in spec.tasks
    ]


def _provider_root(args, results_dir: Optional[Path]):
    """Results root a task writes under, following the provider's subtree."""
    if results_dir is None:
        return None
    from federated_outlier_adaptation.training.matrix import provider_results_root

    return provider_results_root(results_dir, getattr(args, "provider", "nist") or "nist")


def is_complete(line: str, results_dir: Optional[Path] = None) -> bool:
    """Whether every expected output of a task line is already on disk."""
    outputs = expected_outputs(line, results_dir=results_dir)
    return bool(outputs) and all(Path(path).is_file() for path in outputs)


def missing_tasks(
    lines: Iterable[str], results_dir: Optional[Path] = None
) -> list[str]:
    """
    The lines of a task file that still have work to do.

    Exactly the ``--skip-existing`` rule, applied ahead of submission: a line
    whose result files are all present is dropped, everything else - including
    every line whose outputs cannot be predicted - is kept, in the original
    order.
    """
    kept: list[str] = []
    for line in lines:
        text = line.rstrip("\n")
        if not text.strip():
            continue
        if is_complete(text, results_dir=results_dir):
            continue
        kept.append(text)
    return kept


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def task_rounds(line: str) -> int:
    """Federated rounds one run of this task performs."""
    args = parse_task(line)
    if args is None or args.command not in RESUMABLE:
        return 0
    if getattr(args, "rounds", None):
        return int(args.rounds)
    return GRID_ROUNDS if args.command == "grid" else FINAL_ROUNDS


def task_runs(line: str) -> int:
    """
    Federated runs one task line expands into.

    A final task runs one job per aggregation family it is not skipped in - the
    ``fedavg`` variant runs two of the four, because the other two families'
    rules are the same update written differently - and a job without an
    explicit rule runs every rule of its family.  A sweep task runs the whole
    hyperparameter product for each of its (scenario, metadata) tasks.
    """
    args = parse_task(line)
    if args is None or args.command not in RESUMABLE:
        return 0

    if args.command in ("final", "extreme"):
        from federated_outlier_adaptation.aggregation.selector import (
            list_method_names,
            select_class,
        )

        total = 0
        for scenario, metadata, rule in _final_jobs(args):
            if rule is not None:
                total += 1
                continue
            total += len(
                list_method_names(
                    select_class(scenario, metadata),
                    extended=getattr(args, "extended_aggregations", False),
                )
            )
        return total

    spec = _grid_spec(args)
    if spec is None:  # pragma: no cover - a sweep without a static spec
        return 0
    product = 1
    for values in spec.parameters.values():
        product *= max(1, len(list(values)))
    return product * len(spec.tasks)


def task_gpu_hours(
    line: str,
    seconds_per_round: float,
    sweep_seconds_per_round: Optional[float] = None,
) -> float:
    """
    GPU hours one task line is expected to consume.

    ``runs x rounds x seconds-per-round``.  Sweeps run at the sweep budget
    (batch 512, 50 local epochs) rather than the finals budget (batch 64, 100
    local epochs), so they take their own measured unit cost; without one the
    finals cost is reused.
    """
    args = parse_task(line)
    if args is None or args.command not in RESUMABLE:
        return 0.0
    unit = seconds_per_round
    if args.command == "grid":
        unit = seconds_per_round if sweep_seconds_per_round is None else sweep_seconds_per_round
    return task_runs(line) * task_rounds(line) * float(unit) / 3600.0


def plan_gpu_hours(
    tasks: Sequence[str],
    seconds_per_round: float,
    sweep_seconds_per_round: Optional[float] = None,
) -> dict[str, Any]:
    """
    Measured-cost estimate of a whole plan.

    Returns the total GPU hours, the number of runs and the number of task lines
    the model could not price (they cost real time too, but a selection or a
    pool build is minutes, not hours).
    """
    hours = 0.0
    runs = 0
    unpriced = 0
    for line in tasks:
        line_runs = task_runs(line)
        if not line_runs:
            unpriced += 1
            continue
        runs += line_runs
        hours += task_gpu_hours(line, seconds_per_round, sweep_seconds_per_round)
    return {
        "tasks": len(list(tasks)),
        "runs": runs,
        "unpriced_tasks": unpriced,
        "gpu_hours": hours,
        "seconds_per_round": seconds_per_round,
        "sweep_seconds_per_round": (
            seconds_per_round if sweep_seconds_per_round is None else sweep_seconds_per_round
        ),
    }


__all__ = [
    "FINAL_ROUNDS",
    "GRID_ROUNDS",
    "RESUMABLE",
    "expected_outputs",
    "is_complete",
    "missing_tasks",
    "parse_task",
    "plan_gpu_hours",
    "task_gpu_hours",
    "task_rounds",
    "task_runs",
]
