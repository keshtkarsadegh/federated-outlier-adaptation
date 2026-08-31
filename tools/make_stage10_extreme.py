"""
Emit the stage-10 extreme-case client lists, task file and README.

    python tools/make_stage10_extreme.py --root $FOA_RESULTS_DIR/main_v6 \
        [--out stage10_extreme.txt] [--readme stage10_extreme_README.md]

Reads two JSON files and writes three tiny JSON files plus two text files: a
login-node job, no data touched.

The three client lists are written rather than hand-maintained because the
writers they name are a *ranking*, and a ranking that is retyped is a ranking
that can silently go stale. They are read back by ``--outliers-file``, so the
run's own provenance records exactly which writers it federated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_stage8_combos import task_line as combo_task_line

from federated_outlier_adaptation.training.combo_cells import combo_cells
from federated_outlier_adaptation.training.extreme_cells import (
    FULL_ROUNDS,
    extreme_cells,
    rank_cohort,
    winning_combo_id,
)

RESULTS = "$FOA_RESULTS_DIR/main_v6"
BOOKS = f"{RESULTS}/fold_books"
POOLS = f"{RESULTS}/outliers"

#: Measured from the v6 books: rows per fold, for the cost estimate.
COHORT_TRAIN_ROWS = 1355
COHORT_WRITERS = 20
LOCAL_EPOCHS = 5
SECONDS_PER_IMAGE = 1.94e-4
#: Fixed per-round overhead of a family, from the Phase A calibration.
ROUND_OVERHEAD = 0.31


def winning_combo(root) -> dict:
    """
    The stage-8 cell this stage runs: read from ``root``, then looked up.

    Two steps, and both matter. WHICH combination won is a selection result and
    is read from the crowning record; WHAT that combination is - its rule, its
    penalty, its hyper-parameters - is looked up in the stage-8 cell list, so
    the flags this stage copies are the ones that stage measured rather than a
    second statement of them.
    """
    winner = winning_combo_id(root)
    by_id = {cell["id"]: cell for cell in combo_cells()}
    if winner not in by_id:
        raise SystemExit(
            f"FATAL: the crowning names {winner!r}, which is not a stage-8 "
            "combination. The record and the cell list describe different "
            "stages."
        )
    return by_id[winner]


def client_list_path(case: str) -> str:
    return f"{POOLS}/extreme_{case}.json"


def task_line(cell: dict, combo: dict, fold: int, rounds: int) -> str:
    """
    One `foa final` invocation: the winning method, this case's clients.

    Built by taking the stage-8 winner's own line and changing exactly four
    things - the parent, the client list, the participation, and the round
    budget's provenance seed. Copying the winner's flags rather than restating
    them is what keeps "the winning config" true by construction.
    """
    seed = 10200 + fold
    line = combo_task_line(combo, fold, rounds, seed)

    def swap(old: str, new: str) -> None:
        nonlocal line
        # Every substitution must actually bite: a stage-8 line that changed
        # shape would otherwise leave this stage quietly running stage 8.
        if old not in line:
            raise SystemExit(
                f"The stage-8 line no longer contains {old!r}; stage 10 copies "
                "it and can no longer do so safely."
            )
        line = line.replace(old, new)

    swap(
        f"--parent coh8_combo_{combo['id']}_fold{fold}",
        f"--parent coh10_extreme_{cell['id']}_fold{fold}",
    )
    swap(
        f"--outliers-file {POOLS}/cohort_worst20.json",
        f"--outliers-file {client_list_path(cell['case'])}",
    )
    # Full participation: dropping a fifth of a two-client federation is not a
    # participation study, it is a coin flip on whether the round happens.
    swap(
        f" --policy uniform --clients-per-round 16 --sampler-seed {seed}",
        f" --policy all --participation 1.0 --sampler-seed {seed}",
    )
    return line


def task_seconds(cell: dict, rounds: int) -> float:
    """Honest wall-clock estimate of one task, in seconds."""
    per_writer = COHORT_TRAIN_ROWS / COHORT_WRITERS
    images = len(cell["writers"]) * per_writer * LOCAL_EPOCHS
    # One family (the winner is a sequential rule), plus the teacher forward
    # pass the kd+fisher penalty needs on every batch.
    per_round = ROUND_OVERHEAD + 1.4 * images * SECONDS_PER_IMAGE
    # Per-round evaluation is the pooled cohort sets and the old-book series,
    # and does not shrink with the client count - it dominates here.
    per_round += 0.25
    return per_round * rounds + 25.0 + 2.0 + 40.0


def write_client_lists(root: Path, cells, out_dir: Path) -> dict:
    """The three tiny client lists the runs read their populations from."""
    written = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        path = out_dir / f"extreme_{cell['case']}.json"
        path.write_text(json.dumps(cell["clients"], indent=2) + "\n")
        written[cell["case"]] = str(path)
    return written


def header(cells, combo, folds, rounds) -> list[str]:
    total = sum(task_seconds(c, rounds) * len(folds) for c in cells)
    lines = [
        "# stage10_extreme.txt - EXTREME CASES, with the winning method only.",
        "#",
        "# Every earlier stage compared methods. This one does not. The panel is",
        f"# over and the winner is stage 8's {combo['id']}:",
        f"#   aggregation  {combo['agg']['rule']}",
        f"#   penalty      {combo['reg']['space']} at "
        + ", ".join(f"{k}={v:g}" for k, v in combo["reg"]["hypers"].items()),
        "#",
        "# Its flags are COPIED from the stage-8 line, not restated, so \"the",
        "# winning configuration\" is true by construction rather than by",
        "# proofreading. Exactly four things change: the parent, the client list,",
        "# full participation, and the seed.",
        "#",
        "# THREE ARRANGEMENTS OF THE SAME TWO WRITERS.",
        "#",
        "#   single  one client: the cohort's worst writer by g-0 accuracy",
        "#   double  two clients: the two worst distinct writers",
        "#   dual    ONE client holding both of those writers' rows merged",
        "#",
        "# double and dual are the controlled pair: they hold precisely the same",
        "# rows and differ only in whether the aggregation ever sees them",
        "# separately. Any gap between them is what client BOUNDARIES cost with",
        "# the data held constant - a question the twenty-client stages cannot",
        "# ask, because there the boundaries and the data always move together.",
        "#",
        "# The merge is a union PER PARTITION: train is the union of the two",
        "# writers' train rows, validation of their validation rows, test of",
        "# their test rows. Not a re-split of the combined data - a re-split",
        "# would move rows between partitions and dual would no longer hold what",
        "# double holds.",
        "#",
        "# FULL PARTICIPATION. No sampler: dropping a fifth of a two-client",
        "# federation is not a participation study, it is a coin flip deciding",
        "# whether a round happens at all.",
        "#",
        "# THE CLIENT LISTS ARE GENERATED, NOT TYPED. The writers are a ranking -",
        "# the cohort by accuracy under the frozen g-0, lowest first, ties by",
        "# writer id - and a retyped ranking is one that can silently go stale.",
        "# This generator writes outliers/extreme_{single,double,dual}.json and",
        "# the runs read their populations from them, so each run's provenance",
        "# records exactly which writers it federated:",
        "#",
    ]
    for cell in cells:
        lines.append(
            f"#   {cell['case']:<7} writers {cell['writers']}  "
            f"-> {cell['participants']} client(s) {cell['clients']}"
        )
    lines += [
        "#",
        "# EVALUATION is the ladder's, unchanged: final_evaluation carries the",
        "# per-client own test, the pooled clients test, and preservation as the",
        "# mean over the old book's five fold test partitions. For 'dual' the",
        "# per-client column has ONE row, keyed by the merged id, because there",
        "# is one participant.",
        "#",
        "# REQUIRES THE FISHER EXPORT (the penalty is kd+fisher):",
        '#   G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[\'selected_fold\'])" \\',
        "#             $FOA_RESULTS_DIR/main_v6/g0_selection.json)",
        "#   export G0_FOLD",
        "#",
        f"# {len(cells)} cases x {len(folds)} folds = {len(cells) * len(folds)} tasks.",
    ]
    for cell in cells:
        lines.append(
            f"#   {cell['case']:<7} ~{task_seconds(cell, rounds) / 60:.0f} min/task"
        )
    lines += [
        f"# About {total / 3600:.1f} GPU-h in total. These are tiny federations - one or",
        "# two clients of roughly 68 training rows each - so the per-round",
        "# EVALUATION, which does not shrink with the client count, is most of",
        "# the cost.",
        "#",
        "# HOW TO SUBMIT. Gated on stage B, stage C and the G0_FOLD export.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(cells) * len(folds)} \\",
        "#          $J/phase_e.sbatch $J/stage10_extreme.txt",
        "#",
    ]
    return lines


def readme(cells, combo, folds, rounds, lists) -> str:
    lines = [
        "# Stage 10 extreme cases",
        "",
        f"{len(cells)} cases x {len(folds)} folds = {len(cells) * len(folds)} tasks, "
        f"{rounds} rounds each, one family per task.",
        "",
        f"Run with the winning method only - stage 8's `{combo['id']}`: "
        f"`{combo['agg']['rule']}` aggregation with the `{combo['reg']['space']}` "
        "penalty at "
        + ", ".join(f"`{k}={v:g}`" for k, v in combo["reg"]["hypers"].items())
        + ". The flags are copied from that stage's own line, so the "
        "configuration is identical by construction.",
        "",
        "| case | writers | clients | participants | what it is |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| `{cell['case']}` | {', '.join(f'`{w}`' for w in cell['writers'])} | "
            f"{', '.join(f'`{c}`' for c in cell['clients'])} | "
            f"{cell['participants']} | {cell['note']} |"
        )
    lines += [
        "",
        "## The controlled pair",
        "",
        "`double` and `dual` hold **precisely the same rows**. They differ only "
        "in whether the aggregation ever sees them separately. Any gap between "
        "them is what client boundaries cost with the data held constant - which "
        "the twenty-client stages cannot measure, because there the boundaries "
        "and the data always move together.",
        "",
        "The merge is a union per partition - train with train, validation with "
        "validation, test with test - and not a re-split of the combined data. A "
        "re-split would move rows between partitions and `dual` would stop "
        "holding what `double` holds.",
        "",
        "A merged client is named by joining its members with `+`, so the id "
        "carries its own meaning and survives being written to a pool file, "
        "passed through a worker process and read back out of provenance.",
        "",
        "## Client lists",
        "",
        "Generated, not typed: the writers are the cohort ranked by accuracy "
        "under the frozen g-0, lowest first, ties broken by writer id.",
        "",
        "| case | file |",
        "|---|---|",
    ]
    for case, path in lists.items():
        lines.append(f"| `{case}` | `{path}` |")
    lines += [
        "",
        "## Cost",
        "",
        "| case | ~min/task | tasks |",
        "|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| `{cell['case']}` | {task_seconds(cell, rounds) / 60:.0f} | {len(folds)} |"
        )
    lines += [
        "",
        "One or two clients of roughly 68 training rows each, so the per-round "
        "evaluation - which does not shrink with the client count - is most of "
        "the cost. Full participation, no sampler.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Results root (main_v6).")
    parser.add_argument("--out", default="stage10_extreme.txt")
    parser.add_argument("--readme", default="stage10_extreme_README.md")
    parser.add_argument("--lists-dir", default=None, help="Where the client lists go.")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=FULL_ROUNDS)
    args = parser.parse_args()

    root = Path(args.root)
    with open(root / "outliers" / "clients_acc_on_global.json") as handle:
        accuracies = json.load(handle)
    with open(root / "outliers" / "cohort_worst20.json") as handle:
        cohort = json.load(handle)
    cohort = cohort["clients"] if isinstance(cohort, dict) else cohort

    ranked = rank_cohort(accuracies, list(cohort))
    cells = extreme_cells(ranked)
    combo = winning_combo(root)

    lists = write_client_lists(
        root, cells, Path(args.lists_dir) if args.lists_dir else root / "outliers"
    )

    lines = header(cells, combo, args.folds, args.rounds)
    for cell in cells:
        lines.append(f"# --- {cell['case']}: {cell['note']}")
        for fold in args.folds:
            lines.append(task_line(cell, combo, fold, args.rounds))

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.readme).write_text(readme(cells, combo, args.folds, args.rounds, lists))
    runnable = [line for line in lines if line and not line.startswith("#")]
    print(f"wrote {args.out}: {len(runnable)} task lines "
          f"({len(cells)} cases x {len(args.folds)} folds)")
    for case, path in lists.items():
        print(f"  {case:<7} -> {path}")
    for cell in cells:
        print(f"  {cell['case']:<7} writers {cell['writers']} "
              f"clients {cell['clients']} "
              f"~{task_seconds(cell, args.rounds) / 60:.1f} min/task")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
