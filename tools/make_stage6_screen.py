"""
SUPERSEDED as an emitter - see tools/make_digits_p11.py.

Its ``RESULTS`` constant is ``$FOA_RESULTS_DIR/main_v6``, a root from a study
that has since been deleted, and ``tools/make_study_sbatch.py`` refuses any
task file naming it. The aggregation screen this study ran is
``jobs/s09_agg_screen2.txt``, emitted by ``tools/make_digits_p11.py
--jobs-dir <dir>`` and reproducible from it byte for byte.

``task_line`` and ``flag_tokens`` here are still imported - by this file's own
selector and by the stage-8 emitter, both likewise superseded - so the module
is kept rather than deleted. The config-driven replacement for the line
builder is ``federated_outlier_adaptation.training.study_lines``, which says
in its own header why it exists.

Emit the stage-6 screening task file and the README that decodes its cell ids.

    python tools/make_stage6_screen.py [--out PATH] [--readme PATH] [--folds 1 2 3 4 5]

Reads nothing but the cell table in
:mod:`federated_outlier_adaptation.training.agg_cells`, so it is a login-node
job: it writes two text files and touches no data.

One line per (cell, fold).  Every line is stage 5's protocol with one thing
changed - the server rule and its coefficients - so a difference between two
lines is the rule and not the arrangement around it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_outlier_adaptation.training.agg_cells import (
    FULL_ROUNDS,
    SCREEN_ROUNDS,
    SKIPPED_CELLS,
    cells_by_path,
    screen_cells,
)

RESULTS = "$FOA_RESULTS_DIR/main_v6"
BOOKS = f"{RESULTS}/fold_books"
POOLS = f"{RESULTS}/outliers"

#: Cell flag name -> command-line flag.  A boolean flag takes no value.
FLAGS = {
    "server_eta": "--server-eta",
    "weighting": "--weighting",
    "trim_frac": "--trim-frac",
    "server_anchor": "--server-anchor",
    "server_beta": "--server-beta",
    "server_lr": "--server-lr",
    "server_tau": "--server-tau",
    "seq_mix_alpha": "--seq-mix-alpha",
    "client_order": "--client-order",
    "stop_when_global_below_clients": "--stop-when-global-below-clients",
}

BOOLEAN_FLAGS = {"stop_when_global_below_clients"}


def flag_tokens(flags: dict) -> str:
    """The cell's coefficients as command-line tokens, in a stable order."""
    parts = []
    for name in FLAGS:
        if name not in flags:
            continue
        value = flags[name]
        if name in BOOLEAN_FLAGS:
            if value:
                parts.append(FLAGS[name])
            continue
        # repr, not {:g}. Six significant figures is right for a label and
        # wrong for a coefficient: a knob derived from a half-life or a
        # retention lands on a long float, and rounding it changes the run
        # while leaving the task file looking correct.
        parts.append(f"{FLAGS[name]} {value!r}" if isinstance(value, float)
                     else f"{FLAGS[name]} {value}")
    return " ".join(parts)


def task_line(cell: dict, fold: int, rounds: int, sampler_seed: int) -> str:
    """One `foa final` invocation: stage 5's protocol, this cell's rule."""
    parent = f"coh6_agg_{cell['id']}_fold{fold}"
    extra = flag_tokens(cell["flags"])
    return (
        f"foa final --results-dir {RESULTS} --resolution 28 --classes all"
        f" --model fedavg_cnn --trainer BaseTrainer --parent {parent}"
        f" --aggregation {cell['rule']} --extended-aggregations"
        f" --init global --global-name g0"
        f" --outliers-file {POOLS}/cohort_worst20.json"
        f" --fold-book {BOOKS}/cohort20.foldbook.npz --fold {fold}"
        f" --old-book {BOOKS}/old_data.foldbook.npz"
        f" --old-clients-file {POOLS}/old_data.json --old-fold all"
        f" --policy uniform --clients-per-round 16 --sampler-seed {sampler_seed}"
        f" --track-clients --rounds {rounds} --epochs 5 --batch-size 64"
        f" --eval-batch-size 256 --save-final-model --seed {fold}"
        f" --outer-workers 2 --inner-workers 1"
        + (f" {extra}" if extra else "")
    )


def header(cells, folds, rounds) -> list[str]:
    grouped = cells_by_path()
    return [
        "# stage6_screen.txt - STAGE 6 of plan v6: WHAT THE AGGREGATION RULE IS WORTH.",
        "#",
        "# Stage 5 fixed the server rule at plain FedAvg and varied nothing, so it",
        "# cannot say how much of the adaptation - or of the forgetting - was the",
        "# rule's doing. This file varies the rule and nothing else: same twenty",
        "# writers, sixteen of twenty per round, E=5, batch 64, the same outlier",
        "# fold book, the winner g-0 as the ONLY initialisation. A difference",
        "# between two lines is the rule.",
        "#",
        f"# SCREENING HORIZON: {rounds} ROUNDS, NOT {FULL_ROUNDS}. Nothing in this file is a",
        "# reported number. A 25-round result is a ranking signal - reading it as a",
        "# result is reading a race at the quarter mark. The winners are re-run at",
        f"# {FULL_ROUNDS} rounds by stage6_full.txt, which tools/select_agg_screen.py emits",
        "# from these results once the owner has set the budget epsilon.",
        "#",
        f"# {len(cells)} cells x {len(folds)} folds = {len(cells) * len(folds)} tasks:",
        f"#   concurrent {len(grouped['concurrent'])}   sequential "
        f"{len(grouped['sequential'])}   control {len(grouped['control'])}",
        "#",
        "# Every cell id is decoded in stage6_screen_README.md, which is generated",
        "# next to this file. Outputs land under coh6_agg_<cell>_fold<k>.",
        "#",
        "# EACH LINE writes final_evaluation into its accuracies_0.json: the pooled",
        "# clients accuracy on fold-k test rows, the per-client column, and the old",
        "# data's five folds with their mean and spread. Those are what the",
        "# selector ranks on - never the per-round curve's last point.",
        "#",
        "# HOW TO SUBMIT. Gated on stage B (books, client lists) and stage C (g-0).",
        "# Nothing here writes to a shared file, so the array can run wide:",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(cells) * len(folds)}%40 \\",
        "#          $J/phase_e.sbatch $J/stage6_screen.txt",
        "#",
    ]


def readme(cells, folds, rounds) -> str:
    grouped = cells_by_path()
    lines = [
        "# Stage 6 screening cells",
        "",
        f"{len(cells)} cells x {len(folds)} folds = {len(cells) * len(folds)} tasks, "
        f"{rounds} rounds each.",
        "",
        "Every cell runs stage 5's protocol - cohort of 20, 16 per round, E=5, "
        "batch 64, CV-5 outlier book, init from the winner g-0 - and changes only "
        "the server rule and its coefficients. The horizon is a screening horizon: "
        f"these are ranking signals, not reported numbers. Winners are re-run at "
        f"{FULL_ROUNDS} rounds.",
        "",
        "| cell | path | rule | coefficients | what it is |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        coeffs = flag_tokens(cell["flags"]) or "-"
        lines.append(
            f"| `{cell['id']}` | {cell['path']} | `{cell['rule']}` | "
            f"`{coeffs}` | {cell['note']} |"
        )
    lines += [
        "",
        "## Deliberately not run",
        "",
        "Each of these is algebraically the same update as a cell that is run. "
        "Running both would spend GPU time reproducing run-to-run noise and then "
        "report it as two settings.",
        "",
        "| grid point | duplicates |",
        "|---|---|",
    ]
    for name, reason in sorted(SKIPPED_CELLS.items()):
        lines.append(f"| `{name}` | {reason} |")
    lines += [
        "",
        "## Counts by path",
        "",
        "| path | cells | tasks |",
        "|---|---|---|",
    ]
    for path, group in grouped.items():
        lines.append(f"| {path} | {len(group)} | {len(group) * len(folds)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="stage6_screen.txt")
    parser.add_argument("--readme", default="stage6_screen_README.md")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=SCREEN_ROUNDS)
    args = parser.parse_args()

    cells = screen_cells()
    lines = header(cells, args.folds, args.rounds)
    for index, cell in enumerate(cells):
        lines.append(f"# --- {cell['id']}: {cell['note']}")
        for fold in args.folds:
            # One sampler seed per (cell, fold): the participation pattern is
            # reproducible and two cells never share one.
            lines.append(task_line(cell, fold, args.rounds, 62000 + index * 10 + fold))

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.readme).write_text(readme(cells, args.folds, args.rounds))
    runnable = [line for line in lines if line and not line.startswith("#")]
    print(f"wrote {args.out}: {len(runnable)} task lines "
          f"({len(cells)} cells x {len(args.folds)} folds)")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
