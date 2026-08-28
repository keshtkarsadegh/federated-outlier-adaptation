"""
Emit the stage-7 regularisation screening task file and its README.

    python tools/make_stage7_screen.py [--out PATH] [--readme PATH] [--folds 1 2 3 4 5]

Reads nothing but the cell table in
:mod:`federated_outlier_adaptation.training.reg_cells`, so it is a login-node
job: two text files, no data touched.

One line per (cell, fold).  Every line is stage 6's screening protocol with one
thing changed - the penalty added to the client's loss - so a difference between
two lines is the penalty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_outlier_adaptation.training.reg_cells import (
    ANCHOR,
    FAMILIES,
    FULL_ROUNDS,
    SCREEN_ROUNDS,
    SKIPPED_CELLS,
    cells_by_method,
    control_cell,
    screen_cells,
)

RESULTS = "$FOA_RESULTS_DIR/main_v6"
BOOKS = f"{RESULTS}/fold_books"
POOLS = f"{RESULTS}/outliers"

#: The winning g-0 fold's Fisher directory.  Written by that fold's own
#: ``global-train`` run; ``$G0_FOLD`` is exported before submitting, see header.
FISHER_DIR = f"{RESULTS}/g0_fold$G0_FOLD/global_results/fisher"


def _exact(value: float) -> str:
    """
    A float that parses back to the same double.

    ``{:g}`` truncates to six significant figures, which is fine for a label and
    wrong for a value: the distillation lam of alpha=0.3 is 2.3333333333333335,
    and ``2.33333`` is a different objective.  ``repr`` gives the shortest
    representation that round-trips.
    """
    return repr(float(value))


def set_tokens(cell: dict) -> str:
    """The cell's penalty as ``--set NAME=VALUE`` tokens, in a stable order."""
    if cell["trainer"] != "AnchoredTrainer":
        return ""
    parts = [f"space={cell['space']}", f"anchor={ANCHOR}"]
    for name in ("lam", "T", "mix"):
        if name in cell["hypers"]:
            parts.append(f"{name}={_exact(cell['hypers'][name])}")
    if cell["needs_fisher"]:
        parts.append(f"fisher_path={FISHER_DIR}")
    return "--set " + " ".join(parts)


def task_line(cell: dict, fold: int, rounds: int, sampler_seed: int) -> str:
    """One `foa final` invocation: stage 6's protocol, this cell's penalty."""
    parent = f"coh7_reg_{cell['id']}_fold{fold}"
    # --fedprox-convention must precede --set, which is nargs="*" and swallows
    # everything up to the next flag.
    convention = " --fedprox-convention" if cell["fedprox"] else ""
    penalty = set_tokens(cell)
    return (
        f"foa final --results-dir {RESULTS} --resolution 28 --classes all"
        f" --model fedavg_cnn --trainer {cell['trainer']} --parent {parent}"
        f" --aggregation fedavg --init global --global-name g0"
        f" --outliers-file {POOLS}/cohort_worst20.json"
        f" --fold-book {BOOKS}/cohort20.foldbook.npz --fold {fold}"
        f" --old-book {BOOKS}/old_data.foldbook.npz"
        f" --old-clients-file {POOLS}/old_data.json --old-fold all"
        f" --policy uniform --clients-per-round 16 --sampler-seed {sampler_seed}"
        f" --track-clients --rounds {rounds} --epochs 5 --batch-size 64"
        f" --eval-batch-size 256 --save-final-model --seed {fold}"
        f" --outer-workers 2 --inner-workers 1"
        + convention
        + (f" {penalty}" if penalty else "")
    )


def header(cells, folds, rounds) -> list[str]:
    grouped = cells_by_method()
    lines = [
        "# stage7_screen.txt - STAGE 7 of plan v6: WHAT THE REGULARISER IS WORTH.",
        "#",
        "# Stage 6 asked what the SERVER rule is worth. This asks the other half:",
        "# what the CLIENT-SIDE PENALTY is worth. Every line runs stage 6's",
        "# screening protocol - cohort of 20, 16 per round, E=5, batch 64, the",
        "# outlier fold book, the winner g-0 as the only init, plain FedAvg - and",
        "# changes only the term added to the client's loss. The anchor is the",
        "# frozen g-0 in every cell: the penalty measures distance from the model",
        "# the study is trying not to forget.",
        "#",
        "# BOTH SCHEDULES PER TASK. --aggregation fedavg runs the parallel family",
        "# and the cyclic one in one job. A penalty that helps the server average",
        "# and one that helps a sequential walk are different findings, so the",
        "# selector ranks them separately and stage7_full.txt names which family's",
        "# result each winner came from.",
        "#",
        f"# SCREENING HORIZON: {rounds} ROUNDS, NOT {FULL_ROUNDS}. Nothing here is a reported",
        "# number - a 25-round result is a ranking signal. Winners are re-run at",
        f"# {FULL_ROUNDS} rounds by stage7_full.txt, which tools/select_reg_screen.py emits.",
        "#",
        "# ONE SUBSTITUTION - REQUIRED BEFORE SUBMITTING.",
        "#",
        "# The Fisher-weighted cells need the diagonal of the SHIPPED model. This",
        "# protocol never wrote one under the name the artefact resolver looks for",
        "# (global_results/fisher_g0); what exists is the per-fold diagonal each",
        "# global-train run wrote next to its own checkpoint. The 16 fisher and",
        "# fisher_scaled cells therefore point at the WINNING fold's directory,",
        "# and that fold number has to be in the environment:",
        "#",
        '#   G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[\'selected_fold\'])" \\',
        "#             $FOA_RESULTS_DIR/main_v6/g0_selection.json)",
        "#   export G0_FOLD",
        "#",
        "# That is the right object - the Fisher of g-0 on the data g-0 was",
        "# trained on - and it costs no recomputation. Without the export those",
        "# 80 array elements fail at startup on a path with an empty component.",
        "#",
        f"# {len(cells)} cells x {len(folds)} folds = {len(cells) * len(folds)} tasks:",
    ]
    for method, group in grouped.items():
        lines.append(f"#   {method:<14} {len(group):>3} cells   {len(group) * len(folds):>4} tasks")
    lines += [
        "#",
        "# Every cell id is decoded in stage7_screen_README.md, generated next to",
        "# this file. Outputs land under coh7_reg_<cell>_fold<k>.",
        "#",
        "# THE NO-PENALTY CONTROL is param_l2_mu0: the trainer returns zero for any",
        "# lam <= 0, so mu=0 is an exact no-op and not a near-miss of one. A second,",
        "# redundant control (control_none, plain BaseTrainer) is emitted into",
        "# stage7_full.txt: if the two disagree by more than run-to-run noise, the",
        "# penalty machinery is doing something while switched off, and that is",
        "# worth learning from a cheap cell rather than from a reviewer.",
        "#",
        "# HOW TO SUBMIT. Gated on stage B (books, client lists) and stage C (g-0",
        "# and its per-fold Fisher), plus the G0_FOLD export above.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(cells) * len(folds)}%40 \\",
        "#          $J/phase_e.sbatch $J/stage7_screen.txt",
        "#",
    ]
    return lines


def readme(cells, folds, rounds) -> str:
    from federated_outlier_adaptation.training.reg_cells import (
        HYBRID_MIXES,
        kd_lam_of_alpha,
    )

    grouped = cells_by_method()
    lines = [
        "# Stage 7 regularisation cells",
        "",
        f"{len(cells)} cells x {len(folds)} folds = {len(cells) * len(folds)} tasks, "
        f"{rounds} rounds each, both families per task.",
        "",
        "Every cell runs stage 6's screening protocol and changes only the penalty "
        "added to the client's loss. The anchor is the frozen g-0 everywhere. "
        "These are ranking signals, not reported numbers; winners are re-run at "
        f"{FULL_ROUNDS} rounds.",
        "",
        "| cell | method | space | hyperparameters | what it is |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        hypers = ", ".join(f"{k}={v:g}" for k, v in cell["hypers"].items()) or "-"
        lines.append(
            f"| `{cell['id']}` | {cell['method']} | `{cell['space']}` | "
            f"`{hypers}` | {cell['note']} |"
        )
    lines += [
        "",
        "## Counts by method",
        "",
        "| method | cells | tasks |",
        "|---|---|---|",
    ]
    for method, group in grouped.items():
        lines.append(f"| {method} | {len(group)} | {len(group) * len(folds)} |")
    lines += [
        "",
        "## The distillation alpha convention",
        "",
        "The grid is written in the literature's `alpha` and executed in the "
        "trainer's `lam`. `DistillationTrainer` minimises "
        "`alpha*CE + (1-alpha)*T^2*KL`; `AnchoredTrainer` minimises "
        "`CE + lam*T^2*KL`. Dividing the first by `alpha` gives the second with "
        "`lam = (1-alpha)/alpha`, so the two have the same minimiser and the same "
        "gradient direction and differ only by a constant factor on the effective "
        "learning rate.",
        "",
        "| alpha | lam |",
        "|---|---|",
    ]
    from federated_outlier_adaptation.training.reg_cells import KD_ALPHAS

    for alpha in KD_ALPHAS:
        lines.append(f"| {alpha:g} | {kd_lam_of_alpha(alpha):.4g} |")
    lines += [
        "",
        "## Deliberately not screened",
        "",
        "| grid point | why |",
        "|---|---|",
    ]
    for name, reason in SKIPPED_CELLS.items():
        lines.append(f"| `{name}` | {reason} |")
    lines += [
        "",
        f"The hybrid's three cells (blend rho in {{{', '.join(f'{m:g}' for m in HYBRID_MIXES)}}}) "
        "are emitted by `tools/emit_stage7_hybrid.py` once the kd and fisher "
        "winners are known.",
        "",
        "## The Fisher directory",
        "",
        "The 16 Fisher-weighted cells point at "
        "`main_v6/g0_fold$G0_FOLD/global_results/fisher`, the diagonal the "
        "winning fold's own `global-train` run wrote. Export `G0_FOLD` from "
        "`g0_selection.json` before submitting; see the task file's header.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="stage7_screen.txt")
    parser.add_argument("--readme", default="stage7_screen_README.md")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=SCREEN_ROUNDS)
    args = parser.parse_args()

    cells = screen_cells()
    lines = header(cells, args.folds, args.rounds)
    for index, cell in enumerate(cells):
        lines.append(f"# --- {cell['id']}: {cell['note']}")
        for fold in args.folds:
            lines.append(task_line(cell, fold, args.rounds, 72000 + index * 10 + fold))

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.readme).write_text(readme(cells, args.folds, args.rounds))
    runnable = [line for line in lines if line and not line.startswith("#")]
    print(f"wrote {args.out}: {len(runnable)} task lines "
          f"({len(cells)} cells x {len(args.folds)} folds)")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
