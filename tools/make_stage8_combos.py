"""
SUPERSEDED - see tools/study_emit.py combos.

Its ``RESULTS`` constant is ``$FOA_RESULTS_DIR/main_v6``, a root from a study
that has since been deleted, and ``tools/make_study_sbatch.py`` refuses any
task file naming it. The combination stage this study ran is
``jobs/s20_combos4.txt``, emitted by

    python tools/study_emit.py combos --root "$FOA_STUDY_DIR" \
        --out <path> --expect 90

and reproducible from it byte for byte. The seeding is the other reason not
to reuse this file: the current stage seeds a pair on a hash of its identity,
not on its position in a shortlist, so a shortlist can be re-ranked without
moving a seed. Kept as a record.

Emit the stage-8 combination task file and the README that decodes it.

    python tools/make_stage8_combos.py [--out PATH] [--readme PATH] [--folds 1 2 3 4 5]

Reads nothing but the combination table in
:mod:`federated_outlier_adaptation.training.combo_cells`, which in turn reads the
stage-6 and stage-7 cell tables.  Two text files, no data touched: a login-node
job.

Every coefficient is inherited from those tables by cell id and formatted by the
same helpers stages 6 and 7 use, so a number in this file is the same double as
the number in the file it came from.  Nothing here is retyped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_stage6_screen import flag_tokens
from make_stage7_screen import set_tokens

from federated_outlier_adaptation.training.combo_cells import (
    AGG_CELLS,
    FULL_ROUNDS,
    HYBRID_FISHER_CELL,
    HYBRID_KD_CELL,
    HYBRID_MIX,
    REG_CELLS,
    combo_cells,
    combos_by_schedule,
)

RESULTS = "$FOA_RESULTS_DIR/main_v6"
BOOKS = f"{RESULTS}/fold_books"
POOLS = f"{RESULTS}/outliers"


#: Participants this stage draws, and the count its client-scaled knobs
#: are resolved against.
CLIENTS_PER_ROUND = 16


def task_line(combo: dict, fold: int, rounds: int, sampler_seed: int) -> str:
    """One `foa final` invocation: one server rule, one client penalty."""
    parent = f"coh8_combo_{combo['id']}_fold{fold}"
    # RESOLVE FIRST. A cell may hold the stage-independent form of a knob - a
    # half-life in rounds, a retention or a trim count over clients - and
    # flag_tokens has no flag for those names, so passing the stored dictionary
    # drops the coefficient and the arm runs as though the knob were unset.
    # This file emits its own participation, so that is the count to resolve
    # against, not the study's.
    import dataclasses

    from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01
    from federated_outlier_adaptation.training.study_lines import resolve_agg_flags

    server = flag_tokens(resolve_agg_flags(
        combo["agg"]["flags"],
        dataclasses.replace(DIGITS_STUDY01, clients_per_round=CLIENTS_PER_ROUND),
        rounds,
    ))
    # --set is nargs="*" and swallows everything up to the next flag, so it goes
    # last and every other flag goes before it.
    penalty = set_tokens(combo["reg"])
    return (
        f"foa final --results-dir {RESULTS} --resolution 28 --classes all"
        f" --model fedavg_cnn --trainer AnchoredTrainer --parent {parent}"
        f" --aggregation {combo['agg']['rule']} --extended-aggregations"
        f" --init global --global-name g0"
        f" --outliers-file {POOLS}/cohort_worst20.json"
        f" --fold-book {BOOKS}/cohort20.foldbook.npz --fold {fold}"
        f" --old-book {BOOKS}/old_data.foldbook.npz"
        f" --old-clients-file {POOLS}/old_data.json --old-fold all"
        f" --policy uniform --clients-per-round {CLIENTS_PER_ROUND} --sampler-seed {sampler_seed}"
        f" --track-clients --rounds {rounds} --epochs 5 --batch-size 64"
        f" --eval-batch-size 256 --save-final-model --seed {fold}"
        f" --outer-workers 1 --inner-workers 1"
        + (f" {server}" if server else "")
        + (f" {penalty}" if penalty else "")
    )


def header(combos, folds, rounds) -> list[str]:
    grouped = combos_by_schedule()
    fisher = sum(1 for c in combos if c["reg"]["needs_fisher"])
    return [
        "# stage8_combos.txt - STAGE 8 of plan v6: DO THE TWO HALVES COMPOSE?",
        "#",
        "# Stage 6 varied the SERVER rule with no penalty on the clients. Stage 7",
        "# varied the CLIENT penalty under plain FedAvg. Neither can say whether",
        "# the two compose - whether a server rule that preserves and a penalty",
        "# that preserves preserve twice, or whether they are two names for the",
        "# same restraint and stacking them buys nothing. That is the only",
        "# question here, so this is a cross and not another sweep.",
        "#",
        "# ONE COMBINATION IS ONE FAMILY. Unlike stages 6 and 7, these lines do",
        "# NOT run --aggregation fedavg. Every rule named here lives in exactly",
        "# one (scenario, metadata) family, so the aggregation decides the",
        "# schedule and a task produces one result, not two. The concurrent and",
        "# sequential halves are separate experiments with their own winners.",
        "#",
        "# FULL HORIZON: 100 ROUNDS. These are reported numbers, measured under",
        "# the same protocol as stages 5, 6 and 7 - cohort of 20, 16 per round,",
        "# E=5, batch 64, the outlier fold book, init and anchor and teacher all",
        "# the frozen winner g-0 - so all four stages are directly comparable.",
        "#",
        "# THE COEFFICIENTS ARE INHERITED, NOT RETYPED. Every one is looked up",
        "# from the stage-6 and stage-7 cell tables by cell id and formatted by",
        "# the same helpers those stages use. Retyping a float is how",
        "# 2.3333333333333335 becomes 2.33333 - a different objective wearing the",
        "# same label - which stage 7 already caught once.",
        "#",
        f"# THE HYBRID differs between the schedules: mix weights the KD half",
        f"# against the Fisher half, and the two schedules chose differently, so",
        f"# concurrent carries mix={HYBRID_MIX['concurrent']:g} and sequential",
        f"# mix={HYBRID_MIX['sequential']:g}. Both are built from the same two",
        f"# stage-7 winners ({HYBRID_KD_CELL} and {HYBRID_FISHER_CELL}) through the",
        "# same constructor stage 7's own emitter uses.",
        "#",
        "# ONE SUBSTITUTION - REQUIRED BEFORE SUBMITTING.",
        "#",
        f"# {fisher} of the {len(combos)} combinations use the kd+fisher hybrid, which needs",
        "# the Fisher diagonal of the shipped model. v6 never wrote one under the",
        "# name the artefact resolver looks for (global_results/fisher_g0); what",
        "# exists is the per-fold diagonal each global-train run wrote next to its",
        "# own checkpoint. Those lines point at the WINNING fold's directory, so",
        "# that fold number has to be in the environment:",
        "#",
        '#   G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[\'selected_fold\'])" \\',
        "#             $FOA_RESULTS_DIR/main_v6/g0_selection.json)",
        "#   export G0_FOLD",
        "#",
        f"# Without it those {fisher * len(folds)} array elements fail at startup on a path with",
        "# an empty component.",
        "#",
        f"# {len(combos)} combinations x {len(folds)} folds = {len(combos) * len(folds)} tasks:",
        f"#   concurrent  {len(grouped['concurrent'])} combinations   "
        f"{len(grouped['concurrent']) * len(folds)} tasks",
        f"#   sequential  {len(grouped['sequential'])} combinations   "
        f"{len(grouped['sequential']) * len(folds)} tasks",
        "#",
        "# Every combination id is decoded in stage8_combos_README.md, generated",
        "# next to this file. Outputs land under coh8_combo_<agg>_<reg>_fold<k>.",
        "#",
        "# HOW TO SUBMIT. Gated on stage B (books, client lists) and stage C (g-0",
        "# and its per-fold Fisher), plus the G0_FOLD export above.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(combos) * len(folds)}%30 \\",
        "#          $J/phase_e.sbatch $J/stage8_combos.txt",
        "#",
    ]


def readme(combos, folds, rounds) -> str:
    grouped = combos_by_schedule()
    lines = [
        "# Stage 8 combinations",
        "",
        f"{len(combos)} combinations x {len(folds)} folds = {len(combos) * len(folds)} "
        f"tasks, {rounds} rounds each, one family per task.",
        "",
        "Three server rules crossed with three client penalties, per schedule, at "
        "the full horizon. The question is whether the two halves compose: stage 6 "
        "found the server rules with no penalty, stage 7 found the penalties under "
        "plain FedAvg, and neither can say whether stacking them buys anything.",
        "",
        "Every coefficient is inherited from the stage-6 and stage-7 cell tables by "
        "cell id - nothing in this table was retyped.",
        "",
        "| combination | schedule | aggregation | server flags | penalty | penalty hypers |",
        "|---|---|---|---|---|---|",
    ]
    for combo in combos:
        server = flag_tokens(combo["agg"]["flags"]) or "-"
        hypers = ", ".join(f"{k}={v:g}" for k, v in combo["reg"]["hypers"].items()) or "-"
        lines.append(
            f"| `{combo['id']}` | {combo['schedule']} | `{combo['agg']['rule']}` | "
            f"`{server}` | `{combo['reg']['space']}` | `{hypers}` |"
        )
    lines += [
        "",
        "## Where each half came from",
        "",
        "| schedule | server rules (stage-6 cells) | penalties (stage-7 cells) |",
        "|---|---|---|",
    ]
    for schedule in ("concurrent", "sequential"):
        aggs = ", ".join(f"`{name}`" for name in AGG_CELLS[schedule])
        regs = ", ".join(
            [f"`hybrid_mix{HYBRID_MIX[schedule]:g}`"]
            + [f"`{name}`" for name in REG_CELLS]
        )
        lines.append(f"| {schedule} | {aggs} | {regs} |")
    lines += [
        "",
        "## The hybrid",
        "",
        f"Built from `{HYBRID_KD_CELL}` (the KD half: `lam`, `T`) and "
        f"`{HYBRID_FISHER_CELL}` (the Fisher half), blended by `mix` - "
        f"{HYBRID_MIX['concurrent']:g} on the concurrent schedule, "
        f"{HYBRID_MIX['sequential']:g} on the sequential one. The objective is "
        "`lam * (mix * KD + (1 - mix) * Fisher)`, so `mix = 1` would reproduce "
        "the KD cell exactly.",
        "",
        "The six hybrid lines need `fisher_path`, which resolves through the "
        "`$G0_FOLD` export - see the task file's header.",
        "",
        "## Counts by schedule",
        "",
        "| schedule | combinations | tasks |",
        "|---|---|---|",
    ]
    for schedule, group in grouped.items():
        lines.append(f"| {schedule} | {len(group)} | {len(group) * len(folds)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="stage8_combos.txt")
    parser.add_argument("--readme", default="stage8_combos_README.md")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=FULL_ROUNDS)
    args = parser.parse_args()

    combos = combo_cells()
    lines = header(combos, args.folds, args.rounds)
    for index, combo in enumerate(combos):
        lines.append(f"# --- {combo['id']}: {combo['note']}")
        for fold in args.folds:
            lines.append(task_line(combo, fold, args.rounds, 82000 + index * 10 + fold))

    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.readme).write_text(readme(combos, args.folds, args.rounds))
    runnable = [line for line in lines if line and not line.startswith("#")]
    print(f"wrote {args.out}: {len(runnable)} task lines "
          f"({len(combos)} combinations x {len(args.folds)} folds)")
    print(f"wrote {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
