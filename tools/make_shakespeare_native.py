"""
Stage B: the digit study's screens, run natively on Shakespeare.

    python tools/make_shakespeare_native.py --jobs-dir "$FOA_PROJECT_DIR/jobs/v4"

Writes the static task files and a chain script: a login-node job, no GPU.

What this answers that stage A does not
---------------------------------------
Stage A asks whether the transferred winners recover when their strength is
rescaled.  This asks a different question: **what would this task have chosen on
its own?**  The same 171-cell aggregation table and 125-cell regularisation
table the digit study screened, the same two-step protocol, the same
test-ranked top-three rule - selected from scratch on the Shakespeare cohort.

The comparison the two produce is the point of the chapter.  If the native
winners are the transferred ones, the digit selection generalises.  If they are
not, the gap between "the digit winner here" and "the native winner here" is the
price of transferring, measured rather than asserted.

Folds
-----
Defaults to **fold 1 only**, which is this study's rule: one g-0, one client
split, no cross-validation.  ``--folds 1 2 3 4 5`` restores the digit study's
cross-validated shape and multiplies every count and every cost by five.  Both
are one flag apart because the choice is the owner's and the numbers differ by
enough to matter - see the cost table this prints.

The Fisher
----------
Sixteen of the 125 regularisation cells are Fisher-weighted (EWC and the
scaled variant), and they are part of the story rather than something to drop.
They need the Fisher of *this* study's g-0, and it already exists: a
``global-train`` writes ``global_results/fisher/{fisher.pt,global_params.pt}``
next to the model it trains, for any provider - verified on a Shakespeare g-0.
So no Fisher-generation task is needed; what was needed was for the path to stop
naming ``$G0_FOLD``, which a study with one g-0 has no selection record to
supply.  :func:`~...study_lines.fisher_dir` now returns
``$FOA_STUDY_DIR/g0/shakespeare/global_results/fisher`` for this study.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from federated_outlier_adaptation.training import agg_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import SHAKESPEARE_STUDY01 as CFG

#: Seed blocks, one per stage, all clear of every range any study has emitted.
#: s01 runs used 800001-800012 and stage A 801000-801009.
SEEDS = {"agg": 810000, "reg": 812000}

#: Where the chain writes the files it generates.  Inside the study, because the
#: runner refuses a path-valued flag that points anywhere else - and because a
#: file generated from this study's results belongs with them.
GEN_DIR = f"{SL.ROOT}/jobs"

#: Measured on this project's own Shakespeare federated runs (results_v2, m=10,
#: batch 64, 100 rounds): a ten-client round costs 14.3 s on the selected small
#: clients and 36.7 s on the larger rule-based pool, and a penalty trainer adds
#: about six percent over the plain one.  The cohort here is the ten *worst*
#: users under g-0, which is the same kind of population as the 14.3 s case, so
#: that is the centre and the pool figure is the upper bound.
ROUND_SECONDS = (14.3, 36.7)


def _seed(block: str, index: int, fold: int) -> int:
    return SEEDS[block] + index * 10 + fold


def agg_screen(folds) -> List[str]:
    """The 171-cell aggregation table at the screening horizon."""
    return [
        SL.agg_line(CFG, cell, fold, SL.SCREEN_ROUNDS, _seed("agg", index, fold))
        for index, cell in enumerate(agg_cells.screen_cells())
        for fold in folds
    ]


def reg_screen(folds) -> List[str]:
    """The 125-cell regularisation table at the screening horizon."""
    return [
        SL.reg_line(CFG, cell, fold, SL.SCREEN_ROUNDS, _seed("reg", index, fold))
        for index, cell in enumerate(reg_cells.screen_cells())
        for fold in folds
    ]


def counts(folds) -> Dict[str, int]:
    """Every stage's task count, from the tables rather than from memory."""
    n = len(folds)
    return {
        "agg_screen": len(agg_cells.screen_cells()) * n,
        "agg_full": len(SL.AGG_METHODS) * n,
        "reg_screen": len(reg_cells.screen_cells()) * n,
        "reg_hybrid": len(reg_cells.HYBRID_MIXES) * n,
        "reg_full": len(SL.REG_METHODS) * len(SL.FAMILIES) * n,
        "combos": SL.COMBO_TOP_K * SL.COMBO_TOP_K * len(SL.FAMILIES) * n,
    }


def generator_lines(folds) -> Dict[str, List[str]]:
    """
    The in-chain selection steps.

    Each is the hardened emitter: ``--expect`` is mandatory, and a selection
    resting on no measured result is refused outright rather than falling back
    to a row's first cell. ``--rank-by test`` is the owner's documented rule.
    """
    n = counts(folds)
    emit = (f"$FOA_PYTHON $FOA_PROJECT_DIR/repo_foa/tools/study_emit.py"
            f" {{what}} --study {CFG.name} --root {SL.ROOT}")
    return {
        "gen_agg_full": [
            emit.format(what="agg-full")
            + f" --out {GEN_DIR}/s01b_agg_full.txt --expect {n['agg_full']}"
        ],
        "gen_agg_top3": [
            emit.format(what="agg-top3")
            + f" --out {SL.TABLES}/p12_agg_top3.json --rank-by test --expect 1"
        ],
        "gen_reg": [
            emit.format(what="reg-hybrid")
            + f" --out {GEN_DIR}/s01b_reg_hybrid.txt --expect {n['reg_hybrid']}",
            emit.format(what="reg-full")
            + f" --out {GEN_DIR}/s01b_reg_full.txt --expect {n['reg_full']}",
        ],
        "gen_reg_top3": [
            emit.format(what="reg-top3")
            + f" --out {SL.TABLES}/p14_reg_top3.json --rank-by test --expect 1"
        ],
        "gen_combos": [
            emit.format(what="combos")
            + f" --out {GEN_DIR}/s01b_combos.txt --expect {n['combos']}"
        ],
        "gen_winner": [
            emit.format(what="stage-winner")
            + f" --out {SL.TABLES}/p15_stage_winner.json --rank-by test --expect 1"
        ],
    }


def cost(folds) -> List[Tuple[str, int, int, float, float]]:
    """``(stage, tasks, rounds, low GPU-h, high GPU-h)`` from the measured band."""
    n = counts(folds)
    plan = [
        ("agg screen", n["agg_screen"], SL.SCREEN_ROUNDS),
        ("agg full", n["agg_full"], SL.FULL_ROUNDS),
        ("reg screen", n["reg_screen"], SL.SCREEN_ROUNDS),
        ("reg hybrid", n["reg_hybrid"], SL.FULL_ROUNDS),
        ("reg full", n["reg_full"], SL.FULL_ROUNDS),
        ("combos", n["combos"], SL.FULL_ROUNDS),
    ]
    return [
        (name, tasks, rounds,
         tasks * rounds * ROUND_SECONDS[0] / 3600,
         tasks * rounds * ROUND_SECONDS[1] / 3600)
        for name, tasks, rounds in plan
    ]


def header(name: str, what: str, tasks: List[str], folds) -> List[str]:
    return [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        f"# Shakespeare_study01 / stage B: {what}.",
        "#",
        "# THE NATIVE QUESTION. Stage A asks whether the transferred winners",
        "# recover when rescaled. This asks what this task would have chosen on",
        "# its own: the same tables, the same two-step protocol, the same",
        "# test-ranked top-three rule, selected from scratch here.",
        "#",
        f"# FOLDS {list(folds)} - this study trains one g-0 on one client split.",
        "# --folds 1 2 3 4 5 restores the digit shape and multiplies every count",
        "# and every cost by five.",
        "#",
        "# THE FISHER is this study's own g-0 Fisher, written beside the model by",
        "# the g-0 training itself. Sixteen regularisation cells need it; none of",
        "# them names $G0_FOLD, because a study with one g-0 has no selection",
        "# record for the runner to read a fold from.",
        "#",
        f"# {len(tasks)} tasks.",
        "#",
    ]


#: The variables a job needs in its own environment.  ``env.sh`` refuses without
#: ``FOA_PROJECT_DIR`` and derives the rest, but a derived default is only right
#: when the submitting shell's value was the default too - so everything that is
#: set is carried explicitly rather than re-guessed on the compute node.
CARRY = (
    "FOA_PROJECT_DIR FOA_STUDY_DIR FOA_REPO FOA_RESULTS_DIR FOA_DATA_DIR "
    "FOA_NIST28_DIR FOA_SHAKESPEARE_DIR FOA_CACHE_DIR FOA_ENV FOA_MODULES "
    "FOA_HTTP_PROXY FOA_NIST_CLASSES FOA_ACCOUNT FOA_GPU_PARTITION "
    "FOA_CPU_PARTITION"
)

#: What the chain cannot run without, checked before a single job is submitted.
REQUIRED = "FOA_PROJECT_DIR FOA_STUDY_DIR FOA_ACCOUNT FOA_GPU_PARTITION"


def chain_script(folds) -> str:
    """The submission order, with the dependencies that make it one chain."""
    n = counts(folds)
    # rf: the script escapes shell dollars, which are not Python escapes.
    return rf"""#!/bin/bash
# Shakespeare_study01 stage B, submitted as one afterok chain.
#
#   bash s01b_chain.sh          # print what it would submit
#   bash s01b_chain.sh --go     # submit
#
# WHY EVERY sbatch CARRIES AN EXPLICIT --export
# ---------------------------------------------
# This chain was once submitted without one. Slurm did not propagate the
# submitting shell's environment, every element of both screens started with no
# FOA_PROJECT_DIR, env.sh refused, and 296 array elements failed at startup
# across about three attempts each - roughly ten GPU-hours spent discovering
# that a variable was missing. The refusal was right; the submission was wrong.
#
# `--export=ALL` alone is not enough on a site whose default is NONE, because
# ALL is exactly the default being overridden. So every variable that is set is
# named and passed by value.
#
# Every array width below is FIXED, and every generator is given the matching
# --expect. That pairing is the point: a generator that emitted a different
# number exits non-zero, the afterok fails, and the chain stops - rather than an
# array running elements that point at nothing.
set -euo pipefail
GO="${{1:-}}"
J="${{FOA_PROJECT_DIR:-}}/jobs/v4"
S="${{FOA_STUDY_DIR:-}}"
R="${{FOA_PROJECT_DIR:-}}/repo_foa/slurm/study_phase.sbatch"

REQUIRED="{REQUIRED}"
CARRY="{CARRY}"

# ---- preflight: refuse at submit time, not inside 888 job attempts ----------
missing=""
for v in $REQUIRED; do
    eval "value=\${{$v:-}}"
    [ -n "$value" ] || missing="$missing $v"
done
if [ -n "$missing" ]; then
    echo "REFUSED: these are not set:$missing" >&2
    echo "" >&2
    echo "The chain submits 12 jobs and 349 GPU elements. Every one of them" >&2
    echo "would start, fail in env.sh, and cost its startup time to tell you" >&2
    echo "this. Export them and re-run:" >&2
    echo "  export FOA_PROJECT_DIR=/path/to/workspace" >&2
    echo "  export FOA_STUDY_DIR=\"\$FOA_PROJECT_DIR/results_v4/studies/{CFG.name}\"" >&2
    echo "  export FOA_ACCOUNT=... FOA_GPU_PARTITION=..." >&2
    exit 78
fi
if [ ! -f "$R" ]; then
    echo "REFUSED: no runner at $R" >&2
    exit 78
fi
if [ ! -d "$S" ]; then
    echo "REFUSED: no study at $S" >&2
    exit 78
fi

# ---- the export string, built from what is actually set --------------------
EXPORTS="ALL"
for v in $CARRY; do
    eval "value=\${{$v:-}}"
    [ -n "$value" ] && EXPORTS="$EXPORTS,$v=$value"
done

mkdir -p "$S/jobs" "$S/logs"

submit() {{  # submit <name> <array> <taskfile> [dependency]
  local dep=""
  [ -n "${{4:-}}" ] && dep="--dependency=afterok:$4"
  if [ "$GO" != "--go" ]; then
    echo "would: sbatch --job-name=$1 --array=$2 $dep --export=$EXPORTS $R $3" >&2
    echo "JOBID_PLACEHOLDER"
    return
  fi
  # shellcheck disable=SC2086
  sbatch --parsable --job-name="$1" \
      --account="$FOA_ACCOUNT" --partition="$FOA_GPU_PARTITION" --gres=gpu:1 \
      --output="$S/logs/%x_%A_%a.log" --array="$2" \
      --export="$EXPORTS" $dep "$R" "$3"
}}

J1=$(submit agg_screen   1-{n['agg_screen']}   "$J/s01b_agg_screen.txt")
J2=$(submit gen_agg_full 1-1                   "$J/s01b_gen_agg_full.txt" "$J1")
J3=$(submit agg_full     1-{n['agg_full']}     "$S/jobs/s01b_agg_full.txt" "$J2")
J4=$(submit gen_agg_top3 1-1                   "$J/s01b_gen_agg_top3.txt" "$J3")

J5=$(submit reg_screen   1-{n['reg_screen']}   "$J/s01b_reg_screen.txt")
J6=$(submit gen_reg      1-2                   "$J/s01b_gen_reg.txt" "$J5")
J7=$(submit reg_hybrid   1-{n['reg_hybrid']}   "$S/jobs/s01b_reg_hybrid.txt" "$J6")
J8=$(submit reg_full     1-{n['reg_full']}     "$S/jobs/s01b_reg_full.txt" "$J7")
J9=$(submit gen_reg_top3 1-1                   "$J/s01b_gen_reg_top3.txt" "$J8")

JA=$(submit gen_combos   1-1                   "$J/s01b_gen_combos.txt" "$J9")
JB=$(submit combos       1-{n['combos']}       "$S/jobs/s01b_combos.txt" "$JA")
JC=$(submit gen_winner   1-1                   "$J/s01b_gen_winner.txt" "$JB")

echo "chain: $J1 $J2 $J3 $J4 | $J5 $J6 $J7 $J8 $J9 | $JA $JB $JC"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=None, metavar="K")
    args = parser.parse_args()

    folds = tuple(args.folds) if args.folds else SL.folds_of(CFG)
    out_dir = Path(args.jobs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static = {
        "s01b_agg_screen": (agg_screen(folds), "the aggregation screen"),
        "s01b_reg_screen": (reg_screen(folds), "the regularisation screen"),
    }
    for name, (tasks, what) in static.items():
        (out_dir / f"{name}.txt").write_text(
            "\n".join(header(name, what, tasks, folds) + tasks) + "\n"
        )
        print(f"wrote {name}.txt: {len(tasks)} tasks")

    for name, tasks in generator_lines(folds).items():
        (out_dir / f"s01b_{name}.txt").write_text(
            "\n".join(header(name, "an in-chain selection step", tasks, folds) + tasks) + "\n"
        )
        print(f"wrote s01b_{name}.txt: {len(tasks)} task(s)")

    (out_dir / "s01b_chain.sh").write_text(chain_script(folds))
    print("wrote s01b_chain.sh")

    n = counts(folds)
    print(f"\nfolds {list(folds)}   total submitted tasks: {sum(n.values())}")
    low = high = 0.0
    print(f"{'stage':12s} {'tasks':>6s} {'rounds':>7s} {'GPU-h low':>10s} {'high':>8s}")
    for name, tasks, rounds, lo, hi in cost(folds):
        low += lo
        high += hi
        print(f"{name:12s} {tasks:6d} {rounds:7d} {lo:10.1f} {hi:8.1f}")
    print(f"{'TOTAL':12s} {sum(n.values()):6d} {'':7s} {low:10.1f} {high:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
