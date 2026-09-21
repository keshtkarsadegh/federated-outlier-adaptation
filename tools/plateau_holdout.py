#!/usr/bin/env python3
"""
Does the plateau rule's +1.19p survive choosing `k` where it is not measured?

`plateau_rule.py` fixes ``k = 20``, ``eps = 0`` and reports that it buys +1.19p
over the fixed horizon on the 83 full-horizon arms. That cell is the best of the
sixteen ON THOSE SAME 83 ARMS, and the honesty check the tool already carries -
`plateau_grid.csv`'s `nonextreme_mean_score` column - only holds the two extreme
arrangements out. The harder version of the objection is the one a reviewer
actually asks: choose ``(k, eps)`` on a designated subset and evaluate it,
unchanged, on a disjoint one. This tool runs that split in three directions - by
FOLD, by STAGE and by random halves of the ARMS - and reports what the held-out
gain is each time.

    python tools/plateau_holdout.py --root "$FOA_STUDY_DIR"
    python tools/plateau_holdout.py --root "$FOA_STUDY_DIR" \\
        --out "$FOA_STUDY_DIR/tables/stopping"

THE RULE IS NOT REIMPLEMENTED, and neither is the table it is priced against.
`fire_round`, `kept_round`, `outcomes` and the sixteen-cell grid come from
`plateau_rule.py`; the arm reader, the fold mean, the score, the baselines and
the oracle round come from `stopping_table.py` through it. So a cell priced here
is priced exactly as the published table prices it, and the `in-sample` row of
`plateau_holdout_protocols.csv` reproduces `plateau_stages.csv`'s `all` row by
construction rather than by agreement.

THE ONE THING THIS FILE ADDS IS A FOLD-AWARE READ, and it is added because the
published rule runs on the FOLD MEAN of an arm's five folds. `read_arms` drops
which fold a trace came from - the rule never needs to know - so a fold split
cannot be made by cutting a trace in half: there is only one trace, and it
already has all five folds in it. It has to be made by re-averaging the arm over
the selected folds ALONE, which is the trace a study run on those folds alone
would have had, and that needs the fold label back. It is in the run folder's
own name, which is what `read_folds` keeps.

THE ARM POPULATION IS HELD FIXED ACROSS EVERY SPLIT. The five-fold filter is
applied to the full five folds and never to the subset, so a fold protocol
changes the traces and not which arms are in the table. A protocol that silently
dropped arms would be comparing two different studies rather than one study read
two ways.

THE SELECTION OBJECTIVE is `plateau_rule.best_cell`'s, restated on an arbitrary
set of arms rather than on the grid view: over a set of arms, the mean over arms
of the score at the round the rule keeps, maximised over the sixteen cells at the
`checkpoint_best` variant, ties to the smaller patience and then the smaller
margin. A0 and P0 shift every arm's score by the same constant, so on a fixed arm
set ranking by mean kept score and ranking by mean gain over the horizon are the
same ranking - which is why the selection column is a gain and the objective is a
score without the two ever disagreeing.

THE RANDOM HALVES ARE SEEDED AND THE SEEDS ARE FIXED HERE. Each half is cut by
`random.Random(seed).shuffle` over the arms in their sorted order, for
``seed = 0 .. 9``, so a half is a function of the seed alone - not of the
interpreter's global random state, not of the order the run folders were read in,
and not of which protocols ran before it. The views regenerate byte for byte.

WHAT IT FINDS, over the 28 protocols:

    k = 20 is chosen by every one of them, at every margin and on every split
    leave-one-fold-out   held-out gain  1.83p  +/- 0.49p over the five folds
    random halves        held-out gain  1.33p  +/- 0.23p over the ten seeds
    the two extremes     round 8 and round 36 kept under every protocol

so the setting is not an artefact of the arms it was measured on. The extremes
are read off the FULL five-fold traces whatever the setting was chosen on: the
question is whether a `k` picked elsewhere still keeps the rounds the manuscript
quotes, and those rounds are quoted off the published traces.

WHAT IT WRITES, into the stopping bundle beside the four `plateau_*.csv`:

    plateau_holdout_protocols.csv  one line per protocol: what was chosen
                                   where, and what it delivered held out
    plateau_holdout_extremes.csv   the two extreme arrangements under every
                                   protocol's chosen cell
    plateau_holdout_grids.csv      every fold set x every cell, for the record
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from stopping_table import (  # noqa: E402
    SERIES,
    SIGNAL_KEYS,
    STAGES,
    stage_baselines,
    Verdict,
    arm_of,
    baselines,
    cell_and_family,
    in_study,
    judge,
)

from plateau_rule import (  # noqa: E402
    CHECKPOINT_BEST,
    MARGINS,
    PATIENCES,
    PRIMARY_MARGIN,
    PRIMARY_PATIENCE,
    outcomes,
)

#: The five folds every full-horizon arm was run over, in the order the fold
#: labels are printed in.
FOLDS = (1, 2, 3, 4, 5)

#: The two halves the fold protocol cuts, and the stage the stage protocol cuts
#: on: the largest stage of the study, so the split is the one with the most to
#: say rather than the one that happens to be first.
FOLD_HALVES = ((1, 2, 3), (4, 5))
SPLIT_STAGE = "regfull"

#: The seeds the random halves are cut with. Ten is enough to put a standard
#: deviation on the held-out gain and few enough that the ten rows can be read.
RANDOM_SEEDS = tuple(range(10))

#: The stage the extreme table is cut from - `plateau_rule.EXTREME_STAGE`, named
#: here as well because this file cuts it out of a keyed grid rather than a row
#: list.
EXTREME_STAGE = "extreme"

#: No signal rule is priced here - the plateau rule reads the cohort trace only -
#: so the drift grid `judge` would sweep is empty, which also makes it cheap.
NO_DELTAS: Tuple[float, ...] = ()

#: What `plateau_extremes.csv` reports at k = 20, eps = 0 on all five folds: the
#: round kept and the score it was kept at, which are the two numbers section 5
#: of `docs/STOPPING.md` prints. A protocol matches when it keeps the same round
#: at the same score.
#: Re-pinned when each setting started being scored against its own cohort's
#: shipped-model accuracy: the extreme pair is two writers, not the ten-writer
#: cohort, so its A0 is 0.7590 and not 0.8225. The ROUNDS did not move - a stop
#: is an argmax along one trace and a constant added to every round of it
#: changes nothing - and that they did not is the check this table performs.
PUBLISHED_EXTREMES = {"dual": (8, 0.22948574038421832),
                      "double": (36, 0.22032505900628463)}


# ------------------------------------------------------- the fold-aware read
def read_folds(root: Path, prefix: str) -> Dict[Tuple[str, str], Dict[int, Dict[str, List[float]]]]:
    """
    ``{(cell, family): {fold: {series: trace}}}`` - `read_arms` keeping the fold.

    `stopping_table.read_arms` drops which fold a trace came from, because the
    published rule only ever sees the mean over all of them. A fold split needs
    the label back, and it is in the run folder's own name. Everything else -
    which paths are read, which series are kept, which runs `in_study` admits -
    is that function's, line for line, so the two readers cannot come apart on
    the population they see.
    """
    found: Dict[Tuple[str, str], Dict[int, Dict[str, List[float]]]] = defaultdict(dict)
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        cell, family = cell_and_family(Path(path), prefix)
        if cell is None or not in_study(str(Path(path))):
            continue
        folder = next(part for part in Path(path).parts if part.startswith(prefix))
        fold = int(folder.split("_fold")[1].split("_")[0])
        body = next(iter(json.loads(Path(path).read_text()).values()))
        traces = {}
        for name in tuple(SERIES) + tuple(SIGNAL_KEYS):
            values = [v for v in (body.get(name) or []) if v is not None]
            if values:
                traces[name] = values
        found[(cell, family)][fold] = traces
    return dict(found)


def stored_for(per_fold: Dict[int, Dict[str, List[float]]],
               folds: Sequence[int]) -> Dict[str, List[List[float]]]:
    """The `read_arms` shape again, restricted to the folds asked for."""
    stored: Dict[str, List[List[float]]] = defaultdict(list)
    for fold in sorted(folds):
        for name, values in per_fold.get(fold, {}).items():
            stored[name].append(values)
    return dict(stored)


def library(root: Path, tag: str) -> Dict[Tuple[str, str, str], Dict[int, Dict[str, List[float]]]]:
    """Every arm of every hundred-round stage, keyed by its stage as well."""
    found: Dict[Tuple[str, str, str], Dict[int, Dict[str, List[float]]]] = {}
    for stem, _title in STAGES:
        stage = stem.strip("_")
        for (cell, family), per_fold in sorted(read_folds(root, f"{tag}_{stem}").items()):
            found[(stage, cell, family)] = per_fold
    return found


def staged_for(lib, refs: Dict[str, Tuple[float, float]], folds: Sequence[int],
               min_folds: int) -> List[Tuple[str, Verdict]]:
    """
    ``(stage, verdict)`` for every arm, re-averaged over ``folds`` alone.

    ``refs`` is the stage-to-``(A0, P0)`` table: an arm is scored against the
    cohort it actually federated, not against the ten-writer one every stage
    used to share.

    The fold-count filter is applied to the FULL five folds rather than to the
    subset, so the arm population is the published one in every split: a fold
    protocol has to change the traces and not which arms are in the table.
    """
    found: List[Tuple[str, Verdict]] = []
    for (stage, cell, family), per_fold in lib.items():
        if len(per_fold) < min_folds:
            continue
        arm = arm_of(cell, family, stored_for(per_fold, folds))
        if arm is None or arm.rounds < 2:
            continue
        a0, p0 = refs[stage]
        found.append((stage, judge(arm, a0, p0, NO_DELTAS)))
    return found


# ------------------------------------------------------- selection and score
def keyed(staged: Sequence[Tuple[str, Verdict]], patience: int,
          margin: float) -> Dict[Tuple[str, str, str], dict]:
    """Every arm's outcome at one setting, keyed so a subset can be cut from it."""
    rows = outcomes(staged, patience, margin, CHECKPOINT_BEST)
    return {(row["stage"], row["cell"], row["family"]): row for row in rows}


def priced(staged: Sequence[Tuple[str, Verdict]]) -> Dict[Tuple[int, float], Dict]:
    """The whole grid, once, so a split only has to pick rows out of it."""
    return {(patience, margin): keyed(staged, patience, margin)
            for patience in PATIENCES for margin in MARGINS}


def summarise(grid: Dict[Tuple[int, float], Dict], arms: Sequence[tuple],
              cell: Tuple[int, float]) -> dict:
    """Mean fixed, rule, gain and oracle over a set of arms at one setting."""
    rows = [grid[cell][arm] for arm in arms]
    return {
        "arms": len(rows),
        "fixed": st.mean(row["final_score"] for row in rows),
        "rule": st.mean(row["kept_score"] for row in rows),
        "gain": st.mean(row["gain"] for row in rows),
        "oracle": st.mean(row["oracle_score"] for row in rows),
        "fires": sum(row["fired"] for row in rows),
    }


def choose(grid: Dict[Tuple[int, float], Dict],
           arms: Sequence[tuple]) -> Tuple[int, float]:
    """
    `plateau_rule.best_cell`'s rule, on an arbitrary set of arms.

    Maximise the mean kept score; ties to the smaller patience, then the smaller
    margin - that tool's own ``(value, -patience, -margin)`` key, restated on a
    set of arms because the grid view it reads is already a mean over all of
    them.
    """
    return max(grid, key=lambda cell: (summarise(grid, arms, cell)["rule"],
                                       -cell[0], -cell[1]))


def label(folds: Sequence[int]) -> str:
    """The fold set as the views print it: ``45`` is folds four and five."""
    return "".join(str(fold) for fold in sorted(folds))


# -------------------------------------------------------------- the protocols
PROTOCOL_COLUMNS = ("protocol", "selection_set", "sel_arms", "sel_folds", "k", "eps",
                    "sel_gain", "heldout_set", "held_arms", "held_folds",
                    "held_fixed", "held_rule", "held_gain", "held_oracle",
                    "held_fires", "held_gain_at_primary", "primary_is_chosen")


def protocol_row(protocol: str, sel_label: str, sel_grid, sel_arms, sel_folds: str,
                 held_label: str, held_grid, held_arms, held_folds: str) -> dict:
    """
    One selection set priced against one evaluation set.

    `held_gain_at_primary` is the published setting's gain on the same held-out
    set, so a row says both what the protocol chose and what it would have cost
    to have chosen k = 20 regardless - and when the two columns agree, which
    they do on every row here, the protocol has nothing to object to.
    """
    cell = choose(sel_grid, sel_arms)
    sel = summarise(sel_grid, sel_arms, cell)
    held = summarise(held_grid, held_arms, cell)
    primary = summarise(held_grid, held_arms, (PRIMARY_PATIENCE, PRIMARY_MARGIN))
    return {
        "protocol": protocol,
        "selection_set": sel_label,
        "sel_arms": len(sel_arms),
        "sel_folds": sel_folds,
        "k": cell[0],
        "eps": cell[1],
        "sel_gain": sel["gain"],
        "heldout_set": held_label,
        "held_arms": held["arms"],
        "held_folds": held_folds,
        "held_fixed": held["fixed"],
        "held_rule": held["rule"],
        "held_gain": held["gain"],
        "held_oracle": held["oracle"],
        "held_fires": held["fires"],
        "held_gain_at_primary": primary["gain"],
        "primary_is_chosen": int(cell == (PRIMARY_PATIENCE, PRIMARY_MARGIN)),
    }


# --------------------------------------------------------------- the extremes
EXTREME_COLUMNS = ("protocol", "selection_set", "k", "eps", "cell", "family",
                   "fire_round", "kept_round", "kept_score", "final_score", "gain",
                   "matches_published")


def extreme_rows(protocol: str, sel_label: str, cell: Tuple[int, float],
                 full_grid) -> List[dict]:
    """
    What a chosen setting keeps on the two extreme arrangements.

    Always read off the FULL five-fold traces, whatever the setting was chosen
    on: the question is whether a `k` picked elsewhere still keeps the rounds the
    manuscript quotes, and those rounds are quoted off the published traces.
    """
    found: List[dict] = []
    for key, row in sorted(full_grid[cell].items()):
        if key[0] != EXTREME_STAGE:
            continue
        want = PUBLISHED_EXTREMES.get(row["cell"])
        found.append({
            "protocol": protocol,
            "selection_set": sel_label,
            "k": cell[0],
            "eps": cell[1],
            "cell": row["cell"],
            "family": row["family"],
            "fire_round": row["fire_round"],
            "kept_round": row["kept_round"],
            "kept_score": row["kept_score"],
            "final_score": row["final_score"],
            "gain": row["gain"],
            "matches_published": int(
                want is not None and row["kept_round"] == want[0]
                and abs(row["kept_score"] - want[1]) < 1e-12),
        })
    return found


# ------------------------------------------------------------- the fold grids
#: Every fold set the protocols priced, against every cell, cut three ways: all
#: the arms, the split stage, and everything that is not the split stage.
GRID_PARTS = ("all", SPLIT_STAGE, f"non{SPLIT_STAGE}")
GRID_KEYS = ("arms", "fixed", "rule", "gain", "oracle", "fires")
GRID_COLUMNS = ("folds", "k", "eps") + tuple(
    f"{part}_{key}" for part in GRID_PARTS for key in GRID_KEYS)


def grid_rows(grids: Dict[str, Dict], parts: Sequence[Sequence[tuple]]) -> List[dict]:
    """Every fold set against every cell, in fold-label then grid order."""
    return [{"folds": folds, "k": patience, "eps": margin,
             **{f"{name}_{key}": value
                for name, arms in zip(GRID_PARTS, parts)
                for key, value in summarise(grid, arms, (patience, margin)).items()}}
            for folds, grid in sorted(grids.items())
            for patience in PATIENCES
            for margin in MARGINS]


def write_csv(rows: Sequence[dict], columns: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    print(f"  -> {path}  ({len(rows)} rows)")


# ------------------------------------------------------------------ printing
def _p(value: Optional[float]) -> str:
    """A score in points, or a dash where there is none."""
    return "-" if value is None else f"{value * 100:.2f}p"


def show_protocols(rows: Sequence[dict]) -> None:
    """Every protocol: what it chose, where, and what that delivered held out."""
    print("\nWHAT EACH PROTOCOL CHOSE, AND WHAT IT DELIVERED ON THE SET IT DID NOT SEE")
    print(f"  {'protocol':<12}{'selection set':<26}{'k':>4}{'eps':>8}{'sel gain':>10}"
          f"{'held':>6}{'held fixed':>12}{'held gain':>11}{'held oracle':>13}")
    for row in rows:
        print(f"  {row['protocol']:<12}{row['selection_set']:<26}{row['k']:>4}"
              f"{row['eps']:>8g}{_p(row['sel_gain']):>10}{row['held_arms']:>6}"
              f"{_p(row['held_fixed']):>12}{_p(row['held_gain']):>11}"
              f"{_p(row['held_oracle']):>13}")


def show_spread(name: str, rows: Sequence[dict]) -> None:
    """A family of protocols as one line: which k, and the gain's spread."""
    gains = [row["held_gain"] for row in rows]
    chosen = [row["k"] for row in rows]
    counts = ", ".join(f"{k}: {chosen.count(k)}" for k in sorted(set(chosen)))
    print(f"\n  {name}: k chosen {{{counts}}}, held-out gain mean "
          f"{_p(st.mean(gains))} sd {_p(st.stdev(gains))} "
          f"min {_p(min(gains))} max {_p(max(gains))}")


def show_extremes(rows: Sequence[dict]) -> None:
    """The pair the horizon costs most, under every protocol's chosen cell."""
    bad = [row for row in rows if not row["matches_published"]]
    print(f"\nTHE EXTREME ARRANGEMENTS: {len(rows)} checks, "
          f"{len(rows) - len(bad)} keep the published round at the published score")
    for row in bad:
        print(f"  moved: {row['protocol']} / {row['selection_set']} / "
              f"{row['cell']} keeps round {row['kept_round']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory the three CSV views are written into. "
                         "Without it nothing is written and the tables are "
                         "only printed.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--min-folds", type=int, default=5,
                    help="Arms with fewer folds than this are not reported.")
    args = ap.parse_args()

    refs = stage_baselines(args.root)
    lib = library(args.root, args.study_tag)
    if not lib:
        print("no hundred-round arm on disk under this root.")
        return 1

    full = staged_for(lib, refs, FOLDS, args.min_folds)
    grids = {label(FOLDS): priced(full)}
    full_grid = grids[label(FOLDS)]
    arms_all = sorted(full_grid[(PRIMARY_PATIENCE, PRIMARY_MARGIN)])
    by_stage: Dict[str, List[tuple]] = defaultdict(list)
    for key in arms_all:
        by_stage[key[0]].append(key)
    split = by_stage[SPLIT_STAGE]
    unsplit = [arm for arm in arms_all if arm[0] != SPLIT_STAGE]

    rows: List[dict] = []
    extremes: List[dict] = []

    def run(protocol, sel_label, sel_folds, sel_arms, held_label, held_folds, held_arms):
        """Price one protocol, keep its row, and read the extremes under it."""
        for folds in (sel_folds, held_folds):
            if label(folds) not in grids:
                grids[label(folds)] = priced(
                    staged_for(lib, refs, folds, args.min_folds))
        row = protocol_row(protocol, sel_label, grids[label(sel_folds)], sel_arms,
                           label(sel_folds), held_label, grids[label(held_folds)],
                           held_arms, label(held_folds))
        rows.append(row)
        extremes.extend(extreme_rows(protocol, sel_label, (row["k"], row["eps"]), full_grid))
        return row

    # 1. in sample: the published setting, priced where it was chosen ------
    run("in-sample", "all arms, all folds", FOLDS, arms_all,
        "same (in sample)", FOLDS, arms_all)

    # 2. held out by fold --------------------------------------------------
    first, second = FOLD_HALVES
    run("fold", f"all arms, folds {label(first)}", first, arms_all,
        f"all arms, folds {label(second)}", second, arms_all)
    run("fold", f"all arms, folds {label(second)}", second, arms_all,
        f"all arms, folds {label(first)}", first, arms_all)
    lofo = []
    for fold in FOLDS:
        rest = tuple(other for other in FOLDS if other != fold)
        lofo.append(run("fold-loo", f"all arms, folds {label(rest)}", rest, arms_all,
                        f"all arms, fold {fold}", (fold,), arms_all))

    # 3. held out by stage -------------------------------------------------
    run("stage", f"{SPLIT_STAGE} arms", FOLDS, split,
        f"all non-{SPLIT_STAGE} arms", FOLDS, unsplit)
    run("stage", f"all non-{SPLIT_STAGE} arms", FOLDS, unsplit,
        f"{SPLIT_STAGE} arms", FOLDS, split)
    for stem, _title in STAGES:
        stage = stem.strip("_")
        held = by_stage.get(stage) or []
        if not held:
            continue
        run("stage-loo", f"all but {stage}", FOLDS,
            [arm for arm in arms_all if arm[0] != stage], f"{stage} arms", FOLDS, held)

    # 4. held out by a seeded random half of the arms ----------------------
    halves = []
    for seed in RANDOM_SEEDS:
        shuffled = list(arms_all)
        random.Random(seed).shuffle(shuffled)
        cut = len(shuffled) // 2
        halves.append(run("random-half", f"seed {seed} half A", FOLDS, shuffled[:cut],
                          f"seed {seed} half B", FOLDS, shuffled[cut:]))

    show_protocols(rows)
    show_spread("leave-one-fold-out", lofo)
    show_spread("random halves", halves)
    show_extremes(extremes)

    if args.out:
        print()
        write_csv(rows, PROTOCOL_COLUMNS, args.out / "plateau_holdout_protocols.csv")
        write_csv(extremes, EXTREME_COLUMNS, args.out / "plateau_holdout_extremes.csv")
        write_csv(grid_rows(grids, (arms_all, split, unsplit)), GRID_COLUMNS,
                  args.out / "plateau_holdout_grids.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
