#!/usr/bin/env python3
"""
Paired comparisons between arms, folded by fold.

``report_tables.py`` reports what each arm scored.  This reports what the
DIFFERENCE between two arms is worth, which is a different question and has a
different failure mode.

WHY THIS EXISTS.  The combination stage crosses the three best server rules with
the three best client penalties, and the obvious way to read it is to compare
each pair's mean against its two halves' means.  Done that way on this study,
eleven of eighteen combinations "beat both halves" and the stage looks like a
clean positive result.  Pairing the same numbers by fold shows the gains are
+0.04 to +0.73 against a fold standard deviation of 0.22 to 1.52, and that only
one of the eighteen is positive on every fold.  The eleven were an artefact of
reading means without their spread, and nothing in a table of means says so.

So this tool never prints a mean difference without the per-fold differences
next to it, and it labels each row by whether the sign is consistent.  A mean
smaller than the spread it came from is not a finding, and the output should
make that impossible to miss.

WHY PAIRED.  The five folds are the same five partitions for every arm, so a
fold that is harder for one arm is harder for all of them.  Differencing within
a fold removes that shared difficulty; comparing means does not.  Pairing is
what lets a 0.4-point effect be discussed at all - and, here, what shows it
cannot be.

    python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what combos
    python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what composition
    python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what all --csv out/

``combos``       each combination minus its better half.
``composition``  each client penalty minus each server rule - the comparison
                 that asks which half of the update the preservation comes from.

WHERE THE HALVES ARE READ FROM.  A standalone penalty is looked for under the
schedule it was selected on and, failing that, under the bare prefix the blends
were written with; see :func:`penalties`.  A pairing that cannot find one half
prints NO PAIRED RESULT, which is a statement about this tool's search as much
as about the study, and the blends spent a while proving it.

THE SCORE is the study's one selection rule, the same one ``study_emit`` selects
by and ``report_tables`` orders by:

    score = (adaptation - A0) - w * (P0 - preservation)        w = 1

with A0 and P0 the shipped model's own two accuracies, read from the selection
stage's evaluations rather than written down here.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: Weight on a point of forgetting.  One, matching both grids' selections.
FORGETTING_WEIGHT = 1.0

#: Schedules, each selected and reported separately.
FAMILIES = ("concurrent", "sequential")


def baselines(root: Path) -> Tuple[float, float]:
    """
    ``(A0, P0)`` - the shipped model's own accuracies, measured not assumed.

    A0  g-0 on the cohort's rows: what it already gets right on the new clients.
    P0  g-0 on the source population: what it knows before anything touches it.
    """
    def mean_accuracy(name: str) -> Optional[float]:
        path = root / name
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else list(payload.values())
        values = [r["accuracy"] for r in records
                  if isinstance(r, dict) and isinstance(r.get("accuracy"), (int, float))]
        return sum(values) / len(values) if values else None

    a0 = mean_accuracy("g0_perfold_evaluations.json")
    p0 = mean_accuracy("g0_evaluations.json")
    if a0 is None or p0 is None:
        raise SystemExit(
            "FATAL: the shipped model's own accuracies are missing "
            f"({root}/g0_perfold_evaluations.json, g0_evaluations.json). "
            "Every difference here is measured against them and cannot be guessed."
        )
    return a0, p0


def score(adaptation: float, preservation: float, a0: float, p0: float) -> float:
    """The selection rule, in points."""
    return ((adaptation - a0) - FORGETTING_WEIGHT * (p0 - preservation)) * 100.0


def _last(series) -> Optional[float]:
    for value in reversed(list(series or [])):
        if value is not None:
            return float(value)
    return None


def per_fold(root: Path, prefix: str, family: str, a0: float, p0: float) -> Dict[str, Dict[int, float]]:
    """
    ``{cell id: {fold: score}}`` for every run under ``prefix`` in ``family``.

    A payload is claimed by the family it says it is (``scenario``), not by
    where it sits: the screens run both schedules in one task and write both
    into one folder.
    """
    found: Dict[str, Dict[int, float]] = {}
    for run_dir in sorted(root.glob(prefix + "*")):
        stem = run_dir.name[len(prefix):]
        if "_fold" not in stem:
            continue
        cell, tail = stem.rsplit("_fold", 1)
        try:
            fold = int(tail.split("_")[0])
        except ValueError:
            continue
        for payload_path in run_dir.rglob("accuracies_*.json"):
            try:
                data = json.loads(payload_path.read_text())
            except (OSError, ValueError):
                continue
            if data.get("scenario") != family:
                continue
            # REPORT ON TEST: this tool states differences for the paper, so
            # it reads the test halves. The selectors read validation - a
            # difference quoted on the half a winner was chosen on would be
            # reporting the selection back to itself.
            final = data.get("final_evaluation") or {}
            adaptation = (final.get("clients") or {}).get("accuracy")
            preservation = (final.get("old") or {}).get("mean")
            if adaptation is None:
                adaptation = _last(data.get("pool_val_accuracies"))
            if adaptation is None or preservation is None:
                continue
            found.setdefault(cell, {})[fold] = score(adaptation, preservation, a0, p0)
    return found


#: Where a standalone penalty's finals were written, in the order they are
#: tried.  ``{family}`` is filled in per schedule; the bare prefix is the
#: fallback and matches everything the first one does, which is why it only
#: ever fills gaps.
REG_PREFIXES = ("d01_regfull_{family}_", "d01_regfull_")


def penalties(root: Path, family: str, a0: float, p0: float) -> Dict[str, Dict[int, float]]:
    """
    Every standalone penalty of ``family``, whichever way its folder was named.

    Most penalty cells are stored under the schedule they were selected on -
    ``d01_regfull_<family>_<id>_fold*`` - because that run produced one
    schedule's numbers.  The blends are not: one folder per (cell, fold) holds
    BOTH schedules' payloads and is named bare, ``d01_regfull_<id>_fold*``, and
    the two are told apart by each payload's own ``scenario`` field, which
    :func:`per_fold` already filters on.

    Reading only the family-prefixed name is what left every blend row saying
    NO PAIRED RESULT while its runs sat on disk: the finals existed, under a
    name this tool never looked for.  So both names are read and the bare one
    fills only what the family-prefixed one did not resolve - the family
    prefix stays authoritative, and any cell stored bare in future is picked up
    without naming it here.  Ids the bare read invents out of the family-
    prefixed folders (``concurrent_kd_T0p25_a0p9`` and the like) are never
    asked for, because the shortlists name cells and not folders.
    """
    found: Dict[str, Dict[int, float]] = {}
    for prefix in REG_PREFIXES:
        for cell, folds in per_fold(root, prefix.format(family=family),
                                    family, a0, p0).items():
            found.setdefault(cell, folds)
    return found


def paired(left: Dict[int, float], right: Dict[int, float]) -> Optional[dict]:
    """
    ``left - right`` on the folds they share, with the spread that qualifies it.

    ``consistent`` is the honest headline: a mean difference whose sign is not
    the same on every fold has not been shown to be a difference at all.
    """
    folds = sorted(set(left) & set(right))
    if len(folds) < 2:
        return None
    diffs = [left[f] - right[f] for f in folds]
    return {
        "folds": folds,
        "diffs": diffs,
        "mean": st.mean(diffs),
        "sd": st.stdev(diffs),
        "consistent": all(d > 0 for d in diffs) or all(d < 0 for d in diffs),
        "all_positive": all(d > 0 for d in diffs),
    }


def _top_lists(root: Path) -> Tuple[dict, dict]:
    """The two shortlists the combination stage was emitted from."""
    def load(name: str) -> dict:
        path = root / "tables" / name
        if not path.is_file():
            raise SystemExit(
                f"FATAL: {path} is missing. The cross is defined by the "
                "shortlists it was emitted from; guessing them here would "
                "compare arms the stage never ran."
            )
        payload = json.loads(path.read_text())
        return payload.get("top") or payload

    return load("p12_agg_top3.json"), load("p13_reg_top3.json")


def combos(root: Path, a0: float, p0: float) -> List[dict]:
    """Each combination minus whichever of its two halves scored higher."""
    agg_top, reg_top = _top_lists(root)
    rows: List[dict] = []
    for family in FAMILIES:
        combo = per_fold(root, "d01_combo_", family, a0, p0)
        agg_alone = per_fold(root, "d01_aggfull_", family, a0, p0)
        reg_alone = penalties(root, family, a0, p0)
        for agg_id in agg_top.get(family, []):
            for reg_id in reg_top.get(family, []):
                key = f"{agg_id}_{reg_id}"
                if key not in combo:
                    rows.append({"family": family, "left": key, "right": "-",
                                 "missing": True})
                    continue
                shared = sorted(set(agg_alone.get(agg_id, {})) & set(reg_alone.get(reg_id, {})))
                if not shared:
                    rows.append({"family": family, "left": key, "right": "-",
                                 "missing": True})
                    continue
                # the better half is chosen once, on the mean, then differenced
                # fold by fold - choosing it per fold would pick the winner
                # after seeing the answer and bias every difference downward.
                mean_agg = st.mean(agg_alone[agg_id][f] for f in shared)
                mean_reg = st.mean(reg_alone[reg_id][f] for f in shared)
                better_id, better = ((agg_id, agg_alone[agg_id]) if mean_agg >= mean_reg
                                     else (reg_id, reg_alone[reg_id]))
                result = paired(combo[key], better)
                if result is None:
                    rows.append({"family": family, "left": key, "right": better_id,
                                 "missing": True})
                    continue
                result.update({"family": family, "left": key, "right": better_id,
                               "left_mean": st.mean(combo[key].values()),
                               "missing": False})
                rows.append(result)
    return rows


def composition(root: Path, a0: float, p0: float) -> List[dict]:
    """Each client penalty minus each server rule, both run alone."""
    agg_top, reg_top = _top_lists(root)
    rows: List[dict] = []
    for family in FAMILIES:
        agg_alone = per_fold(root, "d01_aggfull_", family, a0, p0)
        reg_alone = penalties(root, family, a0, p0)
        for reg_id in reg_top.get(family, []):
            for agg_id in agg_top.get(family, []):
                result = paired(reg_alone.get(reg_id, {}), agg_alone.get(agg_id, {}))
                if result is None:
                    rows.append({"family": family, "left": reg_id, "right": agg_id,
                                 "missing": True})
                    continue
                result.update({"family": family, "left": reg_id, "right": agg_id,
                               "missing": False})
                rows.append(result)
    return rows


def show(rows: List[dict], title: str) -> None:
    """One table per family, ordered by the mean difference."""
    for family in FAMILIES:
        here = [r for r in rows if r["family"] == family]
        if not here:
            continue
        print(f"\n{'=' * 96}")
        print(f"{title} - {family.upper()}")
        print(f"{'=' * 96}")
        print(f"{'arm':<40}{'minus':<22}{'mean':>7}{'sd':>7}   per-fold")
        print("-" * 96)
        scored = [r for r in here if not r.get("missing")]
        for row in sorted(scored, key=lambda r: r["mean"], reverse=True):
            marks = " ".join(f"{d:+5.1f}" for d in row["diffs"])
            tag = ("  CONSISTENT" if row["all_positive"]
                   else "  all-negative" if row["consistent"] else "")
            print(f"{row['left']:<40}{row['right']:<22}"
                  f"{row['mean']:>+7.2f}{row['sd']:>7.2f}   {marks}{tag}")
        for row in [r for r in here if r.get("missing")]:
            print(f"{row['left']:<40}{row['right']:<22}   NO PAIRED RESULT")
        if scored:
            consistent = sum(1 for r in scored if r["all_positive"])
            print("-" * 96)
            print(f"positive on EVERY fold: {consistent} of {len(scored)}")
            biggest_sd = max(r["sd"] for r in scored)
            biggest_mean = max(abs(r["mean"]) for r in scored)
            if biggest_sd > biggest_mean:
                print("NOTE: the largest fold spread exceeds the largest mean "
                      "difference - no row here is separated from fold noise.")


def write_csv(rows: List[dict], path: Path) -> None:
    """One row per comparison, folds expanded, for the manuscript's appendix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family", "arm", "minus", "mean", "sd",
                         "all_positive", "folds", "diffs"])
        for row in rows:
            if row.get("missing"):
                writer.writerow([row["family"], row["left"], row["right"],
                                 "", "", "", "", ""])
                continue
            writer.writerow([
                row["family"], row["left"], row["right"],
                f"{row['mean']:.4f}", f"{row['sd']:.4f}",
                int(row["all_positive"]),
                " ".join(str(f) for f in row["folds"]),
                " ".join(f"{d:.4f}" for d in row["diffs"]),
            ])


WHAT = {"combos": (combos, "COMBINATION minus its better half"),
        "composition": (composition, "PENALTY minus SERVER RULE, each alone")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path,
                        help="The study root ($FOA_STUDY_DIR).")
    parser.add_argument("--what", default="all",
                        choices=("all",) + tuple(sorted(WHAT)))
    parser.add_argument("--csv", type=Path, default=None,
                        help="Directory to write <what>.csv into.")
    args = parser.parse_args()

    a0, p0 = baselines(args.root)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}   "
          f"w={FORGETTING_WEIGHT:g}")
    print("basis: TEST (final_evaluation). Selection ran on validation.")

    wanted = sorted(WHAT) if args.what == "all" else [args.what]
    for name in wanted:
        builder, title = WHAT[name]
        rows = builder(args.root, a0, p0)
        show(rows, title)
        if args.csv:
            write_csv(rows, args.csv / f"{name}.csv")
            print(f"\nwrote {args.csv / f'{name}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
