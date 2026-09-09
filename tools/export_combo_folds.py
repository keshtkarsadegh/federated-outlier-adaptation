#!/usr/bin/env python3
"""
Each combination against BOTH of its halves, fold by fold.

`compare_arms.py --what combos` differences a combination against whichever of
its two halves scored higher, because a combination is offered as an improvement
on what you already had and what you already had is the better half. The
manuscript makes the stronger claim as well - that a server rule and a client
penalty run together are worth more than either alone - and that claim is about
both halves at once, so it needs both differences side by side. This view is
that table: one row per combination, its per-fold differences against the server
rule alone and against the penalty alone, and the one column that says whether
it cleared both on every fold.

    python tools/export_combo_folds.py --root "$FOA_STUDY_DIR" --out paper/data

WHY BOTH COLUMNS AND NOT THE BETTER ONE. Read on the means, seven of the
eighteen combinations beat both halves and seven rows look like a clean result.
Paired by fold, the gains are smaller than the fold spread they came from and
`beats_both_folds` is 1 on two rows of eighteen. A table that reported only the
better half would have kept that finding out of the manuscript, which is the
whole reason `compare_arms.py` exists; this file makes the harder version of the
same statement checkable.

NOTHING HERE IS DEFINED TWICE. The score, the fold pairing, the per-fold loaders
and the two shortlists the cross was emitted from all come from
`compare_arms.py`. The only thing this file adds is that the difference is taken
against each half rather than against the better one, and the shape it is
written in.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_arms import (  # noqa: E402
    FAMILIES,
    _top_lists,
    baselines,
    paired,
    penalties,
    per_fold,
)

COLUMNS = ("family", "arm", "half_agg", "half_reg", "folds",
           "diffs_vs_agg", "diffs_vs_reg",
           "mean_vs_agg", "sd_vs_agg", "mean_vs_reg", "sd_vs_reg",
           "all_positive_vs_agg", "all_positive_vs_reg", "beats_both_folds")


def rows_of(root: Path, a0: float, p0: float) -> List[dict]:
    """One row per pair of the cross, in the order the cross was emitted."""
    agg_top, reg_top = _top_lists(root)
    rows: List[dict] = []
    for family in FAMILIES:
        combo = per_fold(root, "d01_combo_", family, a0, p0)
        agg_alone = per_fold(root, "d01_aggfull_", family, a0, p0)
        reg_alone = penalties(root, family, a0, p0)
        for agg_id in agg_top.get(family, []):
            for reg_id in reg_top.get(family, []):
                arm = f"{agg_id}_{reg_id}"
                row = {"family": family, "arm": arm,
                       "half_agg": agg_id, "half_reg": reg_id}
                # THE FOLDS ARE THE ONES ALL THREE RAN. Differencing a
                # combination against a half on a fold the half never reached
                # would price it against nothing, and pricing the two halves on
                # different fold sets would make the two columns of one row
                # incomparable.
                shared = sorted(set(combo.get(arm, {}))
                                & set(agg_alone.get(agg_id, {}))
                                & set(reg_alone.get(reg_id, {})))
                only = {fold: combo[arm][fold] for fold in shared} if shared else {}
                against = {
                    "agg": paired(only, {f: agg_alone[agg_id][f] for f in shared})
                            if shared else None,
                    "reg": paired(only, {f: reg_alone[reg_id][f] for f in shared})
                            if shared else None,
                }
                if any(result is None for result in against.values()):
                    rows.append(row)
                    continue
                row["folds"] = " ".join(str(f) for f in shared)
                for half, result in against.items():
                    row[f"diffs_vs_{half}"] = " ".join(f"{d:.4f}" for d in result["diffs"])
                    row[f"mean_vs_{half}"] = f"{result['mean']:.4f}"
                    row[f"sd_vs_{half}"] = f"{result['sd']:.4f}"
                    row[f"all_positive_vs_{half}"] = int(result["all_positive"])
                row["beats_both_folds"] = int(against["agg"]["all_positive"]
                                              and against["reg"]["all_positive"])
                rows.append(row)
    if not rows:
        raise SystemExit(
            f"FATAL: no combination under {root}. The cross is defined by the "
            "shortlists it was emitted from, and neither is on disk here."
        )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    """A pair this study never ran leaves its columns empty rather than zero."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}")
    print("basis: TEST (final_evaluation). Selection ran on validation.")
    rows = rows_of(args.root, a0, p0)
    write_csv(args.out / "combos_folds.csv", rows)
    beats = sum(1 for row in rows if row.get("beats_both_folds") == 1)
    print(f"positive against BOTH halves on every fold: {beats} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
