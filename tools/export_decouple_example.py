#!/usr/bin/env python3
"""
The five writers that show the two selection rankings coming apart.

The cohort is cut in two stages: a detector `g_init` trained on everyone splits
the population into a good and a bad pool, and the shipped model `g-0` - trained
on the good pool only - then re-scores every writer of the bad pool and the
cohorts are cut from THAT ranking. The protocol section defends the second
stage by showing that the two rankings decouple, and this view is the evidence
it quotes: three writers the detector calls its worst that the shipped model has
no difficulty with, and the two writers the shipped model actually struggles on,
with the rank the detector gave them beside.

    python tools/export_decouple_example.py --root "$FOA_STUDY_DIR" --out paper/data

It also runs against the released records on their own, which have no run
directories in them:

    python tools/export_decouple_example.py --root study/artifacts/Digits_study01 \\
        --out paper/data

WHICH WRITERS ARE READ, NOT TYPED. Both blocks come out of the frozen selection
records under `outliers/`:

    outliers/pools.csv             the detector's ranking of the bad pool
    outliers/bad_scores_on_g0.csv  the shipped model's ranking of the same pool
    outliers/cohort_worst20.json   the largest cohort ever cut from that ranking

The first block is the detector's worst writers that the shipped model keeps out
of EVERY cohort - that is, whose `g-0` rank is past the largest cohort this study
cut - which is exactly the claim the section makes: the detector nominates
clients the deployment would never have selected. The second block is the
shipped model's own worst, in its own order. Naming the five here instead would
let the paragraph outlive the selection that produced it, which is the failure
this repository's other export tools exist to prevent.

THE TWO BLOCK SIZES ARE THE ONLY NUMBERS STATED. The section shows three false
alarms and two misses; nothing in the records fixes those counts, so they are
constants here and are the one thing a reader must take from the manuscript
rather than from the study tree.

WHICH ROW IS QUOTED IS THE ONE WITH THE MOST ROWS BEHIND IT. `chosen` marks the
single row `make_numbers.py` reads the two section numbers from, and it is the
false alarm with the largest sample count. The detector's very worst writer is a
starker case - the shipped model serves it perfectly - but on eleven rows, which
is too thin a measurement to headline a paragraph; the section discloses it in
prose instead. Picking the stark row here would have put an eleven-row accuracy
into the manuscript as a number.

NOTHING IS RECOMPUTED. Every accuracy is passed through as the string the record
stores, so this view cannot round a number the selection recorded, and the two
ranks are the ranks those two files assigned.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

#: How many of each block the section shows. Read the docstring before changing
#: either: these two counts are the manuscript's framing, not a measurement.
FALSE_ALARMS = 3
MISSES = 2

COLUMNS = ("writer", "g_init_acc", "g0_acc", "g0_rank", "g0_rank_of",
           "ginit_rank", "ginit_rank_of", "n_rows", "chosen")


def detector_ranking(root: Path) -> Dict[str, Tuple[str, int]]:
    """The bad pool as the detector ranked it: writer -> (accuracy, rank)."""
    path = root / "outliers" / "pools.csv"
    if not path.exists():
        raise SystemExit(
            f"FATAL: {path} is not on disk. The decoupling view is a read of "
            "the two selection rankings, and the detector's is not here."
        )
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            # The good pool is in the same file and was never re-scored, so it
            # has no counterpart to be compared against.
            if row["pool"] != "bad":
                continue
            out[row["writer"]] = (row["accuracy"], int(row["rank"]))
    return out


def shipped_ranking(root: Path) -> Dict[str, Tuple[str, int, int]]:
    """The same pool as g-0 ranked it: writer -> (accuracy, n_rows, rank)."""
    path = root / "outliers" / "bad_scores_on_g0.csv"
    if not path.exists():
        raise SystemExit(
            f"FATAL: {path} is not on disk. The decoupling view is a read of "
            "the two selection rankings, and the shipped model's is not here."
        )
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[row["writer"]] = (row["accuracy"], int(row["n_rows"]),
                                  int(row["rank"]))
    return out


def largest_cohort(root: Path) -> int:
    """How many writers the biggest cohort ever cut from the g-0 ranking took."""
    sizes = []
    for path in sorted((root / "outliers").glob("cohort_worst*.json")):
        with open(path) as handle:
            record = json.load(handle)
        sizes.append(int(record.get("size", len(record.get("clients", ())))))
    if not sizes:
        raise SystemExit(
            f"FATAL: no cohort record under {root / 'outliers'}. Which writers "
            "the shipped model selected is what the first block is defined "
            "against, and it cannot be guessed."
        )
    return max(sizes)


def rows_of(root: Path) -> List[dict]:
    detector = detector_ranking(root)
    shipped = shipped_ranking(root)
    # BOTH RANKINGS MUST COVER THE SAME POPULATION. A rank of 1,074 quoted
    # beside a rank out of some other number would be two different statements
    # printed in one row.
    if set(detector) != set(shipped):
        raise SystemExit(
            f"FATAL: the detector ranked {len(detector)} writers and the "
            f"shipped model {len(shipped)}, and "
            f"{len(set(detector) ^ set(shipped))} of them are in one ranking "
            "only. The two columns of this view would not be about one pool."
        )
    pool = len(detector)
    cohort = largest_cohort(root)

    # The detector's worst that the shipped model keeps out of every cohort.
    alarms = [writer for writer, (_acc, rank) in
              sorted(detector.items(), key=lambda kv: kv[1][1])
              if shipped[writer][2] > cohort][:FALSE_ALARMS]
    # The shipped model's own worst, in its own order.
    misses = [writer for writer, _v in
              sorted(shipped.items(), key=lambda kv: kv[1][2])][:MISSES]
    if len(alarms) < FALSE_ALARMS or len(misses) < MISSES:
        raise SystemExit(
            f"FATAL: the rankings under {root} do not decouple far enough to "
            f"fill this view ({len(alarms)} of {FALSE_ALARMS} false alarms, "
            f"{len(misses)} of {MISSES} misses). The section's paragraph is "
            "about this study's numbers and cannot be written from these."
        )
    # The quoted row is the false alarm resting on the most rows.
    quoted = max(alarms, key=lambda writer: shipped[writer][1])

    rows = []
    for writer in alarms + misses:
        ginit_acc, ginit_rank = detector[writer]
        g0_acc, n_rows, g0_rank = shipped[writer]
        rows.append({"writer": writer,
                     "g_init_acc": ginit_acc,
                     "g0_acc": g0_acc,
                     "g0_rank": g0_rank,
                     "g0_rank_of": pool,
                     "ginit_rank": ginit_rank,
                     "ginit_rank_of": pool,
                     "n_rows": n_rows,
                     "chosen": "yes" if writer == quoted else "no"})
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", "--study", required=True, type=Path, dest="root",
                    help="The study root ($FOA_STUDY_DIR), or the released "
                         "records directory.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    args = ap.parse_args()

    rows = rows_of(args.root)
    print(f"pool: {rows[0]['ginit_rank_of']} writers ranked by both models.")
    print("basis: the two SELECTION rankings, not any adaptation result.")
    for row in rows:
        print(f"  {row['writer']}  detector #{row['ginit_rank']:>4} "
              f"({row['g_init_acc'][:6]})   shipped #{row['g0_rank']:>4} "
              f"({row['g0_acc'][:6]})   {row['n_rows']:>3} rows"
              f"{'   <- quoted' if row['chosen'] == 'yes' else ''}")
    write_csv(args.out / "decouple_example.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
