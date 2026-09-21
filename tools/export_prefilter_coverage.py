#!/usr/bin/env python3
"""
What the coarse pre-filter kept, and what the cohorts actually needed of it.

The detector is two stages and the manuscript describes it as two stages, but
only the second one has a view. The first is `g-init`, trained once on the whole
digit dataset and used to score every writer on the rows that model held out;
the worst `bad_fraction` of that ranking becomes the bad pool. The second is
`g-0`, the shipped model, which scores every writer OF THAT POOL on all of its
rows; the worst k of THAT ranking are the study's cohorts. So the pre-filter
decides which writers the shipped model is ever asked about.

    python tools/export_prefilter_coverage.py --root "$FOA_STUDY_DIR" \\
        --csv "$FOA_STUDY_DIR/tables/paper/"

THE COVERAGE NUMBER IS TRUE AND IT IS NOT EVIDENCE, AND THE VIEW SAYS BOTH.
Every writer of every cohort was kept by the pre-filter - the coverage column
reads 1.0 at five, ten and twenty - and it could not read anything else: the
cohorts are cut out of the pool, so a writer the pre-filter dropped is a writer
`g-0` was never asked to score. The number worth having is the one the records
CAN answer, and it is the next column: how deep into the detector's own ranking
the cohort's writers actually sit. If the worst cohort writer stands at rank 191
of 3,580, then a pre-filter a sixth as wide would have kept the same cohort, and
the width that was used had that much room in it. That is a statement about the
margin the pre-filter ran with, and it is checkable; a recall of the true worst-k
under `g-0` is not, because `g-0` scored the pool and nothing else.

WHAT THE TIE COLUMNS ARE FOR. The detector scores a writer as an accuracy over a
few hundred held-out rows, and most writers get all of them right: of 3,580
writers, 2,946 score exactly 1.0 and the whole ranking takes 79 distinct values.
The cut at 30% therefore lands deep inside that top tie, and the rule breaks ties
by writer id - so several hundred members of the bad pool are in it
alphabetically rather than because the detector said anything about them.
`ranked_above_ties` is how many writers the detector actually separated, and the
cohorts sit well inside it, which is the thing that had to be checked.

Reads `outliers/` and nothing else, so it runs against a bare clone of the
records. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

#: The pre-filter's own two files: the score of every writer it ranked, and the
#: pool it cut out of that ranking.
SCORES = "writer_scores.json"
POOLS = "pools.json"

#: The cohorts, in the order the study cut them, and the file each one is in.
COHORTS = (("five", 5, "cohort_worst5.json"),
           ("ten", 10, "cohort_worst10.json"),
           ("twenty", 20, "cohort_worst20.json"))

COLUMNS = ("cohort", "k", "clients", "kept_by_prefilter", "coverage",
           "deepest_prefilter_rank", "depth_fraction",
           "prefilter_pool", "ranked_writers", "pool_fraction",
           "ranked_above_ties", "distinct_scores", "writers_at_top_score",
           "cut_inside_tie")


def _read(root: Path, name: str) -> dict:
    path = Path(root) / "outliers" / name
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. The pre-filter's own ranking is what this view "
            "is about; recomputing it would need the detector's model and the "
            "whole digit cache."
        )
    return json.loads(path.read_text())


def detector_ranking(root: Path) -> List[tuple]:
    """
    Every writer the pre-filter scored, worst first, ties by writer id.

    THE ORDER IS THE POOL'S OWN, NOT ONE INVENTED HERE. `pools.json` states the
    rule in words and carries the resulting pool in order, so the ranking is
    rebuilt under that rule and then checked against the stored pool: if the two
    disagree, the tie-break or the score file has moved and the ranks below are
    ranks in a different ordering.
    """
    payload = _read(root, SCORES)
    scores: Dict[str, float] = {}
    for entry in payload.get("scores") or []:
        scores.update(entry)
    if not scores:
        raise SystemExit(f"{SCORES} carries no writer scores")
    order = sorted(scores.items(), key=lambda item: (item[1], item[0]))

    pools = _read(root, POOLS)
    bad = pools.get("bad") or []
    rebuilt = [writer for writer, _ in order[:len(bad)]]
    if rebuilt != bad:
        raise SystemExit(
            "FATAL: the pool rebuilt from the detector's scores is not the "
            "pool outliers/pools.json stores, so the ranks this view would "
            "print are ranks in an ordering the study did not cut on."
        )
    return order


def rows_of(root: Path) -> List[dict]:
    """One row per cohort size, with the pre-filter's own shape beside it."""
    order = detector_ranking(root)
    rank = {writer: place for place, (writer, _) in enumerate(order, start=1)}
    pools = _read(root, POOLS)
    pool = set(pools.get("bad") or [])
    ranked = len(order)
    values = [score for _, score in order]
    top = values[-1]
    at_top = sum(1 for score in values if score == top)
    separated = ranked - at_top
    cut = len(pool)

    rows = []
    for name, k, filename in COHORTS:
        clients = _read(root, filename).get("clients") or []
        if len(clients) != k:
            raise SystemExit(
                f"{filename} names {len(clients)} clients and this view "
                f"expects {k}; the cohort sizes are what the macros are "
                "named after."
            )
        kept = [c for c in clients if c in pool]
        deepest = max(rank[c] for c in clients)
        rows.append({
            "cohort": name, "k": k, "clients": len(clients),
            "kept_by_prefilter": len(kept),
            "coverage": "%.4f" % (len(kept) / len(clients)),
            "deepest_prefilter_rank": deepest,
            "depth_fraction": "%.6f" % (deepest / ranked),
            "prefilter_pool": cut, "ranked_writers": ranked,
            "pool_fraction": "%.6f" % (cut / ranked),
            "ranked_above_ties": separated,
            "distinct_scores": len(set(values)),
            "writers_at_top_score": at_top,
            "cut_inside_tie": "1" if cut > separated else "0",
        })
    return rows


def show(rows: List[dict]) -> None:
    first = rows[0]
    print("\nTHE PRE-FILTER, AND WHAT THE COHORTS NEEDED OF IT")
    print(f"basis: outliers/{SCORES}, the detector's accuracy on the rows its")
    print("own fold held out; ties broken by writer id, exactly as the pool was cut.")
    print(f"pool: {first['prefilter_pool']} of {first['ranked_writers']} writers "
          f"({100 * float(first['pool_fraction']):.1f}%); the detector separated "
          f"{first['ranked_above_ties']} of them and scored the rest alike")
    print(f"{'cohort':<9}{'k':>4}{'kept':>6}{'coverage':>10}"
          f"{'deepest rank':>14}{'of ranked':>11}")
    print("-" * 54)
    for r in rows:
        print(f"{r['cohort']:<9}{r['k']:>4}{r['kept_by_prefilter']:>6}"
              f"{float(r['coverage']):>10.2f}{r['deepest_prefilter_rank']:>14}"
              f"{100 * float(r['depth_fraction']):>10.2f}%")
    print("coverage is 1.00 by construction: the cohorts are cut out of the pool,")
    print("so a writer the pre-filter dropped was never scored by the shipped model.")
    print("The deepest rank is what the records can answer: the narrowest")
    print("pre-filter that would have kept the same cohort.")


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root, or study/artifacts/Digits_study01.")
    ap.add_argument("--csv", required=True, type=Path,
                    help="Directory the CSV view is written into.")
    args = ap.parse_args()
    rows = rows_of(args.root)
    show(rows)
    write_csv(args.csv / "prefilter_coverage.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
