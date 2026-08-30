#!/usr/bin/env python3
"""
The three extreme cohorts, cut from the same ranking as every other cohort.

The extreme points ask what federating is worth when there is almost nothing to
federate: one badly served writer alone, the two worst as two clients, and the
same two merged into one client. The three share their writers by construction,
which is what makes the comparison controlled - double and dual hold exactly the
same rows and differ only in whether a client boundary runs between them.

These files existed in the first programme but nothing produced them: they had
been written by hand, and a rebuild from an empty root reached the extreme stage
with no cohort to read. Worse, the reference tasks name their writers literally,
so a hand-made file that went stale would have sent the stage to train a writer
the current ranking never chose, and it would have done it silently. The rule is
small enough to state in one line of code, so it belongs in the programme.

The ranking is the worst-k cohort file, which is already ordered worst-first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--from-cohort", default="cohort_worst10.json",
                    help="Ranked cohort to cut the extremes from.")
    ap.add_argument("--tag", default="digits_extreme")
    args = ap.parse_args()

    outliers = args.study_dir / "outliers"
    payload = json.loads((outliers / args.from_cohort).read_text())
    ranked = payload["clients"] if isinstance(payload, dict) else list(payload)
    if len(ranked) < 2:
        raise SystemExit(f"{args.from_cohort} holds {len(ranked)} clients; need 2")

    worst, second = ranked[0], ranked[1]
    accuracies = payload.get("accuracies", {}) if isinstance(payload, dict) else {}

    cohorts = {
        "extreme_single": {
            "clients": [worst],
            "rule": "the worst writer of the ranking, alone",
        },
        "extreme_double": {
            "clients": [worst, second],
            "rule": "the two worst writers, one client each",
        },
        "extreme_dual": {
            "clients": [f"{worst}+{second}"],
            "rule": ("the same two writers merged into one client; the rows are "
                     "identical to extreme_double and only the client boundary "
                     "differs"),
        },
    }

    for name, body in cohorts.items():
        body.update({
            "tag": args.tag,
            "size": len(body["clients"]),
            "cut_from": args.from_cohort,
            "accuracies": {w: accuracies[w] for w in (worst, second)
                           if w in accuracies},
        })
        path = outliers / f"{name}.json"
        path.write_text(json.dumps(body, indent=2) + "\n")
        print(f"  {name:16s} {body['clients']}")
    print(f"wrote 3 cohorts to {outliers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
