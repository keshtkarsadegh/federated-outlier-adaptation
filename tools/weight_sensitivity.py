#!/usr/bin/env python3
"""
Which winners depend on the weight the selection rule puts on forgetting?

The study selects by

    score = (adaptation - A0) - w * (P0 - preservation)

and runs at ``w = 1``.  ``w`` is a choice, and for a while this repository
claimed it was a free one: ``trade_score``'s docstring said the same cell wins
at one, two and three points.  That was never checked and it is false.  This
tool is what checks it, so the claim in the paper is a measured sensitivity
rather than an assertion.

WHY IT MATTERS.  ``w`` is the only number in the selection rule that is not read
off disk - A0 and P0 are the shipped model's own measured accuracies.  A grid
whose winners move with ``w`` is a grid whose reported winners are partly a
choice of the authors, and the honest way to publish that is a sensitivity row
naming which methods move and which do not.

    python tools/weight_sensitivity.py --root "$FOA_STUDY_DIR" --grid agg
    python tools/weight_sensitivity.py --root "$FOA_STUDY_DIR" --grid reg
    python tools/weight_sensitivity.py --root "$FOA_STUDY_DIR" --grid both --csv out/

Selection is per method - the unit the study ranks within - and, for the
regularisation grid, per (method, schedule), because a penalty that helps a
server average and one that helps a sequential walk are different findings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from federated_outlier_adaptation.training import agg_cells, reg_cells  # noqa: E402
from federated_outlier_adaptation.training import study_lines as SL  # noqa: E402

import select_agg_screen  # noqa: E402
import select_reg_screen  # noqa: E402

#: Weights the winners are recomputed at.  1 is what the study runs; the rest
#: are there to show how far a winner has to be pushed before it moves.
DEFAULT_WEIGHTS: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)


def baselines(root: Path) -> Tuple[float, float]:
    """``(A0, P0)`` - the shipped model's own accuracies, read not assumed."""
    def mean_accuracy(name: str) -> Optional[float]:
        path = root / name
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else list(payload.values())
        values = [r["accuracy"] for r in records
                  if isinstance(r, dict) and isinstance(r.get("accuracy"), (int, float))]
        return sum(values) / len(values) if values else None

    a0, p0 = mean_accuracy("g0_perfold_evaluations.json"), mean_accuracy("g0_evaluations.json")
    if a0 is None or p0 is None:
        raise SystemExit(
            "FATAL: the shipped model's own accuracies are missing "
            f"({root}/g0_perfold_evaluations.json, g0_evaluations.json). "
            "Every score here is measured against them and cannot be guessed."
        )
    return a0, p0


def score(row: dict, a0: float, p0: float, weight: float) -> float:
    """The selection rule at one weight, in points."""
    return ((row["adaptation"]["mean"] - a0)
            - weight * (p0 - row["preservation"]["mean"])) * 100.0


def agg_rows(root: Path) -> Dict[Tuple[str, ...], List[dict]]:
    """Aggregation screen rows, grouped by the unit selection ranks within."""
    cells = agg_cells.screen_cells()
    scored = {r["id"]: r for r in select_agg_screen.summarise(
        select_agg_screen.collect(root, cells, "d01_agg_"))
        if r["adaptation"]["mean"] is not None}
    grouped: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for cell in cells:
        if cell["id"] in scored:
            grouped[(cell["path"], SL.agg_method_of(cell))].append(scored[cell["id"]])
    return grouped


def reg_rows(root: Path) -> Dict[Tuple[str, ...], List[dict]]:
    """Regularisation screen rows, grouped by (schedule, method)."""
    cells = reg_cells.screen_cells()
    by_id = {c["id"]: c for c in cells}
    grouped: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in select_reg_screen.summarise(
            select_reg_screen.collect(root, cells, "d01_reg_")):
        if row["adaptation"]["mean"] is None:
            continue
        cell = by_id.get(row["id"])
        if cell is None:
            continue
        grouped[(row["family"], cell["method"])].append(row)
    return grouped


GRIDS = {"agg": (agg_rows, "AGGREGATION"), "reg": (reg_rows, "REGULARISATION")}


def sensitivity(grouped, a0: float, p0: float, weights) -> List[dict]:
    """Per group: the winning cell at each weight, and whether it moved."""
    out: List[dict] = []
    for key in sorted(grouped):
        siblings = grouped[key]
        winners = {w: max(siblings, key=lambda r: score(r, a0, p0, w)) for w in weights}
        base = winners[weights[0]]
        moved = {w: winners[w]["id"] != base["id"] for w in weights}
        out.append({
            "group": "/".join(key),
            "winners": {w: winners[w]["id"] for w in weights},
            "changed": any(moved.values()),
            "base": base,
            "winner_rows": winners,
        })
    return out


def show(rows: List[dict], title: str, weights, a0: float, p0: float) -> None:
    # sized to the content: a truncated cell id is a different cell
    name_w = max([len(r["group"]) for r in rows] + [len("method")]) + 2
    cell_w = max([len(i) for r in rows for i in r["winners"].values()]) + 2
    header = f"{'method':<{name_w}}" + "".join(
        f"{'w=' + f'{w:g}':<{cell_w}}" for w in weights)
    print(f"\n{'=' * len(header)}")
    print(f"{title} - winner at each forgetting weight")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['group']:<{name_w}}"
        for w in weights:
            line += f"{row['winners'][w]:<{cell_w}}"
        print(line + ("  CHANGED" if row["changed"] else ""))
    changed = [r for r in rows if r["changed"]]
    print("-" * len(header))
    # per weight, so a claim about one weight can be read straight off
    reference = weights[0]
    for w in weights[1:]:
        n = sum(1 for r in rows if r["winners"][w] != r["winners"][reference])
        print(f"  at w={w:g}: {n} of {len(rows)} winners differ from w={reference:g}")
    print(f"{len(changed)} of {len(rows)} winners change at some weight")
    if changed:
        print(f"\nwhat moving costs, on the screen (vs the w={weights[0]:g} winner):")
        for row in changed:
            for w in weights[1:]:
                if row["winners"][w] == row["winners"][weights[0]]:
                    continue
                new, old = row["winner_rows"][w], row["base"]
                da = (new["adaptation"]["mean"] - old["adaptation"]["mean"]) * 100
                dp = (new["preservation"]["mean"] - old["preservation"]["mean"]) * 100
                print(f"  w={w:g}  {row['group']:<24} adapt {da:+6.2f}pt   "
                      f"preserve {dp:+6.2f}pt")
                break


def write_csv(rows: List[dict], path: Path, weights) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method"] + [f"w={w:g}" for w in weights] + ["changed"])
        for row in rows:
            writer.writerow([row["group"]] + [row["winners"][w] for w in weights]
                            + [int(row["changed"])])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--grid", default="both", choices=("both",) + tuple(sorted(GRIDS)))
    parser.add_argument("--weights", type=float, nargs="+", default=list(DEFAULT_WEIGHTS),
                        help="Weights to recompute winners at; the first is the reference.")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    a0, p0 = baselines(args.root)
    weights = tuple(args.weights)
    print(f"shipped on cohort A0={a0:.4f}   shipped on source P0={p0:.4f}")
    print(f"reference weight w={weights[0]:g} - the one the study selects and reports at")

    for name in (sorted(GRIDS) if args.grid == "both" else [args.grid]):
        builder, title = GRIDS[name]
        grouped = builder(args.root)
        if not grouped:
            print(f"\n{title}: no scored rows under this root.")
            continue
        rows = sensitivity(grouped, a0, p0, weights)
        show(rows, title, weights, a0, p0)
        if args.csv:
            write_csv(rows, args.csv / f"weight_sensitivity_{name}.csv", weights)
            print(f"\nwrote {args.csv / f'weight_sensitivity_{name}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
