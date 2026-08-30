#!/usr/bin/env python3
"""
Reopen the ranges whose winner sat on the edge.

The screen sweeps each coefficient over a fixed row of values. When the best
cell of a method is the FIRST or LAST value of that row, the row is the wrong
answer to give: the optimum may lie outside it, and the winner is only "best"
in the sense of "best among what we happened to try". The selector already
detects this and writes it to tables/BOUNDARY_HITS.txt.

Until now the extension that answers those hits was a task file written by hand
for one particular screen. Run against a different cohort it extended a range
that was not at its edge and left the ones that were - which is the same class
of defect as a cohort file made by hand: it looks like part of the programme and
is really a memory of an older one.

This reads the hits and continues each row in the direction it ran out, by two
further values, keeping the geometric step the row already uses. Each axis is
extended on its own, with the winner's other coefficients held fixed, so a cell
differs from the winner in exactly one number.

    python tools/make_boundary_ext.py --root $FOA_STUDY_DIR --out <task file>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

from federated_outlier_adaptation.training import agg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import STUDIES

#: Seeds of the extension. Clear of the screen's block (2000) and the finals'
#: (4000) so no two lines of this study draw the same participation pattern.
SEED_OFFSET = 6000

HIT = re.compile(
    r"^(?P<cell>[^:\t]+): (?P<flag>\w+)=(?P<value>[-\d.e+]+) is the "
    r"(?P<end>LOW|HIGH) end of (?P<row>\[.*\])$"
)


def tag(value: float) -> str:
    """The id spelling this study uses for a number: 0.001 -> 0p001, 1e-05 -> 1em05."""
    if value >= 1 and float(value).is_integer():
        return str(int(value))
    text = repr(float(value))
    if "e" in text:
        mantissa, exponent = text.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
        return f"{mantissa}em{abs(int(exponent)):02d}"
    return text.rstrip("0").rstrip(".").replace(".", "p")


def continue_row(row: list, end: str, steps: int = 2) -> list:
    """Two more values, in the direction the row ran out, at the row's own step."""
    row = sorted(row)
    if len(row) < 2:
        return []
    if end == "LOW":
        ratio = row[1] / row[0]
        out, value = [], row[0]
        for _ in range(steps):
            value = value / ratio
            out.append(value)
    else:
        ratio = row[-1] / row[-2]
        out, value = [], row[-1]
        for _ in range(steps):
            value = value * ratio
            out.append(value)
    # Round away the float dust a repeated multiply leaves behind.
    return [float(f"{v:.6g}") for v in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--study", default="Digits_study01")
    ap.add_argument("--clients-per-round", type=int, default=None)
    ap.add_argument("--label", default="p11/agg-full",
                    help="Only act on hits recorded under this label.")
    args = ap.parse_args()

    cfg = STUDIES[args.study]
    if args.clients_per_round is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, clients_per_round=args.clients_per_round)

    report = args.root / "tables" / "BOUNDARY_HITS.txt"
    if not report.is_file():
        raise SystemExit(f"no boundary report at {report}")

    by_id = {c["id"]: c for c in agg_cells.screen_cells()}
    seen, new_cells = set(), []

    for line in report.read_text().splitlines():
        if not line.strip():
            continue
        label, _, body = line.partition("\t")
        if label != args.label:
            continue
        m = HIT.match(body.strip())
        if not m:
            print(f"  unparsed: {body}")
            continue
        base = by_id.get(m["cell"])
        if base is None:
            print(f"  unknown cell: {m['cell']}")
            continue
        flag, end = m["flag"], m["end"]
        row = ast.literal_eval(m["row"])
        for value in continue_row(row, end):
            flags = dict(base["flags"])
            flags[flag] = value
            # THE ID DECIDES WHICH METHOD THE CELL BELONGS TO. Grouping is by
            # id prefix, not by rule, so a cell named from the rule's last word
            # forms its own method of one and is never compared against the row
            # it was meant to extend. It runs, it is collected, and it loses to
            # nothing.
            method = SL.agg_method_of(base)
            parts = [method]
            parts += [f"{k.split('_')[-1]}{tag(v)}" for k, v in sorted(flags.items())]
            cell_id = "_".join(parts)
            if cell_id in by_id or cell_id in seen:
                continue
            if SL.agg_method_of({"id": cell_id}) != method:
                raise SystemExit(
                    f"{cell_id} would group under "
                    f"{SL.agg_method_of({'id': cell_id})!r}, not {method!r}; "
                    "the extension would never be compared against its own row."
                )
            seen.add(cell_id)
            new_cells.append({
                "id": cell_id,
                "path": base["path"],
                "rule": base["rule"],
                "flags": flags,
                "note": (f"boundary extension of {base['id']}: {flag}={value}, "
                         f"{end.lower()} of the swept row"),
            })

    if not new_cells:
        print("no extension needed: no winner sits at a grid edge.")
        args.out.write_text("# no boundary extension needed\n")
        return 0

    lines = []
    for index, cell in enumerate(new_cells):
        for fold in SL.folds_of(cfg):
            lines.append(SL.agg_line(
                cfg, cell, fold, SL.SCREEN_ROUNDS,
                cfg.seed_base + SEED_OFFSET + index * 10 + fold,
            ))

    header = [
        f"# boundary extension of {args.label}, generated by tools/make_boundary_ext.py",
        "#",
        "# A winner at the end of a swept row is only the best of what was tried.",
        "# Each row below is continued two values further in the direction it ran",
        "# out, at the step the row already uses, with the winner's other",
        "# coefficients held fixed - so a cell differs from its winner in exactly",
        "# one number. Same horizon as the screen, so the results are comparable",
        "# with it directly.",
        "#",
    ] + [f"#   {c['id']}: {c['note']}" for c in new_cells] + ["#"]

    args.out.write_text("\n".join(header + lines) + "\n")

    # THE SELECTOR HAS TO KNOW THESE EXIST. It builds its candidate list from
    # agg_cells.screen_cells(), a fixed list, so a cell invented here would be
    # trained and then ignored - the worst outcome of the three, because the GPU
    # time is spent and the answer is not used. Recording them beside the
    # results lets the selection pick them up without the cell table having to
    # anticipate every extension it might one day need.
    # ACCUMULATE, never replace. The boundary rule can fire more than once - a
    # reopened range can itself end at its new edge - and each round only knows
    # about its own hits. Overwriting would drop the cells of every earlier
    # round from the selector's candidate list while their results sat on disk,
    # which is the same silent loss this file exists to prevent.
    import json as _json
    record = args.root / "tables" / "boundary_ext_cells.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    if record.is_file():
        kept = [c for c in _json.loads(record.read_text())
                if c["id"] not in {n["id"] for n in new_cells}]
    record.write_text(_json.dumps(kept + new_cells, indent=2) + "\n")
    if kept:
        print(f"  kept {len(kept)} cell(s) from earlier boundary rounds")

    print(f"wrote {args.out}: {len(new_cells)} cells x {len(SL.folds_of(cfg))} folds "
          f"= {len(lines)} tasks")
    print(f"wrote {record}")
    for c in new_cells:
        print(f"  {c['id']:<34} {c['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
