#!/usr/bin/env python3
"""
What is actually in a cohort: writers, rows, and label coverage per writer.

WHY THIS EXISTS.  The study's leading penalty is not-true distillation, and
FedNTD was designed for **label-distribution skew** - clients that hold only
some classes forget the ones they never see, so you preserve the teacher's
opinion on the absent classes.  Whether that motivation applies here is a
property of the data, not of the method, and it decides how the result may be
described in the paper.

It does not apply.  These cohorts are **style** outliers - writers the shipped
model serves badly - and they hold essentially every class.  So NTD is being
evaluated outside its motivating regime and wins anyway, which is a different
and stronger claim than "we applied FedNTD".  Making that claim requires this
number, so this prints it rather than leaving it to a remembered impression.

The same table answers the other question the design rests on: how much held-out
data a single writer actually has, which is what makes client weighting (``q``)
worth sweeping at all.

    python tools/describe_cohort.py --root "$FOA_STUDY_DIR" --cohort cohort10
    python tools/describe_cohort.py --root "$FOA_STUDY_DIR" --all --csv out/

Reads the fold book's own metadata, so the writers named are the writers that
ran - not a list re-derived from the outlier file and hoped to match.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

#: Digit classes.  The by-class task has 62; the digit ablation uses the first
#: ten, and a label at or above this is a letter and not part of this study.
DIGIT_CLASSES = 10


def load_book(root: Path, cohort: str) -> dict:
    """The fold book's metadata: which writers, in which order, at what seed."""
    path = root / "fold_books" / f"{cohort}.foldbook.npz"
    if not path.is_file():
        raise SystemExit(
            f"FATAL: {path} is missing. The cohort is defined by the book that "
            "ran; re-deriving it from the outlier file would describe a "
            "different set of writers than the one measured."
        )
    book = np.load(path, allow_pickle=True)
    raw = book["metadata"].item()
    meta = json.loads(raw) if isinstance(raw, str) else raw
    return {"meta": meta, "writer_index": book["writer_index"]}


def load_labels(data_dir: Path) -> np.ndarray:
    path = data_dir / "nist28_labels.npy"
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} is missing.")
    return np.load(path)


def describe(root: Path, data_dir: Path, cohort: str) -> List[dict]:
    """One row per writer: rows held, classes present, per-class spread."""
    book = load_book(root, cohort)
    meta, writer_index = book["meta"], book["writer_index"]
    labels = load_labels(data_dir)
    if len(labels) != len(writer_index):
        raise SystemExit(
            f"FATAL: the label array has {len(labels)} rows and the fold book "
            f"indexes {len(writer_index)}. They describe different datasets."
        )
    table = meta["writer_table"]
    rows: List[dict] = []
    for writer in meta["writers"]:
        index = table.index(writer)
        mask = writer_index == index
        digits = labels[mask]
        digits = digits[digits < DIGIT_CLASSES]
        counts = np.bincount(digits, minlength=DIGIT_CLASSES)
        rows.append({
            "cohort": cohort,
            "writer": writer,
            "rows": int(mask.sum()),
            "digit_rows": int(counts.sum()),
            "classes_present": int((counts > 0).sum()),
            "min_per_class": int(counts.min()),
            "max_per_class": int(counts.max()),
            "missing": [c for c in range(DIGIT_CLASSES) if counts[c] == 0],
        })
    return rows


def show(rows: List[dict], cohort: str) -> None:
    print(f"\n{'=' * 78}\n{cohort}: {len(rows)} writers\n{'=' * 78}")
    print(f"{'writer':<14}{'rows':>7}{'classes':>9}{'min/class':>11}"
          f"{'max/class':>11}   missing")
    print("-" * 78)
    for row in rows:
        miss = ",".join(str(c) for c in row["missing"]) or "-"
        print(f"{row['writer']:<14}{row['digit_rows']:>7}"
              f"{row['classes_present']:>6}/{DIGIT_CLASSES}"
              f"{row['min_per_class']:>11}{row['max_per_class']:>11}   {miss}")
    print("-" * 78)
    full = sum(1 for r in rows if r["classes_present"] == DIGIT_CLASSES)
    totals = [r["digit_rows"] for r in rows]
    print(f"writers holding all {DIGIT_CLASSES} classes: {full} of {len(rows)}")
    print(f"digit rows per writer: {min(totals)} to {max(totals)} "
          f"(mean {sum(totals) / len(totals):.0f})")

    # The verdict is about how much of the label grid is absent, not about how
    # many writers are short. Two writers each missing one class of ten is two
    # empty cells out of two hundred - counting writers would call that skew
    # and it plainly is not.
    cells = len(rows) * DIGIT_CLASSES
    empty = sum(len(r["missing"]) for r in rows)
    share = empty / cells
    print(f"empty (writer, class) cells: {empty} of {cells}  ({share:.1%})")
    if empty == 0:
        print("FULL LABEL COVERAGE: this cohort is not label-skewed. A penalty "
              "motivated by absent classes is being evaluated outside its regime.")
    elif share <= 0.05:
        print("NEAR-FULL LABEL COVERAGE: the skew here is in writing style, not "
              "in which classes a client holds. A penalty motivated by absent "
              "classes is still being evaluated outside its regime.")
    else:
        print("LABEL SKEW PRESENT: classes are genuinely missing from clients.")


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "writer", "rows", "digit_rows",
                         "classes_present", "min_per_class", "max_per_class",
                         "missing"])
        for row in rows:
            writer.writerow([row["cohort"], row["writer"], row["rows"],
                             row["digit_rows"], row["classes_present"],
                             row["min_per_class"], row["max_per_class"],
                             " ".join(str(c) for c in row["missing"])])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Where nist28_labels.npy lives; "
                             "defaults to $FOA_NIST28_DIR or <root>/../../data/nist28.")
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument("--all", action="store_true",
                        help="Every fold book in the study.")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    import os
    data_dir = (args.data_dir or Path(os.environ.get("FOA_NIST28_DIR", ""))
                or args.root.parent.parent / "data" / "nist28")
    if not Path(data_dir).is_dir():
        raise SystemExit(f"FATAL: dataset directory {data_dir} does not exist.")

    if args.all:
        cohorts = sorted(p.name.replace(".foldbook.npz", "")
                         for p in (args.root / "fold_books").glob("*.foldbook.npz")
                         if "cohort" in p.name)
    elif args.cohort:
        cohorts = args.cohort
    else:
        raise SystemExit("give --cohort or --all")

    every: List[dict] = []
    for cohort in cohorts:
        rows = describe(args.root, Path(data_dir), cohort)
        show(rows, cohort)
        every += rows
    if args.csv:
        write_csv(every, args.csv / "cohort_composition.csv")
        print(f"\nwrote {args.csv / 'cohort_composition.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
