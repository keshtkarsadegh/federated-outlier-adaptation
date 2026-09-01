#!/usr/bin/env python3
"""
The two baseline views `fig_baselines` is drawn on: what the shipped model cost
to train, and what fine-tuning one client in isolation does to it.

Neither is a federated result and neither belongs in `report_tables.py`, which
reports arms. These are the two rungs the arms are read against - the model
before anything touched it, and the extreme in the other direction from the
extremes: no federation at all, one client training alone.

    python tools/export_baseline_views.py --root "$FOA_STUDY_DIR" --out paper/data

`g0_training.csv` is g-0's own training history, one row per (fold, epoch), read
from each fold's `global_results/global_metrics.json`. The folds stop at
different epochs because training early-stops on validation, so the file is
ragged by design and anything reading it has to mean over the folds that reached
a given epoch rather than over five.

`isolated_clients.csv` is one row per (cohort client, fold): what the shipped
model already got right on that client's own test rows, and what one client
training alone reaches - from g-0 and from scratch - on its own rows and on the
source population. THE ISOLATION RECORDS STORE TEST EVALUATIONS ONLY, so this
view is on the test basis throughout while the per-round views beside it are on
validation. The two are not comparable and the figure's caption says so.

The shipped model's own per-client accuracies come from
`g0_perfold_evaluations.json` - the same file `report_tables.baselines` reads
`A0` out of - rather than from the isolation records, so the "before" column is
the study's own evaluation of g-0 and not a number the isolation stage recomputed.

TEN SIGNIFICANT FIGURES. Both files carry accuracies as `%.10g`, which is what
the manuscript's own build wrote and is far more precision than any of them is
worth; it is fixed here so that a regenerated view is byte-comparable with the
shipped one rather than differing in the last place for no reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The folds the study cross-validates over, everywhere.
FOLDS = (1, 2, 3, 4, 5)

#: How every accuracy in these two views is written.
PRECISION = "%.10g"

G0_COLUMNS = ("epoch", "fold", "train_acc", "val_acc")

ISOLATED_COLUMNS = (
    "client", "fold", "g0_own_test",
    "iso_g0_own_test", "iso_g0_src_test", "iso_g0_best_val_own", "iso_g0_epochs_run",
    "iso_scratch_own_test", "iso_scratch_src_test", "iso_scratch_best_val_own",
    "iso_scratch_epochs_run",
    "train_samples", "val_samples", "test_samples",
)


def number(value) -> str:
    """One accuracy, at the precision both views are written at."""
    return PRECISION % float(value)


def read_json(path: Path, why: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} is missing; {why} is read from it and not guessed.")
    return json.loads(path.read_text())


# ------------------------------------------------------------- g-0's training
def g0_training(root: Path) -> List[dict]:
    """One row per (fold, epoch) of the shipped model's own training history."""
    rows: List[dict] = []
    for fold in FOLDS:
        path = root / f"g0_fold{fold}" / "global_results" / "global_metrics.json"
        payload = read_json(path, "the shipped model's training history")
        train = payload.get("train_accuracies") or []
        val = payload.get("val_accuracies") or []
        if not train or len(train) != len(val):
            raise SystemExit(
                f"FATAL: {path} carries {len(train)} training and {len(val)} "
                "validation epochs. A history whose two series disagree in "
                "length is not a history of one run."
            )
        for epoch, (train_acc, val_acc) in enumerate(zip(train, val), start=1):
            rows.append({"epoch": epoch, "fold": fold,
                         "train_acc": number(train_acc),
                         "val_acc": number(val_acc)})
    if not rows:
        raise SystemExit(f"FATAL: no g-0 training history under {root}.")
    return rows


# ------------------------------------------------------------ isolated clients
def isolation_records(root: Path, tag: str) -> Dict[str, Dict[int, Dict[str, dict]]]:
    """``{client: {fold: {init: record}}}`` for every isolated run on disk."""
    folder = root / f"isolated_{tag}"
    found: Dict[str, Dict[int, Dict[str, dict]]] = {}
    for init in ("global", "scratch"):
        for path in sorted(folder.glob(f"isolated_{init}_*_fold*.json")):
            stem = path.stem[len(f"isolated_{init}_"):]
            client, _, fold_text = stem.rpartition("_fold")
            try:
                fold = int(fold_text)
            except ValueError:
                continue
            payload = json.loads(path.read_text())
            record = (payload.get("per_client") or {}).get(client)
            if record is None:
                continue
            found.setdefault(client, {}).setdefault(fold, {})[init] = record
    if not found:
        raise SystemExit(
            f"FATAL: no isolated run under {folder}. The rung is part of the "
            "published records; without it this view states nothing."
        )
    return found


def isolated_clients(root: Path, tag: str) -> List[dict]:
    """One row per (client, fold): the shipped model, then each isolated run."""
    perfold = read_json(root / "g0_perfold_evaluations.json",
                        "the shipped model on each cohort client")
    records = isolation_records(root, tag)

    rows: List[dict] = []
    for client in sorted(records):
        for fold in sorted(records[client]):
            pair = records[client][fold]
            if set(pair) != {"global", "scratch"}:
                raise SystemExit(
                    f"FATAL: {client} fold {fold} has only {sorted(pair)} on "
                    "disk. The view prices one against the other and a row with "
                    "one of them is a comparison with itself."
                )
            written = (perfold.get(f"cohort_fold{fold}") or {}).get("per_writer") or {}
            if client not in written:
                raise SystemExit(
                    f"FATAL: the shipped model's evaluation of fold {fold} does "
                    f"not name {client}. Every gain here is measured from it."
                )
            row = {"client": client, "fold": fold,
                   "g0_own_test": number(written[client])}
            for init, prefix in (("global", "iso_g0"), ("scratch", "iso_scratch")):
                record = pair[init]
                convergence = record.get("convergence") or {}
                row[f"{prefix}_own_test"] = number(record["own"])
                row[f"{prefix}_src_test"] = number(record["old"])
                row[f"{prefix}_best_val_own"] = number(convergence["best_val_accuracy"])
                row[f"{prefix}_epochs_run"] = convergence["epochs_run"]
            # The row counts are the client's split, not a result, so they are
            # taken from the run that started at g-0 and asserted to agree with
            # the other rather than written twice.
            for field in ("train_samples", "val_samples", "test_samples"):
                values = {pair[init][field] for init in ("global", "scratch")}
                if len(values) != 1:
                    raise SystemExit(
                        f"FATAL: {client} fold {fold} was trained on two "
                        f"different splits ({field}: {sorted(values)}). The two "
                        "isolated runs are the same client or they are not "
                        "comparable."
                    )
                row[field] = values.pop()
            rows.append(row)
    return rows


def write_csv(path: Path, columns, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)")


VIEWS = ("g0_training", "isolated_clients")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory the CSV views are written into.")
    ap.add_argument("--what", default="all", choices=("all",) + VIEWS)
    ap.add_argument("--tag", default="c10",
                    help="Reference cohort tag, as report_tables.py uses it.")
    args = ap.parse_args()

    print("basis: TEST. The isolation records store test evaluations only, and "
          "g-0's history is its own validation curve.")
    for what in (VIEWS if args.what == "all" else (args.what,)):
        if what == "g0_training":
            write_csv(args.out / "g0_training.csv", G0_COLUMNS, g0_training(args.root))
        else:
            write_csv(args.out / "isolated_clients.csv", ISOLATED_COLUMNS,
                      isolated_clients(args.root, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
