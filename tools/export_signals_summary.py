#!/usr/bin/env python3
"""
The two signal views the forgetting-signals section is written on.

`foa signals` writes a per-run correlation table and a per-arm signal series
into `$FOA_STUDY_DIR/signals/`, and `stopping_table.py` turns the stored arms
into the stop a rule would have taken. The section reads both at once - a
signal is worth something only if it both tracks forgetting AND stops an arm
somewhere useful - and until now the join between them was made by a script
that lived beside the manuscript rather than in this repository. This tool is
that join, so the two extracts the section quotes are reachable from the
published records by a command rather than by a copy.

    python tools/export_signals_summary.py --root "$FOA_STUDY_DIR" --out paper/data

WHAT THE TWO FILES ARE. `signals_summary_extract.csv` is one row per permitted
signal: where the signal can be observed at all, its correlation with
forgetting, and - if the signal is defined on every stopping arm - the best
delta for it and what stopping there costs against running the schedule out.
`signals_extras_extract.csv` is the scalars the same section quotes in prose:
the arm and run counts, the fixed-schedule and oracle means, the two shipped-
model baselines, and how far retention wanders on the extreme arrangements
while the source population is actually being lost.

CORRELATION OF THE DRIFT, NOT OF THE SIGNAL. The stored correlations are of the
RAW signal against forgetting. The section reads the DRIFT, which is the signal
signed so that forgetting makes it grow. A rank correlation is invariant under
the shift and flips with the sign, so the drift correlation is the stored one
times the signal's direction - which is read from
`analysis.forgetting_signals.SIGNAL_DIRECTION`, not typed here. Both the signed
and the raw medians are written, so the flip can be checked rather than trusted.

A SIGNAL IS SCORED ONLY WHERE IT IS DEFINED ON EVERY ARM. A delta that fired on
some arms and was undefined on others would be a mean over a different
population per signal, and the columns would not be comparable down the table.
Such a signal keeps its correlation columns, gets `defined_on_all_arms = 0`, and
its stopping columns are left empty rather than filled with a partial mean.

NOTHING UNDER THE STUDY TREE IS WRITTEN. This is a read of the records and a
write into `--out`.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import stopping_table as ST  # noqa: E402
from federated_outlier_adaptation.analysis.forgetting_signals import (  # noqa: E402
    DEFAULT_DELTAS,
    SIGNAL_DIRECTION,
)
from federated_outlier_adaptation.runners.forgetting_signals import (  # noqa: E402
    SIGNAL_KEYS,
)

#: Where each signal can be read at deployment time. This is the column that
#: decides whether a signal is usable at all, and it is a property of the
#: measurement rather than of any run, so it is stated here.
OBSERVED_ON = {
    "dist_l2_to_global": "model only",
    "dist_fisher_to_global": "model only",
    "dist_fisher_norm_to_global": "model only",
    "retention_known": "client held-out",
    "agreement_with_global": "client held-out",
    "kl_global_to_current": "client held-out",
    "proxy_acc": "public proxy",
    "proxy_kl": "public proxy",
}

#: The correlation columns, in the order they are written. Kept in one place so
#: that a signal with no stopping row and a signal with one carry the same set.
CORR_COLUMNS = ("direction", "n_runs", "median_rho", "median_raw_rho",
                "median_abs_rho", "pooled_rho", "pooled_pearson",
                "share_ge_0p9")

#: The columns only a signal defined on every arm can fill.
STOP_COLUMNS = ("best_delta", "mean_stopped_score", "mean_stop_round",
                "arms_fired", "mean_vs_fixed")

COLUMNS = ("signal", "observed_on") + CORR_COLUMNS + ("defined_on_all_arms",) \
    + STOP_COLUMNS


def correlations(root: Path) -> Tuple[Dict[str, dict], int]:
    """Per-signal correlation with forgetting, from `signal_correlations.csv`."""
    per_run = defaultdict(list)
    pooled = {}
    pooled_pearson = {}
    runs = set()
    path = root / "signals" / "signal_correlations.csv"
    if not path.exists():
        raise SystemExit(
            f"FATAL: {path} is not on disk. The signal views are a read of what "
            "`foa signals` wrote; there is nothing here to summarise."
        )
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target"] != "source_val":
                continue
            try:
                rho = float(row["spearman"])
            except (TypeError, ValueError):
                continue
            if row["run"] == "__pooled__":
                pooled[row["signal"]] = rho
                pooled_pearson[row["signal"]] = float(row["pearson"])
                continue
            # The correlation file is a record of every run the signals module
            # saw, which includes an extreme arrangement the stage no longer
            # defines. Its folders cannot be un-run; its rows are not this
            # study's, and a median taken over them is a median of something
            # else.
            if not ST.in_study(row["run"].split("/", 1)[0]):
                continue
            runs.add(row["run"])
            per_run[row["signal"]].append(rho)

    out = {}
    for signal, values in per_run.items():
        sign = SIGNAL_DIRECTION.get(signal, 1)
        strong = sum(1 for v in values if abs(v) >= 0.9)
        out[signal] = {
            "direction": sign,
            "n_runs": len(values),
            "median_rho": sign * st.median(values),
            "median_raw_rho": st.median(values),
            "median_abs_rho": st.median([abs(v) for v in values]),
            "pooled_rho": sign * pooled[signal] if signal in pooled else None,
            "pooled_pearson": pooled_pearson.get(signal),
            "share_ge_0p9": strong / len(values) if values else None,
        }
    return out, len(runs)


def verdicts(root: Path):
    """Every stopping arm of every stage, judged by the whole-study rule."""
    refs = ST.stage_baselines(root)
    a0, p0 = refs["c10d10"]
    staged = []
    for stem, _title in ST.STAGES:
        # The stage's own cohort, not the study's: the same rule every table
        # this summarises is now read by.
        a0, p0 = refs[stem.strip("_")]
        prefix = f"d01_{stem}"
        found = []
        for (cell, family), stored in sorted(ST.read_arms(root, prefix).items()):
            arm = ST.arm_of(cell, family, stored)
            # AN ARM IS FIVE FOLDS AND MORE THAN ONE ROUND. Anything shorter has
            # no stop to take and would drag the means it is averaged into.
            if arm is None or arm.folds < 5 or arm.rounds < 2:
                continue
            found.append(ST.judge(arm, a0, p0, DEFAULT_DELTAS))
        staged.append((stem.strip("_"), found))
    return staged, a0, p0


def summary_rows(root: Path) -> Tuple[List[dict], list, float, float, int, int]:
    corr, n_runs = correlations(root)
    staged, a0, p0 = verdicts(root)
    every = [v for _stage, vs in staged for v in vs]
    n_arms = len(every)
    if not n_arms:
        raise SystemExit(
            f"FATAL: no stopping arm under {root}. The stopping half of this "
            "view is read from the stored runs, and none of them are here."
        )

    fixed_mean = st.mean(v.final["score"] for v in every)
    oracle_mean = st.mean(v.oracle["score"] for v in every)

    # per (signal, delta): the score at the stop, the stop index, and whether
    # the rule fired at all rather than running the schedule out.
    table = defaultdict(list)
    for v in every:
        last = len(v.values) - 1
        for (signal, delta), index in v.stops.items():
            table[(signal, delta)].append((v.values[index], index, index < last))

    rows = []
    for signal in SIGNAL_KEYS:
        corr_of = {k: corr.get(signal, {}).get(k, "") for k in CORR_COLUMNS}
        eligible = {
            delta: table[(signal, delta)]
            for delta in DEFAULT_DELTAS
            if len(table.get((signal, delta), ())) == n_arms
        }
        if not eligible:
            rows.append({"signal": signal,
                         "observed_on": OBSERVED_ON[signal],
                         "defined_on_all_arms": 0,
                         **{k: "" for k in STOP_COLUMNS},
                         **corr_of})
            continue
        # THE BEST DELTA IS THE ONE THAT SCORES HIGHEST, AND ON A TIE THE ONE
        # THAT GETS THERE SOONEST. Stopping earlier for the same score is the
        # whole point of the rule.
        best = max(
            eligible,
            key=lambda d: (st.mean(s for s, _i, _f in eligible[d]),
                           -st.mean(i for _s, i, _f in eligible[d])),
        )
        entries = eligible[best]
        mean_score = st.mean(s for s, _i, _f in entries)
        rows.append({
            "signal": signal,
            "observed_on": OBSERVED_ON[signal],
            "defined_on_all_arms": 1,
            "best_delta": best,
            "mean_stopped_score": mean_score,
            "mean_stop_round": st.mean(i + 1 for _s, i, _f in entries),
            "arms_fired": sum(1 for _s, _i, f in entries if f),
            "mean_vs_fixed": mean_score - fixed_mean,
            **corr_of,
        })
    return rows, staged, a0, p0, n_arms, n_runs


def extras_rows(root: Path, staged, a0: float, p0: float,
                n_arms: int, n_runs: int) -> List[tuple]:
    """The scalars the section quotes in prose, as key/value pairs."""
    every = [v for _stage, vs in staged for v in vs]
    # the run count the signals module itself reported for the study
    study_runs = ""
    with open(root / "signals" / "signal_selection.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            study_runs = int(row["num_candidates"])
            break

    extras = [
        ("study_runs", study_runs),
        ("signal_runs_with_source_val", n_runs),
        ("stopping_arms", n_arms),
        ("fixed_mean_score", st.mean(v.final["score"] for v in every)),
        ("oracle_mean_score", st.mean(v.oracle["score"] for v in every)),
        ("baseline_a0", a0),
        ("baseline_p0", p0),
    ]
    # RETENTION ON THE EXTREMES is the one signal the section singles out: how
    # far it leaves its ceiling while the source population is actually being
    # lost. Both halves are written per arrangement so the pair can be read
    # against each other rather than taken on the maximum alone.
    extreme = dict(staged)["extreme"]
    dev, fall = [], []
    for v in extreme:
        series = v.arm.signals.get("retention_known") or []
        if series:
            dev.append(max(abs(1.0 - value) for value in series))
        pres = v.arm.preservation
        fall.append(pres[0] - min(pres))
        extras.append((f"retention_dev_{v.arm.cell}",
                       max(abs(1.0 - x) for x in series) if series else ""))
        extras.append((f"source_fall_{v.arm.cell}", pres[0] - min(pres)))
    extras.append(("retention_max_dev_extreme", max(dev) if dev else ""))
    extras.append(("source_max_fall_extreme", max(fall) if fall else ""))
    return extras


def write_summary(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})
    print(f"  -> {path}  ({len(rows)} rows)")


def write_extras(path: Path, extras: List[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "value"])
        for key, value in extras:
            writer.writerow([key, value])
    print(f"  -> {path}  ({len(extras)} keys)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", "--study", required=True, type=Path, dest="root",
                    help="The study root ($FOA_STUDY_DIR).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Directory the two CSV views are written into.")
    args = ap.parse_args()

    rows, staged, a0, p0, n_arms, n_runs = summary_rows(args.root)
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print(f"arms: {n_arms}   runs carrying a source_val correlation: {n_runs}")
    write_summary(args.out / "signals_summary_extract.csv", rows)
    write_extras(args.out / "signals_extras_extract.csv",
                 extras_rows(args.root, staged, a0, p0, n_arms, n_runs))
    usable = sum(1 for row in rows if row["defined_on_all_arms"] == 1)
    print(f"signals defined on every arm: {usable} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
