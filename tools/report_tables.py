#!/usr/bin/env python3
"""
Every table this study reports, from the stored runs.

The numbers in the manuscript come from here and from nowhere else. That is the
point: a figure quoted from a script someone ran once at a terminal cannot be
checked, and cannot be regenerated when a stage is re-run. Nothing in this file
trains anything - it reads the result folders and the selection records that the
programme has already written.

    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what references
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what agg-winners
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what reg-winners
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what all --csv out/

THE RULE THE TABLES ARE ORDERED BY. A configuration is worth what it added on
the new clients less the source knowledge it spent to get it:

    score = (adaptation - A0) - (P0 - preservation)

A0 is the shipped model on the cohort's test rows, P0 the shipped model on the
source population's. Both are read from the selection stage's own evaluations,
never assumed, and a root without them is refused rather than defaulted.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

SIGNALS = ["dist_l2_to_global", "dist_fisher_to_global", "dist_fisher_norm_to_global",
           "agreement_with_global", "kl_global_to_current", "retention_known",
           "proxy_acc", "proxy_kl"]


# --------------------------------------------------------------- baselines
def baselines(root: Path) -> tuple:
    """The shipped model's own two accuracies: A0 on the cohort, P0 on the source."""
    def mean_accuracy(name):
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
            f"{root} has no shipped-model evaluations; every table is measured "
            "against them and they are not guessed."
        )
    return a0, p0


# ------------------------------------------------------------------ runs
def read_runs(root: Path, prefix: str) -> dict:
    """
    Final-round adaptation and preservation of every run under a prefix.

    A run folder holds BOTH schedules - the concurrent (parallel) family and the
    sequential (cyclic) one - so the schedule is part of the key. Reading only
    one of them, or averaging across them, silently answers a different
    question than the table asks.
    """
    out = defaultdict(lambda: {"a": [], "p": [], "signals": []})
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        folder = next(p for p in Path(path).parts if p.startswith(prefix))
        cell = folder[len(prefix):].split("_fold")[0]
        family = "parallel" if Path(path).parent.name == "concurrent_delta" else "cyclic"

        # A REGULARISATION RUN COMPUTES BOTH SCHEDULES, and the folder is named
        # for the one it was selected to serve. Reading the other half is
        # reading a result the selection never chose - the same penalty judged
        # under a schedule it was not picked for. The folder's own tag decides.
        for tag, schedule in (("concurrent_", "parallel"), ("sequential_", "cyclic")):
            if cell.startswith(tag):
                if family != schedule:
                    cell = None
                else:
                    cell = cell[len(tag):]
                break
        if cell is None:
            continue
        body = next(iter(json.loads(Path(path).read_text()).values()))
        adaptation = body.get("heldout_client_accuracies") or []
        preservation = body.get("pool_test_accuracies") or []
        if adaptation:
            out[(cell, family)]["a"].append(adaptation[-1])
        if preservation:
            out[(cell, family)]["p"].append(preservation[-1])
        out[(cell, family)]["signals"].append(
            sum(1 for k in SIGNALS if any(v is not None for v in (body.get(k) or [])))
        )
    return out


def summarise(runs: dict, a0: float, p0: float, folds: int = 5) -> list:
    rows = []
    for (cell, family), v in runs.items():
        if len(v["a"]) < folds:
            continue
        a, p = st.mean(v["a"]), st.mean(v["p"])
        rows.append({
            "cell": cell, "family": family,
            "adaptation": a, "adaptation_sd": st.pstdev(v["a"]),
            "preservation": p, "preservation_sd": st.pstdev(v["p"]),
            "gained": a - a0, "spent": p0 - p,
            "score": (a - a0) - (p0 - p),
            "folds": len(v["a"]), "signals": min(v["signals"]),
        })
    return sorted(rows, key=lambda r: -r["score"])


def show(rows: list, title: str, a0: float, p0: float) -> None:
    print(f"\n{title}")
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print(f"{'cell':<32}{'sched':<10}{'adapt':>8}{'+-':>7}{'preserve':>10}"
          f"{'gained':>9}{'spent':>8}{'score':>8}{'sig':>5}")
    print("-" * 100)
    for r in rows:
        print(f"{r['cell'][:30]:<32}{r['family']:<10}{r['adaptation']:>8.4f}"
              f"{r['adaptation_sd']:>7.4f}{r['preservation']:>10.4f}"
              f"{r['gained']*100:>8.2f}p{r['spent']*100:>7.2f}p"
              f"{r['score']*100:>7.2f}p{r['signals']:>4}/8")


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# ------------------------------------------------------------- references
def references(root: Path, a0: float, p0: float, tag: str) -> list:
    """The four rungs every federated number is read against."""
    rows = []

    def pooled(pattern, key):
        found = []
        for path in glob.glob(str(root / pattern)):
            d = json.loads(Path(path).read_text())
            block = d.get("pooled", {})
            value = block.get(key, d.get(key))
            if isinstance(value, dict):
                value = value.get("mean")
            if value is not None:
                found.append(value)
        return st.mean(found) if found else None

    for label, folder, init in (("isolated (from g-0)", f"isolated_{tag}", "global"),
                                ("isolated (scratch)", f"isolated_{tag}", "scratch"),
                                ("centralized (from g-0)", f"centralized_{tag}", "global"),
                                ("centralized (scratch)", f"centralized_{tag}", "scratch")):
        stem = "isolated" if folder.startswith("isolated") else "centralized"
        a = pooled(f"{folder}/{stem}_{init}_*.json", "own")
        p = pooled(f"{folder}/{stem}_{init}_*.json", "old")
        if a is None:
            continue
        rows.append({"cell": label, "family": "-", "adaptation": a, "adaptation_sd": 0.0,
                     "preservation": p if p is not None else float("nan"),
                     "preservation_sd": 0.0, "gained": a - a0,
                     "spent": p0 - p if p is not None else float("nan"),
                     "score": (a - a0) - (p0 - p) if p is not None else float("nan"),
                     "folds": 5, "signals": 0})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--what", default="all",
                    choices=("all", "references", "agg-screen", "agg-winners",
                             "reg-screen", "reg-winners"))
    ap.add_argument("--tag", default="c10", help="Reference cohort tag.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--csv", type=Path, default=None, help="Also write CSVs here.")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    t = args.study_tag
    wanted = {"all": ("references", "agg-screen", "agg-winners",
                      "reg-screen", "reg-winners")}.get(args.what, (args.what,))

    for what in wanted:
        if what == "references":
            rows = references(args.root, a0, p0, args.tag)
            title = "THE REFERENCE RUNGS"
        else:
            prefix, title = {
                "agg-screen":  (f"{t}_agg_",     "AGGREGATION SCREEN (short horizon: a ranking, not a result)"),
                "agg-winners": (f"{t}_aggfull_", "AGGREGATION WINNERS (full horizon)"),
                "reg-screen":  (f"{t}_reg_",     "REGULARISATION SCREEN (short horizon)"),
                "reg-winners": (f"{t}_regfull_", "REGULARISATION WINNERS (full horizon)"),
            }[what]
            rows = summarise(read_runs(args.root, prefix), a0, p0)
        if not rows:
            print(f"\n{title}\n  nothing on disk yet")
            continue
        show(rows, title, a0, p0)
        if args.csv:
            write_csv(rows, args.csv / f"{what}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
