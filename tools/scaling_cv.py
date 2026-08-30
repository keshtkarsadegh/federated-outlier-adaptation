#!/usr/bin/env python3
"""
The scaling and dropout cells as five-fold means, and what that changes.

The scaling table reports these stages at fold 1, under the rule that they are
robustness probes of fixed winners rather than selections. The runs for the
other four folds exist for the five-client and twenty-percent-dropout stages, so
those cells can be five-fold means at no cost; this prints both so the size of
the correction is visible rather than assumed.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STUDY = Path(sys.argv[1])
STAGES = ["five", "drop20", "c20d10", "c20d20"]
CONFIGS = ["winner", "balanced", "sequential", "control"]


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def read(path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    for body in payload.values():
        if isinstance(body, dict) and "final_evaluation" in body:
            fin = body["final_evaluation"]
            adapt = fin.get("clients", {}).get("accuracy")
            olds = fin.get("old", {}).get("accuracies", {})
            return adapt, (mean(list(olds.values())) if olds else None)
    return None


rows = defaultdict(dict)
for stage in STAGES:
    for config in CONFIGS:
        per_fold = {}
        for fold in (1, 2, 3, 4, 5):
            hits = sorted(STUDY.glob(f"d01_{stage}_{config}_fold{fold}_*/**/summary_0.json"))
            for path in hits:
                got = read(path)
                if got and got[0] is not None:
                    per_fold[fold] = got
                    break
        if per_fold:
            rows[stage][config] = per_fold

print(f"{'stage':8s} {'config':11s} {'folds':>6s} "
      f"{'fold-1 adapt':>13s} {'CV adapt':>16s} {'fold-1 pres':>12s} {'CV pres':>16s} {'shift':>8s}")
print("-" * 96)
out = {}
for stage in STAGES:
    for config in CONFIGS:
        per_fold = rows[stage].get(config)
        if not per_fold:
            continue
        folds = sorted(per_fold)
        a = [per_fold[f][0] for f in folds]
        p = [per_fold[f][1] for f in folds]
        a1 = per_fold.get(1, (None, None))[0]
        p1 = per_fold.get(1, (None, None))[1]
        am, asd = mean(a), (statistics.stdev(a) if len(a) > 1 else 0.0)
        pm, psd = mean(p), (statistics.stdev(p) if len(p) > 1 else 0.0)
        shift = (am - a1) * 100 if (a1 is not None and am is not None) else None
        print(f"{stage:8s} {config:11s} {len(folds):>6d} "
              f"{(f'{a1:.4f}' if a1 else '-'):>13s} "
              f"{(f'{am:.4f} ±{asd:.4f}' if am else '-'):>16s} "
              f"{(f'{p1:.4f}' if p1 else '-'):>12s} "
              f"{(f'{pm:.4f} ±{psd:.4f}' if pm else '-'):>16s} "
              f"{(f'{shift:+.2f}' if shift is not None else '-'):>8s}")
        out[f"{stage}_{config}"] = {
            "folds": folds, "fold1_adapt": a1, "cv_adapt": am, "cv_adapt_sd": asd,
            "fold1_pres": p1, "cv_pres": pm, "cv_pres_sd": psd,
            "adapt_shift_points": shift,
        }

(STUDY / "tables" / "scaling_cv.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"\nwritten: {STUDY / 'tables' / 'scaling_cv.json'}")

moved = [(k, v["adapt_shift_points"]) for k, v in out.items()
         if v["adapt_shift_points"] is not None and abs(v["adapt_shift_points"]) >= 1.0
         and len(v["folds"]) > 1]
if moved:
    print("\ncells where the five-fold mean differs from the fold-1 probe by >= 1 point:")
    for k, s in sorted(moved, key=lambda kv: -abs(kv[1])):
        print(f"  {k:24s} {s:+.2f} pt")
