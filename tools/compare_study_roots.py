#!/usr/bin/env python3
"""
Did the study rebuild itself?

The published tables rest on a cohort chosen by a chain of trained models: a
detector ranks the writers, a cut splits them, a shipped model is trained on one
side and scores the other, and the worst of those become the clients. Every link
is a GPU computation, and GPU arithmetic is not bit-deterministic, so the
question is not whether the files are identical - it is whether the *decisions*
are.

This compares the two roots decision by decision, from the census forward, and
says where they agree and where they diverge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])


def load(root, rel):
    path = root / rel
    return json.loads(path.read_text()) if path.is_file() else None


def clients_of(payload):
    if payload is None:
        return None
    if isinstance(payload, dict):
        return list(payload.get("clients", payload.get("pool", [])))
    return list(payload)


def verdict(ok):
    return "SAME " if ok else "DIFFER"


print(f"old: {OLD}\nnew: {NEW}\n")
rows = []

# --------------------------------------------------------------- the census
a, b = load(OLD, "outliers/writer_counts.json"), load(NEW, "outliers/writer_counts.json")
if a and b:
    same = (a["writers"], a["rows"], a["num_classes"]) == (b["writers"], b["rows"], b["num_classes"])
    rows.append(("writer census", verdict(same),
                 f"{a['writers']} writers / {a['rows']} rows  vs  {b['writers']} / {b['rows']}"))

# ------------------------------------------------------------- the detector
a, b = load(OLD, "ginit_selection.json"), load(NEW, "ginit_selection.json")
if a and b:
    rows.append(("g-init crowned fold", verdict(a["selected_fold"] == b["selected_fold"]),
                 f"fold {a['selected_fold']} (test {a['test_mean']:.4f})  vs  "
                 f"fold {b['selected_fold']} (test {b['test_mean']:.4f})"))

# ------------------------------------------------------------------- pools
a, b = load(OLD, "outliers/pools.json"), load(NEW, "outliers/pools.json")
if a and b:
    rows.append(("bad/good cut rank", verdict(a["cut_rank"] == b["cut_rank"]),
                 f"{a['cut_rank']}  vs  {b['cut_rank']}"))
for name in ("pool_bad", "pool_good"):
    x, y = clients_of(load(OLD, f"outliers/{name}.json")), clients_of(load(NEW, f"outliers/{name}.json"))
    if x and y:
        overlap = len(set(x) & set(y))
        rows.append((f"{name} membership", verdict(set(x) == set(y)),
                     f"{len(x)} vs {len(y)}, overlap {overlap} "
                     f"({100 * overlap / max(len(x), 1):.1f}%)"))

# ------------------------------------------------------- the old population
x, y = clients_of(load(OLD, "outliers/old_data.json")), clients_of(load(NEW, "outliers/old_data.json"))
if x and y:
    overlap = len(set(x) & set(y))
    rows.append(("old population (200)", verdict(set(x) == set(y)),
                 f"overlap {overlap}/{len(x)}"))

# ------------------------------------------------------- the shipped model
a, b = load(OLD, "g0_selection.json"), load(NEW, "g0_selection.json")
if a and b:
    rows.append(("g-0 crowned fold", verdict(a["selected_fold"] == b["selected_fold"]),
                 f"fold {a['selected_fold']} (val {a['val_mean']:.4f} test {a['test_mean']:.4f})  vs  "
                 f"fold {b['selected_fold']} (val {b['val_mean']:.4f} test {b['test_mean']:.4f})"))

# --------------------------------------------------------------- the cohorts
for k in (5, 10, 20):
    x = clients_of(load(OLD, f"outliers/cohort_worst{k}.json"))
    y = clients_of(load(NEW, f"outliers/cohort_worst{k}.json"))
    if x and y:
        overlap = len(set(x) & set(y))
        rows.append((f"cohort worst-{k}", verdict(set(x) == set(y)),
                     f"overlap {overlap}/{k}" +
                     ("" if set(x) == set(y) else
                      f"  only-old {sorted(set(x) - set(y))}  only-new {sorted(set(y) - set(x))}")))

width = max(len(r[0]) for r in rows)
for name, status, detail in rows:
    print(f"  {status}  {name:{width}s}  {detail}")

# ----------------------------------------- the ranking that picks the cohort
a, b = load(OLD, "outliers/bad_acc_on_g0.json"), load(NEW, "outliers/bad_acc_on_g0.json")
def flat(p):
    if isinstance(p, list):
        out = {}
        for row in p:
            out.update(row)
        return out
    return p or {}
a, b = flat(a), flat(b)
if a and b:
    shared = sorted(set(a) & set(b))
    diffs = [abs(a[w] - b[w]) for w in shared]
    ranked_old = sorted(shared, key=lambda w: a[w])[:20]
    ranked_new = sorted(shared, key=lambda w: b[w])[:20]
    print(f"\n  bad-pool scores under g-0: {len(shared)} writers in common")
    print(f"    max |old-new| = {max(diffs):.4f}   mean = {sum(diffs)/len(diffs):.5f}")
    print(f"    worst-20 by that ranking agree on {len(set(ranked_old) & set(ranked_new))}/20")
