#!/usr/bin/env python3
"""
Pin the decisions the study rests on, so a later run can be checked against them.

Every table in the paper is about ten particular writers. Those writers are not
named anywhere by hand: they are the output of a chain of trained models, and
the chain is re-run whenever the study is rebuilt. Averaging the detector folds
and keying the source draw on writer identity removed the two places where that
chain could swing on floating-point noise, but a fix nobody can check is a
promise rather than a guarantee.

This writes one small file that names the decisions and the exact bytes behind
them, and can later re-read it and say whether a fresh root made the same
decisions. It deliberately records DECISIONS, not the contents of model
checkpoints: GPU arithmetic is not bit-deterministic, so two honest runs will
differ in their weights and agree in their conclusions, and it is the
conclusions the paper quotes.

    freeze  writes selection.lock.json under the study root
    verify  compares a study root against a lock and exits non-zero on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: The files that together determine which writers the study trains on.
#: Each is (relative path, is it required).
TRACKED = [
    ("outliers/writer_counts.json", True),
    ("outliers/writer_scores.json", True),
    ("outliers/writer_scores_fold1.json", False),
    ("outliers/writer_scores_fold2.json", False),
    ("outliers/writer_scores_fold3.json", False),
    ("outliers/writer_scores_fold4.json", False),
    ("outliers/writer_scores_fold5.json", False),
    ("outliers/clients_acc_on_global.json", True),
    ("outliers/pools.json", True),
    ("outliers/pool_bad.json", True),
    ("outliers/pool_good.json", True),
    ("outliers/old_data.json", True),
    ("outliers/bad_acc_on_g0.json", True),
    ("outliers/cohort_worst5.json", False),
    ("outliers/cohort_worst10.json", True),
    ("outliers/cohort_worst20.json", False),
    ("ginit_selection.json", True),
    ("g0_selection.json", True),
]

#: The decisions themselves, pulled out of those files so that a drift report
#: can say *what* changed rather than only that a checksum moved.
def decisions(root: Path) -> dict:
    def load(rel):
        path = root / rel
        return json.loads(path.read_text()) if path.is_file() else None

    def members(payload):
        if payload is None:
            return None
        if isinstance(payload, dict):
            return list(payload.get("clients", payload.get("pool", [])))
        return list(payload)

    out = {}
    counts = load("outliers/writer_counts.json")
    if counts:
        out["writers"] = counts.get("writers")
        out["rows"] = counts.get("rows")
        out["classes"] = counts.get("num_classes")

    for name, rel in (("ginit_fold", "ginit_selection.json"),
                      ("g0_fold", "g0_selection.json")):
        sel = load(rel)
        if sel:
            out[name] = sel.get("selected_fold")

    pools = load("outliers/pools.json")
    if pools:
        out["cut_rank"] = pools.get("cut_rank")

    for name, rel in (("pool_bad", "outliers/pool_bad.json"),
                      ("pool_good", "outliers/pool_good.json"),
                      ("old_data", "outliers/old_data.json"),
                      ("cohort_worst5", "outliers/cohort_worst5.json"),
                      ("cohort_worst10", "outliers/cohort_worst10.json"),
                      ("cohort_worst20", "outliers/cohort_worst20.json")):
        who = members(load(rel))
        if who is not None:
            out[name] = sorted(who)
    return out


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(root: Path) -> dict:
    files, missing = {}, []
    for rel, required in TRACKED:
        path = root / rel
        if path.is_file():
            files[rel] = {"sha256": digest(path), "bytes": path.stat().st_size}
        elif required:
            missing.append(rel)
    if missing:
        raise SystemExit(f"cannot freeze: the study root is missing {missing}")
    return {
        "what": ("the decisions that choose the study's clients: the writer "
                 "census, the crowned folds, the pool cut, the source "
                 "population and the cohorts"),
        "why": ("GPU arithmetic is not bit-deterministic, so two honest runs "
                "differ in their weights. What must not differ is which "
                "writers the paper is about."),
        "files": files,
        "decisions": decisions(root),
    }


def compare(lock: dict, root: Path) -> int:
    fresh = decisions(root)
    old = lock["decisions"]
    problems = 0

    keys = sorted(set(old) | set(fresh))
    width = max(len(k) for k in keys)
    for key in keys:
        a, b = old.get(key), fresh.get(key)
        if a == b:
            print(f"  SAME    {key:{width}s}")
            continue
        if isinstance(a, list) and isinstance(b, list):
            overlap = len(set(a) & set(b))
            print(f"  DIFFER  {key:{width}s}  {len(a)} vs {len(b)}, "
                  f"overlap {overlap}")
            only_old, only_new = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            if 0 < len(only_old) <= 12:
                print(f"          gone:  {only_old}")
            if 0 < len(only_new) <= 12:
                print(f"          new:   {only_new}")
        else:
            print(f"  DIFFER  {key:{width}s}  {a!r} vs {b!r}")
        problems += 1

    # A crowned fold that moves is not by itself a failure: the point of
    # averaging the detector was to stop the crown from mattering. Say so.
    benign = {"ginit_fold", "g0_fold"}
    hard = [k for k in keys if old.get(k) != fresh.get(k) and k not in benign]
    if problems and not hard:
        print("\nonly the crowned folds moved. The cohorts are identical, "
              "which is what the averaging was for.")
        return 0
    return 1 if hard else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("freeze", "verify"))
    ap.add_argument("root", type=Path)
    ap.add_argument("--lock", type=Path, default=None,
                    help="Lock file; defaults to <root>/selection.lock.json")
    args = ap.parse_args()

    lock_path = args.lock or (args.root / "selection.lock.json")

    if args.mode == "freeze":
        payload = build(args.root)
        lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        d = payload["decisions"]
        print(f"froze {len(payload['files'])} files -> {lock_path}")
        for key in ("writers", "cut_rank", "ginit_fold", "g0_fold"):
            if key in d:
                print(f"  {key:14s} {d[key]}")
        for key in ("old_data", "cohort_worst5", "cohort_worst10", "cohort_worst20"):
            if key in d:
                print(f"  {key:14s} {len(d[key])} writers")
        return 0

    if not lock_path.is_file():
        raise SystemExit(f"no lock at {lock_path}")
    lock = json.loads(lock_path.read_text())
    print(f"lock: {lock_path}\nroot: {args.root}\n")
    status = compare(lock, args.root)
    print("\n" + ("the study reproduced its own selection."
                  if status == 0 else
                  "THE SELECTION DID NOT REPRODUCE."))
    return status


if __name__ == "__main__":
    sys.exit(main())
