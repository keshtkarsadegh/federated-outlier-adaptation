#!/usr/bin/env python3
"""
Did every run actually record all eight forgetting signals?

The study forbids a running experiment from reading the source population: once
the model is shipped, those writers are gone.  Every number reported on the
source split is an *evaluation* quantity, and a stopping or selection rule that
used it would be reading data it is not allowed to see.  The eight signals in
``runners/forgetting_signals`` are what a server MAY compute, and the point of
recording them is to study stopping and selection under the real constraint.

WHY THIS EXISTS.  Four of the eight were silently empty across an 845-task
stage.  The MNIST proxy set had never been built, so ``proxy_acc`` and
``proxy_kl`` were ``None`` on every round; the winning fold's Fisher was never
promoted to the name the artefact resolver looks for, so the two Fisher
distances were empty too.  Nothing failed.  Every task exited zero, every result
file was written, and half the signal study was missing.

A signal that is absent looks exactly like a signal that is present until
somebody counts, so this counts.

    python tools/check_signals.py --root "$FOA_STUDY_DIR" --prefix d01_reg_
    python tools/check_signals.py --root "$FOA_STUDY_DIR" --all
    python tools/check_signals.py --root "$FOA_STUDY_DIR" --all --expect-rounds 25

Exits non-zero if any signal is missing from any payload, or if a series is
shorter than the rounds the run was supposed to have.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from federated_outlier_adaptation.runners.forgetting_signals import SIGNAL_KEYS  # noqa: E402

#: Prefixes checked by ``--all``, with the horizon each was run at.  A screen
#: and a final differ in length, and a short series is as wrong as an empty one.
#:
#: EVERY STAGE THAT WROTE SIGNALS IS NAMED HERE, not only the stages a grid was
#: searched over.  The check only reports a series SHORTER than the horizon it
#: was given, so a stage missing from this map is not audited loosely - it is
#: not audited at all, and ``--all`` still ends on the sentence that says every
#: payload carries all eight.  The carry settings, the three extreme
#: arrangements and the federated reference cells ran for months before anything
#: counted their signals, because none of them had a line in this dictionary.
DEFAULT_PREFIXES = {
    "d01_agg_": 25,
    "d01_aggfull_": 100,
    "d01_reg_": 25,
    "d01_regfull_": 100,
    "d01_combo_": 100,
    # the carry settings: the selected arms moved onto other federations
    "d01_five_": 100,
    "d01_c10d10_": 100,
    "d01_c20d10_": 100,
    "d01_c20d20_": 100,
    # the three extreme arrangements
    "d01_extreme_": 100,
    # the federated reference cells each cohort is read against
    "d01_c5_m4_control_": 100,
    "d01_c10_m8_control_": 100,
    "d01_c10_m9_control_": 100,
    "d01_c20_m16_control_": 100,
    "d01_c20_m18_control_": 100,
}


def payloads(root: Path, prefix: str):
    """Every ``accuracies_*.json`` under a prefix, with the folder it came from."""
    for run_dir in sorted(root.glob(prefix + "*")):
        for path in sorted(run_dir.rglob("accuracies_*.json")):
            try:
                yield run_dir.name, path, json.loads(path.read_text())
            except (OSError, ValueError):
                yield run_dir.name, path, None


def audit(root: Path, prefix: str, expect_rounds: Optional[int]) -> dict:
    """How many payloads carry each signal, and how long the series are."""
    filled = Counter()
    lengths: Dict[str, Counter] = {key: Counter() for key in SIGNAL_KEYS}
    total = 0
    unreadable: List[str] = []
    for name, path, data in payloads(root, prefix):
        if data is None:
            unreadable.append(str(path))
            continue
        total += 1
        for key in SIGNAL_KEYS:
            series = [v for v in (data.get(key) or []) if v is not None]
            if series:
                filled[key] += 1
                lengths[key][len(series)] += 1
    problems: List[str] = []
    for key in SIGNAL_KEYS:
        if filled[key] < total:
            problems.append(
                f"{prefix}: {key} is empty in {total - filled[key]} of {total} payloads"
            )
        elif expect_rounds is not None:
            short = {n: c for n, c in lengths[key].items() if n < expect_rounds}
            if short:
                problems.append(
                    f"{prefix}: {key} has {sum(short.values())} payload(s) shorter "
                    f"than {expect_rounds} rounds ({sorted(short)})"
                )
    problems += [f"{prefix}: unreadable payload {p}" for p in unreadable]
    return {"prefix": prefix, "total": total, "filled": filled,
            "lengths": lengths, "problems": problems}


def show(result: dict, expect_rounds: Optional[int]) -> None:
    total = result["total"]
    print(f"\n{result['prefix']}  {total} payload(s)"
          + (f", expecting {expect_rounds} rounds" if expect_rounds else ""))
    if not total:
        print("   no payloads under this prefix")
        return
    for key in SIGNAL_KEYS:
        n = result["filled"][key]
        seen = result["lengths"][key]
        span = (f"len {min(seen)}" + (f"-{max(seen)}" if max(seen) != min(seen) else "")
                if seen else "-")
        print(f"   {key:<30}{n:>5}/{total}  {span:<12}"
              f"{'OK' if n == total else 'MISSING'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prefix", action="append", default=None,
                        help="Run-folder prefix to audit; repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="Audit every prefix this study writes.")
    parser.add_argument("--expect-rounds", type=int, default=None,
                        help="Override the horizon a series is checked against.")
    args = parser.parse_args()

    if args.all:
        wanted = dict(DEFAULT_PREFIXES)
    elif args.prefix:
        wanted = {p: args.expect_rounds for p in args.prefix}
    else:
        raise SystemExit("give --prefix or --all")
    if args.expect_rounds is not None:
        wanted = {p: args.expect_rounds for p in wanted}

    print(f"eight signals, {len(SIGNAL_KEYS)} keys: {', '.join(SIGNAL_KEYS)}")
    problems: List[str] = []
    for prefix, rounds in wanted.items():
        result = audit(args.root, prefix, rounds)
        show(result, rounds)
        problems += result["problems"]

    if problems:
        print()
        for p in problems:
            print(f"   {p}")
        print(f"\n{len(problems)} problem(s). A missing signal looks like a present "
              "one until somebody counts.")
        return 1
    print("\nevery payload carries all eight signals, at the expected length.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
