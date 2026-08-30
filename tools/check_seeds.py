#!/usr/bin/env python3
"""
Does every task that trains carry a seed?

This exists because one arm did not. `global-train` seeded, the federated
runner seeded, and `isolated-train` -- 110 of the 135 tasks in a reference
stage -- did not, so the private and centralized baselines drew a fresh
initialisation and a fresh shuffle on every run. Nothing failed and nothing
looked wrong: the arms being compared were reproducible, and the two rungs they
were compared *against* were not.

A command that only reads a model is exempt, because a forward pass has nothing
to seed. Everything that fits a model is not.

    python tools/check_seeds.py <task file> [<task file> ...]

Exits non-zero if any training task is unseeded.
"""

from __future__ import annotations

import shlex
import sys
from collections import defaultdict
from pathlib import Path

#: Commands that fit a model, and therefore need a run seed.
TRAINS = {
    "global-train", "global-train-fl", "isolated-train", "combined-train",
    "base-fl", "all-aggs", "grid", "final", "extreme", "local-finetune",
}

#: Commands that only read a model or a file. A forward pass has no seed.
READS_ONLY = {
    "evaluate-book", "score-writers", "score-pool", "average-scores",
    "writer-counts", "split-pools", "select-outliers", "select-fold",
    "cohort-table", "outlier-figure", "figures", "report", "signals",
    "promote-model", "check-population", "select-eligible", "prepare-data",
}

#: These draw, so they need a seed even though they never train.
DRAWS = {"draw-old-data", "draw-cohort", "fold-book"}


def flag(words: list[str], name: str) -> str | None:
    return words[words.index(name) + 1] if name in words else None


def check(path: Path) -> list[str]:
    problems: list[str] = []
    seeds: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line.startswith("foa "):
            continue
        words = shlex.split(line)
        cmd = words[1]
        seed = flag(words, "--seed")

        if cmd in TRAINS or cmd in DRAWS:
            if seed is None:
                problems.append(f"{path.name}:{n}  {cmd} has no --seed")
            else:
                seeds[cmd].append((n, seed))
        elif cmd not in READS_ONLY:
            problems.append(f"{path.name}:{n}  {cmd} is not classified - "
                            f"add it to TRAINS, DRAWS or READS_ONLY")

    # A seed reused inside one command is worth a look, but it is not always
    # wrong: `final` deliberately uses --seed = fold and separates its arms with
    # --sampler-seed. So this reports, and does not fail.
    for cmd, entries in sorted(seeds.items()):
        seen: dict[str, int] = {}
        dupes = 0
        for n, s in entries:
            key = s if cmd != "final" else f"{s}|{n}"
            if key in seen:
                dupes += 1
            seen[key] = n
        # fold-book SHOULD share a seed: a writer's split is keyed on
        # sha256(seed | fold | writer), so giving the 5-, 10- and 20-client
        # books one seed is exactly what makes a writer hold the same rows in
        # all three, and the sizes comparable.
        if dupes and cmd not in ("final", "fold-book"):
            print(f"  note: {cmd} reuses {dupes} seed(s) across {len(entries)} lines")

    return problems


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    all_problems: list[str] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        n = sum(1 for l in path.read_text().splitlines() if l.strip().startswith("foa "))
        problems = check(path)
        status = "OK " if not problems else "GAP"
        print(f"  {status} {path.name:<28} {n:4d} tasks")
        all_problems += problems

    if all_problems:
        print()
        for p in all_problems:
            print(f"    {p}")
        print(f"\n{len(all_problems)} task(s) that fit a model without a seed.")
        return 1
    print("\nevery task that trains or draws carries a seed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
