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

DO TWO STAGES DRAW THE SAME SEEDS?
----------------------------------
A second failure, found the same way the first was - by looking rather than by
anything breaking. Stages seed as ``seed_base + block + index * 10 + fold`` with
the blocks 1000 apart, which holds until a stage has more than a hundred cells.
The regularisation screen is 140, so it spans 1400 and runs through two blocks;
the combination stage's first emission drew seeds already used by it.

Nothing about that is loud. Both stages run, both write results, and the two
draw the same client-sampling sequence - so their numbers are correlated, and
no output says so. Passing every task file at once cross-checks the spans:

    python tools/check_seeds.py <task file> [<task file> ...]

Exits non-zero if any training task is unseeded, or if two files share a seed.
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


def sampler_seeds(path: Path) -> set[int]:
    """Every ``--sampler-seed`` a task file draws."""
    seeds: set[int] = set()
    for line in path.read_text().splitlines():
        words = line.split()
        if not words or words[0] != "foa":
            continue
        value = flag(words, "--sampler-seed")
        if value is not None:
            try:
                seeds.add(int(value))
            except ValueError:
                continue
    return seeds


def collisions(spans: dict[str, set[int]]) -> list[str]:
    """
    Every pair of task files that would draw the same sampler seed.

    Reported as a problem rather than a note: two stages sharing a sampling
    sequence are not independent, and the correlation is invisible downstream.
    """
    found: list[str] = []
    names = list(spans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = spans[a] & spans[b]
            if shared:
                found.append(
                    f"{a} and {b} share {len(shared)} sampler seed(s), "
                    f"{min(shared)}-{max(shared)}"
                )
    return found


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    all_problems: list[str] = []
    spans: dict[str, set[int]] = {}
    for arg in sys.argv[1:]:
        path = Path(arg)
        n = sum(1 for l in path.read_text().splitlines() if l.strip().startswith("foa "))
        problems = check(path)
        seeds = sampler_seeds(path)
        spans[path.name] = seeds
        status = "OK " if not problems else "GAP"
        span = f"{min(seeds)}-{max(seeds)}" if seeds else "no sampler seeds"
        print(f"  {status} {path.name:<28} {n:4d} tasks   {span}")
        all_problems += problems

    shared = collisions(spans)
    if all_problems or shared:
        print()
        for p in all_problems:
            print(f"    {p}")
        for c in shared:
            print(f"    COLLISION: {c}")
        if all_problems:
            print(f"\n{len(all_problems)} task(s) that fit a model without a seed.")
        if shared:
            print(f"{len(shared)} pair(s) of task files drawing the same seeds.")
        return 1
    print("\nevery task that trains or draws carries a seed, "
          "and no two files share one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
