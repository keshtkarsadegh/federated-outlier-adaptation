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

IS EACH REFERENCE STAGE ON ITS OWN BASE?
----------------------------------------
The third way the same class of failure can arrive. Every other stage derives
its seeds from a block written down in the emitter, so a stage cannot be run on
the wrong one by accident. The reference stages take ``--seed-base`` on the
command line instead, because one generator serves every federation size - and a
base typed differently emits a file that looks right, runs, and draws a
different set of clients than the shipped records were drawn with. Nothing
downstream says so. So the bases the shipped files were emitted with are held
here, beside the files, and a reference file whose seeds leave its own range is
a failure rather than a note.

Exits non-zero if any training task is unseeded, if two files share a seed, or
if a reference stage's seeds sit outside its base.
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

#: The seed base each reference stage was emitted with, keyed on the file that
#: carries it. These are read out of the shipped files rather than chosen here,
#: and `docs/REPRODUCE.md` section 5 lists the same four. The extreme point is
#: on the list with nothing to check: `--only do-nothing` emits five forward
#: passes, which seed nothing, and leaving it off would read as an oversight.
REFERENCE_BASES = {
    "d01_c20_references.txt": 720000,
    "d01_extreme_references.txt": 730000,
    "s03_refs_c10.txt": 740000,
    "d01_c5_references.txt": 750000,
}

#: A stage's slot is the 10000 between its base and the next one - wide enough
#: for the widest stage (twenty client positions at 200 apart is 3800) and
#: narrow enough that the four bases do not overlap. One flat span from a base
#: would not separate them: 720000 plus anything that reaches the pooled arm
#: swallows the other three bases whole, and a five-client file emitted on the
#: twenty-client base would read as correct.
REFERENCE_SLOT = 10000

#: The pooled arm sits this far above the base, deliberately clear of every
#: client's seed, so it gets a second slot of the same width rather than a span
#: stretched to cover the gap between them.
POOLED_OFFSET = 90000


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


def out_of_base(path: Path) -> list[str]:
    """
    Seeds of a reference file that do not sit in that stage's own base.

    Read per command, because the two arms of a reference stage carry their
    drawn seed under different flags: `isolated-train` puts it on ``--seed``,
    while `final` seeds the run with the fold and separates its arms with
    ``--sampler-seed``. Reading ``--seed`` off a `final` line would flag every
    one of them, since a fold is 1 to 5 and no base is.
    """
    base = REFERENCE_BASES.get(path.name)
    if base is None:
        return []

    problems: list[str] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        words = line.strip().split()
        if not words or words[0] != "foa":
            continue
        name = {"isolated-train": "--seed", "final": "--sampler-seed"}.get(words[1])
        if name is None:
            continue
        value = flag(words, name)
        if value is None:
            continue            # the unseeded case is check()'s to report
        try:
            seed = int(value)
        except ValueError:
            continue
        pooled = base + POOLED_OFFSET
        if not (base <= seed < base + REFERENCE_SLOT
                or pooled <= seed < pooled + REFERENCE_SLOT):
            problems.append(
                f"{path.name}:{n}  {name} {seed} is outside the {base} base "
                f"this stage was emitted with, and outside its pooled arm "
                f"at {pooled}"
            )
    return problems


def sampler_seeds(path: Path) -> dict[int, str]:
    """
    ``{sampler seed: the output the line writes}`` for one task file.

    The seed alone is not enough to judge a clash. Two files may legitimately
    carry the SAME task - a smoke file that re-runs four lines of a screen, a
    subset emitted to re-run the winners that moved - and those share a seed
    because they are the same run, not because two runs collided.
    """
    seeds: dict[int, str] = {}
    for line in path.read_text().splitlines():
        words = line.split()
        if not words or words[0] != "foa":
            continue
        value = flag(words, "--sampler-seed")
        if value is None:
            continue
        try:
            seed = int(value)
        except ValueError:
            continue
        seeds[seed] = flag(words, "--parent") or ""
    return seeds


def collisions(spans: dict[str, dict[int, str]]) -> list[str]:
    """
    Every pair of files where one seed drives two DIFFERENT tasks.

    Reported as a problem rather than a note: two stages sharing a sampling
    sequence are not independent, and the correlation is invisible downstream -
    both run, both write results, and nothing in the output says the two were
    drawn together.

    A seed shared by the same ``--parent`` is the same task in two files and is
    not a collision. That distinction matters: without it every smoke file and
    every re-run subset reads as a clash, the check cries wolf, and the next
    real collision is waved through by hand.
    """
    found: list[str] = []
    names = list(spans)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            clashing = [s for s in set(spans[a]) & set(spans[b])
                        if spans[a][s] != spans[b][s]]
            if clashing:
                found.append(
                    f"{a} and {b}: {len(clashing)} seed(s) drive DIFFERENT tasks, "
                    f"{min(clashing)}-{max(clashing)}"
                )
    return found


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    all_problems: list[str] = []
    spans: dict[str, dict[int, str]] = {}
    for arg in sys.argv[1:]:
        path = Path(arg)
        n = sum(1 for l in path.read_text().splitlines() if l.strip().startswith("foa "))
        problems = check(path) + out_of_base(path)
        seeds = sampler_seeds(path)
        spans[path.name] = seeds
        status = "OK " if not problems else "GAP"
        span = f"{min(seeds)}-{max(seeds)}" if seeds else "no sampler seeds"
        base = REFERENCE_BASES.get(path.name)
        tail = f"   base {base}" if base is not None else ""
        print(f"  {status} {path.name:<28} {n:4d} tasks   {span}{tail}")
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
          "no two files share one, and every reference stage is on its own base.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
