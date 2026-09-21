#!/usr/bin/env python3
"""
Does the programme build its own inputs?

A study that starts from a fresh results root has nothing to fall back on: every
file a stage reads must be written by an earlier stage, or the stage stops. The
old tree hid three such holes because the files happened to be there, made by
hand months ago - cohort_worst5.json among them.

This walks the stages in order, collects what each one writes, and reports any
path a stage reads that nothing before it produced. Paths outside the study
root (the packed cache, the archive) are inputs of the programme and are
reported separately rather than as errors.

WHICH FILES ARE WALKED, AND WHY IT IS DECLARED. Stage order here is file order,
so the walk cannot simply glob the jobs directory: the carry settings are named
``d01_*`` and sort before every ``s*`` stage they depend on, which would report
their inputs as unmet. So the ``s*`` stages are walked in name order - that
prefix was assigned in programme order - and the carry settings follow in the
order they ran, named in CARRY.

Anything else in the directory is listed as NOT WALKED with its line count. A
checker that silently examines a subset of a programme is worse than one that
examines none, because its clean verdict is read as covering everything: seven
live stages sat outside the ``s*`` glob and nothing said so.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from federated_outlier_adaptation.training.extreme_cells import (  # noqa: E402
    CASES as EXTREME_CASES,
)

JOBS = Path(sys.argv[1])
EXTERNAL_OK = ("$FOA_NIST28_DIR", "$FOA_DATA_DIR", "$FOA_CACHE_DIR")

READS = ["--scores", "--clients-file", "--outliers-file", "--fold-book",
         "--old-book", "--old-clients-file", "--model-path", "--init-checkpoint",
         "--counts", "--from-pool", "--exclude-file", "--subset-of",
         "--disjoint-from", "--zip"]
WRITES = ["--out", "--csv"]

#: Side effects: commands that write a file the flags do not name.
SIDE_EFFECTS = {
    "global-train": ["global_model", "global_results/fisher"],
    "select-fold": ["g0_model", "g0_selection.json", "ginit_model"],
    "fold-book": [],          # --out names it, ".foldbook.npz" appended
    "score-writers": ["outliers/clients_acc_on_global.json"],
    "score-pool": ["outliers/bad_acc_on_g0.json"],
    "split-pools": ["outliers/pool_bad.json", "outliers/pool_good.json"],
    "select-outliers": [],    # --out names it
    "writer-counts": [],
}


#: Producers that are not `foa` commands, and what they write.
SCRIPT_EFFECTS = {
    "make_extreme_cohorts.py": [
        f"outliers/extreme_{case}.json" for case in EXTREME_CASES
    ],
}


#: What a stage's GENERATOR wrote, on the login node, before the stage ran.
#:
#: Most inputs are produced by a task line of an earlier stage, and those are
#: what this tool follows. The extreme stage's client listings are not:
#: `study_emit.py extreme` cuts them from the same ranking every other cohort
#: comes from and writes them at emit time, in the same command that writes the
#: task file. Nothing in the task file could produce them, so without this the
#: stage reports an unmet dependency for every input that is in fact built by
#: the programme - a false alarm, and a checker nobody believes is worse than
#: none.
#:
#: The entry is keyed by task file rather than by command because that is what
#: the claim is about: THIS file's generator wrote these paths. The dependency
#: is real and it is met on the login node, not in the array.
GENERATOR_EFFECTS = {
    "d01_extreme.txt": (
        "tools/study_emit.py extreme",
        [f"outliers/extreme_{case}.json" for case in EXTREME_CASES],
    ),
}


def tokens(line):
    return line.split()


def flag_value(parts, flag):
    out = []
    for i, tok in enumerate(parts):
        if tok == flag and i + 1 < len(parts):
            out.append(parts[i + 1])
    return out


def normalise(path):
    p = path.replace("$FOA_STUDY_DIR/", "")
    return p


produced = set()
external = set()
problems = []

#: The carry settings and the repair, in the order they ran. They are named
#: d01_* rather than s*, so name order would place them before the stages that
#: build their inputs; the order is stated here instead of inferred.
CARRY = ("d01_c5_references.txt", "d01_c20_references.txt", "d01_five.txt",
         "d01_c20.txt", "d01_extreme.txt", "d01_extreme_references.txt",
         "d01_c10d10.txt", "d01_size_evals_rerun.txt")

stages = sorted(JOBS.glob("s*.txt"))
stages += [JOBS / name for name in CARRY if (JOBS / name).is_file()]

walked = {s.name for s in stages}
skipped = sorted(p for p in JOBS.glob("*.txt") if p.name not in walked)
print(f"stages: {len(stages)}\n")
for stage in stages:
    reads, writes = set(), set()
    for line in stage.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = tokens(line)

        # NOT EVERY PRODUCER IS A `foa` COMMAND. The extreme cohorts are cut by
        # a script, and a checker that only reads `foa` lines would report the
        # stage that consumes them as unmet - which is exactly the kind of false
        # alarm that teaches people to ignore the checker.
        if not line.startswith("foa "):
            for name, made in SCRIPT_EFFECTS.items():
                if name in line:
                    writes.update(made)
            continue
        cmd = parts[1]

        # `global-train --results-dir D` writes D/global_model. Most stages
        # point --results-dir at the study root, but the detector folds each get
        # their own, so the flag has to be read rather than assumed.
        if cmd == "global-train":
            for value in flag_value(parts, "--results-dir"):
                if value.startswith("$FOA_STUDY_DIR"):
                    base = normalise(value).rstrip("/")
                    if base and base != "$FOA_STUDY_DIR":
                        writes.add(f"{base}/global_model")
                        writes.add(f"{base}/global_results/fisher")
        for flag in READS:
            for value in flag_value(parts, flag):
                if value.startswith(EXTERNAL_OK):
                    external.add(value)
                elif value.startswith("$FOA_STUDY_DIR"):
                    reads.add(normalise(value))
        for flag in WRITES:
            for value in flag_value(parts, flag):
                if value.startswith("$FOA_STUDY_DIR"):
                    v = normalise(value)
                    writes.add(v)
                    if flag == "--out" and cmd == "fold-book":
                        writes.add(v + ".foldbook.npz")
        for effect in SIDE_EFFECTS.get(cmd, []):
            writes.add(effect)
        # select-outliers names its output by CONVENTION when --out is absent:
        # mode "worst" with --k N writes outliers/cohort_worstN.json. Counting
        # only --out reported those cohorts as unproduced, which is a checker
        # that cries wolf - and a checker nobody believes is worse than none.
        if cmd == "select-outliers" and not flag_value(parts, "--out"):
            mode = (flag_value(parts, "--mode") or ["sample"])[0]
            k = (flag_value(parts, "--k") or [""])[0]
            if mode == "worst" and k:
                writes.add(f"outliers/cohort_worst{k}.json")
        # the extreme cohorts are written by the study_emit generator, which
        # runs as its own stage; declare them where that stage sits.
        if cmd == "extreme" or "--case" in parts:
            for case in EXTREME_CASES:
                writes.add(f"outliers/extreme_{case}.json")
        # a run writes its own parent folder
        for value in flag_value(parts, "--parent"):
            writes.add(value)

    generator, made = GENERATOR_EFFECTS.get(stage.name, (None, []))
    writes.update(made)

    missing = sorted(r for r in reads if r not in produced and r not in writes)
    status = "OK " if not missing else "GAP"
    print(f"  {status} {stage.name:24s} reads {len(reads):3d}  writes {len(writes):3d}")
    if generator:
        print(f"        {len(made)} input(s) written at emit time by {generator}")
    for m in missing:
        print(f"        MISSING: {m}")
        problems.append((stage.name, m))
    produced |= writes

if skipped:
    print("\nNOT WALKED (in the jobs directory, outside the programme order):")
    for path in skipped:
        lines = sum(1 for line in path.read_text().splitlines()
                    if line.strip() and not line.startswith("#"))
        print(f"  {path.name:24s} {lines:4d} task lines")

print(f"\nexternal inputs (must exist before the run): {sorted(external) or 'none'}")
if problems:
    print(f"\n{len(problems)} unmet dependencies")
    sys.exit(1)
print("\nevery file every stage reads is produced by an earlier stage.")
