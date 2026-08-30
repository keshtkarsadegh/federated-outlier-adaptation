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
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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

stages = sorted(JOBS.glob("s*.txt"))
print(f"stages: {len(stages)}\n")
for stage in stages:
    reads, writes = set(), set()
    for line in stage.read_text().splitlines():
        if not line.startswith("foa "):
            continue
        parts = tokens(line)
        cmd = parts[1]
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
            for case in ("single", "double", "dual"):
                writes.add(f"outliers/extreme_{case}.json")
        # a run writes its own parent folder
        for value in flag_value(parts, "--parent"):
            writes.add(value)

    missing = sorted(r for r in reads if r not in produced and r not in writes)
    status = "OK " if not missing else "GAP"
    print(f"  {status} {stage.name:24s} reads {len(reads):3d}  writes {len(writes):3d}")
    for m in missing:
        print(f"        MISSING: {m}")
        problems.append((stage.name, m))
    produced |= writes

print(f"\nexternal inputs (must exist before the run): {sorted(external) or 'none'}")
if problems:
    print(f"\n{len(problems)} unmet dependencies")
    sys.exit(1)
print("\nevery file every stage reads is produced by an earlier stage.")
