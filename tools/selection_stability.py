#!/usr/bin/env python3
"""
Is the repaired selection stable?

Averaging five detector folds instead of crowning one removes the coin flip, but
it does not by itself prove the ranking is robust - the average of five noisy
models is still an estimate. This measures how much the cut moves when the
evidence behind it is perturbed, without retraining anything:

  leave-one-fold-out   rank on four of the five folds, five ways. If the cohort
                       survives dropping a whole fold, it will survive the much
                       smaller perturbation of retraining the same five.

  single-fold          rank on each fold alone - what the old, crowned rule did.
                       The spread across these five is the instability that was
                       shipping.

The comparison is on the BAD/GOOD cut and on the worst-k membership, because
those are what the study actually consumes.
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

STUDY = Path(sys.argv[1])
BAD_FRACTION = 0.3
KS = (5, 10, 20)


def flat(payload):
    entries = payload["scores"] if isinstance(payload, dict) else payload
    out = {}
    for entry in entries:
        out.update(entry)
    return out


folds = {}
for k in (1, 2, 3, 4, 5):
    path = STUDY / "outliers" / f"writer_scores_fold{k}.json"
    folds[k] = flat(json.loads(path.read_text()))

writers = sorted(set.intersection(*(set(v) for v in folds.values())))
print(f"writers scored by every fold: {len(writers)}\n")


def rank(subset):
    """Writers worst-first under the mean of a subset of folds."""
    mean = {w: sum(folds[k][w] for k in subset) / len(subset) for w in writers}
    return sorted(writers, key=lambda w: (mean[w], w))


full = rank((1, 2, 3, 4, 5))
cut = math.ceil(BAD_FRACTION * len(writers))
full_bad = set(full[:cut])
full_k = {k: set(full[:k]) for k in KS}

print("leave-one-fold-out (the repaired rule, evidence perturbed):")
for dropped in (1, 2, 3, 4, 5):
    subset = tuple(k for k in (1, 2, 3, 4, 5) if k != dropped)
    r = rank(subset)
    bad = set(r[:cut])
    line = [f"  without fold {dropped}: bad pool {len(bad & full_bad)}/{cut}"
            f" ({100 * len(bad & full_bad) / cut:.1f}%)"]
    for k in KS:
        line.append(f"worst-{k} {len(set(r[:k]) & full_k[k])}/{k}")
    print("   ".join(line))

print("\nsingle fold alone (what the crowned rule did):")
for k in (1, 2, 3, 4, 5):
    r = rank((k,))
    bad = set(r[:cut])
    line = [f"  fold {k} only:     bad pool {len(bad & full_bad)}/{cut}"
            f" ({100 * len(bad & full_bad) / cut:.1f}%)"]
    for kk in KS:
        line.append(f"worst-{kk} {len(set(r[:kk]) & full_k[kk])}/{kk}")
    print("   ".join(line))

print("\npairwise agreement of the worst-10 between two single folds:")
worst = []
for a, b in combinations((1, 2, 3, 4, 5), 2):
    overlap = len(set(rank((a,))[:10]) & set(rank((b,))[:10]))
    worst.append(overlap)
print(f"  min {min(worst)}/10   mean {sum(worst) / len(worst):.1f}/10   max {max(worst)}/10")
