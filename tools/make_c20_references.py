#!/usr/bin/env python3
"""
The reference rungs of the twenty-client point.

The scaling table reports the transferred configurations at twenty clients and
nothing else, so those rows have no floor and no ceiling: a reader cannot tell
whether 0.9070 is good, because the shipped model's own accuracy on that cohort,
what a client reaches alone, what pooling the cohort reaches, and what plain
federated learning does are all missing at that size. Every other federation
size in the study has them. This file supplies them.

Four arms, the same four as the main setting:

  do nothing    g-0 on the twenty-client cohort, per fold. Five forward passes.
  isolated      each of the twenty clients alone, from g-0 and from scratch.
  centralized   the pooled cohort, from g-0 and from scratch.
  plain FL      FedAvg from g-0 and from scratch, at BOTH dropout levels,
                because the scaling table reports both and a control that
                exists at only one of them cannot be read against the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON = "--results-dir $FOA_STUDY_DIR --resolution 28 --classes digits"
COHORT = "$FOA_STUDY_DIR/outliers/cohort_worst20.json"
BOOK = "$FOA_STUDY_DIR/fold_books/cohort20.foldbook.npz"
OLD = "$FOA_STUDY_DIR/outliers/old_data.json"
OLD_BOOK = "$FOA_STUDY_DIR/fold_books/old_data.foldbook.npz"
G0 = "$FOA_STUDY_DIR/g0_model"
TRAIN = ("--epochs 100 --early-stopping-patience 10 --min-epochs 20 "
         "--batch-size 64 --eval-batch-size 256")
FOLDS = (1, 2, 3, 4, 5)
SEED_BASE = 720000          # clear of every stage seed the study already used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    study = Path(args.study_dir)
    cohort = json.loads((study / "outliers" / "cohort_worst20.json").read_text())
    clients = cohort["clients"] if isinstance(cohort, dict) else list(cohort)
    if len(clients) != 20:
        raise SystemExit(f"the cohort holds {len(clients)} clients, expected 20")

    lines = []

    # do nothing: the cohort under the shipped model, per fold
    for fold in FOLDS:
        lines.append(
            f"foa evaluate-book {COMMON} --model-path {G0} --fold-book {BOOK} "
            f"--fold {fold} --part test --clients-file {COHORT} --batch-size 256 "
            f"--tag c20_cohort_fold{fold} "
            f"--out $FOA_STUDY_DIR/g0_c20_evaluations.json"
        )

    # isolated: every client alone, both initialisations
    for client in clients:
        for fold in FOLDS:
            for init, extra in (("global", f" --init-checkpoint {G0}"), ("scratch", "")):
                lines.append(
                    f"foa isolated-train {COMMON} --clients-file {COHORT} "
                    f"--only-client {client} --fold-book {BOOK} --fold {fold} "
                    f"--old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD} "
                    f"--init {init}{extra} {TRAIN} "
                    f"--out $FOA_STUDY_DIR/isolated_c20/isolated_{init}_{client}_fold{fold}.json"
                )

    # centralized: the pooled cohort, both initialisations
    for fold in FOLDS:
        for init, extra in (("global", f" --init-checkpoint {G0}"), ("scratch", "")):
            lines.append(
                f"foa isolated-train {COMMON} --pooled --clients-file {COHORT} "
                f"--fold-book {BOOK} --fold {fold} --old-book {OLD_BOOK} "
                f"--old-fold all --old-clients-file {OLD} --init {init}{extra} {TRAIN} "
                f"--out $FOA_STUDY_DIR/centralized_c20/centralized_{init}_fold{fold}.json"
            )

    # plain federated learning, both initialisations and both dropout levels
    for per_round, drop in ((18, "d10"), (16, "d20")):
        for init in ("global", "scratch"):
            for fold in FOLDS:
                seed = SEED_BASE + (0 if drop == "d10" else 50) + \
                       (0 if init == "global" else 100) + fold
                lines.append(
                    f"foa final {COMMON} --model fedavg_cnn --trainer BaseTrainer "
                    f"--parent d01_c20{drop}_control_{init}_fold{fold} "
                    f"--aggregation fedavg --init {init} --global-name g0 "
                    f"--outliers-file {COHORT} --fold-book {BOOK} --fold {fold} "
                    f"--old-book {OLD_BOOK} --old-clients-file {OLD} --old-fold all "
                    f"--policy uniform --clients-per-round {per_round} "
                    f"--sampler-seed {seed} --track-clients --rounds 100 --epochs 5 "
                    f"--batch-size 64 --eval-batch-size 256 --save-final-model "
                    f"--seed {fold} --outer-workers 1 --inner-workers 1"
                )

    out = Path(args.out) if args.out else study / "jobs" / "d01_c20_references.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__doc__.replace("\n", "\n# ").join(["# ", "\n"]).strip() + "\n"
                   + "\n".join(lines) + "\n")
    print(f"{out.name}: {len(lines)} tasks")
    print(f"  do nothing   {len(FOLDS)}")
    print(f"  isolated     {len(clients) * len(FOLDS) * 2}")
    print(f"  centralized  {len(FOLDS) * 2}")
    print(f"  plain FL     {2 * 2 * len(FOLDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
