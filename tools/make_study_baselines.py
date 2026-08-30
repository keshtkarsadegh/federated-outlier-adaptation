#!/usr/bin/env python3
"""
The reference rungs of a study, generated from its configuration.

Four arms, and each answers something no federated number answers alone:

**do nothing** - g-0 on the cohort and on the old population. The adaptation
floor and the preservation ceiling in one pass.

**isolated** - each client trained alone, from g-0 and from scratch. From g-0 it
is the honest alternative to federating: a federated method that cannot beat it
buys nothing but complexity.

**centralized** - the cohort pooled into one training set. What the problem
allows when privacy is not a constraint.

**plain federated learning** - FedAvg from g-0 and from scratch. The arm that
can forget, and therefore the only one with anything for a regulariser or an
aggregation rule to repair.

Everything is named from the study's own configuration - cohort file, fold book,
participation - so a change of design cannot leave this pointing at files the
study no longer writes. That failure has already cost this project two reruns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

COMMON = "--results-dir $FOA_STUDY_DIR --resolution 28 --classes digits"
OLD = "$FOA_STUDY_DIR/outliers/old_data.json"
OLD_BOOK = "$FOA_STUDY_DIR/fold_books/old_data.foldbook.npz"
G0 = "$FOA_STUDY_DIR/g0_model"
TRAIN = ("--epochs 100 --early-stopping-patience 10 --min-epochs 20 "
         "--batch-size 64 --eval-batch-size 256")
FOLDS = (1, 2, 3, 4, 5)


def build(cfg, clients):
    cohort = f"$FOA_STUDY_DIR/outliers/{cfg.cohort_file_name}"
    book = f"$FOA_STUDY_DIR/fold_books/{cfg.cohort_book_name}.foldbook.npz"

    donothing = [
        f"foa evaluate-book {COMMON} --model-path {G0} --fold-book {OLD_BOOK} "
        f"--fold {f} --part test --clients-file {OLD} --batch-size 256 "
        f"--tag old_data_fold{f} --out $FOA_STUDY_DIR/g0_evaluations.json"
        for f in FOLDS
    ] + [
        f"foa evaluate-book {COMMON} --model-path {G0} --fold-book {book} "
        f"--fold {f} --part test --clients-file {cohort} --batch-size 256 "
        f"--tag cohort_fold{f} --out $FOA_STUDY_DIR/g0_perfold_evaluations.json"
        for f in FOLDS
    ]

    isolated = []
    for client in clients:
        for fold in FOLDS:
            for init, extra in (("global", f" --init-checkpoint {G0}"), ("scratch", "")):
                isolated.append(
                    f"foa isolated-train {COMMON} --clients-file {cohort} "
                    f"--only-client {client} --fold-book {book} --fold {fold} "
                    f"--old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD} "
                    f"--init {init}{extra} {TRAIN} "
                    f"--out $FOA_STUDY_DIR/isolated/isolated_{init}_{client}_fold{fold}.json"
                )

    centralized = [
        f"foa isolated-train {COMMON} --pooled --clients-file {cohort} "
        f"--fold-book {book} --fold {fold} --old-book {OLD_BOOK} --old-fold all "
        f"--old-clients-file {OLD} --init {init}{extra} {TRAIN} "
        f"--out $FOA_STUDY_DIR/centralized/centralized_outliers_{init}_fold{fold}.json"
        for fold in FOLDS
        for init, extra in (("global", f" --init-checkpoint {G0}"), ("scratch", ""))
    ]

    fl = []
    for init in ("global", "scratch"):
        for fold in FOLDS:
            seed = cfg.seed_base + 300 + fold + (100 if init == "scratch" else 0)
            fl.append(
                f"foa final {COMMON} --model {cfg.model} --trainer BaseTrainer "
                f"--parent {cfg.tag}_fl_{init}_fold{fold} --aggregation fedavg "
                f"--init {init} --global-name g0 --outliers-file {cohort} "
                f"--fold-book {book} --fold {fold} --old-book {OLD_BOOK} "
                f"--old-clients-file {OLD} --old-fold all --policy uniform "
                f"--clients-per-round {cfg.clients_per_round} --sampler-seed {seed} "
                f"--track-clients --rounds 100 --epochs 5 --batch-size 64 "
                f"--eval-batch-size 256 --save-final-model --seed {fold} "
                f"--outer-workers 2 --inner-workers 1"
            )
    return {"donothing": donothing, "isolated": isolated,
            "centralized": centralized, "fl": fl}


HEAD = {
    "donothing": "# S7a: the shipped model measured on both populations. Forward passes only.\n"
                 "# Lines 1-5 are the preservation ceiling, 6-10 the adaptation floor.\n"
                 "# APPENDS to two shared files, so this stage runs under a throttle of one.",
    "isolated": "# S7b: each client alone, from g-0 and from scratch.\n"
                "# From g-0 this is what a client gets without federating at all; a federated\n"
                "# method that cannot beat it is buying complexity and nothing else. From\n"
                "# scratch it separates the client's own data from the shipped representation.",
    "centralized": "# S7c: the whole cohort pooled - no client boundaries, no aggregation rule.\n"
                   "# The ceiling, and the number that says whether a federated result is close\n"
                   "# to the best available or merely better than doing nothing.",
    "fl": "# S8: plain FedAvg, the arm that can forget.\n"
          "# An isolated client keeps its own model and drifts only as far as its own data\n"
          "# pulls it. FedAvg averages every client's drift each round and sends the average\n"
          "# back out as the next round's starting point, a hundred times over. That\n"
          "# compounding is where forgetting comes from - so if this arm does not forget,\n"
          "# there is nothing here for a regulariser or an aggregation rule to repair.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--study-dir", required=True)
    args = parser.parse_args()

    from federated_outlier_adaptation.training.study_config import config

    cfg = config(args.study)
    import json
    study = Path(args.study_dir)
    cohort = json.loads((study / "outliers" / cfg.cohort_file_name).read_text())
    clients = cohort["clients"] if isinstance(cohort, dict) else list(cohort)
    if len(clients) != cfg.cohort_size:
        # StudyConfig is frozen on purpose: a design point is not something a
        # generator may edit on its way past. So the cohort is carried
        # separately and checked against the design rather than replacing it.
        raise SystemExit(
            f"the cohort file holds {len(clients)} clients and "
            f"{cfg.name} is designed for {cfg.cohort_size}"
        )

    jobs = study / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, lines in build(cfg, clients).items():
        path = jobs / f"{cfg.tag}_{name}.txt"
        path.write_text(HEAD[name] + "\n" + "\n".join(lines) + "\n")
        print(f"{path.name}: {len(lines)} tasks")
        total += len(lines)
    print(f"\ntotal: {total} tasks   ({cfg.describe()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
