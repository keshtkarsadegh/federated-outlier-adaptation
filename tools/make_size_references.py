#!/usr/bin/env python3
"""
The reference rungs of one federation point, for any size.

Every federation size the study reports must be readable against the same four
references, or its rows float: a number means nothing without the shipped
model's own accuracy on that cohort, what a client reaches alone, what pooling
the cohort reaches, and what plain federated learning does at that
participation. Before this tool only the ten-client point had all four.

  do nothing    g-0 on the cohort, per fold. Forward passes, no training.
  isolated      each client alone, from g-0 and from scratch.
  centralized   the pooled cohort, from g-0 and from scratch.
  plain FL      FedAvg from g-0 and from scratch, once per participation rate,
                because a control that exists at one rate cannot be read
                against a row measured at another.

WHAT IS DELIBERATELY NOT REPEATED. Isolated, centralized and do-nothing do not
depend on the participation rate - they have no rounds - so a size measured at
two dropout levels gets them once. Only the federated control is emitted per
rate. Re-running them would spend GPU hours reproducing the same number under a
different file name.

A FEDERATION POINT THAT IS NOT A SIZE still needs the first rung. The extreme
stage federates two writers of the ten-client cohort, and it is scored like
every other setting - against what doing nothing gets on ITS rows, which is not
what doing nothing gets on the ten. It has no isolated, centralized or plain-FL
arm to ask for, because the pair is a controlled comparison of two arrangements
of the same rows and not a size the study reports rungs at, so ``--only
do-nothing`` emits the five forward passes and nothing else.

Usage::

    python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \\
        --cohort cohort_worst20.json --book cohort20 --per-round 18 16 \\
        --tag c20 --seed-base 720000

    python tools/make_size_references.py --study-dir "$FOA_STUDY_DIR" \\
        --cohort extreme_double.json --book cohort10 --only do-nothing \\
        --tag extreme --seed-base 730000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON = "--results-dir $FOA_STUDY_DIR --resolution 28 --classes digits"
OLD = "$FOA_STUDY_DIR/outliers/old_data.json"
OLD_BOOK = "$FOA_STUDY_DIR/fold_books/old_data.foldbook.npz"
G0 = "$FOA_STUDY_DIR/g0_model"
TRAIN = ("--epochs 100 --early-stopping-patience 10 --min-epochs 20 "
         "--batch-size 64 --eval-batch-size 256")
FOLDS = (1, 2, 3, 4, 5)
INITS = (("global", f" --init-checkpoint {G0}"), ("scratch", ""))


def build(clients, cohort, book, per_round, tag, seed_base, full_participation,
          only=None):
    lines = []

    for fold in FOLDS:
        lines.append(
            f"foa evaluate-book {COMMON} --model-path {G0} --fold-book {book} "
            f"--fold {fold} --part test --clients-file {cohort} --batch-size 256 "
            f"--tag {tag}_cohort_fold{fold} "
            f"--out $FOA_STUDY_DIR/g0_{tag}_evaluations.json"
        )
    if only == "do-nothing":
        return lines

    for position, client in enumerate(clients):
        for fold in FOLDS:
            for init, extra in INITS:
                # The seed the private arm never carried. It is a pure function
                # of the cohort's seed base, the client's position, the init and
                # the fold, so it is stable across regenerations and distinct
                # for every cell.
                seed = seed_base + position * 200 + (0 if init == "global" else 100) + fold
                lines.append(
                    f"foa isolated-train {COMMON} --clients-file {cohort} "
                    f"--only-client {client} --fold-book {book} --fold {fold} "
                    f"--old-book {OLD_BOOK} --old-fold all --old-clients-file {OLD} "
                    f"--init {init}{extra} {TRAIN} --seed {seed} "
                    f"--out $FOA_STUDY_DIR/isolated_{tag}/isolated_{init}_{client}_fold{fold}.json"
                )

    for fold in FOLDS:
        for init, extra in INITS:
            # Offset well clear of the per-client block above so the pooled arm
            # can never collide with a client's seed.
            seed = seed_base + 90000 + (0 if init == "global" else 100) + fold
            lines.append(
                f"foa isolated-train {COMMON} --pooled --clients-file {cohort} "
                f"--fold-book {book} --fold {fold} --old-book {OLD_BOOK} "
                f"--old-fold all --old-clients-file {OLD} --init {init}{extra} {TRAIN} "
                f"--seed {seed} "
                f"--out $FOA_STUDY_DIR/centralized_{tag}/centralized_{init}_fold{fold}.json"
            )

    for index, m in enumerate(per_round):
        rate = "full" if full_participation else f"m{m}"
        policy = "--policy all --participation 1.0" if full_participation else \
                 f"--policy uniform --clients-per-round {m}"
        for init, _ in INITS:
            for fold in FOLDS:
                seed = seed_base + index * 200 + (0 if init == "global" else 100) + fold
                lines.append(
                    f"foa final {COMMON} --model fedavg_cnn --trainer BaseTrainer "
                    f"--parent d01_{tag}_{rate}_control_{init}_fold{fold} "
                    f"--aggregation fedavg --init {init} --global-name g0 "
                    f"--outliers-file {cohort} --fold-book {book} --fold {fold} "
                    f"--old-book {OLD_BOOK} --old-clients-file {OLD} --old-fold all "
                    f"{policy} --sampler-seed {seed} --track-clients --rounds 100 "
                    f"--epochs 5 --batch-size 64 --eval-batch-size 256 "
                    f"--save-final-model --seed {fold} "
                    f"--outer-workers 1 --inner-workers 1"
                )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--cohort", required=True, help="file name under outliers/")
    parser.add_argument("--book", required=True, help="fold book stem under fold_books/")
    parser.add_argument("--per-round", type=int, nargs="+", default=(),
                        help="one participation count per dropout level; not "
                             "needed with --only do-nothing")
    parser.add_argument("--only", choices=("do-nothing",), default=None,
                        help="Emit the do-nothing rung alone. For a federation "
                             "point that is scored but is not a size: the "
                             "extreme pair needs its own shipped-model "
                             "accuracy and has no rungs to stand it against.")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--full-participation", action="store_true",
                        help="the extreme cases never drop a client")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    study = Path(args.study_dir)
    payload = json.loads((study / "outliers" / args.cohort).read_text())
    clients = payload["clients"] if isinstance(payload, dict) else list(payload)
    if args.only is None and not args.per_round:
        raise SystemExit("--per-round is required unless --only do-nothing")
    for m in args.per_round:
        if not args.full_participation and not 1 <= m <= len(clients):
            raise SystemExit(
                f"{m} of {len(clients)} is not a participation count for this cohort"
            )

    lines = build(clients,
                  f"$FOA_STUDY_DIR/outliers/{args.cohort}",
                  f"$FOA_STUDY_DIR/fold_books/{args.book}.foldbook.npz",
                  args.per_round, args.tag, args.seed_base,
                  args.full_participation, args.only)

    header = [
        f"# d01_{args.tag}_references.txt - the reference rungs of the "
        f"{len(clients)}-client point.",
        "#",
        "# Generated by tools/make_size_references.py. do-nothing, isolated and",
        "# centralized carry no rounds, so they are emitted once for the size;",
        "# the federated control is emitted once per participation rate.",
        "#",
        f"# cohort {args.cohort}, book {args.book}, participation "
        f"{'full' if args.full_participation else args.per_round}.",
    ]
    if args.only == "do-nothing":
        header = header[:1] + [
            "#",
            "# Generated by tools/make_size_references.py --only do-nothing.",
            "# The do-nothing rung alone: g-0 over this cohort's test rows, per",
            "# fold, no training. This is the A0 the setting's score subtracts,",
            "# and it is not the ten-writer cohort's.",
            "#",
            f"# cohort {args.cohort}, book {args.book}.",
        ]
    out = Path(args.out) if args.out else study / "jobs" / f"d01_{args.tag}_references.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header) + "\n" + "\n".join(lines) + "\n")

    if args.only == "do-nothing":
        print(f"{out.name}: {len(lines)} tasks (do-nothing {len(FOLDS)})")
        return 0
    n_fl = 2 * len(args.per_round) * len(FOLDS)
    print(f"{out.name}: {len(lines)} tasks "
          f"(do-nothing {len(FOLDS)}, isolated {len(clients) * len(FOLDS) * 2}, "
          f"centralized {len(FOLDS) * 2}, plain FL {n_fl})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
