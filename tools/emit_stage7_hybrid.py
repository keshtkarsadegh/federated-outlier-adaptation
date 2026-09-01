"""
SUPERSEDED - see tools/study_emit.py reg-hybrid.

The blend this study ran was emitted per family by

    python tools/study_emit.py reg-hybrid --root "$FOA_STUDY_DIR" \
        --out <path> --expect 15 [--hybrid-family sequential]

into ``jobs/s18_hybrid.txt`` and ``jobs/s19_hybrid_seq.txt``, both
reproducible byte for byte, each writing its own construction record under
``tables/``. That the blend is a per-family object is the substantive
difference: this tool builds one blend from one pair of winners, and the two
schedules select different halves - kd_T0p25_a0p9 with fisher_lam8 in the
concurrent family, kd_T2_a0p99 with fisher_lam0p1 in the sequential one - so
a single blend would price one schedule's penalty on both.

``hybrid_cells`` here is still imported by the tests that pin the blend's
arithmetic, so the module is kept rather than deleted.

Emit the kd+fisher hybrid cells, once the kd and fisher winners are known.

    python tools/emit_stage7_hybrid.py --selection stage7_selection.json \
        [--family concurrent] [--out stage7_hybrid.txt] [--rounds 100]

Or with the two winners named directly, when the selection file is not to hand:

    python tools/emit_stage7_hybrid.py --kd-cell kd_T4_a0p9 --fisher-cell fisher_lam1000

Reads JSON, writes text: a login-node job.

Why this is not in the screen
-----------------------------
``kd+fisher`` blends two penalties with ``mix``: the objective is
``lam * (mix * KD + (1 - mix) * Fisher)``.  Screening it alongside the others
would mean guessing ``lam`` and ``T`` for the KD half and ``lam`` for the Fisher
half before either was known - a three-dimensional grid to answer a question
that only becomes meaningful *after* the two one-dimensional ones are settled.
So it waits, and then costs three cells instead of hundreds.

What it takes from each winner
------------------------------
``T`` and the KD strength come from the winning ``kd`` cell; the Fisher strength
comes from the winning ``fisher`` cell.  The blend's single ``lam`` is set to the
KD winner's, and ``mix`` splits it, so ``mix = 1`` would reproduce the KD winner
exactly - which is what makes the three blend points readable as a line between
the two methods rather than as three unrelated runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from federated_outlier_adaptation.training.reg_cells import (
    ANCHOR,
    FULL_ROUNDS,
    HYBRID_MIXES,
    _cell,
    _fmt,
    screen_cells,
)


def winners_from_selection(path: Path, family: str):
    """The ``kd`` and ``fisher`` winning cell ids of one family."""
    with open(path) as handle:
        report = json.load(handle)
    methods = report.get("selection", {}).get("methods", {})
    kd = methods.get("kd", {}).get(family, {}).get("winner")
    fisher = methods.get("fisher", {}).get(family, {}).get("winner")
    if not kd or not fisher:
        raise SystemExit(
            f"{path} has no {family} winner for kd ({kd!r}) and fisher "
            f"({fisher!r}); the hybrid needs both."
        )
    return kd, fisher


def hybrid_cells(kd_cell_id: str, fisher_cell_id: str):
    """The three blend cells, built from the two winners' hyperparameters."""
    by_id = {cell["id"]: cell for cell in screen_cells()}
    for name in (kd_cell_id, fisher_cell_id):
        if name not in by_id:
            raise SystemExit(f"Unknown stage-7 cell {name!r}.")
    kd, fisher = by_id[kd_cell_id], by_id[fisher_cell_id]
    if kd["method"] != "kd" or fisher["method"] != "fisher":
        raise SystemExit(
            f"Expected a kd cell and a fisher cell; got {kd['method']!r} and "
            f"{fisher['method']!r}."
        )

    cells = []
    for mix in HYBRID_MIXES:
        cells.append(_cell(
            f"hybrid_mix{_fmt(mix)}", "kd+fisher", "kd+fisher",
            f"kd+fisher blend, mix={mix:g} "
            f"(KD from {kd_cell_id}, Fisher from {fisher_cell_id})",
            needs_fisher=True,
            lam=kd["hypers"]["lam"],
            T=kd["hypers"]["T"],
            mix=mix,
        ))
    return cells, kd, fisher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=None, help="stage7_selection.json")
    parser.add_argument("--family", default="concurrent", help="Which family's winners.")
    parser.add_argument("--kd-cell", default=None)
    parser.add_argument("--fisher-cell", default=None)
    parser.add_argument("--out", default="stage7_hybrid.txt")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=FULL_ROUNDS)
    args = parser.parse_args()

    if args.kd_cell and args.fisher_cell:
        kd_id, fisher_id = args.kd_cell, args.fisher_cell
    elif args.selection:
        kd_id, fisher_id = winners_from_selection(Path(args.selection), args.family)
    else:
        raise SystemExit("Give --selection, or both --kd-cell and --fisher-cell.")

    cells, kd, fisher = hybrid_cells(kd_id, fisher_id)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_stage7_screen import task_line

    lines = [
        "# stage7_hybrid.txt - the kd+fisher blend, priced after both winners.",
        "#",
        "# The objective is lam * (mix * KD + (1 - mix) * Fisher), so mix = 1 would",
        "# reproduce the KD winner exactly. That is what makes these three points",
        "# readable as a line between the two methods rather than three unrelated",
        "# runs.",
        "#",
        f"# KD half     from {kd_id}: {kd['note']}",
        f"# Fisher half from {fisher_id}: {fisher['note']}",
        f"# blends      mix in {{{', '.join(f'{m:g}' for m in HYBRID_MIXES)}}}",
        "#",
        "# REQUIRES THE FISHER EXPORT:",
        '#   G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[\'selected_fold\'])" \\',
        "#             $FOA_RESULTS_DIR/main_v6/g0_selection.json)",
        "#   export G0_FOLD",
        "#",
        f"# {len(cells)} cells x {len(args.folds)} folds = {len(cells) * len(args.folds)} tasks",
        "#",
    ]
    for index, cell in enumerate(cells):
        lines.append(f"# --- {cell['id']}: {cell['note']}")
        for fold in args.folds:
            lines.append(task_line(cell, fold, args.rounds, 74000 + index * 10 + fold))

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(cells) * len(args.folds)} task lines "
          f"({len(cells)} blends x {len(args.folds)} folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
