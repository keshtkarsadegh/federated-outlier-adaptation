"""
PARTLY SUPERSEDED, and the split matters.

LIVE: ``collect`` and ``summarise`` here are the regularisation screen's
readers, imported by ``tools/study_emit.py`` and by
``tools/weight_sensitivity.py``. Every regularisation number this study
reports passes through them.

SUPERSEDED: ``emit_full`` writes the prior programme's ``stage7_full.txt``.
The regularisation finals this study ran are ``jobs/s17_reg_full4.txt``,
emitted by ``tools/study_emit.py reg-full`` and reproducible from it byte for
byte. Do not emit finals from here.

Rank the stage-7 regularisation cells per method per family, and emit the finals.

    python tools/select_reg_screen.py --root $FOA_STUDY_DIR \
        [--out stage7_full.txt] [--report stage7_selection.json]

Reads JSON, writes text: a login-node job.

Why the unit of selection is (method, family) and not just (method)
-------------------------------------------------------------------
Every stage-7 task ran ``--aggregation fedavg``, which is two jobs: the parallel
schedule and the cyclic one.  A penalty that stabilises a server average and a
penalty that stabilises a sequential walk are not the same claim, and collapsing
them would let a method win the table on the strength of one schedule while
being useless on the other.  So each method's hyperparameters are chosen
**within each family**, and where the two families choose differently the
emitted finals say which family's result to read from which line.

What it ranks on
----------------
The fold-mean pooled **validation** accuracy of the cohort, per (cell, family).
Selection is per-method argmax over adaptation - unlike stage 6 there is no
preservation budget here, because the whole point of a regulariser is that it
buys preservation, and constraining it would be selecting on the axis the method
exists to move.  The preservation figures are carried into the report next to
every cell so the trade is visible, and the *paper* reports both.

Five folds of one cell are averaged before anything is compared, and the spread
is carried, so a cell that wins by less than its own fold-to-fold noise is
visible as such.

SELECT ON VALIDATION, REPORT ON TEST
------------------------------------
Both axes of a *selection* are read on the halves a selection is allowed to
see: ``pool_val_accuracies`` for adaptation and ``source_val_accuracies`` for
preservation.  The test columns - ``final_evaluation.clients.accuracy`` and
``final_evaluation.old.mean`` - ride along as ``adaptation_test`` and
``preservation_test`` and are what the reporting tools quote, but nothing here
ranks on them.

This was mixed until it was checked: adaptation came from validation while
preservation came from the source *test* partitions, so a winner was chosen on
one half of the data for one axis and the other half for the other.  Making it
consistent moves six of fourteen regularisation winners - none of them the
leading methods, all among the weakest four - and no aggregation winner at all.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from federated_outlier_adaptation.training.reg_cells import (
    FAMILIES,
    FULL_ROUNDS,
    control_cell,
    methods,
    screen_cells,
)

#: Folder prefix of a screening run.  A parameter rather than a constant so
#: that a second study reuses this selector instead of copying it - the
#: selection rule is the science, the prefix is only an address.
PARENT_PREFIX = "coh7_reg_"


def run_pattern(prefix: str = PARENT_PREFIX):
    """The compiled folder matcher for one prefix."""
    return re.compile(
        rf"^{re.escape(prefix)}(?P<cell>.+)_fold(?P<fold>\d+)(?:_.*)?$"
    )

#: ``coh7_reg_<cell>_fold<k>`` plus whatever the driver appends - it builds its
#: output path from ``<parent>_<trainer>_grid_search``, so the fold number is
#: not the tail of the name.  Cell ids never contain ``_fold``.
RUN_DIR = re.compile(rf"^{re.escape(PARENT_PREFIX)}(?P<cell>.+)_fold(?P<fold>\d+)(?:_.*)?$")


def parse_run_dir(name: str, prefix: str = PARENT_PREFIX):
    """``(cell id, fold)`` of a run folder, or ``None`` when it is not one."""
    match = run_pattern(prefix).match(name)
    return None if match is None else (match.group("cell"), int(match.group("fold")))


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _last(series) -> Optional[float]:
    if not series:
        return None
    for value in reversed(list(series)):
        if value is not None:
            return float(value)
    return None


def read_family_runs(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    The two families' results inside one (cell, fold) folder.

    The driver writes one ``accuracies_*.json`` per family job, each carrying
    its own ``scenario``, so the two are told apart by what they say they are
    rather than by where they happen to sit.
    """
    found: Dict[str, Dict[str, Any]] = {}
    for payload_path in sorted(run_dir.rglob("accuracies_*.json")):
        data = _load(payload_path)
        if not data:
            continue
        family = data.get("scenario")
        if family not in FAMILIES:
            continue
        final = data.get("final_evaluation") or {}
        clients = final.get("clients") or {}
        old = final.get("old") or {}
        adaptation = _last(data.get("pool_val_accuracies"))
        fallback = False
        if adaptation is None:
            adaptation = clients.get("accuracy")
            fallback = adaptation is not None
        found[family] = {
            "adaptation": adaptation,
            "adaptation_is_test_fallback": fallback,
            "adaptation_test": clients.get("accuracy"),
            # SELECTION READS VALIDATION ON BOTH AXES - see the module
            # docstring. The test columns ride along for the reporters.
            "preservation": (_last(data.get("source_val_accuracies"))
                             if _last(data.get("source_val_accuracies")) is not None
                             else old.get("mean")),
            "preservation_is_test_fallback":
                _last(data.get("source_val_accuracies")) is None
                and old.get("mean") is not None,
            "preservation_test": old.get("mean"),
            "preservation_sd": old.get("sd"),
            "agg_method_name": data.get("agg_method_name"),
            "path": str(payload_path),
        }
    return found


def collect(root: Path, cells, prefix: str = PARENT_PREFIX) -> Dict[str, Dict[str, Any]]:
    """Every cell's per-fold, per-family runs, keyed by cell id."""
    found: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        folds: Dict[int, Dict[str, Any]] = {}
        for run_dir in sorted(root.glob(f"{prefix}{cell['id']}_fold*")):
            if not run_dir.is_dir():
                continue
            parsed = parse_run_dir(run_dir.name, prefix)
            # The glob is by cell id, but `kd_T1_a0p1` also prefixes nothing
            # else here; the id check keeps that true if ids ever grow.
            if parsed is None or parsed[0] != cell["id"]:
                continue
            families = read_family_runs(run_dir)
            if families:
                folds[parsed[1]] = families
        if folds:
            found[cell["id"]] = {"cell": cell, "folds": folds}
    return found


def _agg(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    usable = [v for v in values if v is not None]
    if not usable:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": mean(usable),
        "sd": pstdev(usable) if len(usable) > 1 else 0.0,
        "n": len(usable),
    }


def summarise(found: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per (cell, family): the fold means of both axes."""
    rows = []
    for cell_id, entry in found.items():
        cell = entry["cell"]
        for family in FAMILIES:
            per_fold = {
                fold: families[family]
                for fold, families in entry["folds"].items()
                if family in families
            }
            if not per_fold:
                continue
            rows.append({
                "id": cell_id,
                "method": cell["method"],
                "family": family,
                "space": cell["space"],
                "hypers": cell["hypers"],
                "note": cell["note"],
                "folds": sorted(per_fold),
                "adaptation": _agg([r["adaptation"] for r in per_fold.values()]),
                "adaptation_test": _agg([r["adaptation_test"] for r in per_fold.values()]),
                "preservation": _agg([r["preservation"] for r in per_fold.values()]),
                "test_fallback": any(
                    r["adaptation_is_test_fallback"] for r in per_fold.values()
                ),
            })
    return rows


def select(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Each method's best cell, within each family.

    Argmax over the fold-mean validation adaptation.  A (method, family) with
    nothing measured has no winner rather than an arbitrary one.
    """
    result: Dict[str, Any] = {"methods": {}}
    for method in methods():
        per_family: Dict[str, Any] = {}
        for family in FAMILIES:
            usable = [
                row for row in rows
                if row["method"] == method
                and row["family"] == family
                and row["adaptation"]["mean"] is not None
            ]
            if not usable:
                per_family[family] = {"winner": None, "considered": 0, "ranked": []}
                continue
            ranked = sorted(usable, key=lambda r: r["adaptation"]["mean"], reverse=True)
            per_family[family] = {
                "winner": ranked[0]["id"],
                "considered": len(usable),
                "ranked": [
                    {
                        "id": r["id"],
                        "adaptation": r["adaptation"]["mean"],
                        "adaptation_sd": r["adaptation"]["sd"],
                        "preservation": r["preservation"]["mean"],
                        "preservation_sd": r["preservation"]["sd"],
                        "folds": len(r["folds"]),
                    }
                    for r in ranked
                ],
            }
        winners = {f: per_family[f]["winner"] for f in FAMILIES}
        per_family["agree"] = (
            winners[FAMILIES[0]] is not None
            and winners[FAMILIES[0]] == winners[FAMILIES[1]]
        )
        result["methods"][method] = per_family
    return result


def emit_full(selection, rows, folds, out: Path, rounds: int) -> int:
    """
    Write each method's winners out at the full horizon.

    Where the two families chose the same cell, one line serves both - the task
    runs ``--aggregation fedavg``, which is both schedules anyway.  Where they
    chose differently, both cells are emitted and the header says which family's
    result to read from which line, because each of those lines still produces
    two family results and only one of them is the selected one.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_stage7_screen import task_line

    by_id = {cell["id"]: cell for cell in screen_cells()}
    by_row = {(r["id"], r["family"]): r for r in rows}

    chosen: List[str] = []
    notes: List[str] = []
    for method in methods():
        info = selection["methods"].get(method, {})
        winners = {f: info.get(f, {}).get("winner") for f in FAMILIES}
        if not any(winners.values()):
            notes.append(f"# {method}: nothing measured in either family; nothing selected")
            continue
        if info.get("agree"):
            cell_id = winners[FAMILIES[0]]
            row = by_row[(cell_id, FAMILIES[0])]
            notes.append(
                f"# {method}: {cell_id} wins in BOTH families "
                f"(validation {row['adaptation']['mean']:.4f}); one line serves both"
            )
            if cell_id not in chosen:
                chosen.append(cell_id)
            continue
        for family in FAMILIES:
            cell_id = winners[family]
            if cell_id is None:
                notes.append(f"# {method}/{family}: nothing measured; nothing selected")
                continue
            row = by_row[(cell_id, family)]
            notes.append(
                f"# {method}/{family}: {cell_id} "
                f"(validation {row['adaptation']['mean']:.4f}) "
                f"- READ THE {family.upper()} RESULT FROM THIS LINE"
            )
            if cell_id not in chosen:
                chosen.append(cell_id)

    control = control_cell()
    for cell_id, cell in list(by_id.items()) + [(control["id"], control)]:
        if cell["method"] in ("param_l2",) and cell["hypers"].get("lam") == 0.0:
            if cell_id not in chosen:
                chosen.append(cell_id)
    if control["id"] not in chosen:
        chosen.append(control["id"])
    by_id[control["id"]] = control

    lines = [
        "# stage7_full.txt - the stage-7 winners at the full horizon.",
        "#",
        "# Emitted by tools/select_reg_screen.py from the 25-round screen. The",
        "# screen ranked; this measures. Every line runs "
        f"{rounds} rounds under the",
        "# same protocol as stage 5 and stage 6, so all three are comparable.",
        "#",
        "# SELECTION: per method, per family, argmax of the fold-mean pooled",
        "# VALIDATION accuracy. No preservation budget - a regulariser exists to",
        "# buy preservation, so constraining that axis would be selecting on the",
        "# thing the method is for. Both axes are in stage7_selection.json.",
        "#",
        "# Each line runs --aggregation fedavg, i.e. BOTH families:",
    ]
    lines += notes
    lines += [
        "#",
        "# Both controls come along: param_l2_mu0 (an exact no-op - the trainer",
        "# returns zero for any lam <= 0) and control_none (plain BaseTrainer). If",
        "# those two disagree by more than run-to-run noise, the penalty machinery",
        "# is doing something while switched off.",
        "#",
        "# REMEMBER THE EXPORT if any winner is a Fisher cell:",
        '#   G0_FOLD=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))[\'selected_fold\'])" \\',
        "#             $FOA_RESULTS_DIR/main_v6/g0_selection.json)",
        "#   export G0_FOLD",
        "#",
        f"# {len(chosen)} cells x {len(folds)} folds = {len(chosen) * len(folds)} tasks",
        "#",
    ]

    order = list(by_id)
    for cell_id in chosen:
        cell = by_id[cell_id]
        lines.append(f"# --- {cell_id}: {cell['note']}")
        for fold in folds:
            index = order.index(cell_id)
            lines.append(task_line(cell, fold, rounds, 73000 + index * 10 + fold))

    out.write_text("\n".join(lines) + "\n")
    return len(chosen) * len(folds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="stage7_full.txt")
    parser.add_argument("--report", default="stage7_selection.json")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rounds", type=int, default=FULL_ROUNDS)
    args = parser.parse_args()

    cells = screen_cells()
    found = collect(Path(args.root), cells)
    rows = summarise(found)
    if not rows:
        raise SystemExit(
            f"No stage-7 screening results under {args.root}; expected folders "
            f"named {PARENT_PREFIX}<cell>_fold<k>."
        )
    selection = select(rows)
    tasks = emit_full(selection, rows, args.folds, Path(args.out), args.rounds)

    Path(args.report).write_text(json.dumps(
        {"cells_found": len(found), "cells_expected": len(cells),
         "rows": len(rows), "selection": selection, "cells": rows}, indent=2,
    ))
    print(f"read {len(found)} of {len(cells)} cells ({len(rows)} cell-family rows)")
    for method in methods():
        info = selection["methods"][method]
        winners = " ".join(f"{f}={info[f]['winner']}" for f in FAMILIES)
        print(f"  {method:<14} {winners}  agree={info['agree']}")
    print(f"wrote {args.out}: {tasks} task lines")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
