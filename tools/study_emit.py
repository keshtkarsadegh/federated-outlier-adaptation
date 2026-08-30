"""
Rank the aggregation screen, and emit the full-horizon file it would feed.

    python tools/study_emit.py agg-full  --study NAME --root $FOA_STUDY_DIR \
        --out d01_p12.txt --expect 90
    python tools/study_emit.py agg-top3  --study NAME --root $FOA_STUDY_DIR \
        --out d01_agg_top3.json --expect 0

Reads JSON and writes text: a login-node job, no GPU.

Why ``--expect`` is mandatory
-----------------------------
A downstream array is written with a fixed ``--array`` range, often before this
runs. A generator that emitted a different number of lines than the range it
feeds would leave Slurm running array elements that point at nothing - or
silently skipping real ones. So every generator is told the count it must
produce and **exits non-zero when it does not**, which fails an ``afterok`` and
halts a chain instead of running garbage.

Boundary winners are flagged, not fatal
---------------------------------------
A winner sitting at the lowest or highest value of its grid row means the
optimum may be outside the range that was searched. That is worth knowing and
worth reporting, and it is not a reason to stop: the run is still the best of
what was tried. Boundary hits go to ``tables/BOUNDARY_HITS.txt`` and to stdout,
and the caller carries on.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import select_agg_screen as agg_selector
import select_reg_screen as reg_selector

from federated_outlier_adaptation.training import agg_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import config


def prefixes(cfg) -> Dict[str, str]:
    """
    The folder prefixes of one study's aggregation stages.

    A study writes into its own root, so these need only be unique within it -
    but they carry the study's tag anyway, because a folder that says which
    study it belongs to cannot be misread when two studies' results end up side
    by side in a table.
    """
    return {
        "agg_screen": f"{cfg.tag}_agg_",
        "agg_full": f"{cfg.tag}_aggfull_",
        "reg_screen": f"{cfg.tag}_reg_",
        "reg_full": f"{cfg.tag}_regfull_",
    }


def emit(lines: List[str], out: Path, expect: int, label: str, header: List[str]) -> int:
    """Write a task file, or fail loudly if it is not the promised size."""
    if expect <= 0:
        print(
            f"FATAL: {label} writes a task file and needs --expect. Without a "
            "count there is nothing to check the emission against, and the "
            "array that consumes it is written with a fixed range.",
            file=sys.stderr,
        )
        return 1
    body = [line for line in lines if line and not line.startswith("#")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header + body) + "\n")
    print(f"{label}: wrote {out} with {len(body)} task lines")
    if len(body) != expect:
        print(
            f"FATAL: {label} emitted {len(body)} lines but the array written for "
            f"it has {expect}. Halted rather than run against a range that does "
            "not match the file.",
            file=sys.stderr,
        )
        return 1
    return 0


def note_boundaries(root: Path, label: str, hits: List[str]) -> None:
    """Record grid-edge winners; never a reason to stop."""
    path = root / "tables" / "BOUNDARY_HITS.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        for hit in hits:
            handle.write(f"{label}\t{hit}\n")
    if hits:
        print(f"BOUNDARY: {len(hits)} winner(s) of {label} sit at a grid edge:")
        for hit in hits:
            print(f"  {hit}")
        print("  (reported, not fatal - the optimum may lie outside the range)")
    else:
        print(f"{label}: no winner sits at a grid edge.")


def numeric_boundary(winner: Dict[str, Any], siblings: List[Dict[str, Any]],
                     key: str) -> List[str]:
    """Which of a winner's numeric coefficients are at the end of their row."""
    hits = []
    values: Dict[str, List[float]] = {}
    for cell in siblings:
        for name, value in cell[key].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(name, []).append(float(value))
    for name, value in winner[key].items():
        row = sorted(set(values.get(name, [])))
        if len(row) < 2 or not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if float(value) == row[0]:
            hits.append(f"{winner['id']}: {name}={value:g} is the LOW end of {row}")
        elif float(value) == row[-1]:
            hits.append(f"{winner['id']}: {name}={value:g} is the HIGH end of {row}")
    return hits


#: How a top-three or a crowning is ordered.  ``val`` is the protocol letter -
#: the test rows are what the paper reports on, so selecting on them is choosing
#: the winner with the number that is supposed to judge it.  ``test`` is the
#: owner's documented departure from that, and every artefact it produces says
#: so and carries BOTH orderings, so a reader never has to infer which was used.
RANK_BY = ("val", "test")

RANK_RULES = {
    "val": "fold-mean VALIDATION adaptation, one cell per method",
    "test": (
        "owner decision A 2026-08-28: test-ranked among fully-reported "
        "finalists, one cell per method"
    ),
}


def rank_key(rank_by: str) -> str:
    """Which column a ranking sorts on."""
    if rank_by not in RANK_BY:
        raise SystemExit(f"Unknown --rank-by {rank_by!r}; expected one of {RANK_BY}")
    return "adaptation" if rank_by == "val" else "adaptation_test"


def ranked_by(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """
    ``rows`` best first, with ties broken by cell id.

    Ties are not hypothetical here - two aggregation methods came out at exactly
    0.9107 on the test column - and without an explicit rule their order would be
    whatever the input happened to be, which is not a property of the data. Score
    descending, then id ascending, so the ordering is reproducible from the
    numbers alone.
    """
    return sorted(rows, key=lambda r: (-r[key]["mean"], r["id"]))


def both_rankings(rows: List[Dict[str, Any]]) -> Dict[str, list]:
    """
    Every finalist ordered both ways, for the record.

    The two columns can disagree - that divergence is itself a finding - so an
    artefact that names one winner carries the other ordering beside it rather
    than leaving a reader to take the choice on trust.
    """
    out: Dict[str, list] = {}
    for name in RANK_BY:
        key = rank_key(name)
        scored = [r for r in rows if r[key]["mean"] is not None]
        out[name] = [
            {"id": r["id"], "adaptation": r["adaptation"]["mean"],
             "adaptation_test": r["adaptation_test"]["mean"]}
            for r in ranked_by(scored, key)
        ]
    return out


def check_measured(label: str, measured: int, total: int, allow: bool) -> bool:
    """
    Whether a selection rests on enough evidence to be worth emitting.

    A screen that failed leaves no results, and every winner then comes from the
    fallback - the first cell of each row, chosen by nothing. The task file that
    comes out of that is indistinguishable from a real one at a glance, which is
    exactly what makes it dangerous: it would run, produce numbers, and be a
    grid's arbitrary first element reported as a selection.

    So a selection over NOTHING is always refused, and a partial one is refused
    unless the caller says in so many words that it knows.

    Returns:
        True when the caller may proceed.
    """
    if measured == 0:
        print(
            f"FATAL: {label} measured NOTHING - every winner would be the "
            "fallback first cell of its row, chosen by no evidence at all. "
            "The screen has not produced results; nothing is emitted.",
            file=sys.stderr,
        )
        return False
    if measured < total and not allow:
        print(
            f"FATAL: {label} measured {measured} of {total} selections; the "
            f"other {total - measured} would fall back to a row's first cell. "
            "Re-run the missing screen elements, or pass --allow-unmeasured to "
            "emit anyway and accept that those lines were not selected.",
            file=sys.stderr,
        )
        return False
    if measured < total:
        print(
            f"WARNING: {label} measured {measured} of {total}; "
            f"{total - measured} winner(s) are fallbacks (--allow-unmeasured)."
        )
    return True


def write_table(payload: Any, root: Path, name: str) -> Path:
    path = root / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  selection -> {path}")
    return path


# --------------------------------------------------------------------------- #
# the screen's winners, and the full-horizon file they would feed
# --------------------------------------------------------------------------- #
def agg_full(cfg, root: Path, out: Path, expect: int, allow_unmeasured: bool = False) -> int:
    """
    Each aggregation method's best cell, as a full-horizon task file.

    No method filtering: every method in the table gets its best coefficients
    re-run, so the full stage compares eighteen methods rather than one winner
    against nothing.
    """
    cells = agg_cells.screen_cells()

    # Cells the boundary rule added after the screen ran. Without this they are
    # trained and then ignored, because the candidate list is fixed at import
    # time and cannot know which ranges a particular cohort will push against.
    extra = root / "tables" / "boundary_ext_cells.json"
    if extra.is_file():
        known = {c["id"] for c in cells}
        added = [c for c in json.loads(extra.read_text()) if c["id"] not in known]
        if added:
            print(f"  boundary extension: {len(added)} extra cell(s) considered")
            cells = cells + added

    rows = agg_selector.summarise(
        agg_selector.collect(root, cells, prefixes(cfg)["agg_screen"])
    )
    by_id = {cell["id"]: cell for cell in cells}
    by_method: Dict[tuple, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_method.setdefault((cell["path"], SL.agg_method_of(cell)), []).append(cell)

    winners, record, hits = [], {}, []
    for key in SL.AGG_METHODS:
        siblings = by_method[key]
        ids = {cell["id"] for cell in siblings}
        scored = [
            row for row in rows
            if row["id"] in ids and row["adaptation"]["mean"] is not None
        ]
        if not scored:
            print(f"  {key}: nothing measured; falling back to the first cell")
            best = siblings[0]
            record["/".join(key)] = {"winner": best["id"], "measured": False}
        else:
            top = max(scored, key=lambda r: r["adaptation"]["mean"])
            best = by_id[top["id"]]
            record["/".join(key)] = {
                "winner": best["id"], "measured": True,
                "adaptation": top["adaptation"]["mean"],
                "adaptation_sd": top["adaptation"]["sd"],
                "preservation": top["preservation"]["mean"],
                "considered": len(scored),
            }
            hits.extend(numeric_boundary(best, siblings, "flags"))
        winners.append(best)

    measured = sum(1 for entry in record.values() if entry.get("measured"))
    if not check_measured("agg-full", measured, len(record), allow_unmeasured):
        return 1

    write_table(record, root, "p11_agg_method_winners.json")
    note_boundaries(root, "p11/agg-full", hits)

    lines = []
    for index, cell in enumerate(winners):
        for fold in SL.folds_of(cfg):
            lines.append(SL.agg_line(
                cfg, cell, fold, SL.FULL_ROUNDS,
                cfg.seed_base + 4000 + index * 10 + fold,
            ))
    return emit(lines, out, expect, "p12/agg-full", [
        "# GENERATED, AND NOT YET AUTHORISED TO RUN.",
        "#",
        "# Each aggregation method's best cell from the 25-round screen, re-run",
        f"# at the full {SL.FULL_ROUNDS}-round horizon: {len(winners)} methods x {len(SL.folds_of(cfg))} folds.",
        "#",
        "# This is the P12 file. P11 is the screen; P12 is gated separately and",
        "# the owner has not authorised it. Do not submit this without that.",
        "#",
    ])


def _method_winner(cfg, root: Path, method: str):
    """The best cell of one method over the screen, and its siblings."""
    cells = agg_cells.screen_cells()
    siblings = [c for c in cells if SL.agg_method_of(c) == method]
    if not siblings:
        raise SystemExit(f"No screening cell belongs to method {method!r}.")
    ids = {c["id"] for c in siblings}
    rows = agg_selector.summarise(
        agg_selector.collect(root, cells, prefixes(cfg)["agg_screen"])
    )
    scored = [
        r for r in rows if r["id"] in ids and r["adaptation"]["mean"] is not None
    ]
    if not scored:
        raise SystemExit(f"Nothing measured for method {method!r}; cannot re-select.")
    top = max(scored, key=lambda r: r["adaptation"]["mean"])
    by_id = {c["id"]: c for c in cells}
    return by_id[top["id"]], siblings, top, len(scored)


def agg_trimmed_patch(cfg, root: Path, out: Path, expect: int) -> int:
    """
    Re-select the trimmed row after its boundary extension, and emit the delta.

    The extension widened one row, so at most one method's winner can move. When
    it does, the honest delta is that method's five lines - not a fresh
    ninety-line full stage, which would re-run seventeen methods whose inputs
    did not change.

    The previously selected cell is read from the record the last selection
    wrote, so the emitter can say which of the two cases this is rather than
    leaving the caller to compare by eye.
    """
    method = "trimmed"
    winner, siblings, top, considered = _method_winner(cfg, root, method)

    record_path = root / "tables" / "p11_agg_method_winners.json"
    previous = None
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        entry = record.get(f"concurrent/{method}") or {}
        previous = entry.get("winner")

    row = sorted(
        {c["flags"]["trim_frac"] for c in siblings if "trim_frac" in c["flags"]}
    )
    hits = numeric_boundary(winner, siblings, "flags")
    print(f"trimmed row now searched: {row}")
    print(f"  previous winner: {previous}")
    print(f"  winner now:      {winner['id']}  "
          f"(validation {top['adaptation']['mean']:.4f} of {considered} scored)")
    if previous is not None and previous == winner["id"]:
        print("  UNCHANGED - the extension confirmed the edge was the optimum.")
        print("  The patch below is identical to what P12 already carries; it is")
        print("  written anyway so the caller can diff it rather than trust this.")
    else:
        print("  CHANGED - the trimmed winner moved into the extended range.")
    note_boundaries(root, "p11ext/agg-trimmed", hits)

    write_table(
        {"method": method, "row": row, "previous_winner": previous,
         "winner": winner["id"], "changed": previous != winner["id"],
         "adaptation": top["adaptation"]["mean"],
         "adaptation_sd": top["adaptation"]["sd"],
         "preservation": top["preservation"]["mean"],
         "considered": considered, "boundary_hits": hits},
        root, "p11ext_trimmed_reselection.json",
    )

    # The trimmed method's index in the full-horizon ordering, so the patch's
    # seeds are the ones P12 would have used for that method.
    index = [k for k in SL.AGG_METHODS].index(("concurrent", method))
    lines = [
        SL.agg_line(cfg, winner, fold, SL.FULL_ROUNDS,
                    cfg.seed_base + 4000 + index * 10 + fold)
        for fold in SL.folds_of(cfg)
    ]
    return emit(lines, out, expect, "p12/trimmed-patch", [
        "# GENERATED, AND NOT AUTHORISED TO RUN - stamped exactly as P12 was.",
        "#",
        "# The trimmed row's re-selection after the boundary extension.",
        f"# row searched: {row}",
        f"# previous winner: {previous}",
        f"# winner now:      {winner['id']}",
        "#",
        "# ONLY THE TRIMMED METHOD. The extension widened one row, so at most one",
        "# method's winner can move; re-running the other seventeen would spend",
        "# GPU time reproducing results whose inputs did not change.",
        "#",
        f"# {len(SL.folds_of(cfg))} tasks at the full {SL.FULL_ROUNDS}-round horizon, carrying the same",
        "# seeds P12 uses for this method, so the patched lines and the ones they",
        "# replace are the same runs with a different coefficient.",
        "#",
    ])


def agg_top3(cfg, root: Path, out: Path, expect: int, rank_by: str = "val") -> int:
    """The three best aggregations per family, for a later combination stage."""
    key = rank_key(rank_by)
    cells = agg_cells.screen_cells()
    rows = agg_selector.summarise(
        agg_selector.collect(root, cells, prefixes(cfg)["agg_full"])
    )
    by_id = {cell["id"]: cell for cell in cells}
    chosen: Dict[str, List[str]] = {}
    orderings: Dict[str, Any] = {}
    for family in ("concurrent", "sequential"):
        scored = [
            row for row in rows
            if by_id[row["id"]]["path"] == family and row[key]["mean"] is not None
        ]
        ranked = ranked_by(scored, key)
        orderings[family] = both_rankings(scored)
        # One cell per METHOD, as the regularisation side already does. Two
        # things make this necessary rather than tidy. A patched method leaves
        # both its old and its new cell with full-horizon results - the
        # superseded one is still on disk - and without this the replacement and
        # the thing it replaced would take two of the three slots and be crossed
        # against each other. And even without a patch, a family's three slots
        # are meant to be three ideas about aggregation, not one rule at three
        # coefficients.
        seen, top = set(), []
        for row in ranked:
            method = SL.agg_method_of(by_id[row["id"]])
            if method in seen:
                continue
            seen.add(method)
            top.append(row["id"])
            if len(top) == SL.TOP_K:
                break
        chosen[family] = top
        if len(chosen[family]) < SL.TOP_K:
            print(
                f"FATAL: only {len(chosen[family])} scored aggregation METHOD(s) "
                f"in {family}; a combination stage needs {SL.TOP_K} distinct ones.",
                file=sys.stderr,
            )
            return 1
    table = write_table(
        {"rule": RANK_RULES[rank_by], "rank_by": rank_by,
         "source": prefixes(cfg)["agg_full"],
         "considered": {f: len(chosen[f]) for f in chosen},
         "rankings": orderings,
         "top": chosen}, root, "p12_agg_top3.json",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != table.resolve():
        out.write_text(json.dumps(chosen, indent=2) + "\n")
    for family, ids in chosen.items():
        print(f"  {family}: {ids}")
    print(f"agg-top3: wrote {out}")
    return 0


# --------------------------------------------------------------------------- #
# the regularisation screen's winners
# --------------------------------------------------------------------------- #
def reg_full(cfg, root: Path, out: Path, expect: int, allow_unmeasured: bool = False) -> int:
    """
    Each penalty's best cell **per family**, re-run at the full horizon.

    Per family, not per method: a penalty that stabilises a server average and
    one that stabilises a sequential walk are different claims, and collapsing
    them would let a method win the table on the strength of one schedule while
    being useless on the other.
    """
    cells = reg_cells.screen_cells()
    rows = reg_selector.summarise(
        reg_selector.collect(root, cells, prefixes(cfg)["reg_screen"])
    )
    by_id = {cell["id"]: cell for cell in cells}
    by_method = reg_cells.cells_by_method()

    winners, record, hits = [], {}, []
    for method in SL.REG_METHODS:
        siblings = by_method[method]
        ids = {cell["id"] for cell in siblings}
        for family in SL.FAMILIES:
            scored = [
                row for row in rows
                if row["id"] in ids and row["family"] == family
                and row["adaptation"]["mean"] is not None
            ]
            if not scored:
                print(f"  {method}/{family}: nothing measured; first cell used")
                best = siblings[0]
                record[f"{method}/{family}"] = {"winner": best["id"], "measured": False}
            else:
                top = max(scored, key=lambda r: r["adaptation"]["mean"])
                best = by_id[top["id"]]
                record[f"{method}/{family}"] = {
                    "winner": best["id"], "measured": True,
                    "adaptation": top["adaptation"]["mean"],
                    "preservation": top["preservation"]["mean"],
                    "considered": len(scored),
                }
                hits.extend(numeric_boundary(best, siblings, "hypers"))
            winners.append((method, family, best))

    measured = sum(1 for entry in record.values() if entry.get("measured"))
    if not check_measured("reg-full", measured, len(record), allow_unmeasured):
        return 1

    write_table(record, root, "p13_reg_method_winners.json")
    note_boundaries(root, "p13/reg-full", hits)

    lines = []
    for index, (method, family, cell) in enumerate(winners):
        for fold in SL.folds_of(cfg):
            lines.append(SL.reg_line(
                cfg, cell, fold, SL.FULL_ROUNDS,
                cfg.seed_base + 8000 + index * 10 + fold, family=family,
            ))
    return emit(lines, out, expect, "p14/reg-full", [
        "# GENERATED, AND NOT YET AUTHORISED TO RUN.",
        "#",
        "# Each penalty's best cell PER FAMILY from the 25-round screen, re-run",
        f"# at the full {SL.FULL_ROUNDS}-round horizon: {len(SL.REG_METHODS)} methods x"
        f" {len(SL.FAMILIES)} families x {len(SL.folds_of(cfg))} folds.",
        "#",
        "# THE FAMILY IS IN THE PARENT, and it has to be: the winners are chosen",
        "# per (method, family) and the same cell often wins on both schedules.",
        "# Without the tag two tasks would name one output folder, race each",
        "# other into it, and leave the selector unable to say which family's",
        "# result it had read. Each task still runs BOTH families; the tag",
        "# records which one the folder is read for.",
        "#",
        "# THERE IS NO ZERO-PENALTY ROW HERE, deliberately. The no-penalty cell",
        "# (param_l2 at mu=0) is an exact no-op, and P09 already runs precisely",
        "# that objective at the same horizon, on the same folds, with the same",
        "# sampler - so the reference exists and running it twice would price the",
        "# same model twice.",
        "#",
        "# This is the P14 file. P13 is the screen; P14 is gated separately and",
        "# the owner has not authorised it.",
        "#",
    ])


def hybrid_cells(root: Path) -> List[Dict[str, Any]]:
    """
    The kd+fisher blend cells, rebuilt from the record its emitter wrote.

    They are not in the screening table - the blend is priced after both of its
    halves are known - so anything that ranks full-horizon results has to be
    told they exist, or the blend could never take a slot it had earned.
    """
    record_path = root / "tables" / "p14_hybrid_construction.json"
    if not record_path.is_file():
        return []
    record = json.loads(record_path.read_text())
    return [
        reg_cells._cell(
            f"hybrid_mix{reg_cells._fmt(mix)}", "kd+fisher", "kd+fisher",
            f"kd+fisher blend, mix={mix:g} (KD from {record['kd_winner']}, "
            f"Fisher from {record['fisher_winner']})",
            needs_fisher=True,
            lam=record["lam"], T=record["T"], mix=mix,
        )
        for mix in record.get("mixes", reg_cells.HYBRID_MIXES)
    ]


def reg_top3(cfg, root: Path, out: Path, expect: int, rank_by: str = "val") -> int:
    """
    The three best penalties per family, for a later combination stage.

    Each family is read from **its own** folders: the full-horizon parents carry
    the family they were selected for, so reading a family's score out of the
    folder tagged for the other one would attribute a result to a selection that
    did not produce it.
    """
    key = rank_key(rank_by)
    cells = reg_cells.screen_cells() + hybrid_cells(root)
    by_id = {cell["id"]: cell for cell in cells}
    chosen: Dict[str, List[str]] = {}
    orderings: Dict[str, Any] = {}
    for family in SL.FAMILIES:
        # Two prefixes. The per-method winners carry the family they were
        # selected for, so they are read from the tagged folders. The hybrid was
        # emitted after that selection and is not per-family, so it lives under
        # the untagged prefix - and it has to compete, or the blend could never
        # take a slot it had earned. The id check inside collect() keeps the
        # untagged pass from swallowing the tagged folders.
        rows = reg_selector.summarise(
            reg_selector.collect(
                root, cells, f"{prefixes(cfg)['reg_full']}{family}_"
            )
        )
        rows += reg_selector.summarise(
            reg_selector.collect(root, cells, prefixes(cfg)["reg_full"])
        )
        scored = [
            row for row in rows
            if row["family"] == family and row[key]["mean"] is not None
        ]
        ranked = ranked_by(scored, key)
        orderings[family] = both_rankings(scored)
        # one cell per method: a family's three slots must be three methods,
        # not three settings of the same one.
        seen, top = set(), []
        for row in ranked:
            if row["method"] in seen:
                continue
            seen.add(row["method"])
            top.append(row["id"])
            if len(top) == SL.TOP_K:
                break
        chosen[family] = top
        if len(top) < SL.TOP_K:
            print(
                f"FATAL: only {len(top)} scored penalty method(s) in {family}; "
                f"a combination stage needs {SL.TOP_K}.",
                file=sys.stderr,
            )
            return 1
    table = write_table(
        {"rule": RANK_RULES[rank_by] + "; the hybrid competes for its own slot",
         "rank_by": rank_by,
         "sources": [f"{prefixes(cfg)['reg_full']}<family>_",
                     prefixes(cfg)["reg_full"]],
         "rankings": orderings,
         "top": chosen},
        root, "p14_reg_top3.json",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.resolve() != table.resolve():
        out.write_text(json.dumps(chosen, indent=2) + "\n")
    for family, ids in chosen.items():
        print(f"  {family}: {ids}")
    print(f"reg-top3: wrote {out}")
    return 0


def reg_patch(cfg, root: Path, out: Path, expect: int,
              method: str = "", family: str = "") -> int:
    """
    Re-select one (method, family) after its row's boundary extension.

    Only the rows that were widened can move a winner, so the honest delta is
    that pair's five lines - not a fresh seventy-line full stage, which would
    re-run thirteen pairs whose inputs did not change.

    The previously selected cell is read from the record the last selection
    wrote, so this says which of the two cases it is rather than leaving the
    caller to compare by eye. And the patch carries the **same seed its P14
    sibling used**, so a patched line and the line it replaces are the same run
    with a different coefficient.
    """
    if method not in SL.REG_METHODS:
        print(
            f"FATAL: --method {method!r} is not one of {SL.REG_METHODS}.",
            file=sys.stderr,
        )
        return 1
    if family not in SL.FAMILIES:
        print(
            f"FATAL: --family {family!r} is not one of {list(SL.FAMILIES)}.",
            file=sys.stderr,
        )
        return 1

    cells = reg_cells.screen_cells()
    siblings = reg_cells.cells_by_method()[method]
    ids = {cell["id"] for cell in siblings}
    rows = reg_selector.summarise(
        reg_selector.collect(root, cells, prefixes(cfg)["reg_screen"])
    )
    scored = [
        r for r in rows
        if r["id"] in ids and r["family"] == family
        and r["adaptation"]["mean"] is not None
    ]
    if not scored:
        print(
            f"FATAL: nothing measured for {method}/{family}; a selection over "
            "nothing would emit a grid's arbitrary first cell as a winner.",
            file=sys.stderr,
        )
        return 1

    top = max(scored, key=lambda r: r["adaptation"]["mean"])
    by_id = {cell["id"]: cell for cell in cells}
    winner = by_id[top["id"]]

    record_path = root / "tables" / "p13_reg_method_winners.json"
    previous = None
    if record_path.is_file():
        previous = (json.loads(record_path.read_text()).get(f"{method}/{family}")
                    or {}).get("winner")

    hits = numeric_boundary(winner, siblings, "hypers")
    print(f"{method}/{family}: {len(siblings)} cells now in the row")
    print(f"  previous winner: {previous}")
    print(f"  winner now:      {winner['id']}  "
          f"(validation {top['adaptation']['mean']:.4f} of {len(scored)} scored)")
    if previous is not None and previous == winner["id"]:
        print("  UNCHANGED - the extension confirmed the edge was the optimum.")
        print("  The patch below is what P14 already carries; it is written")
        print("  anyway so the caller can diff it rather than trust this.")
    else:
        print("  CHANGED - the winner moved into the extended range.")
    note_boundaries(root, f"p13ext/{method}-{family}", hits)

    write_table(
        {"method": method, "family": family, "cells_in_row": len(siblings),
         "previous_winner": previous, "winner": winner["id"],
         "changed": previous != winner["id"],
         "adaptation": top["adaptation"]["mean"],
         "adaptation_sd": top["adaptation"]["sd"],
         "preservation": top["preservation"]["mean"],
         "considered": len(scored), "boundary_hits": hits},
        root, f"p13ext_{method}_{family}_reselection.json",
    )

    # The pair's index in the full-horizon ordering, so the patch's seeds are
    # exactly the ones P14 used for it.
    index = SL.REG_METHODS.index(method) * len(SL.FAMILIES) + \
        list(SL.FAMILIES).index(family)
    lines = [
        SL.reg_line(cfg, winner, fold, SL.FULL_ROUNDS,
                    cfg.seed_base + 8000 + index * 10 + fold, family=family)
        for fold in SL.folds_of(cfg)
    ]
    return emit(lines, out, expect, f"p14/{method}-{family}-patch", [
        "# GENERATED, AND NOT AUTHORISED TO RUN - stamped exactly as P14 was.",
        "#",
        f"# The {method}/{family} re-selection after the boundary extension.",
        f"# previous winner: {previous}",
        f"# winner now:      {winner['id']}",
        "#",
        "# ONLY THIS (METHOD, FAMILY). The extension widened two rows, so only",
        "# the pairs drawing on them can move; re-running the other pairs would",
        "# spend GPU time reproducing results whose inputs did not change.",
        "#",
        f"# {len(SL.folds_of(cfg))} tasks at the full {SL.FULL_ROUNDS}-round horizon, carrying the same",
        "# seeds P14 uses for this pair, so the patched lines and the ones they",
        "# replace are the same runs with a different coefficient.",
        "#",
    ])


def reg_hybrid(cfg, root: Path, out: Path, expect: int) -> int:
    """
    The kd+fisher blend, priced after both of its halves are known.

    A blend of two penalties is only worth pricing once each one's own strength
    has been measured: screening it alongside the others would mean guessing the
    KD half's ``lam`` and ``T`` and the Fisher half's ``lam`` before any of them
    was known - a three-dimensional grid for a question that only becomes
    meaningful after the two one-dimensional ones are settled.

    ``mix`` weights the KD term against the Fisher term, so ``mix = 1`` would
    reproduce the KD winner exactly. That is what makes the three blend points
    readable as a line between the two methods rather than three unrelated runs.
    """
    record_path = root / "tables" / "p13_reg_method_winners.json"
    if not record_path.is_file():
        print(
            f"FATAL: no {record_path}; run the reg-full selection before the "
            "hybrid, which is built from its winners.",
            file=sys.stderr,
        )
        return 1
    record = json.loads(record_path.read_text())

    family = "concurrent"
    kd_id = (record.get(f"kd/{family}") or {}).get("winner")
    fisher_id = (record.get(f"fisher/{family}") or {}).get("winner")
    if not kd_id or not fisher_id:
        print(
            f"FATAL: the {family} selection has no kd ({kd_id!r}) and fisher "
            f"({fisher_id!r}) winner; the hybrid needs both.",
            file=sys.stderr,
        )
        return 1
    unmeasured = [
        name for name, entry in (("kd", record.get(f"kd/{family}") or {}),
                                 ("fisher", record.get(f"fisher/{family}") or {}))
        if not entry.get("measured")
    ]
    if unmeasured:
        print(
            f"FATAL: the {family} winner(s) for {unmeasured} are fallbacks, not "
            "measurements. A blend of two halves that were never measured is "
            "not a blend of anything; re-run the screen first.",
            file=sys.stderr,
        )
        return 1

    by_id = {cell["id"]: cell for cell in reg_cells.screen_cells()}
    kd, fisher = by_id[kd_id], by_id[fisher_id]
    print(f"hybrid from {kd_id} (KD half) and {fisher_id} (Fisher half)")

    lines = []
    for index, mix in enumerate(reg_cells.HYBRID_MIXES):
        cell = reg_cells._cell(
            f"hybrid_mix{reg_cells._fmt(mix)}", "kd+fisher", "kd+fisher",
            f"kd+fisher blend, mix={mix:g} (KD from {kd_id}, Fisher from {fisher_id})",
            needs_fisher=True,
            lam=kd["hypers"]["lam"], T=kd["hypers"]["T"], mix=mix,
        )
        for fold in SL.folds_of(cfg):
            lines.append(SL.hybrid_line(
                cfg, cell, fold, cfg.seed_base + 9000 + index * 10 + fold
            ))
    write_table(
        {"kd_winner": kd_id, "fisher_winner": fisher_id, "family": family,
         "mixes": list(reg_cells.HYBRID_MIXES),
         "lam": kd["hypers"]["lam"], "T": kd["hypers"]["T"]},
        root, "p14_hybrid_construction.json",
    )
    return emit(lines, out, expect, "p14/reg-hybrid", [
        "# GENERATED, AND NOT YET AUTHORISED TO RUN.",
        "#",
        "# The kd+fisher blend, built from the two winners the reg-full",
        "# selection named:",
        f"#   KD half     {kd_id}",
        f"#   Fisher half {fisher_id}",
        f"#   mixes       {', '.join(f'{m:g}' for m in reg_cells.HYBRID_MIXES)}",
        "#",
        "# The objective is lam * (mix * KD + (1 - mix) * Fisher), so mix = 1",
        "# would reproduce the KD winner exactly. That is what makes these three",
        "# points readable as a line between the two methods rather than three",
        "# unrelated runs - and why three is enough.",
        "#",
    ])


# --------------------------------------------------------------------------- #
# the cross, its winner, and the extreme cases
# --------------------------------------------------------------------------- #
def _top_lists(root: Path) -> tuple:
    """The two top-three artefacts, read rather than restated."""
    agg_path = root / "tables" / "p12_agg_top3.json"
    reg_path = root / "tables" / "p14_reg_top3.json"
    missing = [str(p) for p in (agg_path, reg_path) if not p.is_file()]
    if missing:
        raise SystemExit(
            f"FATAL: no {missing}; run agg-top3 and reg-top3 before the cross. "
            "The combination stage is defined by what they selected, and "
            "hard-coding the winners here would let the two disagree."
        )
    agg = json.loads(agg_path.read_text())["top"]
    reg = json.loads(reg_path.read_text())["top"]
    return agg, reg


def combos(cfg, root: Path, out: Path, expect: int) -> int:
    """
    The three best aggregations crossed with the three best penalties, per family.

    Two screens each answered half a question - what the server rule is worth
    with no penalty, and what the penalty is worth under plain FedAvg - and
    neither can say whether the two compose. That is the only question here, so
    it is a cross and not another sweep.

    The pairing never crosses families: a concurrent aggregation is crossed only
    with the penalties selected on the concurrent schedule. Pairing across would
    combine a rule chosen for one loop with a penalty chosen for another and
    report the result as a property of either.
    """
    agg_top, reg_top = _top_lists(root)
    aggs = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    regs = {cell["id"]: cell for cell in reg_cells.screen_cells() + hybrid_cells(root)}

    lines, index, chosen = [], 0, {}
    for family in SL.FAMILIES:
        for agg_id in agg_top.get(family, []):
            for reg_id in reg_top.get(family, []):
                if agg_id not in aggs:
                    raise SystemExit(f"FATAL: unknown aggregation {agg_id!r}.")
                if reg_id not in regs:
                    raise SystemExit(f"FATAL: unknown penalty {reg_id!r}.")
                if aggs[agg_id]["path"] != family:
                    raise SystemExit(
                        f"FATAL: {agg_id!r} is a {aggs[agg_id]['path']} rule and "
                        f"was selected for {family}; the two lists disagree."
                    )
                chosen.setdefault(family, []).append((agg_id, reg_id))
                for fold in SL.folds_of(cfg):
                    lines.append(SL.combo_line(
                        cfg, aggs[agg_id], regs[reg_id], fold,
                        cfg.seed_base + 10000 + index * 10 + fold,
                    ))
                index += 1

    write_table(
        {"rule": "top-3 aggregations x top-3 penalties, within each family",
         "aggregations": agg_top, "penalties": reg_top,
         "pairs": {f: [list(p) for p in v] for f, v in chosen.items()}},
        root, "p15_combination_grid.json",
    )
    return emit(lines, out, expect, "p15/combos", [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        "# The three best aggregations crossed with the three best penalties,",
        "# per family, at the full horizon. Two screens each answered half a",
        "# question and neither can say whether the two compose; that is the",
        "# only question here.",
        "#",
        "# ONE COMBINATION IS ONE FAMILY. Each line names a single aggregation",
        "# rule, which lives in exactly one family, so a task produces one",
        "# result. The pairing never crosses schedules: a concurrent rule is",
        "# crossed only with penalties selected on the concurrent loop.",
        "#",
        f"# aggregations: {agg_top}",
        f"# penalties:    {reg_top}",
        "#",
        f"# {SL.COMBO_TOP_K} x {SL.COMBO_TOP_K} x {len(SL.FAMILIES)} families x"
        f" {len(SL.folds_of(cfg))} folds = {len(lines)} tasks.",
        "#",
    ])


def stage_winner(cfg, root: Path, out: Path, expect: int,
                 rank_by: str = "test") -> int:
    """
    Crown the combination with the best fold-mean validation adaptation.

    Both columns are recorded, not just the one that decided: a winner chosen on
    adaptation has spent something on preservation, and a table that reports
    only the criterion hides the trade it made.
    """
    agg_top, reg_top = _top_lists(root)
    pseudo = [
        {"id": f"{agg_id}_{reg_id}", "path": family, "rule": agg_id,
         "flags": {}, "note": f"{agg_id} x {reg_id}"}
        for family in SL.FAMILIES
        for agg_id in agg_top.get(family, [])
        for reg_id in reg_top.get(family, [])
    ]
    rows = agg_selector.summarise(
        agg_selector.collect(root, pseudo, f"{cfg.tag}_combo_")
    )
    key = rank_key(rank_by)
    scored = [r for r in rows if r[key]["mean"] is not None]
    if not scored:
        print(
            "FATAL: no combination produced a score; there is nothing to crown.",
            file=sys.stderr,
        )
        return 1

    best = ranked_by(scored, key)[0]
    by_id = {cell["id"]: cell for cell in pseudo}
    family = by_id[best["id"]]["path"]
    agg_id, reg_id = _split_combo(best["id"], agg_top, reg_top, family)

    ranked = ranked_by(scored, key)
    path = write_table(
        {"rule": RANK_RULES[rank_by] + " on the clients",
         "rank_by": rank_by,
         "rankings": both_rankings(scored),
         "winner": best["id"], "family": family,
         "aggregation": agg_id, "regulariser": reg_id,
         "adaptation": best["adaptation"]["mean"],
         "adaptation_sd": best["adaptation"]["sd"],
         "adaptation_test": best["adaptation_test"]["mean"],
         "preservation": best["preservation"]["mean"],
         "preservation_sd": best["preservation"]["sd"],
         "considered": len(scored),
         "ranked": [
             {"id": r["id"], "adaptation": r["adaptation"]["mean"],
              "preservation": r["preservation"]["mean"]}
             for r in ranked
         ]},
        root, "p15_stage_winner.json",
    )
    print(f"winner: {best['id']} ({family})")
    print(f"  aggregation {agg_id}   penalty {reg_id}")
    print(f"  ranked by    {rank_by} ({RANK_RULES[rank_by]})")
    print(f"  adaptation   val {best['adaptation']['mean']:.4f} "
          f"test {best['adaptation_test']['mean']:.4f}")
    print(f"  preservation {best['preservation']['mean']:.4f} "
          f"(recorded, not the criterion)")
    print(f"  of {len(scored)} combinations scored")
    if out.resolve() != path.resolve():
        out.write_text(path.read_text())
    return 0


def _split_combo(combo_id: str, agg_top, reg_top, family: str) -> tuple:
    """Which aggregation and which penalty a combination id names."""
    for agg_id in agg_top.get(family, []):
        for reg_id in reg_top.get(family, []):
            if combo_id == f"{agg_id}_{reg_id}":
                return agg_id, reg_id
    raise SystemExit(f"FATAL: cannot split combination id {combo_id!r}.")


def extreme(cfg, root: Path, out: Path, expect: int) -> int:
    """
    The winning combination on one- and two-client federations.

    ``double`` and ``dual`` hold precisely the same rows and differ only in
    whether the aggregation ever sees them separately, so any gap between them
    is what client boundaries cost with the data held constant - which the
    ten-client stages cannot measure, because there the boundaries and the data
    always move together.
    """
    from federated_outlier_adaptation.training.extreme_cells import (
        extreme_cells,
        rank_cohort,
    )

    winner_path = root / "tables" / "p15_stage_winner.json"
    if not winner_path.is_file():
        print(
            f"FATAL: no {winner_path}; crown the combination stage before "
            "running its winner on the extreme cases.",
            file=sys.stderr,
        )
        return 1
    winner = json.loads(winner_path.read_text())
    aggs = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    regs = {cell["id"]: cell for cell in reg_cells.screen_cells() + hybrid_cells(root)}
    agg = aggs[winner["aggregation"]]
    reg = regs[winner["regulariser"]]

    # The worst cohort writers under THIS study's g-0, from the per-fold
    # evaluations that stage already wrote. Not the coarse detector's scores:
    # those rank writers by a model common to every study, and the extreme cases
    # are meant to be this study's own hardest clients under its shipped model.
    # A writer's baseline is the mean of its per-fold accuracies, so no single
    # fold's draw decides the ranking.
    cohort_path = root / "outliers" / cfg.cohort_file_name
    cohort = json.loads(cohort_path.read_text())
    cohort = cohort["clients"] if isinstance(cohort, dict) else list(cohort)

    per_fold_path = root / "g0_perfold_evaluations.json"
    if not per_fold_path.is_file():
        print(f"FATAL: no {per_fold_path}; the ranking comes from it.",
              file=sys.stderr)
        return 1
    gathered: Dict[str, List[float]] = {}
    for record in json.loads(per_fold_path.read_text()).values():
        for writer, value in (record.get("per_writer") or {}).items():
            gathered.setdefault(writer, []).append(float(value))
    accuracies = {w: sum(v) / len(v) for w, v in gathered.items() if v}
    missing = [w for w in cohort if w not in accuracies]
    if missing:
        print(f"FATAL: g-0 scores no fold for {missing}.", file=sys.stderr)
        return 1
    write_table(
        {"rule": "mean of g-0's per-fold accuracy on each cohort writer's test rows",
         "accuracies": {w: accuracies[w] for w in cohort}},
        root, "clients_acc_on_g0.json",
    )

    cells = extreme_cells(rank_cohort(accuracies, cohort))
    lines = []
    for index, cell in enumerate(cells):
        listing = root / "outliers" / f"extreme_{cell['case']}.json"
        listing.write_text(json.dumps(cell["clients"], indent=2) + "\n")
        print(f"  {cell['case']}: {cell['clients']} -> {listing}")
        clients_file = f"{SL.POOLS}/extreme_{cell['case']}.json"
        for fold in SL.folds_of(cfg):
            lines.append(SL.extreme_line(
                cfg, cell["case"], clients_file, agg, reg, fold,
                cfg.seed_base + 11000 + index * 10 + fold,
            ))
    return emit(lines, out, expect, "p16/extreme", [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        f"# The winning combination only: {winner['aggregation']} x {winner['regulariser']}",
        f"# ({winner['family']}), crowned on validation adaptation.",
        "#",
        "# Three arrangements of the same two writers - the cohort's worst two",
        "# under THIS study's g-0, not the coarse detector's scores:",
        "#   single  one client: the worst writer",
        "#   double  two clients: the two worst",
        "#   dual    ONE client holding both of their rows merged",
        "#",
        "# double and dual hold precisely the same rows and differ only in",
        "# whether the aggregation ever sees them separately. Any gap between",
        "# them is what client BOUNDARIES cost with the data held constant - a",
        "# question the ten-client stages cannot ask, because there the",
        "# boundaries and the data always move together.",
        "#",
        "# FULL PARTICIPATION. Dropping a client from a two-client federation is",
        "# not a participation study, it is a coin flip on whether the round",
        "# happens.",
        "#",
        f"# {len(cells)} cases x {len(SL.folds_of(cfg))} folds = {len(lines)} tasks.",
        "#",
    ])


#: The size and dropout stages, as data. Each varies the federation and nothing
#: else; the protocol comes from ``cohort_line`` unchanged. Participation is
#: always ``study_config.participants``, never a number written here, so a stage
#: cannot come to disagree with the rule about what its own dropout means.
#: The three selected configurations, without the plain-FedAvg control. The
#: twenty-client point runs these only: the control's job is to say whether a
#: change is the method or the participation, and the ten-client pair (9-of-10
#: against 8-of-10) already answers that with the control included. Paying for
#: it again at a size where the question is "do the winners hold" would buy
#: nothing the smaller point has not already bought.
WINNERS = ("winner", "balanced", "sequential")

SIZE_STAGES = {
    "five": {
        "tag": "five", "clients": 5, "cohort": "worst5", "book": None,
        "dropout": 0.1, "seed_block": 12000, "record": "p18_five_client.json",
        "what": "the selected configurations at five clients",
    },
    "drop20": {
        "tag": "drop20", "clients": 10, "cohort": "study", "book": None,
        "dropout": 0.2, "seed_block": 14000, "record": "p19_dropout20.json",
        "what": "the selected configurations at twenty percent dropout",
    },
    "c20d10": {
        "tag": "c20d10", "clients": 20, "cohort": "worst20", "book": "cohort20",
        "dropout": 0.1, "seed_block": 15000, "record": "p20_c20_d10.json",
        "only": WINNERS,
        "what": "the selected configurations at twenty clients, ten percent dropout",
    },
    "c20d20": {
        "tag": "c20d20", "clients": 20, "cohort": "worst20", "book": "cohort20",
        "dropout": 0.2, "seed_block": 16000, "record": "p20_c20_d20.json",
        "only": WINNERS,
        "what": "the selected configurations at twenty clients, twenty percent dropout",
    },
}

#: Generators that emit more than one stage into a single task file.
STAGE_GROUPS = {"c20": ("c20d10", "c20d20")}


def worst20_listing(cfg, root: Path):
    """
    The cohort's worst twenty, and the check that it extends the worst ten.

    A larger cohort has to be the smaller one plus the next writers down the
    same ranking, or the twenty-client point is not the ten-client point scaled
    up - it is a different population, and every comparison drawn across the two
    sizes would be measuring that instead. Same ranking, same eligibility rule,
    so the first ten must come back identical; if they do not, something moved
    between the two cuts and it has to stop here.
    """
    listing = root / "outliers" / "cohort_worst20.json"
    if not listing.is_file():
        print(f"FATAL: no {listing}; run the P20 setup chain before the runs.",
              file=sys.stderr)
        return None
    payload = json.loads(listing.read_text())
    clients = payload["clients"] if isinstance(payload, dict) else list(payload)
    if len(clients) != 20:
        print(f"FATAL: {listing.name} holds {len(clients)} clients, not 20.",
              file=sys.stderr)
        return None

    ten = json.loads((root / "outliers" / cfg.cohort_file_name).read_text())
    ten = ten["clients"] if isinstance(ten, dict) else list(ten)
    if clients[:10] != ten:
        print(
            f"FATAL: the first ten of {listing.name} are {clients[:10]} but "
            f"{cfg.cohort_file_name} holds {ten}. The twenty-client cohort does "
            "not extend the ten-client one, so the two sizes are different "
            "populations and nothing across them would be comparable.",
            file=sys.stderr,
        )
        return None
    print(f"  worst-20 extends worst-10; next ten: {clients[10:]}")
    return clients, listing


def worst_five_listing(cfg, root: Path):
    """
    The cohort's worst five, derived from the g-0 ranking and cross-checked.

    Returns ``(clients, accuracies, path)``, or ``None`` when the artefacts
    disagree - which is fatal, because it would mean the writers of a size stage
    are not the writers every other stage federates, and nothing downstream
    would say so.
    """
    from federated_outlier_adaptation.training import five_cells
    from federated_outlier_adaptation.training.extreme_cells import accuracy_map

    ranking_path = root / "outliers" / "bad_acc_on_g0.json"
    scores_path = root / "outliers" / "bad_scores_on_g0.json"
    cohort_path = root / "outliers" / cfg.cohort_file_name
    for path in (ranking_path, scores_path, cohort_path):
        if not path.is_file():
            print(f"FATAL: no {path}; the five-client cohort is derived from it.",
                  file=sys.stderr)
            return None

    accuracies = accuracy_map(json.loads(ranking_path.read_text()))
    derived = sorted(accuracies, key=lambda writer: (accuracies[writer], writer))
    stated = json.loads(scores_path.read_text()).get("ranking") or []
    cohort = json.loads(cohort_path.read_text())
    cohort = cohort["clients"] if isinstance(cohort, dict) else list(cohort)

    size = five_cells.COHORT_SIZE
    chosen = five_cells.worst_five(derived, size)
    for label, other in (("bad_scores_on_g0.json", stated),
                         (cfg.cohort_file_name, cohort)):
        if list(other[:size]) != chosen:
            print(
                f"FATAL: the worst-{size} derived from the accuracies is "
                f"{chosen}, but {label} starts {list(other[:size])}. The "
                "artefacts disagree about the hardest writers; nothing emitted.",
                file=sys.stderr,
            )
            return None

    listing = root / "outliers" / "cohort_worst5.json"
    listing.write_text(json.dumps(
        {"rule": f"the worst {size} of the g-0 ranking the ten-client cohort was cut from",
         "tag": f"{cfg.tag}_cohort_worst{size}", "k": size, "size": size,
         "clients": chosen,
         "accuracies": {writer: accuracies[writer] for writer in chosen},
         "source": ranking_path.name}, indent=2) + "\n")
    print(f"  worst-{size}: {chosen} -> {listing}")
    return chosen, accuracies, listing


def stage_lines(cfg, root: Path, stage: str):
    """
    One stage's task lines and the header block that explains them.

    Every method comparison this study made was made on ten clients with nine
    drawn per round. These stages move the federation and nothing else, so a
    difference is about the participation rather than about the protocol.

    Returns:
        ``(lines, header, recorded)``, or ``None`` when something is wrong
        enough that no file should be written.
    """
    from federated_outlier_adaptation.data.fold_book import FoldBook
    from federated_outlier_adaptation.training import five_cells

    spec = SIZE_STAGES[stage]
    clients_file, chosen, book_name = None, None, spec["book"]
    if spec["cohort"] == "worst5":
        found = worst_five_listing(cfg, root)
        if found is None:
            return None
        chosen, _, listing = found
        clients_file = f"{SL.POOLS}/{listing.name}"
    elif spec["cohort"] == "worst20":
        found = worst20_listing(cfg, root)
        if found is None:
            return None
        chosen, listing = found
        clients_file = f"{SL.POOLS}/{listing.name}"
    else:
        cohort = json.loads((root / "outliers" / cfg.cohort_file_name).read_text())
        chosen = cohort["clients"] if isinstance(cohort, dict) else list(cohort)
        if len(chosen) != spec["clients"]:
            print(
                f"FATAL: {cfg.cohort_file_name} holds {len(chosen)} clients and "
                f"this stage federates {spec['clients']}.", file=sys.stderr,
            )
            return None
    if len(chosen) != spec["clients"]:
        print(f"FATAL: {stage} federates {spec['clients']} clients but its "
              f"list holds {len(chosen)}.", file=sys.stderr)
        return None

    # The ten-client book unless the cohort holds writers it never covered. A
    # book re-cut for writers it DOES cover would give them different rows than
    # the ten-client runs did.
    book_flag = None
    if book_name is None:
        book_path = Path(str(SL.cohort_book(cfg)).replace(SL.ROOT, str(root)))
    else:
        book_path = root / "fold_books" / f"{book_name}.foldbook.npz"
        book_flag = f"{SL.BOOKS}/{book_name}.foldbook.npz"
    if not book_path.is_file():
        print(f"FATAL: no {book_path}; these runs read it.", file=sys.stderr)
        return None
    book = FoldBook.load(str(book_path))
    uncovered = [writer for writer in chosen if not book.covers(writer)]
    if uncovered:
        print(
            f"FATAL: {book_path.name} has no rows for {uncovered}; a cohort "
            "cannot be federated on a book that never split it.",
            file=sys.stderr,
        )
        return None

    # The participation rule, applied rather than restated.
    drawn = five_cells.clients_per_round(spec["clients"], spec["dropout"])
    rule_text = "floor((1 - %g) * %d) = %d" % (
        spec["dropout"], spec["clients"], drawn)

    aggs = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    regs = {cell["id"]: cell for cell in reg_cells.screen_cells() + hybrid_cells(root)}
    grid_path = root / "tables" / "p15_combination_grid.json"
    grid = json.loads(grid_path.read_text())["pairs"] if grid_path.is_file() else {}

    wanted = spec.get("only")
    cells = [c for c in five_cells.configs()
             if wanted is None or c["id"] in wanted]
    if wanted is not None and len(cells) != len(wanted):
        print(f"FATAL: {stage} asks for {list(wanted)} and the table offers "
              f"{[c['id'] for c in cells]}.", file=sys.stderr)
        return None

    lines, recorded = [], []
    for index, cell in enumerate(cells):
        agg = reg = None
        if cell["regulariser"] is not None:
            if cell["aggregation"] not in aggs:
                print(f"FATAL: unknown aggregation {cell['aggregation']!r}.",
                      file=sys.stderr)
                return None
            if cell["regulariser"] not in regs:
                print(f"FATAL: unknown penalty {cell['regulariser']!r}.",
                      file=sys.stderr)
                return None
            agg, reg = aggs[cell["aggregation"]], regs[cell["regulariser"]]
            if agg["path"] != cell["family"]:
                print(
                    f"FATAL: {cell['aggregation']!r} is a {agg['path']} rule and "
                    f"this stage runs it as {cell['family']}.", file=sys.stderr,
                )
                return None
            pair = [cell["aggregation"], cell["regulariser"]]
            if grid and pair not in [list(p) for p in grid.get(cell["family"], [])]:
                print(
                    f"FATAL: {pair} is not a pair the cross ran; a configuration "
                    "carried forward here must be one that stage measured.",
                    file=sys.stderr,
                )
                return None
        survivors = (
            five_cells.trim_survivors(agg["flags"]["trim_frac"], drawn)
            if agg is not None and "trim_frac" in agg["flags"] else None
        )
        recorded.append({
            "id": cell["id"], "family": cell["family"],
            "aggregation": cell["aggregation"], "regulariser": cell["regulariser"],
            "note": cell["note"], "trim_survivors": survivors,
        })
        for fold in SL.folds_of(cfg):
            lines.append(SL.cohort_line(
                cfg, cell, clients_file, agg, reg, fold,
                cfg.seed_base + spec["seed_block"] + index * 10 + fold,
                spec["tag"], drawn, book_flag,
            ))

    write_table(
        {"rule": ("the panel's selected configurations, plus an unmodified "
                  "control, with the federation varied and nothing else"),
         "stage": stage, "clients": chosen, "n_clients": spec["clients"],
         "dropout": spec["dropout"], "clients_per_round": drawn,
         "participation_rule": rule_text,
         "fold_book": book_path.name, "rounds": SL.FULL_ROUNDS,
         "configurations": recorded},
        root, spec["record"],
    )

    ten = cfg.clients_per_round
    trim_note = ["#   f      K=%d survivors   K=%d survivors" % (ten, drawn)] + [
        "#   %-5g  %-14s  %s" % (
            fraction,
            "%d (trim %d)" % (five_cells.trim_survivors(fraction, ten),
                              int(fraction * ten)),
            "%d (trim %d)" % (five_cells.trim_survivors(fraction, drawn),
                              int(fraction * drawn)),
        )
        for fraction in (0.1, 0.2, 0.3, 0.4)
    ]
    header = [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        f"# {spec['what'].upper()}.",
        "#",
        "# Every method comparison in this study was made on ten clients with",
        "# nine drawn per round. This holds the methods, the horizon, the",
        "# shipped model, the anchor, the teacher and the fold book fixed, and",
        "# moves the participation.",
        "#",
        f"# cohort ({len(chosen)}): {chosen}",
        "#",
    ] + ([
        "# THE BOOK IS THE TEN-CLIENT ONE. The cohort is narrowed by the client",
        "# list alone. A book re-cut for writers that book already covers would",
        "# give them different rows than the ten-client runs did, and the",
        "# comparison would be over two splits rather than over two",
        "# federations.",
        "#",
    ] if book_name is None else [
        f"# ITS OWN BOOK: {book_path.name}. A larger cohort has no choice - the",
        "# ten-client book holds no rows for writers it never covered - so this",
        "# one is cut by the SAME rule at the SAME seed (per-writer stratified",
        "# 60/20/20 over 5 folds, seed 42), and its first ten writers are the",
        "# ten-client cohort exactly. The setup chain refuses to go on if they",
        "# are not.",
        "#",
    ]) + [
        f"# PARTICIPATION {drawn} OF {spec['clients']}, at {spec['dropout']:.0%} dropout.",
        f"#   {rule_text}",
        "#   The same rule gives the 9 of 10 every other stage of this study",
        "#   runs and the 16 of 20 of the earlier programme, so this is the",
        "#   study's own participation applied at another rate rather than a",
        "#   number chosen for this stage. The extreme cases are the one",
        "#   documented exception: at one or two clients the rule leaves",
        "#   nobody, and dropping a client from a two-client federation is a",
        "#   coin flip on whether the round happens, so they run at full",
        "#   participation instead.",
        "#",
        f"# WHAT A TRIMMED MEAN DOES AT {drawn} PARTICIPANTS:",
    ] + trim_note
    if any(r["trim_survivors"] == 1 for r in recorded):
        header += [
            "#   f=0.4 leaves ONE surviving update at this size: the rule is a",
            "#   coordinate-wise median of the round, not an average of",
            "#   anything. That is a real change in what the method is and it",
            "#   belongs beside the number it produces.",
        ]
    header += ["#"] + ([
        "# The control is plain FedAvg with no penalty, both schedules in one",
        "# task. Without it a change here could not be attributed: every",
        "# configuration might move together simply because the participation",
        "# did.",
    ] if any(r["id"] == "control" for r in recorded) else [
        "# WINNERS ONLY - no plain-FedAvg control. The control's job is to say",
        "# whether a change is the method or the participation, and the",
        "# ten-client pair (9 of 10 against 8 of 10) already answers that with",
        "# the control included. The question here is whether the winners hold",
        "# at a larger size, and paying for the control again would buy nothing",
        "# that pair has not already bought.",
    ]) + [
        "#",
        f"# {len(recorded)} configurations x {len(SL.folds_of(cfg))} folds = {len(lines)} tasks,",
        f"# {SL.FULL_ROUNDS} rounds each.",
        "#",
    ]
    return lines, header, recorded


def size_stage(cfg, root: Path, out: Path, expect: int, stage: str) -> int:
    """One or several stages into one task file."""
    names = STAGE_GROUPS.get(stage, (stage,))
    lines, header = [], ["# GENERATED, AND NOT AUTHORISED TO RUN."]
    for name in names:
        made = stage_lines(cfg, root, name)
        if made is None:
            return 1
        stage_body, stage_header, _ = made
        lines += stage_body
        header += stage_header[1:]   # its own banner line is already carried
    return emit(lines, out, expect, f"{stage}/size", header)


def five(cfg, root: Path, out: Path, expect: int) -> int:
    """The selected configurations at five clients, four drawn per round."""
    return size_stage(cfg, root, out, expect, "five")


def c20(cfg, root: Path, out: Path, expect: int) -> int:
    """
    The twenty-client point, at both dropout levels, in one file.

    Both levels together because they share a cohort, a book and a setup chain,
    and because the pair is the measurement: one dropout level at a new size
    says nothing the ten-client runs did not already say.
    """
    return size_stage(cfg, root, out, expect, "c20")


def drop20(cfg, root: Path, out: Path, expect: int) -> int:
    """The full cohort at twenty percent dropout: eight of ten per round."""
    return size_stage(cfg, root, out, expect, "drop20")


WHAT = {
    "agg-full": agg_full,
    "combos": combos,
    "stage-winner": stage_winner,
    "extreme": extreme,
    "five": five,
    "drop20": drop20,
    "c20": c20,
    "agg-top3": agg_top3,
    "agg-trimmed-patch": agg_trimmed_patch,
    "reg-full": reg_full,
    "reg-patch": reg_patch,
    "reg-top3": reg_top3,
    "reg-hybrid": reg_hybrid,
}


# --------------------------------------------------------------------------- #
# what each generator reads, what it writes, and the ordering that follows
# --------------------------------------------------------------------------- #
#: Not descended into when working out a generator's inputs.  ``write_table``
#: names its file through a parameter, so its own ``root / "tables" / name`` is
#: about nothing in particular; ``note_boundaries`` appends to a LOG - nothing
#: selects from BOUNDARY_HITS.txt and every generator touches it, so counting it
#: as an input would make every generator look like a reader of a file no
#: generator writes.
_IO_OPAQUE = frozenset({"write_table", "note_boundaries"})

_MODULE = ast.parse(Path(__file__).resolve().read_text())
_DEFS = {node.name: node for node in _MODULE.body
         if isinstance(node, ast.FunctionDef)}


def _reachable(name: str, seen: Optional[set] = None) -> set:
    """Every module-level function ``name`` can reach, itself included."""
    seen = set() if seen is None else seen
    if name in seen or name in _IO_OPAQUE or name not in _DEFS:
        return seen
    seen.add(name)
    for node in ast.walk(_DEFS[name]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _reachable(node.func.id, seen)
    return seen


def _tables_literal(node: ast.AST) -> Optional[str]:
    """The literal ``NAME`` of a ``root / "tables" / "NAME"``, else None."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
        return None
    left, right = node.left, node.right
    if not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
        return None
    if not (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div)
            and isinstance(left.right, ast.Constant)
            and left.right.value == "tables"):
        return None
    return right.value


def table_io(what: str) -> Dict[str, Tuple[str, ...]]:
    """
    Which ``tables/`` artefacts a generator reads, and which it writes.

    Read off this module's own syntax tree rather than declared beside it. A
    declaration would be a second place to be wrong, and going stale is exactly
    how it would fail - quietly, leaving the ordering check below asserting
    something true about a table nobody updated. The source cannot drift from
    itself.

    A name reached through an f-string is not seen: the patch emitters write
    ``p13ext_{method}_{family}_reselection.json`` and nothing reads it, so the
    blind spot costs nothing today. If some later generator ever selects from a
    computed name, this stops covering it - hence the test that asserts the
    derived writes account for every ``write_table`` call in the file.
    """
    writes, touched = set(), set()
    for name in _reachable(WHAT[what].__name__):
        for node in ast.walk(_DEFS[name]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "write_table" and len(node.args) >= 3
                    and isinstance(node.args[2], ast.Constant)
                    and isinstance(node.args[2].value, str)):
                writes.add(node.args[2].value)
            literal = _tables_literal(node)
            if literal is not None:
                touched.add(literal)
    return {"reads": tuple(sorted(touched - writes)),
            "writes": tuple(sorted(writes))}


def step_what(line: str) -> Optional[str]:
    """Which generator a task line invokes, or None when it invokes none."""
    try:
        words = shlex.split(line)
    except ValueError:
        return None
    for previous, word in zip(words, words[1:]):
        if previous.endswith("study_emit.py"):
            return word if word in WHAT else None
    return None


def ordering_violations(steps: Sequence[str]) -> List[str]:
    """
    Consumers packed ahead of their producers, within ONE stage's task file.

    A stage is an array and an array runs by index, so a step that reads what a
    LATER step of the same file writes is not merely untidy: element 1 runs
    first and dies on an artefact element 2 has not written yet. That is how
    ``gen_reg`` died - the hybrid emitter, which is built from the reg-full
    winners, was packed at element 1 and the selection that produces them at
    element 2, under ``%1``. The selection completed; the hybrid had already
    failed, and every dependent stage went with it.

    Only steps of the same stage are compared. Across stages the chain's
    ``afterok`` edges are what order things, and those are checked elsewhere.
    """
    kinds = [step_what(line) for line in steps]
    io = {what: table_io(what) for what in kinds if what is not None}
    problems = []
    for index, what in enumerate(kinds):
        if what is None:
            continue
        for later in range(index + 1, len(kinds)):
            other = kinds[later]
            if other is None:
                continue
            for name in sorted(set(io[what]["reads"]) & set(io[other]["writes"])):
                problems.append(
                    f"element {index + 1} ({what}) reads {name}, which element "
                    f"{later + 1} ({other}) writes: the consumer is packed "
                    f"ahead of its producer and would run first."
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=sorted(WHAT))
    parser.add_argument("--study", default="Digits_study01")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect", type=int, default=0,
        help=(
            "How many task lines the generator must emit. Required for the "
            "generators that write a task file, because a downstream array is "
            "written with a fixed range and a mismatch would point it at lines "
            "that are not there. The selection-only generators (agg-top3, "
            "reg-top3) write a json and ignore it."
        ),
    )
    parser.add_argument(
        "--allow-unmeasured",
        action="store_true",
        help=(
            "Emit even though some selections fell back to a row's first cell "
            "because the screen has no result for them. Off by default: a "
            "selection over nothing produces a task file that looks exactly "
            "like a real one."
        ),
    )
    parser.add_argument(
        "--rank-by", choices=RANK_BY, default=None,
        help=(
            "Which column a top-three or a crowning is ordered on. 'val' is the "
            "protocol letter; 'test' is the owner's documented departure. Either "
            "way the artefact records which was used and carries BOTH orderings."
        ),
    )
    parser.add_argument(
        "--method", default="",
        help="reg-patch only: which penalty's row to re-select.",
    )
    parser.add_argument(
        "--family", default="",
        help="reg-patch only: which schedule's selection to re-emit.",
    )
    parser.add_argument(
        "--clients-per-round", type=int, default=None, metavar="M",
        help=(
            "Participation the emitted lines run at. MUST MATCH THE SCREEN THEY "
            "WERE SELECTED FROM: the grid is searched once, at the harder rate, "
            "and the winners are carried to the other rates. Emitting the "
            "finals at the study's default while the screen ran at another rate "
            "re-runs the winners under conditions they were not chosen under, "
            "and nothing about the resulting file looks wrong."
        ),
    )
    args = parser.parse_args()
    generator = WHAT[args.what]

    # inspect.signature, NOT __code__.co_varnames: co_varnames lists every local
    # name in the function body, so a generator with a `for family in ...` loop
    # looked like it took a `family` argument and was handed one. That is what
    # broke agg-top3.
    accepted = set(inspect.signature(generator).parameters)
    kwargs = {
        name: getattr(args, name)
        for name in ("allow_unmeasured", "method", "family")
        if name in accepted
    }
    if "rank_by" in accepted and args.rank_by is not None:
        kwargs["rank_by"] = args.rank_by
    cfg = config(args.study)
    if args.clients_per_round is not None:
        import dataclasses
        cfg = dataclasses.replace(cfg, clients_per_round=args.clients_per_round)
    return generator(cfg, Path(args.root), Path(args.out), args.expect, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
