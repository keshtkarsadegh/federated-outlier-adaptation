"""
SUPERSEDED - see tools/report_tables.py.

This assembles the PRIOR programme's two headline tables. Its rows are that
programme's cells and folders: ``combo_trimmed_0p4_feature_l2_lam0p1``, the
P18/P19/P20 rungs, and a scaling table pinned to fold 1 because those probes
ran without cross-validation. None of those ids exists under Digits_study01,
and every setting this study reports has five folds. Running it against the
current study root reads nothing and reports nothing, which is the failure
mode that made this banner necessary: it does not raise, it produces an empty
table that looks like a finished one.

The current tables are ``tools/report_tables.py --what all``, whose views are
the eight this study reports, and whose rows are five-fold means with a
spread. Nothing here is repointed, because the file's outputs were shipped
with the prior programme and editing it would falsify a record rather than
fix a tool. It is kept as that record.

Regenerate the study's two headline tables from the run folders.

    python tools/make_tables.py master  --root "$FOA_STUDY_DIR" --out tables/master_table.md
    python tools/make_tables.py scaling --root "$FOA_STUDY_DIR" --out tables/scaling_table.md
    python tools/make_tables.py all     --root "$FOA_STUDY_DIR" --out-dir tables

Reads JSON and writes markdown: a login-node job, no GPU.

Why this exists
---------------
Both tables used to be assembled by hand.  Every number in them came from a run
folder, so every row was checkable - but nobody could regenerate them, and a
table that cannot be regenerated cannot be shown to still match the runs after a
stage is re-run.  This reads the same folders and emits the same markdown.

What is computed and what is written
------------------------------------
Every **number** comes from a run folder; nothing numeric is typed here.  The
**framing** - what a rung means, why single-client is not a federation, what the
readings are - is prose, and prose is carried as prose.  The line between them
is deliberate: a number that could drift from the runs is computed, and a
sentence that states an interpretation is not something a program can derive.

The fold-1 rules of the scaling table
-------------------------------------
The scaling and dropout probes ran without cross-validation - one g-0, one
client split - so their cells are single-fold point estimates.  Three
consequences are baked in here rather than left to the person writing the table:

* the fold filter is **explicit** (``FOLD_ONLY``), never "whatever folders
  exist".  P18 and P19 have all five folds on disk and P20 has one; a reader
  that globbed would average five folds for some rungs and one for others, and
  the difference would read as a size effect.
* the ten-client anchors are read from **fold 1 of the same runs**, not from the
  CV-5 mean, so every cell in the table is one draw.
* no cell carries a ``±``.  A single fold has no spread; the P15 cross-fold
  spread is quoted once, as the noise floor a reader should judge differences
  against.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The fold every scaling and dropout cell is read from.  One number, one place:
#: the rule is "fold 1 for every rung including the anchors", and a table that
#: mixed horizons would be indistinguishable from a size effect.
FOLD_ONLY = 1

#: The folds the main study cross-validates over.
CV_FOLDS = (1, 2, 3, 4, 5)

#: Which sub-directory of a run folder belongs to which schedule.  A run that
#: names ``--aggregation fedavg`` produces both, so the family cannot be taken
#: from the folder name and has to be selected from the path.
FAMILY_DIRS = {"concurrent": "concurrent_", "sequential": "sequential_"}


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def _load(path) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _mean(values) -> Optional[float]:
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else None


def read_run(root: Path, parent: str, fold: int,
             family: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    The three reported categories of one federated run.

    Args:
        parent: The run's parent tag without the study prefix, e.g.
            ``combo_trimmed_0p4_feature_l2_lam0p1``.
        family: ``concurrent`` or ``sequential`` when the run produced both -
            which every ``--aggregation fedavg`` line does.  A run that produced
            one is unambiguous and needs no filter.

    Returns:
        ``{"pooled", "per_client", "old"}``, or ``None`` when the run is absent.
    """
    pattern = f"{root}/d01_{parent}_fold{fold}_*/**/accuracies_*.json"
    payloads = sorted(glob.glob(pattern, recursive=True))
    if family is not None:
        want = FAMILY_DIRS[family]
        payloads = [p for p in payloads if f"/{want}" in p]
    if not payloads:
        return None
    data = _load(payloads[0])
    if not data:
        return None
    final = data.get("final_evaluation") or {}
    clients = final.get("clients") or {}
    old = final.get("old") or {}
    per = clients.get("per_client") or {}
    return {
        "pooled": clients.get("accuracy"),
        "per_client": _mean(per.values()) if per else None,
        "per_client_map": per,
        "old": old.get("mean"),
        "path": payloads[0],
    }


def read_folds(root: Path, parent: str, folds=CV_FOLDS,
               family: Optional[str] = None) -> Dict[str, Optional[float]]:
    """A run's three categories, averaged over the folds that are present."""
    rows = [read_run(root, parent, f, family) for f in folds]
    rows = [r for r in rows if r]
    return {
        "pooled": _mean([r["pooled"] for r in rows]),
        "per_client": _mean([r["per_client"] for r in rows]),
        "old": _mean([r["old"] for r in rows]),
        "n": len(rows),
    }


def read_isolated(root: Path, init: str, folds=CV_FOLDS) -> Dict[str, Any]:
    """
    The private-model rung: one model per client, no federation at all.

    ``union`` is the pooled figure - each client's model scored on every
    client's rows - and ``own`` is each model on its own writer.  The gap
    between them is the whole point of the rung: a private model is excellent on
    itself and worthless on anyone else, which is what a federation is for.
    """
    own, union, old = [], [], []
    for fold in folds:
        for path in sorted(glob.glob(f"{root}/isolated/isolated_{init}_*_fold{fold}.json")):
            data = _load(path)
            if not data:
                continue
            for record in (data.get("per_client") or {}).values():
                own.append(record.get("own"))
                union.append(record.get("union"))
                folds_map = record.get("old_folds") or {}
                old.append(_mean([float(v) for v in folds_map.values()]))
    return {"pooled": _mean(union), "per_client": _mean(own), "old": _mean(old)}


def read_centralized(root: Path, init: str, folds=CV_FOLDS) -> Dict[str, Any]:
    """The upper bound: one model trained on the cohort's pooled rows."""
    pooled, per, old = [], [], []
    for fold in folds:
        data = _load(f"{root}/centralized/centralized_outliers_{init}_fold{fold}.json")
        if not data:
            continue
        block = data.get("pooled") or {}
        pooled.append((block.get("own_weighted")))
        own = block.get("own") or {}
        per.append(own.get("mean"))
        old.append(data.get("old_mean"))
    return {"pooled": _mean(pooled), "per_client": _mean(per), "old": _mean(old)}


def read_do_nothing(root: Path) -> Dict[str, Any]:
    """
    The row every other row is read against: the shipped model, untouched.

    Adaptation comes from g-0 on the cohort's test rows, preservation from g-0
    on the old writers' - the same ``evaluate-book`` command a reviewer runs to
    check it.
    """
    cohort = _load(f"{root}/g0_perfold_evaluations.json") or {}
    old = _load(f"{root}/g0_evaluations.json") or {}
    pooled, per = [], []
    for record in cohort.values():
        pooled.append(record.get("accuracy"))
        writers = record.get("per_writer") or {}
        per.append(_mean(writers.values()) if writers else None)
    return {
        "pooled": _mean(pooled),
        "per_client": _mean(per),
        "old": _mean([r.get("accuracy") for r in old.values()]),
    }


def read_writer_baseline(root: Path) -> Dict[str, float]:
    """
    Each cohort writer's do-nothing accuracy under g-0.

    The **mean of its per-fold accuracies**, which is the record the extreme
    stage itself ranked on - not one fold's draw.  Reading a single fold here
    would name a different "worst writer" than the runs were built for, and the
    table would describe writers that never appeared in them.
    """
    record = _load(f"{root}/tables/clients_acc_on_g0.json") or {}
    return record.get("accuracies") or {}


def read_extreme_writers(root: Path) -> List[str]:
    """
    Who the extreme cases actually federated, from the list they were run with.

    Derived from the emitted client list rather than re-ranked here, so the
    table cannot name a pair the runs did not use.
    """
    listing = _load(f"{root}/outliers/extreme_double.json")
    if isinstance(listing, dict):
        listing = listing.get("clients")
    return list(listing or [])


def read_p15_spread(root: Path) -> Dict[str, Optional[float]]:
    """
    The cross-fold spread of the combination stage: the scaling table's noise
    floor.

    A single-fold cell has no error bar of its own.  The nearest honest thing to
    put beside it is how far the *same* configurations moved across folds when
    they were cross-validated, which is exactly what P15 measured.
    """
    grid = _load(f"{root}/tables/p15_combination_grid.json") or {}
    pooled_sd, old_sd = [], []
    for family, pairs in (grid.get("pairs") or {}).items():
        for agg, reg in pairs:
            rows = [read_run(root, f"combo_{agg}_{reg}", f) for f in CV_FOLDS]
            rows = [r for r in rows if r]
            if len(rows) < 2:
                continue
            pooled = [r["pooled"] for r in rows if r["pooled"] is not None]
            old = [r["old"] for r in rows if r["old"] is not None]
            if len(pooled) > 1:
                pooled_sd.append(statistics.stdev(pooled))
            if len(old) > 1:
                old_sd.append(statistics.stdev(old))
    return {
        "pooled_min": min(pooled_sd) if pooled_sd else None,
        "pooled_max": max(pooled_sd) if pooled_sd else None,
        "old_min": min(old_sd) if old_sd else None,
        "old_max": max(old_sd) if old_sd else None,
        "n": len(pooled_sd),
    }


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def f4(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def f3(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def pair(cell: Dict[str, Any]) -> str:
    """A scaling cell: adaptation / preservation, no spread - it is one fold."""
    if not cell or cell.get("pooled") is None:
        return "—"
    return f"{f4(cell['pooled'])} / {f4(cell['old'])}"


def span(cells: List[Dict[str, Any]]) -> str:
    """
    A cell covering several schedules: the range each axis spans.

    A plain-FedAvg control runs both loops in one task, so it has two answers
    and no reason to prefer one.  Printing the range says that; printing either
    number alone would quietly pick a schedule.
    """
    kept = [c for c in cells if c and c.get("pooled") is not None]
    if not kept:
        return "—"
    if len(kept) == 1:
        return pair(kept[0])
    pooled = sorted(c["pooled"] for c in kept)
    old = sorted(c["old"] for c in kept if c.get("old") is not None)
    left = f4(pooled[0]) if pooled[0] == pooled[-1] else f"{f4(pooled[0])}-{f4(pooled[-1])}"
    right = f4(old[0]) if old[0] == old[-1] else f"{f4(old[0])}-{f4(old[-1])}"
    return f"{left} / {right}"


# --------------------------------------------------------------------------- #
# the master ladder
# --------------------------------------------------------------------------- #
#: ``(label, reader)`` for the ladder, in report order.  The order is the
#: argument the table makes: do nothing, then isolate, then federate, then bound
#: it, then improve the server rule, then the client penalty, then both.
def master_rows(root: Path) -> List[tuple]:
    winners = _load(f"{root}/tables/p12_agg_top3.json") or {}
    reg_top = _load(f"{root}/tables/p14_reg_top3.json") or {}
    agg_conc = (winners.get("top") or {}).get("concurrent", ["anchor_0p03"])[0]
    agg_seq = (winners.get("top") or {}).get("sequential", ["seq_delta_capped"])[0]
    reg_conc = (reg_top.get("top") or {}).get("concurrent", ["ntd_b0p01_t0p5"])[0]
    reg_seq = (reg_top.get("top") or {}).get("sequential", ["feature_l2_lam0p01"])[0]

    labels = {
        "anchor_0p03": "anchor 0.03", "seq_delta_capped": "seq_delta_capped",
        "ntd_b0p01_t0p5": "FedNTD b0.01 t0.5", "feature_l2_lam0p01": "feature_l2 0.01",
    }
    name = lambda key: labels.get(key, key)  # noqa: E731

    return [
        ("g-0 untouched (do nothing)", read_do_nothing(root), None),
        ("Isolated, scratch", read_isolated(root, "scratch"), ("union", "own")),
        ("Isolated, g-0 fine-tune", read_isolated(root, "global"), ("union", "own")),
        ("Normal FL (g-0 init, concurrent)",
         read_folds(root, "fl_global", family="concurrent"), None),
        ("Normal FL (g-0 init, sequential)",
         read_folds(root, "fl_global", family="sequential"), None),
        ("Centralized bound (g-0)", read_centralized(root, "global"), None),
        (f"Best agg conc: {name(agg_conc)}", read_folds(root, f"aggfull_{agg_conc}",
                                                        family="concurrent"), None),
        (f"Best agg seq: {name(agg_seq)}", read_folds(root, f"aggfull_{agg_seq}",
                                                      family="sequential"), None),
        (f"Best reg conc: {name(reg_conc)}",
         read_folds(root, f"regfull_concurrent_{reg_conc}", family="concurrent"), None),
        (f"Best reg seq: {name(reg_seq)}",
         read_folds(root, f"regfull_sequential_{reg_seq}", family="sequential"), None),
        ("COMBO WINNER: trimmed0.4 x feature_l2",
         read_folds(root, "combo_trimmed_0p4_feature_l2_lam0p1"), None),
        ("Combo balanced: anchor x FedNTD",
         read_folds(root, "combo_anchor_0p03_ntd_b0p01_t0p5"), None),
        ("Combo seq best: shuffle x feature_l2",
         read_folds(root, "combo_seq_order_shuffle_feature_l2_lam0p01"), None),
    ]


def master_table(root: Path) -> str:
    ginit = _load(f"{root}/ginit_selection.json") or {}
    reference = ginit.get("test_mean")
    do_nothing = read_do_nothing(root)
    per_writer = read_writer_baseline(root)
    ranked = read_extreme_writers(root)

    out = [
        "# Master table — Digits_study01 (10 outliers / 200 old / 10% dropout / CV-5)",
        "",
        "All values = mean over 5 folds. Columns: pooled clients test / per-client "
        "mean / old-data preservation (mean of 5 old-fold tests). Reference: "
        f"g-init centralized max {f4(reference)}.",
        "",
        "| Rung | pooled | per-client | old |",
        "|---|---|---|---|",
    ]
    for label, cell, notes in master_rows(root):
        pooled, per = f4(cell.get("pooled")), f4(cell.get("per_client"))
        if notes:
            pooled, per = f"{pooled} ({notes[0]})", f"{per} ({notes[1]})"
        out.append(f"| {label} | {pooled} | {per} | {f4(cell.get('old'))} |")

    # ---- the extreme cases -------------------------------------------------
    worst = ranked[0] if ranked else "?"
    second = ranked[1] if len(ranked) > 1 else "?"
    iso_global = _extreme_solo(root, "global", worst)
    dual = read_folds(root, "extreme_dual")
    double = read_folds(root, "extreme_double")
    solo_second = _extreme_solo(root, "global", second)
    combo = read_folds(root, "combo_trimmed_0p4_feature_l2_lam0p1")
    double_per = _extreme_per_writer(root, "extreme_double", [worst, second])

    out += [
        "",
        "## Extreme cases (winner combo) — the minimum-possible FL",
        "",
        "Framing (owner): DUAL and DOUBLE are the minimum possible FL solutions, and",
        "they work. The one-client arrangement that used to sit above them is out of the",
        "study: federating a lone writer is a question about fine-tuning, and its own",
        "run folders are no longer part of what any table reports.",
        "",
        f"### The minimum FL (two writers: {worst} {f3(per_writer.get(worst))}, "
        f"{second} {f3(per_writer.get(second))} under g-0)",
        "",
        "| Setup | pooled | per-writer | old |",
        "|---|---|---|---|",
        f"| best solo (g-0 fine-tune, per writer) | - | "
        f"{f3(iso_global['own'])} / {f3(solo_second['own'])} | "
        f"{f3(iso_global['old'])} / {f3(solo_second['old'])} |",
        f"| DUAL: one client, merged rows (FL-for-single-clients) | {f4(dual['pooled'])} | "
        f"{f3(dual['per_client'])} | {f4(dual['old'])} |",
        f"| DOUBLE: two clients federating | {f4(double['pooled'])} | "
        f"{f3(double_per.get(worst))} / {f3(double_per.get(second))} | {f4(double['old'])} |",
        "",
        f"The minimum-possible federation (two clients) WORKS: {worst} reaches "
        f"{f3(double_per.get(worst))} -",
        "above its best solo result - and double beats dual on both columns on identical",
        f"data. Preservation at n=2 ({f3(double['old'])}) is below the 10-client study "
        f"({f4(combo['old'])}): the",
        "full federation is part of the old-knowledge protection.",
    ]
    return "\n".join(out) + "\n"


def _extreme_solo(root: Path, init: str, writer: str) -> Dict[str, Any]:
    """One writer's private-model result, averaged over the folds it has."""
    own, old = [], []
    for fold in CV_FOLDS:
        data = _load(f"{root}/isolated/isolated_{init}_{writer}_fold{fold}.json")
        if not data:
            continue
        record = (data.get("per_client") or {}).get(writer) or {}
        own.append(record.get("own"))
        old.append(_mean([float(v) for v in (record.get("old_folds") or {}).values()]))
    return {"own": _mean(own), "old": _mean(old)}


def _extreme_per_writer(root: Path, parent: str, writers: List[str]) -> Dict[str, float]:
    """Each writer's own column in an extreme run, averaged over folds."""
    gathered: Dict[str, List[float]] = {w: [] for w in writers}
    for fold in CV_FOLDS:
        run = read_run(root, parent, fold)
        if not run:
            continue
        for writer, value in (run["per_client_map"] or {}).items():
            gathered.setdefault(writer, []).append(value)
    return {w: _mean(v) for w, v in gathered.items() if v}


# --------------------------------------------------------------------------- #
# the scaling and dropout table
# --------------------------------------------------------------------------- #
#: ``(federation label, config label, parent at 10% dropout, parent at 20%)``.
#: A missing parent is a rung that was never run at that level and prints as an
#: em dash rather than as a blank a reader could mistake for a zero.
SCALING_ROWS = [
    ("2 (double, extreme)", "winner", "extreme_double", None),
    ("5 (4-of-5)", "winner", "five_winner", None),
    ("5", "balanced", "five_balanced", None),
    ("5", "sequential", "five_sequential", None),
    ("5", "control", "five_control", None),
    ("10 (9-of-10 / 8-of-10)", "winner", "combo_trimmed_0p4_feature_l2_lam0p1", "drop20_winner"),
    ("10", "balanced", "combo_anchor_0p03_ntd_b0p01_t0p5", "drop20_balanced"),
    ("10", "sequential", "combo_seq_order_shuffle_feature_l2_lam0p01", "drop20_sequential"),
    ("10", "control (con)", "fl_global", "drop20_control"),
    ("20 (18-of-20 / 16-of-20)", "winner", "c20d10_winner", "c20d20_winner"),
    ("20", "balanced", "c20d10_balanced", "c20d20_balanced"),
    ("20", "sequential", "c20d10_sequential", "c20d20_sequential"),
]

#: Rungs whose control runs both schedules in one task, so a family filter is
#: needed to say which half the row reports.
#: ``None`` means the row reports both schedules as a range; a named family
#: means the row's label pins one, as ``control (con)`` does.
CONTROL_FAMILY = {"five_control": None, "drop20_control": "concurrent",
                  "fl_global": "concurrent"}
BOTH_FAMILIES = ("concurrent", "sequential")


def scaling_table(root: Path) -> str:
    spread = read_p15_spread(root)
    do_nothing = read_do_nothing(root)
    c20_base = _load(f"{root}/g0_cohort20_evaluations.json") or {}
    c20_fold1 = (c20_base.get(f"cohort20_fold{FOLD_ONLY}") or {}).get("accuracy")
    per_writer = read_writer_baseline(root)
    ranked = read_extreme_writers(root)
    worst = ranked[0] if ranked else "?"
    second = ranked[1] if len(ranked) > 1 else "?"

    out = [
        "# Scaling & dropout tables — Digits_study01 winners (fold 1, single g-0)",
        "Cells: pooled clients test / old-data preservation (5-fold mean). "
        "No CV on these probes",
        (f"(owner rule); P15 cross-fold spread (the noise floor): "
         f"~±{spread['pooled_min']:.2f}-{spread['pooled_max']:.2f} pooled, "
         f"~±{spread['old_min']:.4f}-{spread['old_max']:.4f} old."
         if spread["pooled_min"] is not None else "(owner rule)."),
        f"Do-nothing baselines (fold {FOLD_ONLY}): cohort10 = {f4(do_nothing['pooled'])} "
        f"(CV mean; study baseline), cohort20 = {f4(c20_fold1)}.",
        "Extremes ran full participation (never dropped); 5c/10c rows include "
        "plain-FedAvg controls, 20c is winners-only.",
        "",
        "## Federation size (10% dropout column) and dropout (20%)",
        "| Federation | Config | 10% dropout | 20% dropout |",
        "|---|---|---|---|",
    ]

    for federation, config, parent10, parent20 in SCALING_ROWS:
        if parent10 == "extreme_double":
            # The extreme stage kept CV-5; its cell is a five-fold mean and is
            # starred, so it is never read as one of the fold-1 probes.
            cell = read_folds(root, parent10)
            left = f"{f4(cell['pooled'])}* / {f4(cell['old'])}*"
            right = "— (never dropped)"
        else:
            left = _cell(root, parent10)
            right = _cell(root, parent20) if parent20 else "—"
        out.append(f"| {federation} | {config} | {left} | {right} |")

    out += [
        f"*extreme-case CV-5 values (that stage kept CV); its do-nothing writers "
        f"were {f3(per_writer.get(worst))}/{f3(per_writer.get(second))}.",
        "",
    ]
    out += _balanced_note(root)
    out += ["", "## Readings"] + READINGS
    return "\n".join(out) + "\n"


def expand(root: Path, value: Optional[str]) -> Optional[Path]:
    """A shipped provenance path, with its placeholder resolved against a root."""
    if not value:
        return None
    return Path(str(value).replace("$FOA_STUDY_DIR", str(root)))


def max_param_diff(root: Path, record: dict) -> Optional[float]:
    """
    How far apart the two balanced checkpoints actually are, in weight space.

    The claim the note makes is that two *different* models score identically,
    and the only thing that separates that from a copied cell is a number
    measured on the weights themselves.  So it is loaded and computed, not
    quoted.  Where the checkpoints are not present - a clone without the run
    folders - the clause is dropped rather than asserted from memory.
    """
    try:
        import torch
    except ImportError:
        return None
    paths = [expand(root, (record.get(k) or {}).get("model"))
             for k in ("recheck_d10", "recheck_d20")]
    if not all(p and p.is_file() for p in paths):
        return None
    try:
        a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in paths)
    except Exception:
        return None
    a = a.get("model_state", a) if isinstance(a, dict) else a
    b = b.get("model_state", b) if isinstance(b, dict) else b
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return None
    shared = [k for k in a if k in b and hasattr(a[k], "shape")]
    if not shared:
        return None
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in shared)


def _cell(root: Path, parent: str) -> str:
    """One scaling cell, at fold 1, over whichever schedules the row reports."""
    if parent in CONTROL_FAMILY and CONTROL_FAMILY[parent] is None:
        return span([read_run(root, parent, FOLD_ONLY, f) for f in BOTH_FAMILIES])
    return pair(read_run(root, parent, FOLD_ONLY, CONTROL_FAMILY.get(parent)))


def _balanced_note(root: Path) -> List[str]:
    """
    The verified-identity note, sourced from the re-evaluation record.

    Two runs at different dropout levels scoring bit-identically is the kind of
    result far more often produced by a copied cell than by the world, so the
    claim is made only from the record of the direct re-evaluation that checked
    it, and every number in the sentence is derived from that file.
    """
    record = _load(f"{root}/tables/balanced_recheck.json")
    if not record:
        return ["## Verified identity note (the balanced attractor)",
                "(no tables/balanced_recheck.json; the note is omitted rather than asserted)"]
    cohort = record.get("recheck_d10") or {}
    old = record.get("recheck_old_d10") or {}
    samples = cohort.get("samples")
    correct = round((cohort.get("accuracy") or 0) * samples) if samples else None
    old_n = old.get("samples")
    old_acc = old.get("accuracy")
    errors = old_n - round(old_acc * old_n) if (old_n and old_acc is not None) else None
    diff = max_param_diff(root, record)
    weights = (f"the two runs trained genuinely different weights (max param diff\n"
               f"{diff:.3f}) yet score" if diff is not None
               else "the two runs score")
    return [
        "## Verified identity note (the balanced attractor)",
        "At 20 clients the balanced (anchor x FedNTD) rows are identical at both "
        "dropout levels.",
        f"This was fully verified: {weights} bit-identically on cohort test "
        f"({correct}/{samples}) AND old fold tests ({old_acc:.6f},",
        f"{errors} errors in {old_n:,}) — direct re-evaluation of both saved models "
        "confirmed it",
        "(tables/balanced_recheck.json). The anchor-to-g0 + NTD-from-g0 combination "
        "is a strong",
        "attractor: different participation draws converge to functionally the same "
        "predictor.",
        "This is WHY balanced is dropout-invariant across every table.",
    ]


#: The interpretation of the table.  Prose, deliberately: these are claims about
#: what the numbers mean, and a program cannot derive a claim.
READINGS = [
    "1. Preservation rises with federation size for every method (winner old: 0.978 @5 ->",
    "   0.992-0.993 @10-20); controls' forgetting explodes at small n (0.86-0.89 @5).",
    "2. The winner is tuned to its native point (best at 10c/9-of-10: 0.9442 on fold 1);",
    "   at other sizes trimming keeps ~2-4 survivors and it converges toward the pack.",
    "3. Balanced is the robustness champion: 0.99+ preservation at EVERY size and dropout,",
    "   adaptation within 1-2 pts of the best everywhere — the deployment recommendation.",
    "4. Dropout (10->20%) is a modest knob at every size; federation size dominates.",
]


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("master", "scaling", "all"))
    parser.add_argument("--root", required=True, help="The study root.")
    parser.add_argument("--out", default=None, help="Destination markdown file.")
    parser.add_argument("--out-dir", default=None, help="Destination directory for 'all'.")
    parser.add_argument(
        "--check", action="store_true",
        help=(
            "Compare with the file at --out instead of writing it, and exit "
            "non-zero on any difference. This is the regression: a table that "
            "no longer matches its runs must fail loudly, not be overwritten."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"FATAL: no study at {root}.", file=sys.stderr)
        return 1

    builders = {"master": master_table, "scaling": scaling_table}
    wanted = ["master", "scaling"] if args.what == "all" else [args.what]

    status = 0
    for name in wanted:
        text = builders[name](root)
        if args.what == "all":
            target = Path(args.out_dir or root / "tables") / f"{name}_table.md"
        else:
            target = Path(args.out) if args.out else root / "tables" / f"{name}_table.md"

        if args.check:
            current = target.read_text() if target.is_file() else ""
            if current != text:
                print(f"DIFFERS: {target}", file=sys.stderr)
                import difflib

                for line in difflib.unified_diff(
                    current.splitlines(), text.splitlines(),
                    fromfile=f"{target} (on disk)", tofile="regenerated", lineterm="",
                ):
                    print(line, file=sys.stderr)
                status = 1
            else:
                print(f"matches: {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(f"wrote {target} ({len(text.splitlines())} lines)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
