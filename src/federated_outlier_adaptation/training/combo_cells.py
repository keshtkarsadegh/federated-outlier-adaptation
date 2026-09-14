"""
Stage 8: the top server rules crossed with the top client penalties.

Stages 6 and 7 each answered half a question.  Stage 6 varied the **server**
rule with no penalty on the clients; stage 7 varied the **client** penalty under
plain FedAvg.  Neither can say whether the two compose - whether a server rule
that preserves and a penalty that preserves preserve *twice*, or whether they
are two names for the same restraint and stacking them buys nothing.

That is the only question this stage asks, so it is a cross and not another
sweep: three aggregations by three regularisers, per schedule, at the full
hundred-round horizon.

One combination is one family
-----------------------------
Unlike stages 6 and 7, a stage-8 line does **not** run ``--aggregation fedavg``.
Each rule named here lives in exactly one (scenario, metadata) family, so the
aggregation decides the schedule and a task produces one result rather than two.
That is what makes the concurrent and sequential halves of the table separate
experiments with their own winners rather than one table read twice.

The numbers are inherited, not retyped
--------------------------------------
Every coefficient here is looked up from the stage-6 and stage-7 cell tables by
cell id.  Retyping a float is how ``2.3333333333333335`` becomes ``2.33333`` - a
different objective wearing the same label - and stage 7 already caught that
once.  The hybrid is rebuilt through the same constructor stage 7's own emitter
uses, from the same two winners, so the blend in this table and the blend in
``stage7_hybrid.txt`` are the same object at a different ``mix``.

The blends differ between the schedules on purpose: ``mix`` weights the KD half
against the Fisher half, and the two schedules chose differently, so the
concurrent half of the cross carries 0.75 and the sequential half 0.5.

The shortlist is read, not written down
---------------------------------------
Which three server rules a schedule contributes is a **selection result**, and
this module reads it from the screen's own record - ``tables/p12_agg_top3.json``
under a study root - rather than holding a copy of it.  A hardcoded shortlist is
correct only until the screen is re-run, and when it stops being correct nothing
says so: the stage still emits, still trains, and reports a cross the current
selection never chose.  This module held one for exactly that long, and two of
its three ``sequential`` ids - ``seq_mix_r0p7`` and ``seq_delta_scaled`` - were
not in the shortlist the study went on to run.  With no record to read it
refuses by name, because a guessed shortlist emits a task file indistinguishable
from a real one and the mistake surfaces only as a table nobody can reproduce.

Every entry point therefore takes a study root, and reads it at call time rather
than at import, so this module can be imported without a study and a generator
pointed at one study cannot carry another's shortlist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from federated_outlier_adaptation.training import agg_cells, reg_cells

#: Rounds every stage-8 line runs for: the reporting horizon, not a screen.
FULL_ROUNDS = 100

#: The two winners the kd+fisher hybrid is built from, by stage-7 cell id.
#: Repointed with the re-ranged reg grid: T=16 left the kd row when the top of
#: that row was dropped as measured-saturated.  Same caveat as the aggregation
#: shortlist below, which is now read from its screen's record - this one is
#: still written down, and should be read from the stage-7 record too.
HYBRID_KD_CELL = "kd_T8_a0p7"
HYBRID_FISHER_CELL = "fisher_lam0p1"

#: Blend weight of the hybrid on each schedule; see the module docstring.
HYBRID_MIX = {"concurrent": 0.75, "sequential": 0.5}

#: The record the three server rules of each schedule are read from, under a
#: study root's ``tables/``.  ``compare_arms.py`` and ``study_emit.py`` read the
#: same file for the same shortlist, so the cross this stage describes and the
#: cross the study ran cannot come apart.
AGG_SHORTLIST_RECORD = "p12_agg_top3.json"

#: The two non-hybrid penalties, by stage-7 cell id.  The hybrid is built rather
#: than looked up, because it was never a screening cell.
#: ``logit_l2_lam0p316`` left the row when it was placed below 1.0, where the
#: earlier work's own sweep shows the live region; 0.1 is its nearest survivor
#: and is also that study's reported setting.
REG_CELLS = (HYBRID_KD_CELL, "logit_l2_lam0p1")


def agg_shortlist(root, schedule: str) -> Tuple[str, ...]:
    """
    The three stage-6 cell ids one schedule contributes, read from ``root``.

    Raises:
        SystemExit: the record is absent, or names no shortlist for this
            schedule.  Refusing is the point; see the module docstring.
    """
    path = Path(root) / "tables" / AGG_SHORTLIST_RECORD
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. The three server rules this cross runs are the "
            "ones the aggregation screen crowned, and with the record absent "
            "they would have to be guessed - which emits, trains and reports "
            "exactly like a real selection. Crown the aggregation screen first."
        )
    top = (json.loads(path.read_text()).get("top") or {}).get(schedule)
    if not top:
        raise SystemExit(
            f"FATAL: {path} shortlists no {schedule!r} rule. That half of the "
            "cross has nothing to run and is not filled in from the other."
        )
    return tuple(top)


def _hybrid_cell(mix: float) -> Dict[str, Any]:
    """
    The kd+fisher blend at one ``mix``, built the way stage 7 builds it.

    Importing the emitter would drag a command-line tool into the package, so
    the construction is repeated here from the same two cells and asserted
    identical by a test - which is the point that actually needs guarding.
    """
    by_id = {cell["id"]: cell for cell in reg_cells.screen_cells()}
    kd = by_id[HYBRID_KD_CELL]
    fisher = by_id[HYBRID_FISHER_CELL]
    return reg_cells._cell(
        f"hybrid_mix{reg_cells._fmt(mix)}", "kd+fisher", "kd+fisher",
        f"kd+fisher blend, mix={mix:g} "
        f"(KD from {HYBRID_KD_CELL}, Fisher from {HYBRID_FISHER_CELL})",
        needs_fisher=True,
        lam=kd["hypers"]["lam"],
        T=kd["hypers"]["T"],
        mix=mix,
    )
    # ``fisher`` is looked up so an unknown id fails here rather than silently
    # producing a blend of one method.


def regularisers(schedule: str) -> List[Dict[str, Any]]:
    """The three penalties of one schedule, hybrid first."""
    by_id = {cell["id"]: cell for cell in reg_cells.screen_cells()}
    cells = [_hybrid_cell(HYBRID_MIX[schedule])]
    cells.extend(by_id[cell_id] for cell_id in REG_CELLS)
    return cells


def aggregations(schedule: str, root) -> List[Dict[str, Any]]:
    """
    The three server rules of one schedule: shortlisted by ``root``, described
    by the stage-6 cell table.

    Two steps, and both matter.  WHICH rules ran is a selection result and is
    read from the screen's record; WHAT each of them is - its rule, its server
    coefficients - is looked up in the cell table, so the flags this stage
    crosses are the ones that stage measured rather than a second statement of
    them.
    """
    by_id = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    shortlist = agg_shortlist(root, schedule)
    missing = [cell_id for cell_id in shortlist if cell_id not in by_id]
    if missing:
        raise SystemExit(
            f"FATAL: {AGG_SHORTLIST_RECORD} shortlists {missing}, which the "
            "stage-6 cell table does not hold. The record and the grid describe "
            "different screens."
        )
    return [by_id[cell_id] for cell_id in shortlist]


def combo_cells(root) -> List[Dict[str, Any]]:
    """Every combination, concurrent first.  Ids are unique by construction."""
    combos: List[Dict[str, Any]] = []
    for schedule in ("concurrent", "sequential"):
        for agg in aggregations(schedule, root):
            for reg in regularisers(schedule):
                combos.append({
                    "id": f"{agg['id']}_{reg['id']}",
                    "schedule": schedule,
                    "agg": agg,
                    "reg": reg,
                    "note": f"{agg['note']}  x  {reg['note']}",
                })
    seen = set()
    for combo in combos:
        if combo["id"] in seen:  # pragma: no cover - guarded by a test
            raise ValueError(f"Duplicate stage-8 combination id {combo['id']!r}")
        seen.add(combo["id"])
    return combos


def combos_by_schedule(root) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"concurrent": [], "sequential": []}
    for combo in combo_cells(root):
        grouped[combo["schedule"]].append(combo)
    return grouped
