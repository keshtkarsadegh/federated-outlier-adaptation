"""
The winners re-run on a five-client federation.

Every method comparison in this study was made on ten clients with nine of them
drawn per round.  That is one federation size, and a result measured at one size
is a result about that size until something else is measured.  This stage holds
the method fixed and moves the size: the same configurations, the same horizon,
the same shipped model, on the cohort's **worst five** writers.

What is varied and what is not
------------------------------
Varied: the number of clients (ten to five) and, with it, the number drawn per
round (nine to four).  Held: the fold book, the old-data book, the initial
model, the anchor, the teacher, the local budget, the horizon, and the three
evaluation categories.  Nothing else moves, so a gap against the ten-client runs
is about the size of the federation rather than about the protocol.

Why four of five
----------------
:func:`~federated_outlier_adaptation.training.study_config.participants` is
``floor((1 - dropout) * n)``: ninety percent of five is four and a half, and a
fractional client is not trained on, so **four** train.  The same rule gives the
nine of ten every other stage of this study runs and the sixteen of twenty of
the earlier programme, so this stage's participation is the study's own rule
applied to a smaller population rather than a number chosen for it.

The extreme cases are the documented exception: at one or two clients the rule
would leave nobody, and they run at full participation instead - see
:func:`~federated_outlier_adaptation.training.study_config.participants`.

Where the configurations come from
----------------------------------
The three selected arms are not written down here.  They are read, every time
:func:`configs` is called, out of the study's own selection records - the
crowning in ``tables/p15_stage_winner.json`` and the cross it crowned in
``tables/p15_combination_grid.json`` - and the module refuses by name when
either is absent.  Written-down winners stay correct only until the selection is
re-run, and when they stop being correct nothing says so.

Why the control is here
-----------------------
Three configurations are the ones the panel selected; the fourth is plain
FedAvg with no penalty at all.  Without it a change between ten and five clients
could not be attributed: every selected configuration might move together simply
because the federation got smaller, and only an unmodified baseline moving the
same way would show that.  The control runs **both** schedules in one task, as
every ``--aggregation fedavg`` line in this study does, because plain averaging
is the same rule on either loop.

What the trimmed mean does at a smaller K
-----------------------------------------
``trimmed_t3`` drops three updates from each end of the ordered
coordinate.  At nine participants that is three from each end and three
survivors; at eight it is three and two; at **five** it is two and **one**, so
the rule degenerates into a single surviving update - a coordinate-wise median
of five rather than an average of anything.  That is a real change in what the
method is, and it belongs in the write-up next to the number it produces.
:func:`trim_survivors` computes it from the same expression the server uses, so
the documented survivor count cannot drift from the applied one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from federated_outlier_adaptation.training import study_config

#: The cohort's worst-five writers are drawn from this many ranked names.
COHORT_SIZE = 5

#: The study's dropout rate, unchanged - only the population it applies to.
DROPOUT_RATE = study_config.DROPOUT_RATE

#: Rounds every configuration runs for: the reporting horizon, unchanged.
FULL_ROUNDS = 100

def clients_per_round(clients: int = COHORT_SIZE,
                      dropout: float = DROPOUT_RATE) -> int:
    """
    How many of ``clients`` a round draws - the study's rule, in one place.

    Delegates to
    :func:`~federated_outlier_adaptation.training.study_config.participants`
    rather than restating it, so this module cannot come to disagree with the
    study configuration about what the study's own participation is.
    """
    return study_config.participants(clients, dropout)


def trim_survivors(fraction: float, participants: int) -> int:
    """
    How many updates a trimmed mean averages at a given participant count.

    Mirrors the server's own ``int(fraction * K)`` and its refusal to trim away
    everything, so the documented survivor count and the applied one are the
    same arithmetic rather than two statements of it.
    """
    trim = int(fraction * participants)
    if trim <= 0 or participants - 2 * trim <= 0:
        return participants
    return participants - 2 * trim


#: The records this stage's arms are read out of.  ``p15_stage_winner.json``
#: carries the crowning and, beside it, every finalist ordered by the rule that
#: crowned it; ``p15_combination_grid.json`` says which schedule each pair
#: belongs to and which pairs the cross actually ran.  Both are written by
#: ``study_emit`` into the study being emitted.
WINNER_RECORD = "p15_stage_winner.json"
COMBINATION_RECORD = "p15_combination_grid.json"

#: The four arms, in report order, described by WHAT EACH ONE IS rather than by
#: which cell currently is it.  The three selected arms carry no aggregation and
#: no penalty here: those are read from the records above, at the moment the
#: stage is emitted, for the study it is emitted from.
#:
#: THIS IS WHY THEY ARE NOT WRITTEN DOWN. A hardcoded winner is correct only
#: until the selection is re-run, and when it stops being correct nothing says
#: so: the stage still emits, still trains, and reports a configuration the
#: current cross never chose. This module carried a previous programme's pairs
#: for exactly that reason, with a comment admitting it.
ARMS: List[Dict[str, Any]] = [
    {"id": "winner", "family": "concurrent", "pick": "strongest",
     "note": "the cross's strongest concurrent pair"},
    {"id": "balanced", "family": "concurrent", "pick": "preserving",
     "note": "the concurrent pair that gave up least preservation"},
    {"id": "sequential", "family": "sequential", "pick": "strongest",
     "note": "the strongest sequential pair"},
    {"id": "control", "family": "both", "pick": None,
     "note": (
         "plain FedAvg, no penalty, both schedules - the baseline that says "
         "whether a change at five clients is the method or the size"
     )},
]


def _record(root, name: str) -> Dict[str, Any]:
    """
    One selection record, or a refusal that names the file.

    Never a default and never a guess: a stage that invented its winners would
    produce a task file indistinguishable from a real one, and the mistake would
    only surface as a table nobody could reproduce.
    """
    path = Path(root) / "tables" / name
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. The configurations this stage carries forward "
            "are the ones the cross selected, and with the record absent they "
            "would have to be guessed - which emits, trains and reports exactly "
            "like a real selection. Crown the combination stage first."
        )
    return json.loads(path.read_text())


def _pairs(root) -> Dict[str, Tuple[str, str, str]]:
    """``{combination id: (family, aggregation, penalty)}``, from the cross."""
    grid = _record(root, COMBINATION_RECORD).get("pairs") or {}
    pairs: Dict[str, Tuple[str, str, str]] = {}
    for family, entries in grid.items():
        for entry in entries:
            agg_id, reg_id = list(entry)[:2]
            pairs[f"{agg_id}_{reg_id}"] = (family, agg_id, reg_id)
    if not pairs:
        raise SystemExit(
            f"FATAL: {Path(root) / 'tables' / COMBINATION_RECORD} names no "
            "pairs; the cross it records ran nothing to carry forward."
        )
    return pairs


def finalists(root) -> List[Dict[str, Any]]:
    """
    Every combination the cross ran, best first, with its schedule and its halves.

    The order is the crowning record's own ``ranked`` - gain less spend, by the
    column that record says it was ranked on - so this stage cannot come to
    disagree with the crowning about which pair was strongest. The schedule and
    the two halves come from the combination grid, because a combination id is
    two ids joined by an underscore and splitting it needs the list it was built
    from.
    """
    record = _record(root, WINNER_RECORD)
    ranked = record.get("ranked") or []
    if not ranked:
        raise SystemExit(
            f"FATAL: {Path(root) / 'tables' / WINNER_RECORD} ranks no "
            "combination; there is nothing to carry forward."
        )
    pairs = _pairs(root)
    rows = []
    for entry in ranked:
        combo_id = entry.get("id")
        if combo_id not in pairs:
            raise SystemExit(
                f"FATAL: {WINNER_RECORD} ranks {combo_id!r}, which "
                f"{COMBINATION_RECORD} does not list as a pair the cross ran. "
                "The two records describe different stages."
            )
        family, agg_id, reg_id = pairs[combo_id]
        rows.append({
            "id": combo_id, "family": family, "aggregation": agg_id,
            "regulariser": reg_id, "preservation": entry.get("preservation"),
        })
    return rows


def configs(root) -> List[Dict[str, Any]]:
    """
    The configurations this stage runs, in report order, read from ``root``.

    Called with the study root rather than resolved on import, so the module can
    be imported without a study and so two studies emitted by one process get
    their own winners rather than whichever was read first.
    """
    rows = finalists(root)
    cells: List[Dict[str, Any]] = []
    for arm in ARMS:
        if arm["pick"] is None:
            cells.append({"id": arm["id"], "family": arm["family"],
                          "aggregation": "fedavg", "regulariser": None,
                          "note": arm["note"]})
            continue
        here = [row for row in rows if row["family"] == arm["family"]]
        if not here:
            raise SystemExit(
                f"FATAL: the crowning ranks no {arm['family']} combination, so "
                f"the {arm['id']!r} arm has nothing to carry forward."
            )
        if arm["pick"] == "strongest":
            chosen = here[0]
        else:
            if any(row["preservation"] is None for row in here):
                raise SystemExit(
                    f"FATAL: the crowning records no preservation for some "
                    f"{arm['family']} combination, and the {arm['id']!r} arm is "
                    "defined by it."
                )
            # Not a pair another arm already carries: two arms naming one
            # configuration would run it twice under two names and report the
            # difference between a cell and itself.
            taken = {(c["aggregation"], c["regulariser"]) for c in cells}
            free = [row for row in here
                    if (row["aggregation"], row["regulariser"]) not in taken]
            if not free:
                raise SystemExit(
                    f"FATAL: every {arm['family']} combination is already "
                    f"carried by an earlier arm, so {arm['id']!r} would repeat "
                    "one. The cross is too small for the arms this stage runs."
                )
            chosen = sorted(free, key=lambda r: (-r["preservation"], r["id"]))[0]
        cells.append({"id": arm["id"], "family": arm["family"],
                      "aggregation": chosen["aggregation"],
                      "regulariser": chosen["regulariser"],
                      "note": arm["note"]})
    return cells


def worst_five(ranking: List[str], size: int = COHORT_SIZE) -> List[str]:
    """
    The first ``size`` names of a worst-first ranking.

    Taken from the ranking artefact rather than restated, so this cohort cannot
    disagree with the ten-client one about who the hardest writers are.
    """
    if len(ranking) < size:
        raise ValueError(
            f"The five-client cohort needs {size} writers; the ranking has "
            f"{len(ranking)}."
        )
    return list(ranking[:size])
