"""
Stage 10: the extreme cases, run only with the method that won.

Every earlier stage compared methods.  This one does not: the panel is over, the
winner is whichever combination the cross crowned, and the only thing that
varies here is how few clients the federation has and how their data is
arranged.  Running a method panel again would be answering a question that has
already been answered, at three times the cost.

Which combination that is is **read**, not written down here:
:func:`winning_combo_id` takes it from the study's own
``tables/p15_stage_winner.json`` and refuses by name when that record is absent.
A winner restated in source is correct only until the crowning is re-run, and
when it stops being correct the stage still emits, still trains, and reports a
method the current cross never chose.

Two arrangements of the same two writers
----------------------------------------
``double``
    a federation of two clients: the two worst distinct writers.
``dual``
    a federation of **one** client whose data is both of those writers' rows
    merged - the same images as ``double``, held by one participant.

That is the whole stage, and the pair is the reason it exists.  They hold
precisely the same rows; they differ only in whether the aggregation ever sees
them separately.  Any gap between them is what client *boundaries* cost, with
the data held constant - which is a question the twenty-client stages cannot
ask, because there the boundaries and the data always move together.

A one-client arrangement is deliberately not here.  Federating a lone writer
answers a question about fine-tuning rather than about aggregation, and it has
no partner to be held constant against; the pair above is the controlled
comparison this stage was built to make.

Full participation
------------------
No sampler.  Dropping 20% of a two-client federation is not a participation
study, it is a coin flip deciding whether a round happens at all, so every
client trains every round and the run reports what the arrangement does rather
than what the draw did.

Who the writers are
-------------------
The two worst writers of the cohort by their accuracy under the frozen g-0,
read from ``outliers/clients_acc_on_global.json`` and restricted to the cohort,
lowest first with ties broken by writer id.  Deterministic, and recorded in the
run's provenance through the client-list files the emitter writes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

#: Rounds every case runs for: the reporting horizon.
FULL_ROUNDS = 100

#: The record the crowning is read from, under a study root's ``tables/``.
WINNER_RECORD = "p15_stage_winner.json"


def winning_combo_id(root) -> str:
    """
    The id of the combination the cross crowned, read from ``root``.

    Read at call time and never at import, so this module can be imported
    without a study and so a generator pointed at one study cannot carry
    another's winner.

    Raises:
        SystemExit: the record is absent or names no winner. Refusing is the
            point: a guessed winner produces a task file that looks exactly like
            a real one, and the error only ever surfaces as a table that cannot
            be reproduced.
    """
    path = Path(root) / "tables" / WINNER_RECORD
    if not path.is_file():
        raise SystemExit(
            f"FATAL: no {path}. This stage runs the winning method only, and "
            "the winner is a selection result rather than something to be "
            "named here. Crown the combination stage first."
        )
    winner = (json.loads(path.read_text()) or {}).get("winner")
    if not winner:
        raise SystemExit(f"FATAL: {path} names no winner.")
    return str(winner)

#: The two arrangements.  Both take the same two writers.
CASES = ("double", "dual")

#: Each case's slot in the stage's sampler-seed block, fixed BY CASE and not by
#: position in :data:`CASES`.
#:
#: An index-based seed makes a run's sampling sequence a function of where its
#: case sits in a list, so a case leaving the ladder renumbers the ones that
#: stay: the emitted file would still look like the file that produced the runs
#: on disk while quietly naming different seeds.  The combination stage learned
#: this the expensive way and hashes its identity instead; here the ladder is
#: short enough that a table of slots says it plainly.  A slot is retired with
#: its case and never reused.
SEED_SLOTS = {"double": 1, "dual": 2}


def case_writers(ranked: List[str], case: str) -> List[str]:
    """
    The writers one case is built from, taken from a worst-first ranking.

    Args:
        ranked: The cohort's writers, worst g-0 accuracy first.
        case: One of :data:`CASES`.

    Returns:
        The two worst writers, for either case.  That ``double`` and ``dual``
        share this line is the point of the pair: they are the same rows, and
        only the client boundary between them differs.
    """
    if case not in CASES:
        raise ValueError(f"Unknown extreme case {case!r}; expected one of {CASES}")
    need = 2
    if len(ranked) < need:
        raise ValueError(
            f"The {case!r} case needs {need} writer(s); the ranking has "
            f"{len(ranked)}."
        )
    return list(ranked[:need])


def case_clients(ranked: List[str], case: str) -> List[str]:
    """
    The **client** list of one case, which is not the same as its writers.

    ``dual`` merges its two writers into one client, so it has two writers and
    one client; ``double`` has two of each.  This is the whole distinction the
    pair exists to measure.
    """
    from federated_outlier_adaptation.data.merged_clients import merged_id

    writers = case_writers(ranked, case)
    if case == "dual":
        return [merged_id(writers)]
    return writers


def extreme_cells(ranked: List[str]) -> List[Dict[str, Any]]:
    """One cell per case, in ladder order."""
    notes = {
        "double": "a federation of two clients: the two worst distinct writers",
        "dual": (
            "a federation of one client holding both of those writers' rows "
            "merged - the same images as 'double', one participant"
        ),
    }
    cells = []
    for case in CASES:
        clients = case_clients(ranked, case)
        cells.append({
            "id": case,
            "case": case,
            "writers": case_writers(ranked, case),
            "clients": clients,
            "participants": len(clients),
            "note": notes[case],
        })
    return cells


def accuracy_map(payload) -> Dict[str, float]:
    """
    ``{writer: accuracy}`` from either shape ``clients_acc_on_global.json`` takes.

    The published pipeline writes it as a **list of one-key dicts**
    (``[{"f0000_14": 0.81}, ...]``); other readers in this codebase build the
    flat mapping directly.  Both are accepted, because guessing wrong here would
    not fail loudly - it would rank the cohort by nothing and pick whichever
    writer sorted first.
    """
    if isinstance(payload, dict):
        return {str(k): float(v) for k, v in payload.items()}
    flat: Dict[str, float] = {}
    for entry in payload or []:
        if isinstance(entry, dict):
            for key, value in entry.items():
                flat[str(key)] = float(value)
    return flat


def rank_cohort(accuracies: Dict[str, float], cohort: List[str]) -> List[str]:
    """
    The cohort's writers, worst g-0 accuracy first, ties broken by writer id.

    Restricted to the cohort on purpose: the extreme cases are drawn from the
    twenty writers every other v6 stage federates, so that a result here is
    comparable with the ladder rather than being about some other writer that
    happens to score badly.

    Raises:
        KeyError: a cohort writer has no recorded accuracy, which would make the
            ranking silently depend on which writers happen to be scored.
    """
    accuracies = accuracy_map(accuracies)
    missing = [writer for writer in cohort if writer not in accuracies]
    if missing:
        raise KeyError(
            f"No g-0 accuracy recorded for {missing}; the extreme-case ranking "
            "needs every cohort writer scored."
        )
    return sorted(cohort, key=lambda writer: (float(accuracies[writer]), writer))
