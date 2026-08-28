"""
Stage 10: the extreme cases, run only with the method that won.

Every earlier stage compared methods.  This one does not: the panel is over, the
winner is the sequential combination stage 8 selected - ``seq_delta_capped``
aggregation with the kd+fisher hybrid at ``mix=0.5`` - and the only thing that
varies here is how few clients the federation has and how their data is
arranged.  Running a method panel again would be answering a question that has
already been answered, at three times the cost.

Three arrangements of the same two writers
------------------------------------------
``single``
    a federation of one client: the cohort's worst writer by g-0 accuracy.  The
    degenerate case - there is nothing to aggregate, so whatever the server rule
    does, it does to one update.
``double``
    a federation of two clients: the two worst distinct writers.
``dual``
    a federation of **one** client whose data is both of those writers' rows
    merged - the same images as ``double``, held by one participant.

``double`` and ``dual`` are the controlled pair.  They hold precisely the same
rows; they differ only in whether the aggregation ever sees them separately.
Any gap between them is what client *boundaries* cost, with the data held
constant - which is a question the twenty-client stages cannot ask, because
there the boundaries and the data always move together.

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

from typing import Any, Dict, List

#: Rounds every case runs for: the reporting horizon.
FULL_ROUNDS = 100

#: The stage-8 combination that won, by cell id.  Its aggregation rule and its
#: penalty are inherited from that table rather than restated here.
WINNING_COMBO = "seq_delta_capped_hybrid_mix0p5"

#: The three arrangements, and how many of the ranked writers each one takes.
CASES = ("single", "double", "dual")


def case_writers(ranked: List[str], case: str) -> List[str]:
    """
    The writers one case is built from, taken from a worst-first ranking.

    Args:
        ranked: The cohort's writers, worst g-0 accuracy first.
        case: One of :data:`CASES`.

    Returns:
        ``single`` takes one writer; ``double`` and ``dual`` take the same two.
        That ``double`` and ``dual`` share this line is the point of the pair.
    """
    if case not in CASES:
        raise ValueError(f"Unknown extreme case {case!r}; expected one of {CASES}")
    if case == "single":
        need = 1
    else:
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
        "single": "a federation of one client: the cohort's worst writer",
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
