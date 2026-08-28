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
``trimmed_0p4`` drops ``int(0.4 * K)`` updates from each end of the ordered
coordinate.  At nine participants that is three from each end and three
survivors; at eight it is three and two; at **five** it is two and **one**, so
the rule degenerates into a single surviving update - a coordinate-wise median
of five rather than an average of anything.  That is a real change in what the
method is, and it belongs in the write-up next to the number it produces.
:func:`trim_survivors` computes it from the same expression the server uses, so
the documented survivor count cannot drift from the applied one.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


#: The four configurations, in report order.  ``regulariser`` of ``None`` means
#: the unmodified baseline: no penalty, plain averaging, both schedules.
CONFIGS: List[Dict[str, Any]] = [
    {
        "id": "winner",
        "family": "concurrent",
        "aggregation": "trimmed_0p4",
        "regulariser": "feature_l2_lam0p1",
        "note": "the cross's strongest concurrent pair",
    },
    {
        "id": "balanced",
        "family": "concurrent",
        "aggregation": "anchor_0p03",
        "regulariser": "ntd_b0p01_t0p5",
        "note": "the concurrent pair that gave up least preservation",
    },
    {
        "id": "sequential",
        "family": "sequential",
        "aggregation": "seq_order_shuffle",
        "regulariser": "feature_l2_lam0p01",
        "note": "the strongest sequential pair",
    },
    {
        "id": "control",
        "family": "both",
        "aggregation": "fedavg",
        "regulariser": None,
        "note": (
            "plain FedAvg, no penalty, both schedules - the baseline that says "
            "whether a change at five clients is the method or the size"
        ),
    },
]


def configs() -> List[Dict[str, Any]]:
    """The configurations this stage runs, in report order."""
    return [dict(cell) for cell in CONFIGS]


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
