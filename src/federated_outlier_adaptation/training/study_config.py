"""
The study this pipeline builds for, as data rather than as scattered constants.

One study is live: ``Digits_study01`` - NIST SD19 digits 0-9, ten outlier
clients, two hundred old writers, nine of ten training per round.

The multi-study grid that used to live here is gone with the 62-class
programme; its folders were deleted, so every configuration in it named a root
that no longer exists.  What survives is the shape - a study is a *population*
and a *participation rate*, and everything else about a run is protocol - and
the one instance that is real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


#: The share of clients a round drops.  The study's own rate; the dropout stages
#: vary it and nothing else.
DROPOUT_RATE = 0.1

#: Guards the float.  ``(1 - 0.8) * 10`` is ``1.9999999999999996`` in IEEE754,
#: and a bare floor of it is 1 where the arithmetic says 2.  Over every
#: federation size to a hundred and every whole-percent rate there are 56 such
#: cases, and they are silent: the run simply trains one client fewer than the
#: rate says.  The nudge
#: is far smaller than the spacing between whole client counts, so it can only
#: repair that representation error - swept over the same grid it moves no
#: participation the exact arithmetic does not move.
_FLOOR_EPS = 1e-9


def participants(n_clients: int, dropout: float = DROPOUT_RATE) -> int:
    """
    How many of ``n_clients`` a round trains on, at a given dropout rate::

        participants = floor((1 - dropout) * n)

    Reads directly: at ten clients and ten percent dropout, nine train; at
    twenty and twenty percent, sixteen; at five and ten percent, four.  A
    fractional client is not trained on, so the kept count is the one that
    floors.

    The extreme cases are an explicit exception
    -----------------------------------------
    At one or two clients this rule floors to zero or one, and dropping a client
    from a two-client federation is not a participation study - it is a coin
    flip on whether the round happens at all.  So the ``single``/``double``/
    ``dual`` cases do not use this function: they run at **full participation**,
    ``--policy all --participation 1.0``, which is what
    :func:`~.study_lines.extreme_line` already emits and says in so many words.
    This function refuses those sizes rather than returning a number that would
    quietly stand in for that decision.

    Args:
        n_clients: Size of the federation.
        dropout: Share of it dropped per round, in ``[0, 1)``.

    Returns:
        The number of clients a round draws.

    Raises:
        ValueError: when the rule leaves nobody to train - which is the signal
            to use the documented full-participation exception, not to round up.
    """
    import math

    if n_clients < 1:
        raise ValueError(f"A federation needs at least one client, got {n_clients!r}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
    kept = math.floor((1.0 - dropout) * n_clients + _FLOOR_EPS)
    if kept < 1:
        raise ValueError(
            f"floor((1 - {dropout:g}) * {n_clients}) is {kept}: this rule leaves "
            "nobody to train. A federation this small runs at FULL "
            "participation (--policy all --participation 1.0), which is the "
            "documented exception the extreme cases use, not a rounding-up of "
            "this rule."
        )
    return kept


@dataclass(frozen=True)
class StudyConfig:
    """One design point, and everything a task line needs to address it."""

    name: str
    cohort_size: int
    old_size: int
    clients_per_round: int
    #: ``digits`` (10 labels, McMahan's head) or ``all`` (62).  Meaningless for a
    #: provider that is not NIST, and omitted from its lines.
    classes: str = "digits"
    #: Which dataset the study federates.  ``nist`` carries the image flags;
    #: anything else names itself and lets the provider supply its own shape.
    provider: str = "nist"
    #: The topology every line declares.  The provider ultimately chooses it -
    #: a sequence dataset has no use for a CNN - but a line that declared the
    #: wrong one would still run and would be recorded, in its own provenance,
    #: as something it is not.
    model: str = "fedavg_cnn"
    #: Whether the study cross-validates.  The digit study does; the transfer
    #: studies deliberately do not - one g-0, one client split - so that a
    #: second modality costs a probe rather than a second full programme.
    cross_validated: bool = True
    #: Prefix of every output folder, so two studies cannot collide in one root.
    tag: str = "d01"
    #: Base of every sampler seed this study emits.  Stages add offsets of a few
    #: thousand, so bases are spaced far apart - and this one sits well clear of
    #: the hand-written seeds the earlier digit stages used.
    seed_base: int = 700000
    #: How the cohort was chosen, for the record.
    cohort_rule: str = "worst"

    @property
    def cohort_file_name(self) -> str:
        return f"cohort_worst{self.cohort_size}.json"

    @property
    def cohort_book_name(self) -> str:
        return f"cohort{self.cohort_size}"

    @property
    def folds(self) -> tuple:
        """
        The folds a run of this study uses.

        A book is always **built** with five folds - preservation is the mean
        over five old-fold test partitions either way, and that is what keeps
        the metric comparable across studies.  What a cross-validated study does
        and a transfer study does not is *train* on all five.
        """
        return (1, 2, 3, 4, 5) if self.cross_validated else (1,)

    @property
    def dropout(self) -> float:
        return 1.0 - self.clients_per_round / self.cohort_size

    def describe(self) -> str:
        return (
            f"{self.cohort_size} outlier clients; {self.old_size} old writers; "
            f"{self.clients_per_round} of {self.cohort_size} per round "
            f"({self.dropout:.0%} dropout); classes={self.classes}"
        )


DIGITS_STUDY01 = StudyConfig(
    name="Digits_study01",
    cohort_size=10,
    old_size=200,
    clients_per_round=9,
    classes="digits",
    tag="d01",
    seed_base=700000,
)


#: Every study this pipeline knows about.
STUDIES: Dict[str, StudyConfig] = {
    DIGITS_STUDY01.name: DIGITS_STUDY01,
}

# The study's own participation is the formula's, not a second statement of it.
# If the two ever part company, every stage already run was run at a rate the
# code no longer describes, and that has to fail here rather than in a table.
assert DIGITS_STUDY01.clients_per_round == participants(
    DIGITS_STUDY01.cohort_size, DROPOUT_RATE
), "the study's clients_per_round no longer matches the participation rule"


def config(name: str) -> StudyConfig:
    """The configuration of one study, by name."""
    if name not in STUDIES:
        raise KeyError(f"Unknown study {name!r}; this pipeline knows {sorted(STUDIES)}")
    return STUDIES[name]
