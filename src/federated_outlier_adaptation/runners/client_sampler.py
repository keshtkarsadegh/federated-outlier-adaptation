"""
Per-round client selection.

The published experiments train every participant in every round.  Larger
client populations make that unrealistic, so the runners obtain the round's
participants from a :class:`ClientSampler` instead.  The default configuration
(``participation=1.0``, ``policy="all"``) returns the full pool in its original
order every round, which is exactly the previous behaviour.

Policies:
    ``all``          every client of the pool, every round (default).
    ``uniform``      a random subset of ``C * N`` clients, drawn per round.
    ``worst_first``  the ``C * N`` clients with the lowest most recent accuracy
                     on their own test split; unknown accuracies and ties are
                     broken randomly.
    ``round_robin``  a cyclic window of ``C * N`` clients over the pool.

All randomised policies draw from a single :class:`random.Random` seeded with
``seed``, so a run is reproducible given the seed and the round order.
"""

from __future__ import annotations

import random
from typing import Iterable, Mapping, Optional, Sequence

POLICIES = ("all", "uniform", "worst_first", "round_robin")


class ClientSampler:
    """
    Choose the participants of one federated round.

    Args:
        pool: Client ids available for training.  Duplicates are allowed and
            are treated as independent participants.
        participation: Fraction ``C`` of the pool taking part per round, in
            ``(0, 1]``.  The number of participants is ``max(1, round(C * N))``.
        clients_per_round: Absolute number ``m`` of clients per round.  When
            given it takes precedence over ``participation``; it is capped at
            the pool size.
        policy: One of :data:`POLICIES`.
        seed: Seed of the sampler's own random number generator.  ``None``
            leaves the draws unseeded.
    """

    def __init__(
        self,
        pool: Sequence[str],
        participation: float = 1.0,
        policy: str = "all",
        seed: Optional[int] = None,
        clients_per_round: Optional[int] = None,
    ):
        if policy not in POLICIES:
            raise ValueError(f"Unknown selection policy {policy!r}; expected one of {POLICIES}")
        if not 0.0 < float(participation) <= 1.0:
            raise ValueError(f"participation must lie in (0, 1], got {participation!r}")
        if clients_per_round is not None and int(clients_per_round) < 1:
            raise ValueError(f"clients_per_round must be >= 1, got {clients_per_round!r}")

        self.pool = list(pool)
        self.participation = float(participation)
        self.clients_per_round = int(clients_per_round) if clients_per_round is not None else None
        self.policy = policy
        self.seed = seed
        self._random = random.Random(seed)
        self._cursor = 0

    # ------------------------------------------------------------------ helpers
    @property
    def is_active(self) -> bool:
        """Whether the sampler changes anything compared to the published runs."""
        if self.clients_per_round is not None and self.clients_per_round < len(self.pool):
            return True
        return self.policy != "all" or self.participation != 1.0

    def num_participants(self) -> int:
        """Number of clients drawn per round."""
        if not self.pool:
            return 0
        if self.clients_per_round is not None:
            return max(1, min(len(self.pool), self.clients_per_round))
        if self.policy == "all":
            return len(self.pool)
        return max(1, min(len(self.pool), round(self.participation * len(self.pool))))

    # ------------------------------------------------------------------ policies
    def _uniform(self, count: int) -> list[str]:
        indices = self._random.sample(range(len(self.pool)), count)
        return [self.pool[i] for i in indices]

    def _worst_first(self, count: int, accuracies: Optional[Mapping[str, float]]) -> list[str]:
        accuracies = accuracies or {}
        order = list(range(len(self.pool)))
        # Shuffle first so that unknown accuracies and ties are broken randomly;
        # the sort below is stable and therefore preserves this random order.
        self._random.shuffle(order)
        order.sort(key=lambda i: accuracies.get(self.pool[i], float("inf")))
        return [self.pool[i] for i in order[:count]]

    def _round_robin(self, count: int) -> list[str]:
        size = len(self.pool)
        chosen = [self.pool[(self._cursor + offset) % size] for offset in range(count)]
        self._cursor = (self._cursor + count) % size
        return chosen

    # -------------------------------------------------------------------- select
    def select(
        self,
        round_index: int = 0,
        accuracies: Optional[Mapping[str, float]] = None,
    ) -> list[str]:
        """
        Participants of round ``round_index``.

        Args:
            round_index: Zero-based round counter (used by the cyclic policy).
            accuracies: Most recent ``{client: accuracy}`` mapping, consumed by
                the ``worst_first`` policy.

        Returns:
            list[str]: Client ids in the order they should be trained.
        """
        if not self.pool:
            return []
        count = self.num_participants()
        if self.policy == "all" and count == len(self.pool):
            return list(self.pool)

        if self.policy == "all":
            # An absolute count smaller than the pool turns "all" into a
            # deterministic cyclic window over the pool.
            return self._round_robin(count)
        if self.policy == "uniform":
            return self._uniform(count)
        if self.policy == "worst_first":
            return self._worst_first(count, accuracies)
        return self._round_robin(count)


def build_sampler(
    pool: Iterable[str],
    participation: float = 1.0,
    policy: str = "all",
    seed: Optional[int] = None,
    clients_per_round: Optional[int] = None,
) -> ClientSampler:
    """Convenience constructor used by the runners."""
    return ClientSampler(
        pool=list(pool),
        participation=participation,
        policy=policy,
        seed=seed,
        clients_per_round=clients_per_round,
    )
