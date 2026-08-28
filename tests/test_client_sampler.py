"""
Per-round client selection: participation counts, determinism and the promise
that the default configuration changes nothing.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.runners.client_sampler import POLICIES, ClientSampler

POOL = [f"c{i}" for i in range(10)]


def test_default_policy_returns_the_whole_pool_in_order():
    sampler = ClientSampler(POOL)
    assert not sampler.is_active
    for round_index in range(4):
        assert sampler.select(round_index) == POOL


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        ClientSampler(POOL, policy="best_first")


@pytest.mark.parametrize("participation", [0.0, -0.5, 1.5])
def test_participation_must_be_a_fraction(participation):
    with pytest.raises(ValueError):
        ClientSampler(POOL, participation=participation, policy="uniform")


@pytest.mark.parametrize("policy", ["uniform", "worst_first", "round_robin"])
@pytest.mark.parametrize("participation,expected", [(0.2, 2), (0.5, 5), (1.0, 10)])
def test_participation_count(policy, participation, expected):
    sampler = ClientSampler(POOL, participation=participation, policy=policy, seed=1)
    assert len(sampler.select(0)) == expected


def test_at_least_one_client_is_drawn():
    sampler = ClientSampler(["a", "b"], participation=0.1, policy="uniform", seed=0)
    assert len(sampler.select(0)) == 1


def test_uniform_is_deterministic_given_the_seed():
    first = ClientSampler(POOL, participation=0.5, policy="uniform", seed=7)
    second = ClientSampler(POOL, participation=0.5, policy="uniform", seed=7)
    other = ClientSampler(POOL, participation=0.5, policy="uniform", seed=8)

    first_rounds = [first.select(r) for r in range(5)]
    second_rounds = [second.select(r) for r in range(5)]
    other_rounds = [other.select(r) for r in range(5)]

    assert first_rounds == second_rounds
    assert first_rounds != other_rounds
    for chosen in first_rounds:
        assert len(set(chosen)) == len(chosen), "uniform draws without replacement"


def test_uniform_varies_between_rounds():
    sampler = ClientSampler(POOL, participation=0.3, policy="uniform", seed=3)
    rounds = [tuple(sampler.select(r)) for r in range(8)]
    assert len(set(rounds)) > 1


def test_worst_first_picks_the_lowest_accuracies():
    sampler = ClientSampler(POOL, participation=0.3, policy="worst_first", seed=0)
    accuracies = {client: 0.9 for client in POOL}
    accuracies["c4"] = 0.1
    accuracies["c7"] = 0.2
    accuracies["c1"] = 0.3
    chosen = sampler.select(1, accuracies)
    assert chosen == ["c4", "c7", "c1"]


def test_worst_first_without_accuracies_is_random_but_seeded():
    first = ClientSampler(POOL, participation=0.5, policy="worst_first", seed=5)
    second = ClientSampler(POOL, participation=0.5, policy="worst_first", seed=5)
    assert first.select(0, {}) == second.select(0, {})


def test_worst_first_breaks_ties_randomly():
    """All-equal accuracies must not always yield the same clients."""
    accuracies = {client: 0.5 for client in POOL}
    sampler = ClientSampler(POOL, participation=0.3, policy="worst_first", seed=11)
    rounds = {tuple(sampler.select(r, accuracies)) for r in range(10)}
    assert len(rounds) > 1


def test_round_robin_cycles_through_the_pool():
    sampler = ClientSampler(POOL, participation=0.2, policy="round_robin")
    assert sampler.select(0) == ["c0", "c1"]
    assert sampler.select(1) == ["c2", "c3"]
    assert sampler.select(2) == ["c4", "c5"]
    seen = ["c0", "c1", "c2", "c3", "c4", "c5"]
    for round_index in range(3, 5):
        seen += sampler.select(round_index)
    assert set(seen) == set(POOL)
    # wraps around
    assert sampler.select(5) == ["c0", "c1"]


def test_empty_pool_selects_nothing():
    sampler = ClientSampler([], participation=0.5, policy="uniform", seed=0)
    assert sampler.select(0) == []


def test_every_policy_is_reachable():
    for policy in POLICIES:
        sampler = ClientSampler(POOL, participation=0.5, policy=policy, seed=0)
        assert sampler.select(0)
