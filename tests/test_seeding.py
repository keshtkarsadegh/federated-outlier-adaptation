"""
Seeding tests.

With a seed, two identical runs must produce bit-identical accuracy traces.
Without a seed the historical (unseeded) behaviour is kept, which is only
asserted indirectly: the run still completes and no ``seed_*`` component is
added to output paths.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.utils.seeding import (
    make_generator,
    seed_suffix,
    set_run_seed,
)

from .test_runners import AGGREGATIONS, run_simulation


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_seeded_runs_are_identical(synthetic_provider, scenario, metadata, agg_name):
    _, first, _ = run_simulation(
        BaseTrainer, synthetic_provider, scenario, metadata, agg_name, seed=1234
    )
    _, second, _ = run_simulation(
        BaseTrainer, synthetic_provider, scenario, metadata, agg_name, seed=1234
    )
    assert first == second


def test_set_run_seed_is_a_noop_without_seed():
    assert set_run_seed(None) is None
    assert make_generator(None) is None


def test_set_run_seed_returns_the_seed():
    assert set_run_seed(7) == 7
    generator = make_generator(7)
    assert generator is not None
    assert generator.initial_seed() == 7


def test_seed_suffix():
    assert seed_suffix(None) == ""
    assert seed_suffix(42) == "seed_42"
