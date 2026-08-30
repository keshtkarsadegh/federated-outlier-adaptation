"""
The selection must survive being run twice.

A rebuild from a fresh results root re-derived a cohort overlapping the
published one by three writers of ten. Neither cause was randomness in the
training: a crowned detector fold decided the ranking, and a seeded
``random.sample`` reshuffled the whole draw whenever the pool moved slightly.
These tests pin both repairs.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.outliers.scoring import (
    _draw_key,
    average_score_files,
    draw_old_data,
)


def _pool(n, prefix="w"):
    return [f"{prefix}{i:05d}" for i in range(n)]


def _draw(pool, size=200, seed=42):
    return sorted(sorted(pool, key=lambda w: _draw_key(seed, w))[:size])


# ------------------------------------------------------------------ the draw
def test_the_draw_is_reproducible_from_the_seed():
    pool = _pool(2000)
    assert _draw(pool) == _draw(list(reversed(pool)))


def test_the_draw_moves_only_as_much_as_the_pool():
    """
    The failure this replaces: a pool that overlapped ninety percent produced a
    draw that overlapped five. A writer's place must depend on its own id.
    """
    pool = _pool(2500)
    thinned = [w for i, w in enumerate(pool) if i % 10]      # drop ten percent
    full, part = set(_draw(pool)), set(_draw(thinned))
    assert len(full & part) >= 170        # ~90% of 200, not ~5%


def test_removing_a_writer_does_not_move_the_others():
    pool = _pool(500)
    before = _draw(pool, size=50)
    victim = next(w for w in pool if w not in before)
    after = _draw([w for w in pool if w != victim], size=50)
    assert before == after


def test_different_seeds_give_different_draws():
    pool = _pool(1000)
    assert _draw(pool, seed=1) != _draw(pool, seed=2)


def test_draw_old_data_uses_the_stable_rule():
    counts = {w: 120 for w in _pool(600)}
    payload = draw_old_data(counts, exclude=[], size=100, seed=7, min_samples=100)
    assert payload["size"] == 100
    assert len(payload["clients"]) == 100
    assert "sha256" in payload["rule"]
    again = draw_old_data(counts, exclude=[], size=100, seed=7, min_samples=100)
    assert payload["clients"] == again["clients"]


def test_draw_old_data_is_stable_when_the_population_shrinks():
    counts = {w: 120 for w in _pool(600)}
    first = draw_old_data(counts, exclude=[], size=100, seed=7, min_samples=100)
    smaller = {w: c for i, (w, c) in enumerate(sorted(counts.items())) if i % 10}
    second = draw_old_data(smaller, exclude=[], size=100, seed=7, min_samples=100)
    kept = set(first["clients"]) & set(second["clients"])
    assert len(kept) >= 80


# ------------------------------------------------------------- the averaging
def test_the_ranking_is_the_mean_over_folds():
    merged = average_score_files([
        {"scores": [{"a": 0.8}, {"b": 0.6}]},
        {"scores": [{"a": 1.0}, {"b": 0.4}]},
    ])
    flat = {k: v for entry in merged["scores"] for k, v in entry.items()}
    assert flat["a"] == pytest.approx(0.9)
    assert flat["b"] == pytest.approx(0.5)
    assert merged["folds"] == 2


def test_a_writer_missing_from_one_fold_is_dropped():
    """
    A mean over a varying number of folds would rank writers on different
    evidence, which is the kind of quiet inconsistency the crowning had.
    """
    merged = average_score_files([
        {"scores": [{"a": 0.8}, {"b": 0.6}]},
        {"scores": [{"a": 1.0}]},
    ])
    flat = {k: v for entry in merged["scores"] for k, v in entry.items()}
    assert set(flat) == {"a"}
    assert merged["dropped_incomplete"] == ["b"]


def test_the_spread_across_folds_is_reported():
    """A writer the folds disagree about should be visible, not averaged away."""
    merged = average_score_files([
        {"scores": [{"a": 0.2}]},
        {"scores": [{"a": 1.0}]},
    ])
    assert merged["fold_spread"]["max"] == pytest.approx(0.8)


def test_averaging_does_not_depend_on_the_order_of_the_folds():
    one = {"scores": [{"a": 0.3}, {"b": 0.9}]}
    two = {"scores": [{"a": 0.7}, {"b": 0.1}]}
    assert average_score_files([one, two])["scores"] == \
           average_score_files([two, one])["scores"]
