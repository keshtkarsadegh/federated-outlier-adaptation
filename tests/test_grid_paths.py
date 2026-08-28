"""Sweep outputs must follow the provider's results subtree."""

from pathlib import Path

from federated_outlier_adaptation import config
from federated_outlier_adaptation.grid_search.common import grid_output_dir


def test_grid_output_dir_follows_provider(monkeypatch, tmp_path):
    """A non-default provider writes under its own root, not the NIST one."""
    monkeypatch.setattr(
        config, "provider_results_dir", lambda name="nist": tmp_path / name
    )

    nist = grid_output_dir("anchored_kd_frozen_grid_search", "concurrent", "weights", 1)
    other = grid_output_dir(
        "anchored_kd_frozen_grid_search",
        "concurrent",
        "weights",
        1,
        provider_name="shakespeare",
    )

    assert nist == tmp_path / "nist" / "anchored_kd_frozen_grid_search" / "seed_1" / "concurrent_weights"
    assert other == tmp_path / "shakespeare" / "anchored_kd_frozen_grid_search" / "seed_1" / "concurrent_weights"
    assert nist != other


def test_explicit_results_dir_wins(tmp_path):
    """An explicit root overrides the provider lookup (resume helper relies on it)."""
    path = grid_output_dir(
        "freeze_grid_search",
        "sequential",
        "delta",
        None,
        results_dir=tmp_path,
        provider_name="cifar10",
    )
    assert path == Path(tmp_path) / "freeze_grid_search" / "sequential_delta"
