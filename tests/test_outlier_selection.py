"""
Client pool selection: the deterministic ``lowest`` mode added for the larger
client populations, and the file names that keep the published list frozen.
"""

from __future__ import annotations

import json

import pytest

from federated_outlier_adaptation.outliers.selection import (
    get_low_acc_writers,
    get_lowest_acc_writers,
    load_client_accuracies,
    selection_filename,
)

ACCURACIES = [
    {"w0": 0.99},
    {"w1": 0.10},
    {"w2": 0.55},
    {"w3": 0.05},
    {"w4": 0.80},
    {"w5": 0.10},
]


@pytest.fixture()
def accuracy_file(tmp_path):
    path = tmp_path / "clients_acc_on_global.json"
    with open(path, "w") as handle:
        json.dump(ACCURACIES, handle)
    return path


def test_load_client_accuracies_flattens_the_file(accuracy_file):
    pairs = load_client_accuracies(accuracy_file)
    assert len(pairs) == len(ACCURACIES)
    assert ("w3", 0.05) in pairs


def test_lowest_mode_returns_the_k_worst_in_order(accuracy_file, tmp_path):
    target = tmp_path / "pool.json"
    selected = get_lowest_acc_writers(accuracy_file, target, k=3)
    assert selected == ["w3", "w1", "w5"]
    with open(target) as handle:
        assert json.load(handle) == selected


def test_lowest_mode_is_deterministic(accuracy_file, tmp_path):
    first = get_lowest_acc_writers(accuracy_file, tmp_path / "a.json", k=4)
    second = get_lowest_acc_writers(accuracy_file, tmp_path / "b.json", k=4)
    assert first == second


def test_lowest_mode_caps_at_the_population_size(accuracy_file, tmp_path):
    selected = get_lowest_acc_writers(accuracy_file, tmp_path / "all.json", k=50)
    assert len(selected) == len(ACCURACIES)


def test_lowest_mode_ignores_bottom_frac(accuracy_file, tmp_path):
    """``bottom_frac`` would keep a single writer; ``lowest`` must ignore it."""
    selected = get_low_acc_writers(
        accuracy_file,
        tmp_path / "pool.json",
        tmp_path,
        k=4,
        bottom_frac=0.005,
        mode="lowest",
    )
    assert selected == ["w3", "w1", "w5", "w2"]


def test_sample_mode_still_samples_from_the_bottom_fraction(accuracy_file, tmp_path):
    selected = get_low_acc_writers(
        accuracy_file, tmp_path / "pool.json", tmp_path, seed=42, k=5, bottom_frac=0.5
    )
    assert set(selected).issubset({"w3", "w1", "w5"})
    assert len(selected) == 3


def test_unknown_mode_is_rejected(accuracy_file, tmp_path):
    with pytest.raises(ValueError):
        get_low_acc_writers(
            accuracy_file, tmp_path / "pool.json", tmp_path, mode="median"
        )


@pytest.mark.parametrize(
    "k,mode,expected",
    [
        (5, "sample", "selected_outliers.json"),
        (8, "sample", "selected_outliers_k8.json"),
        (5, "lowest", "selected_outliers_k5_lowest.json"),
        (20, "lowest", "selected_outliers_k20_lowest.json"),
        (50, "lowest", "selected_outliers_k50_lowest.json"),
    ],
)
def test_selection_filenames(k, mode, expected):
    assert selection_filename(k=k, mode=mode) == expected


def test_select_lowest_pool_writes_under_the_results_root(tmp_path, monkeypatch):
    from federated_outlier_adaptation.outliers import selection

    outliers_dir = tmp_path / "outliers"
    outliers_dir.mkdir(parents=True)
    with open(outliers_dir / "clients_acc_on_global.json", "w") as handle:
        json.dump(ACCURACIES, handle)

    selected, path = selection.select_lowest_pool(results_dir=tmp_path, k=2)
    assert selected == ["w3", "w1"]
    assert path == outliers_dir / "selected_outliers_k2_lowest.json"

    # An existing pool is reused rather than rewritten.
    again, _ = selection.select_lowest_pool(results_dir=tmp_path, k=2)
    assert again == selected


# --------------------------------------------------------------------------- #
# rule-based pools
# --------------------------------------------------------------------------- #
def test_pool_naming_matches_the_shared_helper():
    from federated_outlier_adaptation.outliers.client_accuracy import pool_file_name
    from federated_outlier_adaptation.outliers.selection import pool_filename

    for frac in (0.05, 0.2, 0.5):
        assert pool_filename(pool_frac=frac) == pool_file_name(frac=frac)
    assert pool_filename(pool_threshold=0.9) == pool_file_name(threshold=0.9)
    assert pool_filename(pool_frac=0.05) == "outlier_pool_frac0.05.json"


def test_pool_is_the_bottom_fraction(accuracy_file, tmp_path):
    from federated_outlier_adaptation.outliers.selection import build_outlier_pool

    payload = build_outlier_pool(
        accuracy_file, tmp_path / "outlier_pool_frac0.5.json", pool_frac=0.5
    )
    assert payload["clients"] == ["w3", "w1", "w5"]
    assert payload["rule"] == "fraction"
    assert payload["size"] == 3
    assert payload["eligible_clients"] == len(ACCURACIES)
    assert payload["accuracies"]["w3"] == pytest.approx(0.05)


def test_pool_threshold_rule(accuracy_file, tmp_path):
    from federated_outlier_adaptation.outliers.selection import build_outlier_pool

    payload = build_outlier_pool(
        accuracy_file, tmp_path / "pool.json", pool_threshold=0.6
    )
    assert set(payload["clients"]) == {"w3", "w1", "w5", "w2"}
    assert payload["rule"] == "threshold"


def test_select_outlier_pool_writes_the_shared_name(tmp_path):
    from federated_outlier_adaptation.outliers import selection

    outliers_dir = tmp_path / "outliers"
    outliers_dir.mkdir(parents=True)
    with open(outliers_dir / "clients_acc_on_global.json", "w") as handle:
        json.dump(ACCURACIES, handle)

    payload, path = selection.select_outlier_pool(results_dir=tmp_path, pool_frac=0.5)
    assert path.name == "outlier_pool_frac0.5.json"
    assert payload["clients"] == ["w3", "w1", "w5"]

    again, _ = selection.select_outlier_pool(results_dir=tmp_path, pool_frac=0.5)
    assert again["clients"] == payload["clients"]


def test_load_client_pool_accepts_both_shapes(tmp_path):
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    plain = tmp_path / "list.json"
    with open(plain, "w") as handle:
        json.dump(["a", "b"], handle)
    clients, accuracies = load_client_pool(plain)
    assert clients == ["a", "b"] and accuracies == {}

    structured = tmp_path / "pool.json"
    with open(structured, "w") as handle:
        json.dump({"clients": ["c"], "accuracies": {"c": 0.1}}, handle)
    clients, accuracies = load_client_pool(structured)
    assert clients == ["c"] and accuracies == {"c": 0.1}
