"""
Constrained, validation-based configuration selection.

The rule is "most adaptive configuration whose source accuracy stays within
eps of where it started", decided on validation data; the tests build small
synthetic result trees so the arithmetic is checkable by hand.
"""

from __future__ import annotations

import csv
import json

import pytest

from federated_outlier_adaptation.grid_search import selection


def write_summary(root, folder, entries):
    """Write a ``summary_0.json`` shaped like the driver's output."""
    directory = root / folder
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "summary_0.json", "w") as handle:
        json.dump(entries, handle)
    return directory


def entry(heldout, source_val, **extra):
    block = {
        "heldout_client_accuracies": heldout,
        "source_val_accuracies": source_val,
        "pool_test_accuracies": heldout,
        "pool_val_accuracies": heldout,
        "accuracies": [[a, b] for a, b in zip(heldout, source_val)],
        "config": {"trainer": "BaseTrainer"},
    }
    block.update(extra)
    return block


@pytest.fixture()
def tree(tmp_path):
    """Three configurations with a clear ordering."""
    write_summary(
        tmp_path,
        "run_a",
        {
            # adaptive but forgets 0.02 - outside the budget
            "greedy": entry([0.5] * 5 + [0.90] * 5, [0.99] * 5 + [0.97] * 5),
            # slightly less adaptive, forgets 0.002 - inside the budget
            "balanced": entry([0.5] * 5 + [0.85] * 5, [0.99] * 5 + [0.988] * 5),
            # conservative, forgets nothing
            "cautious": entry([0.5] * 5 + [0.70] * 5, [0.99] * 10),
        },
    )
    return tmp_path


def test_records_are_collected_from_summaries(tree):
    records = selection.collect_records(tree)
    names = {record.name.split("/")[-1] for record in records}
    assert names == {"greedy", "balanced", "cautious"}
    for record in records:
        assert record.complete


def test_adaptation_is_the_tail_mean(tree):
    records = {r.name.split("/")[-1]: r for r in selection.collect_records(tree)}
    assert records["greedy"].adaptation == pytest.approx(0.90)
    assert records["cautious"].adaptation == pytest.approx(0.70)


def test_forgetting_is_measured_against_round_zero(tree):
    records = {r.name.split("/")[-1]: r for r in selection.collect_records(tree)}
    assert records["greedy"].source_reference == pytest.approx(0.99)
    assert records["greedy"].forgetting == pytest.approx(0.02)
    assert records["cautious"].forgetting == pytest.approx(0.0)


def test_the_budget_excludes_the_greedy_configuration(tree):
    result = selection.select_constrained(tree, eps=0.005)
    assert result["selected"]["name"].endswith("balanced")
    assert result["num_feasible"] == 2


def test_a_looser_budget_admits_the_greedy_configuration(tree):
    result = selection.select_constrained(tree, eps=0.05)
    assert result["selected"]["name"].endswith("greedy")


def test_a_tighter_budget_falls_back_to_the_cautious_one(tree):
    result = selection.select_constrained(tree, eps=0.001)
    assert result["selected"]["name"].endswith("cautious")


def test_no_feasible_configuration_selects_nothing(tree):
    result = selection.select_constrained(tree, eps=-1.0)
    assert result["selected"] is None
    assert result["num_feasible"] == 0


def test_the_pareto_front_keeps_the_non_dominated_configurations(tree):
    front = selection.pareto_front(selection.collect_records(tree))
    names = [record.name.split("/")[-1] for record in front]
    # every configuration trades adaptation for forgetting, so none dominates
    assert set(names) == {"cautious", "balanced", "greedy"}
    forgetting = [record.forgetting for record in front]
    assert forgetting == sorted(forgetting)


def test_a_dominated_configuration_is_dropped(tmp_path):
    write_summary(
        tmp_path,
        "run",
        {
            # strictly worse on both axes: less adaptive and forgets more
            "good": entry([0.9] * 5, [0.99] * 5),
            "worse": entry([0.8] * 5, [0.99, 0.99, 0.99, 0.99, 0.97]),
        },
    )
    front = selection.pareto_front(selection.collect_records(tmp_path))
    assert [r.name.split("/")[-1] for r in front] == ["good"]


def test_outputs_are_written(tree):
    selection.select_constrained(tree, eps=0.005)
    with open(tree / "constrained_selection.json") as handle:
        payload = json.load(handle)
    assert payload["eps"] == 0.005
    with open(tree / "pareto_front.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and "adaptation" in rows[0] and "forgetting" in rows[0]


def test_grid_config_blocks_are_read(tmp_path):
    """Sweeps store the series inside the config block's population entry."""
    directory = tmp_path / "sweep" / "concurrent_weights"
    directory.mkdir(parents=True)
    with open(directory / "config_points_100.json", "w") as handle:
        json.dump(
            {
                "config": {
                    "cfg_a": {
                        "trainer": "AnchoredTrainer",
                        "population": {
                            "heldout_client_accuracies": [0.8] * 5,
                            "source_val_accuracies": [0.99] * 5,
                        },
                    }
                }
            },
            handle,
        )
    records = selection.collect_records(tmp_path)
    assert len(records) == 1
    assert records[0].adaptation == pytest.approx(0.8)
    assert records[0].forgetting == pytest.approx(0.0)


def test_runs_without_the_new_series_are_ignored(tmp_path):
    write_summary(tmp_path, "legacy", {"old": {"accuracies": [[0.5, 0.9]]}})
    assert selection.collect_records(tmp_path) == []


def test_empty_tree_is_handled(tmp_path):
    result = selection.select_constrained(tmp_path, eps=0.005)
    assert result["num_records"] == 0
    assert result["selected"] is None
