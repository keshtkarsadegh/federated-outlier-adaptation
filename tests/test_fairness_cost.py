"""
The fairness and cost views: the arithmetic, and the payload they read.

Two things can go wrong here and neither one fails loudly. The distribution can
be computed with a percentile convention that nobody can reproduce by hand, and
the shipped-model reference can be read from a book that covers a *different*
test split of the same writers - which reports a lift that is really the gap
between two fold books. Both are pinned below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import fairness_cost  # noqa: E402


# ------------------------------------------------------------- arithmetic
def test_percentile_interpolates_the_way_a_reader_would():
    """Ten values have no exact quartile; the convention is linear interpolation."""
    sample = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert fairness_cost.percentile(sample, 0.0) == 0.0
    assert fairness_cost.percentile(sample, 0.5) == 2.0
    assert fairness_cost.percentile(sample, 1.0) == 4.0
    assert fairness_cost.percentile(sample, 0.25) == 1.0
    assert fairness_cost.percentile([0.0, 1.0], 0.25) == 0.25


def test_percentile_does_not_need_a_sorted_input():
    assert fairness_cost.percentile([4.0, 0.0, 2.0], 0.5) == 2.0


def test_percentile_of_one_client_is_that_client():
    """The extreme cases run a single client, and a single client has a spread."""
    assert fairness_cost.percentile([0.9375], 0.25) == 0.9375


def test_percentile_refuses_an_empty_sample():
    with pytest.raises(ValueError):
        fairness_cost.percentile([], 0.5)


def test_the_gap_is_the_mean_less_the_worst_served_client():
    """The column the whole table exists for."""
    spread = fairness_cost.distribution([0.5, 0.9, 1.0])
    assert spread["clients"] == 3
    assert spread["min"] == 0.5
    assert spread["max"] == 1.0
    assert spread["mean"] == pytest.approx(0.8)
    assert spread["gap"] == pytest.approx(0.3)


def test_an_even_cohort_lifts_only_its_best_and_the_mean_still_moves():
    """
    A mean can rise while its worst member does not: the failure the fairness
    table is meant to catch, written as a test so the columns cannot both drift.
    """
    before = fairness_cost.distribution([0.6, 0.7, 0.8, 0.9])
    after = fairness_cost.distribution([0.6, 0.7, 0.8, 1.0])
    assert after["mean"] > before["mean"]
    assert after["min"] == before["min"]
    assert after["gap"] > before["gap"]


def test_fold_mean_averages_one_statistic_across_folds():
    rows = [{"min": 0.8}, {"min": 0.9}, {"min": 1.0}]
    assert fairness_cost.fold_mean(rows, "min") == pytest.approx(0.9)


# -------------------------------------------------------------- reference
def test_the_smallest_covering_book_wins():
    """
    Two books cover the cohort and they disagree, because they cut different
    test splits. The one that was written for this cohort is the smaller.
    """
    books = {
        "g0_c20_evaluations.json": {"a": 0.10, "b": 0.10, "c": 0.10, "d": 0.10},
        "g0_perfold_evaluations.json": {"a": 0.90, "b": 0.90},
    }
    assert fairness_cost.pick_reference(["a", "b"], books) == {"a": 0.90, "b": 0.90}


def test_a_cohort_no_book_covers_gets_no_reference():
    """
    The merged writer of the ``dual`` extreme is one client and no book has it.
    Borrowing either half's number would report a lift that was never measured.
    """
    books = {"g0_perfold_evaluations.json": {"f0001_00": 0.8, "f0002_00": 0.7}}
    assert fairness_cost.pick_reference(["f0001_00+f0002_00"], books) == {}


def test_a_partially_covering_book_is_not_used():
    books = {"g0_perfold_evaluations.json": {"a": 0.8}}
    assert fairness_cost.pick_reference(["a", "b"], books) == {}


def test_share_improved_counts_clients_against_their_own_reference():
    finals = {"a": 0.9, "b": 0.5, "c": 0.7}
    reference = {"a": 0.8, "b": 0.6, "c": 0.7}
    assert fairness_cost.share_improved(finals, reference) == pytest.approx(1 / 3)


def test_share_improved_is_not_a_number_when_nothing_is_paired():
    """Not zero: nothing was compared, and zero would read as nobody improved."""
    unpaired = fairness_cost.share_improved({"a": 0.9}, {})
    assert unpaired != unpaired


# ------------------------------------------------------------------ cost
def test_seconds_are_billed_per_folder_not_per_payload():
    """
    A stage that computes both schedules writes two payloads from one job. Two
    payloads counted as two tasks halves the seconds per task and doubles the
    task count, and the total stays right, so nothing looks wrong.
    """
    runs = [
        {"folder": "job1", "body": {"round_seconds": [1.0, 1.0],
                                    "comm_bytes_per_round": [10, 10],
                                    "participants": [["a"], ["a"]], "param_count": 7}},
        {"folder": "job1", "body": {"round_seconds": [2.0, 2.0],
                                    "comm_bytes_per_round": [10, 10],
                                    "participants": [["a"], ["a"]], "param_count": 7}},
    ]
    row = fairness_cost.cost_row(runs)
    assert row["tasks"] == 1
    assert row["payloads"] == 2
    assert row["seconds_per_task"] == pytest.approx(6.0)
    assert row["seconds_per_round"] == pytest.approx(1.5)
    assert row["hours"] == pytest.approx(6.0 / 3600)
    assert row["gb_per_task"] == pytest.approx(40 / 1e9)
    assert row["param_count"] == 7


def test_a_run_with_no_rounds_costs_nothing_and_says_so():
    assert fairness_cost.cost_row([{"folder": "job1", "body": {}}]) == {}


# ------------------------------------------------------- a synthetic root
def _write_run(root: Path, folder: str, fold: int, per_client: dict, seconds: float) -> None:
    path = root / folder / f"fold{fold}_seed_1" / "concurrent_delta"
    path.mkdir(parents=True)
    (path / "summary_0.json").write_text(json.dumps({
        "cell": {
            "round_seconds": [seconds] * 4,
            "client_seconds": [[seconds]] * 4,
            "comm_bytes_per_round": [1000] * 4,
            "participants": [sorted(per_client)] * 4,
            "param_count": 125,
            "final_evaluation": {
                "fold": fold,
                "clients": {"accuracy": sum(per_client.values()) / len(per_client),
                            "per_client": per_client},
                "old": {"mean": 0.99},
            },
        }
    }))


@pytest.fixture
def study(tmp_path: Path) -> Path:
    """
    A study root with one arm over three folds, and the shipped model's book.

    ``c`` is the client the cohort was assembled for: it starts worst and the
    arm lifts it. ``a`` starts best and barely moves.
    """
    root = tmp_path / "study"
    root.mkdir()
    (root / "g0_perfold_evaluations.json").write_text(json.dumps({
        f"cohort_fold{f}": {"fold": f, "accuracy": 0.7, "per_writer":
                            {"a": 0.80, "b": 0.70, "c": 0.60}}
        for f in (1, 2, 3)
    }))
    (root / "g0_evaluations.json").write_text(json.dumps({
        f"old_fold{f}": {"fold": f, "accuracy": 0.99} for f in (1, 2, 3)
    }))
    for fold in (1, 2, 3):
        _write_run(root, f"d01_regfull_concurrent_x_fold{fold}_T_grid_search", fold,
                   {"a": 0.82, "b": 0.80, "c": 0.75}, 2.0)
    return root


def test_the_fairness_row_reads_the_stored_per_client_numbers(study: Path):
    books = fairness_cost.g0_books(study)
    rows = fairness_cost.fairness_rows(study, "d01_regfull_", books, folds=3)
    assert len(rows) == 1
    row = rows[0]
    assert row["cell"] == "x"
    assert row["family"] == "parallel"
    assert row["clients"] == 3
    assert row["min"] == pytest.approx(0.75)
    assert row["max"] == pytest.approx(0.82)
    assert row["mean"] == pytest.approx((0.82 + 0.80 + 0.75) / 3)
    assert row["g0_min"] == pytest.approx(0.60)
    assert row["lift_worst"] == pytest.approx(0.15)
    assert row["improved"] == pytest.approx(1.0)


def test_an_arm_short_of_folds_is_dropped(study: Path):
    books = fairness_cost.g0_books(study)
    assert fairness_cost.fairness_rows(study, "d01_regfull_", books, folds=5) == []


def test_the_client_rows_name_the_worst_served_writer_first(study: Path):
    books = fairness_cost.g0_books(study)
    rows = fairness_cost.client_rows(study, "d01_regfull_", books, folds=3)
    assert [r["client"] for r in rows] == ["c", "b", "a"]
    assert rows[0]["delta"] == pytest.approx(0.15)


def test_an_arm_that_writes_two_payloads_is_billed_once(study: Path):
    """
    ``d01_c10d10_control`` runs both schedules from one job. The fairness table
    splits it into two rows because the two schedules are two results; the cost
    table must not, because they were one allocation.
    """
    root = study
    for fold in (1, 2, 3):
        folder = root / f"d01_c10d10_control_fold{fold}_T_grid_search"
        _write_run(root, folder.name, fold, {"a": 0.8, "b": 0.8, "c": 0.8}, 3.0)
        second = folder / f"fold{fold}_seed_1" / "sequential_weights"
        second.mkdir(parents=True)
        (second / "summary_0.json").write_text(
            (folder / f"fold{fold}_seed_1" / "concurrent_delta" / "summary_0.json").read_text()
        )
    rows = fairness_cost.cost_by_arm(root, "d01")
    assert [r["arm"] for r in rows] == ["c10d10/control"]
    assert rows[0]["tasks"] == 3
    assert rows[0]["payloads"] == 6
    assert rows[0]["seconds_per_task"] == pytest.approx(24.0)


def test_the_cost_row_is_read_from_the_same_root(study: Path):
    rows = fairness_cost.cost_by_stage(study, "d01")
    assert [r["stage"] for r in rows] == ["regularisation finals"]
    assert rows[0]["tasks"] == 3
    assert rows[0]["rounds"] == 4
    assert rows[0]["seconds_per_task"] == pytest.approx(8.0)
    assert rows[0]["mb_per_round"] == pytest.approx(0.001)


def test_the_view_runs_end_to_end(study: Path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["fairness_cost.py", "--root", str(study), "--folds", "3"])
    assert fairness_cost.main() == 0
    printed = capsys.readouterr().out
    assert "PER-CLIENT SPREAD" in printed
    assert "basis: TEST." in printed
    assert "final_evaluation.clients.accuracy" in printed
    assert "final_evaluation.clients.per_client" in printed
    assert "THE PROGRAMME'S MEASURED COMPUTE" in printed


def test_a_root_without_a_per_writer_book_is_refused(study: Path, monkeypatch):
    (study / "g0_perfold_evaluations.json").write_text(json.dumps({
        "cohort_fold1": {"fold": 1, "accuracy": 0.7}
    }))
    monkeypatch.setattr(sys, "argv",
                        ["fairness_cost.py", "--root", str(study),
                         "--what", "fairness", "--folds", "3"])
    with pytest.raises(SystemExit):
        fairness_cost.main()
