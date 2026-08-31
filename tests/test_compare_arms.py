"""
The paired comparison, and the reading error it exists to prevent.

A mean difference is not a result on its own. These tests pin the two things
that make the difference readable: the pairing happens within a fold, and a row
whose sign flips across folds is never labelled consistent.
"""

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import compare_arms as ca  # noqa: E402


# --------------------------------------------------------------------------- #
# the arithmetic
# --------------------------------------------------------------------------- #
def test_the_score_is_gain_less_spend():
    """(adaptation - A0) - (P0 - preservation), in points."""
    assert ca.score(0.9225, 0.9886, 0.8225, 0.9986) == pytest.approx(9.0)
    # a run that changes nothing scores zero
    assert ca.score(0.8225, 0.9986, 0.8225, 0.9986) == pytest.approx(0.0)
    # forgetting is subtracted at w = 1
    assert ca.score(0.8225, 0.9886, 0.8225, 0.9986) == pytest.approx(-1.0)


def test_pairing_is_within_a_fold_not_between_means():
    """
    The folds are shared, so a fold that is hard for one arm is hard for both.

    These two arms have the SAME means, so an unpaired comparison would report
    no difference at all; paired, the left arm is ahead on every fold.
    """
    left = {1: 10.0, 2: 20.0, 3: 30.0}
    right = {1: 9.0, 2: 19.0, 3: 29.0}
    result = ca.paired(left, right)
    assert result["mean"] == pytest.approx(1.0)
    assert result["diffs"] == [1.0, 1.0, 1.0]
    assert result["all_positive"] is True


def test_a_sign_that_flips_is_not_consistent():
    """One good fold and three flat ones is what the combination stage produced."""
    result = ca.paired({1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}, {1: 0.0, 2: 0.0, 3: 0.1, 4: 0.1})
    assert result["mean"] > 0
    assert result["all_positive"] is False
    assert result["consistent"] is False


def test_a_consistently_negative_row_is_flagged_too():
    result = ca.paired({1: 0.0, 2: 0.0}, {1: 1.0, 2: 2.0})
    assert result["consistent"] is True
    assert result["all_positive"] is False


def test_only_shared_folds_are_differenced():
    """An arm missing a fold cannot borrow another arm's."""
    result = ca.paired({1: 5.0, 2: 6.0, 3: 7.0}, {2: 1.0, 3: 2.0})
    assert result["folds"] == [2, 3]
    assert result["diffs"] == [5.0, 5.0]


def test_one_shared_fold_is_refused():
    """A spread cannot be computed from a single pair, so nothing is claimed."""
    assert ca.paired({1: 5.0}, {1: 1.0}) is None
    assert ca.paired({}, {}) is None


# --------------------------------------------------------------------------- #
# reading the study
# --------------------------------------------------------------------------- #
def _run(root: Path, prefix: str, cell: str, fold: int, family: str,
         adaptation: float, preservation: float) -> None:
    d = root / f"{prefix}{cell}_fold{fold}_AnchoredTrainer_grid_search" / "inner"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"accuracies_{fold}.json").write_text(json.dumps({
        "scenario": family,
        "pool_val_accuracies": [0.5, adaptation],
        "final_evaluation": {"old": {"mean": preservation}},
    }))


def test_a_payload_is_claimed_by_the_family_it_says_it_is(tmp_path):
    """The screens write both schedules into one folder."""
    _run(tmp_path, "d01_x_", "cell", 1, "concurrent", 0.90, 0.99)
    _run(tmp_path, "d01_x_", "cell", 2, "sequential", 0.80, 0.99)
    conc = ca.per_fold(tmp_path, "d01_x_", "concurrent", 0.8225, 0.9986)
    seq = ca.per_fold(tmp_path, "d01_x_", "sequential", 0.8225, 0.9986)
    assert set(conc["cell"]) == {1}
    assert set(seq["cell"]) == {2}


def test_missing_baselines_are_fatal_rather_than_assumed(tmp_path):
    with pytest.raises(SystemExit, match="shipped model's own accuracies"):
        ca.baselines(tmp_path)


def test_missing_shortlists_are_fatal_rather_than_guessed(tmp_path):
    """The cross is defined by the lists it was emitted from."""
    with pytest.raises(SystemExit, match="shortlists"):
        ca._top_lists(tmp_path)


def test_the_better_half_is_chosen_on_the_mean_not_per_fold(tmp_path):
    """
    Choosing the better half fold by fold would pick after seeing the answer.

    Here the aggregation half wins folds 1 and 2 while the penalty wins on the
    mean; a per-fold choice would subtract whichever was larger each time and
    drive every difference down.
    """
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "p12_agg_top3.json").write_text(
        json.dumps({"concurrent": ["A"], "sequential": []}))
    (tmp_path / "tables" / "p13_reg_top3.json").write_text(
        json.dumps({"concurrent": ["R"], "sequential": []}))
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))

    for fold, (agg, reg, combo) in enumerate(
            [(0.90, 0.88, 0.91), (0.90, 0.88, 0.91), (0.80, 0.99, 0.99)], start=1):
        _run(tmp_path, "d01_aggfull_", "A", fold, "concurrent", agg, 0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "R", fold, "concurrent", reg, 0.99)
        _run(tmp_path, "d01_combo_", "A_R", fold, "concurrent", combo, 0.99)

    rows = [r for r in ca.combos(tmp_path, 0.8225, 0.9986) if not r["missing"]]
    assert len(rows) == 1
    # R has the higher mean, so every fold is differenced against R
    assert rows[0]["right"] == "R"
    assert rows[0]["diffs"] == pytest.approx([3.0, 3.0, 0.0])
