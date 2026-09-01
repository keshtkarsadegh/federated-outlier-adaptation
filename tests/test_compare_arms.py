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
         adaptation: float, preservation: float, name: str = "") -> None:
    d = root / f"{prefix}{cell}_fold{fold}_AnchoredTrainer_grid_search" / "inner"
    d.mkdir(parents=True, exist_ok=True)
    (d / (name or f"accuracies_{fold}.json")).write_text(json.dumps({
        "scenario": family,
        "pool_val_accuracies": [0.5, adaptation],
        "final_evaluation": {"old": {"mean": preservation}},
    }))


def _both_schedules(root: Path, prefix: str, cell: str, fold: int,
                    concurrent: float, sequential: float,
                    preservation: float) -> None:
    """One folder holding BOTH schedules' payloads, as the blends were run."""
    _run(root, prefix, cell, fold, "concurrent", concurrent, preservation,
         name="accuracies_concurrent.json")
    _run(root, prefix, cell, fold, "sequential", sequential, preservation,
         name="accuracies_sequential.json")


def _study(root: Path, reg_ids) -> None:
    """The two shortlists and the shipped model's own accuracies."""
    (root / "tables").mkdir(exist_ok=True)
    (root / "tables" / "p12_agg_top3.json").write_text(
        json.dumps({"concurrent": ["A"], "sequential": []}))
    (root / "tables" / "p13_reg_top3.json").write_text(
        json.dumps({"concurrent": list(reg_ids), "sequential": []}))
    (root / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (root / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))


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


# --------------------------------------------------------------------------- #
# where a standalone penalty was written
# --------------------------------------------------------------------------- #
def test_a_penalty_stored_without_its_family_is_still_found(tmp_path):
    """
    The blends were written bare, and the pairing could not see them.

    A penalty run for one schedule is stored under that schedule's prefix. The
    blends ran both schedules in one task, so one folder per (cell, fold) holds
    both payloads and is named without a family at all. Looking only under the
    family prefix reported NO PAIRED RESULT for every blend while its finals sat
    on disk - the numbers existed, under a name the tool never asked for.
    """
    for fold in (1, 2, 3):
        _run(tmp_path, "d01_regfull_concurrent_", "R", fold, "concurrent", 0.90, 0.99)
        _both_schedules(tmp_path, "d01_regfull_", "B", fold,
                        concurrent=0.92, sequential=0.80, preservation=0.99)

    conc = ca.penalties(tmp_path, "concurrent", 0.8225, 0.9986)
    assert set(conc) >= {"R", "B"}
    assert sorted(conc["B"]) == [1, 2, 3]
    # the concurrent payload's number, not the sequential one it sits beside
    assert conc["B"][1] == pytest.approx(ca.score(0.92, 0.99, 0.8225, 0.9986))

    seq = ca.penalties(tmp_path, "sequential", 0.8225, 0.9986)
    assert seq["B"][1] == pytest.approx(ca.score(0.80, 0.99, 0.8225, 0.9986))
    # the schedule-specific folder is a concurrent run and says so
    assert "R" not in seq


def test_the_family_prefix_stays_authoritative(tmp_path):
    """
    The bare read fills gaps; it never overrides a cell stored under its family.

    The bare prefix matches everything the family one does, so without that rule
    a folder named for a schedule and a folder named for none would race, and
    which won would depend on the order the directories came back in.
    """
    for fold in (1, 2):
        _run(tmp_path, "d01_regfull_concurrent_", "R", fold, "concurrent", 0.90, 0.99)
        _run(tmp_path, "d01_regfull_", "R", fold, "concurrent", 0.10, 0.99)
    found = ca.penalties(tmp_path, "concurrent", 0.8225, 0.9986)
    assert found["R"][1] == pytest.approx(ca.score(0.90, 0.99, 0.8225, 0.9986))


# --------------------------------------------------------------------------- #
# the blend, against both of the cells it was built from
# --------------------------------------------------------------------------- #
def _hybrid_record(root: Path, family: str, kd: str, fisher: str,
                   mixes=(0.5,)) -> None:
    """One family's construction record, under the literal name it is read by."""
    name = ("p14_hybrid_construction_sequential.json" if family == "sequential"
            else "p14_hybrid_construction.json")
    (root / "tables").mkdir(exist_ok=True)
    (root / "tables" / name).write_text(json.dumps({
        "kd_winner": kd, "fisher_winner": fisher, "family": family,
        "mixes": list(mixes), "lam": 0.1, "T": 2.0,
    }))


def test_a_blend_is_differenced_against_each_parent_not_the_better_one(tmp_path):
    """
    Both directions in one study: cleared on every fold, and lost on every fold.

    A blend is offered as a line between two named methods, so the interesting
    claim is about both ends at once. Differencing against the better parent
    only - the way a combination is read - would report this blend as a clean
    win and never mention that its Fisher parent beats it by three points on
    every fold.
    """
    _hybrid_record(tmp_path, "concurrent", "K", "F")
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))

    for fold, (blend, kd, fisher) in enumerate(
            [(0.90, 0.88, 0.93), (0.91, 0.89, 0.94), (0.92, 0.90, 0.95)], start=1):
        # the blend ran both schedules in one task and is named without a family
        _both_schedules(tmp_path, "d01_regfull_", "hybrid_mix0p5", fold,
                        concurrent=blend, sequential=0.5, preservation=0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "K", fold, "concurrent", kd, 0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "F", fold, "concurrent", fisher, 0.99)

    rows = ca.blends(tmp_path, 0.8225, 0.9986)
    assert [(r["left"], r["right"]) for r in rows] == [
        ("hybrid_mix0p5", "K"), ("hybrid_mix0p5", "F")]
    assert not any(r["missing"] for r in rows)

    against_kd, against_fisher = rows
    assert against_kd["diffs"] == pytest.approx([2.0, 2.0, 2.0])
    assert against_kd["all_positive"] is True
    assert against_fisher["diffs"] == pytest.approx([-3.0, -3.0, -3.0])
    assert against_fisher["all_positive"] is False
    assert against_fisher["consistent"] is True


def test_each_schedule_is_priced_against_its_own_two_winners(tmp_path):
    """
    The blend is a per-family object, and the two schedules blended different
    cells. A view that read one record for both would difference the sequential
    blend against the concurrent selection's halves - parents it never had.
    """
    _hybrid_record(tmp_path, "concurrent", "CK", "CF")
    _hybrid_record(tmp_path, "sequential", "SK", "SF")
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))

    for fold in (1, 2, 3):
        _both_schedules(tmp_path, "d01_regfull_", "hybrid_mix0p5", fold,
                        concurrent=0.90, sequential=0.70, preservation=0.99)
        _both_schedules(tmp_path, "d01_regfull_", "hybrid_seq_mix0p5", fold,
                        concurrent=0.70, sequential=0.90, preservation=0.99)
        for cell, value in (("CK", 0.88), ("CF", 0.89)):
            _run(tmp_path, "d01_regfull_concurrent_", cell, fold, "concurrent",
                 value, 0.99)
        for cell, value in (("SK", 0.86), ("SF", 0.87)):
            _run(tmp_path, "d01_regfull_sequential_", cell, fold, "sequential",
                 value, 0.99)

    rows = ca.blends(tmp_path, 0.8225, 0.9986)
    # the sequential blend carries the seq_ tag its emitter gave it, and each
    # family's rows name that family's two winners and no others
    assert [(r["family"], r["left"], r["right"]) for r in rows] == [
        ("concurrent", "hybrid_mix0p5", "CK"),
        ("concurrent", "hybrid_mix0p5", "CF"),
        ("sequential", "hybrid_seq_mix0p5", "SK"),
        ("sequential", "hybrid_seq_mix0p5", "SF"),
    ]
    assert not any(r["missing"] for r in rows)
    # each blend is scored on its own schedule's payload, not the one beside it
    assert rows[0]["diffs"] == pytest.approx([2.0, 2.0, 2.0])
    assert rows[2]["diffs"] == pytest.approx([4.0, 4.0, 4.0])


def test_a_blend_with_no_finals_says_so_rather_than_being_dropped(tmp_path):
    """NO PAIRED RESULT is a statement; a silently missing row is not."""
    _hybrid_record(tmp_path, "concurrent", "K", "F", mixes=(0.25, 0.5))
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))
    for fold in (1, 2):
        _both_schedules(tmp_path, "d01_regfull_", "hybrid_mix0p5", fold,
                        concurrent=0.90, sequential=0.5, preservation=0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "K", fold, "concurrent", 0.88, 0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "F", fold, "concurrent", 0.89, 0.99)

    rows = ca.blends(tmp_path, 0.8225, 0.9986)
    assert [(r["left"], r["right"], r["missing"]) for r in rows] == [
        ("hybrid_mix0p25", "K", True),
        ("hybrid_mix0p25", "F", True),
        ("hybrid_mix0p5", "K", False),
        ("hybrid_mix0p5", "F", False),
    ]


def test_the_parents_are_never_inferred_from_the_folder_names(tmp_path):
    """
    Without a construction record there is no saying which two cells a blend was
    built from, and guessing would compare it against parents it never had.
    """
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.8225}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.9986}]))
    with pytest.raises(SystemExit, match="construction record"):
        ca.blends(tmp_path, 0.8225, 0.9986)


def test_both_layouts_pair_in_the_cross_and_in_the_composition(tmp_path):
    """Neither view may report a missing half for a run that exists."""
    _study(tmp_path, ["R", "B"])
    for fold in (1, 2, 3):
        _run(tmp_path, "d01_aggfull_", "A", fold, "concurrent", 0.88, 0.99)
        _run(tmp_path, "d01_regfull_concurrent_", "R", fold, "concurrent", 0.90, 0.99)
        _both_schedules(tmp_path, "d01_regfull_", "B", fold,
                        concurrent=0.92, sequential=0.80, preservation=0.99)
        _run(tmp_path, "d01_combo_", "A_R", fold, "concurrent", 0.93, 0.99)
        _run(tmp_path, "d01_combo_", "A_B", fold, "concurrent", 0.94, 0.99)

    rows = ca.combos(tmp_path, 0.8225, 0.9986)
    assert [r["left"] for r in rows] == ["A_R", "A_B"]
    assert not any(r.get("missing") for r in rows)

    rows = ca.composition(tmp_path, 0.8225, 0.9986)
    assert [(r["left"], r["right"]) for r in rows] == [("R", "A"), ("B", "A")]
    assert not any(r.get("missing") for r in rows)
