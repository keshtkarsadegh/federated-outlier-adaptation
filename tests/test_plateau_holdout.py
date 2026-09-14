"""
The held-out check on the plateau rule's setting, and the views that carry it.

`test_plateau.py` pins what the rule MEANS and what the shipped views SAY at the
published setting. This file pins the answer to the objection that setting
invites: k = 20, eps = 0 is the best of sixteen cells on the same 83 arms it is
then reported on, so the question is whether a protocol that chooses it
somewhere and evaluates it somewhere else still gets it.

Three kinds of test, in the order they matter. The first is that the basis is
borrowed rather than copied - the tool's whole claim is that a cell priced here
is priced exactly as `plateau_rule.py` prices it, and a reimplemented rule would
let the two disagree without either failing. The second is what the shipped
views say: 28 protocols, k = 20 on every one of them, no held-out gain below
zero, and the two extreme arrangements keeping the rounds the manuscript quotes
under every protocol. The last regenerates the views from an assembled study
root and compares bytes, which is the only test here that needs the runs.
"""
import csv
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
STOPPING = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "stopping"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import plateau_holdout as holdout  # noqa: E402
import plateau_rule as plateau  # noqa: E402
import stopping_table as stop  # noqa: E402

VIEWS = ("plateau_holdout_protocols.csv", "plateau_holdout_extremes.csv",
         "plateau_holdout_grids.csv")


def _rows(name):
    with open(STOPPING / name, newline="") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------- the basis is not restated
def test_the_rule_and_the_arms_are_the_published_tools_own():
    """
    A cell priced held out has to be priced exactly as the published table
    prices it, or the comparison is between two rules rather than two arm sets.
    Everything the split needs - the grid, the outcome, the variant, the arm
    reader, the score and the baselines - is imported.
    """
    assert holdout.outcomes is plateau.outcomes
    assert holdout.PATIENCES is plateau.PATIENCES
    assert holdout.MARGINS is plateau.MARGINS
    assert holdout.CHECKPOINT_BEST is plateau.CHECKPOINT_BEST
    assert (holdout.PRIMARY_PATIENCE, holdout.PRIMARY_MARGIN) == (20, 0.0)
    assert holdout.arm_of is stop.arm_of
    assert holdout.judge is stop.judge
    assert holdout.baselines is stop.baselines
    assert holdout.SERIES is stop.SERIES
    assert holdout.STAGES is stop.STAGES


def test_a_fold_split_re_averages_the_arm_rather_than_cutting_its_trace():
    """
    The published rule runs on the fold MEAN, so an arm has one trace and a fold
    split cannot be made by cutting it. It is made by re-averaging the arm over
    the selected folds alone - the trace a study run on those folds alone would
    have had - which is why the reader keeps the fold label at all.
    """
    per_fold = {fold: {"pool_val_accuracies": [0.1 * fold, 0.2 * fold],
                       "source_val_accuracies": [1.0, 1.0]}
                for fold in (1, 2, 3, 4, 5)}
    everything = holdout.stored_for(per_fold, (1, 2, 3, 4, 5))
    assert len(everything["pool_val_accuracies"]) == 5
    two = holdout.stored_for(per_fold, (4, 5))
    assert two["pool_val_accuracies"] == [[0.4, 0.8], [0.5, 1.0]]
    arm = holdout.arm_of("a", "parallel", two)
    assert arm.folds == 2
    assert arm.adaptation == pytest.approx([0.45, 0.9])


def test_the_tie_break_is_the_smaller_patience_then_the_smaller_margin():
    """
    `plateau_rule.best_cell` reads a grid view that is already a mean over every
    arm; `choose` has to make the same decision on an arbitrary subset, so the
    key is restated here and this is the test that it was restated correctly.
    """
    def cell(value):
        return {("unit", "a", "parallel"): {
            "final_score": 0.0, "kept_score": value, "gain": value,
            "oracle_score": value, "fired": 1}}

    arms = [("unit", "a", "parallel")]
    grid = {(k, eps): cell(0.5 if (k, eps) in ((5, 0.0), (20, 0.001)) else 0.1)
            for k in holdout.PATIENCES for eps in holdout.MARGINS}
    assert holdout.choose(grid, arms) == (5, 0.0)
    grid[(5, 0.001)] = cell(0.5)
    assert holdout.choose(grid, arms) == (5, 0.0)


def test_a_random_half_is_a_function_of_its_seed_alone():
    """
    The halves are the one thing here that is not determined by the runs, so
    they are cut by `random.Random(seed)` over the arms in sorted order and
    never by the interpreter's global state. Same seed, same half, whatever ran
    before it - which is what lets the views regenerate byte for byte.
    """
    import random

    arms = [("s", f"c{index:02d}", "parallel") for index in range(83)]

    def halves(seed):
        shuffled = list(arms)
        random.Random(seed).shuffle(shuffled)
        cut = len(shuffled) // 2
        return shuffled[:cut], shuffled[cut:]

    random.seed(1234)
    first = halves(3)
    random.seed(4321)
    assert halves(3) == first
    assert halves(4) != first
    left, right = first
    assert not set(left) & set(right)
    assert set(left) | set(right) == set(arms)


# ----------------------------------------------------------- the shipped views
def test_every_holdout_view_ships_beside_the_plateau_views():
    """They answer the plateau rule's own objection, so they travel with it."""
    for name in VIEWS:
        assert (STOPPING / name).is_file(), f"{name} is not in the stopping bundle"


def test_the_shipped_views_carry_the_columns_the_tool_declares():
    """A renamed column is a view nobody can read against its own header."""
    for name, columns in zip(VIEWS, (holdout.PROTOCOL_COLUMNS,
                                     holdout.EXTREME_COLUMNS,
                                     holdout.GRID_COLUMNS)):
        with open(STOPPING / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), f"{name}: header is not what its tool writes"


def test_the_protocols_view_is_the_twenty_eight_protocols():
    """
    One in-sample row, two fold halves, five leave-one-fold-out, two stage
    halves, eight leave-one-stage-out and ten seeded random halves. The count is
    asserted before anything is read off the rows, so a protocol added or
    dropped is read as a change of design rather than of arithmetic.
    """
    rows = _rows("plateau_holdout_protocols.csv")
    assert len(rows) == 28
    counts = {name: sum(1 for row in rows if row["protocol"] == name)
              for name in {row["protocol"] for row in rows}}
    assert counts == {"in-sample": 1, "fold": 2, "fold-loo": 5, "stage": 2,
                      "stage-loo": 8, "random-half": 10}


def test_every_protocol_chooses_the_published_patience():
    """
    The headline. Whatever it is allowed to see - three folds, one fold, one
    stage, half the arms - every protocol picks k = 20, and the margin it picks
    beside it is one of the four the grid carries.
    """
    rows = _rows("plateau_holdout_protocols.csv")
    assert {int(row["k"]) for row in rows} == {holdout.PRIMARY_PATIENCE}
    assert {float(row["eps"]) for row in rows} <= set(holdout.MARGINS)


def test_no_protocol_loses_on_the_set_it_did_not_see():
    """
    A setting chosen elsewhere is allowed to be worse than the horizon, and the
    finding is that it never is: the worst held-out gain over the 28 protocols
    is zero, on the two federations where the rule never fires at all.
    """
    rows = _rows("plateau_holdout_protocols.csv")
    assert all(float(row["held_gain"]) >= 0.0 for row in rows)
    assert all(float(row["held_rule"]) >= float(row["held_fixed"]) for row in rows)
    assert min(float(row["held_gain"]) for row in rows) == pytest.approx(0.0, abs=1e-12)


def test_the_in_sample_row_is_the_published_headline():
    """
    The first row prices the published setting where it was chosen, so it has to
    reproduce `plateau_stages.csv`'s all row - +1.19p over the horizon's 7.74p,
    firing on 50 of the 83 arms - or the two tools are reading different runs.
    """
    row = next(r for r in _rows("plateau_holdout_protocols.csv")
               if r["protocol"] == "in-sample")
    assert int(row["sel_arms"]) == int(row["held_arms"]) == 83
    assert int(row["held_fires"]) == 50 and int(row["primary_is_chosen"]) == 1
    assert float(row["held_gain"]) * 100 == pytest.approx(1.19, abs=0.005)
    assert float(row["held_fixed"]) * 100 == pytest.approx(7.74, abs=0.005)


def test_the_extremes_keep_the_published_rounds_under_every_protocol():
    """
    The pair the rule was written for, read off the full five-fold traces under
    each protocol's chosen cell. `dual` keeps round 8 and `double` round 36 in
    all 28 of them, which is the check that the rounds the manuscript names do
    not depend on where the setting came from.
    """
    rows = _rows("plateau_holdout_extremes.csv")
    assert len(rows) == 56
    assert all(int(row["matches_published"]) == 1 for row in rows)
    for cell, (kept, score) in holdout.PUBLISHED_EXTREMES.items():
        mine = [row for row in rows if row["cell"] == cell]
        assert len(mine) == 28
        assert {int(row["kept_round"]) for row in mine} == {kept}
        assert all(float(row["kept_score"]) == pytest.approx(score, abs=1e-12)
                   for row in mine)


def test_the_grids_view_is_every_fold_set_against_every_cell():
    """
    Thirteen fold sets - all five, the two halves, the five leave-one-out and
    the five singletons - against the sixteen cells, with the arm counts of the
    stage split beside them.
    """
    rows = _rows("plateau_holdout_grids.csv")
    cells = len(holdout.PATIENCES) * len(holdout.MARGINS)
    assert len(rows) == 13 * cells
    assert len({row["folds"] for row in rows}) == 13
    assert all(int(row["all_arms"]) == 83 for row in rows)
    assert all(int(row["regfull_arms"]) + int(row["nonregfull_arms"]) == 83
               for row in rows)


# ------------------------------------------------------ regenerating the views
def test_the_shipped_views_regenerate_byte_for_byte():
    """
    The regression this file exists for, and the one that needs the runs.

    Skipped without them: a clone holds the three views but not the 3,855 run
    folders they were computed from. Against an assembled root
    (`docs/REPRODUCE.md` section 10) the tool has to write these exact bytes,
    which is what makes the seeded halves and the fold labels a contract rather
    than a convention.
    """
    root = os.environ.get("FOA_STUDY_DIR")
    if not root or not (Path(root) / "tables").is_dir():
        pytest.skip("no study root; set FOA_STUDY_DIR to run the full check")

    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch)
        argv = sys.argv
        sys.argv = ["plateau_holdout.py", "--root", root, "--out", str(out)]
        try:
            assert holdout.main() == 0
        finally:
            sys.argv = argv
        for name in VIEWS:
            assert (out / name).read_bytes() == (STOPPING / name).read_bytes(), name
