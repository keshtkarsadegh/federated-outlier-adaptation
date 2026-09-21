"""
The plateau rule: what it means, where its views ship, and what they say.

Three things can go wrong here and only one of them is arithmetic. The rule
itself is four lines and is tested on synthetic traces, because a helper checked
against the numbers it produced only asserts that the runs have not changed.
Its BASIS is borrowed from `stopping_table.py` - the arm reader, the fold mean,
the score and the oracle round - and a copy of any of those would let the two
tables report different final scores for one arm without either failing. And the
four shipped views are read by the manuscript through `make_numbers.locate`,
which looks in four directories, so a view written into the wrong one resolves
nowhere and is found at build time rather than here.

The last test is a different kind. It pins the two headline numbers of
`plateau_stages.csv` against the values the paper quotes, so that a regenerated
view whose arithmetic quietly moved is a failure rather than a diff nobody read.
"""
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
STOPPING = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "stopping"

for entry in (str(TOOLS), str(TOOLS / "paper_figures")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import plateau_rule as plateau  # noqa: E402
import stopping_table as stop  # noqa: E402

#: A trace that climbs for four rounds and then never improves again. With a
#: patience of three the third non-improvement is round 7, and the best round in
#: everything seen up to it is round 4.
CLIMB_THEN_FLAT = [0.10, 0.20, 0.30, 0.40, 0.35, 0.36, 0.34, 0.33, 0.32, 0.31]


# ------------------------------------------------- the basis is not restated
def test_the_arms_and_the_score_are_the_stopping_tables_own():
    """
    One arm has one final score, and the two tables have to print the same one.

    Everything the rule is measured against - which runs make an arm, what the
    fold mean is, what the score is and where the oracle sits - is imported
    rather than reimplemented, so `plateau_arms.csv` and `stopping_all.csv`
    cannot come apart on the arms they share.
    """
    assert plateau.read_arms is stop.read_arms
    assert plateau.arm_of is stop.arm_of
    assert plateau.judge is stop.judge
    assert plateau.baselines is stop.baselines
    assert plateau.one_rule is stop.one_rule
    assert plateau.STAGES is stop.STAGES


# ------------------------------------------------------------------ the rule
def test_the_rule_fires_on_the_kth_round_without_an_improvement():
    """
    Patience is counted in consecutive rounds, so a run that peaks at round 4
    and never beats it fires at round 4 + k and not before.
    """
    assert plateau.fire_round(CLIMB_THEN_FLAT, patience=3, margin=0.0) == 6
    assert plateau.fire_round(CLIMB_THEN_FLAT, patience=5, margin=0.0) == 8


def test_the_rule_keeps_the_best_round_it_had_seen_when_it_fired():
    """
    The server is not deploying the round it stopped on. It has been keeping the
    best cohort round all along, and that is round 4 - index 3 - here.
    """
    fired = plateau.fire_round(CLIMB_THEN_FLAT, patience=3, margin=0.0)
    assert plateau.kept_round(CLIMB_THEN_FLAT, fired) == 3
    assert plateau.scored_round(CLIMB_THEN_FLAT, fired, plateau.CHECKPOINT_BEST) == 3
    assert plateau.scored_round(CLIMB_THEN_FLAT, fired, plateau.STOP_AT_FIRE) == fired


def test_a_run_that_keeps_improving_never_fires_and_is_the_fixed_horizon():
    """
    A rule is allowed to say 'never stop', and it must then score exactly what
    the horizon scores - the last round, which is what the tables report.
    """
    climbing = [0.10 * r for r in range(12)]
    assert plateau.fire_round(climbing, patience=3, margin=0.0) is None
    assert plateau.scored_round(climbing, None, plateau.CHECKPOINT_BEST) == 11
    assert plateau.scored_round(climbing, None, plateau.STOP_AT_FIRE) == 11


def test_only_an_improvement_resets_the_patience():
    """
    A run that oscillates below its own best has plateaued however lively the
    trace looks: two of the three rounds after the peak rise on the round before
    them and neither beats the peak, so the count runs on rather than restarting
    and the rule fires at round 7.
    """
    wobble = [0.10, 0.20, 0.30, 0.40, 0.20, 0.30, 0.20, 0.30, 0.20]
    assert plateau.fire_round(wobble, patience=3, margin=0.0) == 6


def test_the_margin_is_how_much_an_improvement_has_to_be_worth():
    """
    Two hundredths of a point a round is an improvement at eps=0 and is not one
    at eps=0.05, so the same trace never fires under the first and fires under
    the second.
    """
    creeping = [0.50 + 0.02 * r for r in range(9)]
    assert plateau.fire_round(creeping, patience=2, margin=0.0) is None
    assert plateau.fire_round(creeping, patience=2, margin=0.05) == 2


def test_round_zero_is_the_reference_rather_than_a_round_that_can_fire():
    """
    The first stored entry is the shipped model before any client has trained.
    It is the first best-so-far, so the count starts at round 1 and the earliest
    round the rule can fire on is round k - index k, one-based round k+1.
    """
    falling = [0.90] + [0.10] * 8
    assert plateau.fire_round(falling, patience=3, margin=0.0) == 3
    assert plateau.kept_round(falling, 3) == 0


def test_a_tie_for_the_best_round_goes_to_the_earlier_one():
    """
    Two rounds at the same cohort accuracy are not equally good: the later one
    spent more rounds of the source model arriving at it. The same tie-break
    `stopping_table.best_rule` applies between two rules.
    """
    tied = [0.10, 0.40, 0.30, 0.40, 0.20, 0.20, 0.20, 0.20]
    assert plateau.kept_round(tied, plateau.fire_round(tied, 3, 0.0)) == 1


def test_the_rule_reads_the_cohorts_own_accuracy_and_not_the_source_half():
    """
    The whole claim is that this rule is deployable, and it is deployable only
    because it never touches `source_val_accuracies`. An arm whose two halves
    move in opposite directions is where a swapped series would show.
    """
    arm = stop.Arm(cell="a", family="parallel", folds=5,
                   adaptation=CLIMB_THEN_FLAT,
                   preservation=list(reversed(CLIMB_THEN_FLAT)))
    verdict = stop.Verdict(arm=arm, values=[0.0] * len(CLIMB_THEN_FLAT), star=1, stops={})
    assert plateau.cohort_of(verdict) == CLIMB_THEN_FLAT


def test_an_arm_is_priced_against_its_own_last_round():
    """
    `gain` is the difference of two columns of the same row rather than a
    separately computed quantity, and `kept_round` is one-based like every round
    label the stopping tables print.
    """
    arm = stop.Arm(cell="a", family="parallel", folds=5,
                   adaptation=CLIMB_THEN_FLAT, preservation=[1.0] * len(CLIMB_THEN_FLAT))
    values = [0.01 * v for v in range(len(CLIMB_THEN_FLAT))]
    verdict = stop.Verdict(arm=arm, values=values, star=len(values), stops={})
    row = plateau.outcome("unit", verdict, 3, 0.0, plateau.CHECKPOINT_BEST)
    assert row["fire_round"] == 7 and row["kept_round"] == 4
    assert row["kept_score"] == pytest.approx(values[3])
    assert row["final_score"] == pytest.approx(values[-1])
    assert row["gain"] == pytest.approx(row["kept_score"] - row["final_score"])


def test_an_arm_that_never_fires_reports_no_fire_round_and_no_gain():
    """A blank fire round is the tool's sentinel; the row is still written."""
    arm = stop.Arm(cell="a", family="parallel", folds=5,
                   adaptation=[0.10 * r for r in range(8)], preservation=[1.0] * 8)
    verdict = stop.Verdict(arm=arm, values=[0.01 * r for r in range(8)], star=8, stops={})
    row = plateau.outcome("unit", verdict, 3, 0.0, plateau.CHECKPOINT_BEST)
    assert row["fired"] == 0 and row["fire_round"] is None
    assert row["kept_round"] == 8 and row["gain"] == pytest.approx(0.0)


# ---------------------------------------------------------------- the tables
def _staged():
    """Two arms of one stage: one that plateaus early, one that never does."""
    def arm(cell, adaptation):
        return stop.Arm(cell=cell, family="parallel", folds=5,
                        adaptation=adaptation, preservation=[1.0] * len(adaptation))
    return [
        ("unit", stop.Verdict(arm=arm("plateaus", CLIMB_THEN_FLAT),
                              values=[0.01 * r for r in range(10)], star=10, stops={})),
        ("unit", stop.Verdict(arm=arm("climbs", [0.10 * r for r in range(10)]),
                              values=[0.02 * r for r in range(10)], star=10, stops={})),
    ]


def test_the_all_row_is_the_mean_over_arms_and_not_over_stages():
    """
    A stage with twenty-six arms has to weigh twenty-six times a stage with one,
    which is the weighting `stopping_table.one_rule` chooses its rule under and
    the reason the two tables' all rows can be read against each other.
    """
    rows = plateau.outcomes(_staged(), 3, 0.0, plateau.CHECKPOINT_BEST)
    stages = plateau.stage_rows(rows, 3, 0.0, plateau.CHECKPOINT_BEST)
    every = [row for row in stages if row["stage"] == plateau.ALL]
    assert len(every) == 1 and every[0]["arms"] == 2
    assert every[0]["fires"] == 1
    assert every[0]["gain"] == pytest.approx(
        sum(row["gain"] for row in rows) / len(rows))


def test_the_grid_is_every_setting_under_both_variants():
    grid = plateau.grid_rows(_staged())
    assert len(grid) == len(plateau.PATIENCES) * len(plateau.MARGINS) * len(plateau.VARIANTS)
    assert {row["variant"] for row in grid} == set(plateau.VARIANTS)
    assert (plateau.PRIMARY_PATIENCE, plateau.PRIMARY_MARGIN) == (20, 0.0)


# ----------------------------------------------------------- the shipped views
VIEWS = ("plateau_arms.csv", "plateau_stages.csv", "plateau_grid.csv",
         "plateau_extremes.csv")


def _rows(name):
    with open(STOPPING / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_plateau_view_ships_beside_the_stopping_tables():
    """
    They answer the same question off the same arms, so they travel in the same
    bundle - and that bundle is one of the four `make_numbers.locate` looks in.
    """
    import make_numbers

    for name in VIEWS:
        assert (STOPPING / name).is_file(), f"{name} is not in the stopping bundle"
        assert make_numbers.locate(name) == str(STOPPING / name)


def test_the_shipped_views_carry_the_columns_the_tool_declares():
    """A renamed column reaches the manuscript as a missing key, one build later."""
    for name, columns in (("plateau_arms.csv", plateau.ARM_COLUMNS),
                          ("plateau_stages.csv", plateau.STAGE_COLUMNS),
                          ("plateau_grid.csv", plateau.GRID_COLUMNS),
                          ("plateau_extremes.csv", plateau.EXTREME_COLUMNS)):
        with open(STOPPING / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), f"{name}: header is not what its tool writes"


def test_the_shipped_arms_are_the_arms_the_stopping_table_shipped():
    """
    Same arms, same final score, arm for arm. The two views are cut from one
    read of the runs, and if they ever stopped agreeing here one of them would
    be reporting a stage the other no longer sees.
    """
    mine = {(r["stage"], r["cell"], r["family"]): r for r in _rows("plateau_arms.csv")}
    theirs = {(r["stage"], r["cell"], r["family"]): r for r in _rows("stopping_all.csv")}
    assert set(mine) == set(theirs)
    for key, row in mine.items():
        assert float(row["final_score"]) == pytest.approx(float(theirs[key]["final_score"]))


def test_the_shipped_stage_rows_add_up_to_the_shipped_all_row():
    """The all row is a mean over the same 83 arms the stage rows are cut from."""
    stages = _rows("plateau_stages.csv")
    every = [row for row in stages if row["stage"] == plateau.ALL]
    assert len(every) == 1
    parts = [row for row in stages if row["stage"] != plateau.ALL]
    assert sum(int(row["arms"]) for row in parts) == int(every[0]["arms"])
    assert sum(int(row["fires"]) for row in parts) == int(every[0]["fires"])


def test_the_shipped_extremes_are_the_extreme_rows_of_the_shipped_arms():
    extremes = {row["cell"]: row for row in _rows("plateau_extremes.csv")}
    arms = {row["cell"]: row for row in _rows("plateau_arms.csv")
            if row["stage"] == plateau.EXTREME_STAGE}
    assert set(extremes) == set(arms) == {"dual", "double"}
    for cell, row in extremes.items():
        for column in ("fire_round", "kept_round", "kept_score", "final_score", "gain"):
            assert row[column] == arms[cell][column], f"{cell}: {column}"


def test_the_setting_is_not_the_one_the_extremes_would_have_chosen():
    """
    The honesty split, read off the shipped grid rather than asserted in prose.

    k=20, eps=0 is the best of the sixteen cells on the mean over all 83 arms,
    and the two extreme arrangements are the arms with by far the most to gain -
    so the objection is that the setting was fitted to them. Held out entirely,
    the remaining 81 arms choose the same cell.
    """
    grid = [{**row,
             "patience": int(row["patience"]),
             "margin": float(row["margin"]),
             "mean_score": float(row["mean_score"]),
             "nonextreme_mean_score": float(row["nonextreme_mean_score"])}
            for row in _rows("plateau_grid.csv")]
    primary = (plateau.PRIMARY_PATIENCE, plateau.PRIMARY_MARGIN)
    for column in ("mean_score", "nonextreme_mean_score"):
        chosen = plateau.best_cell(grid, plateau.CHECKPOINT_BEST, column)
        assert (chosen["patience"], chosen["margin"]) == primary, column


def test_the_shipped_all_row_still_says_what_the_paper_says():
    """
    The one pin on the values themselves, and it is here on purpose.

    Everything above says what the rule MEANS; nothing above would notice if a
    regenerated view moved the headline by a point. These two numbers - 9.04p
    against the horizon's 7.85p, +1.19p, firing on 50 of the 83 arms - are what
    the manuscript prints, so a regeneration that changes them has to be a
    failing test rather than a diff nobody read.

    THE TWO MEANS MOVED AND THE GAIN DID NOT. Both were 0.11p lower while the
    arms of the five-, twenty- and two-client settings were scored against the
    ten-writer cohort's A0; scoring each against its own moved the level of
    this pool and not the distance between stopping and not stopping, because
    a per-arm constant cancels in a difference of two means over the same
    arms. The gain is the claim, and the gain is unchanged.

    THE POPULATION IS PART OF THE PIN. It was 81 arms at +1.22p until the
    blend's own finals were run: those are two more reg-full arms, they belong
    in every reg-full view, and neither of them fires. So the headline moved
    because the study grew, which is a reason to re-pin the number and never to
    exclude the arms - and the arm count is asserted first so that a future
    change to it is read as a change of population rather than of arithmetic.
    """
    row = next(r for r in _rows("plateau_stages.csv") if r["stage"] == plateau.ALL)
    assert int(row["arms"]) == 83 and int(row["fires"]) == 50
    assert float(row["rule_mean"]) * 100 == pytest.approx(9.04, abs=0.005)
    assert float(row["gain"]) * 100 == pytest.approx(1.19, abs=0.005)
    assert float(row["fixed_mean"]) * 100 == pytest.approx(7.85, abs=0.005)
    assert float(row["oracle_mean"]) * 100 == pytest.approx(9.33, abs=0.005)
    assert float(row["rule_mean"]) - float(row["fixed_mean"]) == pytest.approx(float(row["gain"]))


def test_the_extremes_still_say_what_the_paper_says():
    """
    The pair the rule was written for, and the rounds the prose names.

    The scores are the extreme cohort's own: these two writers held alone, and
    not the ten the search ran on. The rounds are the same four they always
    were, which is what says the baseline fix moved a level and not a decision.
    """
    rows = {row["cell"]: row for row in _rows("plateau_extremes.csv")}
    assert (int(rows["dual"]["fire_round"]), int(rows["dual"]["kept_round"])) == (28, 8)
    assert float(rows["dual"]["kept_score"]) * 100 == pytest.approx(22.95, abs=0.005)
    assert (int(rows["double"]["fire_round"]), int(rows["double"]["kept_round"])) == (56, 36)
    assert float(rows["double"]["kept_score"]) * 100 == pytest.approx(22.03, abs=0.005)
