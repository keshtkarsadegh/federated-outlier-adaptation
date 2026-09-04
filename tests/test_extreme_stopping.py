"""
The stopping arithmetic behind the extreme-case figure.

Two pure helpers carry the whole finding: the round the study's own selection
rule would have picked, and how far the retention signal moved while the source
model came apart. Both are tested on synthetic series rather than on the stored
runs - a helper checked against the numbers it produced would only assert that
the runs have not changed, and these say what the helpers mean.
"""
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import extreme_stopping as es  # noqa: E402

A0, P0 = 0.80, 1.00


# --------------------------------------------------------------- fold means
def test_the_fold_mean_is_truncated_to_the_shortest_fold():
    """
    A round only some folds reached is a mean over a different set of runs.

    Averaging over whatever is present would draw a step at the round the first
    fold ran out, and that step is about which folds finished rather than about
    the training.
    """
    assert es.fold_mean([[1.0, 2.0, 3.0], [3.0, 4.0]]) == [2.0, 3.0]
    assert es.fold_mean([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]) == [0.5, 0.5]


@pytest.mark.parametrize("runs", [[], [[]], [[1.0], []]])
def test_a_fold_mean_with_nothing_to_average_refuses(runs):
    with pytest.raises(ValueError):
        es.fold_mean(runs)


# ------------------------------------------------------- the oracle stopping
def test_the_oracle_stop_maximises_the_studys_own_rule():
    """
    Adaptation climbs and holds while preservation decays: the best trade is the
    round adaptation stops paying, not the last round of the run.
    """
    adaptation = [0.80, 0.90, 0.95, 0.95, 0.95]
    preservation = [1.00, 0.99, 0.98, 0.90, 0.80]
    stop, stopped, final = es.oracle_round(adaptation, preservation, A0, P0)
    assert stop == 3
    assert stopped == pytest.approx(0.13)      # +0.15 gained, 0.02 spent
    assert final == pytest.approx(-0.05)       # +0.15 gained, 0.20 spent


def test_a_run_that_never_stops_paying_stops_at_the_end():
    """Nothing here forces an early stop; the rule is allowed to say 'the horizon'."""
    adaptation = [0.82, 0.86, 0.92]
    preservation = [1.00, 1.00, 1.00]
    stop, stopped, final = es.oracle_round(adaptation, preservation, A0, P0)
    assert stop == 3
    assert stopped == final == pytest.approx(0.12)


def test_a_tie_goes_to_the_earliest_round():
    """
    A later round that only equals an earlier one bought nothing and spent more
    of the source model getting there, so calling it the stop would overstate
    how long adaptation kept paying.
    """
    adaptation = [0.90, 0.90, 0.90]
    preservation = [1.00, 1.00, 1.00]
    assert es.oracle_round(adaptation, preservation, A0, P0)[0] == 1


def test_the_stop_does_not_move_when_the_baselines_do():
    """
    A0 and P0 shift every round's score by the same amount, so which round wins
    cannot depend on them - only on the series. This is why the tables and this
    tool can read baselines from the study's files without the stopping round
    becoming a function of how precisely those were written down.
    """
    adaptation = [0.80, 0.93, 0.95, 0.95]
    preservation = [1.00, 0.99, 0.96, 0.85]
    first = es.oracle_round(adaptation, preservation, A0, P0)
    second = es.oracle_round(adaptation, preservation, A0 + 0.13, P0 - 0.07)
    assert first[0] == second[0]
    assert first[1] - second[1] == pytest.approx(0.13 - 0.07)


def test_the_score_series_is_the_selection_rule_term_by_term():
    values = es.scores([0.85, 0.90], [0.98, 0.94], A0, P0)
    assert values == pytest.approx([0.05 - 0.02, 0.10 - 0.06])


def test_choosing_between_no_rounds_refuses():
    with pytest.raises(ValueError):
        es.oracle_round([], [], A0, P0)


# ------------------------------------------------------- retention blindness
def test_the_blindness_metric_measures_from_one_not_from_the_first_round():
    """
    A retention signal is a claim that the original behaviour survived, and 1.0
    is the claim that all of it did. Measured against its own first round
    instead, a signal that started low and stayed there would look steady.
    """
    blind = es.retention_blindness([0.90, 0.90, 0.90], [1.00, 0.95, 0.90])
    assert blind["max_deviation"] == pytest.approx(0.10)
    assert blind["final_deviation"] == pytest.approx(0.10)


def test_a_signal_pinned_at_one_through_a_collapse_is_the_finding():
    """
    The shape the extreme cases actually show: retention never leaves 1.0 while
    preservation falls by a quarter. The metric has to put those two numbers
    beside each other, because either alone reads as unremarkable.
    """
    blind = es.retention_blindness([1.0] * 5, [1.00, 0.95, 0.90, 0.83, 0.75])
    assert blind["max_deviation"] == 0.0
    assert blind["source_drop"] == pytest.approx(0.25)


def test_a_dip_that_recovered_still_counts_as_having_noticed():
    """The maximum over the run, not the final value: a dip is a detection."""
    blind = es.retention_blindness([1.0, 0.94, 1.0], [1.00, 0.99, 0.98])
    assert blind["max_deviation"] == pytest.approx(0.06)
    assert blind["final_deviation"] == 0.0


@pytest.mark.parametrize("retention,preservation", [([], [1.0]), ([1.0], [])])
def test_the_blindness_metric_needs_both_series(retention, preservation):
    with pytest.raises(ValueError):
        es.retention_blindness(retention, preservation)


# ------------------------------------------------------------- what it reads
def test_the_baselines_come_from_the_study_rather_than_from_this_file(tmp_path):
    """
    Hardcoded baselines are how the ad-hoc preview of these numbers came to sit
    three thousandths of a point away from the committed ones. This tool imports
    the tables' own reader, so there is one definition of A0 and P0 - and a root
    without the shipped model's evaluations is refused rather than defaulted.
    """
    import report_tables

    assert es.baselines is report_tables.baselines
    with pytest.raises(SystemExit, match="shipped-model evaluations"):
        es.baselines(tmp_path)


def test_a_root_with_no_extreme_runs_reads_as_empty(tmp_path):
    """Nothing on disk is not a table of zeros."""
    assert es.read_cases(tmp_path, "d01_extreme_") == {}
    assert es.traces({}, "dual") is None


def test_the_reported_cases_are_the_ones_the_stage_defines():
    """
    Order is this tool's; membership is the cell list's.

    A case reported here that no stage defines would put an arm nothing ran
    into the table, and a case defined there and missing here would drop one
    silently.
    """
    from federated_outlier_adaptation.training.extreme_cells import CASES

    assert set(es.CASES) == set(CASES) == {"double", "dual"}
    assert set(es.WHAT) == set(es.CASES)
