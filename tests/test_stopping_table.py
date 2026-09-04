"""
The arithmetic behind the stopped-against-fixed table.

Three questions decide every row: which round an oracle would have stopped at,
which round a permitted signal's budget fires on, and which single rule a
deployment would have to fix in advance. The first two are borrowed - from
`extreme_stopping` and from the analysis module - and the tests that matter most
here are the ones pinning that they are still borrowed. A stopping table whose
drift semantics have come apart from `foa signals` reports a rule nobody could
run, and it would report it without failing.

The helpers are tested on synthetic traces rather than on the stored runs. A
helper checked against the numbers it produced only asserts that the runs have
not changed; these say what the helpers mean.
"""
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import extreme_stopping as es  # noqa: E402
import report_tables  # noqa: E402
import stopping_table as stop  # noqa: E402

from federated_outlier_adaptation.analysis import forgetting_signals as analysis  # noqa: E402

A0, P0 = 0.80, 1.00
ROUNDS = 6


def make_arm(cell="a", family="parallel", adaptation=None, preservation=None, **signals):
    """An arm whose signals are flat unless a test moves one of them."""
    adaptation = adaptation or [0.80 + 0.02 * r for r in range(ROUNDS)]
    preservation = preservation or [1.00 - 0.005 * r for r in range(ROUNDS)]
    series = {key: [0.0] * len(adaptation) for key in stop.SIGNAL_KEYS}
    series.update(signals)
    return stop.Arm(cell=cell, family=family, folds=5, adaptation=adaptation,
                    preservation=preservation, signals=series)


# ------------------------------------------------- the borrowed definitions
def test_the_stopping_semantics_are_the_analysis_modules_own():
    """
    `foa signals` and this table have to fire on the same round for the same
    budget, or the table prices a rule the study never studied.
    """
    assert stop.stop_round is analysis.stop_round
    assert stop.RunTrajectory is analysis.RunTrajectory
    assert stop.DEFAULT_DELTAS is analysis.DEFAULT_DELTAS


def test_the_oracle_and_the_fold_mean_are_the_extreme_tools_own():
    """
    Run this with --stage extreme and the two rows must reproduce
    extreme_stopping.py round for round. They do because it is the same code.
    """
    assert stop.oracle_round is es.oracle_round
    assert stop.fold_mean is es.fold_mean
    assert stop.scores is es.scores


def test_the_baselines_come_from_the_study_rather_than_from_this_file(tmp_path):
    assert stop.baselines is report_tables.baselines
    with pytest.raises(SystemExit, match="shipped-model evaluations"):
        stop.baselines(tmp_path)


# ---------------------------------------------------------------- the rules
def test_a_rule_fires_at_the_first_round_above_its_budget():
    """
    Agreement falls four hundredths a round, so its drift first passes a budget
    of a tenth at index 3 - which is round 4, one-based like every round label
    the extreme table prints.
    """
    arm = make_arm(agreement_with_global=[1.0 - 0.04 * r for r in range(ROUNDS)])
    verdict = stop.judge(arm, A0, P0, deltas=(0.1,))
    assert verdict.stops[("agreement_with_global", 0.1)] == 3
    assert verdict.at(3)["round"] == 4


def test_a_signal_that_never_moves_far_enough_is_the_fixed_horizon():
    """
    A rule is allowed to say 'never stop', and it must then score exactly what
    the horizon scores. This is why an arm whose best round IS the last one has
    a permitted rule that ties the horizon rather than one that beats it.
    """
    arm = make_arm(proxy_kl=[0.001 * r for r in range(ROUNDS)])
    verdict = stop.judge(arm, A0, P0, deltas=(0.5,))
    index = verdict.stops[("proxy_kl", 0.5)]
    assert index == ROUNDS - 1
    assert verdict.at(index) == verdict.final


def test_round_zero_can_never_be_a_stop():
    """
    The first stored entry is the shipped model before any client has trained,
    and every signal's drift is measured from it, so it is the reference rather
    than a candidate. A rule that could 'stop' there would be recommending that
    nothing be adapted at all.
    """
    arm = make_arm(proxy_kl=[0.0, 9.0, 9.0, 9.0, 9.0, 9.0])
    verdict = stop.judge(arm, A0, P0, deltas=(0.001,))
    assert verdict.stops[("proxy_kl", 0.001)] == 1


def test_the_best_rule_takes_the_higher_score():
    values = [0.00, 0.09, 0.13, 0.05]
    verdict = stop.Verdict(arm=make_arm(), values=values, star=3,
                           stops={("early", 0.1): 1, ("right", 0.1): 2})
    assert stop.best_rule(verdict) == ("right", 0.1)


def test_a_tie_between_two_rules_goes_to_the_one_that_stops_earlier():
    """
    Two rules that reach the same score are not equally good: the later one
    spent more rounds of the source model arriving at it.
    """
    values = [0.00, 0.05, 0.13, 0.13]
    verdict = stop.Verdict(arm=make_arm(), values=values, star=3,
                           stops={("late", 0.1): 3, ("early", 0.1): 2})
    assert stop.best_rule(verdict) == ("early", 0.1)


# ------------------------------------------------------------- the one rule
def _verdicts(*stop_maps):
    values = [0.00, 0.10, 0.02, 0.20]
    other = [0.00, 0.00, 0.20, 0.02]
    return [
        stop.Verdict(arm=make_arm(cell="a"), values=values, star=4, stops=stop_maps[0]),
        stop.Verdict(arm=make_arm(cell="b"), values=other, star=3, stops=stop_maps[1]),
    ]


def test_one_rule_is_chosen_on_the_mean_and_not_on_any_single_arm():
    """
    The first arm would pick A - it scores 0.10 there against B's 0.02 - and the
    second would pick B. Neither preference is deployable, because choosing per
    arm needs the source split. Over the two arms B means 0.11 against A's 0.05,
    so the one rule a deployment could fix in advance is B, and the first arm
    has to live with it.
    """
    verdicts = _verdicts({("A", 1.0): 1, ("B", 1.0): 2}, {("A", 1.0): 1, ("B", 1.0): 2})
    assert stop.best_rule(verdicts[0]) == ("A", 1.0)
    assert stop.best_rule(verdicts[1]) == ("B", 1.0)
    assert stop.one_rule(verdicts) == ("B", 1.0)


def test_a_rule_defined_on_only_some_arms_cannot_be_the_one_rule():
    """
    A rule that is silent on a stage has not been priced on it, and the mean of
    the arms it does reach is a different quantity from every other rule's mean.
    """
    verdicts = _verdicts({("A", 1.0): 1, ("only", 1.0): 3}, {("A", 1.0): 1})
    assert stop.one_rule(verdicts) == ("A", 1.0)


def test_the_one_rule_does_not_depend_on_the_baselines():
    """
    A0 and P0 shift every round of every arm by the same amount, so they cannot
    change which rule wins - only how the winning rule's score reads. The table
    can therefore read them from the study's files without the chosen rule
    becoming a function of how precisely those were written down.
    """
    arms = [
        make_arm(cell="a", proxy_kl=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        make_arm(cell="b", adaptation=[0.80, 0.95, 0.95, 0.90, 0.85, 0.80],
                 proxy_kl=[0.0, 0.05, 0.3, 0.5, 0.7, 0.9]),
    ]
    first = stop.one_rule([stop.judge(a, A0, P0, (0.05, 0.2)) for a in arms])
    second = stop.one_rule([stop.judge(a, A0 + 0.13, P0 - 0.07, (0.05, 0.2)) for a in arms])
    assert first == second


def test_nothing_to_fix_in_advance_when_no_rule_reaches_every_arm():
    assert stop.one_rule([]) is None
    assert stop.one_rule(_verdicts({("A", 1.0): 1}, {("B", 1.0): 1})) is None


# ------------------------------------------------------------- what it reads
def _write(root, folder, scenario, body):
    path = root / folder / "fold1_seed_1" / scenario
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary_0.json").write_text(json.dumps({"job": body}))


def _body(rounds=ROUNDS):
    return {
        "pool_val_accuracies": [0.80 + 0.02 * r for r in range(rounds)],
        "source_val_accuracies": [1.00 - 0.005 * r for r in range(rounds)],
        "proxy_kl": [0.05 * r for r in range(rounds)],
    }


def test_a_penalty_is_read_only_under_the_schedule_it_was_selected_for(tmp_path):
    """
    A regularisation run computes BOTH schedules and the folder is named for the
    one it was selected to serve. Reading the other half would price the penalty
    under a schedule it was never picked for, and the two halves would then be
    averaged into one row as though they were folds of one thing.
    """
    _write(tmp_path, "d01_x_sequential_kd_fold1_T_grid_search", "concurrent_delta", _body())
    _write(tmp_path, "d01_x_sequential_kd_fold1_T_grid_search", "sequential_weights", _body())
    assert set(stop.read_arms(tmp_path, "d01_x_")) == {("kd", "cyclic")}


def test_an_untagged_arm_is_read_under_both_schedules(tmp_path):
    """The schedule is part of the key: an arm that ran both ran two things."""
    _write(tmp_path, "d01_x_control_fold1_T_grid_search", "concurrent_delta", _body())
    _write(tmp_path, "d01_x_control_fold1_T_grid_search", "sequential_weights", _body())
    assert set(stop.read_arms(tmp_path, "d01_x_")) == {
        ("control", "parallel"), ("control", "cyclic")}


def test_an_arm_is_the_fold_mean_truncated_to_the_shortest_fold(tmp_path):
    """
    A round only some folds reached is a mean over a different set of runs, and
    an oracle round chosen at such a round is about which folds finished.
    """
    _write(tmp_path, "d01_x_a_fold1_T_grid_search", "concurrent_delta", _body(rounds=6))
    _write(tmp_path, "d01_x_a_fold2_T_grid_search", "concurrent_delta", _body(rounds=4))
    stored = stop.read_arms(tmp_path, "d01_x_")[("a", "parallel")]
    arm = stop.arm_of("a", "parallel", stored)
    assert arm.folds == 2
    assert arm.rounds == 4


def test_a_root_with_no_runs_reads_as_no_arms(tmp_path):
    assert stop.read_arms(tmp_path, "d01_x_") == {}
    assert stop.arm_of("a", "parallel", {}) is None


def test_the_row_carries_all_three_answers(tmp_path):
    """
    final, oracle and stopped are the three numbers the paper quotes per arm,
    and the two gaps are differences of them rather than separately computed.
    """
    arm = make_arm(proxy_kl=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    verdict = stop.judge(arm, A0, P0, deltas=(0.05,))
    row = stop.row_of(verdict, "unit", ("proxy_kl", 0.05))
    assert row["final_round"] == ROUNDS
    assert row["oracle_round"] == verdict.star
    assert row["oracle_cost"] == pytest.approx(row["oracle_score"] - row["final_score"])
    assert row["one_vs_final"] == pytest.approx(row["one_score"] - row["final_score"])
    assert row["one_signal"] == "proxy_kl"
