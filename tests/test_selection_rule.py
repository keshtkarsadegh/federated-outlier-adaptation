"""
The selection rule: what a run gained, less what it spent.

Selecting on adaptation alone picks, for every method, the setting that
constrains least - the weakest anchor, the largest server step. Every winner is
then the cell closest to plain FedAvg, and the table reports that no method
preserves anything, which is a property of the rule and not of the methods. On
the real screen it moved three of eighteen winners; the one it moved furthest
gained 2.20 points of preservation for 0.09 of adaptation.

These tests pin the rule itself, and pin that it is the SAME rule everywhere a
winner is chosen - the aggregation finals, the aggregation patch, the
regularisation finals and the regularisation patch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import study_emit  # noqa: E402


def _row(cell_id, adaptation, preservation):
    return {
        "id": cell_id,
        "adaptation": {"mean": adaptation, "sd": 0.01},
        "preservation": {"mean": preservation, "sd": 0.01},
    }


A0, P0 = 0.8225, 0.9986


def test_the_rule_prefers_the_better_trade_over_the_greediest():
    """
    The case that motivated it, with the screen's own numbers.

    fedadam's greediest cell gains 10.27 points and spends 2.23; the cell one
    step in gains 9.78 and spends 1.70. Giving up 0.49 of adaptation saves 0.53
    of preservation, so the second is the better buy and the rule must say so.
    """
    greedy = _row("fedadam_tau1e-05", 0.9252, 0.9763)
    better = _row("fedadam_tau1e-04", 0.9203, 0.9816)

    assert greedy["adaptation"]["mean"] > better["adaptation"]["mean"]
    assert (study_emit.trade_score(better, A0, P0)
            > study_emit.trade_score(greedy, A0, P0))


def test_the_rule_does_not_reward_a_configuration_that_barely_moves():
    """
    A pure ratio would crown timidity.

    anchor_0.3 gains 3.04 points for 0.09 - a ratio of 33, far better than any
    real contender - but it reaches 0.8529 when 0.9147 is available. Gain minus
    spend keeps the scale in the answer, so it does not win.
    """
    timid = _row("anchor_h0p125", 0.8529, 0.9977)
    real = _row("seq_delta_scaled", 0.9147, 0.9906)

    timid_ratio = (timid["adaptation"]["mean"] - A0) / (P0 - timid["preservation"]["mean"])
    real_ratio = (real["adaptation"]["mean"] - A0) / (P0 - real["preservation"]["mean"])
    assert timid_ratio > real_ratio, "the ratio really does favour the timid cell"

    assert (study_emit.trade_score(real, A0, P0)
            > study_emit.trade_score(timid, A0, P0))


def test_a_run_that_spends_more_than_it_gains_scores_below_doing_nothing():
    """Doing nothing scores zero; anything worse than that must be negative."""
    wasteful = _row("greedy", A0 + 0.01, P0 - 0.02)
    assert study_emit.trade_score(wasteful, A0, P0) < 0
    assert study_emit.trade_score(_row("still", A0, P0), A0, P0) == 0


@pytest.mark.parametrize("weight", [1.0, 2.0, 3.0])
def test_the_winner_does_not_depend_on_the_weighting(weight):
    """
    Weighting a point of forgetting against a point of adaptation is a choice.
    It is not a delicate one: the screen's winner holds at one, two and three.
    """
    field = [
        _row("seq_delta_scaled", 0.9147, 0.9906),
        _row("fedadam_tau1e-05", 0.9252, 0.9763),
        _row("seq_mix_r0p3", 0.9157, 0.9860),
        _row("fedavgm_b0p5", 0.9174, 0.9828),
        _row("control_fedavg", 0.9043, 0.9910),
    ]
    best = max(field, key=lambda r: (r["adaptation"]["mean"] - A0)
                                    - weight * (P0 - r["preservation"]["mean"]))
    assert best["id"] == "seq_delta_scaled"


def test_the_baselines_are_read_and_not_assumed(tmp_path):
    for name, accuracy in (("g0_perfold_evaluations.json", 0.8225),
                           ("g0_evaluations.json", 0.9986)):
        (tmp_path / name).write_text(json.dumps(
            {str(k): {"fold": k, "accuracy": accuracy} for k in (1, 2, 3, 4, 5)}))
    a0, p0 = study_emit.shipped_baselines(tmp_path)
    assert a0 == pytest.approx(0.8225)
    assert p0 == pytest.approx(0.9986)


def test_a_root_without_the_baselines_is_refused(tmp_path):
    """
    Never a default. A selection measured against a guessed baseline produces a
    task file that looks exactly like a real one.
    """
    with pytest.raises(SystemExit):
        study_emit.shipped_baselines(tmp_path)


def test_every_selection_in_the_programme_uses_the_same_rule():
    """
    Five places pick a winner: the aggregation finals and their patch, the
    regularisation finals and theirs, and the blend's own finals. One rule for
    all five, or the tables were not chosen the same way and cannot be read
    against each other - and the blend is the arm that makes that matter, since
    it is reported in the same table as the seven penalties it was screened
    apart from.
    """
    source = Path(study_emit.__file__).read_text()
    picks = [line for line in source.splitlines() if "max(scored" in line]
    assert len(picks) == 5, f"expected five selection sites, found {len(picks)}"
    for line in picks:
        assert "trade_score" in line, f"selects without the rule: {line.strip()}"


def test_weighting_preservation_higher_would_crown_the_most_constraining_cell():
    """
    Why the weight stays at one for the penalties too.

    A penalty is bought for preservation, so weighting preservation more looks
    natural. It fails the way selecting on adaptation failed, from the other
    side: the winner becomes the cell that spends least because it moves least.
    """
    strong = _row("penalty_very_strong", A0 + 0.005, P0 - 0.001)   # barely moves
    useful = _row("penalty_balanced", A0 + 0.090, P0 - 0.010)      # real adaptation

    at_one = lambda r: (r["adaptation"]["mean"] - A0) - 1.0 * (P0 - r["preservation"]["mean"])
    at_ten = lambda r: (r["adaptation"]["mean"] - A0) - 10.0 * (P0 - r["preservation"]["mean"])

    assert at_one(useful) > at_one(strong)
    assert at_ten(strong) > at_ten(useful), "a heavy weight really does crown timidity"
