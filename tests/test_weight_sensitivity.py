"""
The weight in the selection rule, and whether the winners depend on it.

``trade_score`` once asserted they do not. These tests pin the arithmetic that
showed otherwise, so the paper's sensitivity row stays a measurement.
"""
import json, sys
from pathlib import Path
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import weight_sensitivity as ws  # noqa: E402


def _row(cell_id, adaptation, preservation):
    return {"id": cell_id,
            "adaptation": {"mean": adaptation},
            "preservation": {"mean": preservation}}


def test_the_score_weights_only_the_forgetting_term():
    row = _row("x", 0.9225, 0.9886)          # +10 adapt, -1 preserve
    assert ws.score(row, 0.8225, 0.9986, 1.0) == pytest.approx(9.0)
    assert ws.score(row, 0.8225, 0.9986, 2.0) == pytest.approx(8.0)
    assert ws.score(row, 0.8225, 0.9986, 5.0) == pytest.approx(5.0)


def test_a_bolder_cell_loses_its_lead_as_the_weight_rises():
    """The whole point: which cell wins can depend on w."""
    bold = _row("bold", 0.9400, 0.9880)      # more adaptation, more forgetting
    timid = _row("timid", 0.9250, 0.9950)
    group = {("concurrent", "m"): [bold, timid]}
    rows = ws.sensitivity(group, 0.8225, 0.9986, (1.0, 5.0))
    assert rows[0]["winners"][1.0] == "bold"
    assert rows[0]["winners"][5.0] == "timid"
    assert rows[0]["changed"] is True


def test_a_winner_that_holds_is_not_marked_changed():
    best = _row("best", 0.9300, 0.9980)      # better on BOTH axes
    other = _row("other", 0.9100, 0.9900)
    rows = ws.sensitivity({("c", "m"): [best, other]}, 0.8225, 0.9986, (1.0, 3.0, 5.0))
    assert set(rows[0]["winners"].values()) == {"best"}
    assert rows[0]["changed"] is False


def test_a_single_cell_row_can_never_change():
    rows = ws.sensitivity({("c", "median"): [_row("median", 0.87, 0.995)]},
                          0.8225, 0.9986, (1.0, 2.0, 3.0))
    assert rows[0]["changed"] is False


def test_the_reference_weight_is_the_first_one_given():
    bold, timid = _row("bold", 0.94, 0.988), _row("timid", 0.925, 0.995)
    rows = ws.sensitivity({("c", "m"): [bold, timid]}, 0.8225, 0.9986, (5.0, 1.0))
    # reference is now w=5, so the row is 'changed' because w=1 differs from it
    assert rows[0]["winners"][5.0] == "timid"
    assert rows[0]["changed"] is True


def test_missing_baselines_are_fatal_rather_than_assumed(tmp_path):
    with pytest.raises(SystemExit, match="shipped model's own accuracies"):
        ws.baselines(tmp_path)


def test_baselines_average_the_per_fold_records(tmp_path):
    (tmp_path / "g0_perfold_evaluations.json").write_text(
        json.dumps([{"accuracy": 0.80}, {"accuracy": 0.84}]))
    (tmp_path / "g0_evaluations.json").write_text(json.dumps([{"accuracy": 0.99}]))
    a0, p0 = ws.baselines(tmp_path)
    assert a0 == pytest.approx(0.82)
    assert p0 == pytest.approx(0.99)
