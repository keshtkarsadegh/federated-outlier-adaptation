"""
The eight signals, and the silence that hid four of them.

Four were empty across an 845-task stage because the proxy set was never built
and the Fisher was never promoted. Every task exited zero. These tests pin the
counting that would have caught it.
"""
import json, sys
from pathlib import Path
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import check_signals as cs  # noqa: E402


def _payload(root, cell, fold, rounds=25, empty=(), unreadable=False):
    d = root / f"d01_x_{cell}_fold{fold}_T_grid_search" / "inner"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"accuracies_{fold}.json"
    if unreadable:
        path.write_text("{not json")
        return
    body = {k: ([None] * rounds if k in empty else list(range(rounds)))
            for k in cs.SIGNAL_KEYS}
    path.write_text(json.dumps(body))


def test_a_complete_run_passes(tmp_path):
    _payload(tmp_path, "a", 1)
    result = cs.audit(tmp_path, "d01_x_", 25)
    assert result["total"] == 1
    assert result["problems"] == []


def test_an_empty_signal_is_a_problem_not_a_note(tmp_path):
    """A None-filled series is exactly what the proxy failure produced."""
    _payload(tmp_path, "a", 1, empty=("proxy_acc", "proxy_kl"))
    problems = cs.audit(tmp_path, "d01_x_", 25)["problems"]
    assert len(problems) == 2
    assert any("proxy_acc" in p for p in problems)
    assert any("proxy_kl" in p for p in problems)


def test_a_short_series_is_caught_too(tmp_path):
    """A run that stopped early is as wrong as one that recorded nothing."""
    _payload(tmp_path, "a", 1, rounds=7)
    problems = cs.audit(tmp_path, "d01_x_", 25)["problems"]
    assert len(problems) == len(cs.SIGNAL_KEYS)
    assert all("shorter than 25 rounds" in p for p in problems)


def test_the_length_check_is_skipped_when_no_horizon_is_given(tmp_path):
    _payload(tmp_path, "a", 1, rounds=7)
    assert cs.audit(tmp_path, "d01_x_", None)["problems"] == []


def test_an_unreadable_payload_is_reported(tmp_path):
    _payload(tmp_path, "a", 1, unreadable=True)
    problems = cs.audit(tmp_path, "d01_x_", 25)["problems"]
    assert any("unreadable" in p for p in problems)


def test_all_eight_keys_are_checked(tmp_path):
    assert len(cs.SIGNAL_KEYS) == 8
    _payload(tmp_path, "a", 1, empty=cs.SIGNAL_KEYS)
    assert len(cs.audit(tmp_path, "d01_x_", 25)["problems"]) == 8


# ------------------------------------------------------ what --all reaches
def test_a_long_run_checked_at_the_screen_horizon_passes_silently(tmp_path):
    """
    The check catches a series that is SHORTER than the horizon, and nothing
    else. A hundred-round stage audited at twenty-five is therefore not audited
    loosely - it passes - which is why every stage needs its own line in the
    horizon map rather than a default.
    """
    _payload(tmp_path, "a", 1, rounds=100)
    assert cs.audit(tmp_path, "d01_x_", 25)["problems"] == []


def test_no_prefix_in_the_map_swallows_another():
    """
    A prefix is globbed as ``prefix*``, so a prefix that is a prefix of another
    audits both stages under one heading and at one horizon. The two would then
    share a total, and the shorter stage's runs would be counted against the
    longer stage's horizon - or the reverse, silently.
    """
    for one in cs.DEFAULT_PREFIXES:
        for other in cs.DEFAULT_PREFIXES:
            if one != other:
                assert not other.startswith(one), (one, other)


@pytest.mark.parametrize(
    "prefix",
    ["d01_five_", "d01_c10d10_", "d01_c20d10_", "d01_c20d20_", "d01_extreme_"],
)
def test_the_carried_settings_and_the_extremes_are_audited(prefix):
    """
    These stages carry the stopping analysis. A signal that is empty on them is
    a missing row in the paper's stopping table, not a missing diagnostic.
    """
    assert cs.DEFAULT_PREFIXES[prefix] == 100


@pytest.mark.parametrize(
    "prefix",
    ["d01_c5_m4_control_", "d01_c10_m8_control_", "d01_c10_m9_control_",
     "d01_c20_m16_control_", "d01_c20_m18_control_"],
)
def test_the_federated_reference_cells_are_audited_too(prefix):
    """The rungs every federated number is read against are runs like any other."""
    assert cs.DEFAULT_PREFIXES[prefix] == 100
