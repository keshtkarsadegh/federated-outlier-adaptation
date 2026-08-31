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
