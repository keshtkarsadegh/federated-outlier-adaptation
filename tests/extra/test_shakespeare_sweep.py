"""
Stage A: the strength sweep of the three transferred winners.

The sweep exists because a coefficient is a *rate*, and the two tasks are not
the same size. What must not drift is everything else: if the sweep changed the
horizon, the participation or the fold as well as the strength, its rows would
not be comparable with the transferred references they are read against, and the
recovery it is looking for could not be attributed to the coefficient.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import SHAKESPEARE_STUDY01 as CFG

TOOLS = str(Path(__file__).resolve().parents[2] / "tools")


@pytest.fixture()
def sweep():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_shakespeare_sweep

    return make_shakespeare_sweep


@pytest.fixture()
def lines(sweep):
    return sweep.lines()


def test_ten_runs_three_families(sweep, lines):
    assert len(lines) == 10
    labels = [label for label, _, _ in sweep.sweep_cells()]
    assert labels.count("winner") == 3
    assert labels.count("balanced") == 4
    assert labels.count("sequential") == 3


def test_the_transferred_reference_is_not_re_run(sweep):
    """
    Each family's reference row already ran as ``s01_*``. Re-running it here
    would spend a GPU-hour to produce a number that exists, and invite two
    slightly different values for one configuration.
    """
    swept = {(agg["id"], reg["id"]) for _, agg, reg in sweep.sweep_cells()}
    assert ("trimmed_0p4", "feature_l2_lam0p1") not in swept
    assert ("anchor_0p03", "ntd_b0p01_t0p5") not in swept
    assert ("seq_order_shuffle", "feature_l2_lam0p01") not in swept


def test_only_the_strength_moves(lines):
    """Same rules, horizon, participation, fold and evaluation as the s01 runs."""
    for line in lines:
        assert " --rounds 100 " in line
        assert " --epochs 5 " in line and " --batch-size 64 " in line
        assert f" --clients-per-round {CFG.clients_per_round} " in line
        assert " --fold 1 " in line and " --seed 1" in line
        assert " --old-fold all" in line
        assert " --init global --global-name g0" in line
        assert " --provider shakespeare --model char_lstm" in line
        assert "anchor=frozen" in line, "the penalty still anchors on g-0"


def test_the_balanced_pair_moves_one_hold_at_a_time(sweep):
    """
    Its two holds are the server pull and the client penalty. Moving both at
    once would leave the result unattributable to either.
    """
    rows = [(a, r) for label, a, r in sweep.sweep_cells() if label == "balanced"]
    server = [(a, r) for a, r in rows if r["hypers"]["lam"] == 0.01]
    client = [(a, r) for a, r in rows if a["flags"]["server_anchor"] == 0.03]
    assert sorted(a["flags"]["server_anchor"] for a, _ in server) == [0.1, 0.3]
    assert sorted(r["hypers"]["lam"] for _, r in client) == [0.03, 0.1]
    for _, reg in rows:
        assert reg["hypers"]["T"] == 0.5, "the temperature is not a sweep axis"


def test_the_coefficients_are_emitted_at_full_precision(lines):
    """
    ``{:g}`` is six significant figures, which is fine for a label and wrong for
    a coefficient. 3.16 and 0.316 must appear as themselves.
    """
    joined = "\n".join(lines)
    for value in ("lam=0.316", "lam=1.0", "lam=3.16", "lam=0.03", "lam=0.1"):
        assert value in joined, value
    assert "--server-anchor 0.1 " in joined and "--server-anchor 0.3 " in joined


def test_parents_and_seeds_are_fresh_and_distinct(lines):
    parents = [ln.split(" --parent ")[1].split()[0] for ln in lines]
    seeds = [int(ln.split(" --sampler-seed ")[1].split()[0]) for ln in lines]
    assert len(set(parents)) == len(set(seeds)) == 10
    assert all(p.startswith("s01a_") for p in parents), parents[:2]
    # Its own block: clear of the s01 runs (800001..800012) and of every digit
    # range, so no folder or sampler draw can be claimed twice.
    assert all(801000 <= s < 802000 for s in seeds)


def test_every_line_parses(lines):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for line in lines:
        parser.parse_args(shlex.split(line.replace("$FOA_STUDY_DIR", "/study"))[1:])


def test_no_fisher_and_no_inline_python(lines):
    for line in lines:
        assert line.startswith("foa ")
        assert "fisher" not in line and "G0_FOLD" not in line
