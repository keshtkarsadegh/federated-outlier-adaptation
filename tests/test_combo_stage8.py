"""
Stage 8: the top server rules crossed with the top client penalties.

The table itself is cheap to check.  The thing that actually needed proving is
that the two halves *run together*: every stage-6 rule was screened with a plain
trainer and every stage-7 penalty was screened under plain FedAvg, so no run in
the study had ever put an extended aggregation rule and an anchored trainer in
the same loop at the same time.  Those pairings are turned here, through both
runners, before ninety array elements find out on the cluster.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest
import torch

from federated_outlier_adaptation.aggregation.selector import (
    SKIP_FAMILY,
    resolve_aggregation,
    select_class,
)
from federated_outlier_adaptation.training.combo_cells import (
    AGG_CELLS,
    FULL_ROUNDS,
    HYBRID_FISHER_CELL,
    HYBRID_KD_CELL,
    HYBRID_MIX,
    REG_CELLS,
    combo_cells,
    combos_by_schedule,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

MAX_ROUND = 2
EPOCHS = 1
BATCH_SIZE = 4


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def test_the_cross_is_three_by_three_on_each_schedule():
    grouped = combos_by_schedule()
    assert len(combo_cells()) == 18
    assert len(grouped["concurrent"]) == 9
    assert len(grouped["sequential"]) == 9
    assert FULL_ROUNDS == 100


def test_every_combination_id_is_unique():
    ids = [combo["id"] for combo in combo_cells()]
    assert len(ids) == len(set(ids))


def test_each_schedule_crosses_its_own_three_rules_with_its_own_three_penalties():
    for schedule, group in combos_by_schedule().items():
        assert {c["agg"]["id"] for c in group} == set(AGG_CELLS[schedule])
        assert len({c["reg"]["id"] for c in group}) == 3
        assert all(c["agg"]["path"] == schedule for c in group)


def test_the_hybrid_mix_differs_between_the_schedules():
    """mix weights KD against Fisher, and the two schedules chose differently."""
    for schedule, group in combos_by_schedule().items():
        hybrids = [c["reg"] for c in group if c["reg"]["space"] == "kd+fisher"]
        assert len(hybrids) == 3
        assert {h["hypers"]["mix"] for h in hybrids} == {HYBRID_MIX[schedule]}
    assert HYBRID_MIX["concurrent"] != HYBRID_MIX["sequential"]


def test_the_coefficients_are_the_stage_six_and_seven_doubles_exactly():
    """
    Inherited, not retyped.

    Retyping a float is how 2.3333333333333335 becomes 2.33333 - a different
    objective wearing the same label - which stage 7 already caught once.
    """
    from federated_outlier_adaptation.training import agg_cells, reg_cells

    aggs = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    regs = {cell["id"]: cell for cell in reg_cells.screen_cells()}

    for combo in combo_cells():
        assert combo["agg"]["flags"] == aggs[combo["agg"]["id"]]["flags"]
        if combo["reg"]["space"] == "kd+fisher":
            kd = regs[HYBRID_KD_CELL]
            assert combo["reg"]["hypers"]["lam"] == kd["hypers"]["lam"]
            assert combo["reg"]["hypers"]["T"] == kd["hypers"]["T"]
        else:
            assert combo["reg"]["hypers"] == regs[combo["reg"]["id"]]["hypers"]


def test_the_hybrid_is_what_stage_seven_s_emitter_builds(tools_path):
    """Same two winners, same constructor, same object at a different mix."""
    from emit_stage7_hybrid import hybrid_cells

    built, _, _ = hybrid_cells(HYBRID_KD_CELL, HYBRID_FISHER_CELL)
    by_mix = {cell["hypers"]["mix"]: cell for cell in built}
    for schedule, group in combos_by_schedule().items():
        mine = next(c["reg"] for c in group if c["reg"]["space"] == "kd+fisher")
        theirs = by_mix[HYBRID_MIX[schedule]]
        assert mine["hypers"] == theirs["hypers"]
        assert mine["id"] == theirs["id"]
        assert mine["needs_fisher"] is theirs["needs_fisher"] is True


def test_only_the_hybrid_asks_for_a_fisher():
    fisher = [c for c in combo_cells() if c["reg"]["needs_fisher"]]
    assert len(fisher) == 6
    assert {c["reg"]["space"] for c in fisher} == {"kd+fisher"}


# --------------------------------------------------------------------------- #
# One combination is one family
# --------------------------------------------------------------------------- #
def test_every_combination_resolves_to_exactly_one_family():
    """
    A stage-8 line is not ``--aggregation fedavg``.

    Each rule lives in one (scenario, metadata) family, so a task produces one
    result and the two halves of the cross stay separate experiments.
    """
    for combo in combo_cells():
        families = [
            (scenario, metadata)
            for scenario in ("concurrent", "sequential")
            for metadata in ("weights", "delta")
            if resolve_aggregation(scenario, metadata, combo["agg"]["rule"], extended=True)
            != SKIP_FAMILY
        ]
        assert len(families) == 1, (combo["id"], families)
        assert families[0][0] == combo["schedule"], combo["id"]


# --------------------------------------------------------------------------- #
# The task file
# --------------------------------------------------------------------------- #
def test_every_line_parses_and_carries_the_protocol(tools_path):
    from make_stage8_combos import task_line

    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for combo in combo_cells():
        args = parser.parse_args(shlex.split(task_line(combo, 3, 100, 1))[1:])
        assert args.func.__name__ == "cmd_final"
        assert args.rounds == 100 and args.epochs == 5 and args.batch_size == 64
        assert args.fold == 3 and args.seed == 3
        assert (args.policy, args.clients_per_round) == ("uniform", 16)
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients
        assert args.old_fold == "all" and args.old_book and args.old_clients_file
        assert args.aggregation == combo["agg"]["rule"]
        assert args.aggregation != "fedavg"


def test_the_penalty_reaches_the_trainer_keywords(tools_path):
    from make_stage8_combos import task_line

    from federated_outlier_adaptation.cli import _trainer_overrides, build_parser

    parser = build_parser()
    for combo in combo_cells():
        args = parser.parse_args(shlex.split(task_line(combo, 1, 100, 1))[1:])
        overrides = _trainer_overrides(args)
        assert overrides["space"] == combo["reg"]["space"]
        assert overrides["anchor"] == "frozen"
        for name, value in combo["reg"]["hypers"].items():
            assert float(overrides[name]) == float(value), (combo["id"], name)
        assert ("fisher_path" in overrides) is bool(combo["reg"]["needs_fisher"])


def test_the_server_coefficients_reach_the_server_state(tools_path):
    from make_stage8_combos import task_line

    from federated_outlier_adaptation.aggregation.concurrent_methods import ServerState
    from federated_outlier_adaptation.cli import _server_kwargs, build_parser

    parser = build_parser()
    for combo in combos_by_schedule()["concurrent"]:
        args = parser.parse_args(shlex.split(task_line(combo, 1, 100, 1))[1:])
        state = ServerState(eta=args.server_eta, **_server_kwargs(args))
        flags = combo["agg"]["flags"]
        if "server_anchor" in flags:
            assert state.anchor_lambda == pytest.approx(flags["server_anchor"])
        if "server_eta" in flags:
            assert state.eta == pytest.approx(flags["server_eta"])
        if "server_lr" in flags:
            assert state.server_lr == pytest.approx(flags["server_lr"])
        if "server_tau" in flags:
            assert state.tau == pytest.approx(flags["server_tau"])


def test_the_sequential_mixing_weight_reaches_the_command_line(tools_path):
    from make_stage8_combos import task_line

    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    mixing = [
        c for c in combos_by_schedule()["sequential"]
        if c["agg"]["rule"] == "seq_mix_alpha"
    ]
    assert mixing
    for combo in mixing:
        args = parser.parse_args(shlex.split(task_line(combo, 1, 100, 1))[1:])
        assert args.seq_mix_alpha == pytest.approx(combo["agg"]["flags"]["seq_mix_alpha"])


# --------------------------------------------------------------------------- #
# The pairing itself: an anchored trainer inside an extended aggregation rule
# --------------------------------------------------------------------------- #
def _run(provider, scenario, rule, trainer, **runner_kwargs):
    """Two rounds of one combination, through the runner its schedule uses."""
    from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
    from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner

    metadata = next(
        md for md in ("weights", "delta")
        if resolve_aggregation(scenario, md, rule, extended=True) != SKIP_FAMILY
    )
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider, **runner_kwargs)
    accuracies, _, _, _ = runner.simulate(
        exp_name=f"combo_{rule}",
        global_name="global",
        aggregate_method=getattr(select_class(scenario, metadata), rule),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    return runner, accuracies


CONCURRENT_PAIRINGS = [
    ("con_delta_anchor_lam", {"server_kwargs": {"anchor_lambda": 0.3}}),
    ("con_delta_eta", {"server_eta": 0.1}),
    ("con_delta_fedadam", {"server_kwargs": {"server_lr": 0.000316, "tau": 0.001}}),
]

SEQUENTIAL_PAIRINGS = [
    ("seq_delta_capped", {}),
    ("seq_mix_alpha", {"seq_mix_alpha": 0.05}),
    ("seq_delta_scaled", {}),
]

SPACES = ["kd+fisher", "kd", "logit_l2"]


@pytest.mark.parametrize("rule,runner_kwargs", CONCURRENT_PAIRINGS, ids=lambda v: str(v)[:40])
@pytest.mark.parametrize("space", SPACES)
def test_a_concurrent_rule_and_an_anchored_trainer_turn_together(
    synthetic_provider, rule, runner_kwargs, space
):
    """
    Nothing in the study had ever run these two at once.

    Every stage-6 rule was screened with a plain trainer and every stage-7
    penalty under plain FedAvg, so the first time an extended server rule met an
    anchored client objective would have been ninety array elements deep.
    """
    from federated_outlier_adaptation.trainers.anchored_trainer import AnchoredTrainer

    trainer = AnchoredTrainer(
        provider=synthetic_provider, space=space, anchor="frozen",
        lam=0.42857142857142866, T=16.0, mix=0.75,
    )
    runner, accuracies = _run(
        synthetic_provider, "concurrent", rule, trainer, **runner_kwargs
    )
    assert len(accuracies) == MAX_ROUND
    assert all(clients is not None for clients, _ in accuracies)
    for value in runner.global_model.state_dict().values():
        if value.dtype.is_floating_point:
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("rule,runner_kwargs", SEQUENTIAL_PAIRINGS, ids=lambda v: str(v)[:40])
@pytest.mark.parametrize("space", SPACES)
def test_a_sequential_rule_and_an_anchored_trainer_turn_together(
    synthetic_provider, rule, runner_kwargs, space
):
    from federated_outlier_adaptation.trainers.anchored_trainer import AnchoredTrainer

    trainer = AnchoredTrainer(
        provider=synthetic_provider, space=space, anchor="frozen",
        lam=0.42857142857142866, T=16.0, mix=0.5,
    )
    runner, accuracies = _run(
        synthetic_provider, "sequential", rule, trainer, **runner_kwargs
    )
    assert len(accuracies) == MAX_ROUND
    for value in runner.global_model.state_dict().values():
        if value.dtype.is_floating_point:
            assert torch.isfinite(value).all()


def test_the_server_anchor_still_pulls_when_a_penalty_is_also_running(
    synthetic_provider,
):
    """
    Two restraints, both live.

    The server anchor is a server-side pull towards theta_g and the trainer's
    penalty is a client-side one; the whole question of stage 8 is whether they
    compose, so a run must not silently drop one of them.
    """
    from federated_outlier_adaptation.trainers.anchored_trainer import AnchoredTrainer

    def run(anchor_lambda):
        trainer = AnchoredTrainer(
            provider=synthetic_provider, space="kd", anchor="frozen", lam=0.5, T=16.0
        )
        runner, _ = _run(
            synthetic_provider, "concurrent", "con_delta_anchor_lam", trainer,
            seed=3, server_kwargs={"anchor_lambda": anchor_lambda},
        )
        return runner.global_model.state_dict()

    strong = run(0.3)
    none = run(0.0)
    moved = [
        key for key, value in strong.items()
        if value.dtype.is_floating_point and not torch.allclose(value, none[key])
    ]
    assert moved, "the server anchor made no difference to the final model"
