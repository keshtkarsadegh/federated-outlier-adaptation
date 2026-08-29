"""
Stage 7: the regularisation-contribution screen.

Three things are tested.  The **grids** - that the cell table is the documented
shape and that its alpha convention is the one the trainer actually implements.
The **trainers** - that the extreme ends of those grids are values the penalty
code accepts rather than values that blow up at round one, and that mu = 0 is an
exact no-op and not merely a small number.  And the **selector** - that it ranks
per method per family, which is the whole reason a stage-7 task runs two
families at once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from federated_outlier_adaptation.training.reg_cells import (
    ANCHOR,
    FAMILIES,
    FEATURE_L2_LAMS,
    FISHER_LAMS,
    HYBRID_MIXES,
    KD_ALPHAS,
    KD_TEMPERATURES,
    LOGIT_L2_LAMS,
    LOGIT_L2_T,
    NTD_BETAS,
    NTD_TAUS,
    PARAM_L2_MUS,
    SCREEN_ROUNDS,
    cells_by_method,
    control_cell,
    kd_lam_of_alpha,
    methods,
    screen_cells,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


# --------------------------------------------------------------------------- #
# The grids
# --------------------------------------------------------------------------- #


def test_every_cell_id_is_unique():
    ids = [cell["id"] for cell in screen_cells()]
    assert len(ids) == len(set(ids))


def test_the_grids_are_the_owner_confirmed_values():
    assert PARAM_L2_MUS == (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
    assert FISHER_LAMS == (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)
    assert LOGIT_L2_LAMS == (0.1, 0.316, 1.0, 3.16, 10.0)
    assert FEATURE_L2_LAMS == (1e-2, 0.1, 1.0, 10.0, 1e2)
    assert KD_TEMPERATURES == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0)
    assert KD_ALPHAS == (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)
    assert NTD_BETAS == (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
    assert NTD_TAUS == (0.5, 1.0, 2.0, 3.0, 4.0)
    assert HYBRID_MIXES == (0.25, 0.5, 0.75)


def test_the_kd_grid_is_alphas_converted_to_the_trainer_s_lam():
    from federated_outlier_adaptation.training.reg_cells import kd_cells

    # the screened grid, before the boundary extension was appended
    cells = kd_cells()
    assert len(cells) == 49
    assert len({(c["hypers"]["T"], c["hypers"]["lam"]) for c in cells}) == 49
    # alpha = 0.5 is an equal split, i.e. lam = 1
    assert kd_lam_of_alpha(0.5) == pytest.approx(1.0)
    assert kd_lam_of_alpha(0.9) == pytest.approx(1.0 / 9.0)
    assert kd_lam_of_alpha(1.0) == pytest.approx(0.0)


def test_an_impossible_alpha_is_refused():
    for alpha in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha must lie"):
            kd_lam_of_alpha(alpha)


def test_the_logit_row_pins_the_temperature_it_does_not_use():
    cells = [c for c in screen_cells() if c["method"] == "logit_l2"]
    assert {c["hypers"]["T"] for c in cells} == {LOGIT_L2_T}


def test_the_anchor_is_frozen_everywhere():
    """The premise: the penalty measures distance from the model g-0 is."""
    tools = Path(TOOLS)
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    from make_stage7_screen import set_tokens

    for cell in screen_cells():
        assert f"anchor={ANCHOR}" in set_tokens(cell)


def test_only_the_fisher_spaces_ask_for_a_fisher():
    fisher_cells = [c for c in screen_cells() if c["needs_fisher"]]
    assert len(fisher_cells) == 16
    assert {c["method"] for c in fisher_cells} == {"fisher", "fisher_scaled"}


def test_the_hybrid_is_not_in_the_screen():
    assert "kd+fisher" not in {cell["method"] for cell in screen_cells()}


# --------------------------------------------------------------------------- #
# The trainer accepts the ends of those grids
# --------------------------------------------------------------------------- #
BATCH = 4


@pytest.fixture()
def batch():
    torch.manual_seed(17)
    return torch.rand(BATCH, 1, 128, 128), torch.tensor([0, 1, 2, 1])


def _loss_parts(provider, batch, **kwargs):
    """The (ce, penalty) split of one forward pass under a configuration."""
    from federated_outlier_adaptation.trainers.anchored_trainer import AnchoredTrainer

    images, labels = batch
    trainer = AnchoredTrainer(provider=provider, anchor=ANCHOR, **kwargs)
    torch.manual_seed(7)
    trainer.set_model(provider.make_model())
    trainer.model.eval()
    features = None
    if kwargs.get("space") == "feature_l2":
        from federated_outlier_adaptation.trainers.feature_alignment_trainer import (
            logits_and_features,
        )

        logits, features = logits_and_features(trainer.model, images)
    else:
        logits = trainer.model(images)
    _, parts = trainer.custom_loss_fn(logits, features, labels, images)
    return parts


def test_mu_zero_is_an_exact_no_op(synthetic_provider, batch):
    """
    The control cell has to be a control, not a very small penalty.

    ``penalty()`` returns zero for any ``lam <= 0``, so ``mu = 0`` is the
    unregularised objective bit for bit - which is what lets stage 7 read
    ``param_l2_mu0`` as its own baseline instead of borrowing stage 6's.
    """
    parts = _loss_parts(
        synthetic_provider, batch, space="param_l2", lam=0.0,
        param_l2_convention="fedprox",
    )
    assert parts["penalty"] == 0.0

    plain = _loss_parts(
        synthetic_provider, batch, space="param_l2", lam=1.0,
        param_l2_convention="fedprox",
    )
    # the cross-entropy is untouched either way; only the penalty differs
    assert parts["ce"] == pytest.approx(plain["ce"])
    assert plain["penalty"] > 0.0


@pytest.mark.parametrize("mu", PARAM_L2_MUS)
def test_every_fedprox_mu_is_finite(synthetic_provider, batch, mu):
    parts = _loss_parts(
        synthetic_provider, batch, space="param_l2", lam=mu,
        param_l2_convention="fedprox",
    )
    assert torch.isfinite(torch.tensor(parts["penalty"]))
    assert parts["penalty"] >= 0.0


@pytest.mark.parametrize("lam", FISHER_LAMS)
@pytest.mark.parametrize("space", ["fisher", "fisher_scaled"])
def test_every_ewc_lambda_is_finite(synthetic_provider, batch, space, lam):
    """
    Including lambda = 1e6.

    Eight decades is a wide row, and the top of it is where a penalty that is
    merely large becomes a penalty that overflows. The screen must not discover
    that on the cluster.
    """
    parts = _loss_parts(synthetic_provider, batch, space=space, lam=lam)
    assert torch.isfinite(torch.tensor(parts["penalty"]))


def test_the_scaled_fisher_caps_where_the_plain_one_does_not(synthetic_provider, batch):
    """``fisher_scaled`` is not ``fisher`` rescaled; the cap changes the value."""
    plain = _loss_parts(synthetic_provider, batch, space="fisher", lam=1e6)
    capped = _loss_parts(synthetic_provider, batch, space="fisher_scaled", lam=1e6)
    assert plain["penalty"] != pytest.approx(capped["penalty"])


@pytest.mark.parametrize("temperature", KD_TEMPERATURES)
def test_every_distillation_temperature_is_finite(synthetic_provider, batch, temperature):
    """Including T = 50, where the softened distribution is nearly uniform."""
    parts = _loss_parts(
        synthetic_provider, batch, space="kd",
        lam=kd_lam_of_alpha(0.5), T=temperature,
    )
    assert torch.isfinite(torch.tensor(parts["penalty"]))
    assert parts["penalty"] >= 0.0


@pytest.mark.parametrize("tau", NTD_TAUS)
def test_every_ntd_tau_is_finite(synthetic_provider, batch, tau):
    parts = _loss_parts(synthetic_provider, batch, space="ntd", lam=1.0, T=tau)
    assert torch.isfinite(torch.tensor(parts["penalty"]))


@pytest.mark.parametrize("lam", LOGIT_L2_LAMS)
def test_every_logit_lambda_is_finite(synthetic_provider, batch, lam):
    parts = _loss_parts(
        synthetic_provider, batch, space="logit_l2", lam=lam, T=LOGIT_L2_T
    )
    assert torch.isfinite(torch.tensor(parts["penalty"]))


@pytest.mark.parametrize("lam", FEATURE_L2_LAMS)
def test_every_feature_lambda_is_finite(synthetic_provider, batch, lam):
    parts = _loss_parts(synthetic_provider, batch, space="feature_l2", lam=lam)
    assert torch.isfinite(torch.tensor(parts["penalty"]))


def test_the_logit_space_really_does_ignore_the_temperature(synthetic_provider, batch):
    """The README says so; if it stopped being true the fixed row would be wrong."""
    at_two = _loss_parts(synthetic_provider, batch, space="logit_l2", lam=1.0, T=2.0)
    at_twenty = _loss_parts(synthetic_provider, batch, space="logit_l2", lam=1.0, T=20.0)
    assert at_two["penalty"] == pytest.approx(at_twenty["penalty"])


@pytest.mark.parametrize("mix", HYBRID_MIXES)
def test_the_hybrid_blend_is_between_its_two_halves(synthetic_provider, batch, mix):
    """
    ``lam * (mix * KD + (1 - mix) * Fisher)`` really is a blend.

    If it were not, the three points the emitter produces would not read as a
    line between the two methods, which is the only reason three points is
    enough.
    """
    kd = _loss_parts(synthetic_provider, batch, space="kd", lam=1.0, T=4.0)
    fisher = _loss_parts(synthetic_provider, batch, space="fisher", lam=1.0)
    blend = _loss_parts(
        synthetic_provider, batch, space="kd+fisher", lam=1.0, T=4.0, mix=mix
    )
    expected = mix * kd["penalty"] + (1.0 - mix) * fisher["penalty"]
    assert blend["penalty"] == pytest.approx(expected, rel=1e-5)


# --------------------------------------------------------------------------- #
# The task file
# --------------------------------------------------------------------------- #
def test_every_cell_produces_a_line_the_parser_accepts(tools_path):
    """A cell that will not parse is five array elements that die at startup."""
    import shlex

    from make_stage7_screen import task_line

    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for cell in screen_cells():
        args = parser.parse_args(shlex.split(task_line(cell, 3, 25, 1))[1:])
        assert args.rounds == 25 and args.fold == 3
        assert args.aggregation == "fedavg"
        assert args.trainer == cell["trainer"]
        assert args.init == "global" and args.global_name == "g0"


def test_the_penalty_reaches_the_trainer_keywords(tools_path):
    """``--set`` is only useful if it survives into ``trainer_kwargs``."""
    import shlex

    from make_stage7_screen import task_line

    from federated_outlier_adaptation.cli import _trainer_overrides, build_parser

    parser = build_parser()
    cell = next(c for c in screen_cells() if c["id"] == "kd_T4_a0p9")
    args = parser.parse_args(shlex.split(task_line(cell, 1, 25, 1))[1:])
    overrides = _trainer_overrides(args)
    assert overrides["space"] == "kd"
    assert overrides["anchor"] == ANCHOR
    assert overrides["T"] == pytest.approx(4.0)
    assert overrides["lam"] == pytest.approx(kd_lam_of_alpha(0.9))


def test_the_fedprox_convention_is_on_the_param_l2_lines(tools_path):
    import shlex

    from make_stage7_screen import task_line

    from federated_outlier_adaptation.cli import _trainer_overrides, build_parser

    parser = build_parser()
    for cell in screen_cells():
        args = parser.parse_args(shlex.split(task_line(cell, 1, 25, 1))[1:])
        overrides = _trainer_overrides(args)
        if cell["method"] == "param_l2":
            assert overrides["param_l2_convention"] == "fedprox"
        else:
            assert "param_l2_convention" not in overrides


def test_only_the_fisher_lines_name_a_fisher_directory(tools_path):
    from make_stage7_screen import task_line

    for cell in screen_cells():
        line = task_line(cell, 1, 25, 1)
        assert ("fisher_path=" in line) is bool(cell["needs_fisher"])
        if cell["needs_fisher"]:
            # the winning fold, not a hard-coded one
            assert "g0_fold$G0_FOLD/global_results/fisher" in line


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #
@pytest.fixture()
def selector(tools_path):
    import select_reg_screen

    return select_reg_screen


RUN_SUFFIX = "_AnchoredTrainer_grid_search"


def _write_run(root, cell_id, fold, per_family, suffix=RUN_SUFFIX):
    """One (cell, fold) folder holding one payload per family, as the driver writes."""
    for family, (adaptation, preservation) in per_family.items():
        run_dir = (
            root / f"coh7_reg_{cell_id}_fold{fold}{suffix}"
            / "fold1_seed_1" / f"{family}_delta" / f"base_agg_x_{family}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "accuracies_0.json", "w") as handle:
            json.dump(
                {
                    "scenario": family,
                    "agg_method_name": "x",
                    "accuracies": [[adaptation, 0.5]] * 25,
                    "pool_val_accuracies": [adaptation] * 25,
                    "final_evaluation": {
                        "clients": {"accuracy": adaptation - 0.01},
                        "old": {"mean": preservation, "sd": 0.0},
                    },
                },
                handle,
            )


def test_the_two_families_are_read_out_of_one_folder(selector, tmp_path):
    """One task, two family results; the payloads say which is which."""
    _write_run(tmp_path, "kd_T4_a0p9", 1,
               {"concurrent": (0.70, 0.90), "sequential": (0.60, 0.88)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    assert {r["family"] for r in rows} == set(FAMILIES)
    by_family = {r["family"]: r for r in rows}
    assert by_family["concurrent"]["adaptation"]["mean"] == pytest.approx(0.70)
    assert by_family["sequential"]["adaptation"]["mean"] == pytest.approx(0.60)


def test_the_selector_fold_means_before_it_compares(selector, tmp_path):
    for fold, value in zip((1, 2, 3, 4, 5), (0.1, 0.2, 0.3, 0.4, 0.5)):
        _write_run(tmp_path, "kd_T4_a0p9", fold, {"concurrent": (value, 0.9)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    assert len(rows) == 1
    assert rows[0]["adaptation"]["mean"] == pytest.approx(0.3)
    assert rows[0]["folds"] == [1, 2, 3, 4, 5]


def test_each_method_wins_within_its_own_family(selector, tmp_path):
    """
    A penalty can help the server average and hurt a sequential walk.

    Collapsing the two would let a method win the table on one schedule while
    being useless on the other, so the winners are chosen per family.
    """
    _write_run(tmp_path, "kd_T4_a0p9", 1,
               {"concurrent": (0.90, 0.9), "sequential": (0.40, 0.9)})
    _write_run(tmp_path, "kd_T8_a0p5", 1,
               {"concurrent": (0.50, 0.9), "sequential": (0.80, 0.9)})
    chosen = selector.select(selector.summarise(selector.collect(tmp_path, screen_cells())))
    kd = chosen["methods"]["kd"]

    assert kd["concurrent"]["winner"] == "kd_T4_a0p9"
    assert kd["sequential"]["winner"] == "kd_T8_a0p5"
    assert kd["agree"] is False


def test_agreeing_families_are_recorded_as_agreeing(selector, tmp_path):
    _write_run(tmp_path, "kd_T4_a0p9", 1,
               {"concurrent": (0.90, 0.9), "sequential": (0.80, 0.9)})
    _write_run(tmp_path, "kd_T8_a0p5", 1,
               {"concurrent": (0.50, 0.9), "sequential": (0.40, 0.9)})
    chosen = selector.select(selector.summarise(selector.collect(tmp_path, screen_cells())))
    assert chosen["methods"]["kd"]["agree"] is True


def test_a_method_with_nothing_measured_has_no_winner(selector, tmp_path):
    _write_run(tmp_path, "kd_T4_a0p9", 1, {"concurrent": (0.9, 0.9)})
    chosen = selector.select(selector.summarise(selector.collect(tmp_path, screen_cells())))
    assert chosen["methods"]["ntd"]["concurrent"]["winner"] is None
    assert chosen["methods"]["kd"]["sequential"]["winner"] is None


def test_the_selector_ranks_on_validation_not_on_test(selector, tmp_path):
    _write_run(tmp_path, "kd_T4_a0p9", 1, {"concurrent": (0.80, 0.9)})
    _write_run(tmp_path, "kd_T8_a0p5", 1, {"concurrent": (0.70, 0.9)})
    path = (
        tmp_path / f"coh7_reg_kd_T8_a0p5_fold1{RUN_SUFFIX}" / "fold1_seed_1"
        / "concurrent_delta" / "base_agg_x_concurrent" / "accuracies_0.json"
    )
    payload = json.loads(path.read_text())
    payload["final_evaluation"]["clients"]["accuracy"] = 0.99
    path.write_text(json.dumps(payload))

    chosen = selector.select(selector.summarise(selector.collect(tmp_path, screen_cells())))
    assert chosen["methods"]["kd"]["concurrent"]["winner"] == "kd_T4_a0p9"


def test_agreeing_winners_emit_one_line_for_both_families(selector, tmp_path):
    _write_run(tmp_path, "kd_T4_a0p9", 1,
               {"concurrent": (0.90, 0.9), "sequential": (0.80, 0.9)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows)
    out = tmp_path / "stage7_full.txt"
    tasks = selector.emit_full(chosen, rows, [1, 2, 3, 4, 5], out, rounds=100)

    text = out.read_text()
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    # one winner plus the two controls
    assert tasks == len(lines) == 3 * 5
    assert "wins in BOTH families" in text
    assert sum("coh7_reg_kd_T4_a0p9_fold" in l for l in lines) == 5


def test_disagreeing_winners_emit_both_and_say_which_to_read(selector, tmp_path):
    _write_run(tmp_path, "kd_T4_a0p9", 1,
               {"concurrent": (0.90, 0.9), "sequential": (0.40, 0.9)})
    _write_run(tmp_path, "kd_T8_a0p5", 1,
               {"concurrent": (0.50, 0.9), "sequential": (0.80, 0.9)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows)
    out = tmp_path / "stage7_full.txt"
    tasks = selector.emit_full(chosen, rows, [1], out, rounds=100)

    text = out.read_text()
    assert "READ THE CONCURRENT RESULT FROM THIS LINE" in text
    assert "READ THE SEQUENTIAL RESULT FROM THIS LINE" in text
    # two winners plus the two controls
    assert tasks == 4


def test_the_full_file_always_carries_both_controls(selector, tmp_path):
    _write_run(tmp_path, "ntd_b1_t2", 1, {"concurrent": (0.9, 0.9)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    out = tmp_path / "stage7_full.txt"
    selector.emit_full(selector.select(rows), rows, [1], out, rounds=100)
    text = out.read_text()
    assert "coh7_reg_param_l2_mu0_fold1" in text
    assert "coh7_reg_control_none_fold1" in text


def test_the_emitted_full_lines_parse(selector, tmp_path):
    import shlex

    from federated_outlier_adaptation.cli import build_parser

    _write_run(tmp_path, "fisher_lam1000", 1, {"concurrent": (0.9, 0.9)})
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    out = tmp_path / "stage7_full.txt"
    selector.emit_full(selector.select(rows), rows, [1], out, rounds=100)

    parser = build_parser()
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == 100 and args.func.__name__ == "cmd_final"


def test_the_control_cell_uses_no_anchored_trainer():
    control = control_cell()
    assert control["trainer"] == "BaseTrainer"
    assert control["needs_fisher"] is False


# --------------------------------------------------------------------------- #
# The hybrid emitter
# --------------------------------------------------------------------------- #
def test_the_hybrid_takes_T_from_kd_and_needs_a_fisher(tools_path):
    from emit_stage7_hybrid import hybrid_cells

    cells, kd, fisher = hybrid_cells("kd_T8_a0p3", "fisher_lam10000")
    assert len(cells) == 3
    assert {c["hypers"]["mix"] for c in cells} == set(HYBRID_MIXES)
    for cell in cells:
        assert cell["hypers"]["T"] == pytest.approx(8.0)
        assert cell["hypers"]["lam"] == pytest.approx(kd_lam_of_alpha(0.3))
        assert cell["needs_fisher"] is True
        assert cell["space"] == "kd+fisher"


def test_the_hybrid_refuses_the_wrong_kind_of_cell(tools_path):
    from emit_stage7_hybrid import hybrid_cells

    with pytest.raises(SystemExit, match="Expected a kd cell"):
        hybrid_cells("ntd_b1_t2", "fisher_lam10")
    with pytest.raises(SystemExit, match="Unknown stage-7 cell"):
        hybrid_cells("kd_T8_a0p3", "no_such_cell")


def test_the_hybrid_lines_parse(tools_path, tmp_path):
    import shlex

    from emit_stage7_hybrid import hybrid_cells
    from make_stage7_screen import task_line

    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    cells, _, _ = hybrid_cells("kd_T8_a0p3", "fisher_lam10000")
    for cell in cells:
        line = task_line(cell, 2, 100, 1)
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == 100
        assert "fisher_path=" in line and "mix=" in line


def test_the_emitted_hyperparameters_round_trip_exactly(tools_path):
    """
    A label may round; a value may not.

    ``{:g}`` gives six significant figures, so the distillation lam of
    alpha = 0.3 - 2.3333333333333335 - was emitted as ``2.33333``, which is a
    different objective. Every emitted number must parse back to the double the
    cell table holds.
    """
    import shlex

    from make_stage7_screen import task_line

    from federated_outlier_adaptation.cli import _trainer_overrides, build_parser

    parser = build_parser()
    for cell in screen_cells():
        args = parser.parse_args(shlex.split(task_line(cell, 1, 25, 1))[1:])
        overrides = _trainer_overrides(args)
        for name, value in cell["hypers"].items():
            assert float(overrides[name]) == float(value), (cell["id"], name)


def test_the_screen_is_the_documented_size():
    """
    125 now: the kd and ntd rows gained boundary extensions.

    They are appended rather than inserted, so every earlier cell keeps its
    index - and therefore the sampler seed a screen derives from it.
    """
    grouped = cells_by_method()
    assert len(screen_cells()) == 125
    assert [len(grouped[m]) for m in methods()] == [7, 8, 8, 5, 5, 53, 39]
    assert SCREEN_ROUNDS == 25
