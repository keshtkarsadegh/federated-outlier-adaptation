"""
Stage 6: the aggregation-contribution screen.

Two things are tested here.  The **rules** the screen needs that did not exist -
a server step and a server anchor whose coefficient comes from the state rather
than from the function's name - and the **table** of cells the screen is, which
is the thing that decides how much GPU time stage 6 costs.
"""

from __future__ import annotations

import json

import pytest
import torch

from federated_outlier_adaptation.aggregation.concurrent_methods import (
    ANCHOR_LAMBDA,
    ServerState,
    con_delta_anchor_lam,
    con_delta_eta,
    con_delta_eta1,
    con_delta_weighted_cgd,
)
from federated_outlier_adaptation.training.agg_cells import (
    FEDOPT_LRS,
    FEDOPT_TAUS,
    PATHS,
    SCREEN_ROUNDS,
    SKIPPED_CELLS,
    cells_by_path,
    screen_cells,
)


def _weights(value: float) -> dict:
    return {"w": torch.full((4,), float(value)), "b": torch.full((2,), float(value))}


@pytest.fixture()
def updates():
    """A global model and two client models that moved away from it."""
    global_weights = _weights(0.0)
    clients = [_weights(1.0), _weights(3.0)]
    return global_weights, clients, [1, 1]


# --------------------------------------------------------------------------- #
# The parameterised server step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eta", [0.1, 0.25, 0.5, 0.75, 1.0])
def test_the_server_step_comes_from_the_state(updates, eta):
    """
    One rule, five coefficients.

    The three fixed rules pin eta_s at 0.25, 0.5 and 1.0, which leaves no way to
    ask for 0.1 or 0.75 - the two ends of the screened row.
    """
    global_weights, clients, counts = updates
    state = ServerState(eta=eta)
    result = con_delta_eta(global_weights, clients, counts, server_state=state)
    # equal counts, so the mean update is 2.0 and the step is eta * 2.0
    assert torch.allclose(result["w"], torch.full((4,), 2.0 * eta))


def test_the_swept_step_at_one_is_plain_fedavg(updates):
    global_weights, clients, counts = updates
    swept = con_delta_eta(global_weights, clients, counts, server_state=ServerState(eta=1.0))
    fixed = con_delta_eta1(global_weights, clients, counts, server_state=ServerState())
    plain = con_delta_weighted_cgd(global_weights, clients, counts)
    for key in global_weights:
        assert torch.allclose(swept[key], fixed[key])
        assert torch.allclose(swept[key], plain[key])


def test_the_swept_step_honours_the_weighting(updates):
    """Uniform and proportional differ once the clients differ in size."""
    global_weights, clients, _ = updates
    counts = [1, 9]
    proportional = con_delta_eta(
        global_weights, clients, counts,
        server_state=ServerState(eta=1.0, weighting="proportional"),
    )
    uniform = con_delta_eta(
        global_weights, clients, counts,
        server_state=ServerState(eta=1.0, weighting="uniform"),
    )
    assert not torch.allclose(proportional["w"], uniform["w"])
    assert torch.allclose(uniform["w"], torch.full((4,), 2.0))


# --------------------------------------------------------------------------- #
# The parameterised server anchor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [0.01, 0.03, 0.1, 0.3, 1.0])
def test_the_anchor_pull_comes_from_the_state(lam):
    """theta + eta Delta - lambda (theta - theta_g), with lambda swept."""
    global_weights = _weights(2.0)
    clients = [_weights(4.0)]
    state = ServerState(eta=1.0, anchor_lambda=lam)
    state.frozen_global = _weights(0.0)

    result = con_delta_anchor_lam(global_weights, clients, [1], server_state=state)
    # step: 2 + 1*(4-2) = 4; pull: -lam * (2 - 0)
    assert torch.allclose(result["w"], torch.full((4,), 4.0 - lam * 2.0))


def test_an_anchor_pull_of_zero_is_the_plain_step():
    """The low end of the row is a genuine control, not a near-miss."""
    global_weights = _weights(2.0)
    clients = [_weights(4.0)]
    state = ServerState(eta=1.0, anchor_lambda=0.0)
    state.frozen_global = _weights(0.0)

    anchored = con_delta_anchor_lam(global_weights, clients, [1], server_state=state)
    plain = con_delta_eta(
        global_weights, clients, [1], server_state=ServerState(eta=1.0)
    )
    for key in global_weights:
        assert torch.allclose(anchored[key], plain[key])


def test_the_anchor_default_matches_the_fixed_rule_it_replaces():
    assert ANCHOR_LAMBDA == 0.1
    assert ServerState().anchor_lambda == 0.1


def test_a_negative_anchor_pull_is_refused():
    """A negative lambda would push *away* from g-0, which is not the rule."""
    with pytest.raises(ValueError, match="cannot be negative"):
        ServerState(anchor_lambda=-0.1)


def test_the_anchor_is_in_the_provenance():
    assert ServerState(anchor_lambda=0.3).hyperparameters()["anchor_lambda"] == 0.3


# --------------------------------------------------------------------------- #
# The command line reaches the state
# --------------------------------------------------------------------------- #
def test_the_server_anchor_flag_reaches_the_server_state():
    from federated_outlier_adaptation.cli import _server_kwargs, build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["final", "--trainer", "BaseTrainer",
         "--aggregation", "con_delta_anchor_lam", "--server-anchor", "0.03"]
    )
    kwargs = _server_kwargs(args)
    assert kwargs["anchor_lambda"] == pytest.approx(0.03)
    assert ServerState(**kwargs).anchor_lambda == pytest.approx(0.03)


def test_an_unset_server_anchor_leaves_the_default_alone():
    from federated_outlier_adaptation.cli import _server_kwargs, build_parser

    args = build_parser().parse_args(["final", "--trainer", "BaseTrainer"])
    assert "anchor_lambda" not in _server_kwargs(args)


# --------------------------------------------------------------------------- #
# The cell table
# --------------------------------------------------------------------------- #
def test_the_screen_is_the_documented_size():
    """
    96 after the reparameterisation.

    Three knobs are now swept as the quantity whose meaning is stable across
    horizons and federation sizes - a half-life, a retention, a client count -
    and the weighting as an exponent rather than as three names. FedAdam and
    FedYogi use the grid of the paper that introduced them, which is where most
    of the reduction from 171 comes from.
    """
    cells = screen_cells()
    grouped = cells_by_path()
    assert len(cells) == 96
    assert len(grouped["concurrent"]) == 83
    assert len(grouped["sequential"]) == 11
    assert len(grouped["control"]) == 2
    assert SCREEN_ROUNDS == 25


def test_every_cell_id_is_unique():
    ids = [cell["id"] for cell in screen_cells()]
    assert len(ids) == len(set(ids))


def test_the_fedopt_grids_are_the_stated_shape():
    assert len(FEDOPT_LRS) == 4 and len(FEDOPT_TAUS) == 7
    for prefix in ("fedadam", "fedyogi"):
        cells = [c for c in screen_cells() if c["id"].startswith(prefix)]
        assert len(cells) == 28
        assert len({(c["flags"]["server_lr"], c["flags"]["server_tau"]) for c in cells}) == 28


def test_every_rule_named_by_a_cell_exists_in_exactly_one_family():
    """
    A cell whose rule exists nowhere is 5 array elements that fail at startup.

    The controls name the ``fedavg`` variant key, which deliberately spans two
    families; every other cell must resolve to exactly one.
    """
    from federated_outlier_adaptation.aggregation.selector import (
        AGG_VARIANTS,
        SKIP_FAMILY,
        resolve_aggregation,
    )

    for cell in screen_cells():
        rule = cell["rule"]
        if rule in AGG_VARIANTS:
            assert cell["path"] == "control"
            continue
        families = [
            (scenario, metadata)
            for scenario in ("concurrent", "sequential")
            for metadata in ("weights", "delta")
            if resolve_aggregation(scenario, metadata, rule, extended=True) != SKIP_FAMILY
        ]
        assert len(families) == 1, (cell["id"], rule, families)


def test_every_cell_flag_has_a_command_line_flag():
    """A coefficient with no flag is a cell that silently runs at the default."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from make_stage6_screen import FLAGS

    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()

    # A cell may store the stage-independent form of a knob - a half-life in
    # rounds, a retention over clients, a count of clients - which the emitter
    # converts. Those names are legitimately absent from the flag table; what
    # must have a command-line flag is what is actually emitted.
    from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01
    from federated_outlier_adaptation.training.study_lines import resolve_agg_flags

    for cell in screen_cells():
        for name in resolve_agg_flags(cell["flags"], DIGITS_STUDY01, 25):
            assert name in FLAGS, (cell["id"], name)
    # The flags live on the `final` subparser, so probe it rather than the top
    # level: parsing is also a stronger check than looking the name up.
    args = parser.parse_args(["final", "--trainer", "BaseTrainer"])
    for name in FLAGS:
        assert hasattr(args, name), name


def test_the_skipped_grid_points_say_what_they_duplicate():
    assert "weight_proportional" in SKIPPED_CELLS
    assert "seq_delta_fedavg_update" in SKIPPED_CELLS
    assert "seq_delta_progressive_update" in SKIPPED_CELLS
    ids = {cell["id"] for cell in screen_cells()}
    # nothing is both run and recorded as skipped
    assert not (ids & set(SKIPPED_CELLS))


def test_the_sequential_delta_duplicates_really_are_duplicates():
    """
    The claim in SKIPPED_CELLS, checked rather than asserted in a comment.

    ``theta + alpha (theta_k - theta)`` is ``(1 - alpha) theta + alpha theta_k``,
    so the delta form of the progressive rule is the incremental rule.
    """
    from federated_outlier_adaptation.aggregation.sequential_methods import (
        seq_delta_progressive_update,
        seq_incremental_update,
    )

    global_weights = _weights(2.0)
    client = _weights(5.0)
    kwargs = dict(num_clients=4, index=3)
    left = seq_incremental_update(global_weights, client, **kwargs)
    right = seq_delta_progressive_update(global_weights, client, **kwargs)
    for key in global_weights:
        assert torch.allclose(left[key], right[key])


# --------------------------------------------------------------------------- #
# The selector
# --------------------------------------------------------------------------- #
#: What the driver actually names a run folder: it builds its output path from
#: ``<parent>_<trainer>_grid_search``, so the fold number is not the tail.
RUN_SUFFIX = "_BaseTrainer_grid_search"


def _write_run(root, cell_id, fold, adaptation, preservation, suffix=RUN_SUFFIX):
    run_dir = root / f"coh6_agg_{cell_id}_fold{fold}{suffix}" / "fold1_seed_1" / "job"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "accuracies_0.json", "w") as handle:
        json.dump(
            {
                "accuracies": [[adaptation, 0.5]] * 25,
                "pool_val_accuracies": [adaptation] * 25,
                "final_evaluation": {
                    "clients": {"accuracy": adaptation - 0.01},
                    "old": {"mean": preservation, "sd": 0.0},
                },
            },
            handle,
        )


@pytest.fixture()
def selector(tmp_path, monkeypatch):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "tools"))
    import select_agg_screen

    return select_agg_screen


def test_the_selector_reads_the_folder_the_runner_actually_writes(selector, tmp_path):
    """
    Run folders carry the driver's ``_<trainer>_grid_search`` suffix.

    The selector used to read the fold number off the tail of the folder name,
    so ``coh6_agg_eta_1_fold3_BaseTrainer_grid_search`` parsed as fold
    ``"3_BaseTrainer_grid_search"``, failed ``isdigit()``, and was skipped -
    silently, for all 845 of them, reporting that the screen had produced
    nothing.  Both forms must be read.
    """
    _write_run(tmp_path, "eta_1", 3, 0.7, 0.9)                 # suffixed
    _write_run(tmp_path, "eta_1", 4, 0.9, 0.9, suffix="")      # bare
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))

    assert len(rows) == 1
    assert rows[0]["folds"] == [3, 4]
    assert rows[0]["adaptation"]["mean"] == pytest.approx(0.8)


def test_the_folder_name_is_parsed_into_a_cell_and_a_fold(selector):
    parse = selector.parse_run_dir
    assert parse("coh6_agg_eta_1_fold3_BaseTrainer_grid_search") == ("eta_1", 3)
    assert parse("coh6_agg_eta_1_fold3") == ("eta_1", 3)
    # a cell id with underscores of its own survives the split
    assert parse("coh6_agg_fedadam_lr0p01_tau0p001_fold5_BaseTrainer_grid_search") == (
        "fedadam_lr0p01_tau0p001", 5
    )
    assert parse("coh6_fl_g0_fold1_BaseTrainer_grid_search") is None
    assert parse("something_else") is None


def test_a_cell_does_not_collect_another_cell_s_folders(selector, tmp_path):
    """``fedadam_lr0p1`` globs ``fedadam_lr0p1_tau...`` too; ids must match."""
    _write_run(tmp_path, "fedadam_lr0p1_tau0p001", 1, 0.7, 0.9)
    found = selector.collect(tmp_path, screen_cells())
    assert set(found) == {"fedadam_lr0p1_tau0p001"}


def test_the_selector_fold_means_before_it_compares(selector, tmp_path):
    """Five folds of one cell are averaged; a cell wins only on average."""
    for fold, value in zip((1, 2, 3, 4, 5), (0.1, 0.2, 0.3, 0.4, 0.5)):
        _write_run(tmp_path, "eta_1", fold, value, 0.8)
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    assert len(rows) == 1
    assert rows[0]["adaptation"]["mean"] == pytest.approx(0.3)
    assert rows[0]["adaptation"]["n"] == 5
    assert rows[0]["folds"] == [1, 2, 3, 4, 5]


def test_the_budget_picks_the_best_adapter_within_it(selector, tmp_path):
    """
    The constrained rule, in one picture.

    ``eta_0p1`` adapts worst and preserves best; ``eta_1`` adapts best and
    preserves worst.  With a budget of 0.02 the middle cell is the only one that
    both stays inside the budget and beats the conservative one.
    """
    for fold in (1, 2):
        _write_run(tmp_path, "eta_0p1", fold, 0.50, 0.90)   # preserves best
        _write_run(tmp_path, "eta_0p5", fold, 0.70, 0.885)  # inside a 0.02 budget
        _write_run(tmp_path, "eta_1", fold, 0.90, 0.60)     # far outside it
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows, epsilon=0.02)["paths"]["concurrent"]

    assert chosen["winner"] == "eta_0p5"
    assert chosen["eligible"] == 2 and chosen["considered"] == 3
    assert chosen["preservation_floor"] == pytest.approx(0.88)


def test_a_wider_budget_buys_more_adaptation(selector, tmp_path):
    for fold in (1, 2):
        _write_run(tmp_path, "eta_0p1", fold, 0.50, 0.90)
        _write_run(tmp_path, "eta_1", fold, 0.90, 0.60)
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    assert selector.select(rows, epsilon=0.01)["paths"]["concurrent"]["winner"] == "eta_0p1"
    assert selector.select(rows, epsilon=0.50)["paths"]["concurrent"]["winner"] == "eta_1"


def test_a_path_with_nothing_measured_has_no_winner(selector, tmp_path):
    for fold in (1, 2):
        _write_run(tmp_path, "eta_1", fold, 0.9, 0.6)
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows, epsilon=0.01)
    assert chosen["paths"]["sequential"]["winner"] is None
    assert chosen["paths"]["sequential"]["eligible"] == 0


def test_the_selector_ranks_on_validation_not_on_test(selector, tmp_path):
    """
    Selecting on the test rows would be choosing the winner with the number
    that is supposed to judge it.  The two cells are ordered one way on
    validation and the other way on test; validation must decide.
    """
    for fold in (1, 2):
        _write_run(tmp_path, "eta_0p5", fold, 0.80, 0.90)
        _write_run(tmp_path, "eta_1", fold, 0.70, 0.90)
    # make the test figures disagree with the validation ones
    for fold in (1, 2):
        path = (
            tmp_path / f"coh6_agg_eta_1_fold{fold}{RUN_SUFFIX}"
            / "fold1_seed_1" / "job" / "accuracies_0.json"
        )
        payload = json.loads(path.read_text())
        payload["final_evaluation"]["clients"]["accuracy"] = 0.99
        path.write_text(json.dumps(payload))

    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    assert selector.select(rows, epsilon=0.05)["paths"]["concurrent"]["winner"] == "eta_0p5"


def test_the_full_horizon_file_carries_the_winners_and_the_controls(selector, tmp_path):
    for fold in (1, 2):
        _write_run(tmp_path, "eta_0p5", fold, 0.80, 0.90)
        _write_run(tmp_path, "seq_fedavg", fold, 0.60, 0.88)
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows, epsilon=0.05)

    out = tmp_path / "stage6_full.txt"
    tasks = selector.emit_full(chosen, rows, [1, 2, 3, 4, 5], out, rounds=100)
    text = out.read_text()
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]

    # two winners plus the two controls, five folds each
    assert tasks == len(lines) == 4 * 5
    assert "--rounds 100" in lines[0]
    assert any("coh6_agg_eta_0p5_fold" in l for l in lines)
    assert any("coh6_agg_seq_fedavg_fold" in l for l in lines)
    assert any("coh6_agg_control_fedavg_fold" in l for l in lines)


def test_the_emitted_full_lines_parse(selector, tmp_path):
    import shlex

    from federated_outlier_adaptation.cli import build_parser

    for fold in (1, 2):
        _write_run(tmp_path, "fedadam_lr0p01_tau0p001", fold, 0.80, 0.90)
    rows = selector.summarise(selector.collect(tmp_path, screen_cells()))
    chosen = selector.select(rows, epsilon=0.05)
    out = tmp_path / "stage6_full.txt"
    selector.emit_full(chosen, rows, [1], out, rounds=100)

    parser = build_parser()
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == 100 and args.func.__name__ == "cmd_final"
