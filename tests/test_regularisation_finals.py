"""
The regularisation family's finals run with the selected lambda, not the default.

``plan_regularisation_family`` emits the sweeps and then either the finals -
when the per-sweep selection files are already on disk - or the single
``foa select --emit-finals`` line that produces them.  The tests build small
sweep trees whose winner is known by construction and check both shapes, the
values that end up on the command line, and that every emitted line parses.
"""

from __future__ import annotations

import json
import shlex

import pytest

from federated_outlier_adaptation.cli import build_parser, main as cli_main
from federated_outlier_adaptation.grid_search import selection
from federated_outlier_adaptation.training import matrix

ROUNDS = 6


def population(adaptation: float, forgetting: float) -> dict:
    """
    A configuration that ends at ``adaptation`` and forgets ``forgetting``.

    The selection averages the last five rounds, so the source curve drops once
    after round 0 and then stays put: the averaged drop is exactly
    ``forgetting`` and the budget arithmetic is checkable by hand.
    """
    source = [0.99] + [0.99 - forgetting] * (ROUNDS - 1)
    return {
        "heldout_client_accuracies": [adaptation] * ROUNDS,
        "pool_val_accuracies": [adaptation] * ROUNDS,
        "pool_test_accuracies": [adaptation] * ROUNDS,
        "source_val_accuracies": source,
        "accuracies": [[adaptation, source[r]] for r in range(ROUNDS)],
    }


def write_sweep(root, space: str, anchor: str, entries: dict) -> "object":
    """
    One stored anchored sweep.

    ``entries`` maps an experiment name to ``(kwargs, adaptation, forgetting)``,
    exactly the shape ``run_grid_config`` writes into ``config_points_*.json``.
    """
    folder = root / f"anchored_{space.replace('+', '_')}_{anchor}_grid_search"
    directory = folder / "concurrent_weights"
    directory.mkdir(parents=True, exist_ok=True)

    configs = {}
    for name, (kwargs, adaptation, forgetting) in entries.items():
        configs[name] = {
            "trainer": "AnchoredTrainer",
            "trainer_kwargs": {**kwargs, "space": space, "anchor": anchor},
            "hyperparameters": dict(kwargs),
            "scenario": "concurrent",
            "metadata": "weights",
            "population": population(adaptation, forgetting),
        }
    with open(directory / "config_points_100.json", "w") as handle:
        json.dump({"config": configs}, handle)
    return folder


@pytest.fixture()
def sweeps(tmp_path):
    """
    Two sweeps: one with a temperature, one without.

    In both, the most adaptive configuration forgets too much and the runner-up
    is the one a 0.005 budget admits, so the selected value is not the extreme
    of the grid.
    """
    write_sweep(
        tmp_path,
        "kd",
        "frozen",
        {
            "anchored_lam1000_T8": ({"lam": 1000, "T": 8}, 0.95, 0.02),
            "anchored_lam10_T4": ({"lam": 10, "T": 4}, 0.90, 0.004),
            "anchored_lam0.001_T1": ({"lam": 0.001, "T": 1}, 0.70, 0.0),
        },
    )
    write_sweep(
        tmp_path,
        "kd+fisher",
        "current",
        {
            "anchored_lam100": ({"lam": 100}, 0.88, 0.001),
            "anchored_lam1": ({"lam": 1}, 0.80, 0.0),
        },
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# discovery and selection
# --------------------------------------------------------------------------- #
def test_sweeps_are_discovered_with_their_space_and_anchor(sweeps):
    found = {(space, anchor) for _, space, anchor in selection.discover_sweeps(sweeps)}
    assert found == {("kd", "frozen"), ("kd+fisher", "current")}


def test_the_folder_name_identifies_a_sweep_without_provenance(tmp_path):
    """``kd+fisher`` flattens to ``kd_fisher`` in the path and is recovered."""
    folder = tmp_path / "anchored_kd_fisher_current_grid_search"
    folder.mkdir(parents=True)
    assert selection.sweep_configuration(folder) == ("kd+fisher", "current")

    folder = tmp_path / "anchored_fisher_scaled_frozen_grid_search"
    folder.mkdir(parents=True)
    assert selection.sweep_configuration(folder) == ("fisher_scaled", "frozen")


def test_a_sweep_written_under_its_own_parent_is_still_found(tmp_path):
    """``foa grid --parent`` puts a prefix in front of the default folder name."""
    folder = tmp_path / "smoke_grid_anchored_logit_l2_frozen_grid_search"
    folder.mkdir(parents=True)
    assert selection.sweep_configuration(folder) == ("logit_l2", "frozen")
    assert [space for _, space, _ in selection.discover_sweeps(tmp_path)] == ["logit_l2"]


def test_only_the_swept_values_reach_the_command_line(tmp_path):
    """
    A space without a temperature must not carry the trainer's default ``T``.

    The sweep engine records exactly the keys the grid varied under
    ``trainer_kwargs``, while ``hyperparameters`` lists every attribute the
    trainer happens to have, default temperature included.
    """
    write_sweep(tmp_path, "param_l2", "frozen", {"anchored_lam1": ({"lam": 1}, 0.8, 0.0)})
    folder = tmp_path / "anchored_param_l2_frozen_grid_search"
    payload = json.loads((folder / "concurrent_weights" / "config_points_100.json").read_text())
    payload["config"]["anchored_lam1"]["hyperparameters"]["T"] = 4.0
    (folder / "concurrent_weights" / "config_points_100.json").write_text(json.dumps(payload))

    selected = selection.select_for_sweep(folder, eps=0.005, write=False)
    assert selected["hyperparameters"] == {"lam": 1}
    task = matrix.reg_final_task("param_l2", "frozen", 0.005, 1, selected["hyperparameters"])
    assert "T=" not in task
    assert "lam=1 " in task


def test_an_unrelated_folder_is_not_a_sweep(tmp_path):
    folder = tmp_path / "anchored_nonsense_grid_search"
    folder.mkdir(parents=True)
    assert selection.sweep_configuration(folder) is None
    assert selection.discover_sweeps(tmp_path) == []


def test_the_budget_decides_which_lambda_is_selected(sweeps):
    folder = sweeps / "anchored_kd_frozen_grid_search"

    tight = selection.select_for_sweep(folder, eps=0.005, write=False)
    assert tight["hyperparameters"] == {"lam": 10, "T": 4}
    assert tight["space"] == "kd" and tight["anchor"] == "frozen"

    loose = selection.select_for_sweep(folder, eps=0.05, write=False)
    assert loose["hyperparameters"] == {"lam": 1000, "T": 8}

    strict = selection.select_for_sweep(folder, eps=0.0, write=False)
    assert strict["hyperparameters"] == {"lam": 0.001, "T": 1}


def test_a_space_without_a_temperature_selects_lambda_only(sweeps):
    folder = sweeps / "anchored_kd_fisher_current_grid_search"
    payload = selection.select_for_sweep(folder, eps=0.005, write=False)
    assert payload["hyperparameters"] == {"lam": 100}


def test_an_impossible_budget_selects_nothing(sweeps):
    folder = sweeps / "anchored_kd_frozen_grid_search"
    payload = selection.select_for_sweep(folder, eps=-1.0, write=False)
    assert payload["selected"] is None
    assert payload["hyperparameters"] == {}


def test_the_selection_is_written_next_to_its_sweep(sweeps):
    folder = sweeps / "anchored_kd_frozen_grid_search"
    selection.select_for_sweep(folder, eps=0.005)
    path = folder / selection.selection_filename(0.005)
    assert path.is_file()
    assert selection.load_selection(folder, 0.005)["hyperparameters"] == {"lam": 10, "T": 4}
    # A budget that was never selected has no file and reads back as absent.
    assert selection.load_selection(folder, 0.01) is None


def test_selections_are_collected_per_sweep_and_budget(sweeps):
    for folder, _, _ in selection.discover_sweeps(sweeps):
        for eps in (0.005, 0.01):
            selection.select_for_sweep(folder, eps=eps)
    payloads = selection.collect_selections(sweeps, (0.005, 0.01))
    assert len(payloads) == 4
    assert {p["space"] for p in payloads} == {"kd", "kd+fisher"}


# --------------------------------------------------------------------------- #
# the emitted task lines
# --------------------------------------------------------------------------- #
def test_a_final_line_carries_the_selected_values():
    task = matrix.reg_final_task("kd", "frozen", 0.005, seed=2, hyperparameters={"lam": 10, "T": 4})
    assert "--trainer AnchoredTrainer" in task
    assert "--parent reg_kd_frozen_eps0005" in task
    assert '--set space="kd" anchor="frozen" T=4 lam=10' in task
    assert "--seed 2" in task
    assert "--skip-existing" in task
    assert "--aggregation fedavg" in task


def test_a_final_line_without_a_selection_keeps_the_trainer_defaults():
    task = matrix.reg_final_task("param_l2", "current", 0.01, seed=1)
    assert '--set space="param_l2" anchor="current" --aggregation' in task


def test_the_parent_folder_encodes_space_anchor_and_budget():
    assert matrix.reg_parent("kd+fisher", "current", 0.0025) == "reg_kd_fisher_current_eps00025"


def test_finals_are_emitted_once_per_selection_and_seed(sweeps):
    for folder, _, _ in selection.discover_sweeps(sweeps):
        selection.select_for_sweep(folder, eps=0.005)
    tasks = matrix.emit_regularisation_finals(sweeps, eps_values=(0.005,), seeds=(1, 2, 3))
    assert len(tasks) == 2 * 3
    assert sum("lam=10" in task and "T=4" in task for task in tasks) == 3
    assert sum("lam=100" in task for task in tasks) == 3


def test_a_sweep_the_budget_rejected_produces_no_final(sweeps):
    folder = sweeps / "anchored_kd_frozen_grid_search"
    selection.select_for_sweep(folder, eps=-1.0)
    assert matrix.emit_regularisation_finals(sweeps, eps_values=(-1.0,), seeds=(1,)) == []


@pytest.mark.parametrize("provider", ["shakespeare", "cifar10"])
def test_emitted_finals_follow_the_dataset(sweeps, provider):
    for folder, _, _ in selection.discover_sweeps(sweeps):
        selection.select_for_sweep(folder, eps=0.005)
    tasks = matrix.emit_regularisation_finals(
        sweeps, eps_values=(0.005,), seeds=(1,), provider=provider
    )
    assert all(f"--provider {provider}" in task for task in tasks)


def test_every_emitted_final_parses_as_a_cli_invocation(sweeps):
    for folder, _, _ in selection.discover_sweeps(sweeps):
        for eps in (0.0025, 0.005, 0.01):
            selection.select_for_sweep(folder, eps=eps)
    parser = build_parser()
    tasks = matrix.emit_regularisation_finals(sweeps)
    assert tasks
    for task in tasks:
        parser.parse_args(shlex.split(task)[1:])


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_the_plan_emits_the_finals_once_the_selections_exist(sweeps):
    for folder, _, _ in selection.discover_sweeps(sweeps):
        for eps in matrix.SELECTION_EPS:
            selection.select_for_sweep(folder, eps=eps)

    tasks = matrix.plan_tasks("regularisation_family", results_dir=sweeps)
    grids = [t for t in tasks if t.startswith("foa grid")]
    finals = [t for t in tasks if t.startswith("foa final")]
    assert len(grids) == len(matrix.ANCHOR_SPACES) * len(matrix.ANCHOR_KINDS) * len(
        matrix.GRID_SEEDS
    )
    # Two sweeps exist on disk, at three budgets, over three seeds.
    assert len(finals) == 2 * len(matrix.SELECTION_EPS) * len(matrix.POOL_SEEDS)
    assert not any(t.startswith("foa select") for t in tasks)
    assert max(tasks.index(t) for t in grids) < min(tasks.index(t) for t in finals)


def test_an_empty_results_root_defers_to_the_selection(tmp_path):
    tasks = matrix.plan_tasks("regularisation_family", results_dir=tmp_path)
    assert not any(t.startswith("foa final") for t in tasks)
    assert tasks[-1].startswith("foa select ")
    assert "--emit-finals" in tasks[-1]
    assert f"--root {tmp_path} " in tasks[-1]
    assert str(tmp_path / matrix.REG_FINALS_TASK_FILE) in tasks[-1]


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_writes_the_selections_and_the_task_file(sweeps, tmp_path, capsys):
    out = tmp_path / "tasks_reg.txt"
    code = cli_main(
        [
            "select",
            "--root",
            str(sweeps),
            "--emit-finals",
            "--eps",
            "0.005",
            "0.01",
            "--seeds",
            "1",
            "2",
            "--out",
            str(out),
        ]
    )
    assert code == 0

    for folder, _, _ in selection.discover_sweeps(sweeps):
        for eps in (0.005, 0.01):
            assert (folder / selection.selection_filename(eps)).is_file()

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2 * 2 * 2
    assert all(line.startswith("foa final --trainer AnchoredTrainer") for line in lines)
    assert any("--parent reg_kd_frozen_eps0005" in line for line in lines)
    assert any("--parent reg_kd_fisher_current_eps001" in line for line in lines)

    report = json.loads(capsys.readouterr().out)
    assert report["sweeps"] == 2
    assert report["tasks"] == 8


def test_a_plain_selection_still_takes_a_single_budget(sweeps, capsys):
    """``foa select --eps 0.005`` keeps behaving exactly as it did."""
    assert cli_main(["select", "--root", str(sweeps), "--eps", "0.005"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["eps"] == 0.005
    assert (sweeps / "constrained_selection.json").is_file()
