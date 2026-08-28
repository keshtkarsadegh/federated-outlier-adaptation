"""
CLI surface of the new features: the two extra sweeps, the client-population
flags, the deterministic pool selection, the resume flag and the matrix command.
"""

from __future__ import annotations

import importlib

import pytest

from federated_outlier_adaptation.cli import (
    GRID_METHODS,
    SELECTION_MODES,
    SELECTION_POLICIES,
    build_parser,
)
from federated_outlier_adaptation.runners.client_sampler import POLICIES


def _command_names(parser):
    for action in parser._actions:
        if action.dest == "command" and action.choices:
            return set(action.choices)
    return set()


def test_matrix_command_is_wired_up():
    assert "matrix" in _command_names(build_parser())


@pytest.mark.parametrize("method", ["feature", "freeze"])
def test_new_grid_methods_resolve(method):
    module_name, func_name = GRID_METHODS[method]
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


def test_cli_policies_match_the_sampler():
    assert set(SELECTION_POLICIES) == set(POLICIES)


def test_selection_modes():
    """
    ``worst`` joined the three published modes: a fixed cohort, enrolled once
    for the whole of training, as opposed to a pool that per-round selection
    samples from and that can be re-cut at any time.
    """
    assert set(SELECTION_MODES) == {"sample", "lowest", "pool", "worst"}


def test_select_outliers_accepts_the_lowest_mode():
    args = build_parser().parse_args(["select-outliers", "--mode", "lowest", "--k", "20"])
    assert args.mode == "lowest"
    assert args.k == 20


def test_select_outliers_defaults_to_the_published_selection():
    args = build_parser().parse_args(["select-outliers"])
    assert args.mode == "sample"
    assert args.k == 5


@pytest.mark.parametrize("command", ["final", "extreme"])
def test_population_flags_are_available(command):
    base = ["final", "--trainer", "BaseTrainer"] if command == "final" else ["extreme"]
    args = build_parser().parse_args(
        base
        + [
            "--outliers-file",
            "pool.json",
            "--participation",
            "0.5",
            "--policy",
            "worst_first",
            "--sampler-seed",
            "3",
            "--track-clients",
            "--stop-when-global-below-clients",
            "--skip-existing",
        ]
    )
    assert args.outliers_file == "pool.json"
    assert args.participation == 0.5
    assert args.policy == "worst_first"
    assert args.sampler_seed == 3
    assert args.track_clients is True
    assert args.stop_when_global_below_clients is True
    assert args.skip_existing is True


@pytest.mark.parametrize("command", ["final", "extreme"])
def test_population_defaults_are_inert(command):
    base = ["final", "--trainer", "BaseTrainer"] if command == "final" else ["extreme"]
    args = build_parser().parse_args(base)
    assert args.participation == 1.0
    assert args.policy == "all"
    assert args.sampler_seed is None
    assert args.track_clients is False
    assert args.stop_when_global_below_clients is False
    assert args.skip_existing is False


def test_base_fl_supports_the_stop_rule_and_its_own_parent():
    args = build_parser().parse_args(
        ["base-fl", "--parent", "prove_fl_stop", "--stop-when-global-below-clients"]
    )
    assert args.parent == "prove_fl_stop"
    assert args.stop_when_global_below_clients is True
    assert build_parser().parse_args(["base-fl"]).parent == "prove_fl"


def test_matrix_arguments():
    args = build_parser().parse_args(
        ["matrix", "--plan", "e1_seeds", "--out", "tasks.txt", "--minutes-per-task", "9"]
    )
    assert args.plan == "e1_seeds"
    assert args.out == "tasks.txt"
    assert args.minutes_per_task == 9.0


def test_matrix_list_needs_no_plan():
    args = build_parser().parse_args(["matrix", "--list"])
    assert args.list is True
    assert args.plan is None


# --------------------------------------------------------------------------- #
# federated pre-training, named global models, extended aggregations
# --------------------------------------------------------------------------- #
def test_global_train_fl_command_is_wired_up():
    assert "global-train-fl" in _command_names(build_parser())


def test_global_train_fl_defaults():
    args = build_parser().parse_args(["global-train-fl"])
    assert args.rounds == 100
    assert args.participation == 0.1
    assert args.local_epochs == 1
    assert args.global_name == "global_fl"
    assert args.force is False


def test_ntd_grid_method_resolves():
    module_name, func_name = GRID_METHODS["ntd"]
    module = importlib.import_module(module_name)
    assert callable(getattr(module, func_name))


@pytest.mark.parametrize(
    "argv",
    [
        ["base-fl"],
        ["all-aggs"],
        ["grid", "--method", "kd"],
        ["final", "--trainer", "BaseTrainer"],
        ["extreme"],
    ],
    ids=lambda v: v[0],
)
def test_global_name_defaults_to_the_published_model(argv):
    assert build_parser().parse_args(argv).global_name == "global"


@pytest.mark.parametrize(
    "argv",
    [
        ["base-fl"],
        ["all-aggs"],
        ["grid", "--method", "kd"],
        ["final", "--trainer", "BaseTrainer"],
        ["extreme"],
    ],
    ids=lambda v: v[0],
)
def test_global_name_is_settable(argv):
    args = build_parser().parse_args(argv + ["--global-name", "global_fl"])
    assert args.global_name == "global_fl"


@pytest.mark.parametrize(
    "argv", [["all-aggs"], ["final", "--trainer", "BaseTrainer"]], ids=lambda v: v[0]
)
def test_extended_aggregations_is_off_by_default(argv):
    assert build_parser().parse_args(argv).extended_aggregations is False
    assert build_parser().parse_args(argv + ["--extended-aggregations"]).extended_aggregations


# --------------------------------------------------------------------------- #
# round / epoch / batch-size overrides
# --------------------------------------------------------------------------- #
BUDGET_COMMANDS = {
    "final": ["final", "--trainer", "BaseTrainer"],
    "extreme": ["extreme"],
    "grid": ["grid", "--method", "kd"],
}


@pytest.mark.parametrize("command", sorted(BUDGET_COMMANDS))
def test_budget_overrides_default_to_the_published_values(command):
    """Unset means ``None``, which leaves the published budget in place."""
    args = build_parser().parse_args(BUDGET_COMMANDS[command])
    assert args.rounds is None
    assert args.epochs is None
    assert args.batch_size is None


@pytest.mark.parametrize("command", sorted(BUDGET_COMMANDS))
def test_budget_overrides_are_settable(command):
    args = build_parser().parse_args(
        BUDGET_COMMANDS[command]
        + ["--rounds", "2", "--epochs", "1", "--batch-size", "32"]
    )
    assert args.rounds == 2
    assert args.epochs == 1
    assert args.batch_size == 32


def test_unset_budget_overrides_are_not_forwarded():
    """The driver defaults stay authoritative when no flag is given."""
    from federated_outlier_adaptation.cli import _budget_kwargs

    args = build_parser().parse_args(["final", "--trainer", "BaseTrainer"])
    assert _budget_kwargs(args) == {}

    args = build_parser().parse_args(
        ["final", "--trainer", "BaseTrainer", "--rounds", "2", "--epochs", "1"]
    )
    assert _budget_kwargs(args) == {"rounds": 2, "epochs": 1}


def test_final_and_extreme_pass_the_budget_to_the_driver():
    from federated_outlier_adaptation.training.final_experiments import budget_kwargs

    assert budget_kwargs() == {}
    assert budget_kwargs(rounds=2, epochs=1, batch_size=32) == {
        "max_round": 2,
        "epochs": 1,
        "batch_size": 32,
    }


def test_the_grid_engine_keeps_the_published_sweep_budget():
    from federated_outlier_adaptation.grid_search.common import (
        GRID_BATCH_SIZE,
        GRID_EPOCHS,
        GRID_MAX_ROUND,
        resolve_budget,
    )

    assert (GRID_BATCH_SIZE, GRID_EPOCHS, GRID_MAX_ROUND) == (512, 50, 50)
    assert resolve_budget() == (512, 50, 50)
    assert resolve_budget(batch_size=8, epochs=1, max_round=2) == (8, 1, 2)
    assert resolve_budget(max_round=2) == (512, 50, 2)


@pytest.mark.parametrize(
    "module_name,func_name", sorted(set(GRID_METHODS.values())), ids=lambda v: str(v)
)
def test_every_sweep_accepts_the_budget_overrides(module_name, func_name):
    import inspect

    func = getattr(importlib.import_module(module_name), func_name)
    parameters = inspect.signature(func).parameters
    for name in ("batch_size", "epochs", "max_round", "folder"):
        assert name in parameters, (func_name, name)
        assert parameters[name].default is None


def test_grid_writes_into_the_published_folder_by_default():
    assert build_parser().parse_args(["grid", "--method", "kd"]).parent is None
    args = build_parser().parse_args(["grid", "--method", "kd", "--parent", "smoke_kd"])
    assert args.parent == "smoke_kd"


def test_local_finetune_parent_defaults_to_the_published_folder():
    assert build_parser().parse_args(["local-finetune"]).parent == "local_finetune"
    args = build_parser().parse_args(["local-finetune", "--parent", "smoke_local"])
    assert args.parent == "smoke_local"


# --------------------------------------------------------------------------- #
# dataset providers
# --------------------------------------------------------------------------- #
def test_provider_defaults_to_nist():
    for argv in (["final", "--trainer", "BaseTrainer"], ["grid", "--method", "kd"], ["base-fl"]):
        assert build_parser().parse_args(argv).provider == "nist"


@pytest.mark.parametrize("provider", ["shakespeare", "cifar10"])
def test_provider_is_selectable(provider):
    args = build_parser().parse_args(
        ["final", "--trainer", "BaseTrainer", "--provider", provider]
    )
    assert args.provider == provider


def test_prepare_data_dataset_option():
    args = build_parser().parse_args(
        ["prepare-data", "--zip", "x.tar.gz", "--dataset", "cifar10", "--num-clients", "200"]
    )
    assert args.dataset == "cifar10"
    assert args.num_clients == 200


def test_prepare_data_proxy_reserve_defaults_to_zero():
    args = build_parser().parse_args(["prepare-data", "--zip", "x.tar.gz", "--dataset", "cifar10"])
    assert args.proxy_size == 0
    assert args.clients_name is None
    with_reserve = build_parser().parse_args(
        ["prepare-data", "--zip", "x.tar.gz", "--dataset", "cifar10", "--proxy-size", "1000"]
    )
    assert with_reserve.proxy_size == 1000


def test_prepare_data_accepts_the_mnist_proxy_set():
    args = build_parser().parse_args(["prepare-data", "--dataset", "mnist", "--raw-dir", "d"])
    assert args.dataset == "mnist"
    assert args.raw_dir == "d"
    assert args.zip is None


def test_signals_defaults():
    args = build_parser().parse_args(["signals"])
    assert args.root is None
    assert args.eps == 0.005
    assert args.window == 5
    assert args.deltas is None
    assert args.no_plots is False


def test_global_train_accepts_several_selection_sizes():
    args = build_parser().parse_args(
        ["global-train", "--provider", "shakespeare", "--k-values", "5", "20", "50"]
    )
    assert args.k_values == [5, 20, 50]


def test_cli_providers_match_the_registry():
    from federated_outlier_adaptation.cli import PROVIDERS
    from federated_outlier_adaptation.providers import available_providers

    assert set(PROVIDERS) == set(available_providers())


# --------------------------------------------------------------------------- #
# evaluation path and in-sample thinning
# --------------------------------------------------------------------------- #
EVAL_COMMANDS = {
    "final": ["final", "--trainer", "BaseTrainer"],
    "extreme": ["extreme"],
    "grid": ["grid", "--method", "kd"],
}


@pytest.mark.parametrize("command", sorted(EVAL_COMMANDS))
def test_eval_path_defaults_to_the_cache(command):
    """``None`` on the command line, resolved to the cache by the runners."""
    from federated_outlier_adaptation.utils.eval_cache import (
        DEFAULT_EVAL_PATH,
        resolve_eval_path,
    )

    args = build_parser().parse_args(EVAL_COMMANDS[command])
    assert args.eval_path is None
    assert args.insample_every == 1
    assert resolve_eval_path(args.eval_path) == DEFAULT_EVAL_PATH == "cache"


@pytest.mark.parametrize("command", sorted(EVAL_COMMANDS))
def test_eval_path_and_insample_every_are_settable(command):
    args = build_parser().parse_args(
        EVAL_COMMANDS[command] + ["--eval-path", "loader", "--insample-every", "5"]
    )
    assert args.eval_path == "loader"
    assert args.insample_every == 5


def test_cli_eval_paths_match_the_cache_module():
    from federated_outlier_adaptation.cli import EVAL_PATHS as CLI_PATHS
    from federated_outlier_adaptation.utils.eval_cache import EVAL_PATHS

    assert set(CLI_PATHS) == set(EVAL_PATHS)


def test_an_unknown_eval_path_is_rejected():
    from federated_outlier_adaptation.utils.eval_cache import resolve_eval_path

    with pytest.raises(ValueError):
        resolve_eval_path("gpu")


@pytest.mark.parametrize(
    "module_name,func_name", sorted(set(GRID_METHODS.values())), ids=lambda v: str(v)
)
def test_every_sweep_accepts_the_evaluation_options(module_name, func_name):
    import inspect

    func = getattr(importlib.import_module(module_name), func_name)
    parameters = inspect.signature(func).parameters
    assert parameters["eval_path"].default is None
    assert parameters["insample_every"].default == 1


# --------------------------------------------------------------------------- #
# the anchored grid ranges
# --------------------------------------------------------------------------- #
def test_anchored_grid_ranges_default_to_the_published_ones():
    args = build_parser().parse_args(["grid", "--method", "anchored"])
    assert args.lams is None
    assert args.temperatures is None


def test_anchored_grid_ranges_are_settable():
    args = build_parser().parse_args(
        ["grid", "--method", "anchored", "--lams", "0.1", "1", "--temperatures", "2", "8"]
    )
    assert args.lams == [0.1, 1.0]
    assert args.temperatures == [2.0, 8.0]


def test_matrix_resume_and_measured_cost_options():
    args = build_parser().parse_args(
        [
            "matrix",
            "--plan",
            "resume",
            "--from",
            "tasks.txt",
            "--seconds-per-round",
            "4.5",
            "--sweep-seconds-per-round",
            "9",
        ]
    )
    assert args.plan == "resume"
    assert args.source == "tasks.txt"
    assert args.seconds_per_round == 4.5
    assert args.sweep_seconds_per_round == 9.0
