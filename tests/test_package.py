"""
Package-level checks: every module imports, the CLI is wired up, the trainer
registry resolves every trainer, and the aggregation selector exposes the method
names the experiment scripts reference.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import federated_outlier_adaptation as package
from federated_outlier_adaptation import config
from federated_outlier_adaptation.aggregation.selector import CLASS_MAP, select_class
from federated_outlier_adaptation.cli import GRID_METHODS, build_parser
from federated_outlier_adaptation.trainers.registry import available_trainers, get_trainer_class

EXPECTED_TRAINERS = [
    "BaseTrainer",
    "CFAlignedFeatureTrainer",
    "CFLogitConsistencyTrainer",
    "CFProxTrainer",
    "DistillationEWCTrainer",
    "DistillationTrainer",
    "EWCTrainer",
]

# Aggregation names referenced by the grid searches and training drivers.
REFERENCED_AGGREGATIONS = {
    ("concurrent", "weights"): ["con_weighted_cgw", "con_weighted_cw", "con_capped_cgw"],
    ("concurrent", "delta"): ["con_delta_weighted_cgd", "con_delta_capped_cgd"],
    ("sequential", "weights"): ["seq_fedavg_update"],
    ("sequential", "delta"): ["seq_delta_fedavg_update"],
}


def test_every_module_imports():
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)


def test_no_legacy_import_root():
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        source = importlib.import_module(module.name).__file__
        if source is None:
            continue
        with open(source, "r", encoding="utf-8") as handle:
            text = handle.read()
        assert "from src." not in text, module.name
        assert "import src." not in text, module.name


def test_config_defaults_are_under_the_repo():
    assert config.DATA_DIR.name == "data" or config.DATA_DIR.is_absolute()
    assert config.CACHE_DIR.is_absolute()
    assert config.LOG_DIR.is_absolute()
    assert config.outliers_file(5) == config.SELECTED_OUTLIERS_JSON
    assert config.outliers_file(3).name == "selected_outliers_k3.json"


def test_env_value_prefers_the_current_prefix(monkeypatch):
    monkeypatch.delenv("FOA_TEST_SETTING", raising=False)
    monkeypatch.delenv("FAL_TEST_SETTING", raising=False)
    assert config.env_value("TEST_SETTING") is None

    # The name of the earlier repository is still honoured ...
    monkeypatch.setenv("FAL_TEST_SETTING", "legacy")
    assert config.env_value("TEST_SETTING") == "legacy"

    # ... but the current one wins when both are set.
    monkeypatch.setenv("FOA_TEST_SETTING", "current")
    assert config.env_value("TEST_SETTING") == "current"
    assert config.env_name("TEST_SETTING") == "FOA_TEST_SETTING"


@pytest.mark.parametrize("name", EXPECTED_TRAINERS)
def test_registry_resolves_trainer(name):
    cls = get_trainer_class(name)
    assert cls.__name__ == name


def test_registry_lists_all_trainers():
    assert set(EXPECTED_TRAINERS).issubset(set(available_trainers()))


@pytest.mark.parametrize("key,names", list(REFERENCED_AGGREGATIONS.items()), ids=str)
def test_referenced_aggregations_exist(key, names):
    scenario, metadata = key
    cls = select_class(scenario, metadata)
    available, _ = cls.list_methods()
    for name in names:
        assert name in available, f"{name} missing from {cls.__name__}"


def test_selector_rejects_unknown_combination():
    assert set(CLASS_MAP) == {"sequential", "concurrent"}
    with pytest.raises(ValueError):
        select_class("sequential", "nope")


def test_cli_parser_exposes_every_command():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for action in actions:
        if action.dest == "command":
            commands = set(action.choices)
    expected = {
        "prepare-data",
        "global-train",
        "select-outliers",
        "combined-train",
        "base-fl",
        "all-aggs",
        "grid",
        "final",
        "extreme",
        "select",
        "signals",
        "figures",
    }
    assert expected.issubset(commands)


def test_cli_grid_methods_resolve():
    for method, (module_name, func_name) in GRID_METHODS.items():
        module = importlib.import_module(module_name)
        assert callable(getattr(module, func_name)), method


def test_cli_help_exits_cleanly(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
