"""
Smoke tests of the federated loops.

Every trainer is run through both runners with a representative aggregation
method of each of the four families (concurrent/sequential x weights/delta).
The tests assert that the loop completes, produces one accuracy pair per round
and records the new timing/communication instrumentation.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.cf_aligned_feature_trainer import (
    CFAlignedFeatureTrainer,
)
from federated_outlier_adaptation.trainers.cf_logit_consistency_trainer import (
    CFLogitConsistencyTrainer,
)
from federated_outlier_adaptation.trainers.cf_prox_trainer import CFProxTrainer
from federated_outlier_adaptation.trainers.distillation_ewc_trainer import (
    DistillationEWCTrainer,
)
from federated_outlier_adaptation.trainers.distillation_trainer import DistillationTrainer
from federated_outlier_adaptation.trainers.ewc_trainer import EWCTrainer
from federated_outlier_adaptation.aggregation.selector import select_class

TRAINERS = [
    BaseTrainer,
    EWCTrainer,
    DistillationTrainer,
    DistillationEWCTrainer,
    CFProxTrainer,
    CFLogitConsistencyTrainer,
    CFAlignedFeatureTrainer,
]

# One representative aggregation method per family.
AGGREGATIONS = [
    ("concurrent", "weights", "con_weighted_cgw"),
    ("concurrent", "delta", "con_delta_weighted_cgd"),
    ("sequential", "weights", "seq_fedavg_update"),
    ("sequential", "delta", "seq_delta_fedavg_update"),
]

MAX_ROUND = 2
EPOCHS = 1
BATCH_SIZE = 4


def run_simulation(trainer_cls, provider, scenario, metadata, agg_name, seed=None):
    """Run one short federated simulation and return the runner."""
    trainer = trainer_cls(provider=provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider, seed=seed)
    agg_method = getattr(select_class(scenario, metadata), agg_name)
    accuracies, _, _, results_dir = runner.simulate(
        exp_name=f"test_{agg_name}",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    return runner, accuracies, results_dir


@pytest.mark.parametrize("trainer_cls", TRAINERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_trainer_runs_through_runner(
    synthetic_provider, trainer_cls, scenario, metadata, agg_name
):
    runner, accuracies, results_dir = run_simulation(
        trainer_cls, synthetic_provider, scenario, metadata, agg_name
    )

    assert len(accuracies) == MAX_ROUND
    for clients_acc, global_acc in accuracies:
        assert 0.0 <= clients_acc <= 1.0
        assert 0.0 <= global_acc <= 1.0
    assert results_dir == synthetic_provider.results_dir


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_instrumentation_is_recorded(synthetic_provider, scenario, metadata, agg_name):
    runner, _, _ = run_simulation(
        BaseTrainer, synthetic_provider, scenario, metadata, agg_name
    )
    metrics = runner.instrumentation

    assert len(metrics["round_seconds"]) == MAX_ROUND
    assert len(metrics["client_seconds"]) == MAX_ROUND
    assert all(len(per_round) == 3 for per_round in metrics["client_seconds"])
    assert len(metrics["comm_bytes_per_round"]) == MAX_ROUND
    assert all(value > 0 for value in metrics["comm_bytes_per_round"])
    assert metrics["param_count"] > 0
    assert metrics["device"].startswith("cpu")


def test_single_outlier_none_keeps_all_clients(synthetic_provider):
    """A falsy client restriction must not filter anything out."""
    trainer = BaseTrainer(provider=synthetic_provider)
    runner = BaseConcurrentRunner(trainer=trainer, provider=synthetic_provider, single_outlier=None)
    agg_method = getattr(select_class("concurrent", "weights"), "con_weighted_cgw")
    runner.simulate(
        exp_name="test_all_clients",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=1,
    )
    assert runner.selected_outliers == synthetic_provider.selected_clients()


def test_single_outlier_restricts_clients(synthetic_provider):
    trainer = BaseTrainer(provider=synthetic_provider)
    runner = BaseConcurrentRunner(
        trainer=trainer, provider=synthetic_provider, single_outlier=["c0"]
    )
    agg_method = getattr(select_class("concurrent", "weights"), "con_weighted_cgw")
    runner.simulate(
        exp_name="test_one_client",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=1,
    )
    assert runner.selected_outliers == ["c0"]


# --------------------------------------------------------------------------- #
# Where the per-round source series is measured
# --------------------------------------------------------------------------- #
def _no_writer_split(provider, monkeypatch):
    """Make the provider behave like a v6 results root: no writer_split.json."""

    def missing(*_args, **_kwargs):
        raise FileNotFoundError(
            f"[Errno 2] No such file or directory: '{provider.results_dir}/writer_split.json'"
        )

    monkeypatch.setattr(provider, "global_client_ids", missing, raising=False)
    monkeypatch.setattr(provider, "local_client_ids", missing, raising=False)


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_a_root_without_a_writer_split_still_runs(
    synthetic_provider, monkeypatch, scenario, metadata, agg_name
):
    """
    The source series is a diagnostic and must not be able to kill a run.

    This is the stage-5 failure: ``get_writers_split`` opened
    ``writer_split.json`` unconditionally, so a v6 results root - which never
    writes one - died in setup, before the first round, with a
    ``FileNotFoundError``, having spent a GPU allocation on nothing.
    """
    _no_writer_split(synthetic_provider, monkeypatch)

    runner, accuracies, _ = run_simulation(
        BaseTrainer, synthetic_provider, scenario, metadata, agg_name
    )
    assert len(accuracies) == MAX_ROUND
    # The clients series is untouched; the source series is simply absent.
    assert all(clients is not None for clients, _ in accuracies)
    assert all(source is None for _, source in accuracies)
    assert runner.population_info()["source_series"]["source"] == "none"
    assert runner.population_info()["source_val_accuracies"] == []


def test_the_missing_series_is_warned_about_not_swallowed(
    synthetic_provider, monkeypatch, caplog
):
    from federated_outlier_adaptation.runners import source_series

    _no_writer_split(synthetic_provider, monkeypatch)
    warnings = []
    monkeypatch.setattr(
        source_series.NistLogger, "warning", lambda message: warnings.append(message)
    )
    _, _, info = source_series.source_loaders(synthetic_provider, batch_size=BATCH_SIZE)

    assert info["source"] == "none"
    assert warnings and "no writer split" in warnings[0]
    assert "--old-book" in warnings[0]


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_a_legacy_root_keeps_the_writer_split_series(
    synthetic_provider, scenario, metadata, agg_name
):
    """v4 and v5 runs are unchanged: the split is there, so the split is used."""
    runner, accuracies, _ = run_simulation(
        BaseTrainer, synthetic_provider, scenario, metadata, agg_name
    )
    info = runner.population_info()["source_series"]
    assert info["source"] == "writer_split"
    assert info["writers"] == len(synthetic_provider.global_client_ids())
    assert all(source is not None for _, source in accuracies)
    assert len(runner.population_info()["source_val_accuracies"]) == MAX_ROUND


def test_the_stop_rule_never_fires_without_a_source_number(
    synthetic_provider, monkeypatch
):
    """The oracle stop rule compares against the source number; there isn't one."""
    _no_writer_split(synthetic_provider, monkeypatch)

    trainer = BaseTrainer(provider=synthetic_provider)
    runner = BaseConcurrentRunner(
        trainer=trainer, provider=synthetic_provider,
        stop_when_global_below_clients=True,
    )
    agg_method = getattr(select_class("concurrent", "weights"), "con_weighted_cgw")
    accuracies, _, _, _ = runner.simulate(
        exp_name="test_no_source_stop",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    assert len(accuracies) == MAX_ROUND
    assert runner.stopped_early is False
