"""
The early-stopped FedAvg arm.

``BaseTrainer`` reproduces the published baseline by default - every local epoch
is run and the best-validation model is kept.  With ``early_stopping=True`` it
adopts the stopping rule every regularised trainer of this package already uses,
so an arm that differs only in the objective can be compared against them.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.registry import build_trainer

BATCH_SIZE = 4


class CountingTrainer(BaseTrainer):
    """A trainer that records how many local epochs it actually ran."""

    def __init__(self, provider, **kwargs):
        super().__init__(provider=provider, **kwargs)
        self.epochs_run = 0

    def train(self, train_loader, eval_loader, epochs=10, patience=None):
        super().train(train_loader, eval_loader, epochs=epochs, patience=patience)
        self.epochs_run = len(self.val_accuracies)


class ConstantModel(nn.Module):
    """A model whose validation accuracy can never improve."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        rows = x.reshape(x.shape[0], -1).shape[0]
        return self.bias.expand(rows, self.bias.shape[0]) * 0.0


def _loaders(provider):
    train, evaluation, _ = provider.build_dataset(
        "c0", train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    return train, evaluation


# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
def test_early_stopping_is_off_by_default(synthetic_provider):
    trainer = BaseTrainer(provider=synthetic_provider)
    assert trainer.early_stopping is False
    assert trainer.patience == 5


def test_the_default_runs_every_epoch(synthetic_provider):
    """The published baseline: no epoch is skipped, whatever validation does."""
    trainer = CountingTrainer(synthetic_provider)
    trainer.set_model(ConstantModel(synthetic_provider.num_classes))
    train_loader, eval_loader = _loaders(synthetic_provider)
    trainer.train(train_loader, eval_loader, epochs=8)
    assert trainer.epochs_run == 8


def test_early_stopping_leaves_the_loop_after_the_patience(synthetic_provider):
    """
    A model that cannot improve stops after ``patience`` non-improving epochs.

    The first epoch always improves on ``-inf``, so the loop runs
    ``1 + patience`` epochs and then stops.
    """
    trainer = CountingTrainer(synthetic_provider, early_stopping=True, patience=2)
    trainer.set_model(ConstantModel(synthetic_provider.num_classes))
    train_loader, eval_loader = _loaders(synthetic_provider)
    trainer.train(train_loader, eval_loader, epochs=8)
    assert trainer.epochs_run == 3


def test_the_patience_can_be_given_per_call(synthetic_provider):
    trainer = CountingTrainer(synthetic_provider, early_stopping=True, patience=5)
    trainer.set_model(ConstantModel(synthetic_provider.num_classes))
    train_loader, eval_loader = _loaders(synthetic_provider)
    trainer.train(train_loader, eval_loader, epochs=8, patience=1)
    assert trainer.epochs_run == 2


def test_early_stopping_keeps_the_best_validation_model(synthetic_provider):
    """The same contract as the regularised trainers: the best model survives."""
    trainer = BaseTrainer(provider=synthetic_provider, early_stopping=True, patience=1)
    trainer.set_model(synthetic_provider.make_model())
    train_loader, eval_loader = _loaders(synthetic_provider)
    trainer.train(train_loader, eval_loader, epochs=4)
    assert trainer.val_accuracies
    # The model left behind is the best-validation one, not the last one.
    assert trainer.evaluate(eval_loader) == pytest.approx(max(trainer.val_accuracies))


# --------------------------------------------------------------------------- #
# the registry and the provenance record
# --------------------------------------------------------------------------- #
def test_the_registry_accepts_the_flag(synthetic_provider):
    trainer = build_trainer(
        "BaseTrainer",
        {"early_stopping": True, "patience": 3},
        provider=synthetic_provider,
    )
    assert trainer.early_stopping is True
    assert trainer.patience == 3


def test_the_flag_is_recorded_in_the_provenance_block(synthetic_provider):
    from federated_outlier_adaptation.utils.provenance import trainer_hyperparameters

    trainer = BaseTrainer(provider=synthetic_provider, early_stopping=True, patience=3)
    recorded = trainer_hyperparameters(trainer)
    assert recorded["early_stopping"] is True
    assert recorded["patience"] == 3


def test_the_cli_parses_the_override():
    from federated_outlier_adaptation.cli import _parse_overrides

    assert _parse_overrides(["early_stopping=true"], None) == {"early_stopping": True}


def test_the_plans_emit_the_early_stopped_arm():
    from federated_outlier_adaptation.training import matrix

    tasks = matrix.plan_tasks("lean_nist_finals_pre")
    early = [task for task in tasks if "early_stopping=true" in task]
    assert early
    for task in early:
        assert "--trainer BaseTrainer" in task
    assert any("--parent lean_m10_uniform_fedavg_es" in task for task in tasks)
    assert any("--parent lean_m10_uniform_fedavg " in task + " " for task in tasks)
