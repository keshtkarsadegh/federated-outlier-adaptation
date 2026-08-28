"""
Not-true distillation: the masking of the true class, the degenerate cases and
the trainer running through both runners.
"""

from __future__ import annotations

import math

import pytest
import torch

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.ntd_trainer import (
    NTDTrainer,
    not_true_log_softmax,
)

AGGREGATIONS = [
    ("concurrent", "weights", "con_weighted_cgw"),
    ("sequential", "weights", "seq_fedavg_update"),
]
MAX_ROUND = 2


def test_true_class_receives_no_probability():
    logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, -1.0]])
    labels = torch.tensor([0, 1])
    log_p = not_true_log_softmax(logits, labels, tau=1.0)

    assert math.isinf(log_p[0, 0].item()) and log_p[0, 0].item() < 0
    assert math.isinf(log_p[1, 1].item()) and log_p[1, 1].item() < 0

    probabilities = log_p.exp()
    assert probabilities[0, 0].item() == 0.0
    assert probabilities[1, 1].item() == 0.0
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)


def test_temperature_flattens_the_not_true_distribution():
    logits = torch.tensor([[0.0, 4.0, 1.0]])
    labels = torch.tensor([0])
    sharp = not_true_log_softmax(logits, labels, tau=1.0).exp()
    smooth = not_true_log_softmax(logits, labels, tau=5.0).exp()
    assert smooth[0, 1] < sharp[0, 1]
    assert smooth[0, 2] > sharp[0, 2]


def test_identical_logits_give_zero_divergence(synthetic_provider):
    trainer = NTDTrainer(provider=synthetic_provider, beta=1.0, tau=2.0)
    images = torch.rand(4, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 0])
    trainer.teacher_model.eval()
    with torch.no_grad():
        teacher_logits = trainer.teacher_model(images)
    loss = trainer.not_true_loss(teacher_logits, labels, images)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_divergence_is_positive_for_different_logits(synthetic_provider):
    trainer = NTDTrainer(provider=synthetic_provider, beta=1.0, tau=1.0)
    images = torch.rand(4, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 0])
    torch.manual_seed(3)
    student_logits = torch.randn(4, 3, requires_grad=True)
    loss = trainer.not_true_loss(student_logits, labels, images)
    assert loss.item() > 0
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_zero_beta_is_plain_cross_entropy(synthetic_provider):
    trainer = NTDTrainer(provider=synthetic_provider, beta=0.0)
    images = torch.rand(4, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 1])
    logits = trainer.model(images)
    total, parts = trainer.custom_loss_fn(logits, labels, images)
    assert parts["ntd"] == 0.0
    assert torch.allclose(total, trainer.criterion(logits, labels))


def test_teacher_is_frozen(synthetic_provider):
    trainer = NTDTrainer(provider=synthetic_provider)
    assert not any(p.requires_grad for p in trainer.teacher_model.parameters())
    assert not trainer.teacher_model.training


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_ntd_trainer_runs_through_the_runners(
    synthetic_provider, scenario, metadata, agg_name
):
    trainer = NTDTrainer(provider=synthetic_provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=synthetic_provider)
    agg_method = getattr(select_class(scenario, metadata), agg_name)
    accuracies, *_ = runner.simulate(
        exp_name="test_ntd",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=4,
        epochs=1,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    assert len(accuracies) == MAX_ROUND
    for clients_acc, global_acc in accuracies:
        assert 0.0 <= clients_acc <= 1.0
        assert 0.0 <= global_acc <= 1.0


def test_registry_resolves_the_trainer():
    from federated_outlier_adaptation.trainers.registry import get_trainer_class

    assert get_trainer_class("NTDTrainer").__name__ == "NTDTrainer"
