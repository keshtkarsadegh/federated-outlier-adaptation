"""
The unified regularisation family.

The point of :class:`AnchoredTrainer` is that the published objectives are
special cases of one formula, so most of these tests are equivalences: the
family trainer, configured as the table in its module docstring says, must
compute the same loss as the trainer it replaces.
"""

from __future__ import annotations

import copy

import pytest
import torch

from federated_outlier_adaptation.trainers.anchored_trainer import (
    ANCHOR_SPACES,
    AnchoredTrainer,
)
from federated_outlier_adaptation.trainers.cf_logit_consistency_trainer import (
    CFLogitConsistencyTrainer,
)
from federated_outlier_adaptation.trainers.cf_prox_trainer import CFProxTrainer
from federated_outlier_adaptation.trainers.distillation_trainer import DistillationTrainer
from federated_outlier_adaptation.trainers.ewc_trainer import EWCTrainer
from federated_outlier_adaptation.trainers.feature_alignment_trainer import (
    FeatureAlignmentTrainer,
    logits_and_features,
)
from federated_outlier_adaptation.trainers.ntd_trainer import NTDTrainer

BATCH = 4


@pytest.fixture()
def batch():
    torch.manual_seed(17)
    images = torch.rand(BATCH, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 1])
    return images, labels


def _pair(provider, anchored, published):
    """Give both trainers the same weights, so only the objective differs."""
    torch.manual_seed(5)
    model = provider.make_model()
    anchored.set_model(copy.deepcopy(model))
    published.set_model(copy.deepcopy(model))
    anchored.model.eval()
    published.model.eval()
    return anchored, published


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("space", ANCHOR_SPACES)
def test_every_space_constructs(synthetic_provider, space):
    trainer = AnchoredTrainer(provider=synthetic_provider, space=space, lam=0.1)
    assert trainer.space == space


def test_unknown_space_and_anchor_are_rejected(synthetic_provider):
    with pytest.raises(ValueError):
        AnchoredTrainer(provider=synthetic_provider, space="cosine")
    with pytest.raises(ValueError):
        AnchoredTrainer(provider=synthetic_provider, anchor="previous")


def test_zero_lambda_is_plain_cross_entropy(synthetic_provider, batch):
    images, labels = batch
    trainer = AnchoredTrainer(provider=synthetic_provider, space="kd", lam=0.0)
    trainer.model.eval()
    logits = trainer.model(images)
    total, parts = trainer.custom_loss_fn(logits, None, labels, images)
    assert parts["penalty"] == 0.0
    assert torch.allclose(total, trainer.criterion(logits, labels))


# --------------------------------------------------------------------------- #
# equivalences with the published trainers
# --------------------------------------------------------------------------- #
def test_param_l2_reproduces_the_prox_objective(synthetic_provider, batch):
    images, labels = batch
    lam = 0.7
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(provider=synthetic_provider, space="param_l2", lam=lam),
        CFProxTrainer(provider=synthetic_provider, lambda_prox=lam),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)
    theirs = published.criterion(logits, labels) + lam * published._prox_loss()
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_logit_l2_reproduces_the_logit_consistency_objective(synthetic_provider, batch):
    images, labels = batch
    lam = 0.3
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(provider=synthetic_provider, space="logit_l2", lam=lam),
        CFLogitConsistencyTrainer(provider=synthetic_provider, lambda_consis=lam),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)
    theirs, _ = published.custom_loss_fn(logits, labels, images)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_feature_l2_reproduces_the_feature_alignment_objective(synthetic_provider, batch):
    images, labels = batch
    lam = 0.4
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(provider=synthetic_provider, space="feature_l2", lam=lam),
        FeatureAlignmentTrainer(provider=synthetic_provider, beta=lam),
    )
    logits, features = logits_and_features(anchored.model, images)
    ours, _ = anchored.custom_loss_fn(logits, features, labels, images)
    theirs, _ = published.custom_loss_fn(logits, features, labels, images)
    assert torch.allclose(ours, theirs, atol=1e-6)


def test_ntd_space_reproduces_the_ntd_objective(synthetic_provider, batch):
    images, labels = batch
    lam, temperature = 0.6, 2.0
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(provider=synthetic_provider, space="ntd", lam=lam, T=temperature),
        NTDTrainer(provider=synthetic_provider, beta=lam, tau=temperature),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)
    theirs, _ = published.custom_loss_fn(logits, labels, images)
    assert torch.allclose(ours, theirs, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.5, 0.95])
def test_kd_space_matches_distillation_up_to_the_alpha_convention(
    synthetic_provider, batch, alpha
):
    """``alpha * (CE + lam * T^2 KL)`` with ``lam = (1-alpha)/alpha``."""
    images, labels = batch
    temperature = 4.0
    lam = (1.0 - alpha) / alpha
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(provider=synthetic_provider, space="kd", lam=lam, T=temperature),
        DistillationTrainer(provider=synthetic_provider, T=temperature, alpha=alpha),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)
    theirs = published.distillation_loss(logits, labels, images)
    assert torch.allclose(alpha * ours, theirs, atol=1e-6)


def test_fisher_space_reproduces_ewc_without_the_dynamic_scaling(synthetic_provider, batch):
    images, labels = batch
    lam = 3.0
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(
            provider=synthetic_provider, space="fisher", lam=lam, normalise_fisher=False
        ),
        EWCTrainer(provider=synthetic_provider, ewc_lambda=lam),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)

    penalty = 0.0
    for name, parameter in published.model.named_parameters():
        penalty = penalty + (
            published.fisher[name] * (parameter - published.global_params[name]) ** 2
        ).sum()
    theirs = published.criterion(logits, labels) + (lam / 2.0) * penalty
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_fisher_scaled_reproduces_the_ewc_lambda_cap(synthetic_provider, batch):
    images, labels = batch
    lam = 8.0
    anchored, published = _pair(
        synthetic_provider,
        AnchoredTrainer(
            provider=synthetic_provider, space="fisher_scaled", lam=lam, normalise_fisher=False
        ),
        EWCTrainer(provider=synthetic_provider, ewc_lambda=lam),
    )
    logits = anchored.model(images)
    ours, _ = anchored.custom_loss_fn(logits, None, labels, images)
    theirs = published.ewc_loss(logits, labels)
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_normalised_fisher_has_mean_one(synthetic_provider):
    trainer = AnchoredTrainer(provider=synthetic_provider, space="fisher", lam=1.0)
    total = sum(float(v.sum()) for v in trainer.fisher.values())
    count = sum(v.numel() for v in trainer.fisher.values())
    assert total / count == pytest.approx(1.0, rel=1e-5)


# --------------------------------------------------------------------------- #
# anchors
# --------------------------------------------------------------------------- #
def test_frozen_anchor_ignores_set_model(synthetic_provider):
    trainer = AnchoredTrainer(provider=synthetic_provider, space="param_l2", anchor="frozen")
    before = {k: v.clone() for k, v in trainer.anchor_state.items()}

    torch.manual_seed(99)
    other = synthetic_provider.make_model()
    trainer.set_model(other)

    for key, value in before.items():
        assert torch.equal(value, trainer.anchor_state[key]), key


def test_current_anchor_follows_the_received_model(synthetic_provider):
    trainer = AnchoredTrainer(provider=synthetic_provider, space="param_l2", anchor="current")
    torch.manual_seed(123)
    other = synthetic_provider.make_model()
    trainer.set_model(other)

    for key, value in other.state_dict().items():
        assert torch.allclose(value, trainer.anchor_state[key].cpu()), key


def test_current_anchor_makes_the_penalty_vanish_at_the_start(synthetic_provider, batch):
    """A client that has not moved yet is exactly at its own anchor."""
    images, labels = batch
    trainer = AnchoredTrainer(
        provider=synthetic_provider, space="param_l2", anchor="current", lam=5.0
    )
    torch.manual_seed(7)
    trainer.set_model(synthetic_provider.make_model())
    trainer.model.eval()
    logits = trainer.model(images)
    _, parts = trainer.custom_loss_fn(logits, None, labels, images)
    assert parts["penalty"] == pytest.approx(0.0, abs=1e-9)


def test_current_anchor_moves_the_teacher_for_output_spaces(synthetic_provider, batch):
    images, _ = batch
    trainer = AnchoredTrainer(
        provider=synthetic_provider, space="logit_l2", anchor="current", lam=1.0
    )
    torch.manual_seed(31)
    other = synthetic_provider.make_model().eval()
    trainer.set_model(other)
    trainer.teacher_model.eval()
    with torch.no_grad():
        assert torch.allclose(trainer.teacher_model(images), other(images), atol=1e-6)


# --------------------------------------------------------------------------- #
# hybrid and end to end
# --------------------------------------------------------------------------- #
def test_hybrid_splits_the_budget(synthetic_provider, batch):
    images, labels = batch
    lam = 2.0
    hybrid = AnchoredTrainer(
        provider=synthetic_provider, space="kd+fisher", lam=lam, mix=0.25, T=2.0
    )
    kd_only = AnchoredTrainer(provider=synthetic_provider, space="kd", lam=lam, T=2.0)
    fisher_only = AnchoredTrainer(provider=synthetic_provider, space="fisher", lam=lam)
    for trainer in (hybrid, kd_only, fisher_only):
        torch.manual_seed(3)
        trainer.set_model(synthetic_provider.make_model())
        trainer.model.eval()

    logits = hybrid.model(images)
    combined = hybrid.penalty(logits, None, labels, images)
    kd_term = kd_only.penalty(logits, None, labels, images)
    fisher_term = fisher_only.penalty(logits, None, labels, images)
    assert torch.allclose(combined, 0.25 * kd_term + 0.75 * fisher_term, atol=1e-5)


@pytest.mark.parametrize("space", ["param_l2", "kd", "fisher", "feature_l2", "ntd"])
def test_anchored_trainer_runs_through_a_runner(synthetic_provider, space):
    from federated_outlier_adaptation.aggregation.selector import select_class
    from federated_outlier_adaptation.runners.concurrent_runner import (
        BaseConcurrentRunner,
    )

    trainer = AnchoredTrainer(provider=synthetic_provider, space=space, lam=0.1)
    runner = BaseConcurrentRunner(trainer=trainer, provider=synthetic_provider)
    agg = getattr(select_class("concurrent", "weights"), "con_weighted_cgw")
    accuracies, *_ = runner.simulate(
        exp_name="test_anchored",
        global_name="global",
        aggregate_method=agg,
        batch_size=4,
        epochs=1,
        max_round=2,
        grid_Search=False,
    )
    assert len(accuracies) == 2


def test_registry_resolves_the_trainer():
    from federated_outlier_adaptation.trainers.registry import get_trainer_class

    assert get_trainer_class("AnchoredTrainer").__name__ == "AnchoredTrainer"
