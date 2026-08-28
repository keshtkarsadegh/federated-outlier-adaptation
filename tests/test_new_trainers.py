"""
The trainers added for the feature-alignment (E2) and layer-freezing (E5)
experiments, exercised through both federated runners.

Besides the smoke test, the freezing assertions check the three properties the
experiment depends on: frozen weights do not move, the optimizer never sees
them, and the normalisation statistics inside the frozen part stay put even
though the model is repeatedly put into training mode.
"""

from __future__ import annotations

import copy

import pytest
import torch

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.feature_alignment_trainer import (
    FeatureAlignmentTrainer,
)
from federated_outlier_adaptation.trainers.freeze_trainer import (
    DistillationFreezeTrainer,
    FreezeTrainer,
)

AGGREGATIONS = [
    ("concurrent", "weights", "con_weighted_cgw"),
    ("concurrent", "delta", "con_delta_weighted_cgd"),
    ("sequential", "weights", "seq_fedavg_update"),
    ("sequential", "delta", "seq_delta_fedavg_update"),
]

MAX_ROUND = 2
EPOCHS = 1
BATCH_SIZE = 4


def run_simulation(trainer, provider, scenario, metadata, agg_name):
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider)
    agg_method = getattr(select_class(scenario, metadata), agg_name)
    accuracies, _, _, _ = runner.simulate(
        exp_name=f"test_{agg_name}",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    return runner, accuracies


@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_feature_alignment_trainer_runs(synthetic_provider, scenario, metadata, agg_name):
    trainer = FeatureAlignmentTrainer(provider=synthetic_provider)
    _, accuracies = run_simulation(
        trainer, synthetic_provider, scenario, metadata, agg_name
    )
    assert len(accuracies) == MAX_ROUND
    for clients_acc, global_acc in accuracies:
        assert 0.0 <= clients_acc <= 1.0
        assert 0.0 <= global_acc <= 1.0


def test_feature_alignment_teacher_is_frozen(synthetic_provider):
    trainer = FeatureAlignmentTrainer(provider=synthetic_provider)
    assert not any(p.requires_grad for p in trainer.teacher_model.parameters())
    assert not trainer.teacher_model.training


def test_feature_alignment_zero_beta_is_plain_cross_entropy(synthetic_provider):
    trainer = FeatureAlignmentTrainer(provider=synthetic_provider, beta=0.0)
    images = torch.rand(4, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 0])
    logits, features = trainer.model(images, return_features=True)
    total, parts = trainer.custom_loss_fn(logits, features, labels, images)
    assert parts["align"] == 0.0
    assert torch.allclose(total, trainer.criterion(logits, labels))


@pytest.mark.parametrize("scope", ["conv", "body"])
@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_freeze_trainer_runs(synthetic_provider, scope, scenario, metadata, agg_name):
    trainer = FreezeTrainer(provider=synthetic_provider, scope=scope)
    _, accuracies = run_simulation(
        trainer, synthetic_provider, scenario, metadata, agg_name
    )
    assert len(accuracies) == MAX_ROUND


def test_freeze_trainer_rejects_unknown_scope(synthetic_provider):
    with pytest.raises(ValueError):
        FreezeTrainer(provider=synthetic_provider, scope="everything")


@pytest.mark.parametrize("scope", ["conv", "body"])
def test_frozen_parameters_do_not_change(synthetic_provider, scope):
    trainer = FreezeTrainer(provider=synthetic_provider, scope=scope)
    model = synthetic_provider.make_model()
    trainer.set_model(model)

    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    frozen = set(trainer.frozen_parameter_names())
    assert frozen, "the scope must freeze at least one parameter"

    train_loader, eval_loader, _ = synthetic_provider.build_dataset(
        ["c0"], train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    trainer.train(train_loader, eval_loader, epochs=2)

    after = dict(trainer.model.named_parameters())
    for name in frozen:
        assert torch.equal(before[name], after[name].detach()), name

    trainable = [n for n in before if n not in frozen]
    assert any(
        not torch.equal(before[n], after[n].detach()) for n in trainable
    ), "the trainable head must have moved"


def test_frozen_batchnorm_statistics_do_not_move(synthetic_provider):
    trainer = FreezeTrainer(provider=synthetic_provider, scope="body")
    trainer.set_model(synthetic_provider.make_model())

    buffers_before = {
        name: buffer.detach().clone()
        for name, buffer in trainer.model.named_buffers()
        if name.startswith("features.")
    }
    assert buffers_before, "the synthetic model must expose normalisation buffers"

    train_loader, eval_loader, _ = synthetic_provider.build_dataset(
        ["c1"], train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    trainer.train(train_loader, eval_loader, epochs=2)

    buffers_after = dict(trainer.model.named_buffers())
    for name, value in buffers_before.items():
        assert torch.equal(value, buffers_after[name].detach()), name


def test_freezing_survives_set_model(synthetic_provider):
    """The runners hand a fresh deep copy over every round."""
    trainer = FreezeTrainer(provider=synthetic_provider, scope="body")
    fresh = synthetic_provider.make_model()
    for parameter in fresh.parameters():
        parameter.requires_grad = True

    trainer.set_model(fresh)
    frozen = set(trainer.frozen_parameter_names())
    for name, parameter in trainer.model.named_parameters():
        assert parameter.requires_grad is (name not in frozen), name

    optimizer_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    for name, parameter in trainer.model.named_parameters():
        if name in frozen:
            assert id(parameter) not in optimizer_params, name

    for module in trainer.frozen_modules():
        assert not module.training


def test_distillation_freeze_trainer_reuses_the_kd_objective(synthetic_provider):
    from federated_outlier_adaptation.trainers.distillation_trainer import (
        DistillationTrainer,
    )

    trainer = DistillationFreezeTrainer(provider=synthetic_provider, scope="conv")
    reference = DistillationTrainer(provider=synthetic_provider)
    reference.model = copy.deepcopy(trainer.model)

    images = torch.rand(4, 1, 128, 128)
    labels = torch.tensor([0, 1, 2, 1])
    trainer.model.eval()
    reference.model.eval()
    with torch.no_grad():
        outputs = trainer.model(images)
        theirs = reference.distillation_loss(outputs, labels, images)
        ours = trainer.batch_loss(outputs, labels, images)
    assert torch.allclose(ours, theirs)


def test_distillation_freeze_trainer_runs(synthetic_provider):
    trainer = DistillationFreezeTrainer(provider=synthetic_provider, scope="body")
    _, accuracies = run_simulation(
        trainer, synthetic_provider, "concurrent", "weights", "con_weighted_cgw"
    )
    assert len(accuracies) == MAX_ROUND
