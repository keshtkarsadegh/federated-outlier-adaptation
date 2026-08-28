"""
Federated pre-training of the global model, and the ``--global-name`` plumbing
that lets every downstream phase start from it.
"""

from __future__ import annotations

import json

import pytest
import torch

from federated_outlier_adaptation.trainers import artefacts
from federated_outlier_adaptation.trainers.registry import build_trainer
from federated_outlier_adaptation.training.federated_pretraining import (
    GLOBAL_FL_NAME,
    FederatedPretraining,
)


# --------------------------------------------------------------------------- #
# artefact naming
# --------------------------------------------------------------------------- #
def test_default_global_name_uses_the_provider_paths(synthetic_provider):
    assert artefacts.is_default("global")
    assert artefacts.is_default(None)
    assert artefacts.global_model_path(synthetic_provider, "global") == (
        synthetic_provider.global_model_path
    )
    assert artefacts.fisher_dir(synthetic_provider, "global") == synthetic_provider.fisher_dir


def test_named_global_model_gets_its_own_paths(synthetic_provider):
    root = synthetic_provider.results_dir
    assert artefacts.global_model_path(synthetic_provider, "global_fl") == (
        root / "global_fl_model"
    )
    assert artefacts.fisher_dir(synthetic_provider, "global_fl") == (
        root / "global_results" / "fisher_fl"
    )


@pytest.mark.parametrize(
    "name,tag", [("global_fl", "fl"), ("global_warm", "warm"), ("other", "other")]
)
def test_fisher_tag(name, tag):
    assert artefacts.fisher_tag(name) == tag


# --------------------------------------------------------------------------- #
# registry injection
# --------------------------------------------------------------------------- #
def test_registry_leaves_the_default_run_untouched(synthetic_provider):
    trainer = build_trainer("DistillationTrainer", None, provider=synthetic_provider)
    assert str(trainer.global_model_path) == str(synthetic_provider.global_model_path)


def test_registry_points_the_teacher_at_the_named_model(synthetic_provider, tmp_path):
    # A second checkpoint under the alternative name.
    target = artefacts.global_model_path(synthetic_provider, "global_fl")
    torch.save(synthetic_provider.make_model().state_dict(), target)

    trainer = build_trainer(
        "DistillationTrainer", None, provider=synthetic_provider, global_name="global_fl"
    )
    assert str(trainer.global_model_path) == str(target)


def test_registry_points_ewc_at_the_named_fisher(synthetic_provider):
    fisher_dir = artefacts.fisher_dir(synthetic_provider, "global_fl")
    fisher_dir.mkdir(parents=True, exist_ok=True)
    model = synthetic_provider.make_model()
    torch.save(
        {n: torch.ones_like(p) for n, p in model.named_parameters()},
        fisher_dir / "fisher.pt",
    )
    torch.save(
        {n: p.detach().clone() for n, p in model.named_parameters()},
        fisher_dir / "global_params.pt",
    )

    trainer = build_trainer(
        "EWCTrainer", None, provider=synthetic_provider, global_name="global_fl"
    )
    assert str(trainer.fisher_path) == str(fisher_dir)


def test_explicit_kwargs_win_over_the_injection(synthetic_provider):
    explicit = str(synthetic_provider.global_model_path)
    trainer = build_trainer(
        "DistillationTrainer",
        {"global_model_path": explicit},
        provider=synthetic_provider,
        global_name="global_fl",
    )
    assert str(trainer.global_model_path) == explicit


# --------------------------------------------------------------------------- #
# the federated pre-training loop
# --------------------------------------------------------------------------- #
def test_federated_pretraining_writes_its_own_artefacts(synthetic_provider):
    pretraining = FederatedPretraining(
        provider=synthetic_provider,
        batch_size=4,
        participation=1.0,
        local_epochs=1,
        seed=7,
    )
    baseline = synthetic_provider.global_model_path.read_bytes()

    metrics = pretraining.run(rounds=2)

    assert pretraining.model_path.is_file()
    assert pretraining.model_path != synthetic_provider.global_model_path
    assert pretraining.model_path.name == f"{GLOBAL_FL_NAME}_model"
    # The centrally trained checkpoint is untouched.
    assert synthetic_provider.global_model_path.read_bytes() == baseline

    assert (pretraining.fisher_path / "fisher.pt").is_file()
    assert (pretraining.fisher_path / "global_params.pt").is_file()
    assert pretraining.fisher_path != synthetic_provider.fisher_dir

    assert len(metrics["round_accuracies"]) == 2
    assert all(0.0 <= value <= 1.0 for value in metrics["round_accuracies"])
    assert metrics["participants"] == [["g0"], ["g0"]]
    assert metrics["config"]["agg_method_name"] == "con_weighted_cw"
    assert metrics["config"]["global_name"] == GLOBAL_FL_NAME

    with open(pretraining.metrics_path) as handle:
        stored = json.load(handle)
    assert stored["rounds"] == 2


def test_pretrained_model_can_be_loaded_by_the_provider_topology(synthetic_provider):
    pretraining = FederatedPretraining(
        provider=synthetic_provider, batch_size=4, participation=1.0, local_epochs=1, seed=1
    )
    pretraining.run(rounds=1)

    model = synthetic_provider.make_model()
    model.load_state_dict(torch.load(pretraining.model_path, map_location="cpu"))


def test_partial_participation_is_seeded(synthetic_provider):
    """Two runs with the same seed must sample the same clients."""
    from federated_outlier_adaptation.runners.client_sampler import ClientSampler

    pool = [f"g{i}" for i in range(10)]
    first = ClientSampler(pool, participation=0.1, policy="uniform", seed=42)
    second = ClientSampler(pool, participation=0.1, policy="uniform", seed=42)
    assert [first.select(r) for r in range(5)] == [second.select(r) for r in range(5)]
