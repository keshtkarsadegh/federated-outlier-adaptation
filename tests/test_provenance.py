"""
Provenance and instrumentation helpers.

Every result file carries a ``config`` block describing how it was produced;
these tests pin down its shape and the hyperparameter capture.
"""

from __future__ import annotations

import hashlib

import torch

from federated_outlier_adaptation.utils import instrumentation, provenance

REQUIRED_KEYS = {
    "trainer",
    "trainer_kwargs",
    "hyperparameters",
    "scenario",
    "metadata",
    "agg_method_name",
    "batch_size",
    "epochs",
    "max_round",
    "seed",
    "outliers_file",
    "outlier_writers",
    "global_model_path",
    "global_model_sha256",
    "fisher_dir",
    "torch_version",
    "device",
    "python_version",
    "git_commit",
    "hostname",
    "started_at",
    "finished_at",
}


def test_run_config_has_required_keys(synthetic_provider):
    from federated_outlier_adaptation.trainers.distillation_trainer import DistillationTrainer

    trainer = DistillationTrainer(provider=synthetic_provider, T=4, alpha=0.5)
    block = provenance.build_run_config(
        trainer_name="DistillationTrainer",
        trainer=trainer,
        trainer_kwargs={"T": 4, "alpha": 0.5},
        scenario="concurrent",
        metadata="weights",
        agg_method_name="con_weighted_cgw",
        batch_size=64,
        epochs=1,
        max_round=2,
        seed=7,
        outliers_file=synthetic_provider.outliers_file,
        global_model_path=synthetic_provider.global_model_path,
        fisher_dir=synthetic_provider.fisher_dir,
        started_at=provenance.utc_now(),
    )

    assert REQUIRED_KEYS.issubset(block)
    assert block["hyperparameters"]["T"] == 4
    assert block["hyperparameters"]["alpha"] == 0.5
    assert block["hyperparameters"]["learning_rate"] == 1e-3
    assert block["outlier_writers"] == synthetic_provider.selected_clients()
    assert len(block["global_model_sha256"]) == 64
    assert block["seed"] == 7


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"federated")
    assert provenance.sha256_of(path) == hashlib.sha256(b"federated").hexdigest()
    assert provenance.sha256_of(tmp_path / "missing") is None


def test_communication_volume_counts_both_directions():
    state = {"w": torch.zeros(10, dtype=torch.float32)}
    payload = instrumentation.state_dict_bytes(state)
    assert payload == 40
    assert instrumentation.state_dict_params(state) == 10
    assert instrumentation.round_communication_bytes(payload, 3) == 240


def test_timing_summary():
    summary = instrumentation.summarise_timing([1.0, 3.0], [[0.5], [1.5]])
    assert summary["total_seconds"] == 4.0
    assert summary["mean_round_seconds"] == 2.0
    assert summary["client_seconds"] == [[0.5], [1.5]]


def test_device_name():
    assert instrumentation.device_name("cpu") == "cpu"
    assert instrumentation.device_name("cuda", "NVIDIA A100") == "cuda (NVIDIA A100)"
