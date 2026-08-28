"""
federated_pretraining.py - obtain the global model by FedAvg instead of central training.

The published pipeline trains the global model centrally on the pooled data of
the server-side clients (``provider.global_client_ids()``, 108 NIST writers).
This module produces an alternative starting point with the same data but a
federated procedure: each server-side client is a FedAvg client, a fraction of
them is sampled per round, they train locally for a few epochs and the server
averages the resulting models proportionally to their sample counts.

Defaults follow the standard FedAvg setting:

    participation C = 0.1, local epochs E = 1, rounds T = 100,
    proportional weighting p_k = n_k / N, server step size 1.0.

Everything is written next to - never over - the centrally trained artefacts:

    <results>/global_fl_model                        state dict
    <results>/global_results/global_fl_metrics.json  per-round curves
    <results>/global_results/fisher_fl/              Fisher information

Downstream phases start from it with ``--global-name global_fl``; the trainers
resolve their teacher checkpoint and Fisher directory through
:mod:`federated_outlier_adaptation.trainers.artefacts`, which follows the
same naming convention.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

from federated_outlier_adaptation import config
from federated_outlier_adaptation.aggregation.concurrent_methods import con_weighted_cw
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.providers import default_provider
from federated_outlier_adaptation.runners.client_sampler import ClientSampler
from federated_outlier_adaptation.trainers import artefacts
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.training.fisher import compute_and_save_fisher_and_params
from federated_outlier_adaptation.utils import provenance
from federated_outlier_adaptation.utils.seeding import set_run_seed

#: Name of the federated pre-trained model; see ``trainers.artefacts``.
GLOBAL_FL_NAME = "global_fl"

DEFAULT_ROUNDS = 100
DEFAULT_PARTICIPATION = 0.1
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 64
#: Split seed of the published pipeline; reused so the test set is identical.
SPLIT_SEED = 42


class FederatedPretraining:
    """
    FedAvg over the server-side clients, producing an alternative global model.

    Args:
        provider: Dataset provider; defaults to the configured one.
        results_dir: Results root; defaults to the provider's.
        global_name: Output name of the model, ``"global_fl"`` by default.
        batch_size: Batch size of local training and of every evaluation.
        participation: Fraction of clients sampled per round.
        local_epochs: Local epochs each sampled client runs per round.
        seed: Seed of the sampler and of the run RNGs.
    """

    def __init__(
        self,
        provider=None,
        results_dir: Optional[Path] = None,
        global_name: str = GLOBAL_FL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
        participation: float = DEFAULT_PARTICIPATION,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        seed: Optional[int] = SPLIT_SEED,
    ):
        self.provider = provider or default_provider()
        self.results_dir = Path(results_dir) if results_dir else Path(
            getattr(self.provider, "results_dir", config.RESULTS_DIR)
        )
        self.global_name = global_name
        self.batch_size = batch_size
        self.participation = participation
        self.local_epochs = local_epochs
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = BaseTrainer(provider=self.provider)
        self.global_model = self.provider.make_model().to(self.device)

        self.clients = list(self.provider.global_client_ids())
        self.model_path = self.results_dir / f"{self.global_name}_model"
        self.results_path = self.results_dir / "global_results"
        self.metrics_path = self.results_path / f"{self.global_name}_metrics.json"
        self.fisher_path = artefacts.fisher_dir(self.provider, self.global_name)

        self.round_accuracies: list[float] = []
        self.round_participants: list[list[str]] = []
        self.round_seconds: list[float] = []

    # ------------------------------------------------------------------ paths
    def path_init(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.fisher_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- data
    def central_loaders(self):
        """
        The loaders the centrally trained global model used.

        ``train_rate=0.6`` / ``eval_rate=0.4`` over the pooled server-side
        clients for training and Fisher information, and the full pool as the
        test set - exactly what ``run_global_training`` builds, so the two
        models are evaluated on the same data.
        """
        train_loader, eval_loader, _ = self.provider.build_dataset(
            self.clients,
            train_rate=0.6,
            eval_rate=0.4,
            batch_size=self.batch_size,
            seed=SPLIT_SEED,
        )
        _, _, test_loader = self.provider.build_dataset(
            self.clients,
            train_rate=0.0,
            eval_rate=0.0,
            batch_size=self.batch_size,
            seed=SPLIT_SEED,
        )
        return train_loader, eval_loader, test_loader

    def client_loaders(self, client_id):
        """Local train/validation loaders of one client."""
        train_loader, eval_loader, _ = self.provider.build_dataset(
            client_id,
            train_rate=0.6,
            eval_rate=0.4,
            batch_size=self.batch_size,
            seed=SPLIT_SEED,
        )
        return train_loader, eval_loader

    # ------------------------------------------------------------------ rounds
    def run_round(self, participants) -> None:
        """Train the sampled clients locally and average their models."""
        client_weights, sample_counts = [], []
        for client_id in participants:
            train_loader, eval_loader = self.client_loaders(client_id)
            if train_loader is None or eval_loader is None:  # pragma: no cover
                continue
            self.trainer.set_model(self.global_model)
            self.trainer.train(train_loader, eval_loader, self.local_epochs)
            trained = self.trainer.get_model()
            client_weights.append(
                copy.deepcopy(
                    {
                        key: value
                        for key, value in trained.state_dict().items()
                        if value.dtype.is_floating_point
                    }
                )
            )
            sample_counts.append(self.provider.sample_count(client_id))

        if not client_weights:  # pragma: no cover - empty round
            return

        global_weights = {
            key: value
            for key, value in self.global_model.state_dict().items()
            if value.dtype.is_floating_point
        }
        # Proportional weighting with a server step size of 1.0 is plain FedAvg,
        # which is exactly what con_weighted_cw computes.
        aggregated = con_weighted_cw(global_weights, client_weights, sample_counts)
        self.global_model.load_state_dict(aggregated, strict=False)

    def evaluate(self, test_loader) -> float:
        self.trainer.set_model(self.global_model)
        return float(self.trainer.evaluate(test_loader))

    # -------------------------------------------------------------------- run
    def run(self, rounds: int = DEFAULT_ROUNDS) -> dict[str, Any]:
        """
        Run the federated pre-training and store model, metrics and Fisher.

        Returns:
            dict: The metrics block that was written to disk.
        """
        self.path_init()
        set_run_seed(self.seed)
        timer = provenance.RunTimer()

        train_loader, _, test_loader = self.central_loaders()
        sampler = ClientSampler(
            self.clients,
            participation=self.participation,
            policy="uniform" if self.participation < 1.0 else "all",
            seed=self.seed,
        )

        NistLogger.info(
            f"Federated pre-training: {len(self.clients)} clients, "
            f"C={self.participation}, E={self.local_epochs}, T={rounds}"
        )

        for round_index in range(rounds):
            round_t0 = time.perf_counter()
            participants = sampler.select(round_index)
            self.run_round(participants)
            accuracy = self.evaluate(test_loader)

            self.round_participants.append(list(participants))
            self.round_accuracies.append(accuracy)
            self.round_seconds.append(time.perf_counter() - round_t0)
            NistLogger.info(
                f"Round {round_index + 1}/{rounds} | "
                f"{len(participants)} clients | test acc {accuracy:.4f}"
            )

        torch.save(self.global_model.state_dict(), self.model_path)
        NistLogger.info(f"Saved federated global model to {self.model_path}")

        compute_and_save_fisher_and_params(
            model=self.global_model,
            dataloader=train_loader,
            criterion=nn.CrossEntropyLoss(),
            device=self.device,
            fisher_path=self.fisher_path,
        )
        timer.stop()

        metrics = {
            "test_accuracy": self.round_accuracies[-1] if self.round_accuracies else None,
            "round_accuracies": self.round_accuracies,
            "participants": self.round_participants,
            "round_seconds": self.round_seconds,
            "rounds": rounds,
            "participation": self.participation,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "num_clients": len(self.clients),
            "model_path": str(self.model_path),
            "fisher_dir": str(self.fisher_path),
            "config": provenance.build_run_config(
                trainer_name="BaseTrainer",
                trainer=self.trainer,
                scenario="concurrent",
                metadata="weights",
                agg_method_name="con_weighted_cw",
                batch_size=self.batch_size,
                epochs=self.local_epochs,
                max_round=rounds,
                seed=self.seed,
                selected_writers=self.clients,
                global_model_path=self.model_path,
                fisher_dir=self.fisher_path,
                parent_name=self.global_name,
                started_at=timer.started_at,
                finished_at=timer.finished_at,
                extra={
                    "wall_seconds": timer.seconds,
                    "participation": self.participation,
                    "policy": sampler.policy,
                    "sampler_seed": self.seed,
                    "global_name": self.global_name,
                },
            ),
        }
        with open(self.metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        NistLogger.info(f"Saved federated pre-training metrics to {self.metrics_path}")
        return metrics


def run_federated_pretraining(
    rounds: int = DEFAULT_ROUNDS,
    participation: float = DEFAULT_PARTICIPATION,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: Optional[int] = SPLIT_SEED,
    global_name: str = GLOBAL_FL_NAME,
    provider=None,
    force: bool = False,
) -> Mapping[str, Any]:
    """
    Entry point of ``foa global-train-fl``.

    Args:
        force: Re-run even when ``<results>/<global_name>_model`` exists.
    """
    pretraining = FederatedPretraining(
        provider=provider,
        global_name=global_name,
        batch_size=batch_size,
        participation=participation,
        local_epochs=local_epochs,
        seed=seed,
    )
    if pretraining.model_path.exists() and not force:
        NistLogger.info(
            f"{pretraining.model_path} already exists; pass --force to re-run."
        )
        if pretraining.metrics_path.exists():
            with open(pretraining.metrics_path) as handle:
                return json.load(handle)
        return {"model_path": str(pretraining.model_path)}
    return pretraining.run(rounds=rounds)


if __name__ == "__main__":  # pragma: no cover
    run_federated_pretraining()
