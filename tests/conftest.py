"""
Shared fixtures.

The suite never touches the real NIST dataset or the published results.  It runs
against a tiny synthetic provider: random 128x128 grayscale images, three
clients, a three-class model and freshly generated "frozen" artefacts (global
model, Fisher information, baseline metrics) written into a temporary results
root.  That exercises exactly the code paths the real experiments use while
staying inside a few seconds of CPU time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

from federated_outlier_adaptation.providers.base import DatasetProvider  # noqa: E402

IMAGE_SIZE = 128
NUM_CLASSES = 3
SAMPLES_PER_CLIENT = 12


class TinyCNN(nn.Module):
    """Small convolutional model with the same interface as the NIST model."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 4 * 4, 16)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(16, num_classes)

    def penultimate(self, x):
        """Post-``fc1`` ReLU activation, mirroring the NIST model's accessor."""
        x = self.features(x)
        x = self.flatten(x)
        return torch.relu(self.fc1(x))

    def forward(self, x, return_features=False):
        features = self.penultimate(x)
        logits = self.fc2(self.dropout(features))
        if return_features:
            return logits, features
        return logits


class FeaturesOnlyCNN(TinyCNN):
    """Model that implements ``penultimate`` but not ``return_features``."""

    def forward(self, x):
        return self.fc2(self.dropout(self.penultimate(x)))


class SyntheticProvider(DatasetProvider):
    """In-memory provider with a fixed, reproducible synthetic dataset."""

    name = "synthetic"

    def __init__(self, results_dir: Path, num_clients: int = 3, seed: int = 0):
        self.results_dir = Path(results_dir)
        self.num_clients = num_clients
        self._clients = [f"c{i}" for i in range(num_clients)]
        self._global_clients = ["g0"]

        generator = torch.Generator().manual_seed(seed)
        self._data: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for client in self._clients + self._global_clients:
            images = torch.rand(
                SAMPLES_PER_CLIENT, 1, IMAGE_SIZE, IMAGE_SIZE, generator=generator
            )
            labels = torch.arange(SAMPLES_PER_CLIENT) % NUM_CLASSES
            self._data[client] = (images, labels)

    # ----------------------------------------------------------------- clients
    def all_client_ids(self) -> List[str]:
        return list(self._clients) + list(self._global_clients)

    def local_client_ids(self) -> List[str]:
        return list(self._clients)

    def global_client_ids(self) -> List[str]:
        return list(self._global_clients)

    def selected_clients(self, k: int = 5) -> List[str]:
        return list(self._clients[:k])

    # -------------------------------------------------------------------- data
    def build_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ):
        if isinstance(client_ids, str):
            client_ids = [client_ids]
        images = torch.cat([self._data[c][0] for c in client_ids])
        labels = torch.cat([self._data[c][1] for c in client_ids])

        n = len(labels)
        n_train = int(n * train_rate)
        n_val = int(n * eval_rate)

        def loader(start, stop):
            if stop <= start:
                return None
            dataset = TensorDataset(images[start:stop], labels[start:stop])
            if loader_seed is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            generator = torch.Generator().manual_seed(int(loader_seed))
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        return loader(0, n_train), loader(n_train, n_train + n_val), loader(n_train + n_val, n)

    def sample_count(self, client_id: str) -> int:
        return len(self._data[client_id][1])

    # ------------------------------------------------------------------- model
    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    def make_model(self):
        return TinyCNN(self.num_classes)

    # --------------------------------------------------------------- artefacts
    @property
    def global_model_path(self) -> Path:
        return self.results_dir / "global_model"

    @property
    def fisher_dir(self) -> Path:
        return self.results_dir / "global_results" / "fisher"

    @property
    def outliers_file(self) -> Path:
        return self.results_dir / "outliers" / "selected_outliers.json"

    # ------------------------------------------------------------------- setup
    def write_artefacts(self) -> None:
        """Create the frozen artefacts the runners and trainers read."""
        (self.results_dir / "global_results").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "global_clients_results").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "outliers").mkdir(parents=True, exist_ok=True)
        self.fisher_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(0)
        model = self.make_model()
        torch.save(model.state_dict(), self.global_model_path)

        fisher = {
            name: torch.ones_like(param) * 1e-3
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        params = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(fisher, self.fisher_dir / "fisher.pt")
        torch.save(params, self.fisher_dir / "global_params.pt")

        with open(self.results_dir / "writer_split.json", "w") as handle:
            json.dump(
                {"local_writers": self._clients, "global_writers": self._global_clients},
                handle,
            )
        with open(self.outliers_file, "w") as handle:
            json.dump(self._clients, handle)

        with open(self.results_dir / "global_results" / "global_metrics.json", "w") as handle:
            json.dump({"test_accuracy": 0.5}, handle)
        with open(
            self.results_dir / "global_clients_results" / "global_clients_global_metrics.json", "w"
        ) as handle:
            json.dump({"test_accuracy": 0.5}, handle)
        with open(
            self.results_dir / "global_clients_results" / "global_clients_all_outliers_metrics.json",
            "w",
        ) as handle:
            json.dump({"test_accuracy": 0.5}, handle)


@pytest.fixture(scope="session")
def synthetic_provider(tmp_path_factory) -> SyntheticProvider:
    """A ready-to-use synthetic provider with its artefacts on disk."""
    results_dir = tmp_path_factory.mktemp("results")
    provider = SyntheticProvider(results_dir)
    provider.write_artefacts()
    return provider


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Keep the suite on the CPU regardless of the host."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)


def make_trainer(trainer_cls, provider) -> Any:
    """Instantiate a trainer against the synthetic provider."""
    return trainer_cls(provider=provider)


# --------------------------------------------------------------------------- #
# v6 fixtures: a small multi-writer cache and the two fold books over it
#
# These live here rather than in one test module because three files now build
# runs on them - the pipeline tests, the stage-5 evaluation tests and the
# extreme-case tests - and a fixture copied into each would be three fixtures
# that can drift apart.
# --------------------------------------------------------------------------- #
@pytest.fixture()
def cohort_cache(tmp_path):
    """
    A cache with four cohort writers and two old-data writers.

    Every writer holds six samples in each of three classes, which is enough for
    a 60/20/20 book fold to leave all three parts non-empty - the shape stage 5
    actually runs on, as opposed to the one-sample-per-class corner that
    ``thin_cache`` exists to cover.
    """
    from federated_outlier_adaptation.data import sd19_labels
    from federated_outlier_adaptation.data.emnist_convert import conversion_info

    out = tmp_path / "cohort"
    out.mkdir()
    writers = [f"c000{i}_11" for i in range(4)] + [f"o000{i}_22" for i in range(2)]
    rows = []
    for index, _ in enumerate(writers):
        for label in range(3):
            rows.extend([(index, label)] * 6)

    images = np.lib.format.open_memmap(
        out / "nist28_images.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 28, 28)
    )
    for row, (writer, label) in enumerate(rows):
        # A different constant per (writer, label) so a model can in principle
        # tell them apart; the tests never depend on it learning anything.
        images[row] = np.uint8(40 * label + 5 * writer + 1)
    images.flush()
    del images
    np.save(out / "nist28_labels.npy", np.array([l for _, l in rows], dtype=np.uint8))
    np.save(out / "nist28_writers.npy", np.array([w for w, _ in rows], dtype=np.int32))
    with open(out / "nist28_index.json", "w") as handle:
        json.dump(
            {
                "resolution": 28,
                "classes": "all",
                "num_classes": 62,
                "class_map": {str(k): v for k, v in sd19_labels.class_map("all").items()},
                "conversion": conversion_info(),
                "rows": len(rows),
                "writers": writers,
            },
            handle,
        )
    return out


@pytest.fixture()
def stage5_setup(cohort_cache, tmp_path):
    """A provider, the cohort's writers, a cohort book and an old book."""
    from federated_outlier_adaptation.data.fold_book import build_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r5", cache_dir=cohort_cache, resolution=28, classes="all"
    )
    cohort_writers = [f"c000{i}_11" for i in range(4)]
    old_writers = [f"o000{i}_22" for i in range(2)]
    cohort = build_fold_book(provider.dataset, writers=cohort_writers, folds=5, seed=11)
    old = build_fold_book(provider.dataset, writers=old_writers, folds=5, seed=13)
    return provider, cohort_writers, cohort, old_writers, old
