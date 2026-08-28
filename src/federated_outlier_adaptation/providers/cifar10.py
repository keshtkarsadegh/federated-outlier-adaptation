"""
CIFAR-10 provider with Dirichlet-partitioned clients.

Same protocol as NIST: a seeded subset of the clients forms the pooled global
training set, the remaining clients are held out, scored by the global model,
and the ``k`` lowest-accuracy ones become the federated participants.

Artefacts live under ``RESULTS_DIR/cifar10/`` with the NIST file names
(``writer_split.json``, ``global_model``, ``global_results/fisher/``,
``outliers/``).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.cifar10 import NUM_CLASSES, Cifar10Data
from federated_outlier_adaptation.models.cifar_cnn import CifarCNN
from federated_outlier_adaptation.providers.base import DatasetProvider, proxy_digest

#: Clients used for pooled (server-side) training; the remainder is held out.
#: With the default 200-client partition this is a 100 / 100 split, which keeps
#: the held-out population large enough for a bottom-fraction outlier pool.
#: The earlier 100-client layout is reached with
#: ``Cifar10Provider(num_global_clients=80)`` on a dataset prepared with
#: ``--num-clients 100``.
DEFAULT_GLOBAL_CLIENTS = 100

#: Minimum pool size for a held-out client to be eligible as a participant.
DEFAULT_MIN_SELECTION_SAMPLES = 20


class Cifar10Provider(DatasetProvider):
    """CIFAR-10: 10 classes, 200 Dirichlet(0.3) clients, compact CNN."""

    name = "cifar10"

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        npz_path: Optional[Path] = None,
        clients_path: Optional[Path] = None,
        outliers_file: Optional[Path] = None,
        num_global_clients: int = DEFAULT_GLOBAL_CLIENTS,
        split_seed: int = 42,
        min_selection_samples: int = DEFAULT_MIN_SELECTION_SAMPLES,
        model_kwargs: Optional[Dict[str, Any]] = None,
        augment_global: bool = True,
    ):
        self.results_dir = Path(results_dir) if results_dir else config.CIFAR10_RESULTS_DIR
        self.data_dir = Path(data_dir) if data_dir else config.CIFAR10_DIR
        self._outliers_file = Path(outliers_file) if outliers_file else None
        self.num_global_clients = int(num_global_clients)
        self.split_seed = int(split_seed)
        self.min_selection_samples = int(min_selection_samples)
        self.model_kwargs = dict(model_kwargs or {})
        self.augment_global = bool(augment_global)

        self.dataset = Cifar10Data(
            npz_path=npz_path if npz_path else self.data_dir / config.CIFAR10_NPZ_NAME,
            clients_path=(
                clients_path if clients_path else self.data_dir / config.CIFAR10_CLIENTS_NAME
            ),
        )
        self._split: Optional[dict] = None

    # ----------------------------------------------------------------- paths
    @property
    def writer_split_path(self) -> Path:
        return self.results_dir / "writer_split.json"

    @property
    def outliers_dir(self) -> Path:
        return self.results_dir / "outliers"

    @property
    def clients_acc_on_global_path(self) -> Path:
        return self.outliers_dir / "clients_acc_on_global.json"

    @property
    def global_model_path(self) -> Path:
        return self.results_dir / "global_model"

    @property
    def fisher_dir(self) -> Path:
        return self.results_dir / "global_results" / "fisher"

    @property
    def outliers_file(self) -> Path:
        if self._outliers_file is not None:
            return self._outliers_file
        published = self.outliers_dir / "selected_outliers.json"
        if published.is_file():
            return published
        return self.selected_clients_path(5)

    def selected_clients_path(self, k: int = 5) -> Path:
        return self.outliers_dir / f"selected_clients_k{k}.json"

    def pool_path(self, frac: float = 0.05) -> Path:
        """Location of the rule-based outlier pool of a bottom fraction."""
        from federated_outlier_adaptation.outliers.client_accuracy import pool_file_name

        return self.outliers_dir / pool_file_name(frac=frac)

    def outlier_pool(self, frac: float = 0.05) -> List[str]:
        """
        The rule-based outlier pool the participants are sampled from.

        Reads the stored pool file when it exists and derives the pool from
        ``clients_acc_on_global.json`` otherwise.
        """
        from federated_outlier_adaptation.outliers.client_accuracy import build_pool, load_pool

        path = self.pool_path(frac)
        if path.is_file():
            return load_pool(path)
        return list(
            build_pool(
                self.clients_acc_on_global_path, frac=frac, eligible=self.eligible_clients()
            )["clients"]
        )

    # --------------------------------------------------------------- clients
    def make_client_split(self, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Deterministic (local, global) partition of the Dirichlet clients."""
        clients = sorted(self.dataset.all_clients())
        shuffled = list(clients)
        random.Random(self.split_seed if seed is None else int(seed)).shuffle(shuffled)
        n_global = min(self.num_global_clients, len(clients))
        global_clients = sorted(shuffled[:n_global])
        local_clients = sorted(shuffled[n_global:])
        return local_clients, global_clients

    def _client_split(self) -> dict:
        if self._split is None:
            if self.writer_split_path.is_file():
                with open(self.writer_split_path) as handle:
                    self._split = json.load(handle)
            else:
                local_clients, global_clients = self.make_client_split()
                self._split = {"local_writers": local_clients, "global_writers": global_clients}
        return self._split

    def set_client_split(self, local_clients: List[str], global_clients: List[str]) -> None:
        """Adopt a split that was just (re)generated, replacing any cached one."""
        self._split = {
            "local_writers": list(local_clients),
            "global_writers": list(global_clients),
        }

    def all_client_ids(self) -> List[str]:
        return self.dataset.all_clients()

    def local_client_ids(self) -> List[str]:
        return list(self._client_split()["local_writers"])

    def global_client_ids(self) -> List[str]:
        return list(self._client_split()["global_writers"])

    def eligible_clients(self, min_samples: Optional[int] = None) -> List[str]:
        threshold = self.min_selection_samples if min_samples is None else int(min_samples)
        return [
            client
            for client in self.local_client_ids()
            if self.dataset.sample_count(client) >= threshold
        ]

    def selected_clients(self, k: int = 5) -> List[str]:
        """The ``k`` lowest-accuracy held-out clients."""
        if self._outliers_file is not None:
            with open(self._outliers_file) as handle:
                return list(json.load(handle))

        path = self.selected_clients_path(k)
        if not path.is_file() and k == 5:
            path = self.outliers_dir / "selected_outliers.json"
        if path.is_file():
            with open(path) as handle:
                return list(json.load(handle))

        from federated_outlier_adaptation.outliers.client_accuracy import (
            lowest_accuracy_clients,
        )

        return lowest_accuracy_clients(
            self.clients_acc_on_global_path,
            k=k,
            eligible=self.eligible_clients(),
        )

    # ------------------------------------------------------------------ data
    def build_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ):
        return self.dataset.build_dataset(
            client_ids,
            seed=seed,
            train_rate=train_rate,
            eval_rate=eval_rate,
            batch_size=batch_size,
            loader_seed=loader_seed,
        )

    def build_training_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ):
        """
        Loaders for pooled server-side training.

        Identical to :meth:`build_dataset` except that the training loader
        applies the standard CIFAR-10 augmentation (padded random crop and
        horizontal flip).  Validation and test loaders stay deterministic, and
        client-side adaptation never augments.
        """
        return self.dataset.build_dataset(
            client_ids,
            seed=seed,
            train_rate=train_rate,
            eval_rate=eval_rate,
            batch_size=batch_size,
            loader_seed=loader_seed,
            augment=self.augment_global,
        )

    def sample_count(self, client_id: str) -> int:
        return self.dataset.sample_count(client_id)

    # ----------------------------------------------------------------- model
    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes or NUM_CLASSES

    def make_model(self) -> CifarCNN:
        kwargs = {
            "num_classes": self.num_classes,
            "widths": (64, 128, 256),
            "fc_dim": 256,
            "dropout_rate": 0.1,
            "use_dropout": True,
        }
        kwargs.update(self.model_kwargs)
        return CifarCNN(**kwargs)

    # ----------------------------------------------------------------- proxy
    def proxy_loader(self, batch_size: int = 64):
        """
        The evaluation images reserved by ``prepare --proxy-size``.

        They are assigned to no client, so the server may score a model on them
        without reading anyone's data.  A partition prepared without a proxy
        reserve returns ``None`` and the proxy signals stay ``None``.
        """
        dataset = self.dataset.proxy_dataset()
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def proxy_info(self) -> Dict[str, Any]:
        """Description and content hash of the reserved proxy set."""
        rows = self.dataset.proxy_rows()
        if rows.size == 0:
            return {}
        source = self.dataset.index.get("source_sha256")
        return {
            "name": "cifar10_reserved_test",
            "description": (
                "CIFAR-10 evaluation images held back from the Dirichlet "
                "partition, so they belong to no client"
            ),
            "path": str(self.dataset.clients_path),
            "source_sha256": source,
            "size": int(rows.size),
            "hash": proxy_digest([source, int(rows.size), rows.tobytes().hex()]),
        }

    # ------------------------------------------------------------ provenance
    def dataset_files(self) -> Dict[str, str]:
        return {
            "cifar10_npz": str(self.dataset.npz_path),
            "cifar10_clients": str(self.dataset.clients_path),
        }
