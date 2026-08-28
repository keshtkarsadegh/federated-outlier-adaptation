"""
NIST SD19 provider.

Wraps the dataset layer and the model and exposes the frozen artefacts of the
pipeline (writer split, selected outliers, global model, Fisher information).

Two settings of the same dataset are served, chosen by ``resolution`` and
``classes`` (or by ``FOA_NIST_RESOLUTION`` / ``FOA_NIST_CLASSES``):

``128`` / ``digits``
    the published pipeline - :class:`~federated_outlier_adaptation.data.datasets.NistDataset`
    over the 128x128 packed cache, ten classes;
``28`` / ``all``
    the by-writer FEMNIST task of plan v4 -
    :class:`~federated_outlier_adaptation.data.nist28.Nist28Dataset` over the
    packed 28x28 cache produced by the EMNIST conversion, 62 classes.
    ``classes="digits"`` restricts the same cache to the ten digits, which is
    the character-set ablation.

The model is selected independently (``model`` or ``FOA_MODEL``):
:class:`~federated_outlier_adaptation.models.fedavg_cnn.FedAvgCNN` (the default,
the reference topology of McMahan et al.) or
:class:`~federated_outlier_adaptation.model.FlexibleCNN` (kept as the
architecture ablation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from torch.utils.data import DataLoader

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.datasets import NistDataset
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.model import FlexibleCNN
from federated_outlier_adaptation.models.fedavg_cnn import FedAvgCNN
from federated_outlier_adaptation.providers.base import DatasetProvider, proxy_digest

#: Seed of the proxy subsample, so a reduced proxy set is still reproducible.
PROXY_SEED = 42


def default_proxy_size() -> Optional[int]:
    """
    Size of the MNIST proxy set: the whole 10 000-image test set by default.

    Scoring 10 000 images at 128x128 once per round is the same order as the
    source-side evaluation the runners already make, but on a long sweep it is
    still real time.  ``FOA_NIST_PROXY_SIZE`` caps it with a deterministic,
    seeded subsample; the value is recorded in the proxy hash, so two runs with
    different caps can never be mistaken for each other.
    """
    raw = config.env_value("NIST_PROXY_SIZE")
    if not raw:
        return None
    try:
        size = int(raw)
    except ValueError:
        NistLogger.info(f"Ignoring {config.env_name('NIST_PROXY_SIZE')}={raw!r}: not an integer.")
        return None
    return size if size > 0 else None


class NistProvider(DatasetProvider):
    """NIST SD19 by writer; see the module docstring for the two settings."""

    name = "nist"

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        outliers_file: Optional[Path] = None,
        use_cache: bool = True,
        mnist_dir: Optional[Path] = None,
        proxy_size: Optional[int] = None,
        proxy_seed: int = PROXY_SEED,
        resolution: Optional[int] = None,
        classes: Optional[str] = None,
        model: Optional[str] = None,
        require_trainable: Optional[bool] = None,
    ):
        self.results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self._outliers_file = Path(outliers_file) if outliers_file else None
        self.resolution = int(resolution) if resolution else config.nist_resolution()
        self.classes = str(classes) if classes else config.nist_classes()
        self.model_name = str(model) if model else config.model_name()
        if self.resolution == 28:
            from federated_outlier_adaptation.data.nist28 import Nist28Dataset

            self.dataset = Nist28Dataset(
                cache_dir=Path(cache_dir) if cache_dir else config.NIST28_DIR,
                classes=self.classes,
            )
        else:
            if self.classes != "digits":
                raise ValueError(
                    "The 128x128 cache holds the digits only; run the 62-class "
                    "task at --resolution 28."
                )
            self.dataset = NistDataset(
                data_dir=self.data_dir,
                labels_json=self.data_dir / "by_write" / "digits_labels.json",
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        self.require_trainable = (
            config.require_trainable() if require_trainable is None else bool(require_trainable)
        )
        if self.require_trainable and hasattr(self.dataset, "trainable_writers"):
            # ``outliers.selection.eligible_of`` looks this attribute up, so
            # binding it only on request keeps every existing pool unchanged.
            self.eligible_clients = self._eligible_clients

        self._writer_split: Optional[dict] = None
        if hasattr(self.dataset, "split_writers"):
            # The global-training phase looks this attribute up to decide
            # whether the provider draws its own split.
            self.make_client_split = self._make_client_split

        self.mnist_dir = Path(mnist_dir) if mnist_dir else config.MNIST_DIR
        self.proxy_size = default_proxy_size() if proxy_size is None else int(proxy_size)
        self.proxy_seed = int(proxy_seed)
        self._mnist = None
        self._mnist_loaded = False
        self._proxy_info: Optional[dict] = None

    # ----------------------------------------------------------------- clients
    def _split(self) -> dict:
        if self._writer_split is None:
            with open(self.results_dir / "writer_split.json", "r") as handle:
                self._writer_split = json.load(handle)
        return self._writer_split

    def all_client_ids(self) -> List[str]:
        return self.dataset.all_writers()

    def local_client_ids(self) -> List[str]:
        return list(self._split()["local_writers"])

    def global_client_ids(self) -> List[str]:
        return list(self._split()["global_writers"])

    @property
    def outliers_dir(self) -> Path:
        return self.results_dir / "outliers"

    @property
    def clients_acc_on_global_path(self) -> Path:
        return self.outliers_dir / "clients_acc_on_global.json"

    @property
    def outliers_file(self) -> Path:
        if self._outliers_file is not None:
            return self._outliers_file
        return self.outliers_dir / "selected_outliers.json"

    def pool_path(self, frac: float = 0.05) -> Path:
        """Location of the rule-based outlier pool of a bottom fraction."""
        from federated_outlier_adaptation.outliers.client_accuracy import pool_file_name

        return self.outliers_dir / pool_file_name(frac=frac)

    def outlier_pool(self, frac: float = 0.05) -> List[str]:
        """
        The rule-based outlier pool the participants are sampled from.

        Reads the stored pool file when it exists and derives the pool from
        ``clients_acc_on_global.json`` otherwise.  The published five-writer
        selection is untouched: it lives in ``selected_outliers.json`` and is
        still what ``selected_clients`` returns.
        """
        from federated_outlier_adaptation.outliers.client_accuracy import (
            build_pool,
            load_pool,
        )

        path = self.pool_path(frac)
        if path.is_file():
            return load_pool(path)
        return list(build_pool(self.clients_acc_on_global_path, frac=frac)["clients"])

    def selected_clients(self, k: int = 5) -> List[str]:
        path = self.outliers_file
        if self._outliers_file is None and k != 5:
            path = self.results_dir / "outliers" / f"selected_outliers_k{k}.json"
        with open(path, "r") as handle:
            return list(json.load(handle))

    # -------------------------------------------------------------------- data
    def build_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ) -> Tuple[Any, Any, Any]:
        from federated_outlier_adaptation.data.merged_clients import expand

        # A merged client ("w1+w2") is its members: the book unions their rows
        # per partition, which is exactly what the 'dual' extreme case means.
        return self.dataset.build_dataset(
            expand(client_ids),
            seed=seed,
            train_rate=train_rate,
            eval_rate=eval_rate,
            batch_size=batch_size,
            loader_seed=loader_seed,
        )

    def sample_count(self, client_id: str) -> int:
        from federated_outlier_adaptation.data.merged_clients import members

        # Aggregation weights a client by how much data it holds, and a merged
        # client holds all of its members' - so the counts add.
        return sum(
            self.dataset.get_sample_count(member) for member in members(client_id)
        )

    # ------------------------------------------------------------------- model
    @property
    def num_classes(self) -> int:
        """10 for the digit task, 62 for the by-class task."""
        return int(getattr(self.dataset, "num_classes", 10))

    def make_model(self):
        """The run's model: ``FedAvgCNN`` by default, ``FlexibleCNN`` as ablation."""
        if self.model_name == "flexible_cnn":
            return FlexibleCNN(
                num_classes=self.num_classes,
                complexity=3,
                dropout_rate=0.1,
                use_dropout=True,
                regularization=False,
                input_size=self.resolution,
            )
        return FedAvgCNN(num_classes=self.num_classes, input_size=self.resolution)

    def make_teacher_model(self):
        """
        Teacher/anchor model used by the distillation-style trainers.

        Kept separate because the published ``FlexibleCNN`` teacher was
        constructed without the explicit ``regularization=False`` argument; the
        resulting module is identical but the call is reproduced verbatim.
        """
        if self.model_name == "flexible_cnn":
            return FlexibleCNN(
                num_classes=self.num_classes,
                complexity=3,
                dropout_rate=0.1,
                use_dropout=True,
                input_size=self.resolution,
            )
        return FedAvgCNN(num_classes=self.num_classes, input_size=self.resolution)

    def model_info(self) -> dict:
        """Topology, parameter count and setting of the run, for provenance."""
        model = self.make_model()
        return {
            "model": self.model_name,
            "model_class": type(model).__name__,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "num_classes": self.num_classes,
            "resolution": self.resolution,
            "classes": self.classes,
        }

    def _eligible_clients(self) -> List[str]:
        """
        Clients a pool may contain, when the run asked for trainable ones only.

        Bound as ``eligible_clients`` only under ``require_trainable`` (or
        ``FOA_REQUIRE_TRAINABLE=1``); the pool writers look the attribute up and
        keep every client when it is absent, which is the default.
        """
        eligible = self.dataset.trainable_writers()
        total = len(self.dataset.all_writers())
        if len(eligible) < total:
            NistLogger.info(
                f"{total - len(eligible)} of {total} writers cannot form a "
                "local training split and are excluded from the pool."
            )
        return eligible

    def _make_client_split(self, seed: int = 42, global_size: float = 0.03):
        """
        Draw the local/source writer split of the 28x28 setting.

        Bound as ``make_client_split`` only when the dataset can do it: the
        128x128 pipeline keeps its frozen ``writer_split.json`` and derives the
        split from the manifest instead, and the global-training phase decides
        between the two by the presence of this attribute.
        """
        return self.dataset.split_writers(global_size=global_size, seed=seed)

    # --------------------------------------------------------------- artefacts
    @property
    def global_model_path(self) -> Path:
        return self.results_dir / "global_model"

    @property
    def fisher_dir(self) -> Path:
        return self.results_dir / "global_results" / "fisher"

    # ----------------------------------------------------------------- proxy
    @property
    def mnist_npz_path(self) -> Path:
        return self.mnist_dir / config.MNIST_NPZ_NAME

    def mnist(self):
        """
        The prepared MNIST proxy set, or ``None`` when it has not been built.

        A checkout without ``mnist.npz`` keeps working exactly as before: the
        proxy loader is then absent and the two proxy signals report ``None``.
        """
        if not self._mnist_loaded:
            from federated_outlier_adaptation.data.mnist import MnistData

            self._mnist_loaded = True
            self._mnist = MnistData.open_if_available(
                npz_path=self.mnist_npz_path,
                index_path=self.mnist_dir / config.MNIST_INDEX_NAME,
            )
            if self._mnist is None:
                NistLogger.debug(
                    f"No MNIST proxy set at {self.mnist_npz_path}; proxy signals disabled."
                )
        return self._mnist

    def proxy_loader(self, batch_size: int = 64):
        """
        The MNIST test set rendered into the NIST input format.

        Public data from a different writer population, so the server may read
        it without touching the source writers.  The loader does not shuffle, so
        the cached reference predictions stay aligned across rounds.
        """
        data = self.mnist()
        if data is None:
            return None
        dataset = data.proxy_dataset(
            size=self.proxy_size, seed=self.proxy_seed, resolution=self.resolution
        )
        if len(dataset) == 0:  # pragma: no cover - empty archive
            return None
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def proxy_info(self) -> dict:
        """Description and content hash of the MNIST proxy set."""
        if self._proxy_info is not None:
            return dict(self._proxy_info)
        data = self.mnist()
        if data is None:
            self._proxy_info = {}
            return {}
        from federated_outlier_adaptation.data.mnist import INVERT_POLARITY
        from federated_outlier_adaptation.utils.provenance import sha256_of

        rows = data.rows(size=self.proxy_size, seed=self.proxy_seed)
        source_hash = sha256_of(self.mnist_npz_path)
        self._proxy_info = {
            "name": "mnist_test",
            "description": (
                "MNIST test set, inverted to the NIST polarity (black ink on "
                f"white) and resized to {self.resolution}x{self.resolution} "
                "with the NIST transform"
            ),
            "path": str(self.mnist_npz_path),
            "source_sha256": source_hash,
            "size": int(rows.size),
            "seed": self.proxy_seed,
            "invert_polarity": INVERT_POLARITY,
            "resolution": self.resolution,
            "hash": proxy_digest(
                [source_hash, int(rows.size), self.proxy_seed, INVERT_POLARITY, self.resolution]
            ),
        }
        return dict(self._proxy_info)

    # ------------------------------------------------------------ provenance
    def dataset_files(self) -> dict:
        files = getattr(self.dataset, "files", None)
        if files is not None:
            return dict(files())
        return {
            "digits_labels": str(self.dataset.labels_json),
            "nist_cache_index": str(self.dataset.cache_dir / config.CACHE_INDEX_NAME),
        }


_DEFAULT_PROVIDER: Optional[NistProvider] = None


def default_provider() -> NistProvider:
    """Process-wide default provider (NIST with the configured directories)."""
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = NistProvider()
    return _DEFAULT_PROVIDER


def get_provider(name: str = "nist", **kwargs) -> DatasetProvider:
    """Look up a provider by name."""
    if name.lower() in ("nist", "", None):
        return NistProvider(**kwargs) if kwargs else default_provider()
    raise ValueError(f"Unknown dataset provider: {name!r}")
