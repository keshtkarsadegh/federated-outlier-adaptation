"""
Dataset/model provider interface.

A provider bundles everything an experiment needs to know about a dataset: how
to build client loaders, which clients exist, where the frozen global model and
Fisher information live, and how to construct the model.  Runners, trainers and
training drivers talk to a provider instead of importing NIST-specific classes,
so a second dataset can be added without touching the federated logic.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple


def proxy_digest(parts: Iterable[Any]) -> str:
    """
    Stable content hash of a proxy-set definition.

    ``parts`` are the things that fully determine which samples the proxy set
    contains - a dataset checksum, the reserved row indices, the role ids - and
    are folded into one SHA-256 so a result file can prove which proxy set the
    signals were measured on.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class DatasetProvider(ABC):
    """Minimal contract every dataset must satisfy."""

    #: Short identifier used in log messages and provenance records.
    name: str = "base"

    # ----------------------------------------------------------------- clients
    @abstractmethod
    def all_client_ids(self) -> List[str]:
        """Every client id known to the dataset."""

    @abstractmethod
    def local_client_ids(self) -> List[str]:
        """Clients held out of the global (server-side) training set."""

    @abstractmethod
    def global_client_ids(self) -> List[str]:
        """Clients whose data trains and evaluates the global model."""

    @abstractmethod
    def selected_clients(self, k: int = 5) -> List[str]:
        """The ``k`` low-accuracy clients used as the federated participants."""

    # -------------------------------------------------------------------- data
    @abstractmethod
    def build_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ) -> Tuple[Any, Any, Any]:
        """Return ``(train_loader, eval_loader, test_loader)`` for the clients."""

    def build_training_dataset(
        self,
        client_ids,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        batch_size: int = 64,
        seed: Optional[int] = 42,
        loader_seed: Optional[int] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Loaders for pooled (server-side) training of the global model.

        Identical to :meth:`build_dataset` unless a dataset applies a
        training-time augmentation that must not leak into client adaptation or
        evaluation.  The default forwards, so nothing changes for NIST.
        """
        return self.build_dataset(
            client_ids,
            train_rate=train_rate,
            eval_rate=eval_rate,
            batch_size=batch_size,
            seed=seed,
            loader_seed=loader_seed,
        )

    @abstractmethod
    def sample_count(self, client_id: str) -> int:
        """Number of samples a client contributes to aggregation weighting."""

    #: Evaluation splits :meth:`evaluation_arrays` understands.
    EVALUATION_SPLITS = ("insample", "heldout")

    def evaluation_arrays(
        self,
        client_ids,
        split: str = "insample",
        batch_size: int = 64,
        loader_seed: Optional[int] = None,
    ):
        """
        Raw tensors of an evaluation split, for the evaluation cache.

        Returns ``(payload, labels, decoder)`` - the dataset's own compact
        storage form plus the batched equivalent of its per-sample transform -
        or ``None`` when the dataset cannot be materialised, in which case the
        caller keeps its ``DataLoader``.  See
        :mod:`federated_outlier_adaptation.utils.eval_cache`.

        ``insample`` is the split the published in-sample metrics use (the
        clients' whole data); ``heldout`` is the 40% the clients never train on.
        The default implementation goes through :meth:`build_dataset`, so every
        provider supports it and a dataset that gains a compact form gains the
        cheap path automatically.
        """
        from federated_outlier_adaptation.utils.eval_cache import materialise

        if split not in self.EVALUATION_SPLITS:
            raise ValueError(
                f"Unknown evaluation split {split!r}; expected one of {self.EVALUATION_SPLITS}"
            )
        rates = (0.0, 0.0) if split == "insample" else (0.6, 0.4)
        loaders = self.build_dataset(
            client_ids,
            train_rate=rates[0],
            eval_rate=rates[1],
            batch_size=batch_size,
            loader_seed=loader_seed,
        )
        loader = loaders[2] if split == "insample" else loaders[1]
        if loader is None:
            return None
        return materialise(loader.dataset)

    # ------------------------------------------------------------------- model
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output classes."""

    @abstractmethod
    def make_model(self):
        """Construct an untrained model with the dataset's default topology."""

    def make_teacher_model(self):
        """Construct the frozen teacher/anchor model (same topology by default)."""
        return self.make_model()

    # --------------------------------------------------------------- artefacts
    @property
    @abstractmethod
    def global_model_path(self) -> Path:
        """Location of the frozen global (teacher) model state dict."""

    @property
    @abstractmethod
    def fisher_dir(self) -> Path:
        """Directory holding ``fisher.pt`` and ``global_params.pt``."""

    @property
    def outliers_file(self) -> Optional[Path]:
        """Location of the selected-client list, when the dataset has one."""
        return None

    def dataset_files(self) -> dict:
        """
        Prepared dataset files of this provider, for the provenance record.

        Maps a short key to an absolute path; the provenance helper adds the
        SHA-256 of every entry that exists.
        """
        return {}

    # ----------------------------------------------------------------- proxy
    def proxy_loader(self, batch_size: int = 64):
        """
        Server-side proxy set, or ``None`` when the dataset defines none.

        The proxy set is public data that belongs to neither the source
        population nor the clients, so the server may read it without violating
        the constraint the study works under.  It is used only to *observe* a
        run (see
        :mod:`federated_outlier_adaptation.runners.forgetting_signals`);
        no trainer and no aggregation rule ever touches it.

        Implementations return a deterministic, non-shuffling ``DataLoader``.
        The default returns ``None``, so a provider that defines no proxy set
        behaves exactly as it did before the signals existed.
        """
        return None

    def proxy_info(self) -> dict:
        """
        Description and content hash of the proxy set, for the provenance block.

        Returns an empty mapping when the provider defines no proxy set.
        """
        return {}

    # ------------------------------------------------------------------ helper
    def restrict(self, client_ids: Sequence[str], subset) -> List[str]:
        """
        Filter ``client_ids`` to ``subset`` when a subset is given.

        A falsy ``subset`` (``None``, ``[]``, ``""``) means "keep everything",
        which is what the extreme-case experiments rely on.  The legacy sentinel
        string ``"None"`` is accepted with the same meaning.
        """
        if not subset or subset == "None":
            return list(client_ids)
        return [cid for cid in client_ids if cid in subset]
