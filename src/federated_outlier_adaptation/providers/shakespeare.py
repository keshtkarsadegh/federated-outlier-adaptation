"""
LEAF Shakespeare provider.

Clients are speaking roles (``<PLAY>_<ROLE>``), the model is the LEAF reference
character LSTM and the protocol is the NIST one: a seeded fraction of the roles
forms the pooled *global* set that trains the baseline model, the remaining
roles are held out, scored by that model, and the ``k`` lowest-accuracy ones
become the federated participants.

Artefacts live under ``RESULTS_DIR/shakespeare/`` with the same file names the
NIST pipeline uses (``writer_split.json``, ``global_model``,
``global_results/fisher/``, ``outliers/``), so runners, trainers and plots need
no dataset-specific branch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.shakespeare import (
    NUM_LETTERS,
    ShakespeareData,
    ShakespeareSequenceDataset,
)
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.models.char_lstm import CharLSTM
from federated_outlier_adaptation.providers.base import DatasetProvider, proxy_digest

#: Fraction of roles that forms the pooled global training set.  Ten percent of
#: the roles carry roughly a tenth of the four million sequences, which keeps
#: the pooled set comfortably above the two hundred thousand sequences the
#: baseline needs while leaving the vast majority of roles held out.
DEFAULT_GLOBAL_FRACTION = 0.10

#: A held-out role needs at least this many sequences to be eligible as a
#: federated participant; below that its accuracy estimate is too noisy.
DEFAULT_MIN_SELECTION_SAMPLES = 100

#: Held-out roles carved out as the server-side proxy set.  They are removed
#: from the eligible population, so they never become participants and never
#: enter an outlier pool.
DEFAULT_PROXY_ROLES = 20

#: Seed of that carve-out; the same 42 the dataset split uses.
PROXY_SEED = 42

#: Sequences drawn from the proxy roles.  The twenty roles carry far more
#: windows than a per-round evaluation should cost, so a deterministic
#: subsample of this size is used.
DEFAULT_PROXY_SEQUENCES = 2000

#: Upper bound on the share of the eligible population the proxy set may take.
#: With the real corpus (954 eligible roles) it is never binding; it exists so
#: that a small or toy corpus keeps a usable client population instead of being
#: swallowed by the carve-out.
DEFAULT_PROXY_MAX_FRACTION = 0.25


class ShakespeareProvider(DatasetProvider):
    """LEAF Shakespeare: 80-symbol next-character prediction, roles as clients."""

    name = "shakespeare"

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        npz_path: Optional[Path] = None,
        index_path: Optional[Path] = None,
        outliers_file: Optional[Path] = None,
        global_fraction: float = DEFAULT_GLOBAL_FRACTION,
        split_seed: int = 42,
        min_selection_samples: int = DEFAULT_MIN_SELECTION_SAMPLES,
        model_kwargs: Optional[Dict[str, Any]] = None,
        proxy_roles: int = DEFAULT_PROXY_ROLES,
        proxy_sequences: int = DEFAULT_PROXY_SEQUENCES,
        proxy_seed: int = PROXY_SEED,
        proxy_max_fraction: float = DEFAULT_PROXY_MAX_FRACTION,
    ):
        self.results_dir = Path(results_dir) if results_dir else config.SHAKESPEARE_RESULTS_DIR
        self.data_dir = Path(data_dir) if data_dir else config.SHAKESPEARE_DIR
        self._outliers_file = Path(outliers_file) if outliers_file else None
        self.global_fraction = float(global_fraction)
        self.split_seed = int(split_seed)
        self.min_selection_samples = int(min_selection_samples)
        self.model_kwargs = dict(model_kwargs or {})

        self.dataset = ShakespeareData(
            npz_path=npz_path if npz_path else self.data_dir / config.SHAKESPEARE_NPZ_NAME,
            index_path=index_path if index_path else self.data_dir / config.SHAKESPEARE_INDEX_NAME,
        )
        self._split: Optional[dict] = None

        self.proxy_roles = int(proxy_roles)
        self.proxy_sequences = int(proxy_sequences)
        self.proxy_seed = int(proxy_seed)
        self.proxy_max_fraction = float(proxy_max_fraction)
        self._proxy_clients: Optional[List[str]] = None

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
            # A pool file written before the proxy roles existed may still name
            # one of them; the reserved roles are filtered out on read, so the
            # stored file stays valid documentation of the rule that produced it
            # while the run itself keeps the two populations disjoint.
            return self._without_proxy(load_pool(path), f"the stored pool {path.name}")
        return list(
            build_pool(
                self.clients_acc_on_global_path, frac=frac, eligible=self.eligible_clients()
            )["clients"]
        )

    # --------------------------------------------------------------- clients
    def make_client_split(self, seed: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Deterministic (local, global) partition of the roles.

        A seeded shuffle picks ``global_fraction`` of the roles as the pooled
        server-side set; the rest are held out.  Called by the global-training
        phase when ``writer_split.json`` does not exist yet.
        """
        users = sorted(self.dataset.all_users())
        shuffled = list(users)
        random.Random(self.split_seed if seed is None else int(seed)).shuffle(shuffled)
        n_global = max(1, round(len(users) * self.global_fraction))
        global_users = sorted(shuffled[:n_global])
        local_users = sorted(shuffled[n_global:])
        return local_users, global_users

    def _client_split(self) -> dict:
        if self._split is None:
            if self.writer_split_path.is_file():
                with open(self.writer_split_path) as handle:
                    self._split = json.load(handle)
            else:
                local_users, global_users = self.make_client_split()
                self._split = {"local_writers": local_users, "global_writers": global_users}
        return self._split

    def set_client_split(self, local_clients: List[str], global_clients: List[str]) -> None:
        """Adopt a split that was just (re)generated, replacing any cached one."""
        self._split = {
            "local_writers": list(local_clients),
            "global_writers": list(global_clients),
        }

    def all_client_ids(self) -> List[str]:
        return self.dataset.all_users()

    def local_client_ids(self) -> List[str]:
        return list(self._client_split()["local_writers"])

    def global_client_ids(self) -> List[str]:
        return list(self._client_split()["global_writers"])

    def proxy_clients(self) -> List[str]:
        """
        The held-out roles reserved as the server-side proxy set.

        All roles of this corpus belong either to the pooled global set or to
        the held-out population, so a proxy set has to be carved out of one of
        them.  It is taken from the held-out side with a seeded shuffle of the
        roles that carry enough sequences to be scored, which makes the choice
        deterministic and independent of any accuracy file, and the resulting
        roles are then removed from :meth:`eligible_clients` - they can never be
        participants and never enter an outlier pool.
        """
        if self._proxy_clients is None:
            if self.proxy_roles <= 0:
                self._proxy_clients = []
                return []
            candidates = sorted(
                client
                for client in self._client_split()["local_writers"]
                if self.dataset.sample_count(client) >= self.min_selection_samples
            )
            allowed = int(len(candidates) * self.proxy_max_fraction)
            take = max(0, min(self.proxy_roles, allowed))
            shuffled = list(candidates)
            random.Random(self.proxy_seed).shuffle(shuffled)
            self._proxy_clients = sorted(shuffled[:take])
        return list(self._proxy_clients)

    def _without_proxy(self, clients: Sequence[str], context: str) -> List[str]:
        """Drop the reserved proxy roles from a client list."""
        reserved = set(self.proxy_clients())
        kept = [client for client in clients if client not in reserved]
        dropped = len(clients) - len(kept)
        if dropped:
            NistLogger.info(
                f"Excluded {dropped} reserved proxy role(s) from {context}; "
                f"{len(kept)} remain."
            )
        return kept

    def eligible_clients(self, min_samples: Optional[int] = None) -> List[str]:
        """
        Held-out roles with enough sequences to be scored reliably.

        The roles reserved as the proxy set are excluded, so the population the
        participants are drawn from is disjoint from the data the server uses to
        observe the run.
        """
        threshold = self.min_selection_samples if min_samples is None else int(min_samples)
        candidates = [
            client
            for client in self.local_client_ids()
            if self.dataset.sample_count(client) >= threshold
        ]
        return self._without_proxy(candidates, "the eligible held-out roles")

    def selected_clients(self, k: int = 5) -> List[str]:
        """
        The ``k`` lowest-accuracy held-out roles.

        Reads the list written by the global-training phase.  When it is absent
        the list is derived on the fly from ``clients_acc_on_global.json``.
        """
        if self._outliers_file is not None:
            with open(self._outliers_file) as handle:
                return list(json.load(handle))

        path = self.selected_clients_path(k)
        if not path.is_file() and k == 5:
            path = self.outliers_dir / "selected_outliers.json"
        if path.is_file():
            with open(path) as handle:
                return self._without_proxy(list(json.load(handle)), f"the list {path.name}")

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

    def sample_count(self, client_id: str) -> int:
        return self.dataset.sample_count(client_id)

    # ----------------------------------------------------------------- model
    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes or NUM_LETTERS

    def make_model(self) -> CharLSTM:
        kwargs = {"vocab": self.num_classes, "embed": 8, "hidden": 256, "layers": 2}
        kwargs.update(self.model_kwargs)
        return CharLSTM(**kwargs)

    # ----------------------------------------------------------------- proxy
    def proxy_offsets(self) -> np.ndarray:
        """
        Window offsets of the proxy set: the reserved roles, subsampled.

        The twenty roles together carry far more windows than a per-round
        evaluation should cost, so a deterministic sample of
        ``proxy_sequences`` of them is drawn once (seeded with
        :data:`PROXY_SEED`).
        """
        roles = self.proxy_clients()
        if not roles:
            return np.empty(0, dtype=np.int64)
        offsets = np.concatenate([self.dataset.window_offsets(role) for role in roles])
        if 0 < self.proxy_sequences < offsets.size:
            rng = np.random.default_rng(self.proxy_seed)
            offsets = offsets[rng.choice(offsets.size, size=self.proxy_sequences, replace=False)]
        return np.sort(offsets)

    def proxy_loader(self, batch_size: int = 64):
        """
        Sequences of the reserved held-out roles, in a fixed order.

        Those roles belong to neither the pooled global set nor the eligible
        client population, so scoring a model on them reads nobody's data.
        """
        offsets = self.proxy_offsets()
        if offsets.size == 0:
            return None
        dataset = ShakespeareSequenceDataset(
            self.dataset.tokens, offsets, self.dataset.seq_length
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def proxy_info(self) -> Dict[str, Any]:
        """Description and content hash of the reserved-role proxy set."""
        roles = self.proxy_clients()
        if not roles:
            return {}
        offsets = self.proxy_offsets()
        return {
            "name": "shakespeare_reserved_roles",
            "description": (
                f"{len(roles)} held-out roles carved out with seed "
                f"{self.proxy_seed} and excluded from the eligible population"
            ),
            "path": str(self.dataset.index_path),
            "roles": list(roles),
            "size": int(offsets.size),
            "seed": self.proxy_seed,
            "hash": proxy_digest(
                [self.index_checksum(), ",".join(roles), int(offsets.size), self.proxy_seed]
            ),
        }

    def index_checksum(self) -> Optional[str]:
        """Corpus checksum recorded by the preparation step, if present."""
        return self.dataset.index.get("corpus_sha256") or self.dataset.index.get("source_sha256")

    # ------------------------------------------------------------ provenance
    def dataset_files(self) -> Dict[str, str]:
        return {
            "shakespeare_npz": str(self.dataset.npz_path),
            "shakespeare_index": str(self.dataset.index_path),
        }
