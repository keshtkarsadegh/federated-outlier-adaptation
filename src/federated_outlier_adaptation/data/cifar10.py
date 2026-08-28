"""
CIFAR-10 with Dirichlet-partitioned clients.

Dataset: A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*,
Technical Report, University of Toronto, 2009.  The Python archive
``cifar-10-python.tar.gz`` is read **in memory** with :mod:`tarfile` - it is
never extracted - and stored as a single ``cifar10.npz`` (uint8 images, labels,
train/test flag) plus one ``cifar10_clients.json`` index, keeping the file count
at two.

Client partition: 200 clients drawn from a ``Dirichlet(alpha=0.3)`` distribution
over the ten labels of the 50 000 training images (seeded with 42).  The 10 000
evaluation images are then distributed over the same clients with each client's
own label proportions, so that a client's pool has a consistent label profile.
200 clients keep the held-out population large enough for a rule-based outlier
*pool* with per-round client selection; ``--num-clients 100`` reproduces the
smaller partition.
Each client therefore owns one pool of samples; the train/validation/test split
of that pool is performed by :meth:`Cifar10Data.build_dataset` with exactly the
rates the NIST pipeline uses (60 / 20 / 20 by default, and 60 / 40 / 0 or
0 / 0 / 100 where the runners ask for it), so the federated protocol is
identical across datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.splits import split_indices
from federated_outlier_adaptation.utils.eval_cache import Decoder
from federated_outlier_adaptation.utils.seeding import make_generator

NUM_CLASSES = 10
IMAGE_SIZE = 32
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

#: Channel statistics of the 50 000 training images, the constants used by the
#: standard CIFAR-10 recipes.
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def _cifar10_u8(payload: torch.Tensor) -> torch.Tensor:
    """
    Batched form of :meth:`Cifar10ImageDataset.__getitem__` without augmentation.

    ``[N, H, W, 3]`` uint8 -> ``[N, 3, H, W]`` normalised float, in the same
    order of operations the per-sample path uses.
    """
    mean = torch.tensor(MEAN, dtype=torch.float32, device=payload.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32, device=payload.device).view(1, 3, 1, 1)
    images = payload.permute(0, 3, 1, 2).float().div_(255.0)
    return (images - mean) / std


#: Decoder of the cached CIFAR-10 evaluation sets.
CIFAR10_U8 = Decoder("cifar10_u8", _cifar10_u8)

#: Default client count.  Chosen so that the held-out half still contains a
#: hundred clients, which a bottom-fraction outlier pool needs.
DEFAULT_NUM_CLIENTS = 200
DEFAULT_ALPHA = 0.3
DEFAULT_SEED = 42

#: Evaluation images held back from every client as the server-side proxy set.
#: ``0`` is the default of :func:`prepare`, which reproduces the partition that
#: existed before the proxy sets, so ``--proxy-size`` is opt-in.
DEFAULT_PROXY_SIZE = 0

_TRAIN_BATCHES = tuple(f"data_batch_{i}" for i in range(1, 6))
_TEST_BATCH = "test_batch"


# ---------------------------------------------------------------- extraction
def read_archive(archive_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decode ``cifar-10-python.tar.gz`` without writing anything to disk.

    Returns:
        ``(images, labels, is_test)`` with ``images`` of shape
        ``[60000, 32, 32, 3]`` (uint8), ``labels`` uint8 and ``is_test`` bool.
    """
    archive_path = Path(archive_path)
    payloads: Dict[str, dict] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in _TRAIN_BATCHES and name != _TEST_BATCH:
                continue
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - corrupt archive
                continue
            payloads[name] = pickle.loads(handle.read(), encoding="bytes")

    missing = [n for n in (*_TRAIN_BATCHES, _TEST_BATCH) if n not in payloads]
    if missing:
        raise ValueError(f"{archive_path} is missing the batches {missing}")

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    is_test: List[np.ndarray] = []
    for name in (*_TRAIN_BATCHES, _TEST_BATCH):
        batch = payloads[name]
        data = np.asarray(batch[b"data"], dtype=np.uint8)
        data = data.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE).transpose(0, 2, 3, 1)
        images.append(data)
        labels.append(np.asarray(batch[b"labels"], dtype=np.uint8))
        is_test.append(np.full(data.shape[0], name == _TEST_BATCH, dtype=bool))

    return np.concatenate(images), np.concatenate(labels), np.concatenate(is_test)


# ---------------------------------------------------------------- partition
def dirichlet_partition(
    labels: Sequence[int],
    num_clients: int = DEFAULT_NUM_CLIENTS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Split sample indices over clients with a per-label Dirichlet prior.

    For every label the samples carrying it are shuffled and dealt out to the
    clients in the proportions of one ``Dirichlet(alpha)`` draw, the standard
    non-IID construction for federated CIFAR-10.

    Args:
        labels: Label of every sample.
        num_clients: Number of clients to create.
        alpha: Concentration of the Dirichlet prior; smaller is more skewed.
        seed: Seed of the draw and of the shuffles.

    Returns:
        ``(client_indices, proportions)`` where ``proportions`` is the
        ``[num_clients, num_labels]`` matrix of realised label shares.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)

    buckets: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    for label in classes:
        pool = np.flatnonzero(labels == label)
        pool = pool[rng.permutation(pool.size)]
        weights = rng.dirichlet(np.full(num_clients, alpha))
        cuts = (np.cumsum(weights) * pool.size).astype(np.int64)[:-1]
        for client, part in enumerate(np.split(pool, cuts)):
            buckets[client].append(part)

    client_indices = [
        np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        for parts in buckets
    ]

    counts = np.zeros((num_clients, classes.size), dtype=np.int64)
    for client, indices in enumerate(client_indices):
        if indices.size:
            values, occurrences = np.unique(labels[indices], return_counts=True)
            counts[client, np.searchsorted(classes, values)] = occurrences
    totals = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(counts, np.maximum(totals, 1), dtype=np.float64)
    return client_indices, proportions


def distribute_by_proportions(
    labels: Sequence[int],
    proportions: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> List[np.ndarray]:
    """
    Deal a second pool of samples out with given per-client label proportions.

    Used for the 10 000 CIFAR-10 evaluation images so that a client's pool keeps
    the label profile its training share defines.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    num_clients = proportions.shape[0]
    rng = np.random.default_rng(seed + 1)

    buckets: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    for column, label in enumerate(classes):
        pool = np.flatnonzero(labels == label)
        pool = pool[rng.permutation(pool.size)]
        weights = proportions[:, column].astype(np.float64)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(num_clients, 1.0 / num_clients)
        cuts = (np.cumsum(weights) * pool.size).astype(np.int64)[:-1]
        for client, part in enumerate(np.split(pool, cuts)):
            buckets[client].append(part)

    return [
        np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        for parts in buckets
    ]


def reserve_proxy(
    rows: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Set aside a label-balanced, deterministic proxy sample.

    The reserved rows are removed from the pool the clients are dealt from, so
    no client ever sees them and a server-side evaluation on the proxy set stays
    independent of every client's data.

    Args:
        rows: Candidate sample indices (the evaluation images).
        labels: Label of every entry of ``rows``.
        size: Number of samples to reserve; ``0`` reserves nothing.
        seed: Seed of the per-label shuffles.

    Returns:
        ``(proxy_rows, remaining_rows)``, both sorted.
    """
    rows = np.asarray(rows, dtype=np.int64)
    empty = np.empty(0, dtype=np.int64)
    if size <= 0 or rows.size == 0:
        return empty, rows
    size = min(int(size), int(rows.size))

    labels = np.asarray(labels)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed + 2)
    quota = size // int(classes.size)

    picked: List[np.ndarray] = []
    leftover: List[np.ndarray] = []
    for label in classes:
        group = rows[labels == label]
        group = group[rng.permutation(group.size)]
        take = min(quota, group.size)
        picked.append(group[:take])
        leftover.append(group[take:])

    chosen = np.concatenate(picked) if picked else empty
    rest = np.concatenate(leftover) if leftover else empty
    rest = rest[rng.permutation(rest.size)]
    missing = size - int(chosen.size)
    if missing > 0:
        chosen = np.concatenate([chosen, rest[:missing]])
        rest = rest[missing:]
    return np.sort(chosen), np.sort(rest)


def build_clients(
    labels: np.ndarray,
    is_test: np.ndarray,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    proxy_size: int = DEFAULT_PROXY_SIZE,
) -> Tuple[List[str], List[np.ndarray], np.ndarray, dict]:
    """
    Assign the 60 000 images to clients, optionally reserving a proxy sample.

    ``proxy_size`` evaluation images are held back before the clients are dealt
    from the pool, so they belong to no client at all.  The default ``0``
    reproduces the earlier partition exactly.
    """
    train_rows = np.flatnonzero(~is_test)
    test_rows = np.flatnonzero(is_test)

    proxy_rows, test_rows = reserve_proxy(
        test_rows, labels[test_rows], size=int(proxy_size), seed=seed
    )

    train_parts, proportions = dirichlet_partition(
        labels[train_rows], num_clients=num_clients, alpha=alpha, seed=seed
    )
    test_parts = distribute_by_proportions(labels[test_rows], proportions, seed=seed)

    client_ids = [f"c{index:03d}" for index in range(num_clients)]
    pools = [
        np.sort(np.concatenate([train_rows[train_parts[i]], test_rows[test_parts[i]]]))
        for i in range(num_clients)
    ]

    sizes = np.asarray([pool.size for pool in pools])
    stats = {
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "num_samples": int(sizes.sum()),
        "min_client_samples": int(sizes.min()),
        "max_client_samples": int(sizes.max()),
        "mean_client_samples": float(sizes.mean()),
        "empty_clients": int((sizes == 0).sum()),
        "train_pool_samples": int(sum(part.size for part in train_parts)),
        "test_pool_samples": int(sum(part.size for part in test_parts)),
        "proxy_size": int(proxy_rows.size),
    }
    return client_ids, pools, proxy_rows, stats


# --------------------------------------------------------------- preparation
def prepare(
    archive_path: Path,
    out_dir: Optional[Path] = None,
    num_clients: int = DEFAULT_NUM_CLIENTS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    proxy_size: int = DEFAULT_PROXY_SIZE,
    clients_name: Optional[str] = None,
    log=print,
) -> dict:
    """
    Build ``cifar10.npz`` and ``cifar10_clients.json``.  Deterministic.

    Args:
        proxy_size: Evaluation images reserved as the server-side proxy set and
            therefore assigned to no client.  ``0`` (the default) reproduces the
            partition that existed before the proxy sets.
        clients_name: File name of the client index; defaults to
            ``cifar10_clients.json``.  Writing a partition under a second name
            keeps an existing one reachable.
    """
    archive_path = Path(archive_path)
    out_dir = Path(out_dir) if out_dir else config.CIFAR10_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"reading {archive_path}")
    images, labels, is_test = read_archive(archive_path)

    npz_path = out_dir / config.CIFAR10_NPZ_NAME
    clients_path = out_dir / (clients_name or config.CIFAR10_CLIENTS_NAME)
    np.savez(npz_path, images=images, labels=labels, is_test=is_test)

    client_ids, pools, proxy_rows, stats = build_clients(
        labels, is_test, num_clients=num_clients, alpha=alpha, seed=seed, proxy_size=proxy_size
    )
    index = {
        "dataset": "cifar10",
        "source": archive_path.name,
        "source_sha256": _sha256(archive_path),
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "mean": list(MEAN),
        "std": list(STD),
        "array_file": npz_path.name,
        "clients": {cid: pool.tolist() for cid, pool in zip(client_ids, pools)},
        "proxy": proxy_rows.tolist(),
        **stats,
    }
    with open(clients_path, "w") as handle:
        json.dump(index, handle)

    summary = {
        "npz": str(npz_path),
        "clients_index": str(clients_path),
        "npz_bytes": npz_path.stat().st_size,
        "num_images": int(images.shape[0]),
        **stats,
    }
    log(
        f"{images.shape[0]} images over {num_clients} clients "
        f"({proxy_rows.size} reserved as the proxy set) -> {npz_path}"
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------- access
class Cifar10ImageDataset(Dataset):
    """
    Normalised CIFAR-10 images, optionally augmented.

    ``__getitem__`` returns ``(x, y)`` with ``x`` a ``[3, 32, 32]`` float tensor
    and ``y`` the integer class id.  Augmentation - four-pixel padded random
    crop plus a horizontal flip, the standard CIFAR-10 recipe - is applied only
    when ``augment`` is set, which the pooled (server-side) training does.
    """

    def __init__(self, images: np.ndarray, rows: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.images = images
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels)
        self.augment = bool(augment)
        self._mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.rows.size)

    def __getitem__(self, item):
        row = int(self.rows[item])
        image = torch.from_numpy(np.asarray(self.images[row], dtype=np.uint8))
        image = image.permute(2, 0, 1).float().div_(255.0)
        if self.augment:
            image = self._augment(image)
        image = (image - self._mean) / self._std
        return image, int(self.labels[item])

    def eval_payload(self):
        """
        Compact form of the dataset for the evaluation cache.

        Only the evaluation configuration can be cached: an augmenting dataset
        draws a fresh crop and flip per sample, so it has no fixed tensor to
        materialise and keeps its loader.
        """
        if self.augment:
            return None
        rows = np.asarray(self.images[self.rows], dtype=np.uint8)
        payload = torch.from_numpy(np.ascontiguousarray(rows))
        labels = torch.as_tensor(np.asarray(self.labels, dtype=np.int64), dtype=torch.long)
        return payload, labels, CIFAR10_U8

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(image, (4, 4, 4, 4), mode="reflect")
        top = int(torch.randint(0, 9, (1,)).item())
        left = int(torch.randint(0, 9, (1,)).item())
        image = padded[:, top : top + IMAGE_SIZE, left : left + IMAGE_SIZE]
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=(2,))
        return image


class Cifar10Data:
    """Read-only view over a prepared CIFAR-10 dataset and its client index."""

    def __init__(self, npz_path: Optional[Path] = None, clients_path: Optional[Path] = None):
        self.npz_path = Path(npz_path) if npz_path else config.CIFAR10_NPZ
        self.clients_path = Path(clients_path) if clients_path else config.CIFAR10_CLIENTS_JSON
        with open(self.clients_path) as handle:
            self.index = json.load(handle)

        with np.load(self.npz_path) as archive:
            self.images: np.ndarray = archive["images"]
            self.labels: np.ndarray = archive["labels"]
            self.is_test: np.ndarray = archive["is_test"]

        self.clients: Dict[str, np.ndarray] = {
            cid: np.asarray(rows, dtype=np.int64) for cid, rows in self.index["clients"].items()
        }
        self.client_ids: List[str] = list(self.index["clients"].keys())
        #: Evaluation images reserved as the server-side proxy set; empty for a
        #: partition prepared without ``--proxy-size``.
        self.proxy: np.ndarray = np.asarray(self.index.get("proxy") or [], dtype=np.int64)

    # ------------------------------------------------------------- metadata
    def __len__(self) -> int:
        return len(self.client_ids)

    @property
    def num_classes(self) -> int:
        return int(self.index.get("num_classes", NUM_CLASSES))

    def all_clients(self) -> List[str]:
        return list(self.client_ids)

    def sample_count(self, client: str) -> int:
        return int(self.clients[client].size)

    def proxy_rows(self) -> np.ndarray:
        """Rows of the reserved proxy set; empty when none was reserved."""
        return np.asarray(self.proxy, dtype=np.int64)

    def proxy_dataset(self) -> Optional["Cifar10ImageDataset"]:
        """The reserved proxy set as a torch dataset, or ``None``."""
        rows = self.proxy_rows()
        if rows.size == 0:
            return None
        return Cifar10ImageDataset(self.images, rows, self.labels[rows], augment=False)

    def label_counts(self, client: str) -> np.ndarray:
        counts = np.zeros(self.num_classes, dtype=np.int64)
        rows = self.clients[client]
        if rows.size:
            values, occurrences = np.unique(self.labels[rows], return_counts=True)
            counts[values.astype(np.int64)] = occurrences
        return counts

    # -------------------------------------------------------------- loaders
    def build_dataset(
        self,
        clients,
        seed: Optional[int] = 42,
        train_rate: float = 0.6,
        eval_rate: float = 0.2,
        is_stratified: bool = True,
        batch_size: int = 64,
        loader_seed: Optional[int] = None,
        augment: bool = False,
    ):
        """
        Train/validation/test DataLoaders for one or more clients.

        The split rule is the one NIST uses: the clients' samples are pooled,
        grouped by label, shuffled with ``seed`` and cut at ``train_rate`` /
        ``eval_rate``.  ``augment`` applies the training-time augmentation to
        the training loader only.
        """
        if isinstance(clients, str):
            clients = [clients]
        known = [client for client in clients if client in self.clients]
        if not known:
            return None, None, None

        rows = np.concatenate([self.clients[client] for client in known])
        if rows.size == 0:
            return None, None, None
        labels = self.labels[rows]

        train, val, test = split_indices(
            rows,
            labels,
            train_rate=train_rate,
            eval_rate=eval_rate,
            seed=seed,
            is_stratified=is_stratified,
        )

        def loader(part: np.ndarray, use_augmentation: bool = False):
            if part.size == 0:
                return None
            dataset = Cifar10ImageDataset(
                self.images, part, self.labels[part], augment=use_augmentation
            )
            generator = make_generator(loader_seed)
            if generator is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        return loader(train, augment), loader(val), loader(test)


# ---------------------------------------------------------------- entry point
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CIFAR-10 with Dirichlet clients from the raw Python archive."
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="cifar-10-python.tar.gz (read in memory, never extracted).",
    )
    parser.add_argument("--out-dir", default=None, help="Destination of the npz and the index.")
    parser.add_argument("--num-clients", type=int, default=DEFAULT_NUM_CLIENTS)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--proxy-size",
        type=int,
        default=DEFAULT_PROXY_SIZE,
        help="Evaluation images reserved as the server-side proxy set (0: none).",
    )
    parser.add_argument(
        "--clients-name",
        default=None,
        help="File name of the client index (default: cifar10_clients.json).",
    )
    args = parser.parse_args(argv)

    raw = Path(args.raw) if args.raw else config.CIFAR10_RAW_ARCHIVE
    summary = prepare(
        archive_path=raw,
        out_dir=args.out_dir,
        num_clients=args.num_clients,
        alpha=args.alpha,
        seed=args.seed,
        proxy_size=args.proxy_size,
        clients_name=args.clients_name,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
