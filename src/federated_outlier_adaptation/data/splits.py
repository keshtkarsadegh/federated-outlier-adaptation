"""
Client-level train/validation/test splitting shared by the dataset providers.

The NIST pipeline splits a client's samples with
:meth:`~federated_outlier_adaptation.data.datasets.NistDataset.build_dataset`:
the samples of all requested clients are pooled, grouped by label, shuffled with
a fixed seed and cut at ``train_rate`` / ``eval_rate``.  The remaining fraction
becomes the test split.  Rates of ``0.0`` are meaningful and are used throughout
the code base to obtain a pure test loader.

This module re-implements exactly that rule over plain integer index arrays so
that datasets which are not backed by a file manifest (LEAF Shakespeare,
CIFAR-10) follow the identical protocol.  ``datasets.py`` itself is untouched,
so the published NIST numbers cannot move.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SplitIndices = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _cut(n: int, train_rate: float, eval_rate: float) -> Tuple[int, int]:
    """The two cut points NIST uses: ``int(n * rate)``, truncated."""
    n_train = int(n * train_rate)
    n_val = int(n * eval_rate)
    return n_train, n_val


def stratified_split(
    indices: Sequence[int],
    labels: Sequence[int],
    train_rate: float = 0.6,
    eval_rate: float = 0.2,
    seed: Optional[int] = 42,
) -> SplitIndices:
    """
    Label-preserving split of ``indices``.

    Args:
        indices: Sample indices to split.
        labels: Label of every entry of ``indices`` (same length).
        train_rate: Fraction of each label group used for training.
        eval_rate: Fraction of each label group used for validation.
        seed: Seed of the per-group shuffle.  ``None`` shuffles unseeded.

    Returns:
        ``(train, val, test)`` index arrays.
    """
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels)
    if indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    rng = np.random.default_rng(seed)
    train: List[np.ndarray] = []
    val: List[np.ndarray] = []
    test: List[np.ndarray] = []

    # Groups are visited in order of first appearance, mirroring the
    # ``defaultdict(list)`` iteration order of the NIST implementation.
    order: Dict[int, None] = {}
    for label in labels.tolist():
        order.setdefault(label, None)

    for label in order:
        group = indices[labels == label]
        group = group[rng.permutation(group.size)]
        n_train, n_val = _cut(group.size, train_rate, eval_rate)
        train.append(group[:n_train])
        val.append(group[n_train : n_train + n_val])
        test.append(group[n_train + n_val :])

    def joined(parts: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    return joined(train), joined(val), joined(test)


def random_split(
    indices: Sequence[int],
    train_rate: float = 0.6,
    eval_rate: float = 0.2,
    seed: Optional[int] = 42,
) -> SplitIndices:
    """Unstratified counterpart of :func:`stratified_split`."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()

    rng = np.random.default_rng(seed)
    shuffled = indices[rng.permutation(indices.size)]
    n_train, n_val = _cut(shuffled.size, train_rate, eval_rate)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def split_indices(
    indices: Sequence[int],
    labels: Optional[Sequence[int]] = None,
    train_rate: float = 0.6,
    eval_rate: float = 0.2,
    seed: Optional[int] = 42,
    is_stratified: bool = True,
) -> SplitIndices:
    """Dispatch to the stratified or the random split, as NIST does."""
    if is_stratified and labels is not None:
        return stratified_split(indices, labels, train_rate, eval_rate, seed)
    return random_split(indices, train_rate, eval_rate, seed)
