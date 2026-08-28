"""
Mixing source data back into a client's local training set.

This is the *data-sharing upper bound* of the study: the one thing that would
obviously stop a client from forgetting the source population is letting it keep
training on source data, which the privacy constraint the rest of the work
operates under forbids.  Running it anyway gives the ceiling every
constraint-respecting method is measured against - and the gap between that
ceiling and a method is the price of the constraint.

Two modes, both drawing from the old data's *training* split only:

``equal``
    a fresh random sample per epoch, ``multiplier`` times as many old images as
    the client has training images of its own.  ``multiplier=1`` is a balanced
    50/50 mixture; the access-to-old-data study sweeps 1, 2, 4 and 8.
``full``
    the whole old training pool, every epoch.

Which rows may be shared, and which may never be
------------------------------------------------
Under the v6 protocol the shared population is the hundred-writer **old-data
draw**, and only the **train** partition of its fold book.  The old book's
validation and test partitions are the preservation measurement: a row that
enters a client's training set and is later scored as "knowledge the model
retained" measures nothing at all.  :func:`old_train_dataset` therefore reads
the book's ``train`` part and nothing else, and a test asserts the shared rows
are disjoint from every validation and test partition of every fold.

This replaces the v4-era pool, which came from ``provider.global_client_ids()``
- the frozen 3% writer split.  That population is not the old data, is not
disjoint from the outlier cohort by construction, and does not exist in a v6
results root at all.

The sample is redrawn **per epoch**, which is what makes ``equal`` a share of
the source distribution rather than one fixed extra subset.  The redraw happens
in a sampler rather than in the dataset, because a ``DataLoader`` calls
``iter(sampler)`` exactly once per epoch: that is the only hook that fires at
epoch boundaries without the trainer having to know anything about it.

Determinism
-----------
The sampler is seeded from the run seed, the round and the client, and advances
its own epoch counter, so a rerun of the same (seed, round, client) draws the
same source images in the same order - and two different clients in the same
round draw different ones.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler

#: The modes ``--source-share`` understands.  ``off`` is the default and leaves
#: every loader exactly as it was.
SOURCE_SHARE_MODES = ("off", "equal", "full")


def old_book_rows(book, writers, fold: int, part: str):
    """The rows one part of one fold of the old book holds, sorted."""
    rows: list = []
    for writer in writers:
        rows.extend(book.part(fold, writer, part))
    return sorted(rows)


def old_train_dataset(provider, book, writers, fold: int, batch_size: int = 256):
    """
    The old data's fold-k **train** partition, as a dataset to share from.

    Args:
        provider: Dataset provider over the packed cache.
        book: The old-data fold book.
        writers: The old-data writers; ``None`` means everyone the book covers.
        fold: Which fold's train rows.  The run's own fold, so the shared data
            and the client's data come from the same split of the world.
        batch_size: Batch size of the intermediate loader.

    Returns:
        The dataset, or ``None`` when there is nothing to share.
    """
    from federated_outlier_adaptation.training.evaluate import rows_loader

    writers = list(writers) if writers else list(book.writers)
    rows = old_book_rows(book, writers, fold, "train")
    loader = rows_loader(provider, rows, batch_size)
    return None if loader is None else loader.dataset


def source_pool_dataset(provider, batch_size: int, loader_seed: Optional[int] = None):
    """
    The legacy source pool: the frozen 3% writer split's training half.

    Kept so a v4 or v5 results root reproduces exactly what it did.  A v6 run
    passes its old book instead and goes through :func:`old_train_dataset`; see
    the module docstring for why the two are not the same population.

    Returns:
        The dataset, or ``None`` when the source population has no training data.
    """
    train_loader, _, _ = provider.build_dataset(
        provider.global_client_ids(),
        train_rate=0.6,
        eval_rate=0.4,
        batch_size=batch_size,
        loader_seed=loader_seed,
    )
    return None if train_loader is None else train_loader.dataset


def share_size(
    mode: str,
    client_size: int,
    source_size: int,
    cap: Optional[int] = None,
    multiplier: float = 1.0,
) -> int:
    """
    How many source images join one epoch of one client's training set.

    Args:
        mode: One of :data:`SOURCE_SHARE_MODES`.
        client_size: The client's own training samples.
        source_size: Size of the source training pool.
        cap: Optional upper bound, applied to either mode.  The ``full`` mode is
            otherwise the whole pool, which on this dataset is two orders of
            magnitude more data per client-epoch than the client itself holds
            and costs accordingly; a documented cap keeps the arm affordable and
            is recorded with the run.
        multiplier: How many times the client's own training volume the
            ``equal`` mode draws.  1.0 is the balanced mixture; the
            access-to-old-data study sweeps 1, 2, 4 and 8.  Ignored by ``full``,
            which is the whole pool by definition.

    Returns:
        The number of source images per epoch; 0 when the mode is ``off``.
        Never more than the pool holds - asking for eight times a client's data
        when the pool is smaller than that gives the pool, and the info block
        records both numbers so a saturated arm is visible rather than silent.
    """
    if mode not in SOURCE_SHARE_MODES:
        raise ValueError(f"Unknown source-share mode {mode!r}; expected one of {SOURCE_SHARE_MODES}")
    if float(multiplier) <= 0:
        raise ValueError(f"The source-share multiplier must be positive; got {multiplier!r}")
    if mode == "off" or source_size <= 0 or client_size <= 0:
        return 0
    wanted = int(round(client_size * float(multiplier))) if mode == "equal" else source_size
    wanted = min(wanted, source_size)
    if cap:
        wanted = min(wanted, int(cap))
    return int(wanted)


def sampler_seed(run_seed: Optional[int], round_index: int, client_id: str) -> int:
    """
    A seed that is a pure function of the run, the round and the client.

    The client id goes through a digest rather than :func:`hash`, whose salt
    changes between processes - and every client trains in a worker process.
    """
    digest = hashlib.sha256(
        f"{run_seed}|{round_index}|{client_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


class SourceShareSampler(Sampler):
    """
    Indices of one epoch: every client sample plus a fresh source draw.

    The concatenated dataset is ``[client..., source...]``, so source index
    ``i`` is ``client_size + i`` here.

    Args:
        client_size: Number of client samples (all of them, every epoch).
        source_size: Size of the source pool to draw from.
        share: Source samples per epoch; ``share >= source_size`` takes the pool
            whole.
        seed: Base seed; the epoch counter is added to it on every pass.
    """

    def __init__(self, client_size: int, source_size: int, share: int, seed: int):
        self.client_size = int(client_size)
        self.source_size = int(source_size)
        self.share = max(0, int(share))
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.client_size + self.share

    #: Multipliers of the (seed, epoch) mix below - the constants of a
    #: well-tested 64-bit linear congruential generator.
    _SEED_MULTIPLIER = 6364136223846793005
    _EPOCH_MULTIPLIER = 1442695040888963407

    def epoch_seed(self, epoch: int) -> int:
        """
        The generator seed of one epoch.

        The epoch is *mixed* into the base seed rather than added to it.  Adding
        would make two clients whose base seeds differ by one draw the same
        images one epoch apart - the sort of collision that produces a quietly
        wrong experiment rather than a crash.
        """
        return (
            self.seed * self._SEED_MULTIPLIER + int(epoch) * self._EPOCH_MULTIPLIER
        ) % (2 ** 63)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch_seed(self._epoch))
        self._epoch += 1

        order = list(range(self.client_size))
        if self.share and self.source_size:
            if self.share >= self.source_size:
                drawn = list(range(self.source_size))
            else:
                drawn = torch.randperm(self.source_size, generator=generator)[
                    : self.share
                ].tolist()
            order.extend(self.client_size + index for index in drawn)

        shuffled = torch.randperm(len(order), generator=generator).tolist()
        return iter([order[position] for position in shuffled])


def build_shared_loader(
    client_dataset,
    source_dataset,
    mode: str,
    batch_size: int,
    seed: int,
    cap: Optional[int] = None,
    multiplier: float = 1.0,
) -> Tuple[Optional[DataLoader], Dict[str, Any]]:
    """
    A training loader over the client's data plus a per-epoch source share.

    Args:
        client_dataset: The client's own training dataset.
        source_dataset: The source writers' training dataset.
        mode: One of :data:`SOURCE_SHARE_MODES`.
        batch_size: Batch size of the produced loader.
        seed: Seed of the per-epoch draw; see :func:`sampler_seed`.
        cap: Optional bound on the source images per epoch.

    Returns:
        ``(loader, info)``.  With ``mode="off"`` - or with nothing to draw from -
        the loader is ``None``, telling the caller to keep the loader it already
        has, and ``info`` records why.
    """
    client_size = len(client_dataset) if client_dataset is not None else 0
    source_size = len(source_dataset) if source_dataset is not None else 0
    share = share_size(mode, client_size, source_size, cap, multiplier)

    info: Dict[str, Any] = {
        "mode": mode,
        "multiplier": float(multiplier),
        "client_samples": client_size,
        "source_pool": source_size,
        "source_per_epoch": share,
        # What the multiplier asked for, next to what the pool could give: an
        # arm that saturated is a different experiment from one that did not.
        "requested_per_epoch": (
            int(round(client_size * float(multiplier))) if mode == "equal" else source_size
        ),
        "cap": int(cap) if cap else None,
        "samples_per_epoch": client_size + share,
    }
    if share <= 0:
        return None, info

    combined = ConcatDataset([client_dataset, source_dataset])
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=SourceShareSampler(client_size, source_size, share, seed),
    )
    return loader, info


# --------------------------------------------------------------------------- #
# Monitoring without training: blending the old data into the *validation* set
# --------------------------------------------------------------------------- #
def blend_counts(client_size: int, old_size: int, rho: float) -> int:
    """
    How many old-data rows join a client's validation set.

    ``rho`` is the share of the blended set that comes from the old book, so
    ``n_old = n_client * rho / (1 - rho)``, bounded by what the old set holds.
    ``rho = 0.5`` gives equal counts, which is the case the study runs: with
    equal counts the accuracy over the blend is the arithmetic mean of the two
    accuracies, so "a 50/50 blend" is literally what the number is.

    Raises:
        ValueError: ``rho`` outside ``[0, 1)``.  ``rho = 1`` would be an
            all-old validation set with no client rows in it, which is not a
            blend and is not what any arm asks for.
    """
    rho = float(rho)
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"The validation blend must lie in [0, 1); got {rho!r}")
    if rho == 0.0 or client_size <= 0 or old_size <= 0:
        return 0
    return int(min(old_size, round(client_size * rho / (1.0 - rho))))


def blended_val_loader(
    client_val_loader,
    old_val_dataset,
    rho: float,
    batch_size: int,
    seed: int,
) -> Tuple[Optional[DataLoader], Dict[str, Any]]:
    """
    A validation loader that watches the old data without training on it.

    This is the monitoring-only arm of the access-to-old-data study: the client
    trains on its own rows alone - no old row ever enters a gradient - but the
    criterion the early-stopping rule reads is a blend of how the client is
    doing and how the old population is doing.  It is the cheapest possible form
    of access: the model is allowed to *look* at the old data, not to learn from
    it.

    The old rows come from the old book's **validation** partition, never its
    test partition, so the preservation measurement stays untouched.

    Args:
        client_val_loader: The client's own validation loader.
        old_val_dataset: The old book's fold-k validation rows.
        rho: Share of the blended set drawn from the old data; 0.5 is equal
            counts.
        batch_size: Batch size of the produced loader.
        seed: Seed of the subsample, so the blend is the same on every rerun.

    Returns:
        ``(loader, info)``.  The loader is ``None`` when there is nothing to
        blend, telling the caller to keep the loader it already has.
    """
    client_dataset = None if client_val_loader is None else client_val_loader.dataset
    client_size = len(client_dataset) if client_dataset is not None else 0
    old_size = len(old_val_dataset) if old_val_dataset is not None else 0
    wanted = blend_counts(client_size, old_size, rho)

    info: Dict[str, Any] = {
        "blend": float(rho),
        "client_val_samples": client_size,
        "old_val_pool": old_size,
        "old_val_samples": wanted,
        "blended_samples": client_size + wanted,
    }
    if wanted <= 0:
        return None, info

    generator = torch.Generator()
    generator.manual_seed(int(seed) % (2 ** 63))
    drawn = torch.randperm(old_size, generator=generator)[:wanted].tolist()
    from torch.utils.data import Subset

    combined = ConcatDataset([client_dataset, Subset(old_val_dataset, sorted(drawn))])
    # Evaluation, so no shuffling: the accuracy over a set does not depend on
    # the order it is walked in, and a fixed order keeps reruns identical.
    return DataLoader(combined, batch_size=batch_size, shuffle=False), info
