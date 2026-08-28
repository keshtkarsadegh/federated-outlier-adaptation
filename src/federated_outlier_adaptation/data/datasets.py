"""
NIST SD19 "by_write" dataset access.

Two interchangeable back-ends produce byte-identical tensors:

* :class:`NISTDigitDataset` reads the individual PNG files under ``data/by_write``
  (the original path, used when no cache is present);
* :class:`MemmapDigitDataset` reads from the packed cache built by
  :mod:`federated_outlier_adaptation.data.nist_cache`, which stores all
  images in a single ``.npy`` file and therefore needs two inodes instead of
  ~400 000.

:class:`NistDataset` picks the cache automatically when it exists and falls back
to the PNG reader otherwise.  Split logic, seeding and transform are unchanged
from the published experiments.
"""

from __future__ import annotations

import json
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from federated_outlier_adaptation import config
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.utils.eval_cache import GRAYSCALE_U8
from federated_outlier_adaptation.utils.seeding import make_generator

#: Normalisation constants of the project's image transform.
TRANSFORM_MEAN = (0.5,)
TRANSFORM_STD = (0.5,)


#: Edge length of the published NIST pipeline.
DEFAULT_SIZE = 128


def build_transform(size: int = DEFAULT_SIZE) -> transforms.Compose:
    """
    The single image transform used by every phase of the project.

    Args:
        size: Edge length the images are resized to.  128 is the published
            resolution; the 28x28 cache passes its own, in which case the
            ``Resize`` is a no-op on already converted images and only keeps the
            three-stage shape the evaluation cache recognises.
    """
    return transforms.Compose(
        [
            transforms.Resize((int(size), int(size))),
            transforms.ToTensor(),
            transforms.Normalize(TRANSFORM_MEAN, TRANSFORM_STD),
        ]
    )


def resize_stage(transform) -> Optional[transforms.Resize]:
    """
    The ``Resize`` of a standard project transform, or ``None``.

    The evaluation cache stores images in their ``uint8`` form and applies the
    ``ToTensor``/``Normalize`` tail as one batched operation.  That is only
    equivalent when the transform really is
    ``Resize -> ToTensor -> Normalize((0.5,), (0.5,))``; anything else - a
    different normalisation, an augmentation - makes this return ``None`` and
    the caller keeps the per-sample path.  The returned ``Resize`` is the very
    object the dataset uses, so applying it during materialisation cannot drift
    from what the loader did.
    """
    stages = getattr(transform, "transforms", None)
    if not stages or len(stages) != 3:
        return None
    resize, to_tensor, normalize = stages
    if not isinstance(resize, transforms.Resize):
        return None
    if not isinstance(to_tensor, transforms.ToTensor):
        return None
    if not isinstance(normalize, transforms.Normalize):
        return None
    if tuple(normalize.mean) != TRANSFORM_MEAN or tuple(normalize.std) != TRANSFORM_STD:
        return None
    return resize


def grayscale_payload(images, labels, resize) -> tuple[torch.Tensor, torch.Tensor, object]:
    """
    Stack PIL grayscale images into the cache's ``uint8`` payload.

    ``resize`` is applied while the image is still ``uint8`` - the same object,
    the same interpolation - so the batched ``ToTensor``/``Normalize`` that
    follows reproduces the loader's float tensors exactly.
    """
    arrays = [np.asarray(resize(image), dtype=np.uint8) for image in images]
    payload = torch.from_numpy(np.stack(arrays)) if arrays else torch.empty(0, dtype=torch.uint8)
    return payload, torch.as_tensor(list(labels), dtype=torch.long), GRAYSCALE_U8


class NISTDigitDataset(Dataset):
    """Reads one grayscale PNG per sample from disk."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def eval_payload(self):
        """Compact form of the whole dataset for the evaluation cache."""
        resize = resize_stage(self.transform)
        if resize is None:
            return None
        images = (Image.open(path).convert("L") for path in self.image_paths)
        return grayscale_payload(images, self.labels, resize)


class MemmapDigitDataset(Dataset):
    """
    Reads samples from the packed cache instead of individual PNG files.

    ``rows`` are indices into the cached array; the transform pipeline is the
    same object used by :class:`NISTDigitDataset`, so the produced tensors are
    identical.
    """

    def __init__(self, cache, rows, labels, transform=None):
        self.cache = cache
        self.rows = list(rows)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image = self.cache.image(self.rows[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def eval_payload(self):
        """
        Compact form of the whole dataset for the evaluation cache.

        The packed cache already stores ``uint8`` images, so the rows are simply
        unpacked once and stacked; the resize of the transform is applied to the
        same ``uint8`` images the loader would have resized.
        """
        resize = resize_stage(self.transform)
        if resize is None:
            return None
        arrays = [self.cache.array(row) for row in self.rows]
        height, width = self.cache.height, self.cache.width
        if tuple(resize.size) == (height, width):
            # The stored images already have the target size, which is the
            # normal case for this dataset; skip the round trip through PIL.
            payload = (
                torch.from_numpy(np.stack(arrays))
                if arrays
                else torch.empty(0, dtype=torch.uint8)
            )
            return payload, torch.as_tensor(list(self.labels), dtype=torch.long), GRAYSCALE_U8
        images = (Image.fromarray(array, mode="L") for array in arrays)
        return grayscale_payload(images, self.labels, resize)


class NistDataset:
    """
    Client-level dataset builder.

    Args:
        data_dir: Root that contains ``by_write/``.  Defaults to the configured
            :data:`~federated_outlier_adaptation.config.DATA_DIR`.
        labels_json: Manifest mapping relative image path -> digit label.
        cache_dir: Location of the packed cache.  When the cache is absent the
            PNG reader is used.
        use_cache: Set to False to force the PNG reader.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        labels_json: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.labels_json = Path(labels_json) if labels_json else config.DIGITS_LABELS_JSON
        self.cache_dir = Path(cache_dir) if cache_dir else config.CACHE_DIR
        self.use_cache = use_cache
        self._cache = None
        self._cache_loaded = False
        # The default NIST provider is a process-wide singleton and the drivers
        # fan out over threads, so several of them race for the first access to
        # the packed cache.  Opening it is guarded and the "loaded" flag is only
        # published once the cache object itself is in place.
        self._cache_lock = threading.Lock()
        self._label_lock = threading.Lock()
        self._label_dict: Optional[dict] = None
        self._writer_samples: Optional[dict] = None
        self._sample_counts: dict = {}

    # ------------------------------------------------------------------ cache
    @property
    def cache(self):
        """
        The packed cache if available, else ``None``.

        Opened once per instance and safe to reach from several threads: the
        instance is shared (``providers.nist.default_provider`` is a
        process-wide singleton) while the drivers run their jobs in a thread
        pool, so a naive "set the flag, then load" would let a second thread see
        ``_cache is None`` while the first is still opening the array.  That
        thread would silently build a PNG-backed dataset and fail on the first
        batch wherever the archive was never unpacked - which is everywhere,
        since unpacking it is what the cache exists to avoid.
        """
        if not self.use_cache:
            return None
        if not self._cache_loaded:
            with self._cache_lock:
                if not self._cache_loaded:
                    from federated_outlier_adaptation.data.nist_cache import NistDigitCache

                    cache = NistDigitCache.open_if_available(self.cache_dir)
                    self._cache = cache
                    self._cache_loaded = True
                    if cache is not None:
                        NistLogger.info(f"Using packed NIST cache at {self.cache_dir}")
        return self._cache

    # ------------------------------------------------------------------ labels
    def label_dict(self) -> dict:
        """The digit-label manifest, loaded once per instance."""
        if self._label_dict is None:
            with self._label_lock:
                if self._label_dict is None:
                    with open(self.labels_json, "r") as handle:
                        self._label_dict = json.load(handle)
        return self._label_dict

    def writer_samples(self) -> dict:
        """
        Relative image paths and labels grouped by writer id, computed once.

        The grouping is a pure function of the manifest, so caching it does not
        change any split: callers receive the same lists in the same order as
        the previous per-call grouping produced.
        """
        if self._writer_samples is None:
            labels = self.label_dict()  # acquires the label lock itself; take it before locking here
            with self._label_lock:
                if self._writer_samples is None:
                    grouped = defaultdict(list)
                    for rel_path, label in labels.items():
                        parts = rel_path.split("/")
                        if len(parts) < 3:
                            continue
                        grouped[parts[2]].append((rel_path, label))  # e.g. f0302_47
                    self._writer_samples = dict(grouped)
        return self._writer_samples

    def all_writers(self) -> List[str]:
        """Every writer id present in the manifest, sorted."""
        return sorted({path.split("/")[2] for path in self.label_dict() if path.count("/") >= 2})

    # ------------------------------------------------------------------- utils
    def get_sample_count(self, writer_id):
        """
        Return the number of training samples available for this writer/client.
        Used in aggregation weighting.
        """
        count = self._sample_counts.get(writer_id)
        if count is None:
            # The count is the size of the writer's full sample list; it does not
            # depend on the split, so it is computed once from the grouped manifest.
            count = len(self.writer_samples().get(writer_id, []))
            self._sample_counts[writer_id] = count
        return count

    def visualize_dataset_samples(self, data_loader, class_names=None, num_batches=1):
        for batch_idx, (images, labels) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            images = images[:8]  # Show up to 8 samples per batch
            labels = labels[:8]

            fig, axs = plt.subplots(1, len(images), figsize=(15, 3))
            for i, (img, label) in enumerate(zip(images, labels)):
                img = img.squeeze()  # Remove channel dim (1, H, W) -> (H, W)
                img = TF.to_pil_image((img * 0.5 + 0.5).clamp(0, 1))  # Undo normalization

                axs[i].imshow(img, cmap="gray")
                lbl = class_names[label.item()] if class_names else str(label.item())
                axs[i].set_title(f"Label: {lbl}")
                axs[i].axis("off")
            plt.show()
            NistLogger.info("Plotted the samples from dataset")

    # ------------------------------------------------------------------- build
    def build_dataset(
        self,
        writers,
        seed=42,
        train_rate=0.6,
        eval_rate=0.2,
        is_stratified=True,
        batch_size=64,
        loader_seed=None,
    ):
        """
        Build train/validation/test DataLoaders for one or more writers.

        The split itself is unchanged: samples are grouped per writer, shuffled
        with ``random.seed(seed)`` and cut at ``train_rate``/``eval_rate``.

        Args:
            loader_seed: Optional seed for the DataLoader shuffling generator.
                ``None`` (the default) keeps the historical unseeded behaviour.
        """
        if isinstance(writers, str):
            writers = [writers]

        writer_samples = self.writer_samples()

        # The 60/20/20 split of every client follows the configured fold, so a
        # fold is a different draw over the same clients rather than a different
        # client population.  Without one the seed is untouched and the split is
        # byte-for-byte the published one.
        from federated_outlier_adaptation.utils.seeding import (
            configured_fold,
            fold_seed,
        )

        random.seed(fold_seed(seed, configured_fold()))

        transform = build_transform()

        # Stratified split (preserve label balance)
        def stratified_split(data):
            from federated_outlier_adaptation.data.fold_book import share_holdout

            label_map = defaultdict(list)
            for path, label in data:
                label_map[label].append(path)

            test_rate = max(0.0, 1.0 - train_rate - eval_rate)
            # A separate stream for the tie-breaks: drawing them from the module
            # RNG would advance the sequence ``random.shuffle`` uses below, and
            # every class after the first would then be shuffled differently.
            tie_breaker = random.Random(hash((seed, "holdout")) & 0xFFFFFFFF)
            train, val, test = [], [], []
            for label, imgs in label_map.items():
                random.shuffle(imgs)
                n = len(imgs)
                n_train = int(n * train_rate)
            # The held-out rows are shared between validation and test in
            # proportion to the two rates rather than by two independent floors.
            # With 62 classes and a few images per class per writer, the floor
            # rule gave validation nothing from the small classes and test
            # everything, which made the two non-exchangeable; see
            # ``fold_book.share_holdout``.  Every existing rate idiom - 0.6/0.4,
            # 0.0/0.0, 0.6/0.0 - is unchanged.
                n_val, _ = share_holdout(
                    n - n_train, eval_rate, test_rate, tie_breaker
                )

                train.extend((img, label) for img in imgs[:n_train])
                val.extend((img, label) for img in imgs[n_train : n_train + n_val])
                test.extend((img, label) for img in imgs[n_train + n_val :])
            return train, val, test

        # Random split (no label balancing)
        def random_split(data):
            random.shuffle(data)
            n = len(data)
            n_train = int(n * train_rate)
            n_val = int(n * eval_rate)
            train = data[:n_train]
            val = data[n_train : n_train + n_val]
            test = data[n_train + n_val :]
            return train, val, test

        split_fn = stratified_split if is_stratified else random_split

        cache = self.cache

        def make_loader(data):
            if not data:
                return None
            rel_paths = [p for p, _ in data]
            labels = [l for _, l in data]
            if cache is not None and cache.has_all(rel_paths):
                dataset = MemmapDigitDataset(cache, cache.rows_for(rel_paths), labels, transform)
            else:
                if cache is not None:
                    # The archive is never unpacked, so a PNG fallback is a
                    # crash waiting for the first batch.  Say so here, where the
                    # cause is still visible.
                    NistLogger.warning(
                        f"The packed cache at {self.cache_dir} does not cover all "
                        f"{len(rel_paths)} requested images; falling back to the "
                        f"PNG reader under {self.data_dir}."
                    )
                image_paths = [self.data_dir / p for p in rel_paths]
                dataset = NISTDigitDataset(image_paths, labels, transform)
            generator = make_generator(loader_seed)
            if generator is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        data = []
        for wid in writers:
            data.extend(writer_samples.get(wid, []))
        if not data:
            return None, None, None
        train, val, test = split_fn(data)
        return make_loader(train), make_loader(val), make_loader(test)


def split_local_global_writers(
    global_size: float,
    seed: int = 42,
    labels_json: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    Partition the writers of the manifest into a local and a global set.

    Writers are read from ``digits_labels.json`` rather than from a directory
    glob, so the split no longer depends on the dataset being unpacked on disk.

    Args:
        global_size: Fraction of writers assigned to the global set.
        seed: Seed of the pre-shuffle (the subsequent ``train_test_split`` keeps
            its historical ``random_state=42``).
        labels_json: Manifest location; defaults to the configured one.
    """
    path = Path(labels_json) if labels_json else config.DIGITS_LABELS_JSON
    with open(path, "r") as handle:
        label_dict = json.load(handle)

    all_writers = {
        rel_path.split("/")[2] for rel_path in label_dict if rel_path.count("/") >= 2
    }

    random.seed(seed)
    all_writers = sorted(all_writers)
    random.shuffle(all_writers)

    local_writers, global_writers = train_test_split(
        all_writers, test_size=global_size, random_state=42
    )

    return local_writers, global_writers
