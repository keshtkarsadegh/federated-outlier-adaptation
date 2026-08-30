"""
The packed 28x28, 62-class NIST SD19 cache and the dataset built on it.

The cache is the by-writer form of the FEMNIST task: every scan of the
``by_write`` archive, converted with
:mod:`federated_outlier_adaptation.data.emnist_convert`, labelled through the
``by_class`` checksum log (see
:mod:`federated_outlier_adaptation.data.sd19_labels`) and stored in four files:

``nist28_images.npy``
    ``uint8`` ``[N, 28, 28]`` - the images.
``nist28_labels.npy``
    ``uint8`` ``[N]`` - the label index, in the by-class order of
    :data:`~federated_outlier_adaptation.data.sd19_labels.CLASS_ORDER`.
``nist28_writers.npy``
    ``int32 [N]`` - the row's writer, as an index into the writer table.
``nist28_index.json``
    The writer table, the class map, the applied conversion, the counts and the
    checksum of the archive it was built from.

Four files, whatever the sample count: the archive is streamed and never
extracted, and no per-image file is ever written.  Rows are ordered by
(writer, path), so every writer's samples are contiguous.

The reader exposes ``image``/``array``/``height``/``width``, which is the
interface :class:`~federated_outlier_adaptation.data.datasets.MemmapDigitDataset`
reads through, so the dataset layer is the published one with a different
backing store.
"""

from __future__ import annotations

import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from federated_outlier_adaptation.data import sd19_labels
from federated_outlier_adaptation.data.emnist_convert import (
    TARGET_SIZE,
    conversion_info,
    convert_image,
)
from federated_outlier_adaptation.logging_utils import NistLogger

class _FixedChoice:
    """
    A stand-in random source that always resolves a tie the same way.

    ``trainable_writers`` asks a yes/no question about a writer, so it must not
    depend on a coin toss: an odd held-out row is counted as validation's, which
    is the optimistic reading and therefore the right one for "could this writer
    be trained at all".
    """

    @staticmethod
    def random() -> float:
        return 0.0


_NO_TIES = _FixedChoice()


#: File names of the packed cache.
IMAGES_NAME = "nist28_images.npy"
LABELS_NAME = "nist28_labels.npy"
WRITERS_NAME = "nist28_writers.npy"
INDEX_NAME = "nist28_index.json"

#: Images timed at the start of a build to project the total runtime.
TIMING_SAMPLE = 2000


def _decode(zip_file: zipfile.ZipFile, name: str) -> np.ndarray:
    """Decode one archive member into a ``uint8`` grayscale array."""
    with zip_file.open(name) as handle:
        return np.asarray(Image.open(handle).convert("L"), dtype=np.uint8)


def build_cache(
    zip_path,
    out_dir,
    by_class_log=None,
    by_write_log=None,
    classes: str = "all",
    size: int = TARGET_SIZE,
    timing_sample: int = TIMING_SAMPLE,
    progress_every: int = 20000,
    log=NistLogger.info,
) -> dict:
    """
    Build the packed 28x28 cache by streaming ``by_write.zip``.

    The two checksum logs are downloaded next to the archive when they are not
    given and not already present; they are a few tens of megabytes of text and
    are the only way to attach a by-class label to a by-write path without
    fetching the second archive.

    Args:
        zip_path: Path of ``by_write.zip``.  Never extracted.
        out_dir: Destination of the four cache files.
        by_class_log, by_write_log: The checksum logs; ``None`` resolves them
            next to the archive and downloads them if missing.
        classes: ``"all"`` (62 classes) or ``"digits"`` (the ablation).
        size: Edge length of the converted images.
        timing_sample: Images after which the projected total runtime is logged,
            so a job says early what it will cost.
        progress_every: Progress log interval in images.
        log: Callable used for progress output.

    Returns:
        A summary dict with the row count, the writer count, the elapsed time
        and the four output paths.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if by_class_log is None or by_write_log is None:
        resolved_class = zip_path.parent / sd19_labels.BY_CLASS_MD5_NAME
        resolved_write = zip_path.parent / sd19_labels.BY_WRITE_MD5_NAME
        if not (resolved_class.is_file() and resolved_write.is_file()):
            log(f"Fetching the SD19 checksum logs into {zip_path.parent}")
            resolved_class, resolved_write = sd19_labels.download_md5_logs(zip_path.parent)
        by_class_log = by_class_log or resolved_class
        by_write_log = by_write_log or resolved_write

    labels = sd19_labels.build_label_map(by_class_log, by_write_log, classes=classes, log=log)
    if not labels:
        raise ValueError(
            "The checksum logs produced no labelled by_write path; check that "
            f"{by_class_log} and {by_write_log} are the SD19 logs."
        )

    zip_file = zipfile.ZipFile(zip_path)
    present = set(zip_file.namelist())
    missing = [path for path in labels if path not in present]
    if missing:
        log(f"WARNING: {len(missing)} labelled paths are absent from the archive")
    paths = sorted(path for path in labels if path in present)

    # Rows are grouped by writer so that a client's samples are contiguous.
    paths.sort(key=lambda path: (sd19_labels.writer_of(path) or "", path))
    writer_ids = sorted({sd19_labels.writer_of(path) or "" for path in paths})
    writer_row = {writer: index for index, writer in enumerate(writer_ids)}

    total = len(paths)
    log(f"{total} images, {len(writer_ids)} writers, {classes} classes, target {size}x{size}")

    images = np.lib.format.open_memmap(
        out_dir / IMAGES_NAME, mode="w+", dtype=np.uint8, shape=(total, size, size)
    )
    label_array = np.zeros(total, dtype=np.uint8)
    writer_array = np.zeros(total, dtype=np.int32)

    started = time.time()
    projected = None
    for row, path in enumerate(paths):
        images[row] = convert_image(_decode(zip_file, path), size=size)
        label_array[row] = labels[path]
        writer_array[row] = writer_row[sd19_labels.writer_of(path) or ""]

        if timing_sample and row + 1 == timing_sample:
            elapsed = time.time() - started
            projected = elapsed / timing_sample * total
            log(
                f"timing sample: {timing_sample} images in {elapsed:.1f}s "
                f"-> projected {projected / 60.0:.1f} min for {total} images"
            )
        if progress_every and (row + 1) % progress_every == 0:
            elapsed = time.time() - started
            eta = elapsed / (row + 1) * (total - row - 1)
            log(f"{row + 1}/{total}  {elapsed:.0f}s  eta {eta:.0f}s")

    images.flush()
    del images
    np.save(out_dir / LABELS_NAME, label_array)
    np.save(out_dir / WRITERS_NAME, writer_array)
    zip_file.close()

    from federated_outlier_adaptation.utils.provenance import sha256_of

    seconds = time.time() - started
    counts = np.bincount(label_array, minlength=sd19_labels.CLASS_COUNTS[classes])
    meta = {
        "resolution": int(size),
        "classes": classes,
        "num_classes": int(sd19_labels.CLASS_COUNTS[classes]),
        "class_map": {str(k): v for k, v in sd19_labels.class_map(classes).items()},
        "conversion": conversion_info(size=size),
        "rows": int(total),
        "writers": writer_ids,
        "missing": len(missing),
        "class_counts": {str(index): int(value) for index, value in enumerate(counts)},
        # THE INDEX MUST BE REPRODUCIBLE, so nothing about *this* run goes in
        # it. Two builds of the same archive on two nodes produced identical
        # image, label and writer arrays and differed only here: one recorded
        # 574.99 seconds and the other 577.97. A cache that cannot be
        # checksummed is a cache whose integrity nobody can check, and a
        # spurious mismatch sends someone hunting a corruption that does not
        # exist. The elapsed time is still logged and still returned to the
        # caller; it is simply not part of the artefact.
        #
        # The archive is named by its hash rather than by where it happened to
        # sit, for the same reason: the hash is what identifies the release,
        # and an absolute path would make the index differ between machines
        # while describing identical data.
        "source_archive": Path(zip_path).name,
        "source_sha256": sha256_of(zip_path),
        "by_class_md5_log": Path(by_class_log).name,
        "by_write_md5_log": Path(by_write_log).name,
    }
    with open(out_dir / INDEX_NAME, "w") as handle:
        json.dump(meta, handle)

    log(f"done in {seconds:.0f}s -> {out_dir}")
    return {
        "rows": total,
        "writers": len(writer_ids),
        "classes": classes,
        "seconds": seconds,
        "projected_seconds": projected,
        "missing": len(missing),
        "images_path": str(out_dir / IMAGES_NAME),
        "labels_path": str(out_dir / LABELS_NAME),
        "writers_path": str(out_dir / WRITERS_NAME),
        "index_path": str(out_dir / INDEX_NAME),
    }


class Nist28Cache:
    """
    Read-only view over a packed 28x28 cache.

    ``image(row)`` returns a PIL ``L`` image and ``array(row)`` the raw
    ``uint8`` array, so the dataset layer can use the very same transform
    pipeline as the 128x128 backend.
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / INDEX_NAME) as handle:
            meta = json.load(handle)
        self.meta = meta
        self.resolution = int(meta.get("resolution", TARGET_SIZE))
        self.height = self.width = self.resolution
        self.stored_classes: str = str(meta.get("classes", "all"))
        self.writer_ids: List[str] = list(meta.get("writers") or [])
        self.class_map: Dict[int, str] = {
            int(key): value for key, value in (meta.get("class_map") or {}).items()
        }
        self._images = np.load(self.cache_dir / IMAGES_NAME, mmap_mode="r")
        self.labels = np.load(self.cache_dir / LABELS_NAME)
        self.writer_index = np.load(self.cache_dir / WRITERS_NAME)

    # ------------------------------------------------------------------ lookup
    @classmethod
    def open_if_available(cls, cache_dir) -> Optional["Nist28Cache"]:
        """Return a cache instance, or ``None`` when the files are absent."""
        cache_dir = Path(cache_dir)
        for name in (INDEX_NAME, IMAGES_NAME, LABELS_NAME, WRITERS_NAME):
            if not (cache_dir / name).is_file():
                return None
        try:
            return cls(cache_dir)
        except (OSError, ValueError, KeyError):  # pragma: no cover - broken cache
            return None

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def array(self, row: int) -> np.ndarray:
        """The raw ``uint8`` image of a row."""
        return np.asarray(self._images[row], dtype=np.uint8)

    def image(self, row: int) -> Image.Image:
        """The row as a PIL grayscale image."""
        return Image.fromarray(self.array(row), mode="L")


class Nist28Dataset:
    """
    Client-level dataset builder over the packed 28x28 cache.

    The split logic - ``random.seed(seed)``, per-label shuffle, cut at
    ``train_rate``/``eval_rate`` - is the one of
    :class:`~federated_outlier_adaptation.data.datasets.NistDataset`, so the two
    resolutions differ in their samples and not in how a client is split.

    Args:
        cache_dir: Directory holding the four cache files.
        classes: ``"all"`` keeps every class of the cache; ``"digits"`` restricts
            the samples to labels 0-9, which is the ablation of plan v4.
    """

    def __init__(self, cache_dir, classes: str = "all"):
        if classes not in sd19_labels.CLASS_SETS:
            raise ValueError(
                f"Unknown class set {classes!r}; expected one of {sd19_labels.CLASS_SETS}"
            )
        self.cache_dir = Path(cache_dir)
        self.classes = classes
        self._cache: Optional[Nist28Cache] = None
        self._writer_samples: Optional[Dict[str, List[tuple]]] = None
        self._sample_counts: dict = {}
        self._fold_book = None
        self._fold_book_loaded = False

    # ------------------------------------------------------------------ cache
    @property
    def cache(self) -> Nist28Cache:
        """The packed cache, opened once per instance."""
        if self._cache is None:
            cache = Nist28Cache.open_if_available(self.cache_dir)
            if cache is None:
                raise FileNotFoundError(
                    f"No packed 28x28 cache under {self.cache_dir}; build it with "
                    "'foa prepare-data --resolution 28 --classes all'."
                )
            self._cache = cache
            NistLogger.info(f"Using packed 28x28 NIST cache at {self.cache_dir}")
        return self._cache

    @property
    def num_classes(self) -> int:
        """Output units the cache's character set needs."""
        return sd19_labels.CLASS_COUNTS[self.classes]

    # -------------------------------------------------------------- fold book
    @property
    def fold_book(self):
        """
        The configured fold book, or ``None``.

        Opened once per instance.  A book plus a fold makes
        :meth:`build_dataset` read its splits off disk rather than re-deriving
        them, so the same rows are train, validation and test in every process
        that ever touches this fold.
        """
        if not self._fold_book_loaded:
            from federated_outlier_adaptation import config
            from federated_outlier_adaptation.data.fold_book import FoldBook

            self._fold_book_loaded = True
            path = config.fold_book()
            if path is not None and Path(path).is_file():
                self._fold_book = FoldBook.load(path)
                NistLogger.info(
                    f"Using the fold book at {path}: "
                    f"{len(self._fold_book.writers)} writers, "
                    f"{self._fold_book.folds} folds."
                )
            elif path is not None:
                raise FileNotFoundError(
                    f"No fold book at {path}. Build it with 'foa fold-book' "
                    "before pointing a run at it."
                )
        return self._fold_book

    def book_split(self, writers, fold):
        """
        The three parts of ``writers`` in ``fold``, from the book.

        Returns ``None`` when there is no book, no fold, or the book does not
        cover every requested writer - in which case the caller falls back to
        the seeded split and nothing about it changes.
        """
        book = self.fold_book
        if book is None or fold is None:
            return None
        if not all(book.covers(writer) for writer in writers):
            return None
        return book.split(fold, writers)

    # ----------------------------------------------------------------- writers
    def writer_samples(self) -> Dict[str, List[tuple]]:
        """``{writer: [(row, label), ...]}``, computed once per instance."""
        if self._writer_samples is None:
            cache = self.cache
            limit = self.num_classes
            grouped: Dict[str, List[tuple]] = defaultdict(list)
            writer_ids = cache.writer_ids
            for row, (writer, label) in enumerate(
                zip(cache.writer_index.tolist(), cache.labels.tolist())
            ):
                if label >= limit:
                    continue
                grouped[writer_ids[writer]].append((row, int(label)))
            self._writer_samples = dict(grouped)
        return self._writer_samples

    def all_writers(self) -> List[str]:
        """Every writer with at least one sample of the selected class set."""
        return sorted(self.writer_samples())

    def get_sample_count(self, writer_id) -> int:
        """Number of samples a writer contributes to aggregation weighting."""
        count = self._sample_counts.get(writer_id)
        if count is None:
            count = len(self.writer_samples().get(writer_id, []))
            self._sample_counts[writer_id] = count
        return count

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

        The arguments and the split are those of
        :meth:`~federated_outlier_adaptation.data.datasets.NistDataset.build_dataset`.
        """
        from torch.utils.data import DataLoader

        from federated_outlier_adaptation.data.datasets import (
            MemmapDigitDataset,
            build_transform,
        )
        from federated_outlier_adaptation.utils.seeding import make_generator

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
        transform = build_transform(self.cache.resolution)

        def stratified_split(data):
            from federated_outlier_adaptation.data.fold_book import share_holdout

            label_map = defaultdict(list)
            for row, label in data:
                label_map[label].append(row)

            test_rate = max(0.0, 1.0 - train_rate - eval_rate)
            # A separate stream for the tie-breaks: drawing them from the module
            # RNG would advance the sequence ``random.shuffle`` uses below, and
            # every class after the first would then be shuffled differently.
            tie_breaker = random.Random(hash((seed, "holdout")) & 0xFFFFFFFF)
            train, val, test = [], [], []
            for label, rows in label_map.items():
                random.shuffle(rows)
                n = len(rows)
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
                train.extend((row, label) for row in rows[:n_train])
                val.extend((row, label) for row in rows[n_train : n_train + n_val])
                test.extend((row, label) for row in rows[n_train + n_val :])
            return train, val, test

        def random_split(data):
            random.shuffle(data)
            n = len(data)
            n_train = int(n * train_rate)
            n_val = int(n * eval_rate)
            return data[:n_train], data[n_train : n_train + n_val], data[n_train + n_val :]

        split_fn = stratified_split if is_stratified else random_split

        def make_loader(data):
            if not data:
                return None
            rows = [row for row, _ in data]
            labels = [label for _, label in data]
            dataset = MemmapDigitDataset(self.cache, rows, labels, transform)
            generator = make_generator(loader_seed)
            if generator is None:
                return DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        # A fold book, when one is configured and covers these writers, *is* the
        # split: the rows it recorded are the split, and no rate or seed here
        # can move them.  This is what makes one fold mean the same thing in
        # every process that ever reads it.
        from federated_outlier_adaptation.utils.seeding import configured_fold

        booked = self.book_split(list(writers), configured_fold())
        if booked is not None:
            labels = dict(
                (row, label)
                for writer in writers
                for row, label in writer_samples.get(writer, [])
            )
            return tuple(
                make_loader([(row, labels[row]) for row in booked[part] if row in labels])
                for part in ("train", "val", "test")
            )

        data = []
        for writer in writers:
            data.extend(writer_samples.get(writer, []))
        if not data:
            return None, None, None
        train, val, test = split_fn(data)
        return make_loader(train), make_loader(val), make_loader(test)

    def class_names(self) -> Dict[int, str]:
        """The label -> character map of the cache, restricted to this class set."""
        limit = self.num_classes
        return {
            label: character
            for label, character in self.cache.class_map.items()
            if label < limit
        }

    def trainable_writers(self, train_rate: float = 0.6, eval_rate: float = 0.4) -> List[str]:
        """
        Writers whose local split yields both a train and a validation sample.

        :meth:`build_dataset` splits *per label* and truncates
        (``int(n_label * rate)``), so a writer holding a single sample in a
        class contributes nothing to that class's train and validation halves.
        A writer with one sample in *every* class therefore has both halves
        empty and cannot be trained at all.  That is a 62-class effect: the
        writer that held roughly eighteen samples per class in the ten-class
        setting holds about one here.

        The rule is a pure function of the manifest - it counts, it does not
        build loaders - so it is cheap enough to apply when a pool is written.

        Returns:
            The eligible writer ids, sorted.
        """
        from federated_outlier_adaptation.data.fold_book import share_holdout

        test_rate = max(0.0, 1.0 - train_rate - eval_rate)
        eligible = []
        for writer, rows in self.writer_samples().items():
            counts: Dict[int, int] = defaultdict(int)
            for _, label in rows:
                counts[label] += 1
            # The same arithmetic ``build_dataset`` performs, so this answers
            # "can this writer be trained?" about the split that will actually
            # be built rather than about a different one.
            n_train = sum(int(count * train_rate) for count in counts.values())
            n_eval = sum(
                share_holdout(
                    count - int(count * train_rate), eval_rate, test_rate, _NO_TIES
                )[0]
                for count in counts.values()
            )
            if n_train > 0 and n_eval > 0:
                eligible.append(writer)
        return sorted(eligible)

    # ------------------------------------------------------------- writer split
    def split_writers(self, global_size: float = 0.03, seed: int = 42):
        """
        Partition the writers into a local and a global (source) population.

        Identical in shape to
        :func:`~federated_outlier_adaptation.data.datasets.split_local_global_writers`:
        the writer ids are sorted, shuffled with ``seed`` and cut by
        ``train_test_split(test_size=global_size, random_state=42)``.

        Returns:
            ``(local_writers, global_writers)``.
        """
        writers = self.all_writers()
        random.seed(seed)
        random.shuffle(writers)
        local_writers, global_writers = train_test_split(
            writers, test_size=global_size, random_state=42
        )
        return list(local_writers), list(global_writers)

    # --------------------------------------------------------------- provenance
    def files(self) -> Dict[str, str]:
        """The cache files, for the provenance record."""
        return {
            "nist28_images": str(self.cache_dir / IMAGES_NAME),
            "nist28_labels": str(self.cache_dir / LABELS_NAME),
            "nist28_writers": str(self.cache_dir / WRITERS_NAME),
            "nist28_index": str(self.cache_dir / INDEX_NAME),
        }
