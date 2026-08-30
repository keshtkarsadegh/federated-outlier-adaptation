"""
Fold books: the cross-validation splits, written down once and reused forever.

A seed is a promise that a split can be recomputed; a **fold book** is the split
itself, on disk.  The difference matters here because the same folds are read by
every later experiment - the shipped model, the outlier study, the access-to-old-
data arms - often months apart and always in different processes.  A seed keeps
that promise only as long as nothing about the splitting code, the sample order
or the library versions changes; an index file keeps it unconditionally.

Layout
------
One ``.npz`` and nothing else:

``assignment``  ``int8 [folds, N]`` - for every fold, what each row of the packed
                cache is: 0 train, 1 validation, 2 test, and
                :data:`OUTSIDE` for a row belonging to a writer this book does
                not cover.
``metadata``    a JSON blob: the writers, the rates, the seed, the per-fold
                counts and the provenance of the cache it was built against.

One byte per row per fold - the whole 814k-image dataset over five folds is four
megabytes - so a book can cover every writer without a second thought, and
reading it back is a single ``np.load``.

The split itself
----------------
Per writer, per fold, **stratified by label**: each class's rows are shuffled
with a seed derived from (book seed, fold, writer) and cut at the two rates.  A
class with too few rows to fill a part contributes nothing to it, which is the
same truncating rule the on-the-fly splits use - so a writer that cannot form a
training split here is exactly the writer that could not form one there.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Parts of a split, in the order they are numbered in ``assignment``.
PARTS = ("train", "val", "test")

#: Marker for a row whose writer this book does not cover.
OUTSIDE = np.int8(-1)

#: Folds a book holds unless asked otherwise.
DEFAULT_FOLDS = 5

#: Default 60/20/20.
DEFAULT_TRAIN_RATE = 0.6
DEFAULT_EVAL_RATE = 0.2

#: File suffix of a written book.
SUFFIX = ".foldbook.npz"


def _writer_seed(seed: int, fold: int, writer: str) -> int:
    """A split seed that is a pure function of the book, the fold and the writer."""
    import hashlib

    digest = hashlib.sha256(f"{seed}|{fold}|{writer}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def share_holdout(
    remaining: int,
    eval_rate: float,
    test_rate: float,
    generator: random.Random,
) -> Tuple[int, int]:
    """
    Split a class's non-training rows between validation and test.

    Why this is not ``int(count * eval_rate)``
    ------------------------------------------
    The obvious rule - take ``floor(c * 0.2)`` for validation and give test
    whatever is left - is badly biased whenever classes are small, and at 62
    classes with roughly 3.6 images per class per writer they always are.  For a
    class of four, ``floor(4 * 0.2) == 0``: validation gets nothing and test gets
    both held-out rows.  Pooled over a population that produces a
    45.7 / 5.6 / 48.8 split rather than 60/20/20, and - much worse - validation
    is then drawn *only* from a writer's well-represented classes while test is
    dominated by its rare ones.  Measured on realistic counts: 0 % of validation
    but 85 % of test came from classes with four or fewer examples.  The two are
    not exchangeable at all, and a model scores far lower on test than on
    validation for reasons that have nothing to do with the model.

    Instead the held-out rows are shared **in proportion to the two rates**, with
    the leftover row going one way or the other at random.  That keeps every
    existing idiom intact - ``0.6/0.4`` still means "no test set",
    ``0.0/0.0`` still means "everything is test", ``0.6/0.0`` still means "no
    validation" - while ``0.6/0.2`` finally produces two exchangeable halves.

    Args:
        remaining: Rows of this class that are not training rows.
        eval_rate, test_rate: The nominal shares of the two held-out parts.
        generator: The per-writer, per-fold random source.

    Returns:
        ``(n_val, n_test)``.
    """
    if remaining <= 0:
        return 0, 0
    total = eval_rate + test_rate
    if total <= 0:
        return 0, remaining
    exact = remaining * (eval_rate / total)
    n_val = int(exact)
    # The one row the floors leave over goes to whichever part its fractional
    # part favours, and on a tie - the 0.2/0.2 case, which is the common one -
    # to either, with no standing preference.
    if remaining - n_val > int(remaining - exact):
        fraction = exact - n_val
        if fraction > 0.5 or (fraction == 0.5 and generator.random() < 0.5):
            n_val += 1
    return n_val, remaining - n_val


def split_rows(
    rows_by_label: Dict[int, List[int]],
    seed: int,
    train_rate: float = DEFAULT_TRAIN_RATE,
    eval_rate: float = DEFAULT_EVAL_RATE,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Cut one writer's rows into train / validation / test, stratified by label.

    The **training** rows are ``floor(c * train_rate)`` per class, exactly as
    they have always been: that keeps this function's train partition identical
    to what it produced before, so a model already trained against a book stays
    valid when the book is rebuilt.  What changed is the held-out rows, which
    are now shared evenly between validation and test instead of validation
    taking ``floor(c * eval_rate)`` and test taking the rest - see
    :func:`share_holdout` for why that mattered so much.

    ``eval_rate`` is kept in the signature because callers pass it and it is
    recorded with the book, but the even share makes it redundant for any rate
    that leaves validation and test the same nominal size.

    Args:
        rows_by_label: ``{label: [row, ...]}`` of one writer.
        seed: Seed of the per-label shuffle.
        train_rate, eval_rate: The rates; see above.

    Returns:
        Three sorted row lists.
    """
    generator = random.Random(seed)
    # A SEPARATE stream for the tie-breaks.  Drawing them from ``generator``
    # would advance the same sequence the per-class shuffles come from, so
    # every class after the first would shuffle differently and the training
    # rows would move - which is exactly what must not happen, because models
    # already trained against a book have to stay valid when it is rebuilt.
    tie_breaker = random.Random(seed ^ 0x5F5F5F5F)

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for label in sorted(rows_by_label):
        rows = list(rows_by_label[label])
        generator.shuffle(rows)
        count = len(rows)
        n_train = int(count * train_rate)
        n_val, _ = share_holdout(
            count - n_train,
            eval_rate,
            max(0.0, 1.0 - train_rate - eval_rate),
            tie_breaker,
        )
        train += rows[:n_train]
        val += rows[n_train : n_train + n_val]
        test += rows[n_train + n_val :]
    return sorted(train), sorted(val), sorted(test)


class FoldBook:
    """
    A written-down cross-validation split.

    Args:
        assignment: ``int8 [folds, N]`` part index per row per fold.
        metadata: The book's provenance; see the module docstring.
        writer_index: ``int32 [N]`` writer of each row, from the packed cache -
            kept so per-writer lookups need no second pass over the cache.
    """

    def __init__(self, assignment: np.ndarray, metadata: Dict[str, Any], writer_index=None):
        self.assignment = np.asarray(assignment, dtype=np.int8)
        self.metadata = dict(metadata)
        self.writers: List[str] = list(metadata.get("writers") or [])
        self._writer_index = writer_index
        self._rows_by_writer: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ shape
    @property
    def folds(self) -> int:
        """Number of folds the book holds."""
        return int(self.assignment.shape[0])

    @property
    def rows(self) -> int:
        """Rows of the cache the book was built against."""
        return int(self.assignment.shape[1])

    def covers(self, writer: str) -> bool:
        """
        Whether this book holds a split for a client.

        A merged client is covered when **every one of its members** is: a
        bundle half of whose writers are missing is not a client this book can
        split, and answering ``True`` would let a caller build a training set
        out of the half that happens to be there.
        """
        from federated_outlier_adaptation.data.merged_clients import members

        known = set(self.writers)
        return all(member in known for member in members(writer))

    # ------------------------------------------------------------------ lookup
    def _writer_rows(self) -> Dict[str, np.ndarray]:
        if self._rows_by_writer is None:
            grouped: Dict[str, List[int]] = defaultdict(list)
            table = self.metadata.get("writer_table") or self.writers
            for row, index in enumerate(np.asarray(self._writer_index)):
                grouped[table[int(index)]].append(row)
            self._rows_by_writer = {
                writer: np.asarray(rows, dtype=np.int64)
                for writer, rows in grouped.items()
            }
        return self._rows_by_writer

    def part(self, fold: int, writer: str, part: str) -> List[int]:
        """
        The rows of one writer's one part in one fold.

        Args:
            fold: 1-based fold number.
            writer: Writer id.
            part: One of :data:`PARTS`.

        Returns:
            Sorted row indices into the packed cache; empty when the book does
            not cover the writer or the part came out empty.
        """
        if part not in PARTS:
            raise ValueError(f"Unknown part {part!r}; expected one of {PARTS}")
        if not 1 <= int(fold) <= self.folds:
            raise ValueError(f"Fold {fold} is outside 1..{self.folds}")
        from federated_outlier_adaptation.data.merged_clients import members

        # A CLIENT MAY BE SEVERAL WRITERS. Its rows are the union of theirs, per
        # part - each writer keeps its own split and nothing is re-cut, so a
        # merged client holds exactly what its members hold and no row moves
        # between partitions. A plain id has one member: itself.
        wanted = PARTS.index(part)
        by_writer = self._writer_rows()
        found = []
        for member in members(writer):
            rows = by_writer.get(member)
            if rows is None or rows.size == 0:
                continue
            assigned = self.assignment[int(fold) - 1, rows]
            found.extend(int(row) for row in rows[assigned == wanted])
        return sorted(found)

    def split(self, fold: int, writers: Sequence[str]) -> Dict[str, List[int]]:
        """The three parts of one fold, pooled over several writers."""
        pooled: Dict[str, List[int]] = {part: [] for part in PARTS}
        for writer in writers:
            for part in PARTS:
                pooled[part].extend(self.part(fold, writer, part))
        return {part: sorted(rows) for part, rows in pooled.items()}

    def counts(self, fold: int) -> Dict[str, int]:
        """How many rows each part of one fold holds."""
        assigned = self.assignment[int(fold) - 1]
        return {part: int((assigned == index).sum()) for index, part in enumerate(PARTS)}

    # -------------------------------------------------------------------- i/o
    @classmethod
    def load(cls, path, writer_index=None) -> "FoldBook":
        """Read a book written by :func:`write_fold_book`."""
        path = Path(path)
        payload = np.load(path, allow_pickle=False)
        metadata = json.loads(str(payload["metadata"]))
        index = payload["writer_index"] if "writer_index" in payload else writer_index
        return cls(payload["assignment"], metadata, writer_index=index)


def build_fold_book(
    dataset,
    writers: Optional[Iterable[str]] = None,
    folds: int = DEFAULT_FOLDS,
    seed: int = 42,
    train_rate: float = DEFAULT_TRAIN_RATE,
    eval_rate: float = DEFAULT_EVAL_RATE,
    tag: Optional[str] = None,
) -> FoldBook:
    """
    Materialise the folds of a set of writers.

    Args:
        dataset: A :class:`~federated_outlier_adaptation.data.nist28.Nist28Dataset`.
        writers: Writers to cover; ``None`` covers every writer of the dataset.
        folds: How many folds to draw.
        seed: Base seed; every writer's every fold derives its own from it.
        train_rate, eval_rate: The 60/20/20 rates.
        tag: Free-text label stored with the book.

    Returns:
        The book, ready to be written.
    """
    cache = dataset.cache
    samples = dataset.writer_samples()
    covered = sorted(samples) if writers is None else sorted(set(writers))
    missing = [writer for writer in covered if writer not in samples]
    if missing:
        raise ValueError(f"These writers are not in the dataset: {missing[:5]}")

    total_rows = len(cache)
    assignment = np.full((int(folds), total_rows), OUTSIDE, dtype=np.int8)

    per_fold_counts = []
    unsplittable: Dict[str, List[str]] = {}
    for fold in range(1, int(folds) + 1):
        empty_here: List[str] = []
        for writer in covered:
            rows_by_label: Dict[int, List[int]] = defaultdict(list)
            for row, label in samples[writer]:
                rows_by_label[label].append(row)
            train, val, test = split_rows(
                rows_by_label,
                seed=_writer_seed(seed, fold, writer),
                train_rate=train_rate,
                eval_rate=eval_rate,
            )
            for index, rows in enumerate((train, val, test)):
                if rows:
                    assignment[fold - 1, np.asarray(rows, dtype=np.int64)] = index
            if not train or not val:
                empty_here.append(writer)
        if empty_here:
            unsplittable[str(fold)] = empty_here
        counts = {
            part: int((assignment[fold - 1] == index).sum())
            for index, part in enumerate(PARTS)
        }
        per_fold_counts.append(counts)

    metadata = {
        "tag": tag,
        "folds": int(folds),
        "seed": int(seed),
        "train_rate": float(train_rate),
        "eval_rate": float(eval_rate),
        "writers": covered,
        "writer_table": list(cache.writer_ids),
        "rows": total_rows,
        "fold_counts": per_fold_counts,
        "unsplittable": unsplittable,
        "cache_dir": str(dataset.cache_dir),
        "classes": dataset.classes,
    }
    return FoldBook(assignment, metadata, writer_index=cache.writer_index)


def write_fold_book(book: FoldBook, path) -> Path:
    """
    Write a book to ``path``, adding :data:`SUFFIX` when it has none.

    Returns:
        The file written.
    """
    path = Path(path)
    if not path.name.endswith(SUFFIX):
        path = path.with_name(path.name + SUFFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        assignment=book.assignment,
        writer_index=np.asarray(book._writer_index, dtype=np.int32),
        metadata=json.dumps(book.metadata),
    )
    return path
