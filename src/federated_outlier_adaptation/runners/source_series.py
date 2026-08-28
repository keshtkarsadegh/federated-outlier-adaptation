"""
Where the per-round *source* series is measured, and what to do when there is nowhere.

Both federated runners score the server model once a round on a "global" set and
record it next to the clients number.  That set used to be one thing and one
thing only: the writers named in ``results/writer_split.json``, the frozen 3%
source population of the published pipeline.  Two problems with that, and the
first one is fatal.

**It is not optional.**  ``provider.global_client_ids()`` opens that file, so a
results root without one cannot start a federated run at all - the loop dies in
setup, before the first round, with a ``FileNotFoundError``.  The v6 protocol
draws its populations with ``select-outliers`` and ``draw-old-data`` and never
writes a writer split, so every v6 federated run was unstartable.

**It is not the old data.**  The 3% split is cut across *all* writers, and the
v6 cohort is cut by score across all writers too, so the two can overlap: a
cohort writer can sit in the source set and be counted as "preserved knowledge"
while the federation is actively training on it.  It is also the expensive
evaluation of the round, at roughly 25k images.

So the series follows the old-data book when there is one: fold k's *test* rows,
where k is the fold the run itself was split by, at roughly 6k rows.  That makes
it a cheap, honest, clearly-labelled **diagnostic** - it says *when* something
happened over the rounds.  It is not the preservation figure.  The preservation
figure is the five-fold set computed on the final server model by
:mod:`federated_outlier_adaptation.training.final_eval`, and nothing here
replaces it.

The three cases, in order:

``book``
    an old-data book was supplied - fold k's test rows, and its validation rows
    for the source-validation series.
``writer_split``
    no book, but the results root has the frozen split - the published
    behaviour, bit-for-bit, so every v4 and v5 run is unchanged.
``none``
    neither - a warning, and the run proceeds with no source series rather than
    dying in setup over a diagnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from federated_outlier_adaptation.logging_utils import NistLogger


def _book_loaders(
    provider,
    old_book: str,
    old_clients_file: Optional[str],
    batch_size: int,
    fold: Optional[int],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """The old-data book's fold-k test and validation rows."""
    from federated_outlier_adaptation.data.fold_book import FoldBook
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.training.evaluate import rows_loader

    path = Path(old_book)
    if not path.is_file():
        raise FileNotFoundError(
            f"No old-data fold book at {path}. The per-round source series of a "
            "run without a writer split is read off it."
        )
    book = FoldBook.load(path)
    writers = list(load_client_pool(old_clients_file)[0]) if old_clients_file else list(book.writers)
    # A run always names its fold; without one there is no k to take, and fold 1
    # is recorded rather than guessed at silently.
    chosen = int(fold) if fold is not None else 1

    def rows_of(part: str):
        rows: list[int] = []
        for writer in writers:
            rows.extend(book.part(chosen, writer, part))
        return sorted(rows)

    test_rows, val_rows = rows_of("test"), rows_of("val")
    test_loader = rows_loader(provider, test_rows, batch_size)
    val_loader = rows_loader(provider, val_rows, batch_size)
    info = {
        "source": "old_book",
        "old_book": str(path),
        "fold": chosen,
        "fold_given": fold is not None,
        "writers": len(writers),
        "test_samples": len(test_rows),
        "val_samples": len(val_rows),
        "diagnostic": True,
    }
    NistLogger.info(
        f"Per-round source series: old-data book fold {chosen}, "
        f"{len(test_rows)} test rows from {len(writers)} writers. "
        "This is a diagnostic; the reported preservation figure is the "
        "five-fold evaluation of the final server model."
    )
    return test_loader, val_loader, info


def _writer_split_loaders(provider, batch_size: int, loader_seed) -> Tuple[Any, Any, Dict[str, Any]]:
    """The published path: the frozen 3% source population of this results root."""
    global_writers = provider.global_client_ids()
    _, _, test_loader = provider.build_dataset(
        global_writers,
        train_rate=0.0,
        eval_rate=0.0,
        batch_size=batch_size,
        loader_seed=loader_seed,
    )
    # The source-side validation split: the global writers' 40% evaluation part,
    # built with the same split seed the global model was trained with, so
    # validation-based selection never touches the test numbers.
    _, val_loader, _ = provider.build_dataset(
        global_writers,
        train_rate=0.6,
        eval_rate=0.4,
        batch_size=batch_size,
        loader_seed=loader_seed,
    )
    return test_loader, val_loader, {"source": "writer_split", "writers": len(global_writers)}


def source_loaders(
    provider,
    batch_size: int,
    loader_seed=None,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    fold: Optional[int] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    The per-round source test and validation loaders, and where they came from.

    Args:
        provider: Dataset provider over the packed cache.
        batch_size: Evaluation batch size.
        loader_seed: Split seed of the legacy path.
        old_book: Path to the old-data fold book, when the run has one.
        old_clients_file: The old-data writer list, or ``None`` for everyone the
            book covers.
        fold: The fold the run was split by; the book's fold k is scored.

    Returns:
        ``(test loader, validation loader, info)``.  Both loaders are ``None``
        when there is neither a book nor a writer split, in which case the run
        keeps going without the series - it is a diagnostic, and a diagnostic
        must not be able to kill a job that has GPU-hours to spend.
    """
    if old_book:
        return _book_loaders(provider, old_book, old_clients_file, batch_size, fold)

    try:
        return _writer_split_loaders(provider, batch_size, loader_seed)
    except FileNotFoundError as missing:
        NistLogger.warning(
            f"No per-round source series: this results root has no writer split "
            f"({missing}) and no --old-book was given. The run proceeds without "
            "it; the series is a diagnostic and the reported preservation "
            "figure comes from the final-model evaluation instead."
        )
        return None, None, {"source": "none", "reason": str(missing)}
