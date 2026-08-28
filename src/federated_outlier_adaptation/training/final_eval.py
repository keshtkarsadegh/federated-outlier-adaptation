"""
The three things the final server model is asked, once the rounds are over.

Stage 5 of the v6 protocol trains a plain federated model on the outlier
cohort, and the per-round curves it logs are not what the paper reports.  A
curve is measured against whatever the loop happened to have loaded that round;
the reported numbers are measured against the fold books, on the **final**
server model, after the last aggregation.  That is what this module produces.

Three categories, and why each is separate
------------------------------------------
``clients``
    the pooled accuracy over all twenty cohort writers' fold-k **test** rows,
    plus the same model's accuracy on each writer's test rows on its own.  The
    pooled figure is the adaptation headline; the per-writer column is what
    shows whether the federation lifted the cohort or lifted three writers and
    left seventeen where they were.  A mean of a pooled number cannot tell those
    two apart.

``old``
    the preservation axis: the model's accuracy on the old data's test rows,
    measured on **every fold of the old-data book separately**.  Five numbers,
    reported with their mean and standard deviation.  The five folds are five
    partitions of one population, so the spread across them is the error bar on
    preservation - merging them into a single set would throw exactly that away
    and report a false precision.

The runner also keeps a per-round source series, which is cheap and useful for
seeing *when* forgetting happened.  It is not this number: it is measured on the
pre-book source notion of "the global writers", and the preservation figure the
paper reports is the five-fold set computed here.

Everything is read off the books, so every number is on rows the model never
trained on, and every number is reproducible from the files alone.
"""

from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from federated_outlier_adaptation.logging_utils import NistLogger


def resolve_folds(book, folds: Any = "all") -> List[int]:
    """
    Which folds of ``book`` to evaluate.

    ``"all"`` (the default) is every fold it holds; an integer, or a sequence of
    them, is those folds - kept so a cheap run can ask for one.
    """
    if folds is None or (isinstance(folds, str) and folds.lower() == "all"):
        return list(range(1, book.folds + 1))
    if isinstance(folds, (list, tuple)):
        return [int(fold) for fold in folds]
    return [int(folds)]


def old_data_evaluation(
    provider,
    model,
    old_book,
    writers: Optional[Sequence[str]] = None,
    folds: Any = "all",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    The model's accuracy on each fold of the old-data book's test rows.

    Args:
        provider: Dataset provider over the packed cache.
        model: The final server model.
        old_book: Fold book of the old data.
        writers: The old-data writers; defaults to everyone the book covers.
        folds: ``"all"`` (the default) or specific folds.
        batch_size: Evaluation batch size.

    Returns:
        The per-fold accuracies and sample counts, with their mean and
        population standard deviation.
    """
    from federated_outlier_adaptation.training.evaluate import evaluate_on_book

    writers = list(writers) if writers else list(old_book.writers)
    chosen = resolve_folds(old_book, folds)

    per_fold: Dict[str, Optional[float]] = {}
    samples: Dict[str, int] = {}
    for fold in chosen:
        result = evaluate_on_book(
            provider, model, old_book, fold=fold, writers=writers,
            part="test", batch_size=batch_size,
        )
        per_fold[str(fold)] = result["accuracy"]
        samples[str(fold)] = result["samples"]

    measured = [value for value in per_fold.values() if value is not None]
    return {
        "folds": chosen,
        "accuracies": per_fold,
        "samples": samples,
        "mean": mean(measured) if measured else None,
        "sd": (pstdev(measured) if len(measured) > 1 else 0.0) if measured else None,
        "n": len(measured),
        "writers": len(writers),
    }


def final_model_evaluation(
    provider,
    model,
    cohort_book=None,
    fold: Optional[int] = None,
    clients: Optional[Sequence[str]] = None,
    old_book=None,
    old_writers: Optional[Sequence[str]] = None,
    old_folds: Any = "all",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Score the final server model on the cohort's fold and on the old data.

    Either half may be absent - a run without a cohort book still reports the
    old-data folds, and a run without an old book still reports the cohort -
    so this never turns a finished run into a failed one over a missing path.

    Args:
        provider: Dataset provider over the packed cache.
        model: The final server model, after the last aggregation.
        cohort_book: Fold book of the outlier cohort.
        fold: Which of its folds this run trained on.
        clients: The cohort writers; defaults to everyone the book covers.
        old_book: Fold book of the old data.
        old_writers: The old-data writers.
        old_folds: ``"all"`` (the default) or specific folds.
        batch_size: Evaluation batch size.

    Returns:
        The payload: the pooled and per-client cohort figures, and the
        five-fold old-data set.
    """
    from federated_outlier_adaptation.training.evaluate import evaluate_on_book

    payload: Dict[str, Any] = {"fold": None if fold is None else int(fold)}

    if cohort_book is not None and fold is not None:
        writers = list(clients) if clients else list(cohort_book.writers)
        result = evaluate_on_book(
            provider, model, cohort_book, fold=fold, writers=writers,
            part="test", batch_size=batch_size,
        )
        payload["clients"] = {
            "accuracy": result["accuracy"],
            "samples": result["samples"],
            "writers": result["writers"],
            # The column, not a mean of it: which writers the federation
            # actually moved is the whole point of scoring them separately.
            "per_client": result["per_writer"],
        }
        NistLogger.info(
            f"Final server model, cohort fold {fold}: "
            f"{_fmt(result['accuracy'])} over {result['samples']} test images "
            f"from {result['writers']} writers."
        )

    if old_book is not None:
        payload["old"] = old_data_evaluation(
            provider, model, old_book, writers=old_writers,
            folds=old_folds, batch_size=batch_size,
        )
        NistLogger.info(
            "Final server model, old data: "
            f"{_fmt(payload['old']['mean'])} +/- {_fmt(payload['old']['sd'])} "
            f"over {payload['old']['n']} fold(s)."
        )

    return payload


def evaluate_final_model(
    provider,
    model,
    clients: Optional[Sequence[str]] = None,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    old_folds: Any = "all",
    batch_size: int = 256,
) -> Optional[Dict[str, Any]]:
    """
    The worker-side entry point: open the books from paths and score the model.

    The cohort book and the fold come from the run's own environment - they are
    what the whole run was split by - so only the old-data half needs naming.
    Returns ``None`` when neither half is available, which is what an ordinary
    run without books does, so nothing about such a run changes.

    Args:
        provider: Dataset provider over the packed cache.
        model: The final server model.
        clients: The cohort writers the run trained on.
        old_book: Path to the old-data fold book, or ``None``.
        old_clients_file: Path to the old-data writer list, or ``None`` for
            everyone the old book covers.
        old_folds: ``"all"`` (the default) or specific folds.
        batch_size: Evaluation batch size.
    """
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.data.fold_book import FoldBook
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.utils.seeding import configured_fold

    cohort_book = None
    fold = configured_fold()
    book_path = config.fold_book()
    if book_path is not None and Path(book_path).is_file() and fold is not None:
        cohort_book = FoldBook.load(book_path)

    old = None
    old_writers = None
    if old_book:
        path = Path(old_book)
        if not path.is_file():
            raise FileNotFoundError(
                f"No old-data fold book at {path}. The preservation numbers of "
                "a stage-5 run are read off it."
            )
        old = FoldBook.load(path)
        if old_clients_file:
            old_writers = load_client_pool(old_clients_file)[0]

    if cohort_book is None and old is None:
        return None

    payload = final_model_evaluation(
        provider,
        model,
        cohort_book=cohort_book,
        fold=fold,
        clients=clients,
        old_book=old,
        old_writers=old_writers,
        old_folds=old_folds,
        batch_size=batch_size,
    )
    if book_path is not None:
        payload["fold_book"] = str(book_path)
    if old_book:
        payload["old_book"] = str(old_book)
    return payload


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"
