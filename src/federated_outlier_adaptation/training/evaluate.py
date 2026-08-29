"""
Scoring a saved model on one part of one fold.

Stage 3 reports two numbers per fold of g-0, and they are the study's two
headline axes:

* **preservation** - g-0's accuracy on the *old data's* test rows, the reference
  every later method is measured against;
* **adaptation gap** - g-0's accuracy on the *outlier cohort's* test rows, which
  is what the whole study exists to close.

Both are read out of fold books, so both are computed on rows the model in
question never trained on, and both are reproducible from the files alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from federated_outlier_adaptation.logging_utils import NistLogger


def rows_dataset(dataset, rows, labels):
    """
    A torch dataset over explicit cache rows, whatever the modality.

    The book addresses samples by row, and every book-driven command - scoring,
    per-fold evaluation, the isolated and centralized arms - ends up here. It
    used to construct an image dataset directly, which is what confined the
    whole fold-book apparatus to NIST. Now it asks the dataset how to realise
    its own rows and only falls back to the packed-image path when the dataset
    does not say.

    Args:
        dataset: The provider's dataset.
        rows: Row indices, already filtered to ones the dataset knows.
        labels: Label per row, in the same order. A dataset that recovers its
            own labels from the row may
            ignore them.
    """
    build = getattr(dataset, "rows_dataset", None)
    if build is not None:
        return build(rows, labels)

    from federated_outlier_adaptation.data.datasets import (
        MemmapDigitDataset,
        build_transform,
    )

    return MemmapDigitDataset(
        dataset.cache, rows, labels, build_transform(dataset.cache.resolution)
    )


def rows_loader(provider, rows, batch_size: int = 256, shuffle: bool = False):
    """
    An evaluation loader over explicit cache rows.

    Args:
        provider: Dataset provider over the packed cache.
        rows: Row indices; rows whose writer the dataset does not know are
            dropped, so a caller may pass a book's part verbatim.
        batch_size: Batch size.
        shuffle: Whether to shuffle - off for evaluation, on for training.

    Returns:
        A ``DataLoader``, or ``None`` when there is nothing to load.
    """
    from torch.utils.data import DataLoader

    dataset = provider.dataset
    labels = {
        row: label
        for rows_of_writer in dataset.writer_samples().values()
        for row, label in rows_of_writer
    }
    kept = [row for row in rows if row in labels]
    if not kept:
        return None
    return DataLoader(
        rows_dataset(dataset, kept, [labels[row] for row in kept]),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def accuracy_of(model, loader, device=None) -> Optional[float]:
    """Accuracy of ``model`` over ``loader``; ``None`` when there is nothing."""
    import torch

    if loader is None:
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            correct += int(model(images).max(1)[1].eq(targets).sum())
            total += int(targets.size(0))
    return correct / total if total else None


def evaluate_on_book(
    provider,
    model,
    book,
    fold: int,
    writers: Sequence[str],
    part: str = "test",
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Accuracy of ``model`` on one part of one fold, pooled and per writer.

    Args:
        provider: Dataset provider over the packed cache.
        model: The model to score.
        book: The fold book the rows come from.
        fold: Which fold.
        writers: Whose rows to score on.
        part: ``"train"``, ``"val"`` or ``"test"``.
        batch_size: Evaluation batch size.

    Returns:
        The pooled accuracy, the sample count, and the per-writer accuracies.
    """
    import torch
    from torch.utils.data import DataLoader

    dataset = provider.dataset
    samples = dataset.writer_samples()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    per_writer: Dict[str, float] = {}
    pooled_correct = pooled_total = 0

    from federated_outlier_adaptation.data.merged_clients import members

    for writer in writers:
        # A merged client ("w1+w2") is scored as one client on the union of its
        # members' rows of this partition, and reported under the merged id -
        # it is one participant, so it is one row of the per-client column.
        parts = members(writer)
        rows = [row for member in parts for row in book.part(fold, member, part)]
        labels = {}
        for member in parts:
            labels.update(dict(samples.get(member, [])))
        rows = sorted(row for row in rows if row in labels)
        if not rows:
            continue
        loader = DataLoader(
            rows_dataset(dataset, rows, [labels[row] for row in rows]),
            batch_size=batch_size,
            shuffle=False,
        )
        correct = total = 0
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                correct += int(model(images).max(1)[1].eq(targets).sum())
                total += int(targets.size(0))
        per_writer[writer] = correct / total if total else 0.0
        pooled_correct += correct
        pooled_total += total

    accuracy = pooled_correct / pooled_total if pooled_total else None
    NistLogger.info(
        f"fold {fold} {part}: {accuracy if accuracy is None else round(accuracy, 4)} "
        f"over {pooled_total} images from {len(per_writer)} writers"
    )
    return {
        "fold": int(fold),
        "part": part,
        "accuracy": accuracy,
        "samples": pooled_total,
        "writers": len(per_writer),
        "per_writer": per_writer,
    }


def write_evaluation(payload: Dict[str, Any], path) -> Path:
    """Write an evaluation payload, merging into an existing file if there is one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            with open(path) as handle:
                existing = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[str(payload.get("tag") or payload.get("fold"))] = payload
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2)
    return path
