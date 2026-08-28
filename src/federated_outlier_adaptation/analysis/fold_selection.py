"""
Choosing one model out of a cross-validation run, and writing it down.

Five folds produce five models.  Exactly one of them is the artefact everything
downstream is built on - g-init, and later g-0 - so which one, and on what
evidence, has to be a recorded decision rather than a convention.

The rule
--------
**The fold with the highest validation accuracy wins.**  Validation, never test:
the test halves of these folds are what the study reports its numbers on, and a
model chosen by them would make every later comparison self-congratulatory.
Ties break on the lower fold number, so the choice is deterministic.

What is written
---------------
``<root>/<name>_model``           the chosen fold's checkpoint, copied
``<root>/<name>_selection.json``  every fold's validation and test accuracy, the
                                  mean and spread across folds, which fold won,
                                  where its checkpoint came from and the SHA-256
                                  of what was written

The spread is the number the paper reports next to the mean - five folds of one
configuration, not five configurations - so it is computed here rather than left
to be recovered from five separate metric files later.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from federated_outlier_adaptation.logging_utils import NistLogger

#: Where one fold of a cross-validated training leaves its metrics.
METRICS_NAME = "global_metrics.json"

#: ...and its checkpoint.
MODEL_NAME = "global_model"


def fold_dir(root, prefix: str, fold: int) -> Path:
    """The directory one fold of a cross-validated training wrote into."""
    return Path(root) / f"{prefix}_fold{int(fold)}"


def read_fold(root, prefix: str, fold: int) -> Optional[Dict[str, Any]]:
    """
    One fold's validation and test accuracy, or ``None`` when it has not run.

    The validation accuracy is the best epoch's, taken from the convergence
    block the training records; a run without one falls back to the maximum of
    the per-epoch series, which is the same quantity.
    """
    directory = fold_dir(root, prefix, fold)
    metrics_path = directory / "global_results" / METRICS_NAME
    if not metrics_path.is_file():
        return None
    try:
        with open(metrics_path) as handle:
            metrics = json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return None

    convergence = metrics.get("convergence") or {}
    validation = convergence.get("best_val_accuracy")
    if validation is None:
        series = [v for v in (metrics.get("val_accuracies") or []) if isinstance(v, (int, float))]
        validation = max(series) if series else None

    return {
        "fold": int(fold),
        "val_accuracy": validation,
        "test_accuracy": metrics.get("test_accuracy"),
        "epochs_run": convergence.get("epochs_run"),
        "best_epoch": convergence.get("best_epoch"),
        "stopped_early": convergence.get("stopped_early"),
        "metrics_path": str(metrics_path),
        "model_path": str(directory / MODEL_NAME),
    }


def read_evaluations(path, folds: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """
    Per-fold validation and test accuracy from an ``evaluate-book`` file.

    The file is ``{tag: payload}`` and each payload carries its own ``fold`` and
    ``part``, so the tags themselves are free-form.  This is what makes a
    selection recomputable without retraining: the checkpoints are unchanged,
    only the rows they are scored on were wrong.

    Returns:
        ``{fold: {"val_accuracy": …, "test_accuracy": …}}``.
    """
    with open(Path(path)) as handle:
        stored = json.load(handle)
    if not isinstance(stored, dict):
        raise ValueError(f"{path} is not an evaluations file.")

    wanted = set(int(fold) for fold in folds)
    found: Dict[int, Dict[str, Any]] = {}
    for payload in stored.values():
        if not isinstance(payload, dict):
            continue
        fold, part = payload.get("fold"), payload.get("part")
        if fold is None or part not in ("val", "test") or int(fold) not in wanted:
            continue
        found.setdefault(int(fold), {})[f"{part}_accuracy"] = payload.get("accuracy")
    return found


def select_best_fold(
    root,
    prefix: str,
    name: str,
    folds: Sequence[int] = (1, 2, 3, 4, 5),
    results_dir=None,
    evaluations=None,
) -> Dict[str, Any]:
    """
    Pick the best fold of a cross-validated training and persist it.

    Args:
        root: Where the per-fold directories live.
        prefix: Their common prefix, e.g. ``"ginit"`` for ``ginit_fold1``.
        name: Name of the artefact to write, e.g. ``"ginit"``.
        folds: Fold numbers to consider.
        results_dir: Where the artefact is written; defaults to ``root``.
        evaluations: An ``evaluate-book`` file to take the per-fold validation
            and test accuracies from, instead of the metrics each training run
            wrote.  This is the recompute path: when the *evaluation* was wrong
            but the checkpoints were not, the folds are re-scored and re-selected
            without anything being retrained.

    Returns:
        The selection payload, which is also written next to the model.

    Raises:
        FileNotFoundError: No fold has produced metrics yet.
    """
    from federated_outlier_adaptation.utils.provenance import sha256_of

    root = Path(root)
    destination = Path(results_dir) if results_dir else root

    recomputed = read_evaluations(evaluations, folds) if evaluations else {}

    entries: List[Dict[str, Any]] = []
    for fold in folds:
        entry = read_fold(root, prefix, fold)
        if entry is None and int(fold) in recomputed:
            # The training's own metrics may be absent or, as here, simply not
            # the numbers we now trust; the checkpoint is what matters.
            entry = {
                "fold": int(fold),
                "val_accuracy": None,
                "test_accuracy": None,
                "model_path": str(fold_dir(root, prefix, fold) / MODEL_NAME),
            }
        if entry is None:
            continue
        if int(fold) in recomputed:
            entry = dict(entry)
            entry["reported_val_accuracy"] = entry.get("val_accuracy")
            entry["reported_test_accuracy"] = entry.get("test_accuracy")
            entry.update(recomputed[int(fold)])
            entry["source"] = "recomputed"
        entries.append(entry)

    scored = [e for e in entries if e["val_accuracy"] is not None]
    if not scored:
        missing = [f"{prefix}_fold{fold}" for fold in folds]
        raise FileNotFoundError(
            f"No fold of {prefix!r} under {root} has a validation accuracy yet "
            f"(looked for {missing}). Run the folds before selecting one."
        )

    # Highest validation accuracy; ties to the lower fold number.
    best = max(scored, key=lambda e: (e["val_accuracy"], -e["fold"]))

    val_values = [e["val_accuracy"] for e in scored]
    test_values = [e["test_accuracy"] for e in scored if e["test_accuracy"] is not None]

    source = Path(best["model_path"])
    if not source.is_file():
        raise FileNotFoundError(
            f"Fold {best['fold']} reported a validation accuracy but its "
            f"checkpoint is not at {source}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{name}_model"
    shutil.copyfile(source, target)

    payload = {
        "name": name,
        "prefix": prefix,
        "rule": "highest validation accuracy; ties to the lower fold",
        "accuracy_source": "recomputed" if recomputed else "training metrics",
        "selected_fold": best["fold"],
        "folds": entries,
        "val_mean": mean(val_values),
        "val_spread": pstdev(val_values) if len(val_values) > 1 else 0.0,
        "test_mean": mean(test_values) if test_values else None,
        "test_spread": (pstdev(test_values) if len(test_values) > 1 else 0.0)
        if test_values
        else None,
        "n_folds": len(scored),
        "source_model": str(source),
        "model": str(target),
        "sha256": sha256_of(target),
    }
    selection_path = destination / f"{name}_selection.json"
    with open(selection_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    NistLogger.info(
        f"{name}: fold {best['fold']} of {len(scored)} wins on validation "
        f"({best['val_accuracy']:.4f}); val mean {payload['val_mean']:.4f} "
        f"+/- {payload['val_spread']:.4f} -> {target}"
    )
    return payload
