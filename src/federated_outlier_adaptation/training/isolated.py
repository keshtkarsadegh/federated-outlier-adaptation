"""
Isolated training: one private model per outlier client, and what it costs.

Stage 4 of the v6 protocol.  Each of the twenty cohort writers trains a model of
its own on its own data - no aggregation, no other client's gradients - which is
the arrangement the whole study is arguing against.  The point is to price it,
and the price only shows up if each private model is scored on three different
things:

``own``
    the client's own fold-k test rows.  This is the number isolation *wins* on:
    a model fitted to one writer's handwriting should be very good at that
    writer.
``union``
    every cohort client's fold-k test rows, pooled.  Each private model is
    scored on the whole cohort, which is where isolation fails: twenty models
    that are each excellent on one writer are, collectively, twenty models that
    are bad on nineteen.
``old``
    the old data's test rows - the preservation axis, i.e. what the private
    model still knows about the population the shipped model was built for.
    Measured on **every fold of the old-data book separately**: five numbers
    per client, reported with their mean and standard deviation.  Five
    evaluations rather than one merged set, because the folds are five
    partitions of the same population and the spread across them *is* the error
    bar on the preservation figure - merging them throws exactly that away.

Nothing is written but numbers
------------------------------
Twenty models per fold and init, over five folds and two inits, is two hundred
checkpoints nobody will ever load again.  They are evaluated while they are in
memory and then dropped; the file this writes holds the three accuracies per
client, the pooled figures and the provenance, and nothing else.

The pooled rung: one model over all of it
-----------------------------------------
``pooled=True`` is a different experiment in the same machinery.  Instead of
twenty private models it trains **one** model on the union of every cohort
writer's fold-k train rows, early-stopped on the union of their validation rows,
and then scores that single model on the same three sets.  It is the rung
between isolation and federation: what one model achieves when the privacy
constraint is simply lifted and all twenty writers' data sits in one place.  It
is the ceiling the federated arm is trying to reach without moving any data, so
the gap between them is the price of the constraint.

The three evaluations mean the same things they do above, with one simplification:
``own`` and ``union`` are now measured on the same model, so ``union`` *is* the
pooled adaptation figure and the per-client column says which writers that one
model actually serves.

One writer at a time, and the union that does not shrink with it
----------------------------------------------------------------
``only_client`` trains and scores a single cohort writer, so the hundred
(writer, fold) cells can run as a hundred short independent jobs.  The ``union``
set is deliberately *not* narrowed with it: it stays every cohort client's
fold-k test rows, because the question it answers - what one private model is
worth to the whole group - is meaningless against a group of one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence

from federated_outlier_adaptation.logging_utils import NistLogger

#: Initialisations a private model can start from.
INITIALISATIONS = ("global", "scratch")


def _pooled(entries: Sequence[tuple]) -> Optional[float]:
    """Sample-count weighted mean of ``[(accuracy, count), ...]``."""
    usable = [(a, c) for a, c in entries if a is not None and c]
    total = sum(count for _, count in usable)
    if not total:
        return None
    return sum(accuracy * count for accuracy, count in usable) / total


def _spread(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mean and population spread of the per-model figures."""
    usable = [v for v in values if v is not None]
    if not usable:
        return {"mean": None, "spread": None, "min": None, "max": None, "n": 0}
    return {
        "mean": mean(usable),
        "spread": pstdev(usable) if len(usable) > 1 else 0.0,
        "min": min(usable),
        "max": max(usable),
        "n": len(usable),
    }


def isolated_training(
    provider,
    clients: Sequence[str],
    cohort_book,
    fold: int,
    old_book,
    old_fold: Any = "all",
    old_clients: Optional[Sequence[str]] = None,
    only_client: Optional[str] = None,
    pooled: bool = False,
    init: str = "global",
    init_checkpoint: Optional[str] = None,
    epochs: int = 100,
    batch_size: int = 64,
    patience: Optional[int] = 10,
    min_epochs: int = 20,
    eval_batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Train one private model per client and score each on the three sets.

    Args:
        provider: Dataset provider over the packed cache.
        clients: The cohort writers.
        cohort_book: Fold book holding the cohort's splits.
        fold: Which of its folds this run trains on.
        old_book: Fold book of the old data.
        old_fold: ``"all"`` (the default) evaluates every fold of the old book
            separately; an integer evaluates that one fold, kept for
            compatibility with runs made before the five-fold rule.
        old_clients: The old-data writers; defaults to everyone that book covers.
        only_client: Train and score just this writer.  The union evaluation
            still spans every client in ``clients`` - see the module docstring.
        pooled: Train ONE model on the union of every client's fold-k train
            rows instead of one model per client - the centralized-on-outliers
            rung.  Mutually exclusive with ``only_client``.
        init: ``"global"`` starts every client from ``init_checkpoint``;
            ``"scratch"`` from a fresh initialisation.
        init_checkpoint: The shipped model, for ``init="global"``.
        epochs: Ceiling on local epochs; the convergence rule normally stops
            well before it.
        batch_size: Training batch size.
        patience, min_epochs: The convergence rule, applied per client - each
            one converges on its own handful of images.
        eval_batch_size: Batch size of the three evaluations.

    Returns:
        The payload: per-client rows, the pooled figures and the provenance.
    """
    import torch

    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
    from federated_outlier_adaptation.training.evaluate import accuracy_of, rows_loader
    from federated_outlier_adaptation.utils.provenance import sha256_of

    if init not in INITIALISATIONS:
        raise ValueError(f"Unknown init {init!r}; expected one of {INITIALISATIONS}")
    if init == "global" and not init_checkpoint:
        raise ValueError("init='global' needs the checkpoint every client starts from")

    clients = list(clients)
    if not clients:
        raise ValueError("Isolated training needs at least one client.")

    if pooled and only_client is not None:
        raise ValueError(
            "--pooled trains one model over every client; --only-client trains "
            "one client's private model. Pick one."
        )
    if only_client is not None and only_client not in clients:
        raise ValueError(
            f"--only-client {only_client!r} is not in the cohort "
            f"({len(clients)} writers)."
        )
    training_clients = [only_client] if only_client is not None else clients

    old_folds = (
        list(range(1, old_book.folds + 1))
        if isinstance(old_fold, str) and old_fold.lower() == "all"
        else [int(old_fold)]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The two evaluation sets that do not depend on the client: built once.
    union_rows: List[int] = []
    for client in clients:
        union_rows.extend(cohort_book.part(fold, client, "test"))
    union_loader = rows_loader(provider, sorted(union_rows), eval_batch_size)

    old_writers = list(old_clients) if old_clients else list(old_book.writers)
    old_loaders: Dict[int, Any] = {}
    old_sizes: Dict[int, int] = {}
    for candidate in old_folds:
        rows: List[int] = []
        for writer in old_writers:
            rows.extend(old_book.part(candidate, writer, "test"))
        loader = rows_loader(provider, sorted(rows), eval_batch_size)
        old_loaders[candidate] = loader
        old_sizes[candidate] = 0 if loader is None else len(loader.dataset)

    NistLogger.info(
        f"Isolated training, fold {fold}, init {init}: "
        f"{len(training_clients)} of {len(clients)} clients trained; "
        f"union test {0 if union_loader is None else len(union_loader.dataset)} "
        f"images; old test over folds {old_folds} "
        f"({sum(old_sizes.values())} images across them)."
    )

    # The starting weights, loaded once and copied per client.
    template = provider.make_model()
    checkpoint_info: Dict[str, Any] = {}
    if init == "global":
        path = Path(init_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {path}. Isolated training from the shipped "
                "model needs g-0 to exist."
            )
        template.load_state_dict(torch.load(path, map_location=device))
        checkpoint_info = {
            "init_checkpoint": str(path),
            "init_checkpoint_sha256": sha256_of(path),
        }

    if pooled:
        return _pooled_training(
            provider=provider,
            clients=clients,
            cohort_book=cohort_book,
            fold=fold,
            union_loader=union_loader,
            old_loaders=old_loaders,
            old_sizes=old_sizes,
            old_folds=old_folds,
            old_writers=old_writers,
            template=template,
            device=device,
            init=init,
            checkpoint_info=checkpoint_info,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            min_epochs=min_epochs,
            eval_batch_size=eval_batch_size,
        )

    # REFUSE A CLIENT THE BOOK DOES NOT COVER, AND ONLY THAT.
    #
    # Two different things look alike downstream and must not be treated alike.
    # A client the book covers but which this fold leaves too thin to train is
    # DATA: it is recorded, skipped, and the other clients carry the fold. A
    # client the book does not cover at all is a CONFIGURATION ERROR - the book
    # was built over the wrong writers, or over writers rather than over the
    # clients those writers make up - and it yields no rows for any fold.
    #
    # The second used to be silent. A hundred array elements of the multi-writer
    # study ran with merged ids the book had never heard of, trained nothing,
    # wrote "trained": 0 and exited zero. Nothing downstream could tell that
    # from a real result.
    uncovered = [client for client in training_clients if not cohort_book.covers(client)]
    if uncovered:
        raise ValueError(
            f"the fold book does not cover {len(uncovered)} of the "
            f"{len(training_clients)} client(s) requested: {', '.join(uncovered)}. "
            "It yields no rows for them in any fold, so this run would train on "
            "nothing and report success. Check that the book was built over the "
            "writers these clients are made of."
        )

    per_client: Dict[str, Dict[str, Any]] = {}
    own_entries: List[tuple] = []
    skipped: List[str] = []

    for index, client in enumerate(training_clients, start=1):
        train_rows = cohort_book.part(fold, client, "train")
        val_rows = cohort_book.part(fold, client, "val")
        test_rows = cohort_book.part(fold, client, "test")

        train_loader = rows_loader(provider, train_rows, batch_size, shuffle=True)
        val_loader = rows_loader(provider, val_rows, batch_size)
        if train_loader is None or val_loader is None:
            # A writer this fold cannot train is recorded, not silently dropped.
            NistLogger.warning(
                f"{client}: fold {fold} leaves it {len(train_rows)} train and "
                f"{len(val_rows)} validation rows; no private model is trained."
            )
            skipped.append(client)
            per_client[client] = {
                "skipped": True,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "test_samples": len(test_rows),
            }
            continue

        trainer = BaseTrainer(provider=provider, early_stopping=patience is not None)
        trainer.set_model(copy.deepcopy(template))
        trainer.train(train_loader, val_loader, epochs, patience, min_epochs)
        model = trainer.get_model()

        own_loader = rows_loader(provider, test_rows, eval_batch_size)
        own = accuracy_of(model, own_loader, device)

        # Every fold of the old book, separately.
        per_fold = {
            candidate: accuracy_of(model, loader, device)
            for candidate, loader in old_loaders.items()
        }
        measured = [value for value in per_fold.values() if value is not None]
        row = {
            "own": own,
            "union": accuracy_of(model, union_loader, device),
            "old_folds": {str(k): v for k, v in sorted(per_fold.items())},
            "old_mean": mean(measured) if measured else None,
            "old_sd": (pstdev(measured) if len(measured) > 1 else 0.0)
            if measured
            else None,
            # ``old`` is that mean under a short name, so the pooled block and
            # any reader has a single number to reach for.
            "old": mean(measured) if measured else None,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "test_samples": len(test_rows),
            "convergence": dict(getattr(trainer, "convergence", {}) or {}),
        }
        per_client[client] = row
        own_entries.append((own, len(test_rows)))
        NistLogger.info(
            f"[{index}/{len(clients)}] {client}: own={_fmt(own)} "
            f"union={_fmt(row['union'])} "
            f"old={_fmt(row['old_mean'])} +/- {_fmt(row['old_sd'])} "
            f"over {len(measured)} fold(s)"
        )

        # The model has now told us everything it is going to; drop it rather
        # than write two hundred checkpoints nobody will load again.
        del trainer, model

    trained = [row for row in per_client.values() if not row.get("skipped")]
    payload = {
        "stage": "isolated",
        "init": init,
        "fold": int(fold),
        "old_folds": old_folds,
        "old_fold": old_folds[0] if len(old_folds) == 1 else "all",
        "clients": clients,
        "trained_clients": training_clients,
        "only_client": only_client,
        "trained": len(trained),
        "skipped": skipped,
        "per_client": per_client,
        "pooled": {
            # Own is sample-weighted: the clients differ in how much test data
            # they have, and the question is per-image.
            "own_weighted": _pooled(own_entries),
            # Union and old are the same set for every model, so what matters is
            # the spread across models, not a weighting.
            "own": _spread([row.get("own") for row in trained]),
            "union": _spread([row.get("union") for row in trained]),
            "old": _spread([row.get("old") for row in trained]),
        },
        "union_test_samples": 0 if union_loader is None else len(union_loader.dataset),
        "old_test_samples": {str(k): v for k, v in sorted(old_sizes.items())},
        "old_writers": len(old_writers),
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping_patience": patience,
        "min_epochs": min_epochs,
        **checkpoint_info,
    }
    return payload


def _pooled_training(
    provider,
    clients: Sequence[str],
    cohort_book,
    fold: int,
    union_loader,
    old_loaders: Dict[int, Any],
    old_sizes: Dict[int, int],
    old_folds: List[int],
    old_writers: Sequence[str],
    template,
    device: str,
    init: str,
    checkpoint_info: Dict[str, Any],
    epochs: int,
    batch_size: int,
    patience: Optional[int],
    min_epochs: int,
    eval_batch_size: int,
) -> Dict[str, Any]:
    """
    Centralized-on-outliers: one model over the pooled cohort.

    The training set is the union of every client's fold-k **train** rows and
    the early-stopping set is the union of their **validation** rows, so the
    convergence rule is applied once to one model rather than twenty times to
    twenty.  The three evaluations are the ones the isolated arm uses, which is
    what makes the two rungs comparable line for line.

    Returns:
        The payload, in the same shape as the isolated one: a per-client column,
        the pooled figures, the five old-data folds and the provenance.
    """
    import copy

    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
    from federated_outlier_adaptation.training.evaluate import accuracy_of, rows_loader

    train_rows: List[int] = []
    val_rows: List[int] = []
    for client in clients:
        train_rows.extend(cohort_book.part(fold, client, "train"))
        val_rows.extend(cohort_book.part(fold, client, "val"))
    train_rows, val_rows = sorted(train_rows), sorted(val_rows)

    train_loader = rows_loader(provider, train_rows, batch_size, shuffle=True)
    val_loader = rows_loader(provider, val_rows, batch_size)
    if train_loader is None or val_loader is None:
        raise ValueError(
            f"The pooled cohort has {len(train_rows)} train and {len(val_rows)} "
            f"validation rows in fold {fold}; there is nothing to train on."
        )

    NistLogger.info(
        f"Centralized on the outliers, fold {fold}, init {init}: one model over "
        f"{len(train_rows)} pooled train rows from {len(clients)} writers, "
        f"early-stopped on {len(val_rows)} pooled validation rows."
    )

    trainer = BaseTrainer(provider=provider, early_stopping=patience is not None)
    trainer.set_model(copy.deepcopy(template))
    trainer.train(train_loader, val_loader, epochs, patience, min_epochs)
    model = trainer.get_model()

    # The per-client column: the one model, on each writer's own test rows.
    per_client: Dict[str, Dict[str, Any]] = {}
    own_entries: List[tuple] = []
    for client in clients:
        test_rows = cohort_book.part(fold, client, "test")
        own = accuracy_of(model, rows_loader(provider, test_rows, eval_batch_size), device)
        per_client[client] = {
            "own": own,
            "train_samples": len(cohort_book.part(fold, client, "train")),
            "val_samples": len(cohort_book.part(fold, client, "val")),
            "test_samples": len(test_rows),
        }
        own_entries.append((own, len(test_rows)))

    per_fold = {
        candidate: accuracy_of(model, loader, device)
        for candidate, loader in old_loaders.items()
    }
    measured = [value for value in per_fold.values() if value is not None]
    union = accuracy_of(model, union_loader, device)

    NistLogger.info(
        f"Centralized on the outliers: union={_fmt(union)} "
        f"old={_fmt(mean(measured) if measured else None)} +/- "
        f"{_fmt(pstdev(measured) if len(measured) > 1 else 0.0 if measured else None)} "
        f"over {len(measured)} fold(s)"
    )

    return {
        "stage": "centralized_outliers",
        "pooled_arm": True,
        "init": init,
        "fold": int(fold),
        "old_folds": old_folds,
        "old_fold": old_folds[0] if len(old_folds) == 1 else "all",
        "clients": list(clients),
        # One model over all of them, which is the whole point of this rung.
        "models": 1,
        "trained": 1,
        "skipped": [],
        "pooled_train_rows": len(train_rows),
        "pooled_val_rows": len(val_rows),
        "per_client": per_client,
        "pooled": {
            "own_weighted": _pooled(own_entries),
            "own": _spread([row.get("own") for row in per_client.values()]),
            # One model, so ``union`` is a single number and not a spread over
            # models: it *is* the pooled adaptation figure of this rung.
            "union": union,
        },
        "old_folds_accuracy": {str(k): v for k, v in sorted(per_fold.items())},
        "old_mean": mean(measured) if measured else None,
        "old_sd": (pstdev(measured) if len(measured) > 1 else 0.0) if measured else None,
        "old": mean(measured) if measured else None,
        "union_test_samples": 0 if union_loader is None else len(union_loader.dataset),
        "old_test_samples": {str(k): v for k, v in sorted(old_sizes.items())},
        "old_writers": len(old_writers),
        "convergence": dict(getattr(trainer, "convergence", {}) or {}),
        "epochs": epochs,
        "batch_size": batch_size,
        "early_stopping_patience": patience,
        "min_epochs": min_epochs,
        **checkpoint_info,
    }


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_payload(payload: Dict[str, Any], path) -> Path:
    """Write one ``(init, fold)`` result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    NistLogger.info(f"Isolated training -> {path}")
    return path
