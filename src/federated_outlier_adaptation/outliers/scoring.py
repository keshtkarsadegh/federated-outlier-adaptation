"""
Scoring every writer with g-init, and drawing the two populations from it.

Stage 2 of the v6 protocol.  Three decisions come out of one pass:

* **what each writer is worth under g-init**, measured only on data g-init never
  trained on;
* **the outlier cohort** - the worst writers that can actually be trained;
* **the old data** - a seeded random draw from everyone else.

The scoring rule, and why it is the only honest one here
--------------------------------------------------------
g-init is trained on the whole dataset, so *every* writer contributed training
images to it.  Scoring a writer on all of its data would therefore be scoring the
model on its own training set, and the resulting ranking would say more about
memorisation than about difficulty.

So each writer is scored on **its own held-out partition of the fold g-init was
selected from**: the validation and test rows that fold assigned it, which are
exactly the rows that fold's model did not train on.  The fold book makes that
partition available long after the training that produced it, which is what the
book exists for.
"""

from __future__ import annotations

import json
from collections import defaultdict
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from federated_outlier_adaptation.logging_utils import NistLogger

#: Parts of a writer's split that g-init did not train on.
HELDOUT_PARTS = ("val", "test")


def holdout_rows(book, fold: int, writer: str) -> List[int]:
    """The rows of one writer that the given fold held out."""
    rows: List[int] = []
    for part in HELDOUT_PARTS:
        rows.extend(book.part(fold, writer, part))
    return sorted(rows)


def score_writers(
    provider,
    model,
    book,
    fold: int,
    writers: Optional[Sequence[str]] = None,
    batch_size: int = 256,
    log_every: int = 250,
) -> Dict[str, Any]:
    """
    Accuracy of ``model`` on every writer's held-out partition of ``fold``.

    Args:
        provider: Dataset provider (its dataset must be the packed 28x28 one).
        model: The model to score with; moved to the available device.
        book: The fold book the partition is read from.
        fold: Which fold's held-out partition to use.
        writers: Writers to score; defaults to every writer the book covers.
        batch_size: Evaluation batch size.
        log_every: Progress interval, in writers.

    Returns:
        ``{"fold", "scores", "skipped", "samples", ...}`` - ``scores`` is a list
        of single-entry dicts, the same shape ``clients_acc_on_global.json`` has
        always had, so every existing reader of that file works unchanged.
    """
    import torch
    from torch.utils.data import DataLoader

    from federated_outlier_adaptation.data.datasets import (
        build_transform,
    )
    from federated_outlier_adaptation.training.evaluate import rows_dataset

    dataset = provider.dataset
    samples = dataset.writer_samples()
    covered = list(writers) if writers is not None else list(book.writers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    scores: List[Dict[str, float]] = []
    per_writer_samples: Dict[str, int] = {}
    skipped: List[str] = []

    for index, writer in enumerate(covered, start=1):
        rows = holdout_rows(book, fold, writer)
        if not rows:
            skipped.append(writer)
            continue
        labels = dict(samples.get(writer, []))
        rows = [row for row in rows if row in labels]
        if not rows:
            skipped.append(writer)
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
                predicted = model(images).max(1)[1]
                correct += int(predicted.eq(targets).sum())
                total += int(targets.size(0))
        accuracy = correct / total if total else 0.0
        scores.append({writer: accuracy})
        per_writer_samples[writer] = total

        if log_every and index % log_every == 0:
            NistLogger.info(f"scored {index}/{len(covered)} writers")

    NistLogger.info(
        f"Scored {len(scores)} writers on fold {fold}'s held-out partition "
        f"({len(skipped)} had none)."
    )
    return {
        "fold": int(fold),
        "rule": (
            "each writer scored only on the validation and test rows this fold "
            "held out, i.e. rows the fold's model never trained on"
        ),
        "scores": scores,
        "samples": per_writer_samples,
        "skipped": skipped,
        "writers": len(scores),
    }


def eligible_for_old_data(
    samples_by_writer: Dict[str, int],
    exclude: Iterable[str],
    min_samples: int,
    splittable: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Writers the old-data draw may take from, sorted.

    Three conditions, all of them recorded with the draw: not in the outlier
    cohort, at least ``min_samples`` images, and - when a list is given - able to
    form a training split at all.
    """
    banned = set(exclude)
    allowed = set(splittable) if splittable is not None else None
    return sorted(
        writer
        for writer, count in samples_by_writer.items()
        if writer not in banned
        and count >= int(min_samples)
        and (allowed is None or writer in allowed)
    )


def draw_old_data(
    samples_by_writer: Dict[str, int],
    exclude: Iterable[str],
    size: int = 100,
    seed: int = 20260824,
    min_samples: int = 100,
    splittable: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Draw the old-data population, once and reproducibly.

    A *draw*, not a ranking: the old data is meant to be an ordinary slice of the
    population, so taking the best or the largest writers would build the
    preservation reference out of an unrepresentative sample.  The draw is a
    seeded ``random.sample`` over the eligible writers in sorted order, so it
    depends on the seed and the eligibility rule and on nothing else.

    Returns:
        The payload, including the eligibility rule and the seed, ready to write.
    """
    eligible = eligible_for_old_data(
        samples_by_writer, exclude=exclude, min_samples=min_samples, splittable=splittable
    )
    if len(eligible) < int(size):
        raise ValueError(
            f"Only {len(eligible)} writers are eligible for the old-data draw of "
            f"{size} (>= {min_samples} samples, not in the cohort"
            + (", splittable" if splittable is not None else "")
            + "). Lower --old-size or --min-samples."
        )

    generator = random.Random(int(seed))
    drawn = sorted(generator.sample(eligible, int(size)))
    return {
        "rule": "seeded uniform draw without replacement from the eligible writers",
        "seed": int(seed),
        "size": int(size),
        "min_samples": int(min_samples),
        "splittable_only": splittable is not None,
        "excluded": sorted(set(exclude)),
        "eligible_clients": len(eligible),
        "clients": drawn,
        "samples": {writer: samples_by_writer[writer] for writer in drawn},
    }


def write_json(payload: Dict[str, Any], path) -> Path:
    """Write a payload and return where it went."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path


def draw_cohort(
    accuracies: Dict[str, float],
    size: int,
    seed: int,
    pool: Optional[int] = None,
    splittable: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    best: bool = False,
) -> Dict[str, Any]:
    """
    Draw a cohort at random from a ranked population, once and reproducibly.

    The T1 studies *rank* their cohort - the k worst writers, no randomness at
    all.  T2 and T3 draw one instead, and the difference is the point of the
    design: a ranked cohort is the worst case, a drawn one is a sample, and a
    method that only helps the worst case is a narrower result than a method
    that helps a sample.

    Args:
        accuracies: ``{writer: accuracy}`` under the shared detector.
        size: How many writers to draw.
        seed: Seed of the draw; the study's own, so two studies of the same size
            are two draws rather than one draw run twice.
        pool: Restrict the draw to the ``pool`` lowest-scoring eligible writers
            (T2, "the 300 worst").  ``None`` draws from everyone eligible (T3),
            which may by chance include outliers - that is allowed, and is what
            makes T3 a study of an *ordinary* federation rather than one
            selected for being unremarkable.
        splittable: Writers that can form a local training split; ``None``
            applies no such filter.
        exclude: Writers to leave out regardless.
        best: Rank from the top instead of the bottom.  The cohort is always
            drawn from the worst end; the *reference* population of the
            qualitative figure is drawn from the best, because "what a typical
            writer looks like" means a writer the detector had no trouble with.

    Returns:
        The payload, carrying the rule, the seed and the pool it drew from.
    """
    excluded = set(exclude or ())
    eligible = [
        writer for writer in sorted(accuracies)
        if writer not in excluded
        and (splittable is None or writer in set(splittable))
    ]
    # Worst first, ties by writer id, so the pool restriction is a function of
    # the scores and of nothing else.
    ranked = sorted(
        eligible,
        key=lambda writer: (float(accuracies[writer]), writer),
        reverse=bool(best),
    )
    population = ranked[: int(pool)] if pool else ranked

    if len(population) < int(size):
        raise ValueError(
            f"Only {len(population)} writers are eligible for a cohort draw of "
            f"{size}"
            + (f" from the {pool} worst" if pool else "")
            + (" (splittable only)" if splittable is not None else "")
            + ". Lower the cohort size or widen the pool."
        )

    generator = random.Random(int(seed))
    drawn = generator.sample(population, int(size))
    return {
        "rule": (
            "seeded uniform draw without replacement from the "
            + (f"{pool} {'highest' if best else 'lowest'}-scoring eligible writers"
               if pool else "eligible writers")
        ),
        "end": "best" if best else "worst",
        "seed": int(seed),
        "k": int(size),
        "size": int(size),
        "pool": int(pool) if pool else None,
        "splittable_only": splittable is not None,
        "eligible_clients": len(eligible),
        "population_clients": len(population),
        # Sorted for readability; the draw itself is what the seed fixes.
        "clients": sorted(drawn),
        "accuracies": {writer: float(accuracies[writer]) for writer in sorted(drawn)},
    }


def writer_counts(dataset) -> Dict[str, Any]:
    """
    How much of each class every writer holds - the selection's paper trail.

    The cohort is cut from writer *accuracies*, but a reader of the paper will
    ask the next question immediately: were the worst writers simply the ones
    with the least data, or the fewest examples of the hard classes?  That is
    not answerable from a ranking alone, so the counts are banked next to it.

    The counts are of the **selected class set**.  Under ``--classes digits`` a
    writer's total is its digit rows, not its rows - roughly half - and writers
    holding no digits at all are absent from the dataset entirely rather than
    present with a zero.

    Returns:
        The per-writer totals, the per-writer per-class counts, and the
        population summary a table row is built from.
    """
    samples = dataset.writer_samples()
    num_classes = dataset.num_classes

    per_writer: Dict[str, int] = {}
    per_class: Dict[str, Dict[str, int]] = {}
    class_totals: Dict[int, int] = {label: 0 for label in range(num_classes)}

    for writer in sorted(samples):
        counts: Dict[int, int] = defaultdict(int)
        for _, label in samples[writer]:
            counts[int(label)] += 1
        per_writer[writer] = sum(counts.values())
        per_class[writer] = {str(label): counts[label] for label in sorted(counts)}
        for label, count in counts.items():
            class_totals[label] = class_totals.get(label, 0) + count

    totals = sorted(per_writer.values())
    covered = [len(per_class[writer]) for writer in per_writer]
    return {
        "classes": getattr(dataset, "classes", None),
        "num_classes": num_classes,
        "writers": len(per_writer),
        "rows": sum(per_writer.values()),
        "per_writer_total": per_writer,
        "per_writer_per_class": per_class,
        "class_totals": {str(k): v for k, v in sorted(class_totals.items())},
        "summary": {
            "min": totals[0] if totals else None,
            "max": totals[-1] if totals else None,
            "mean": (sum(totals) / len(totals)) if totals else None,
            "median": totals[len(totals) // 2] if totals else None,
            # A writer missing a class cannot contribute to it, which is the
            # first thing to check when a writer scores badly.
            "writers_with_all_classes": sum(1 for n in covered if n == num_classes),
            "min_classes_covered": min(covered) if covered else None,
        },
    }


def write_counts_csv(payload: Dict[str, Any], path) -> Path:
    """The same table as a CSV, one row per writer, one column per class."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in range(int(payload["num_classes"]))]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["writer", "total", *labels])
        for name, total in payload["per_writer_total"].items():
            counts = payload["per_writer_per_class"].get(name, {})
            writer.writerow([name, total, *(counts.get(label, 0) for label in labels)])
    return path


def cohort_table(
    clients: Sequence[str],
    accuracies: Dict[str, float],
    counts: Dict[str, Any],
    book=None,
    extra_scores: Optional[Dict[str, Dict[str, float]]] = None,
    rule: Optional[str] = None,
) -> Dict[str, Any]:
    """
    The cohort, with everything a reader needs to argue with the selection.

    The cohort is cut from a ranking, and a ranking on its own invites two
    objections that it cannot answer: *were these simply the writers with the
    least data?* and *were they the ones missing the hard classes?*  So each
    row carries the accuracy that selected the writer, its rank in the whole
    population, how many rows it holds, how those rows fall across the classes,
    and - when a fold book is given - how many rows each fold puts in train,
    validation and test.

    The rank is over **every scored writer**, not over the cohort: "rank 3 of
    3,580" is the statement worth printing, and "rank 3 of 10" is not.

    Args:
        clients: The cohort, in whatever order the selection produced.
        accuracies: ``{writer: accuracy}`` under the detector.
        counts: The payload of :func:`writer_counts`.
        book: The cohort's fold book, for the per-fold split sizes.
        extra_scores: Further named rankings to carry as columns, e.g. the
            coarse detector's accuracy next to the shipped model's.  A reader
            comparing the two can see whether the writers the shipped model
            finds hardest are the ones the coarse pass already flagged.
        rule: How the cohort was cut, for the record.

    Returns:
        The table payload: one row per client, plus the population context the
        ranks are relative to.
    """
    clients = list(clients)
    ranking = sorted(accuracies, key=lambda writer: (float(accuracies[writer]), writer))
    rank_of = {writer: index + 1 for index, writer in enumerate(ranking)}
    per_class = counts.get("per_writer_per_class", {})
    totals = counts.get("per_writer_total", {})
    num_classes = int(counts.get("num_classes", 10))

    rows = []
    for writer in clients:
        row: Dict[str, Any] = {
            "writer": writer,
            "accuracy": float(accuracies[writer]) if writer in accuracies else None,
            "rank": rank_of.get(writer),
            "of": len(ranking),
            "total_rows": totals.get(writer),
            "per_class": {
                str(label): per_class.get(writer, {}).get(str(label), 0)
                for label in range(num_classes)
            },
        }
        row["classes_covered"] = sum(1 for v in row["per_class"].values() if v)
        for name, table in (extra_scores or {}).items():
            value = table.get(writer)
            row[name] = float(value) if value is not None else None
        if book is not None:
            folds: Dict[str, Dict[str, int]] = {}
            for fold in range(1, book.folds + 1):
                folds[str(fold)] = {
                    part: len(book.part(fold, writer, part))
                    for part in ("train", "val", "test")
                }
            row["folds"] = folds
        rows.append(row)

    scored = [row["accuracy"] for row in rows if row["accuracy"] is not None]
    held = [row["total_rows"] for row in rows if row["total_rows"] is not None]
    population = sorted(totals.values())
    return {
        "classes": counts.get("classes"),
        "num_classes": num_classes,
        "size": len(rows),
        "rule": rule or "the k lowest-scoring splittable writers under the detector",
        "extra_columns": sorted(extra_scores or {}),
        "rows": rows,
        "population": {
            "scored_writers": len(ranking),
            "median_rows": population[len(population) // 2] if population else None,
            "mean_accuracy": (sum(accuracies.values()) / len(accuracies))
            if accuracies else None,
        },
        "summary": {
            "min_accuracy": min(scored) if scored else None,
            "max_accuracy": max(scored) if scored else None,
            "min_rows": min(held) if held else None,
            "max_rows": max(held) if held else None,
            "mean_rows": (sum(held) / len(held)) if held else None,
        },
    }


def write_cohort_table_csv(payload: Dict[str, Any], path) -> Path:
    """The same table as a CSV: one row per writer, one column per class."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in range(int(payload["num_classes"]))]
    folded = any("folds" in row for row in payload["rows"])
    fold_ids = sorted(
        {fold for row in payload["rows"] for fold in row.get("folds", {})},
        key=int,
    )
    extra = list(payload.get("extra_columns") or [])
    header = ["writer", "accuracy", "rank", "of", "total_rows", "classes_covered"]
    header += extra
    header += [f"class_{label}" for label in labels]
    if folded:
        for fold in fold_ids:
            header += [f"f{fold}_train", f"f{fold}_val", f"f{fold}_test"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in payload["rows"]:
            line = [
                row["writer"], row["accuracy"], row["rank"], row["of"],
                row["total_rows"], row["classes_covered"],
            ]
            line += [row.get(name) for name in extra]
            line += [row["per_class"].get(label, 0) for label in labels]
            if folded:
                for fold in fold_ids:
                    part = row.get("folds", {}).get(fold, {})
                    line += [part.get("train"), part.get("val"), part.get("test")]
            writer.writerow(line)
    return path


def split_pools(
    accuracies: Dict[str, float],
    bad_fraction: float = 0.30,
    eligible: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Cut the population into a BAD pool and a GOOD pool by the coarse detector.

    g-init is a *coarse* instrument, and this is the only thing it is asked to
    do: separate the writers worth looking at from the writers that are not.
    It does not choose the cohort - the shipped model g-0 does that, later,
    scoring only the bad pool.  The division of labour matters because g-init
    trained on every writer, so its ranking is contaminated by memorisation in a
    way a *cut* can absorb and a *selection* cannot.

    The GOOD pool is where the old data is drawn from, which is what makes the
    later scoring clean: g-0 is trained on GOOD writers and then scores BAD
    ones, so no writer is ever scored by a model that trained on it.

    Args:
        accuracies: ``{writer: accuracy}`` under the coarse detector.
        bad_fraction: Share of the ranking that is BAD; the cut is at
            ``ceil(bad_fraction * n)`` so the bad pool is never short.
        eligible: Writers allowed in either pool; ``None`` allows all.

    Returns:
        The two lists with the rule, the cut rank and the accuracy there.
    """
    import math

    allowed = None if eligible is None else set(eligible)
    ranked = sorted(
        (w for w in accuracies if allowed is None or w in allowed),
        key=lambda writer: (float(accuracies[writer]), writer),
    )
    if not ranked:
        raise ValueError("No eligible writer has an accuracy to rank.")
    if not 0.0 < float(bad_fraction) < 1.0:
        raise ValueError(
            f"The bad fraction must lie in (0, 1); got {bad_fraction!r}"
        )

    cut = math.ceil(float(bad_fraction) * len(ranked))
    cut = max(1, min(cut, len(ranked)))
    bad, good = ranked[:cut], ranked[cut:]
    return {
        "rule": (
            f"the worst {bad_fraction:.0%} of the coarse detector's ranking, "
            "worst first, ties by writer id"
        ),
        "bad_fraction": float(bad_fraction),
        "ranked_writers": len(ranked),
        "excluded_by_eligibility": (
            len(accuracies) - len(ranked) if allowed is not None else 0
        ),
        "cut_rank": cut,
        "accuracy_at_cut": float(accuracies[bad[-1]]),
        "first_good_accuracy": float(accuracies[good[0]]) if good else None,
        "bad_count": len(bad),
        "good_count": len(good),
        "bad": bad,
        "good": good,
        "accuracies": {writer: float(accuracies[writer]) for writer in ranked},
    }


def write_pools_csv(payload: Dict[str, Any], path) -> Path:
    """One row per writer: its accuracy, its rank and which pool it landed in."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accuracies = payload["accuracies"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["writer", "accuracy", "rank", "pool"])
        rank = 0
        for pool in ("bad", "good"):
            for name in payload[pool]:
                rank += 1
                writer.writerow([name, accuracies[name], rank, pool])
    return path


def score_pool(provider, model, clients: Sequence[str], batch_size: int = 256) -> Dict[str, Any]:
    """
    Score a model on **every** row of each named writer.

    Not a held-out partition: the point of this pass is that the model has never
    seen these writers at all, so there is nothing to hold out from.  Using all
    of a writer's data is what makes the estimate as tight as the writer's own
    data allows - which matters, because the writers being scored are the small,
    difficult ones.

    Args:
        provider: Dataset provider over the packed cache.
        model: The scoring model.
        clients: The writers to score.
        batch_size: Evaluation batch size.

    Returns:
        The per-writer accuracies with their row counts, worst first.
    """
    import torch

    from federated_outlier_adaptation.training.evaluate import accuracy_of, rows_loader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = provider.dataset.writer_samples()

    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    skipped: List[str] = []
    for writer in clients:
        rows = [row for row, _ in samples.get(writer, [])]
        loader = rows_loader(provider, rows, batch_size)
        accuracy = accuracy_of(model, loader, device)
        if accuracy is None:
            skipped.append(writer)
            continue
        scores[writer] = float(accuracy)
        counts[writer] = len(rows)

    ranked = sorted(scores, key=lambda writer: (scores[writer], writer))
    values = [scores[writer] for writer in ranked]
    return {
        "rule": "the model's accuracy on every row the writer holds",
        "writers": len(ranked),
        "rows": sum(counts.values()),
        "skipped": skipped,
        "ranking": ranked,
        "scores": [{writer: scores[writer]} for writer in ranked],
        "accuracies": {writer: scores[writer] for writer in ranked},
        "samples": {writer: counts[writer] for writer in ranked},
        "rank": {writer: index + 1 for index, writer in enumerate(ranked)},
        "summary": {
            "min": values[0] if values else None,
            "max": values[-1] if values else None,
            "mean": (sum(values) / len(values)) if values else None,
            "median": values[len(values) // 2] if values else None,
        },
    }


def write_scores_csv(payload: Dict[str, Any], path) -> Path:
    """One row per writer: accuracy, rows held, rank - worst first."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["writer", "accuracy", "n_rows", "rank"])
        for name in payload["ranking"]:
            writer.writerow([
                name, payload["accuracies"][name],
                payload["samples"][name], payload["rank"][name],
            ])
    return path
