import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from torch import nn
from tqdm import tqdm

from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.providers import default_provider

base_criterion = nn.CrossEntropyLoss()
"""
selecting_outliers.py

Purpose:
    Evaluate how well a trained global model performs on individual clients’ test sets,
    in order to identify low-performing ("outlier") clients in federated adaptive learning.

Key Features:
    - Iterates over a set of local writers (clients).
    - Evaluates global model accuracy on each client’s test set.
    - Collects per-client accuracy results into a list of dicts.
    - Supports caching results to avoid redundant evaluation.
    - Persists results to JSON for downstream selection/analysis.
    - Ranks clients by accuracy and writes the k lowest-accuracy client lists.

Inputs:
    - global_model: Trained PyTorch model (e.g., CNN).
    - local_writers: List of client IDs.
    - clients_acc_on_global_path: Path to JSON file for saving/loading results.
    - force_generate: If True, recompute results even if JSON exists.

Outputs:
    - clients_acc_on_global.json (list of {client_id: accuracy})
    - selected_clients_k<k>.json (list of client IDs, the format of
      selected_outliers.json)
    - Console logs per client and summary.

Dependencies:
    - torch, tqdm, json
    - federated_outlier_adaptation.logging_utils.NistLogger
    - federated_outlier_adaptation.providers

The module is dataset agnostic: every dataset access goes through the provider,
so it serves NIST, LEAF Shakespeare and CIFAR-10 alike.  With ``provider=None``
the NIST default provider is used and the behaviour is the published one.
"""


def select_outliers(
    global_model,
    local_writers,
    clients_acc_on_global_path: Path,
    force_generate=False,
    provider=None,
    batch_size: int = 64,
):
    """
        Evaluate a global model on all clients’ test sets and record their accuracies.

        Args:
            global_model (torch.nn.Module):
                Trained PyTorch model to be evaluated.
            local_writers (list[str]):
                List of client/writer IDs whose test sets will be used.
            clients_acc_on_global_path (Path):
                Path to JSON file for saving or loading evaluation results.
            force_generate (bool, optional):
                If True, recompute client accuracies even if the JSON file exists.
                If False (default), load cached results if available.
            provider (DatasetProvider, optional):
                Source of the client loaders.  Defaults to the NIST provider.
            batch_size (int, optional):
                Evaluation batch size.  Defaults to 64, the published value.

        Process:
            1. If cached JSON exists (and force_generate is False), load and return results.
            2. Otherwise:
                - For each client, build their test dataset.
                - Run evaluation with the global model (CrossEntropyLoss).
                - Compute accuracy per client.
                - Log and collect results into a list of dicts.
            3. Save results to JSON.

        Returns:
            list[dict]:
                A list of {client_id: accuracy} entries, one per client.
        """
    clients_acc_on_global_path = Path(clients_acc_on_global_path)
    if not force_generate and  clients_acc_on_global_path.exists():
        NistLogger.info(f"Loading clients with accuracy on global model from {clients_acc_on_global_path}")
        with open(clients_acc_on_global_path, "r") as f:
            clients_acc_on_global = json.load(f)
        return clients_acc_on_global
# === Evaluate Only on Test Set Per Client ===
    clients_acc_on_global = []
    provider = provider or default_provider()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for client_id in tqdm(local_writers, desc="Processing Clients"):
        _, _, test_loader = provider.build_dataset(
            client_id, train_rate=0.0, eval_rate=0.0, batch_size=batch_size
        )

        if test_loader is None :
            NistLogger.info(f"Skipping client {client_id}: No test data.")
            continue

        global_model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                loss = base_criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total if test_total > 0 else 0.0

        NistLogger.info(f"Client '{client_id}' — Test Accuracy: {test_acc:.4f}")

        clients_acc_on_global.append({f"{client_id}": test_acc})

    NistLogger.info("\nList of all clients with accuracy using trained global model")

    clients_acc_on_global_path.parent.mkdir(parents=True, exist_ok=True)
    with open(clients_acc_on_global_path, "w") as f:
        json.dump(clients_acc_on_global, f)
    return clients_acc_on_global


# --------------------------------------------------------------- ranking
def load_client_accuracies(clients_acc_on_global_path: Path) -> List[Tuple[str, float]]:
    """
    Read ``clients_acc_on_global.json`` as ``(client_id, accuracy)`` pairs.

    The stored format is the published one: a list of single-entry dicts.
    """
    with open(clients_acc_on_global_path, "r") as f:
        payload = json.load(f)
    return [(client_id, value) for entry in payload for client_id, value in entry.items()]


def lowest_accuracy_clients(
    clients_acc_on_global_path: Path,
    k: int = 5,
    eligible: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    The ``k`` clients with the lowest accuracy under the global model.

    Deterministic - unlike
    :func:`~federated_outlier_adaptation.outliers.selection.get_low_acc_writers`,
    which samples randomly inside the worst ``bottom_frac`` of the clients and
    defines the published NIST outlier set.  This ranking is used by the
    additional datasets, which have no frozen selection to preserve.

    Args:
        clients_acc_on_global_path: The per-client accuracy record.
        k: Number of clients to return.
        eligible: Optional restriction, e.g. clients with enough samples.

    Returns:
        Client ids, lowest accuracy first.
    """
    pairs = load_client_accuracies(clients_acc_on_global_path)
    if eligible is not None:
        allowed = set(eligible)
        pairs = [pair for pair in pairs if pair[0] in allowed]
    pairs.sort(key=lambda pair: (pair[1], pair[0]))
    return [client_id for client_id, _ in pairs[:k]]


def write_selected_clients(
    clients_acc_on_global_path: Path,
    outliers_dir: Path,
    k: int = 5,
    eligible: Optional[Iterable[str]] = None,
    mirror_published_name: bool = True,
) -> Tuple[Path, List[str]]:
    """
    Write the ``k`` lowest-accuracy clients to ``selected_clients_k<k>.json``.

    The file format is the one of the published ``selected_outliers.json``: a
    plain JSON list of client ids.  For ``k == 5`` the list is mirrored to
    ``selected_outliers.json`` as well, so that the runners' default artefact
    layout is complete for the new datasets.  NIST is unaffected: this function
    is never called on the NIST results root by the existing pipeline.

    Returns:
        ``(path, client_ids)``.
    """
    outliers_dir = Path(outliers_dir)
    outliers_dir.mkdir(parents=True, exist_ok=True)
    selected = lowest_accuracy_clients(clients_acc_on_global_path, k=k, eligible=eligible)

    target = outliers_dir / f"selected_clients_k{k}.json"
    with open(target, "w") as f:
        json.dump(selected, f, indent=2)
    NistLogger.info(f"Selected {len(selected)} lowest-accuracy clients -> {target}")

    if mirror_published_name and k == 5:
        mirror = outliers_dir / "selected_outliers.json"
        with open(mirror, "w") as f:
            json.dump(selected, f, indent=2)
        NistLogger.info(f"Mirrored the k=5 selection to {mirror}")
    return target, selected


# ------------------------------------------------------------------- pools
def pool_file_name(frac: Optional[float] = None, threshold: Optional[float] = None) -> str:
    """File name of an outlier pool: ``outlier_pool_frac<f>.json``."""
    if threshold is not None:
        return f"outlier_pool_thr{threshold:g}.json"
    return f"outlier_pool_frac{frac:g}.json"


#: Severity tags of a client pool.  They name a cut of the accuracy
#: distribution - which one is decided from the histogram of the run's own
#: theta_g - and are carried into the provenance of every run drawn from the
#: pool, so that a result says which severity it belongs to without the reader
#: having to match numbers back to a threshold.
SEVERITIES = ("mild", "moderate", "severe")


def build_pool(
    clients_acc_on_global_path: Path,
    frac: Optional[float] = 0.05,
    threshold: Optional[float] = None,
    eligible: Optional[Iterable[str]] = None,
    severity: Optional[str] = None,
) -> Dict[str, object]:
    """
    The rule-based outlier pool of a dataset.

    Two equivalent rules are supported and exactly one is applied:

    * ``frac`` - the worst ``frac`` of the eligible held-out clients, sized
      ``max(1, int(frac * n))``, the truncating rule
      :func:`~federated_outlier_adaptation.outliers.selection.get_low_acc_writers`
      already uses for the published NIST bottom fraction;
    * ``threshold`` - every eligible held-out client whose accuracy under the
      global model is strictly below the threshold.

    Unlike the published selection the pool is deterministic: it is the ranking
    itself, not a sample drawn from it.  Per-round client selection happens on
    top of the pool, so the pool records the baseline accuracy of every member.

    Args:
        clients_acc_on_global_path: The per-client accuracy record.
        frac: Bottom fraction; ignored when ``threshold`` is given.
        threshold: Accuracy cut-off - every eligible client scoring strictly
            below it.  This is the rule the severities are stated in, because a
            threshold is a property of the accuracy distribution while a
            fraction is a property of the population size.
        eligible: Optional restriction, e.g. clients with enough samples.
        severity: Optional tag (:data:`SEVERITIES`) recorded with the pool.

    Returns:
        The pool payload: ``clients`` (worst first), ``accuracies``, the rule
        that produced it and the population it was drawn from.
    """
    pairs = load_client_accuracies(clients_acc_on_global_path)
    if eligible is not None:
        allowed = set(eligible)
        pairs = [pair for pair in pairs if pair[0] in allowed]
    pairs.sort(key=lambda pair: (pair[1], pair[0]))

    if threshold is not None:
        selected = [pair for pair in pairs if pair[1] < float(threshold)]
    else:
        if frac is None:
            raise ValueError("Either frac or threshold must be given")
        size = max(1, int(float(frac) * len(pairs)))
        selected = pairs[:size]

    accuracies = [value for _, value in selected]
    if severity is not None and severity not in SEVERITIES:
        raise ValueError(f"Unknown severity {severity!r}; expected one of {SEVERITIES}")

    return {
        "rule": "threshold" if threshold is not None else "fraction",
        "severity": severity,
        "frac": None if threshold is not None else float(frac),
        "threshold": None if threshold is None else float(threshold),
        "size": len(selected),
        "eligible_clients": len(pairs),
        "clients": [client for client, _ in selected],
        "accuracies": {client: value for client, value in selected},
        "min_accuracy": min(accuracies) if accuracies else None,
        "max_accuracy": max(accuracies) if accuracies else None,
        "mean_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else None,
        "source": str(clients_acc_on_global_path),
    }


def write_pool(
    clients_acc_path: Path,
    out_dir: Path,
    frac: Optional[float] = 0.05,
    threshold: Optional[float] = None,
    eligible: Optional[Iterable[str]] = None,
    severity: Optional[str] = None,
) -> Tuple[Path, Dict[str, object]]:
    """
    Write the rule-based outlier pool to ``out_dir``.

    The file is ``outlier_pool_frac<f>.json`` (``outlier_pool_thr<t>.json`` for
    the threshold rule) and lists the client ids together with their baseline
    accuracy under the global model; see :func:`build_pool` for the payload.

    Returns:
        ``(path, payload)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_pool(
        clients_acc_path,
        frac=frac,
        threshold=threshold,
        eligible=eligible,
        severity=severity,
    )

    target = out_dir / pool_file_name(frac=frac, threshold=threshold)
    with open(target, "w") as f:
        json.dump(payload, f, indent=2)
    NistLogger.info(
        f"Outlier pool of {payload['size']} / {payload['eligible_clients']} clients -> {target}"
    )
    return target, payload


def load_pool(path: Path) -> List[str]:
    """
    Read a pool file and return its client ids.

    A plain JSON list is accepted as well, so a pool written in the simpler
    format of ``selected_outliers.json`` still loads.
    """
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return list(payload)
    return list(payload["clients"])


def accuracy_summary(
    clients_acc_on_global_path: Path,
    eligible: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Distribution of the held-out clients' accuracies under the global model."""
    pairs = load_client_accuracies(clients_acc_on_global_path)
    if eligible is not None:
        allowed = set(eligible)
        pairs = [pair for pair in pairs if pair[0] in allowed]
    values = sorted(value for _, value in pairs)
    if not values:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        position = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
        return values[position]

    return {
        "count": len(values),
        "min": values[0],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "median": percentile(0.50),
        "mean": sum(values) / len(values),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": values[-1],
    }


# --------------------------------------------------------------------------- #
# fixed cohorts
# --------------------------------------------------------------------------- #
#: Rule tag of a fixed cohort, recorded in the payload.
COHORT_RULE = "worst_k"


def cohort_file_name(k: int = 20) -> str:
    """File name of a fixed cohort: ``cohort_worst<k>.json``."""
    return f"cohort_worst{int(k)}.json"


def build_cohort(
    clients_acc_on_global_path: Path,
    k: int = 20,
    eligible: Optional[Iterable[str]] = None,
    tag: Optional[str] = None,
) -> Dict[str, object]:
    """
    The ``k`` worst-scoring eligible clients, as a fixed enrolment.

    A *pool* is a population that per-round selection samples from and that can
    be re-cut at any time; a *cohort* is an enrolment fixed once for the whole
    of training, which is what a study of a named group of clients needs.  The
    rule is deterministic - the ranking itself, worst first, with ties broken by
    client id - so the same accuracy record always yields the same twenty
    writers.

    ``eligible`` is applied *before* the cut, and the ids it removes from the
    bottom ``k`` are recorded under ``excluded_from_cohort``: a writer that is
    too small to form a training split would otherwise be enrolled precisely
    because it scores worst, and the record has to say that it was dropped and
    which writer took its place.

    Args:
        clients_acc_on_global_path: The per-client accuracy record.
        k: Cohort size.
        eligible: Clients allowed to be enrolled; ``None`` allows all of them.
        tag: Optional label stored with the cohort.

    Returns:
        The cohort payload.  It carries ``clients`` and ``accuracies``, so
        ``--outliers-file`` reads it exactly like a pool file.
    """
    ranked = load_client_accuracies(clients_acc_on_global_path)
    ranked.sort(key=lambda pair: (pair[1], pair[0]))

    if eligible is None:
        allowed = None
        kept = list(ranked)
    else:
        allowed = set(eligible)
        kept = [pair for pair in ranked if pair[0] in allowed]

    size = max(0, int(k))
    selected = kept[:size]
    # Who the unfiltered ranking would have enrolled, and did not.
    unfiltered = [client for client, _ in ranked[:size]]
    chosen = {client for client, _ in selected}
    excluded = [client for client in unfiltered if client not in chosen]

    accuracies = [value for _, value in selected]
    return {
        "rule": COHORT_RULE,
        "tag": tag,
        "k": size,
        "size": len(selected),
        "clients": [client for client, _ in selected],
        "accuracies": {client: value for client, value in selected},
        "scored_clients": len(ranked),
        "eligible_clients": len(kept),
        "ineligible_clients": len(ranked) - len(kept),
        "excluded_from_cohort": excluded,
        "min_accuracy": min(accuracies) if accuracies else None,
        "max_accuracy": max(accuracies) if accuracies else None,
        "mean_accuracy": (sum(accuracies) / len(accuracies)) if accuracies else None,
        "source": str(clients_acc_on_global_path),
    }


def write_cohort(
    clients_acc_path: Path,
    out_dir: Path,
    k: int = 20,
    eligible: Optional[Iterable[str]] = None,
    tag: Optional[str] = None,
) -> Tuple[Path, Dict[str, object]]:
    """
    Write the fixed cohort to ``out_dir/cohort_worst<k>.json``.

    Returns:
        ``(path, payload)``; see :func:`build_cohort`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cohort(clients_acc_path, k=k, eligible=eligible, tag=tag)

    target = out_dir / cohort_file_name(k)
    with open(target, "w") as f:
        json.dump(payload, f, indent=2)
    NistLogger.info(
        f"Cohort of {payload['size']} clients (worst of {payload['eligible_clients']} "
        f"eligible, {payload['ineligible_clients']} excluded by the eligibility "
        f"rule) -> {target}"
    )
    if payload["excluded_from_cohort"]:
        NistLogger.warning(
            "These clients scored inside the bottom "
            f"{payload['k']} but are not enrolled: {payload['excluded_from_cohort']}"
        )
    return target, payload


#: File name of the per-client accuracy table.
ACCURACY_CSV_NAME = "clients_acc_on_global.csv"


def write_accuracy_table(
    clients_acc_on_global_path: Path,
    out_dir: Path,
    eligible: Optional[Iterable[str]] = None,
    name: str = ACCURACY_CSV_NAME,
) -> Tuple[Path, Dict[str, float]]:
    """
    Write every held-out client's accuracy under the global model as a CSV.

    The JSON record is the source of truth and is left untouched; this is the
    same numbers in the form the severity cuts are read off - one row per
    client, worst first, with the rank and the cumulative fraction of the
    population, so a threshold and the pool size it implies can be read
    directly off the table.  The summary of the distribution is returned with
    it, so a caller can print both.

    Returns:
        ``(path, summary)``.
    """
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_client_accuracies(clients_acc_on_global_path)
    if eligible is not None:
        allowed = set(eligible)
        pairs = [pair for pair in pairs if pair[0] in allowed]
    pairs.sort(key=lambda pair: (pair[1], pair[0]))

    target = out_dir / name
    total = len(pairs) or 1
    with open(target, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "client", "accuracy", "cumulative_fraction"])
        for rank, (client, accuracy) in enumerate(pairs, start=1):
            writer.writerow([rank, client, f"{accuracy:.6f}", f"{rank / total:.6f}"])
    NistLogger.info(f"Per-client accuracy table of {len(pairs)} clients -> {target}")
    return target, accuracy_summary(clients_acc_on_global_path, eligible=eligible)


def selected_client_accuracies(
    clients_acc_on_global_path: Path,
    clients: Sequence[str],
) -> Dict[str, float]:
    """Pre-adaptation accuracy of the given clients under the global model."""
    table = dict(load_client_accuracies(clients_acc_on_global_path))
    return {client: table.get(client) for client in clients}


# --------------------------------------------------------------- entry point
def main(argv=None):
    """
    Write outlier pools from an existing per-client accuracy record.

    The record itself is produced by the global-training phase; this entry point
    only applies the pool rules, so it needs no GPU and no dataset arrays beyond
    the provider's eligibility check.

        python -m federated_outlier_adaptation.outliers.client_accuracy \\
            --provider shakespeare --pool-frac 0.05 0.2
    """
    import argparse

    from federated_outlier_adaptation.providers import get_provider

    parser = argparse.ArgumentParser(description="Write the rule-based outlier pools.")
    parser.add_argument("--provider", default="nist", help="nist | shakespeare | cifar10")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--pool-frac", type=float, nargs="*", default=[0.05, 0.2])
    parser.add_argument("--pool-threshold", type=float, nargs="*", default=[])
    args = parser.parse_args(argv)

    kwargs = {}
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    provider = get_provider(args.provider, **kwargs)

    accuracies_path = Path(
        getattr(
            provider,
            "clients_acc_on_global_path",
            Path(provider.results_dir) / "outliers" / "clients_acc_on_global.json",
        )
    )
    eligible = provider.eligible_clients() if hasattr(provider, "eligible_clients") else None
    out_dir = Path(provider.results_dir) / "outliers"

    written = {}
    for frac in args.pool_frac:
        path, payload = write_pool(accuracies_path, out_dir, frac=frac, eligible=eligible)
        written[str(path)] = {
            "size": payload["size"],
            "eligible_clients": payload["eligible_clients"],
            "min_accuracy": payload["min_accuracy"],
            "max_accuracy": payload["max_accuracy"],
            "mean_accuracy": payload["mean_accuracy"],
        }
    for threshold in args.pool_threshold:
        path, payload = write_pool(
            accuracies_path, out_dir, frac=None, threshold=threshold, eligible=eligible
        )
        written[str(path)] = {
            "size": payload["size"],
            "eligible_clients": payload["eligible_clients"],
            "min_accuracy": payload["min_accuracy"],
            "max_accuracy": payload["max_accuracy"],
            "mean_accuracy": payload["mean_accuracy"],
        }

    print(json.dumps({"provider": provider.name, "pools": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
