
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt

from federated_outlier_adaptation.logging_utils import NistLogger

"""
outlier_selection.py

Purpose:
    Identify and visualize low-performing (low-accuracy) clients/writers
    from global evaluation results in federated adaptive learning.

Key Features:
    - Loads per-client accuracy metrics on a global test set.
    - Ranks clients by accuracy (descending).
    - Selects the bottom 0.5% of writers (at least 1).
    - Randomly samples 5 writers from this bottom group (seeded).
    - Alternatively (mode="lowest") takes the k lowest-accuracy writers
      deterministically, for the larger client populations.
    - Saves selected outlier IDs as JSON.
    - Produces plots:
        * All writers with bottom 0.5% highlighted.
        * Bottom 0.5% subset with selected writers emphasized.

Inputs:
    - clients_acc_on_global_path: JSON file with client accuracies.
    - selected_outliers_path: Output file for selected outlier IDs.
    - outliers_result_path: Directory for saving plots.

Outputs:
    - selected_outliers.json (chosen writer IDs)
    - all_none_contributor_writers.png
    - bottom_005_selected_writers.png

Dependencies:
    - matplotlib, json, random
    - federated_outlier_adaptation.logging_utils.NistLogger
"""


def load_client_accuracies(clients_acc_on_global_path):
    """
    Read the per-client accuracy file and return ``[(client_id, acc), ...]``.

    Args:
        clients_acc_on_global_path (Path or str): JSON file formatted as a list
            of single-entry dicts, ``[{client_id: acc}, ...]``.

    Returns:
        list[tuple[str, float]]: Flattened (client, accuracy) pairs.
    """
    with open(clients_acc_on_global_path, "r") as f:
        writers_acc_on_global = json.load(f)
    return [(k_id, v) for entry in writers_acc_on_global for k_id, v in entry.items()]


def get_lowest_acc_writers(clients_acc_on_global_path, selected_outliers_path, k=20):
    """
    Take the ``k`` lowest-accuracy clients deterministically.

    Unlike :func:`get_low_acc_writers` this performs no sampling and ignores
    ``bottom_frac``: the clients are sorted by accuracy ascending (ties broken
    by client id, so the result does not depend on the file order) and the
    first ``k`` are written out.

    Args:
        clients_acc_on_global_path (Path or str): Per-client accuracy file.
        selected_outliers_path (Path or str): Output JSON file.
        k (int): Number of clients to keep.

    Returns:
        list[str]: The selected client ids, lowest accuracy first.
    """
    flattened = load_client_accuracies(clients_acc_on_global_path)
    ordered = sorted(flattened, key=lambda item: (item[1], item[0]))
    selected_outliers = [client_id for client_id, _ in ordered[: max(0, int(k))]]

    with open(selected_outliers_path, "w") as f:
        json.dump(selected_outliers, f, indent=2)
    NistLogger.info(
        f"Lowest-accuracy writers: {len(selected_outliers)}, saved at {selected_outliers_path}"
    )
    return selected_outliers


def selection_filename(k=5, mode="sample"):
    """
    File name of a selected-client list.

    ``mode="sample"`` with ``k == 5`` keeps the published
    ``selected_outliers.json``; every other combination gets its own name so no
    frozen artefact is ever overwritten.
    """
    if mode == "lowest":
        return f"selected_outliers_k{int(k)}_lowest.json"
    if int(k) == 5:
        return "selected_outliers.json"
    return f"selected_outliers_k{int(k)}.json"


def outliers_dir_of(provider=None, results_dir=None) -> Path:
    """
    Directory holding a dataset's selection artefacts.

    Providers may expose ``outliers_dir`` (the additional datasets do); the
    NIST layout is ``<results>/outliers``.
    """
    from federated_outlier_adaptation import config

    if provider is not None:
        directory = getattr(provider, "outliers_dir", None)
        if directory is not None:
            return Path(directory)
        root = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
        return root / "outliers"
    root = Path(results_dir) if results_dir else Path(config.RESULTS_DIR)
    return root / "outliers"


def accuracies_path_of(provider=None, results_dir=None) -> Path:
    """Per-client accuracy record of a dataset."""
    if provider is not None:
        path = getattr(provider, "clients_acc_on_global_path", None)
        if path is not None:
            return Path(path)
    return outliers_dir_of(provider, results_dir) / "clients_acc_on_global.json"


def eligible_of(provider=None):
    """Clients a dataset considers eligible, or ``None`` for "all of them"."""
    if provider is None:
        return None
    accessor = getattr(provider, "eligible_clients", None)
    return accessor() if callable(accessor) else None


def pool_filename(pool_frac=None, pool_threshold=None):
    """
    File name of a rule-based outlier pool.

    Delegates to
    :func:`~federated_outlier_adaptation.outliers.client_accuracy.pool_file_name`
    so that every dataset writes the same names
    (``outlier_pool_frac0.05.json`` / ``outlier_pool_thr0.9.json``).
    """
    from federated_outlier_adaptation.outliers.client_accuracy import pool_file_name

    return pool_file_name(frac=pool_frac, threshold=pool_threshold)


def build_outlier_pool(
    clients_acc_on_global_path,
    pool_path,
    pool_frac=0.05,
    pool_threshold=None,
    eligible=None,
    severity=None,
):
    """
    Define the outlier pool by a rule instead of a fixed list.

    Thin wrapper around
    :func:`~federated_outlier_adaptation.outliers.client_accuracy.write_pool`,
    so every dataset - the published NIST writers included - uses one writer and
    one naming scheme (``outlier_pool_frac0.05.json``).  See there for the two
    rules (bottom fraction and accuracy threshold) and the payload shape.

    Args:
        pool_path: Where the pool should end up; only its directory is used,
            the file name comes from the shared naming rule.

    Returns:
        dict: The pool payload, including the client ids and their baseline
        accuracies under the frozen global model.
    """
    from federated_outlier_adaptation.outliers.client_accuracy import write_pool

    _, payload = write_pool(
        Path(clients_acc_on_global_path),
        Path(pool_path).parent,
        frac=pool_frac,
        threshold=pool_threshold,
        eligible=eligible,
        severity=severity,
    )
    return payload


def load_client_pool(path):
    """
    Read a selected-client list or a rule-based pool file.

    Accepts both shapes so that ``--outliers-file`` can point at the published
    ``selected_outliers*.json`` (a plain list) and at an
    ``outlier_pool_*.json`` (a dict with the ids and their baseline
    accuracies).

    Returns:
        tuple[list[str], dict[str, float]]: ``(client_ids, baseline accuracies)``;
        the accuracies are empty for a plain list.
    """
    with open(path, "r") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        clients = list(data.get("clients", []))
        accuracies = dict(data.get("accuracies", {}))
        return clients, accuracies
    return list(data), {}


#: Keys of a pool file that describe *which* pool it is, as opposed to who is
#: in it.  They are copied into the provenance of every run drawn from the pool.
POOL_META_KEYS = ("rule", "severity", "tag", "k", "frac", "threshold", "size",
                  "eligible_clients", "ineligible_clients", "excluded_from_cohort",
                  "min_accuracy", "max_accuracy", "mean_accuracy")


def load_pool_meta(path) -> dict:
    """
    The rule and severity of a pool file, or an empty mapping.

    A plain ``selected_outliers.json`` list carries no such description and
    yields ``{}``, so a run against the published five-writer selection records
    exactly what it recorded before.
    """
    with open(path, "r") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    meta = {key: data[key] for key in POOL_META_KEYS if key in data}
    meta["path"] = str(path)
    return meta


def select_outlier_pool(
    results_dir=None,
    pool_frac=0.05,
    pool_threshold=None,
    force=False,
    provider=None,
    severity=None,
    write_table=False,
):
    """
    Write a rule-based outlier pool under a dataset's results root.

    Args:
        results_dir (Path or str, optional): Results root; ignored when a
            provider is given.
        pool_frac (float): Bottom fraction of clients to keep.
        pool_threshold (float, optional): Accuracy threshold rule; wins over
            ``pool_frac`` when given.
        force (bool): Rebuild even if the file exists.
        provider: Dataset provider whose paths and eligibility rule are used.
        severity (str, optional): ``mild`` / ``moderate`` / ``severe``, recorded
            with the pool and carried into the provenance of every run that
            draws from it.
        write_table (bool): Also write the per-client accuracy table the
            severity cuts are read off.

    Returns:
        tuple[dict, Path]: The pool payload and the file written.
    """
    outliers_dir = outliers_dir_of(provider, results_dir)
    outliers_dir.mkdir(parents=True, exist_ok=True)
    target = outliers_dir / pool_filename(pool_frac=pool_frac, pool_threshold=pool_threshold)

    accuracies_path = accuracies_path_of(provider, results_dir)
    if write_table:
        from federated_outlier_adaptation.outliers.client_accuracy import (
            write_accuracy_table,
        )

        _, summary = write_accuracy_table(
            accuracies_path, outliers_dir, eligible=eligible_of(provider)
        )
        NistLogger.info(f"Held-out accuracy distribution: {summary}")

    if target.exists() and not force:
        with open(target, "r") as handle:
            payload = json.load(handle)
        NistLogger.info(f"Loaded existing outlier pool from {target}")
        return payload, target

    payload = build_outlier_pool(
        accuracies_path,
        target,
        pool_frac=pool_frac,
        pool_threshold=pool_threshold,
        eligible=eligible_of(provider),
        severity=severity,
    )
    return payload, target


def select_lowest_pool(results_dir=None, k=20, force=False, provider=None):
    """
    Write ``selected_outliers_k<k>_lowest.json`` under a dataset's results root.

    Deterministic counterpart of ``foa select-outliers``: no sampling, no
    plots, no seed.  Existing files are reused unless ``force`` is set, so the
    pools of a running matrix stay stable.

    Args:
        results_dir (Path or str, optional): Results root; ignored when a
            provider is given.
        k (int): Number of lowest-accuracy clients to keep.
        force (bool): Re-select even if the file already exists.
        provider: Dataset provider whose paths and eligibility rule are used.

    Returns:
        tuple[list[str], Path]: The selected client ids and the file written.
    """
    from federated_outlier_adaptation.outliers.client_accuracy import (
        lowest_accuracy_clients,
    )

    outliers_dir = outliers_dir_of(provider, results_dir)
    outliers_dir.mkdir(parents=True, exist_ok=True)
    target = outliers_dir / selection_filename(k=k, mode="lowest")

    if target.exists() and not force:
        with open(target, "r") as f:
            selected = json.load(f)
        NistLogger.info(f"Loaded existing client pool from {target}")
        return selected, target

    selected = lowest_accuracy_clients(
        accuracies_path_of(provider, results_dir), k=k, eligible=eligible_of(provider)
    )
    with open(target, "w") as f:
        json.dump(selected, f, indent=2)
    NistLogger.info(
        f"Lowest-accuracy clients: {len(selected)}, saved at {target}"
    )
    return selected, target


def select_cohort(
    results_dir=None,
    k=20,
    force=False,
    provider=None,
    tag=None,
    write_table=False,
    scores=None,
):
    """
    Write the fixed cohort of the ``k`` worst-scoring eligible clients.

    A cohort is an enrolment, not a pool: it is decided once from the global
    model's per-client accuracies and then held fixed for the whole of training,
    while per-round selection decides only who of the enrolled clients shows up.

    Args:
        results_dir (Path or str, optional): Results root; ignored when a
            provider is given.
        k (int): Cohort size.
        force (bool): Rebuild even if the file exists.
        provider: Dataset provider whose paths and eligibility rule are used.
            With ``--require-trainable`` the rule removes clients that cannot
            form a local training split, which matters here more than anywhere
            else: those clients are exactly the ones that score worst.
        tag (str, optional): Label stored with the cohort.
        write_table (bool): Also write the per-client accuracy table.
        scores (Path or str, optional): The ranking to cut from.  Defaults to
            the root's ``clients_acc_on_global.json``.  The digits study passes
            the *shipped* model's scores over the bad pool instead, because the
            coarse detector only decides who is worth looking at and the shipped
            model decides who the cohort is.

    Returns:
        tuple[dict, Path]: The cohort payload and the file written.
    """
    from federated_outlier_adaptation.outliers.client_accuracy import (
        cohort_file_name,
        write_cohort,
    )

    outliers_dir = outliers_dir_of(provider, results_dir)
    outliers_dir.mkdir(parents=True, exist_ok=True)
    target = outliers_dir / cohort_file_name(k)
    accuracies_path = (
        Path(scores) if scores else accuracies_path_of(provider, results_dir)
    )

    if write_table:
        from federated_outlier_adaptation.outliers.client_accuracy import (
            write_accuracy_table,
        )

        _, summary = write_accuracy_table(
            accuracies_path, outliers_dir, eligible=eligible_of(provider)
        )
        NistLogger.info(f"Held-out accuracy distribution: {summary}")

    if target.exists() and not force:
        with open(target, "r") as handle:
            payload = json.load(handle)
        NistLogger.info(f"Loaded existing cohort from {target}")
        return payload, target

    _, payload = write_cohort(
        accuracies_path, outliers_dir, k=k, eligible=eligible_of(provider), tag=tag
    )
    return payload, target


def get_low_acc_writers(
    clients_acc_on_global_path,
    selected_outliers_path,
    outliers_result_path,
    seed=42,
    k=5,
    bottom_frac=0.005,
    mode="sample",
):
    """
        Select and visualize low-performing clients/writers.

        Args:
            clients_acc_on_global_path (Path or str):
                Path to JSON file containing client accuracies, formatted as
                a list of dicts: [{client_id: acc}, ...].
            selected_outliers_path (Path or str):
                Path to JSON file where selected outlier IDs will be saved.
            outliers_result_path (Path):
                Directory where plots will be saved.
            seed (int, optional):
                Random seed for reproducibility when sampling outliers.
                Defaults to 42.
            k (int, optional):
                Number of outliers to sample from the bottom group.  Defaults to
                5, the value used for the published selection.
            bottom_frac (float, optional):
                Fraction of the worst-performing clients to sample from.
                Defaults to 0.005 (the bottom 0.5%).
            mode (str, optional):
                ``"sample"`` (default) reproduces the published selection: draw
                ``k`` writers at random from the bottom ``bottom_frac``.
                ``"lowest"`` takes the ``k`` lowest-accuracy writers
                deterministically and skips the sampling plots; see
                :func:`get_lowest_acc_writers`.

        Process:
            1. Load accuracies and flatten to (client_id, acc) tuples.
            2. Sort clients by accuracy (descending).
            3. Take the bottom ``bottom_frac`` of writers (at least 1).
            4. Randomly select ``k`` writers from this group (or fewer if not enough).
            5. Save selected writer IDs to JSON.
            6. Generate two plots:
                - All writers with the bottom group highlighted.
                - Bottom group only with the chosen writers annotated.

        Returns:
            list[str]:
                List of selected writer IDs (outliers).
        """
    if mode == "lowest":
        return get_lowest_acc_writers(
            clients_acc_on_global_path, selected_outliers_path, k=k
        )
    if mode != "sample":
        raise ValueError(f"Unknown selection mode {mode!r}; expected 'sample' or 'lowest'")

    # === Load and sort writers ===
    # Convert to flat list of tuples: [(client_id, acc), ...]
    flattened = load_client_accuracies(clients_acc_on_global_path)

    # Sort by accuracy descending
    sorted_writers = sorted(flattened, key=lambda x: x[1], reverse=True)
    n_total = len(sorted_writers)
    n_bottom = max(1, int(bottom_frac * n_total))  # at least 1

    bottom_005_percent_writers = sorted_writers[-n_bottom:]

    # Randomly sample k (or fewer if not enough)
    n_sample = min(k, len(bottom_005_percent_writers))
    seed = int(str(seed))  # ✅ Ensure it's int-compatible
    random.seed(seed)  # You can choose any integer seed value
    selected_5_writers = random.sample(bottom_005_percent_writers, n_sample)
    selected_outliers = [selected_5_writers[i][0] for i in range(n_sample)]

    with open(selected_outliers_path, "w") as f:
        json.dump(selected_outliers, f, indent=2)
    NistLogger.info(f"Low Acc Writers: {len(selected_outliers)}, saved at {selected_outliers_path}")

    # ---- Plot 1: All clients, highlight bottom 0.2% ----
    all_accs = [acc for _, acc in sorted_writers]
    highlight_005_accs = [acc for _, acc in bottom_005_percent_writers]
    highlight_005_indices = [i for i, (_, acc) in enumerate(sorted_writers) if (_, acc) in bottom_005_percent_writers]

    plt.figure(figsize=(12, 6))
    plt.plot(all_accs, label="All None Contributor Writers", color='gray')
    plt.scatter(highlight_005_indices, highlight_005_accs, color='red', label='Bottom 0.05%', zorder=5)
    plt.title("Writer Accuracies with Bottom 0.05% Highlighted")
    plt.xlabel("Writer Index (sorted by accuracy)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outliers_result_path/"all_none_contributor_writers.png")

    # ---- Plot 2: Bottom 0.2% only, highlight selected 5 ----
    bottom_accs = [acc for _, acc in bottom_005_percent_writers]
    selected_indices = [i for i, (_, acc) in enumerate(bottom_005_percent_writers) if (_, acc) in selected_5_writers]
    selected_accs = [acc for i, (_, acc) in enumerate(bottom_005_percent_writers) if i in selected_indices]

    # Set global font size
    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(12, 6))
    plt.plot(bottom_accs, label="Bottom 0.05%", color='orange')
    plt.scatter(selected_indices, selected_accs, color='blue', label='Selected Clients', zorder=5)

    # Annotate each selected point
    for i, (x, y) in enumerate(zip(selected_indices, selected_accs)):
        plt.text(
            x, y,  # position
            f"{y:.2f}",  # text: show accuracy (2 decimals)
            fontsize=16,  # font size for labels
            ha='center',  # horizontal alignment
            va='bottom'  # vertical alignment
        )
    plt.title("Bottom 0.05% Writers with Selected Writers Highlighted")
    plt.xlabel("Writer Index (within bottom 0.05%)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outliers_result_path/"bottom_005_selected_writers.png")
    return selected_outliers


