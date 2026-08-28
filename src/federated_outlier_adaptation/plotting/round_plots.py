
def _plot_accuracies_over_rounds_sequential_impl(
    accuracies,
    global_clients_all_metrics_acc,
    global_clients_metric_acc,
    results_path
):
    # Unpack aggregated accuracies per round
    aggregated_all_clients, aggregated_global = zip(*accuracies)

    plt.figure(figsize=(10, 6))

    def plot_with_min_max(line, label, color):
        max_val = max(line)
        min_val = min(line)
        max_idx = line.index(max_val)
        min_idx = line.index(min_val)

        # Add max/min info to legend label
        label_with_stats = f"{label} (max={max_val:.4f}, min={min_val:.4f})"

        plt.plot(line, label=label_with_stats, color=color)
        plt.scatter([max_idx], [max_val], color=color, marker='o')
        plt.scatter([min_idx], [min_val], color=color, marker='x')

        # Optional: annotations
        plt.annotate(f"max: {max_val:.4f}", (max_idx, max_val),
                     textcoords="offset points", xytext=(0, 5),
                     ha='center', color=color)
        plt.annotate(f"min: {min_val:.4f}", (min_idx, min_val),
                     textcoords="offset points", xytext=(0, -15),
                     ha='center', color=color)

    # Main curves from accuracies
    plot_with_min_max(list(aggregated_all_clients), "Agg-Model on Clients data Acc", "green")
    plot_with_min_max(list(aggregated_global), "Agg-Model on Global data Acc", "red")

    # Constant reference lines (without global_metrics_acc)
    plt.axhline(
        y=global_clients_all_metrics_acc,
    color="blue",  # distinct color

        linestyle="--",
        label=f"Global+Clients on clients data Acc ({global_clients_all_metrics_acc:.4f})"
    )
    plt.axhline(
        y=global_clients_metric_acc,
        color="purple",
        linestyle="-.",  # dash-dot
        label=f"Global+Clients on global data Acc ({global_clients_metric_acc:.4f})"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Rounds (Sequential)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / "accuracies_over_rounds_sequential.png")
    plt.close()


def _plot_accuracies_over_rounds_all_together_impl(
    accuracies,
    global_clients_all_metrics_acc,
    global_clients_metric_acc,
    results_path
):
    # Unpack aggregated accuracies per round
    aggregated_all_clients, aggregated_global = zip(*accuracies)

    plt.figure(figsize=(10, 6))

    def plot_with_min_max(line, label, color):
        max_val = max(line)
        min_val = min(line)
        max_idx = line.index(max_val)
        min_idx = line.index(min_val)

        # Add max/min info to legend label
        label_with_stats = f"{label} (max={max_val:.4f}, min={min_val:.4f})"

        plt.plot(line, label=label_with_stats, color=color)
        plt.scatter([max_idx], [max_val], color=color, marker='o')
        plt.scatter([min_idx], [min_val], color=color, marker='x')

        # Optional: annotations
        plt.annotate(f"max: {max_val:.4f}", (max_idx, max_val),
                     textcoords="offset points", xytext=(0, 5),
                     ha='center', color=color)
        plt.annotate(f"min: {min_val:.4f}", (min_idx, min_val),
                     textcoords="offset points", xytext=(0, -15),
                     ha='center', color=color)

    # Main curves from accuracies
    plot_with_min_max(list(aggregated_all_clients), "Agg-Model on Clients data Acc", "green")
    plot_with_min_max(list(aggregated_global), "Agg-Model on Global data Acc", "red")

    # Constant reference lines (without global_metrics_acc)
    plt.axhline(
        y=global_clients_all_metrics_acc,
    color="blue",  # distinct color

        linestyle="--",
        label=f"Global+Clients on clients data Acc ({global_clients_all_metrics_acc:.4f})"
    )
    plt.axhline(
        y=global_clients_metric_acc,
        color="purple",
        linestyle="-.",  # dash-dot
        label=f"Global+Clients on global data Acc ({global_clients_metric_acc:.4f})"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Rounds (All Together)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / "accuracies_over_rounds_all_together.png")
    plt.close()


def _plot_accuracies_over_rounds_impl(
    accuracies,
    global_clients_all_metrics_acc: float,
    global_clients_metric_acc: float,
    results_path,
    scenario: str,
    metadata: str,
    tag: str | None = None,  # append e.g. agg name to the filename
):
    """
    accuracies: list[tuple(list_like, list_like)] => per-round (agg_all_clients_acc, agg_global_acc)
    """
    # Unpack aggregated accuracies per round
    aggregated_all_clients, aggregated_global = zip(*accuracies)

    plt.figure(figsize=(10, 6))

    def plot_with_min_max(line, label, color):
        line = list(line)
        max_val = max(line)
        min_val = min(line)
        max_idx = line.index(max_val)
        min_idx = line.index(min_val)

        # Add max/min info to legend label
        label_with_stats = f"{label} (max={max_val:.4f}, min={min_val:.4f})"

        plt.plot(line, label=label_with_stats, color=color)
        plt.scatter([max_idx], [max_val], color=color, marker='o')
        plt.scatter([min_idx], [min_val], color=color, marker='x')

        # Optional: annotations
        plt.annotate(f"max: {max_val:.4f}", (max_idx, max_val),
                     textcoords="offset points", xytext=(0, 5),
                     ha='center', color=color)
        plt.annotate(f"min: {min_val:.4f}", (min_idx, min_val),
                     textcoords="offset points", xytext=(0, -15),
                     ha='center', color=color)

    # Main curves from accuracies
    plot_with_min_max(aggregated_all_clients, "Accuracy on Clients data", "green")
    plot_with_min_max(aggregated_global, "Accuracy on Global data", "red")

    # Constant reference lines
    plt.axhline(
        y=global_clients_all_metrics_acc,
        color="blue",
        linestyle="--",
        label=f"Ground Truth Accuracy (Combined-Model) on Clients Data ({global_clients_all_metrics_acc:.4f})"
    )
    plt.axhline(
        y=global_clients_metric_acc,
        color="purple",
        linestyle="-.",
        label=f"Ground Truth Accuracy (Combined-Model) on Global Data ({global_clients_metric_acc:.4f})"
    )

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    title = f"Accuracy Over Rounds ({scenario}, {metadata})"
    if tag:
        title += f" • {tag}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fname = f"accuracies_over_rounds_{scenario}_{metadata}"
    if tag:
        fname += f"_{tag}"
    plt.savefig(results_path / f"{fname}.png")
    plt.close()


import threading

import matplotlib.pyplot as plt

#: pyplot keeps global state; the training driver calls these helpers from
#: several threads, so every helper runs under one lock.
_PYPLOT_LOCK = threading.RLock()

def _plot_overlay_accuracies_per_scenario_impl(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path,
    scenario: str,
    metadata: str,
    ref_clients_acc: float | None = None,   # constant per (scenario, metadata)
    ref_global_acc: float | None = None,    # constant per (scenario, metadata)
):
    """
    results_by_method: { agg_name: [(acc_clients_round0, acc_global_round0), ...], ... }
    Draws one figure overlaying all methods. Optionally draws reference lines once.
    """
    plt.figure(figsize=(11, 7))

    for agg_name, pairs in sorted(results_by_method.items()):
        acc_clients = [a for (a, _) in pairs]
        acc_global  = [b for (_, b) in pairs]
        plt.plot(acc_clients, label=f"{agg_name} (clients)", linestyle="-")
        plt.plot(acc_global,  label=f"{agg_name} (global)",  linestyle="--")

    if ref_clients_acc is not None:
        plt.axhline(ref_clients_acc, linestyle=":", label=f"Ref clients={ref_clients_acc:.4f}")
    if ref_global_acc is not None:
        plt.axhline(ref_global_acc, linestyle=":", label=f"Ref global={ref_global_acc:.4f}")

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title(f"All Aggregation Methods • {scenario}, {metadata}")
    plt.legend(ncols=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"overlay_all_methods_{scenario}_{metadata}.png")
    plt.close()


def plot_accuracies_over_rounds_sequential(*args, **kwargs):
    """Thread-safe wrapper; see :func:`_plot_accuracies_over_rounds_sequential_impl`."""
    with _PYPLOT_LOCK:
        return _plot_accuracies_over_rounds_sequential_impl(*args, **kwargs)

def plot_accuracies_over_rounds_all_together(*args, **kwargs):
    """Thread-safe wrapper; see :func:`_plot_accuracies_over_rounds_all_together_impl`."""
    with _PYPLOT_LOCK:
        return _plot_accuracies_over_rounds_all_together_impl(*args, **kwargs)

def plot_accuracies_over_rounds(*args, **kwargs):
    """Thread-safe wrapper; see :func:`_plot_accuracies_over_rounds_impl`."""
    with _PYPLOT_LOCK:
        return _plot_accuracies_over_rounds_impl(*args, **kwargs)

def plot_overlay_accuracies_per_scenario(*args, **kwargs):
    """Thread-safe wrapper; see :func:`_plot_overlay_accuracies_per_scenario_impl`."""
    with _PYPLOT_LOCK:
        return _plot_overlay_accuracies_per_scenario_impl(*args, **kwargs)
