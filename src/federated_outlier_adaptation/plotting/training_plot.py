from matplotlib import pyplot as plt


def plot_accuracies_over_rounds(
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
