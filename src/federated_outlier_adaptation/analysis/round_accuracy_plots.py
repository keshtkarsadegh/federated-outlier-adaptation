"""Round-accuracy and overlay figure helpers with fixed reference baselines, used across the results root."""

import matplotlib.pyplot as plt






def plot_accuracies_over_rounds(
    accuracies,
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
        y=0.941972920696325,
        color="blue",
        linestyle="--",
        label=f"Ground Truth Accuracy (Combined-Model) on Clients Data ({0.941972920696325:.4f})"
    )
    plt.axhline(
        y=0.9957132050096663,
        color="purple",
        linestyle="-.",
        label=f"Ground Truth Accuracy (Combined-Model) on Global Data ({0.9957132050096663:.4f})"
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



def plot_overlay_accuracies_per_scenario(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path,
    scenario: str,
    metadata: str,

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

    plt.axhline(y=0.9957132050096663, color='r', linestyle='--',
                label=f"Test Accuracy on Global Data: {0.9957132050096663:.4f}")
    plt.axhline(y=0.941972920696325, color='r', linestyle='--',
                label=f"Test Accuracy on Ouliers Data: {0.941972920696325:.4f}")

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title(f"All Aggregation Methods • {scenario}, {metadata}")
    plt.legend(ncols=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"overlay_all_methods_{scenario}_{metadata}.png")
    plt.close()
