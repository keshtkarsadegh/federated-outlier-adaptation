"""Per-client training/evaluation curves for the outlier clients, run on results/outliers/results."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

from federated_outlier_adaptation import config

DEFAULT_RESULTS_SUBFOLDER = Path("outliers") / "results"


def plot_client_training_progress(
    input_path: Path,
    output_dir: Path | None = None,
    font_size: int = 24,
    dpi: int = 250,
    figsize=(12, 8),
    extra_y_margin: float = 0.02,  # expand y-axis range by ±2%
):
    """
    Plot training and evaluation curves for each client.
    Keeps true values but increases vertical space (y-limits)
    to better distinguish close test accuracies.
    """

    with open(input_path, "r") as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = input_path.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "legend.fontsize": font_size - 1,
    })

    for client_id, data in results.items():
        if client_id.startswith("__"):
            continue

        train_acc = data["train_accuracies"]
        eval_acc = data["eval_accuracies"]
        global_acc = data["global_test_accuracy"]
        all_clients_acc = data["all_client_test_accuracies"]

        label_train = f"Train Accuracy (max={train_acc[-1]:.3f})"
        label_eval = f"Eval Accuracy (max={eval_acc[-1]:.3f})"
        label_global = f"Test Accuracy on Global Data  ({global_acc:.3f})"
        label_all_clients = f"Test Accuracy on Clients Data ({all_clients_acc:.3f})"

        fig, ax = plt.subplots(figsize=figsize)

        # --- Plot curves ---
        ax.plot(train_acc, label=label_train, linewidth=2.0, color="#1f77b4")
        ax.plot(eval_acc, label=label_eval, linewidth=2.0, color="#ff7f0e")

        # --- Reference lines ---
        ax.axhline(y=global_acc, color="red", linestyle="--", linewidth=2.2, label=label_global)
        ax.axhline(y=all_clients_acc, color="blue", linestyle=":", linewidth=2.2, alpha=0.9, label=label_all_clients)

        # --- Dynamic vertical scaling ---
        all_values = train_acc + eval_acc + [global_acc, all_clients_acc]
        y_min, y_max = min(all_values), max(all_values)
        margin = (y_max - y_min) * 0.5 + extra_y_margin  # expand upper/lower range
        y_min = max(0, y_min - margin)
        y_max = min(1.05, y_max + margin)
        ax.set_ylim(y_min, y_max)

        # --- Labels and styling ---
        ax.set_title(f"Client {client_id}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Accuracy")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", frameon=True)

        save_path = output_dir / f"{client_id}_training_plot.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)

        print(f"✅ Saved plot for {client_id}: {save_path}")



def plot_outlier_training_progress(results_dir: str | Path | None = None) -> None:
    """
    Plot the training progress of every outlier client from the stored run results.

    ``results_dir`` defaults to ``config.RESULTS_DIR / "outliers" / "results"``,
    the folder these figures were produced from; the input JSON and the ``plots``
    output directory are resolved inside it.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR / DEFAULT_RESULTS_SUBFOLDER

    plot_client_training_progress(
        input_path=root / "outlier_training_results.json"
    )


if __name__ == "__main__":
    plot_outlier_training_progress()
