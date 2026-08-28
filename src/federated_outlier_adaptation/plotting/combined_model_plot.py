"""
Training curve of the combined model (global writers + selected outliers).

Draws train/validation accuracy over epochs with the two test-accuracy reference
lines (global data and clients data) reported in the paper.
"""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from federated_outlier_adaptation import config


def plot_combined_model(
    json_path: Optional[Path] = None,
    clients_metric_json_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    font_size: int = 18,
    dpi: int = 200,
):
    """
    Plot training and validation accuracy curves with test accuracy references.

    Args:
        json_path: ``global_clients_global_metrics.json``.
        clients_metric_json_path: ``global_clients_all_outliers_metrics.json``.
        save_path: Destination PNG.
        font_size: Base font size applied to every text element.
        dpi: Figure resolution.
    """
    results_dir = config.GLOBAL_CLIENTS_RESULTS_DIR
    json_path = Path(json_path) if json_path else results_dir / "global_clients_global_metrics.json"
    clients_metric_json_path = (
        Path(clients_metric_json_path)
        if clients_metric_json_path
        else results_dir / "global_clients_all_outliers_metrics.json"
    )
    save_path = Path(save_path) if save_path else results_dir / "global_clients_training_plot.png"

    # --- Load data ---
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(clients_metric_json_path, "r") as f:
        clients_data = json.load(f)
    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]
    test_accuracy = data["test_accuracy"]
    clients_accuracy = clients_data["test_accuracy"]

    # --- Create figure first ---
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    # --- Plot curves ---
    ax.plot(train_accuracies, label="Train Accuracy", color="blue", linewidth=3)
    ax.plot(val_accuracies, label="Validation Accuracy", color="red", linewidth=3)

    # Thicker dashed test accuracy lines
    ax.axhline(
        y=test_accuracy,
        color="gray",
        linestyle=(0, (5, 5)),
        linewidth=4,
        label=f"Test Accuracy on Global Data: {test_accuracy:.4f}",
    )
    ax.axhline(
        y=clients_accuracy,
        color="green",
        linestyle=(0, (5, 5)),
        linewidth=4,
        label=f"Test Accuracy on Clients Data: {clients_accuracy:.4f}",
    )

    # --- Apply font sizes explicitly ---
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Accuracy", fontsize=font_size)
    ax.set_title("Training / Validation Accuracy over Epochs", fontsize=font_size + 2)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    legend = ax.legend(loc="lower right", frameon=True)
    for text in legend.get_texts():
        text.set_fontsize(font_size)

    ax.grid(True, linestyle="dotted", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    plot_combined_model()
