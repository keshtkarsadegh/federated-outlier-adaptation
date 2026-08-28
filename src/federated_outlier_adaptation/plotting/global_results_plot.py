"""Training curve of the global (server-side) model."""

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from federated_outlier_adaptation import config


def plot_global_training_progress(
    json_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    font_size: int = 12,
    dpi: int = 200,
):
    """
    Plot training and validation accuracy curves and show a test accuracy reference line.
    Font size is applied consistently to all text elements.
    """
    json_path = Path(json_path) if json_path else config.GLOBAL_RESULTS_DIR / "global_metrics.json"
    save_path = (
        Path(save_path) if save_path else config.GLOBAL_RESULTS_DIR / "global_training_plot.png"
    )

    # --- Load data ---
    with open(json_path, "r") as f:
        data = json.load(f)

    train_accuracies = data["train_accuracies"]
    val_accuracies = data["val_accuracies"]
    test_accuracy = data["test_accuracy"]

    # --- Create figure first ---
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    # --- Plot curves ---
    ax.plot(train_accuracies, label="Train Accuracy", color="blue", linewidth=3)
    ax.plot(val_accuracies, label="Validation Accuracy", color="red", linewidth=3)

    # Thicker dashed test accuracy line (ticker-style)
    ax.axhline(
        y=test_accuracy,
        color="green",
        linestyle=(0, (5, 5)),
        linewidth=4,
        label=f"Test Accuracy: {test_accuracy:.4f}"
    )

    # --- Apply font sizes explicitly ---
    ax.set_xlabel("Epoch", fontsize=font_size)
    ax.set_ylabel("Accuracy", fontsize=font_size)
    ax.set_title("Training / Validation Accuracy over Epochs", fontsize=font_size + 2)

    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # Legend
    legend = ax.legend(loc="lower right", frameon=True)
    for text in legend.get_texts():
        text.set_fontsize(font_size)

    # Grid
    ax.grid(True, linestyle="dotted", alpha=0.6)

    # --- Save ---
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    plot_global_training_progress(font_size=18)
