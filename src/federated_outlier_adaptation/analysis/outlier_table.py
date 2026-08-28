"""Overall-means table figure for the outlier clients, run on results/outliers/results."""

import json
import matplotlib.pyplot as plt
from pathlib import Path

from federated_outlier_adaptation import config

DEFAULT_RESULTS_SUBFOLDER = Path("outliers") / "results"


def plot_summary_table(json_path: Path, figsize=(8, 3), font_size=20, dpi=200,
                       output_path: str | Path | None = None, show: bool = False):
    """
    Plot the __overall_means__ section from outlier_summary.json as a table.
    The first column (metric names) has a darker gray background,
    white bold text, and a slightly larger font than the numeric column.
    ``output_path`` defaults to summary_table_plot.png next to ``json_path``.
    """
    json_path = Path(json_path)
    output_path = Path(output_path) if output_path else json_path.parent / "summary_table_plot.png"

    # --- Load data ---
    with open(json_path, "r") as f:
        data = json.load(f)

    summary = data.get("__overall_means__", None)
    if summary is None:
        raise KeyError("__overall_means__ not found in JSON file")

    # --- Prepare table data ---
    table_data = [
        ["Mean Accuracy on Global Data", f"{summary['mean_global_test_accuracy']:.3f}"],
        ["Mean Accuracy on Clients Data", f"{summary['mean_all_client_test_accuracies']:.3f}"],
        ["Mean Max Eval Accuracy", f"{summary['mean_best_eval_accuracy']:.3f}"],
    ]

    # --- Plot table ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
    )

    # --- Basic styling ---
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    # --- Adjust individual cell sizes and styles ---
    for (row, col), cell in table.get_celld().items():
        width, height = cell.get_width(), cell.get_height()
        if col == 0:  # first column
            cell.set_width(width * 2.35)
            cell.set_facecolor("#555555")  # darker gray
            text_obj = cell.get_text()
            text_obj.set_color("white")
            text_obj.set_fontsize(font_size * 1.2)
            text_obj.set_fontweight("bold")
        else:  # numeric column
            cell.set_width(width * 0.5)
            cell.get_text().set_fontsize(font_size)
            cell.get_text().set_fontweight("bold")

        cell.set_height(height * 1.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()


# === Usage ===
def plot_outlier_summary_table(results_dir: str | Path | None = None, show: bool = False) -> None:
    """
    Draw the overall-means table of the outlier training summary.

    ``results_dir`` defaults to ``config.RESULTS_DIR / "outliers" / "results"``,
    the folder this figure was produced from; ``outlier_summary.json`` is read
    from it and ``summary_table_plot.png`` is written back into it.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR / DEFAULT_RESULTS_SUBFOLDER

    plot_summary_table(root / "outlier_summary.json", show=show)


if __name__ == "__main__":
    plot_outlier_summary_table()
