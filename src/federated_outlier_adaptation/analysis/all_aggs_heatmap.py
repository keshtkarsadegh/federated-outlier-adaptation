"""Four-panel metric heatmap over all aggregation methods, run on results/all_aggs_fl_BaseTrainer_grid_search."""

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from federated_outlier_adaptation import config

DEFAULT_RESULTS_SUBFOLDER = "all_aggs_fl_BaseTrainer_grid_search"


def generate_heatmap_for_all_aggs(input_file=None, output_path=None, results_dir=None):
    """
    Draw one column heatmap per summary metric for every aggregation method.

    ``results_dir`` defaults to ``config.RESULTS_DIR / "all_aggs_fl_BaseTrainer_grid_search"``,
    the folder this figure was produced from.  ``input_file`` and ``output_path``
    default to ``aggregated_results.csv`` and ``aggregated_results_large.png``
    inside that folder.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR / DEFAULT_RESULTS_SUBFOLDER
    input_file = Path(input_file) if input_file else root / "aggregated_results.csv"
    output_path = Path(output_path) if output_path else root / "aggregated_results_large.png"

    # --- Load CSV ---
    df = pd.read_csv(input_file, header=None, names=[
        "scenario", "metadata", "agg_method_name",
        "min_acc_on_global_data", "max_acc_on_clients_data",
        "adaptation_stability", "global_knowledge_drop"
    ])

    # Remove accidental header-like row if any
    df = df[df["agg_method_name"] != "agg_method_name"]

    # --- Ensure numeric columns ---
    numeric_cols = [
        "min_acc_on_global_data",
        "max_acc_on_clients_data",
        "adaptation_stability",
        "global_knowledge_drop",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={
        "min_acc_on_global_data": "Min Acc on Global Data",
        "max_acc_on_clients_data": "Max Acc on Clients Data",
        "adaptation_stability": "Adaptation Stability",
        "global_knowledge_drop": "Global Knowledge Drop"
    })
    numeric_cols = [   "Min Acc on Global Data",
     "Max Acc on Clients Data",
    "Adaptation Stability",
    "Global Knowledge Drop"
    ]
    # --- Reorder: sequential methods last ---
    seq_rows = df[df["agg_method_name"].str.startswith("seq")]
    non_seq_rows = df[~df["agg_method_name"].str.startswith("seq")]
    df = pd.concat([non_seq_rows, seq_rows])

    # --- Styling ---
    sns.set(font_scale=2.5)
    annot_fontsize = 32
    label_fontsize = 32
    title_fontsize = 32

    # --- Large figure size ---
    n_methods = len(df)
    fig_width = 36
    fig_height = max(18, n_methods * 1.2)  # scale with rows
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(fig_width, fig_height), sharey=True)

    # --- Plot ---
    for i, col in enumerate(numeric_cols):
        sns.heatmap(
            df.set_index("agg_method_name")[[col]],
            annot=True, fmt=".2f",
            cmap="viridis",
            cbar=False,
            ax=axes[i],
            annot_kws={"size": annot_fontsize, "weight": "bold"},
            linewidths=0.6, linecolor="gray"
        )

        # axes[i].set_title(col.replace("_", " ").title(), fontsize=title_fontsize, weight="bold")
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")

        # Tick labels
        axes[i].tick_params(axis="x", labelsize=label_fontsize, width=2, pad=12)
        axes[i].tick_params(axis="y", labelsize=label_fontsize, width=2, pad=12)
        for tick in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            tick.set_fontweight("bold")

        # Remove clutter
        axes[i].tick_params(top=False, labeltop=False)
        axes[i].spines["top"].set_visible(False)
        axes[i].xaxis.tick_bottom()

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    generate_heatmap_for_all_aggs()
