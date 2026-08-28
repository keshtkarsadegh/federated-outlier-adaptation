"""Side-by-side clients/global accuracy heatmaps for the paper, run on results/final_result_DistillationTrainer_grid_search."""

# An earlier fixed-font version of the same figure is kept below for reference.
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # ----- Data -----
# clients_data = {
#     ("Sequential", "Weights"): {
#         "seq_fedavg_update": 0.9284,
#         "seq_fixed_ratio_update": 0.9458,
#         "seq_incremental_update_": 0.9052,
#         "seq_equal_update": 0.9033,
#     },
#     ("Sequential", "Delta"): {
#         "seq_delta_capped": 0.9439,
#         "seq_delta_fedavg_update": 0.9400,
#         "seq_delta_scaled": 0.9362,
#         "seq_delta_prograssive_update": 0.9110,
#     },
#     ("Concurrent", "Weights"): {
#         "con_capped_cgw": 0.9323,
#         "con_scaled_cgw": 0.9342,
#         "con_scaled_cw": 0.9439,
#         "con_capped_cw": 0.9304,
#         "con_weighted_cgw": 0.9304,
#         "con_weighted_cw": 0.9439,
#     },
#     ("Concurrent", "Delta"): {
#         "con_delta_capped_cgd": 0.9342,
#         "con_delta_weighted_cgd": 0.9400,
#         "con_delta_scaled_cgd": 0.9323,
#     },
# }
#
# global_data = {
#     ("Sequential", "Weights"): {
#         "seq_fedavg_update": 0.9916,
#         "seq_fixed_ratio_update": 0.9906,
#         "seq_incremental_update_": 0.9891,
#         "seq_equal_update": 0.9845,
#     },
#     ("Sequential", "Delta"): {
#         "seq_delta_capped": 0.9925,
#         "seq_delta_fedavg_update": 0.9924,
#         "seq_delta_scaled": 0.9908,
#         "seq_delta_prograssive_update": 0.9880,
#     },
#     ("Concurrent", "Weights"): {
#         "con_capped_cgw": 0.9934,
#         "con_scaled_cgw": 0.9928,
#         "con_scaled_cw": 0.9918,
#         "con_capped_cw": 0.9919,
#         "con_weighted_cgw": 0.9918,
#         "con_weighted_cw": 0.9894,
#     },
#     ("Concurrent", "Delta"): {
#         "con_delta_capped_cgd": 0.9917,
#         "con_delta_weighted_cgd": 0.9906,
#         "con_delta_scaled_cgd": 0.9906,
#     },
# }
#
# cols = pd.MultiIndex.from_product(
#     [["Sequential", "Concurrent"], ["Weights", "Delta"]],
#     names=["Scenario", "UpdateType"]
# )
#
# df_clients = pd.DataFrame(clients_data, columns=cols)
# df_global = pd.DataFrame(global_data, columns=cols)
#
# # ----- Plot -----
# fig, axes = plt.subplots(1, 2, figsize=(22, 10))
#
# def draw_heatmap(df, ax, cmap, title, show_yticks=True):
#     sns.heatmap(df, annot=True, fmt=".4f", cmap=cmap, cbar=False,
#                 ax=ax, annot_kws={"size": 11}, yticklabels=show_yticks)
#
#     # remove duplicate right-side labels
#     ax.tick_params(axis="y", which="both", right=False, labelright=False)
#
#     if not show_yticks:
#         ax.set_yticklabels([])
#
#     ax.set_title(title, fontsize=16, fontweight="bold")
#     ax.set_xticks([])
#
#     xpos = range(len(df.columns))
#     level0 = df.columns.get_level_values(0)
#     level1 = df.columns.get_level_values(1)
#
#     # bottom headers (Weights/Delta)
#     ax.set_xticks([x + 0.5 for x in xpos], labels=level1, rotation=0, fontsize=12)
#
#     # top headers (Sequential/Concurrent)
#     for group in ["Sequential", "Concurrent"]:
#         indices = [i for i, val in enumerate(level0) if val == group]
#         if indices:
#             start, end = min(indices), max(indices) + 1
#             ax.annotate(group,
#                         xy=((start + end) / 2, -0.25),  # slightly lower for better centering
#                         xycoords=("data", "axes fraction"),
#                         ha="center", va="bottom", fontsize=13, fontweight="bold")
#             ax.axvline(x=end, color="black", linewidth=1.4)
#
#     # inner dividers between Weights and Delta
#     for idx in range(1, len(level1)):
#         if level1[idx] == "Delta":
#             ax.axvline(x=idx, color="black", linewidth=1.0)
#
#     if show_yticks:
#         ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
#
#     # remove that annoying MultiIndex axis label
#     ax.set_xlabel("")
#
# # Left heatmap (with method names)
# draw_heatmap(df_clients, axes[0], "Blues", "Accuracies on Clients data", show_yticks=True)
# # Right heatmap (without method names)
# draw_heatmap(df_global, axes[1], "Greens", "Accuracies on Global data", show_yticks=False)
#
# plt.tight_layout()
# plt.savefig("accuracies_final.png", dpi=300, bbox_inches="tight")
# plt.show()

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from federated_outlier_adaptation import config

DEFAULT_RESULTS_SUBFOLDER = "final_result_DistillationTrainer_grid_search"

# ----- Data -----
clients_data = {
    ("Sequential", "Weights"): {
        "seq_fedavg_update": 0.9284,
        "seq_fixed_ratio_update": 0.9458,
        "seq_incremental_update_": 0.9052,
        "seq_equal_update": 0.9033,
    },
    ("Sequential", "Delta"): {
        "seq_delta_capped": 0.9439,
        "seq_delta_fedavg_update": 0.9400,
        "seq_delta_scaled": 0.9362,
        "seq_delta_prograssive_update": 0.9110,
    },
    ("Concurrent", "Weights"): {
        "con_capped_cgw": 0.9323,
        "con_scaled_cgw": 0.9342,
        "con_scaled_cw": 0.9439,
        "con_capped_cw": 0.9304,
        "con_weighted_cgw": 0.9304,
        "con_weighted_cw": 0.9439,
    },
    ("Concurrent", "Delta"): {
        "con_delta_capped_cgd": 0.9342,
        "con_delta_weighted_cgd": 0.9400,
        "con_delta_scaled_cgd": 0.9323,
    },
}

global_data = {
    ("Sequential", "Weights"): {
        "seq_fedavg_update": 0.9916,
        "seq_fixed_ratio_update": 0.9906,
        "seq_incremental_update_": 0.9891,
        "seq_equal_update": 0.9845,
    },
    ("Sequential", "Delta"): {
        "seq_delta_capped": 0.9925,
        "seq_delta_fedavg_update": 0.9924,
        "seq_delta_scaled": 0.9908,
        "seq_delta_prograssive_update": 0.9880,
    },
    ("Concurrent", "Weights"): {
        "con_capped_cgw": 0.9934,
        "con_scaled_cgw": 0.9928,
        "con_scaled_cw": 0.9918,
        "con_capped_cw": 0.9919,
        "con_weighted_cgw": 0.9918,
        "con_weighted_cw": 0.9894,
    },
    ("Concurrent", "Delta"): {
        "con_delta_capped_cgd": 0.9917,
        "con_delta_weighted_cgd": 0.9906,
        "con_delta_scaled_cgd": 0.9906,
    },
}

def draw_heatmap(df, ax, cmap, title, show_yticks=True,
                 fontsize_title=18, fontsize_top=14, fontsize_bottom=12,
                 fontsize_numbers=11, fontsize_methods=11):
    sns.heatmap(df, annot=True, fmt=".4f", cmap=cmap, cbar=False,
                ax=ax, annot_kws={"size": fontsize_numbers}, yticklabels=show_yticks)

    # remove duplicate right-side labels
    ax.tick_params(axis="y", which="both", right=False, labelright=False)

    if not show_yticks:
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
    ax.set_xticks([])

    xpos = range(len(df.columns))
    level0 = df.columns.get_level_values(0)
    level1 = df.columns.get_level_values(1)

    # bottom headers (Weights/Delta)
    ax.set_xticks([x + 0.5 for x in xpos], labels=level1, rotation=0, fontsize=fontsize_bottom)

    # top headers (Sequential/Concurrent)
    for group in ["Sequential", "Concurrent"]:
        indices = [i for i, val in enumerate(level0) if val == group]
        if indices:
            start, end = min(indices), max(indices) + 1
            ax.annotate(group,
                        xy=((start + end) / 2, -0.25),
                        xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=fontsize_top, fontweight="bold")
            ax.axvline(x=end, color="black", linewidth=1.4)

    # inner dividers between Weights and Delta
    for idx in range(1, len(level1)):
        if level1[idx] == "Delta":
            ax.axvline(x=idx, color="black", linewidth=1.0)

    if show_yticks:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsize_methods)

    # remove that annoying MultiIndex axis label
    ax.set_xlabel("")

def plot_min_max_accuracy_heatmaps(results_dir: str | Path | None = None,
                                   output_path: str | Path | None = None,
                                   show: bool = False) -> None:
    """
    Draw the paired clients/global accuracy heatmaps from the tabulated results above.

    ``results_dir`` defaults to ``config.RESULTS_DIR / "final_result_DistillationTrainer_grid_search"``,
    the folder this figure belongs to; ``output_path`` defaults to
    ``accuracies_with_fonts.png`` inside that folder.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR / DEFAULT_RESULTS_SUBFOLDER
    output_path = Path(output_path) if output_path else root / "accuracies_with_fonts.png"

    cols = pd.MultiIndex.from_product(
        [["Sequential", "Concurrent"], ["Weights", "Delta"]],
        names=["Scenario", "UpdateType"]
    )

    df_clients = pd.DataFrame(clients_data, columns=cols)
    df_global = pd.DataFrame(global_data, columns=cols)

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Left heatmap (with method names)
    draw_heatmap(df_clients, axes[0], "Blues", "Accuracies on Clients data",
                 show_yticks=True, fontsize_title=18, fontsize_top=18,
                 fontsize_bottom=18, fontsize_numbers=18, fontsize_methods=18)

    # Right heatmap (without method names)
    draw_heatmap(df_global, axes[1], "Greens", "Accuracies on Global data",
                 show_yticks=False, fontsize_title=18, fontsize_top=18,
                 fontsize_bottom=18, fontsize_numbers=18, fontsize_methods=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()


if __name__ == "__main__":
    plot_min_max_accuracy_heatmaps()
