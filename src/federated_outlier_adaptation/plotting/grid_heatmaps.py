#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import List, Union

# -------------------------------------------------
# Extract hyperparameter values from method strings
# -------------------------------------------------
def label_values(labels: Union[str, List[str]], df) -> List[tuple]:
    values = (
        df["method"]
        .str.extractall(r"([0-9.]+)")
        .unstack()
        .dropna(axis=1, how="all")
        .astype(float)
        .apply(lambda row: row.dropna().tolist(), axis=1)
        .tolist()
    )

    if isinstance(labels, str):
        labels = [labels]
        values = [[v] for v in values]

    client_scores = df["clients"].tolist()
    global_scores = df["global"].tolist()

    return [
        (
            ", ".join(f"{lab}:{val}" for lab, val in zip(labels, row)),
            client,
            glob,
        )
        for row, client, glob in zip(values, client_scores, global_scores)
    ]

# -------------------------------------------------
# Subfolder ordering
# -------------------------------------------------
def order_subfolders(subfolders):
    order = ["sequential_weights","sequential_delta",
             "concurrent_weights", "concurrent_delta"]
    return sorted(subfolders, key=lambda sf: order.index(os.path.basename(sf).lower()))

# -------------------------------------------------
# Main plotting function
# -------------------------------------------------
def plot_grid_heatmaps(root_folder: str,
                                  fig_title: str,
                                  output_name: str = "heatmaps.png",
                                  height=3):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    subfolders = order_subfolders(subfolders)

    all_clients_vals, all_global_vals = [], []
    for sf in subfolders:
        csvs = [f for f in os.listdir(sf) if f.endswith(".csv")]
        if not csvs:
            continue
        df = pd.read_csv(os.path.join(sf, csvs[0]))
        if not {"clients", "global"}.issubset(df.columns):
            continue
        all_clients_vals.extend(df["clients"].tolist())
        all_global_vals.extend(df["global"].tolist())

    # normalization for colormaps
    clients_min, clients_max = min(all_clients_vals), max(all_clients_vals)
    global_min, global_max   = min(all_global_vals), max(all_global_vals)

    clients_cmap = plt.get_cmap("Blues")
    clients_norm = colors.Normalize(vmin=clients_min, vmax=clients_max)
    global_cmap  = plt.get_cmap("GnBu")
    global_norm  = colors.Normalize(vmin=global_min, vmax=global_max)

    fig, axes = plt.subplots(1, len(subfolders), figsize=(4.5 * len(subfolders), height))
    if len(subfolders) == 1:
        axes = [axes]

    for ax, sf in zip(axes, subfolders):
        name = os.path.basename(sf)
        pretty = " ".join(p.capitalize() for p in name.split("_"))

        csvs = [f for f in os.listdir(sf) if f.endswith(".csv")]
        if not csvs:
            continue
        df = pd.read_csv(os.path.join(sf, csvs[0]))
        if not {"clients", "global"}.issubset(df.columns):
            continue

        # --- Extract labels ---
        if df["method"].str.startswith("aligned_").any():
            triples = label_values(["β"], df)
            col_widths = [2.0, 1.5, 1.5]  # Params wide, Clients & Global 15% bigger


        elif df["method"].str.startswith("prox_").any():
            triples = label_values(["λ"], df)
            col_widths = [2.0, 1.5, 1.5]  # Params wide, Clients & Global 15% bigger

        elif df["method"].str.startswith("ewc_").any():
            triples = label_values(["λ"], df)
            col_widths = [2.0, 1.5, 1.5]  # Params wide, Clients & Global 15% bigger

        elif df["method"].str.startswith("distill_T").any():
            triples = label_values(["T", "α"], df)
            col_widths = [2.0, 0.8, 0.8]  # Params wide, Clients & Global 15% bigger

        elif df["method"].str.startswith("logit_").any():
            triples = label_values(["λ"], df)
            col_widths = [2.0, 1.5, 1.5]  # Params wide, Clients & Global 15% bigger

        elif df["method"].str.startswith("T_").any():
            triples = label_values(["T", "α", "λ"], df)
            col_widths = [2.0, 0.7, 0.7]  # Params wide, Clients & Global 15% bigger
        else:
            triples = [("", c, g) for c, g in zip(df["clients"], df["global"])]
            col_widths = [2.0, 2, 2]  # Params wide, Clients & Global 15% bigger

        labels, clients_vals, global_vals = zip(*triples)

        # --- Custom column widths ---

        # --- Custom column widths ---
        # col_widths = [2.0, 0.8, 0.8]  # Params wide, Clients & Global 15% bigger
        col_edges = np.cumsum([0] + col_widths)
        row_edges = np.arange(0, len(clients_vals) + 1)
        col_centers = [(col_edges[i] + col_edges[i + 1]) / 2 for i in range(len(col_widths))]

        # --- Draw Params column (gray) ---
        ax.pcolormesh([col_edges[0], col_edges[1]], row_edges,
                      np.zeros((len(clients_vals), 1)),
                      cmap=colors.ListedColormap([[0.9, 0.9, 0.9]]),
                      shading="auto")
        # --- Draw Params column (white) ---
        ax.pcolormesh([col_edges[0], col_edges[1]], row_edges,
                      np.zeros((len(clients_vals), 1)),
                      cmap=colors.ListedColormap([[1.0, 1.0, 1.0]]),  # white background
                      shading="auto")

        # --- Draw Clients column (heatmap) ---
        ax.pcolormesh([col_edges[1], col_edges[2]], row_edges,
                      np.array(clients_vals).reshape(len(clients_vals), 1),
                      cmap=clients_cmap, norm=clients_norm, shading="auto")

        # --- Draw Global column (heatmap) ---
        ax.pcolormesh([col_edges[2], col_edges[3]], row_edges,
                      np.array(global_vals).reshape(len(global_vals), 1),
                      cmap=global_cmap, norm=global_norm, shading="auto")


        for row_idx, (lab, cval, gval) in enumerate(zip(labels, clients_vals, global_vals)):
            y_center = row_idx + 0.5  # correct center

            # Params text
            txtcol =  "black"
            ax.text(col_centers[0], y_center, lab,
                    ha="center", va="center", fontsize=10,fontweight="bold", color="grey")

            # Clients text
            c_color = clients_cmap(clients_norm(cval))
            txtcol = "white" if (0.299 * c_color[0] + 0.587 * c_color[1] + 0.114 * c_color[2]) < 0.5 else "black"
            ax.text(col_centers[1], y_center, f"{cval:.4f}",
                    ha="center", va="center", fontsize=10, color=txtcol)

            # Global text
            g_color = global_cmap(global_norm(gval))
            txtcol = "white" if (0.299 * g_color[0] + 0.587 * g_color[1] + 0.114 * g_color[2]) < 0.5 else "black"
            ax.text(col_centers[2], y_center, f"{gval:.4f}",
                    ha="center", va="center", fontsize=10, color=txtcol)

        # ensure correct orientation
        ax.set_ylim(0, len(clients_vals))
        ax.invert_yaxis()

        # --- Formatting ---
        ax.set_xticks(col_centers)
        ax.set_xticklabels(["Params", "Clients", "Global"], fontsize=9)
        ax.set_xlim(0, sum(col_widths))
        ax.set_ylim(0, len(clients_vals))
        ax.set_yticks(np.arange(0.5, len(clients_vals) + 0.5))
        ax.set_yticklabels(range(len(clients_vals)), fontsize=7)
        ax.set_title(pretty, fontsize=11, pad=12)

        # Vertical separators
        for x in col_edges[1:-1]:
            ax.axvline(x, color="black", linewidth=1)

    plt.subplots_adjust(wspace=0.25)
    save_path = os.path.join(root_folder, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Heatmap saved at: {save_path}")

# -------------------------------------------------
# Grid searches rendered for the paper
# -------------------------------------------------
GRID_SEARCHES = [
    {"path": "prox_grid_search", "name": "Proximal"},
    {"path": "logit_consistency_grid_search", "name": "Logistic Consistency"},
    {"path": "aligned_feature_grid_search", "name": "Aligned Feature"},
    {"path": "distillation_grid_search", "name": "Knowledge Distillation"},
    {"path": "ewc_grid_search", "name": "Elastic Weight Consolidation"},
    {
        "path": "distil_ewc_grid_search",
        "name": "Knowledge Distillation + Elastic Weight Consolidation",
    },
    {
        "path": "single_outlier_distillation_grid_search",
        "name": "Knowledge Distillation (Single Outlier)",
    },
    {
        "path": "double_outlier_distillation_grid_search",
        "name": "Knowledge Distillation (Double Outlier)",
    },
    {
        "path": "dual_outlier_distillation_grid_search",
        "name": "Knowledge Distillation (Dual Outlier)",
    },
]


def plot_all_grid_heatmaps(results_dir=None, height=2):
    """Render one heatmap figure per grid search under ``results_dir``."""
    from federated_outlier_adaptation import config

    root = Path(results_dir) if results_dir else config.RESULTS_DIR
    for entry in GRID_SEARCHES:
        folder = root / entry["path"]
        if not folder.is_dir():
            print(f"Skipping missing grid search folder: {folder}")
            continue
        plot_grid_heatmaps(
            str(folder),
            f"Top Accuracies on (Clients and Global Data) for {entry['name']} Grid Search",
            height=height,
        )


if __name__ == "__main__":
    plot_all_grid_heatmaps()
