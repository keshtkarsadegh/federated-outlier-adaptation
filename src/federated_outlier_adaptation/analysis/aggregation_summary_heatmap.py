"""Per-grid-search summary heatmaps (top accuracies on clients/global data), run on the results root."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from federated_outlier_adaptation import config

# --- Helper: extract numeric parameter values from "method" strings ---
def extract_param_values(df):
    values = (
        df["method"]
        .str.extractall(r"([0-9.]+)")
        .unstack()
        .dropna(axis=1, how="all")
        .astype(float)
    )
    param_cols = [f"param_{i}" for i in range(values.shape[1])]
    values.columns = param_cols
    df = pd.concat([df.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    return df, param_cols

# --- Folder ordering ---
def order_subfolders(subfolders):
    order = ["sequential_weights","sequential_delta","concurrent_weights","concurrent_delta"]
    return sorted(subfolders, key=lambda sf: order.index(os.path.basename(sf).lower()))

# --- Main plotting function ---
def plot_heatmaps_from_subfolders(root_folder: str | Path,
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

        df, param_cols = extract_param_values(df)
        num_params = len(param_cols)

        # Guess Greek symbols by number of params (optional mapping)
        greek_map = {0: "β", 1: "λ", 2: "T", 3: "α"}
        param_labels = []
        for i, col in enumerate(param_cols):
            # try to infer symbol from method name
            if "T" in df["method"].iloc[0]:
                symbols = ["T", "α", "λ"][:num_params]
            elif "aligned_" in df["method"].iloc[0]:
                symbols = ["β"]
            elif "prox_" in df["method"].iloc[0]:
                symbols = ["λ"]
            elif "ewc_" in df["method"].iloc[0]:
                symbols = ["λ"]
            elif "logit_" in df["method"].iloc[0]:
                symbols = ["λ"]
            else:
                symbols = [f"P{i+1}" for i in range(num_params)]
            param_labels = symbols
        xticklabels = param_labels + ["Clients", "Global"]

        clients_vals = df["clients"].values
        global_vals = df["global"].values

        col_widths = [1.2] * num_params + [1.5, 1.5]
        col_edges = np.cumsum([0] + col_widths)
        row_edges = np.arange(0, len(df) + 1)
        col_centers = [(col_edges[i] + col_edges[i + 1]) / 2 for i in range(len(col_widths))]

        # --- Draw parameter columns (gray) ---
        for i in range(num_params):
            ax.pcolormesh([col_edges[i], col_edges[i+1]], row_edges,
                          np.zeros((len(df), 1)),
                          cmap=colors.ListedColormap([[0.97, 0.97, 0.97]]),
                          shading="auto")

        # --- Clients & Global heatmaps ---
        ax.pcolormesh([col_edges[num_params], col_edges[num_params+1]], row_edges,
                      np.array(clients_vals).reshape(len(df), 1),
                      cmap=clients_cmap, norm=clients_norm, shading="auto")
        ax.pcolormesh([col_edges[num_params+1], col_edges[num_params+2]], row_edges,
                      np.array(global_vals).reshape(len(df), 1),
                      cmap=global_cmap, norm=global_norm, shading="auto")

        # --- Text annotations ---
        for row_idx in range(len(df)):
            y_center = row_idx + 0.5
            # Parameter numbers
            for i, pcol in enumerate(param_cols):
                val = df[pcol].iloc[row_idx]
                before, after = str(val).split('.')
                if "T" in df["method"].iloc[0]:
                    if int(before) != 0:
                        ax.text(col_centers[i], y_center, f"{val:.1f}".rstrip('0').rstrip('.'),
                                ha="center", va="center", fontsize=13, color="black")
                    else:
                        ax.text(col_centers[i], y_center, f"{val:.3f}",
                                ha="center", va="center", fontsize=13, color="black")
                else:
                    ax.text(col_centers[i], y_center, f"{val:.3f}",
                            ha="center", va="center", fontsize=14, color="black")
            # Clients
            cval = clients_vals[row_idx]
            c_col = clients_cmap(clients_norm(cval))
            c_txt = "white" if (0.299*c_col[0]+0.587*c_col[1]+0.114*c_col[2]) < 0.5 else "black"
            ax.text(col_centers[num_params], y_center, f"{cval:.4f}",
                    ha="center", va="center", fontsize=13, color=c_txt)
            # Global
            gval = global_vals[row_idx]
            g_col = global_cmap(global_norm(gval))
            g_txt = "white" if (0.299*g_col[0]+0.587*g_col[1]+0.114*g_col[2]) < 0.5 else "black"
            ax.text(col_centers[num_params+1], y_center, f"{gval:.4f}",
                    ha="center", va="center", fontsize=13, color=g_txt)

        # --- Grid lines and formatting ---
        ax.set_ylim(0, len(df))
        ax.invert_yaxis()
        ax.set_xticks(col_centers)
        ax.set_xticklabels(xticklabels, fontsize=12)
        ax.set_xlim(0, sum(col_widths))
        ax.set_yticks(np.arange(0.5, len(df)+0.5))
        ax.set_yticklabels(range(len(df)), fontsize=12)
        ax.set_title(pretty, fontsize=14, pad=16)

        # Draw table lines between *all* columns
        for x in col_edges[1:-1]:
            ax.axvline(x, color="black", linewidth=1)
        # Horizontal lines between rows
        for y in row_edges[1:-1]:
            ax.axhline(y, color="black", linewidth=0.5)

    plt.subplots_adjust(wspace=0.25)
    save_path = os.path.join(root_folder, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Heatmap saved at: {save_path}")


# -------------------------------------------------
# Batch entry point
# -------------------------------------------------
def plot_all_grid_search_summary_heatmaps(results_dir: str | Path | None = None) -> None:
    """
    Draw the summary heatmap of every grid search listed below.

    ``results_dir`` defaults to the results root (``config.RESULTS_DIR``); each
    grid search folder is resolved relative to it and every figure is written
    next to the data it summarises.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR

    args = [
        {"path":"prox_grid_search","name":"Proximal"},
        {"path":"logit_consistency_grid_search","name":"Logistic Consistency"},
        {"path":"aligned_feature_grid_search","name":"Aligned Feature"},
        {"path":"distillation_grid_search","name":"Knowledge Distillation"},
        {"path":"ewc_grid_search","name":"Elastic Weight Consolidation"},
        {"path":"distil_ewc_grid_search","name":"Knowledge Distillation + Elastic Weight Consolidation"},
    {"path":"single_outlier_distillation_grid_search","name":"Knowledge Distillation (Single Outlier)"},
        {"path":"double_outlier_distillation_grid_search","name":"Knowledge Distillation (Two Outlier)"},
        {"path":"dual_outlier_distillation_grid_search","name":"Knowledge Distillation (Dual Outlier)"},]

    for arg in args:
        plot_heatmaps_from_subfolders(
            root / arg["path"],
            f"Top Accuracies on (Clients and Global Data) for {arg['name']} Grid Search",
            height=2
        )


if __name__ == "__main__":
    plot_all_grid_search_summary_heatmaps()
