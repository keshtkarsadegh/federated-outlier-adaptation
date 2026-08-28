"""Two-column (clients/global) grid-search heatmaps per scenario, run on the results root."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from typing import List, Union

from federated_outlier_adaptation import config


def label_values(labels: Union[str, List[str]], df) -> List[tuple]:
    # extract numeric values
    values = (
        df["method"]
        .str.extractall(r"([0-9.]+)")   # extract all numbers
        .unstack()                      # reshape so each row gets its numbers
        .dropna(axis=1, how="all")      # drop empty columns
        .astype(float)                  # convert to floats
        .apply(lambda row: row.dropna().tolist(), axis=1)  # row-wise list of floats
        .tolist()
    )

    # normalize labels into a list
    if isinstance(labels, str):
        labels = [labels]
        # wrap single values in lists to match structure
        values = [[v] for v in values]

    client_scores = df["clients"].tolist()
    global_scores = df["global"].tolist()

    # return triples (labeled_string, clients, global)
    return [
        (
            ",".join(f"{lab}:{val}" for lab, val in zip(labels, row)),
            client,
            glob,
        )
        for row, client, glob in zip(values, client_scores, global_scores)
    ]




def order_subfolders(subfolders):
    order = ["sequential_weights","sequential_delta",
             "concurrent_weights", "concurrent_delta"]
    return sorted(subfolders, key=lambda sf: order.index(os.path.basename(sf).lower()))

def plot_heatmaps_from_subfolders(root_folder: str | Path,
                                  fig_title: str,
                                  output_name: str = "heatmaps.png",height=3):
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    subfolders = order_subfolders(subfolders)

    # --- Collect all values (including concurrent_delta) ---
    all_clients_vals, all_global_vals = [], []
    all_parameters_vals=[]
    for sf in subfolders:
        csvs = [f for f in os.listdir(sf) if f.endswith(".csv")]
        if not csvs:
            continue
        df = pd.read_csv(os.path.join(sf, csvs[0]))
        if not {"clients", "global"}.issubset(df.columns):
            continue
        all_clients_vals.extend(df["clients"].tolist())
        all_global_vals.extend(df["global"].tolist())
        if df["method"].startswith("aliened_"):
            all_parameters_vals.extend(label_values("β",df))
        elif df["method"].startswith("prox_"):
            all_parameters_vals.extend(label_values("λ",df))
        elif df["method"].startswith("ewc_"):
            all_parameters_vals.extend(label_values("λ",df))
        elif df["method"].startswith("distil_T"):
            all_parameters_vals.extend(label_values(["T","α"],df))
        elif df["method"].startswith("logit_"):
            all_parameters_vals.extend(label_values("λ",df))
        elif df["method"].startswith("T_"):
            all_parameters_vals.extend(label_values(["T","α","λ"],df))
        else:
            continue








    # --- Build bins from actual values (with step = 0.0001) ---
    step_clients  = 0.0001
    step_global = 0.0001

    clients_min, clients_max   = min(all_clients_vals), max(all_clients_vals)
    global_min, global_max = min(all_global_vals), max(all_global_vals)

    clients_bounds  = np.arange(clients_min,  clients_max  + step_clients,  step_clients)
    global_bounds = np.arange(global_min, global_max + step_global, step_global)

    print(f"Clients Acc range: {clients_min:.6f} – {clients_max:.6f}, bins={len(clients_bounds)}")
    print(f"Global Acc range: {global_min:.6f} – {global_max:.6f}, bins={len(global_bounds)}")

    # --- Colormaps (fantasy blue) ---
    green_base = plt.get_cmap("Blues")  # clients → greenish
    blue_base = plt.get_cmap("GnBu")  # Global → fantasy blue

    clients_cmap = colors.ListedColormap(green_base(np.linspace(0, 1, len(clients_bounds))))
    clients_norm = colors.BoundaryNorm(clients_bounds, ncolors=len(clients_bounds))

    global_cmap = colors.ListedColormap(blue_base(np.linspace(0, 1, len(global_bounds))))
    global_norm = colors.BoundaryNorm(global_bounds, ncolors=len(global_bounds))

    # --- Plotting ---
    fig, axes = plt.subplots(1, len(subfolders), figsize=(2.5 * len(subfolders), height))
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

        clients_vals  = df["clients"].to_numpy()
        global_vals = df["global"].to_numpy()

        # Heatmap for all subfolders (no rounding applied to values!)
        clients_rgba  = clients_cmap(clients_norm(clients_vals))
        global_rgba = global_cmap(global_norm(global_vals))

        # Stack into (rows, 2, RGBA)
        table_colors = np.zeros((len(clients_vals), 2, 4))
        table_colors[:, 0, :] = clients_rgba
        table_colors[:, 1, :] = global_rgba

        ax.imshow(table_colors, aspect="auto")

        # Write numbers with adaptive font color, show 4 decimals
        for i in range(len(clients_vals)):
            for j, (val, col) in enumerate([(clients_vals[i],  clients_rgba[i]),
                                            (global_vals[i], global_rgba[i])]):
                r, g, b, _ = col
                bright = 0.299*r + 0.587*g + 0.114*b
                txtcol = "white" if bright < 0.5 else "black"
                ax.text(j, i, f"{val:.4f}",
                        ha="center", va="center",
                        fontsize=10,  color=txtcol)

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clients", "Global"], fontsize=9)
        ax.set_yticks(range(len(clients_vals)))
        ax.set_yticklabels(range(len(clients_vals)), fontsize=8)
        ax.set_title(pretty, fontsize=11, pad=12)

        # Draw vertical separator between clients and Global
        ax.axvline(x=0.5, color="black", linewidth=1)

    # plt.suptitle(fig_title, fontsize=14, y=1.05)
    plt.subplots_adjust(wspace=0.15)
    save_path = os.path.join(root_folder, output_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Heatmap saved at: {save_path}")


# Example usage:
# plot_heatmaps_from_subfolders(
#     "distil_aligned_feature_grid_search",
#     "Top Accuracies (clients and Global) for Knowledge Distillation Grid Search"
# )

def plot_all_grid_search_heatmaps(results_dir: str | Path | None = None) -> None:
    """
    Draw the two-column heatmap of every grid search listed below.

    ``results_dir`` defaults to the results root (``config.RESULTS_DIR``); each
    grid search folder is resolved relative to it and every figure is written
    next to the data it summarises.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR

    args=[
          {"path":"prox_grid_search","name":"Proximal"},
          {"path":"logit_consistency_grid_search","name":"Logistic Consistency"},
          {"path":"aligned_feature_grid_search","name":"Aligned Feature"},
          {"path":"distillation_grid_search","name":"Knowledge Distillation"},
          {"path":"distillation_custom_grid_search","name":"Knowledge Distillation Custom"},
          {"path":"ewc_grid_search","name":"Elastic Weight Consolidation"},
    {"path":"distil_ewc_grid_search","name":"Knowledge Distillation + Elastic Weight Consolidation"}]

    for arg in args:
        plot_heatmaps_from_subfolders(root / arg["path"],f"Top Accuracies on (clients and Global Data) for {arg['name']} Grid Search",height=2)


if __name__ == "__main__":
    plot_all_grid_search_heatmaps()