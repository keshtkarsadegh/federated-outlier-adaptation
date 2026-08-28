"""Three-column (params/clients/global) grid-search heatmaps per scenario, run on the results root."""

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
    print(values)
    # return triples (labeled_string, clients, global)
    return [
        (
            ",".join(f"{lab}:{",".join(f"{lab}:{val}" for lab, val in zip(labels, row))}" for lab, val in zip(labels, row)),
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
        if df["method"].str.startswith("aliened_").any():
            all_parameters_vals.extend(label_values("β", df))
        elif df["method"].str.startswith("prox_").any():
            all_parameters_vals.extend(label_values("λ", df))
        elif df["method"].str.startswith("ewc_").any():
            all_parameters_vals.extend(label_values("λ", df))
        elif df["method"].str.startswith("distill_T").any():
            all_parameters_vals.extend(label_values(["T", "α"], df))
        elif df["method"].str.startswith("logit_").any():
            all_parameters_vals.extend(label_values("λ", df))
        elif df["method"].str.startswith("T_").any():
            all_parameters_vals.extend(label_values(["T", "α", "λ"], df))

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

        # --- Get parameter labels ---
        if df["method"].str.startswith("aliened_").any():
            triples = label_values("β", df)
        elif df["method"].str.startswith("prox_").any():
            triples = label_values("λ", df)
        elif df["method"].str.startswith("ewc_").any():
            triples = label_values("λ", df)
        elif df["method"].str.startswith("distil_T").any():
            triples = label_values(["T", "α"], df)
        elif df["method"].str.startswith("logit_").any():
            triples = label_values("λ", df)
        elif df["method"].str.startswith("T_").any():
            triples = label_values(["T", "α", "λ"], df)
        else:
            triples = [("", c, g) for c, g in zip(df["clients"], df["global"])]

        labels, clients_vals, global_vals = zip(*triples)

        # --- Heatmap colors for Clients & Global only ---
        clients_rgba = clients_cmap(clients_norm(clients_vals))
        global_rgba = global_cmap(global_norm(global_vals))

        # Make table with 3 columns: Params (gray), Clients (colored), Global (colored)
        table_colors = np.ones((len(clients_vals), 3, 4))  # all white
        table_colors[:, 0, :] = [0.9, 0.9, 0.9, 1.0]  # Params column = light gray
        table_colors[:, 1, :] = clients_rgba
        table_colors[:, 2, :] = global_rgba

        ax.imshow(table_colors, aspect="auto")

        # --- Write text into cells ---
        for i, (lab, cval, gval) in enumerate(zip(labels, clients_vals, global_vals)):
            # Params column (black text)
            ax.text(0, i, lab, ha="center", va="center", fontsize=8, color="black")

            # Clients column
            r, g, b, _ = clients_rgba[i]
            txtcol = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(1, i, f"{cval:.4f}", ha="center", va="center", fontsize=9, color=txtcol)

            # Global column
            r, g, b, _ = global_rgba[i]
            txtcol = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
            ax.text(2, i, f"{gval:.4f}", ha="center", va="center", fontsize=9, color=txtcol)

        # --- Formatting ---
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Params", "Clients", "Global"], fontsize=9)
        ax.set_yticks(range(len(clients_vals)))
        ax.set_yticklabels(range(len(clients_vals)), fontsize=8)
        ax.set_title(pretty, fontsize=11, pad=12)

        # Vertical separators
        ax.axvline(x=0.5, color="black", linewidth=1)
        ax.axvline(x=1.5, color="black", linewidth=1)


# Example usage:
# plot_heatmaps_from_subfolders(
#     "distil_aligned_feature_grid_search",
#     "Top Accuracies (clients and Global) for Knowledge Distillation Grid Search"
# )

def plot_all_labelled_grid_search_heatmaps(results_dir: str | Path | None = None) -> None:
    """
    Draw the labelled three-column heatmap of every grid search listed below.

    ``results_dir`` defaults to the results root (``config.RESULTS_DIR``); each
    grid search folder is resolved relative to it.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR

    args=[
          {"path":"prox_grid_search","name":"Proximal"},
          {"path":"logit_consistency_grid_search","name":"Logistic Consistency"},
          {"path":"aligned_feature_grid_search","name":"Aligned Feature"},
          {"path":"distillation_grid_search","name":"Knowledge Distillation"},
          {"path":"ewc_grid_search","name":"Elastic Weight Consolidation"},
    {"path":"distil_ewc_grid_search","name":"Knowledge Distillation + Elastic Weight Consolidation"}]

    for arg in args:
        plot_heatmaps_from_subfolders(root / arg["path"],f"Top Accuracies on (clients and Global Data) for {arg['name']} Grid Search",height=2)


if __name__ == "__main__":
    plot_all_labelled_grid_search_heatmaps()