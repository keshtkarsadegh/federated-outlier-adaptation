"""Proximal-regularizer overlay figures with inside legends, run on results/final_result_CFProxTrainer_grid_search."""

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

from federated_outlier_adaptation import config

DEFAULT_RESULTS_SUBFOLDER = "final_result_CFProxTrainer_grid_search"


def plot_overlay_accuracies_per_scenario(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path: Path,
    scenario: str,
    metadata: str,
    ref_clients_acc: float | None = None,
    ref_global_acc: float | None = None,
    font_size: int = 14,
    highlight_map: dict[str, bool] | None = None,  # True=full color, False=faded
):
    """
    Plot two subplots (clients/global accuracies) per aggregation method,
    showing percentage values (0–100%) with 2 decimal places.
    """
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 3,
        'legend.fontsize': font_size - 1,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    # helper to format y-axis as percentage
    percent_fmt = FuncFormatter(lambda y, _: f"{y:.2f}%")

    # -----------------------
    # Plot client accuracies
    # -----------------------
    all_clients_vals = []
    for idx, (agg_name, pairs) in enumerate(sorted(results_by_method.items())):
        acc_clients = [a * 100 for (a, _) in pairs]  # convert to %
        all_clients_vals.extend(acc_clients)
        color = plt.cm.tab10(idx % 10)
        highlight = highlight_map.get(agg_name, True) if highlight_map else True
        alpha = 1.0 if highlight else 0.2
        ax1.plot(acc_clients, linestyle="-", label=f"{agg_name}", color=color, alpha=alpha)

    if ref_clients_acc is not None:
        ref_clients_acc *= 100
        ax1.axhline(
            ref_clients_acc, linestyle=":", color="black", linewidth=2.5,
            label=f"Ref: clients={ref_clients_acc:.2f}%"
        )

    # min/max
    if all_clients_vals:
        min_client = min(all_clients_vals) - 1.0
        max_client = max(all_clients_vals)
        range_client = max_client - min_client
        margin_client = 0.02 * range_client
        ax1.set_ylim(min_client - margin_client, max_client + margin_client)
    else:
        min_client = max_client = None

    ax1.yaxis.set_major_formatter(percent_fmt)
    ax1.set_ylabel("Accuracy on Clients Data (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.9,
               borderpad=0.8, fancybox=True, edgecolor="gray")

    # -----------------------
    # Plot global accuracies
    # -----------------------
    all_global_vals = []
    for idx, (agg_name, pairs) in enumerate(sorted(results_by_method.items())):
        acc_global = [b * 100 for (_, b) in pairs]  # convert to %
        all_global_vals.extend(acc_global)
        color = plt.cm.tab10(idx % 10)
        highlight = highlight_map.get(agg_name, True) if highlight_map else True
        alpha = 1.0 if highlight else 0.2
        ax2.plot(acc_global, linestyle="--", label=f"{agg_name}", color=color, alpha=alpha)

    if ref_global_acc is not None:
        ref_global_acc *= 100
        ax2.axhline(
            ref_global_acc, linestyle=":", color="black", linewidth=2.5,
            label=f"Ref: global={ref_global_acc:.2f}%"
        )

    # min/max
    if all_global_vals:
        min_global = min(all_global_vals) - 2.0
        max_global = max(all_global_vals)
        range_global = max_global - min_global
        margin_global = 0.02 * range_global
        ax2.set_ylim(min_global - margin_global, max_global + margin_global)
    else:
        min_global = max_global = None

    ax2.yaxis.set_major_formatter(percent_fmt)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy on Global Data (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", ncol=2, frameon=True, facecolor="white", framealpha=0.9,
               borderpad=0.8, fancybox=True, edgecolor="gray")

    # --- Layout & save ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"All Aggregation Methods ({scenario.capitalize()}, {metadata.capitalize()})",
                 fontsize=font_size + 2, y=0.995)

    output_path = results_path / f"overlay_two_plots_insideleg_boldref_{scenario}_{metadata}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "min_client": min_client, "max_client": max_client,
        "min_global": min_global, "max_global": max_global,
    }




def extract_from_ctaegory(path: str, keyword: str) -> str:
    idx = path.find(keyword)
    if idx == -1:
        raise ValueError(f"Keyword '{keyword}' not found in path: {path}")
    return path[idx:]

def extract_after_keyword(path: str, keyword: str) -> str:

    idx = path.find(keyword)
    if idx == -1:
        raise ValueError(f"Keyword '{keyword}' not found in path: {path}")
    return path[idx + len(keyword):].lstrip("/\\")

# ----------------------------
# Sequential Aggregation Methods (explicit dictionaries)
# ----------------------------

# 1️⃣ Sequential weights-based aggregation methods
SEQ_WEIGHTS_METHODS_TRUE = {
    "seq_fixed_ratio_update": True,
    "seq_equal_update": False,
    "seq_fedavg_update": False,
    "seq_incremental_update": True,
}

# 2️⃣ Sequential delta-based aggregation methods
SEQ_DELTA_METHODS_TRUE = {
    "seq_delta_fedavg_update": False,
    "seq_delta_scaled": False,
    "seq_delta_capped": True,
    "seq_delta_progressive_update": True,
}

# 3️⃣ Concurrent weights-based aggregation methods
CONCURRENT_WEIGHTS_METHODS_TRUE = {
    "con_weighted_cw": False,
    "con_weighted_cgw": False,
    "con_scaled_cw": False,
    "con_scaled_cgw": False,
    "con_capped_cw": False,
    "con_capped_cgw": True,
}

# 4️⃣ Concurrent delta-based aggregation methods
CONCURRENT_DELTA_METHODS_TRUE = {
    "con_delta_weighted_cgd": False,
    "con_delta_scaled_cgd": False,
    "con_delta_capped_cgd": True,
}


def plot_regularizer_scenario_overlays(results_dir: str | Path | None = None) -> None:
    """
    Draw one overlay figure per scenario subfolder of the proximal-regularizer grid search.

    ``results_dir`` defaults to ``config.RESULTS_DIR / "final_result_CFProxTrainer_grid_search"``,
    the folder these figures were produced from.  Each figure is written into the
    scenario subfolder it summarises.
    """
    root = Path(results_dir) if results_dir else config.RESULTS_DIR / DEFAULT_RESULTS_SUBFOLDER

    # get all subfolders (not files)
    subfolders = [f for f in root.iterdir() if f.is_dir()]

    for subfolder in subfolders:
        with open(f"{subfolder}/summary_0.json", "r", encoding="utf-8") as f:
            json_summary = json.load(f)
        scenario,metadata=subfolder.name.split("_")
        base_line_acc_on_global_data = ""
        base_line_acc_on_clients_data = ""
        if scenario=="sequential" and metadata=="weights":
            highlight_map=SEQ_WEIGHTS_METHODS_TRUE
        elif scenario=="sequential" and metadata=="delta":
            highlight_map=SEQ_DELTA_METHODS_TRUE
        elif scenario=="concurrent" and metadata=="weights":
            highlight_map=CONCURRENT_WEIGHTS_METHODS_TRUE
        elif scenario=="concurrent" and metadata=="delta":
            highlight_map=CONCURRENT_DELTA_METHODS_TRUE
        else:
            raise ValueError(f"Scenario '{scenario}' not supported")


        results_by_method={}
        for item in json_summary:
            dict_data=json_summary[item]
            base_line_acc_on_global_data=dict_data["global_clients_metric_acc"]
            base_line_acc_on_clients_data=dict_data["global_clients_all_metrics_acc"]
            json_path=extract_from_ctaegory(dict_data["json_path"],f"{scenario}_{metadata}")
            with open(root / json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            agg_method=    json_data["agg_method_name"]
            accuracies=json_data["accuracies"]
            results_by_method[agg_method]=accuracies

        plot_overlay_accuracies_per_scenario(results_by_method=results_by_method,results_path=Path(subfolder),scenario=scenario,metadata=metadata,
                                                     ref_global_acc=base_line_acc_on_global_data,
                                                     ref_clients_acc=base_line_acc_on_clients_data
                                                    ,font_size=16,highlight_map=highlight_map )


if __name__ == "__main__":
    plot_regularizer_scenario_overlays()

