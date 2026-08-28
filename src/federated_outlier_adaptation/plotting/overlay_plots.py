"""Overlay plots comparing aggregation methods within one scenario."""

import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path




def plot_overlay_accuracies_per_scenario(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path: Path,
    scenario: str,
    metadata: str,
    ref_clients_acc: float | None = None,
    ref_global_acc: float | None = None,
    font_size: int = 14,
    highlight_map: dict[str, bool] | None = None,
):
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 3,
        'legend.fontsize': font_size - 1,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    percent_fmt = FuncFormatter(lambda y, _: f"{y:.2f}%")

    # Client accuracies
    all_clients_vals = []
    for idx, (agg_name, pairs) in enumerate(sorted(results_by_method.items())):
        acc_clients = [a * 100 for (a, _) in pairs]
        all_clients_vals.extend(acc_clients)
        color = plt.cm.tab10(idx % 10)
        highlight = highlight_map.get(agg_name, True) if highlight_map else True
        alpha = 1.0 if highlight else 0.2
        ax1.plot(acc_clients, "-", label=agg_name, color=color, alpha=alpha)

    if ref_clients_acc is not None:
        ref_clients_acc *= 100
        ax1.axhline(ref_clients_acc, ":", color="black", linewidth=2.5,
                    label=f"Ref: clients={ref_clients_acc:.2f}%")

    if all_clients_vals:
        low, high = min(all_clients_vals), max(all_clients_vals)
        margin = 0.02 * (high - low)
        ax1.set_ylim(low - margin, high + margin)

    ax1.yaxis.set_major_formatter(percent_fmt)
    ax1.set_ylabel("Accuracy on Clients Data (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", ncol=2, frameon=True, facecolor="white",
               framealpha=0.9, borderpad=0.8, fancybox=True, edgecolor="gray")

    # Global accuracies
    all_global_vals = []
    for idx, (agg_name, pairs) in enumerate(sorted(results_by_method.items())):
        acc_global = [b * 100 for (_, b) in pairs]
        all_global_vals.extend(acc_global)
        color = plt.cm.tab10(idx % 10)
        highlight = highlight_map.get(agg_name, True) if highlight_map else True
        alpha = 1.0 if highlight else 0.2
        ax2.plot(acc_global, "--", label=agg_name, color=color, alpha=alpha)

    if ref_global_acc is not None:
        ref_global_acc *= 100
        ax2.axhline(ref_global_acc, ":", color="black", linewidth=2.5,
                    label=f"Ref: global={ref_global_acc:.2f}%")

    if all_global_vals:
        low, high = min(all_global_vals), max(all_global_vals)
        margin = 0.02 * (high - low)
        ax2.set_ylim(low - margin, high + margin)

    ax2.yaxis.set_major_formatter(percent_fmt)
    ax2.set_xlabel("Federated Round")
    ax2.set_ylabel("Accuracy on Global Data (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", ncol=2, frameon=True, facecolor="white",
               framealpha=0.9, borderpad=0.8, fancybox=True, edgecolor="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(f"All Aggregation Methods ({scenario.capitalize()}, {metadata.capitalize()})",
                 fontsize=font_size + 2, y=0.995)

    out = results_path / f"overlay_two_plots_insideleg_boldref_{scenario}_{metadata}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def plot_overlay_all_scenarios(root_dir):
    root_dir = Path(root_dir).resolve()
    print(f"🧭 Using root directory: {root_dir}")

    SEQ_WEIGHTS_METHODS_TRUE = {
        "seq_fixed_ratio_update": True,
        "seq_equal_update": False,
        "seq_fedavg_update": False,
        "seq_incremental_update": True,
    }
    SEQ_DELTA_METHODS_TRUE = {
        "seq_delta_fedavg_update": False,
        "seq_delta_scaled": False,
        "seq_delta_capped": True,
        "seq_delta_progressive_update": True,
    }
    CONCURRENT_WEIGHTS_METHODS_TRUE = {
        "con_weighted_cw": False,
        "con_weighted_cgw": False,
        "con_scaled_cw": False,
        "con_scaled_cgw": False,
        "con_capped_cw": False,
        "con_capped_cgw": True,
    }
    CONCURRENT_DELTA_METHODS_TRUE = {
        "con_delta_weighted_cgd": False,
        "con_delta_scaled_cgd": False,
        "con_delta_capped_cgd": True,
    }

    for subfolder in root_dir.iterdir():
        if not subfolder.is_dir():
            continue

        summary = subfolder / "summary_0.json"
        if not summary.exists():
            print(f"⚠️  Skipping {subfolder.name}, no summary_0.json.")
            continue

        print(f"📂 Processing {subfolder.name}")
        with open(summary, "r", encoding="utf-8") as f:
            json_summary = json.load(f)

        try:
            scenario, metadata = subfolder.name.split("_", 1)
        except ValueError:
            print(f"⚠️  Invalid folder naming: {subfolder.name}")
            continue

        if scenario == "sequential" and metadata == "weights":
            highlight_map = SEQ_WEIGHTS_METHODS_TRUE
        elif scenario == "sequential" and metadata == "delta":
            highlight_map = SEQ_DELTA_METHODS_TRUE
        elif scenario == "concurrent" and metadata == "weights":
            highlight_map = CONCURRENT_WEIGHTS_METHODS_TRUE
        elif scenario == "concurrent" and metadata == "delta":
            highlight_map = CONCURRENT_DELTA_METHODS_TRUE
        else:
            print(f"⚠️  Unsupported scenario: {scenario}_{metadata}")
            continue

        results_by_method = {}
        for item, info in json_summary.items():
            rel_path = Path(info["json_path"])
            json_path = rel_path if rel_path.is_absolute() else (root_dir / rel_path)
            json_path = json_path.resolve()

            if not json_path.exists():
                print(f"❌ Missing: {json_path}")
                continue

            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            results_by_method[data["agg_method_name"]] = data["accuracies"]

        if results_by_method:
            plot_overlay_accuracies_per_scenario(
                results_by_method,
                results_path=subfolder,
                scenario=scenario,
                metadata=metadata,
                ref_global_acc=info.get("global_clients_metric_acc"),
                ref_clients_acc=info.get("global_clients_all_metrics_acc"),
                font_size=16,
                highlight_map=highlight_map,
            )
            print(f"✅ Done: {subfolder.name}")
        else:
            print(f"⚠️  No valid results for {subfolder.name}")


# Historical alias: the notebooks import the sweep helper under both names.
plot_overlay_all_scenarios_new = plot_overlay_all_scenarios
