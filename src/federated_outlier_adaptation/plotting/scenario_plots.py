





import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracies_over_rounds(
    accuracies,
    global_clients_all_metrics_acc: float,
    global_clients_metric_acc: float,
    results_path: Path,
    scenario: str,
    metadata: str,
    tag: str | None = None,
    font_size: int = 14,  # 👈 adjustable font size
):
    """
    accuracies: list[tuple(list_like, list_like)] => per-round (agg_all_clients_acc, agg_global_acc)
    """
    # Apply global font size settings
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
    })

    # Unpack accuracies per round
    aggregated_all_clients, aggregated_global = zip(*accuracies)
    plt.figure(figsize=(11, 7),dpi=200)

    def plot_with_min_max(line, label, color):
        line = list(line)
        max_val = max(line)
        min_val = min(line)
        max_idx = line.index(max_val)
        min_idx = line.index(min_val)

        label_with_stats = f"{label} (max={max_val:.4f}, min={min_val:.4f})"
        plt.plot(line, label=label_with_stats, color=color, linewidth=2)
        plt.scatter([max_idx], [max_val], color=color, marker='o', s=60)
        plt.scatter([min_idx], [min_val], color=color, marker='x', s=60)

        plt.annotate(f"max: {max_val:.4f}", (max_idx, max_val),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', color=color, fontsize=font_size - 2)
        plt.annotate(f"min: {min_val:.4f}", (min_idx, min_val),
                     textcoords="offset points", xytext=(0, -18),
                     ha='center', color=color, fontsize=font_size - 2)

    # Plot curves
    plot_with_min_max(aggregated_all_clients, "Accuracy on Clients data", "green")
    plot_with_min_max(aggregated_global, "Accuracy on Global data", "red")

    # Reference lines
    plt.axhline(
        y=global_clients_all_metrics_acc,
        color="blue", linestyle="--", linewidth=1.5,
        label=f"Ground Truth Accuracy (Combined-Model) on Clients Data ({global_clients_all_metrics_acc:.4f})"
    )
    plt.axhline(
        y=global_clients_metric_acc,
        color="purple", linestyle="-.", linewidth=1.5,
        label=f"Ground Truth Accuracy (Combined-Model) on Global Data ({global_clients_metric_acc:.4f})"
    )

    # Titles and labels
    plt.xlabel("Federated Round", labelpad=10)
    plt.ylabel("Accuracy", labelpad=10)
    title = f"Accuracy Over Rounds ({scenario}, {metadata})"
    if tag:
        title += f" • {tag}"
    plt.title(title, pad=15)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(results_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_overlay_accuracies_per_scenario(
    results_by_method: dict[str, list[tuple[float, float]]],
    results_path,
    scenario: str,
    metadata: str,
    ref_clients_acc: float | None = None,   # constant per (scenario, metadata)
    ref_global_acc: float | None = None,    # constant per (scenario, metadata)
):
    """
    results_by_method: { agg_name: [(acc_clients_round0, acc_global_round0), ...], ... }
    Draws one figure overlaying all methods. Optionally draws reference lines once.
    """
    plt.figure(figsize=(11, 7))

    for agg_name, pairs in sorted(results_by_method.items()):
        acc_clients = [a for (a, _) in pairs]
        acc_global  = [b for (_, b) in pairs]
        plt.plot(acc_clients, label=f"{agg_name} (clients)", linestyle="-")
        plt.plot(acc_global,  label=f"{agg_name} (global)",  linestyle="--")

    if ref_clients_acc is not None:
        plt.axhline(ref_clients_acc, linestyle=":", label=f"Ref clients={ref_clients_acc:.4f}")
    if ref_global_acc is not None:
        plt.axhline(ref_global_acc, linestyle=":", label=f"Ref global={ref_global_acc:.4f}")

    plt.xlabel("Federated Round")
    plt.ylabel("Accuracy")
    plt.title(f"All Aggregation Methods • {scenario}, {metadata}")
    plt.legend(ncols=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"overlay_all_methods_{scenario}_{metadata}.png")
    plt.close()

import json
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

def regenerate_scenario_plots(root_dir="."):
    """
    Re-render the per-scenario accuracy plots of an existing results folder.

    ``root_dir`` is the ``<parent>_<trainer>_grid_search`` directory whose
    ``<scenario>_<metadata>`` subfolders hold ``summary_0.json``.
    """
    current_dir = Path(root_dir)

    # get all subfolders (not files)
    subfolders = [f for f in current_dir.iterdir() if f.is_dir()]


    for subfolder in subfolders:
        with open(f"{subfolder}/summary_0.json", "r", encoding="utf-8") as f:
            json_summary = json.load(f)
        scenario,metadata=str(subfolder).split("_")
        for item in json_summary:
            dict_data=json_summary[item]
            base_line_acc_on_global_data=dict_data["global_clients_metric_acc"]
            base_line_acc_on_clients_data=dict_data["global_clients_all_metrics_acc"]
            json_path=extract_from_ctaegory(dict_data["json_path"],f"{scenario}_{metadata}")
            plot_path = extract_from_ctaegory(dict_data["plot_path"],f"{scenario}_{metadata}")
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            agg_method=    json_data["agg_method_name"]
            accuracies=json_data["accuracies"]
            plot_accuracies_over_rounds(accuracies=accuracies,global_clients_all_metrics_acc=base_line_acc_on_clients_data,
                                        global_clients_metric_acc=base_line_acc_on_global_data,
                                        scenario=scenario,
                                        metadata=metadata,
                                        tag=agg_method,results_path=Path(plot_path),font_size=16)
            print()



if __name__ == "__main__":
    regenerate_scenario_plots(".")
