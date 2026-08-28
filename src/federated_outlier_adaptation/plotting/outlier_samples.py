import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image

from federated_outlier_adaptation import config
from federated_outlier_adaptation.providers import default_provider

"""
visualize_global_with_outlier_samples.py

Purpose:
    Generate side-by-side visualizations comparing the mean digit images
    of global writers with example samples from selected outlier writers
    in the NIST dataset.

Capabilities:
    - Computes per-digit global mean images across all global writers.
    - Selects sample digits from chosen outlier writers.
    - Renders a table-like grid where:
        • Row 0 = global mean images (per digit).
        • Rows 1+ = outlier writers, one row per writer.
    - Highlights digits consistently with borders and aligned labels.

Outputs:
    - Figure displayed in a tight grid layout.
    - Optional save to disk as PNG with high resolution.

Usage:
    $ python visualize_global_with_outlier_samples.py
    (shows interactive plot; provide save_path to also save PNG)
"""

def visualize_global_with_outlier_samples(
    nist_dataset,
    global_writers: list[str],
    outlier_writers: list[str],
    save_path: str | Path = None,
    n_examples: int = 5,
):
    """
        Visualize global mean digit images against example samples from outlier writers.

        Args:
            nist_dataset (NistDataset): Dataset handler for loading writers' samples.
            global_writers (list[str]): List of writer IDs considered "global."
            outlier_writers (list[str]): List of selected outlier writer IDs.
            save_path (str | Path, optional): If provided, save the figure to this path.
            n_examples (int): Number of outlier writers to visualize (default=5).

        Process:
            1. Load global and outlier datasets.
            2. Compute mean digit images across all global writers.
            3. For each selected outlier writer, fetch sample digits.
            4. Arrange results into a bordered, tight table:
                - Top row: global means per digit.
                - Subsequent rows: one per outlier writer.

        Output:
            - Interactive Matplotlib figure.
            - If save_path is specified, saves as a high-resolution PNG.

        Notes:
            • Each subplot is tightly bordered for a clean, table-like appearance.
            • Missing digits for a writer are shown as blank white cells.
    """
    import matplotlib.patches as patches

    print("Loading samples for visualization...")

    # --- Load data ---
    global_loader, _, _ = nist_dataset.build_dataset(global_writers, batch_size=64, train_rate=1.0, eval_rate=0.0)
    outlier_loader, _, _ = nist_dataset.build_dataset(outlier_writers, batch_size=64, train_rate=1.0, eval_rate=0.0)

    def collect_images(data_loader):
        imgs, labels = [], []
        for batch_imgs, batch_labels in tqdm(data_loader, desc="Collecting"):
            imgs.extend(batch_imgs.squeeze(1).numpy())
            labels.extend(batch_labels.numpy())
        return np.array(imgs), np.array(labels)

    global_imgs, global_labels = collect_images(global_loader)
    outlier_imgs, outlier_labels = collect_images(outlier_loader)

    digits = sorted(set(global_labels.tolist()) & set(outlier_labels.tolist()))
    n_digits = len(digits)

    # --- compute global mean per digit ---
    global_means = {d: global_imgs[global_labels == d].mean(axis=0) for d in digits}

    selected_writers = outlier_writers[:n_examples]

    # gather their samples
    label_dict_path = nist_dataset.nist_data_path.digits_labels_json
    with open(label_dict_path, "r") as f:
        label_dict = json.load(f)

    from collections import defaultdict
    from PIL import Image
    writer_to_imgs = defaultdict(list)
    for rel_path, lbl in label_dict.items():
        writer_id = rel_path.split("/")[2]
        if writer_id in selected_writers:
            writer_to_imgs[writer_id].append((nist_dataset.nist_data_path.nist_root_path / rel_path, lbl))

    total_rows = 1 + len(selected_writers)

    # --- Create figure ---
    fig, axes = plt.subplots(total_rows, n_digits, figsize=(2.2 * n_digits, 2.1 * total_rows))
    plt.subplots_adjust(left=0.32, right=0.98, top=0.96, bottom=0.04, wspace=0.0, hspace=0.0)

    # --- Helper to draw borders ---
    def draw_border(ax):
        rect = patches.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            color="black",
            linewidth=0.7,
            fill=False,
        )
        ax.add_patch(rect)

    # --- Top row: global means ---
    for j, d in enumerate(digits):
        ax = axes[0, j]
        ax.imshow(global_means[d], cmap="gray", aspect="auto", extent=(0, 1, 0, 1))
        ax.set_title(f"{d}", fontsize=26, pad=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        draw_border(ax)
        if j == 0:
            ax.set_ylabel(
                "Global Mean",
                fontsize=24,
                rotation=0,
                labelpad=100,
                va="center",
                fontweight="bold",
            )

    # --- Outlier rows ---
    for i, wid in enumerate(selected_writers, start=1):
        samples = writer_to_imgs.get(wid, [])
        digit_to_img = {lbl: str(path) for path, lbl in samples}

        for j, d in enumerate(digits):
            ax = axes[i, j]
            if d in digit_to_img:
                img = np.array(Image.open(digit_to_img[d]).convert("L").resize((128, 128)))
                ax.imshow(img, cmap="gray", aspect="auto", extent=(0, 1, 0, 1))
            else:
                ax.imshow(np.ones((128, 128)), cmap="gray", vmin=0, vmax=1, aspect="auto", extent=(0, 1, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            draw_border(ax)

            if j == 0:
                ax.set_ylabel(
                    wid,
                    fontsize=22,
                    rotation=0,
                    labelpad=85,
                    va="center",
                    fontweight="bold",
                )

    # --- Force tight placement manually ---
    fig.canvas.draw()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for ax in fig.axes:
        pos = ax.get_position()
        pos.y0 -= 0.001  # small shift to remove subpixel gaps
        pos.y1 += 0.001
        ax.set_position(pos)

    fig.patch.set_facecolor("white")
    fig.patch.set_linewidth(1.2)
    fig.patch.set_edgecolor("black")

    if save_path:
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved figure at: {save_path}")

    plt.show()



# ==========================================================
# MAIN EXECUTION
# ==========================================================
def render_default_figure(save_path=None, provider=None, n_examples: int = 5):
    """Draw the figure for the frozen writer split and outlier selection."""
    provider = provider or default_provider()
    global_writers = provider.global_client_ids()
    outlier_writers = provider.selected_clients()
    print(f"Number of global writers: {len(global_writers)}")
    print(f"Number of outlier writers: {len(outlier_writers)}")

    return visualize_global_with_outlier_samples(
        provider.dataset,
        global_writers=global_writers,
        outlier_writers=outlier_writers,
        save_path=save_path or (config.OUTLIERS_DIR / "visualizing_outliers.png"),
        n_examples=n_examples,
    )


if __name__ == "__main__":
    render_default_figure()
