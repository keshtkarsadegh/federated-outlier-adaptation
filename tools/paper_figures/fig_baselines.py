#!/usr/bin/env python3
"""The shipped model and what isolation does: g-0's own training per fold
(left) and the per-client isolated fine-tuning results (right).

Left panel basis: g-0's stored training history (training and validation
accuracy on the source population, per fold; folds stop at different epochs
because training early-stops on validation).  Right panel basis: test rows,
fold-mean per client --- the isolation records store test evaluations only,
and the caption says so.
"""

import csv
import os

import figstyle as st
import matplotlib.pyplot as plt

FOLD_COLOURS = [st.BLUE, st.ORANGE, st.GREEN, st.VERMILION, st.PURPLE]


def g0_history():
    path = st.locate("g0_training.csv")
    folds = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            folds.setdefault(int(row["fold"]), []).append(
                (int(row["epoch"]), float(row["train_acc"]), float(row["val_acc"])))
    for fold in folds.values():
        fold.sort()
    return folds


def isolation():
    path = st.locate("isolated_clients.csv")
    per = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            per.setdefault(row["client"], []).append(
                (float(row["g0_own_test"]),
                 float(row["iso_g0_own_test"]),
                 float(row["iso_g0_src_test"])))
    out = []
    for rows_ in per.values():
        n = len(rows_)
        out.append(tuple(sum(r[i] for r in rows_) / n for i in range(3)))
    out.sort(key=lambda t: t[0])
    return out


def main():
    st.setup()
    fig = plt.figure(figsize=(st.WIDTH, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.0],
                            width_ratios=[1.0, 0.55])
    ax1 = fig.add_subplot(grid[0, :])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    history = g0_history()
    longest = max(len(s_) for s_ in history.values())
    for fold, series in sorted(history.items()):
        epochs = [e for e, _, _ in series]
        val = [v for _, _, v in series]
        ax1.plot(epochs, val, color=st.LIGHT, lw=0.5, zorder=1)
        best = max(range(len(val)), key=lambda i: val[i])
        ax1.plot(epochs[best], val[best], marker="o", ms=2.6,
                 color=st.GREY, zorder=3)
    mean_val = []
    for e in range(1, longest + 1):
        vals = [s_[e - 1][2] for s_ in history.values() if len(s_) >= e]
        mean_val.append(sum(vals) / len(vals))
    ax1.plot(range(1, longest + 1), mean_val, color=st.BLUE, lw=1, zorder=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Source accuracy")
    ax1.set_title("The shipped model $g$-$0$", loc="left")
    ax1.set_ylim(0.96, 1.002)
    ax1.grid(True, color="#EEEEEE", lw=0.5)
    handles = [
        plt.Line2D([], [], color=st.BLUE, lw=1, label="validation, fold-mean"),
        plt.Line2D([], [], color=st.LIGHT, lw=0.5, label="single folds"),
        plt.Line2D([], [], color=st.GREY, marker="o", ms=2.4, lw=0.5,
                   label="selected epoch"),
    ]
    ax1.legend(handles=handles, loc="lower right")

    rows_ = isolation()
    ys = list(range(1, len(rows_) + 1))
    for y, (before, own, src_) in zip(ys, rows_):
        ax2.plot([before, own], [y, y], color=st.LIGHT, lw=0.86, zorder=1)
    ax2.scatter([r[0] for r in rows_], ys, s=22, facecolors="white",
                edgecolors=st.BLUE, lw=0.79, zorder=2,
                label="shipped model $g$-$0$")
    ax2.scatter([r[1] for r in rows_], ys, s=22, color=st.BLUE,
                zorder=3, label="after fine-tuning on this client")
    ax2.set_xlabel("Accuracy on the client's own rows")
    ax2.set_ylabel("Client (sorted)")
    ax2.set_title("What each client gains", loc="left")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([str(y) for y in ys])
    ax2.set_xlim(0.72, 1.02)
    ax2.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax2.grid(True, axis="x", color="#EEEEEE", lw=0.5)
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles2, labels2, loc="outside lower center", ncol=2,
               fontsize=6.3, handletextpad=0.4, columnspacing=1.0)

    import csv as _csv
    with open(st.locate("references.csv"), newline="") as handle:
        ref = next(_csv.DictReader(handle))
    p0 = float(ref["preservation"]) + float(ref["spent"])
    for y, (before, own, src_) in zip(ys, rows_):
        ax3.plot([src_, p0], [y, y], color=st.LIGHT, lw=0.86, zorder=1)
    ax3.scatter([r[2] for r in rows_], ys, s=26, color=st.VERMILION,
                marker="x", lw=0.86, zorder=3,
                label="after fine-tuning on this client")
    ax3.axvline(p0, color=st.GREY, lw=0.58, ls=":")
    ax3.text(p0 - 0.0012, 0.62, "$g$-$0$", ha="right", va="bottom",
             fontsize=6.5, color=st.GREY)
    ax3.set_xlim(0.955, 1.003)
    ax3.set_xticks([0.96, 0.98, 1.0])
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks(ys)
    ax3.set_yticklabels([])
    ax3.set_xlabel("Source-population accuracy")
    ax3.set_title("What the source loses", loc="left")
    ax3.grid(True, axis="x", color="#EEEEEE", lw=0.5)

    os.makedirs(st.OUT, exist_ok=True)
    fig.savefig(os.path.join(st.OUT, "fig_baselines.pdf"))


if __name__ == "__main__":
    main()
