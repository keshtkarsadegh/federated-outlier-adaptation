#!/usr/bin/env python3
"""What the selected server rules do, each readable against the control.

Four panels: preservation on the left, adaptation on the right.  Every panel
carries the two references everything is measured against --- the plain
federated-averaging control as a black dashed curve, drawn on top so it is
never hidden, and the shipped model's own accuracy as the dotted horizontal
line every arm starts from.  The three selected rules are split across the two
rows so that no two of them overlap in the same panel.

    python fig_aggs.py           # regenerates fig_aggs.pdf from ../data

Reads data/traces_aggfull.csv.  Renders only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import figstyle as fs

CONTROL = ("control_fedavg", fs.BLACK, (0, (5, 2)), 1.3, 9)
ROWS = [
    [("anchor_h2", fs.BLUE, "-", 0.5, 6),
     ("eta_0p95", fs.GREEN, "-", 1.1, 5)],
    [("weight_q0", fs.PURPLE, (0, (4, 1, 1, 1)), 1.2, 5)],
]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_aggfull.csv")
    for arm, _, _, _, _ in [CONTROL] + ROWS[0] + ROWS[1]:
        if arm not in series:
            raise SystemExit("traces_aggfull.csv carries no arm %r" % arm)
    start_src = series[CONTROL[0]]["src"][0]
    start_cohort = series[CONTROL[0]]["cohort"][0]

    figure, axes = plt.subplots(2, 2, figsize=(fs.WIDTH, 4.1), sharex=True)
    for arms, (left, right) in zip(ROWS, axes):
        for axis, key, title, ylim, start in (
                (left, "src", fs.SRC_TITLE, fs.SRC_YLIM, start_src),
                (right, "cohort", fs.COHORT_TITLE, fs.COHORT_YLIM,
                 start_cohort)):
            axis.grid(True, color="0.90", linewidth=0.5, zorder=0)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.set_xlim(-2, 102)
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.set_ylim(*ylim)
            axis.set_ylabel("Source accuracy" if key == "src"
                            else "Cohort accuracy")
            axis.set_title(title, pad=4, loc="left")
            axis.axhline(start, color=fs.GREY, linestyle=(0, (1, 2)),
                         linewidth=0.5, zorder=1)
            axis.annotate("shipped model", (100, start),
                          textcoords="offset points", xytext=(-2, 3),
                          ha="right", va="bottom", fontsize=6.4,
                          color=fs.GREY)
            for arm, colour, style, width, z in arms + [CONTROL]:
                fs.draw(axis, rounds, series[arm][key], colour, style,
                        width, z=z)
            fs.mark_start(axis, start, text=None)
    for axis in axes[1]:
        axis.set_xlabel("Round")

    figure.legend(handles=fs.handles(
        [(fs.label(a), c, s) for a, c, s, _, _ in
         [CONTROL] + ROWS[0] + ROWS[1]]),
        loc="lower center", ncol=2, columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.0))
    fs.save(figure, "fig_aggs", rect=(0, 0.105, 1, 1))
    fs.caption("fig_aggs", """
        The three selected server rules on the parallel schedule with no
        client-side penalty, over the full hundred-round budget on the
        ten-client cohort, fold-mean over the five folds on the validation
        halves.  Every panel repeats the federated-averaging control (black,
        dashed) and the shipped model's accuracy (dotted); the rules are
        split across the rows so their curves do not overlap.  A rule helps
        only where it rises above the control: the anchor (blue) is the only
        one that does, on preservation.  Validation basis, so the endpoints
        are not the test-set figures the tables report.
        """)


if __name__ == "__main__":
    main()
