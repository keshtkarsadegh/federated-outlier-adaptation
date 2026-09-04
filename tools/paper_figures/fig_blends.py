#!/usr/bin/env python3
"""The distillation-plus-elastic-weight mixtures against their two parents.

One row of panels per schedule, so the parents and their own mixtures are
never drawn over the other schedule's: within each row the two parents carry
the interval and the three mixtures should be seen sitting outside it, at or
above the elastic-weight parent on the left and far above the distillation
parent, which is the figure's claim.

    python fig_blends.py           # regenerates fig_blends.pdf from ../data

Reads data/traces_blends.csv.  Renders only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import figstyle as fs

ROWS = [
    ("Parallel schedule", [
        ("kd_T0p25_a0p9", fs.VERMILION, "-", 1, 6),
        ("fisher_lam8", fs.BLUE, "-", 1, 6),
        ("hybrid_mix0p5", fs.GREEN, "-", 1, 7),
    ]),
    ("Cyclic schedule", [
        ("kd_T2_a0p99", fs.VERMILION, "-", 1, 6),
        ("fisher_lam0p1", fs.BLUE, "-", 1, 6),
        ("hybrid_seq_mix0p5", fs.GREEN, "-", 1, 7),
    ]),
]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_blends.csv")

    figure, axes = plt.subplots(2, 2, figsize=(fs.WIDTH, 4.3), sharex=True)
    for (row_title, arms), (left, right) in zip(ROWS, axes):
        for axis, key, title, ylabel, ylim in (
                (left, "src", fs.SRC_TITLE, fs.SRC_LABEL, fs.SRC_YLIM),
                (right, "cohort", fs.COHORT_TITLE, fs.COHORT_LABEL,
                 fs.COHORT_YLIM)):
            axis.grid(True, color="0.90", linewidth=0.6, zorder=0)
            axis.set_axisbelow(True)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.set_xlim(-2, 102)
            axis.set_xticks([0, 25, 50, 75, 100])
            axis.set_ylim(*ylim)
            axis.set_ylabel("Source accuracy" if key == "src" else "Cohort accuracy")
            axis.set_title("%s --- %s" % (title, row_title), pad=4,
                           loc="left")
        for arm, colour, style, width, z in arms:
            if arm not in series:
                raise SystemExit("traces_blends.csv carries no arm %r" % arm)
            fs.draw(left, rounds, series[arm]["src"], colour, style, width,
                    z=z)
            fs.draw(right, rounds, series[arm]["cohort"], colour, style,
                    width, z=z)
        fs.mark_start(left, series[arms[0][0]]["src"][0], text=None)
        fs.mark_start(right, series[arms[0][0]]["cohort"][0], text=None)
    for axis in axes[1]:
        axis.set_xlabel("Round")

    figure.legend(handles=fs.handles([
        ("distillation parent", fs.VERMILION, "-"),
        ("elastic-weight parent", fs.BLUE, "-"),
        ("the blend, $m{=}0.5$ (selected)", fs.GREEN, "-"),
    ]), loc="lower center", ncol=3, columnspacing=1.6,
        bbox_to_anchor=(0.5, 0.0))
    fs.save(figure, "fig_blends", rect=(0, 0.075, 1, 1))
    fs.caption("fig_blends", """
        The distillation-plus-elastic-weight mixtures against the two parents
        each was built from, one row of panels per schedule so the two
        constructions are never drawn over each other: accuracy on the source
        population (left, preservation) and on the outlier cohort (right,
        adaptation), fold-mean over the five folds on the validation halves,
        from the shipped model at round 0.  In each row the parents are the red and blue
        curves --- distillation, which adapts and does not preserve, and the
        elastic-weight term, the reverse --- and the green curve is the
        selected blend ($m{=}0.5$; the other two ratios behave alike and are
        in the released records).  The reader should see the green curve
        sitting outside the interval its parents span: at or above the blue
        parent on the left and far above it on the right, on the cyclic row
        overtaking even the distillation parent.  Validation basis, so the endpoints are not the
        test-set figures the tables report.
        """)


if __name__ == "__main__":
    main()
