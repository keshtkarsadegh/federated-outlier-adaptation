#!/usr/bin/env python3
"""The problem, in one control run.

Plain federated averaging is carried, unprotected, over the whole hundred-round
budget on the study's own ten-client cohort.  The left panel is the source
population the shipped model was trained on; the right is the outlier cohort
the federation is adapting to.  Two horizontal references are drawn on each
panel: the shipped model's own accuracy, which is where every arm starts, and
the centralized pooled ceiling, which is what the same adaptation reaches when
the data are not split across clients at all.  The gap between the control
curve and the ceiling on the right, against the distance the control has
fallen from the shipped model on the left, is the trade this paper is about.

    python fig_problem.py        # regenerates fig_problem.pdf from ../data

Reads data/traces_control.csv and data/references.csv.  Renders only: every
number drawn is read from those files.
"""

from __future__ import annotations

import figstyle as fs


ARM = "control_fedavg"


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_control.csv")
    refs = {row["cell"]: row for row in fs.read_rows("references.csv")}
    ceiling = refs["centralized (from g-0)"]
    ceiling_src = float(ceiling["preservation"])
    ceiling_cohort = float(ceiling["adaptation"])

    src = series[ARM]["src"]
    cohort = series[ARM]["cohort"]
    shipped_src, shipped_cohort = src[0], cohort[0]

    print("control arm: %s" % ARM)
    print("  shipped model   source %.4f  cohort %.4f" % (shipped_src, shipped_cohort))
    print("  round %d        source %.4f  cohort %.4f"
          % (rounds[-1], src[-1], cohort[-1]))
    print("  pooled ceiling  source %.4f  cohort %.4f" % (ceiling_src, ceiling_cohort))

    figure, left, right = fs.panels(height=2.6)

    for axis, values, ref_shipped, ref_ceiling, ship_va, ceil_va in (
            (left, src, shipped_src, ceiling_src, "bottom", "top"),
            (right, cohort, shipped_cohort, ceiling_cohort, "bottom", "bottom")):
        axis.axhline(ref_shipped, color=fs.GREY, linestyle=(0, (1, 2)),
                     linewidth=0.5, zorder=2)
        axis.axhline(ref_ceiling, color=fs.GREEN, linestyle=(0, (5, 2)),
                     linewidth=0.5, zorder=2)
        fs.draw(axis, rounds, values, fs.VERMILION, "-", width=0.5, z=5)
        axis.annotate("shipped model", (100, ref_shipped),
                      textcoords="offset points", xytext=(-2, 2 if ship_va == "bottom" else -2),
                      ha="right", va=ship_va, fontsize=6.6, color=fs.GREY)
        axis.annotate("centralized ceiling", (100, ref_ceiling),
                      textcoords="offset points", xytext=(-2, 2 if ceil_va == "bottom" else -2),
                      ha="right", va=ceil_va, fontsize=6.6, color=fs.GREEN)

    left.annotate(fs.label(ARM), (68, src[68]), textcoords="offset points",
                  xytext=(0, -6), ha="center", va="top", fontsize=7,
                  color=fs.VERMILION)
    right.annotate(fs.label(ARM), (62, cohort[62]), textcoords="offset points",
                   xytext=(0, -5), ha="center", va="top", fontsize=7,
                   color=fs.VERMILION)
    fs.mark_start(left, shipped_src, text=None)
    fs.mark_start(right, shipped_cohort, text=None)

    fs.save(figure, "fig_problem")
    fs.caption("fig_problem", """
        Unprotected federated averaging over the whole budget, on the study's
        ten-client outlier cohort: accuracy on the source population (left,
        preservation) and on the outlier cohort (right, adaptation) against the
        communication round, fold-mean over the five folds on the validation
        halves.  Round 0 is the shipped model's own evaluation, so the curve
        begins at the dotted reference on both panels.  The dashed line is the
        centralized pooled ceiling, the same adaptation run with the cohort's
        data not split across clients.  The reader should see the two halves of
        one trade: the cohort curve climbs steeply for some tens of rounds and
        then flattens well short of the pooled ceiling, while over the same
        rounds the source curve leaves the shipped model and keeps falling to
        the end of the budget, with no sign of settling.  Validation basis, as
        all per-round curves here are, so the endpoints are not the test-set
        figures the tables report.
        """)


if __name__ == "__main__":
    main()
