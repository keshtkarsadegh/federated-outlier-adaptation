#!/usr/bin/env python3
"""The distillation-plus-elastic-weight blends, against the two parents.

Each schedule's own distillation winner and its own elastic-weight winner were
mixed at three ratios, and the three mixtures are drawn here against the two
penalties they were built from.  Colour names the role --- the two parents and
the three mixing ratios --- and the line style names the schedule the family
was tuned under, so a mixture can be read against its own parents without
crossing schedules.

    python fig_blends.py         # regenerates fig_blends.pdf from ../data

Reads data/traces_blends.csv.  Renders only.
"""

from __future__ import annotations

import figstyle as fs


PARALLEL = [
    ("kd_T0p25_a0p9", fs.PURPLE),
    ("fisher_lam8", fs.BLUE),
    ("hybrid_mix0p25", fs.ORANGE),
    ("hybrid_mix0p5", fs.GREEN),
    ("hybrid_mix0p75", fs.VERMILION),
]
CYCLIC = [
    ("kd_T2_a0p99", fs.PURPLE),
    ("fisher_lam0p1", fs.BLUE),
    ("hybrid_seq_mix0p25", fs.ORANGE),
    ("hybrid_seq_mix0p5", fs.GREEN),
    ("hybrid_seq_mix0p75", fs.VERMILION),
]
GROUPS = [("Parallel schedule", PARALLEL, "-"),
          ("Cyclic schedule", CYCLIC, (0, (4, 1.6)))]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_blends.csv")

    figure, left, right = fs.panels(height=3.5)
    for _, group, style in GROUPS:
        for arm, colour in group:
            if arm not in series:
                raise SystemExit("traces_blends.csv carries no arm %r" % arm)
            print("  %-20s source %.4f -> %.4f   cohort %.4f -> %.4f"
                  % (arm, series[arm]["src"][0], series[arm]["src"][-1],
                     series[arm]["cohort"][0], series[arm]["cohort"][-1]))
            fs.draw(left, rounds, series[arm]["src"], colour, style, z=4)
            fs.draw(right, rounds, series[arm]["cohort"], colour, style, z=4)
    first = GROUPS[0][1][0][0]
    fs.mark_start(left, series[first]["src"][0], text=None)
    fs.mark_start(right, series[first]["cohort"][0], text=None)

    for index, (title, group, style) in enumerate(GROUPS):
        legend = figure.legend(
            handles=fs.handles([(fs.label(a), c, style) for a, c in group]),
            title=title, loc="lower left", alignment="left",
            bbox_to_anchor=(0.045 + 0.485 * index, 0.005))
        legend.get_title().set_fontsize(7.4)
        figure.add_artist(legend)

    fs.save(figure, "fig_blends", rect=(0, 0.275, 1, 1))
    fs.caption("fig_blends", """
        The distillation-plus-elastic-weight mixtures against the two penalties
        each was built from, over the full hundred-round budget on the study's
        ten-client cohort: accuracy on the source population (left,
        preservation) and on the outlier cohort (right, adaptation), fold-mean
        over the five folds on the validation halves.  Colour names the role,
        line style the schedule the family was tuned under, and round 0 is the
        shipped model, so all ten curves begin at one point.  Within each
        schedule the reader should see that the mixtures do not fall between
        their parents.  On the left every mixture sits at or above the
        elastic-weight parent, the better-preserving of the two, and far above
        the distillation parent; on the right the cyclic mixtures overtake
        their own distillation parent at the higher ratios while the parallel
        ones give a little of that adaptation back.  Holding the better
        parent's preservation without paying the adaptation the other parent
        bought is the whole argument for mixing rather than choosing.  Validation basis, so the
        endpoints are not the test-set figures the tables report.
        """)


if __name__ == "__main__":
    main()
