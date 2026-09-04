#!/usr/bin/env python3
"""The two extreme arrangements, and when they should have stopped.

At one or two clients there is nothing left to average, and the fixed
hundred-round budget stops being a schedule and becomes a hazard: every
arrangement reaches its best trade in the first tens of rounds and then spends
the rest of the horizon taking the source model apart.  Two rounds are marked
on every curve.  The filled marker is the ORACLE stop --- the round that
maximises the study's own selection score along the validation trace, which
needs a source split to read and is therefore a bound, not a method.  The open
marker is the round the study's one permitted rule fires: a fall of more than
five points in the stored proxy-set accuracy, measured from the run's first
round.

    python fig_extremes.py       # regenerates fig_extremes.pdf from ../data

Reads data/traces_extreme.csv and data/extreme_stop_rounds.csv, both of which
carry the rounds already computed; nothing is re-derived here.
"""

from __future__ import annotations

import figstyle as fs
from matplotlib.lines import Line2D


ARMS = [
    ("dual", fs.BLUE, "-"),
    ("double", fs.GREEN, (0, (4, 1.6))),
]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_extreme.csv")
    stops = {row["arm"]: row for row in fs.read_rows("extreme_stop_rounds.csv")}

    figure, left, right = fs.panels(height=2.9, wide=True)
    for arm, colour, style in ARMS:
        if arm not in series or arm not in stops:
            raise SystemExit("no trace or no stopping row for %r" % arm)
        oracle = int(stops[arm]["oracle_round"])
        rule = int(stops[arm]["rule_round"])
        print("  %-7s oracle round %3d, rule (%s > %s) round %3d"
              % (arm, oracle, stops[arm]["rule_signal"], stops[arm]["rule_delta"], rule))
        for axis, key in ((left, "src"), (right, "cohort")):
            values = series[arm][key]
            fs.draw(axis, rounds, values, colour, style, z=4)
            axis.plot([oracle], [values[oracle]], marker="o", markersize=4.2,
                      color=colour, markeredgecolor="white", markeredgewidth=0.5,
                      linestyle="none", zorder=9)
            axis.plot([rule], [values[rule]], marker="s", markersize=4.6,
                      markerfacecolor="white", markeredgecolor=colour,
                      markeredgewidth=0.5, linestyle="none", zorder=9)
    handles = fs.handles([(fs.label(a), c, s) for a, c, s in ARMS]) + [
        Line2D([], [], marker="o", markersize=4.2, color=fs.GREY,
               linestyle="none", label="best round"),
        Line2D([], [], marker="s", markersize=4.6, markerfacecolor="white",
               markeredgecolor=fs.GREY, markeredgewidth=0.5, linestyle="none",
               label="early stop fires"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=4,
                  columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))

    fs.save(figure, "fig_extremes", rect=(0, 0.10, 1, 1))
    fs.caption("fig_extremes", """
        The two extreme arrangements under the fixed hundred-round budget,
        both running the crowned configuration at full participation: the two
        worst-served writers held as two clients, and the same writers' rows
        merged into one client, so the two hold identical data and differ
        only in whether the aggregation ever sees a client boundary.
        Accuracy on the source population (left) and on the two writers' own
        rows (right), fold-mean on the validation halves.  Filled markers
        give each arrangement's best round by the selection score; open
        markers the round the early-stopping rule fires.  Adaptation is
        complete at the filled marker while preservation is still near the
        shipped model; every round after only destroys.
        """)


if __name__ == "__main__":
    main()
