#!/usr/bin/env python3
"""The three smallest federations, and when they should have stopped.

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
    ("single", fs.VERMILION, "-"),
    ("dual", fs.BLUE, (0, (4, 1.6))),
    ("double", fs.GREEN, (0, (1, 1.4))),
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
                      color=colour, markeredgecolor="white", markeredgewidth=0.6,
                      linestyle="none", zorder=9)
            axis.plot([rule], [values[rule]], marker="s", markersize=4.6,
                      markerfacecolor="white", markeredgecolor=colour,
                      markeredgewidth=0.72, linestyle="none", zorder=9)
    fs.mark_start(left, series[ARMS[0][0]]["src"][0], text=None)
    fs.mark_start(right, series[ARMS[0][0]]["cohort"][0], text=None)

    left.legend(handles=fs.handles([(fs.label(a), c, s) for a, c, s in ARMS]),
                loc="lower left", bbox_to_anchor=(-0.01, -0.02))
    markers = [
        Line2D([], [], marker="o", markersize=4.2, color=fs.GREY,
               linestyle="none", label="oracle stop"),
        Line2D([], [], marker="s", markersize=4.6, markerfacecolor="white",
               markeredgecolor=fs.GREY, markeredgewidth=0.72, linestyle="none",
               label="rule fires"),
    ]
    right.legend(handles=markers, loc="lower right", bbox_to_anchor=(1.01, -0.02))

    fs.save(figure, "fig_extremes")
    fs.caption("fig_extremes", """
        The three smallest arrangements --- one client holding the cohort's
        worst writer, two clients holding its worst two, and those same two
        writers' rows merged into a single client --- over the full
        hundred-round budget: accuracy on the source population (left,
        preservation) and on the outlier cohort (right, adaptation), fold-mean
        over the five folds on the validation halves, with round 0 the shipped
        model itself.  Note the wider vertical scale: these curves leave the
        range the earlier figures are drawn on.  A filled marker gives each
        arm's oracle stop, the round maximising the study's selection score
        along the trace, and an open marker the round the one permitted
        stopping rule fires, a fall of more than five points in the stored
        proxy-set accuracy.  The reader should see that adaptation is
        essentially finished by the filled marker in every arm, so everything
        the source curve loses to the right of it was spent for nothing, and
        that the rule always fires well after the oracle --- late enough that
        most of the damage is already done, which is the honest limit of what
        the stored signals support.  Validation basis, so the endpoints are not
        the test-set figures the tables report.
        """)


if __name__ == "__main__":
    main()
