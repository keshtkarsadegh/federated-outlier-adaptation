#!/usr/bin/env python3
"""What the server rule alone can do.

The three selected aggregation rules of the frozen record, at the full
hundred-round horizon, against the plain federated-averaging control they were
selected over.  No client-side penalty is present in any of these four arms, so
whatever separates the curves is the aggregation rule and nothing else.

    python fig_aggs.py           # regenerates fig_aggs.pdf from ../data

Reads data/traces_aggfull.csv.  Renders only.
"""

from __future__ import annotations

import figstyle as fs


# Drawn in the order they are argued: the control first, so it sits under the
# three rules that were chosen over it.
ARMS = [
    ("control_fedavg", fs.VERMILION, (0, (5, 2)), 6),
    ("anchor_h2", fs.BLUE, "-", 5),
    ("eta_0p95", fs.GREEN, "-", 4),
    ("weight_q0", fs.PURPLE, (0, (4, 1, 1, 1)), 4),
]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_aggfull.csv")

    for arm, _, _, _ in ARMS:
        if arm not in series:
            raise SystemExit("traces_aggfull.csv carries no arm %r" % arm)
        print("  %-18s source %.4f -> %.4f   cohort %.4f -> %.4f"
              % (arm, series[arm]["src"][0], series[arm]["src"][-1],
                 series[arm]["cohort"][0], series[arm]["cohort"][-1]))

    figure, left, right = fs.panels(height=2.6)
    for arm, colour, style, z in ARMS:
        fs.draw(left, rounds, series[arm]["src"], colour, style, z=z)
        fs.draw(right, rounds, series[arm]["cohort"], colour, style, z=z)
    fs.mark_start(left, series[ARMS[0][0]]["src"][0], text=None)
    fs.mark_start(right, series[ARMS[0][0]]["cohort"][0], text=None)

    left.legend(handles=fs.handles([(fs.label(a), c, s) for a, c, s, _ in ARMS]),
                loc="lower left", bbox_to_anchor=(-0.01, -0.02))
    fs.save(figure, "fig_aggs")
    fs.caption("fig_aggs", """
        The three selected server rules of the frozen record against the plain
        federated-averaging control, over the full hundred-round budget, on the
        study's ten-client cohort: accuracy on the source population (left,
        preservation) and on the outlier cohort (right, adaptation), fold-mean
        over the five folds on the validation halves.  All four arms run the
        same parallel schedule with no client-side penalty, and round 0 is the
        shipped model itself, so the four curves begin at one point and every
        later difference is the aggregation rule alone.  The reader should see
        that the four are all but indistinguishable on the right --- adaptation
        is the same to within half a point --- and that only one of them
        separates on the left: the server anchor pulls clear of the control
        from roughly the thirtieth round on and stays there, while the damped
        step and the uniform weighting end where the control ends.  Every arm,
        the anchor included, is still falling when the budget runs out, so the
        server rule slows the loss without stopping it.  Validation basis, so the endpoints are not the test-set
        figures the tables report.
        """)


if __name__ == "__main__":
    main()
