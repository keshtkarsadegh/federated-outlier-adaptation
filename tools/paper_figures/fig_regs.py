#!/usr/bin/env python3
"""What a client-side penalty alone can do.

One arm per regularisation family --- the strength that family's own search
selected --- plus the no-penalty control they are all measured against, over
the full hundred-round budget.  Every arm here runs the cyclic schedule, which
is the schedule the no-penalty control was selected under, so the schedule is
held fixed and the only thing that varies across the seven curves is the
penalty.

    python fig_regs.py           # regenerates fig_regs.pdf from ../data

Reads data/traces_regfull.csv.  Renders only.
"""

from __future__ import annotations

import figstyle as fs


# Grouped the way the section argues them: the unprotected control, then the
# three output-space penalties, then the three that constrain weights or
# features.
ARMS = [
    # Four curves only: the falling control, the two output-space leaders,
    # and one elastic-weight representative for the trailing claim.  The
    # remaining families are in the table; more lines here buried the point.
    ("param_l2_mu0", fs.BLACK, (0, (5, 2)), 0.5, 8),
    ("logit_l2_lam0p003", fs.BLUE, "-", 0.5, 7),
    ("ntd_b0p01_t2", fs.VERMILION, "-", 0.5, 7),
    ("fisher_lam0p1", fs.SKY, (0, (4, 1, 1, 1)), 0.5, 4),
]

def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_regfull.csv")

    for arm, _, _, _, _ in ARMS:
        if arm not in series:
            raise SystemExit("traces_regfull.csv carries no arm %r" % arm)
        print("  %-22s source %.4f -> %.4f   cohort %.4f -> %.4f"
              % (arm, series[arm]["src"][0], series[arm]["src"][-1],
                 series[arm]["cohort"][0], series[arm]["cohort"][-1]))

    figure, left, right = fs.panels(height=3.15)
    for arm, colour, style, width, z in ARMS:
        fs.draw(left, rounds, series[arm]["src"], colour, style, width, z=z)
        fs.draw(right, rounds, series[arm]["cohort"], colour, style, width, z=z)
    fs.mark_start(left, series[ARMS[0][0]]["src"][0], text=None)
    fs.mark_start(right, series[ARMS[0][0]]["cohort"][0], text=None)

    figure.legend(handles=fs.handles([(fs.label(a), c, s) for a, c, s, _, _ in ARMS]),
                  loc="lower center", ncol=2, columnspacing=1.6,
                  bbox_to_anchor=(0.5, 0.0))
    fs.save(figure, "fig_regs", rect=(0, 0.16, 1, 1))
    fs.caption("fig_regs", """
        One arm per regularisation family, at the strength that family's own
        search selected, against the no-penalty control, over the full
        hundred-round budget on the study's ten-client cohort: accuracy on the
        source population (left, preservation) and on the outlier cohort
        (right, adaptation), fold-mean over the five folds on the validation
        halves.  All seven arms run the same cyclic schedule and start from the
        shipped model at round 0, so the curves differ only in the penalty.
        The reader should see the separation the previous figure did not have:
        every penalty holds the source population far above the unprotected
        control, which alone keeps falling steeply to the end of the budget,
        and the better ones do so without buying that preservation with
        adaptation --- FedNTD and logit matching end above the control on both
        panels at once, while the elastic-weight penalties, which preserve
        most, are the two that visibly trail on the right.  Validation basis, so the endpoints are not the
        test-set figures the tables report.  Only the four curves the text argues from are drawn
        --- the falling control, logit matching, FedNTD, and one
        elastic-weight arm --- the remaining families are read from the
        table.
        """)


if __name__ == "__main__":
    main()
