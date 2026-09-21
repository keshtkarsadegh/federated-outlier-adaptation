#!/usr/bin/env python3
"""The crowned combination, beside each of its two components.

The combination the frozen record crowned is a server rule and a client-side
penalty run together.  It is drawn here against the same server rule with no
penalty, the same penalty under the plain server rule, and the unprotected
control both components were selected over, so the reader can see which
component carries which part of the result.

    python fig_combo.py          # regenerates fig_combo.pdf from ../data

Reads data/traces_combo.csv.  Renders only.
"""

from __future__ import annotations

import figstyle as fs


ARMS = [
    # The claim is a pair: the crowned combination (thick black) rides on its
    # penalty component (red dashed) --- those two carry the argument; the server
    # rule alone and the control are context, thin and light.
    ("control_fedavg", fs.LIGHT, (0, (5, 2)), 0.5, 3),
    ("anchor_h2", fs.SKY, (0, (1, 1.2)), 0.5, 4),
    ("ntd_b0p01_t0p5", fs.VERMILION, (0, (5, 2)), 0.5, 7),
    ("anchor_h2_ntd_b0p01_t0p5", fs.BLACK, "-", 0.5, 8),
]


def main():
    fs.setup()
    rounds, series = fs.read_traces("traces_combo.csv")

    for arm, _, _, _, _ in ARMS:
        if arm not in series:
            raise SystemExit("traces_combo.csv carries no arm %r" % arm)
        print("  %-26s source %.4f -> %.4f   cohort %.4f -> %.4f"
              % (arm, series[arm]["src"][0], series[arm]["src"][-1],
                 series[arm]["cohort"][0], series[arm]["cohort"][-1]))

    figure, left, right = fs.panels(height=2.75)
    for arm, colour, style, width, z in ARMS:
        fs.draw(left, rounds, series[arm]["src"], colour, style, width, z=z)
        fs.draw(right, rounds, series[arm]["cohort"], colour, style, width, z=z)

    names = {
        "control_fedavg": fs.label("control_fedavg"),
        "anchor_h2": fs.short("anchor_h2") + ", no penalty",
        "ntd_b0p01_t0p5": fs.short("ntd_b0p01_t0p5") + ", plain server",
        "anchor_h2_ntd_b0p01_t0p5": fs.label("anchor_h2_ntd_b0p01_t0p5")
                                    + " (crowned)",
    }
    figure.legend(handles=fs.handles([(names[a], c, s)
                                      for a, c, s, _, _ in ARMS]),
                  loc="lower center", ncol=2, columnspacing=1.6,
                  bbox_to_anchor=(0.5, 0.0))

    fs.save(figure, "fig_combo", rect=(0, 0.16, 1, 1))
    # RAW.  This caption sets two greek letters, and "\b" and "\t" are a
    # backspace and a tab to Python long before they are \beta and \tau to
    # LaTeX.  A plain string here put a literal 0x08 into the caption file
    # and ate the tau, and neither shows up in a diff.
    fs.caption("fig_combo", r"""
        The crowned combination against each of its two components and against
        the unprotected control, over the full hundred-round budget on the
        study's ten-client cohort: accuracy on the source population (left,
        preservation) and on the outlier cohort (right, adaptation), fold-mean
        over the five folds on the validation rows.  All four arms run the
        parallel schedule and start from the shipped model at round 0; the
        server rule is the anchor at $h=2R$ and the penalty is FedNTD at
        $\beta=0.01$, $\tau=0.5$.  The reader should see that the two
        components do not contribute symmetrically: the penalty alone already
        carries almost all of the preservation the combination holds on the
        left, where the server rule alone recovers only a small part of the
        control's loss, while on the right the combination sits above both
        components --- so the server rule is buying adaptation on top of a
        penalty that is doing the preserving.
        Validation basis, so the endpoints are not the test-set figures the
        tables report.
        """)


if __name__ == "__main__":
    main()
