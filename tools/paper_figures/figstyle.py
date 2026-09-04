#!/usr/bin/env python3
"""Shared style and data access for the six accuracy-over-rounds figures.

Every figure in this group answers the same question in the same shape: what a
configuration does to the two populations, round by round, over the fixed
hundred-round budget.  Keeping the shape in one module is what lets a reader
carry a reading from one figure to the next --- the panels sit in the same
order, the axes carry the same limits, and a colour means the same arm
wherever it appears.

    left panel  = source population, "preservation"
    right panel = outlier cohort,    "adaptation"

BASIS.  Every curve is a fold-mean over the five folds of the frozen record,
read from the VALIDATION halves the study's own selection is allowed to see
(``source_val_accuracies`` and ``pool_val_accuracies``).  They are therefore
not the test-set numbers the tables report, and the two must not be quoted
against each other.

ROUND 0 IS THE SHIPPED MODEL.  Round 0 of every CSV carries the shipped
model's own two accuracies, so every arm in a figure starts from one shared
point and the curves show only what adaptation then spends and buys.

SIZE.  The figures are drawn at the manuscript's own text width (372 pt), so
they are placed at natural size and the 8 pt type in them prints as 8 pt.
Nothing here is rasterised: the PDFs are vector throughout.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from paper_names import label, short  # noqa: E402,F401  (re-exported)


def _data_dirs():
    """Where the CSV views are read from, in the order they are tried.

    These scripts have two homes and read the same files in both.  In the
    manuscript checkout they sit in ``<paper>/figures`` beside a ``data``
    directory; in the source repository they sit in ``tools/paper_figures``
    and the views travel in the study's metadata core, where the ones the
    figures need are split across two directories because one bundle is
    written by ``report_tables.py`` and the other by ``export_traces.py``.
    ``FOA_PAPER_DATA`` overrides both with an explicit path (or several,
    separated the way ``PATH`` is), which is what a reader who unpacked the
    records elsewhere will want.
    """
    override = os.environ.get("FOA_PAPER_DATA")
    if override:
        return [os.path.abspath(part) for part in override.split(os.pathsep) if part]
    beside = os.path.join(os.path.dirname(HERE), "data")
    if os.path.isdir(beside):
        return [beside]
    tables = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                          "study", "artifacts", "Digits_study01", "tables")
    return [os.path.join(tables, "paper_figures"), os.path.join(tables, "paper")]


#: Every directory a view may be read from, and the first of them, kept under
#: its old name because the figures address it directly.
DATA_DIRS = _data_dirs()
DATA = DATA_DIRS[0]

#: Where the PDFs and their draft captions are written.  Beside these scripts,
#: which is what the manuscript build wants, unless a reader running them out
#: of a source checkout asks for somewhere that is not tracked.
OUT = os.environ.get("FOA_PAPER_OUT") or HERE


def locate(name):
    """One view, from the first directory that carries it.

    A missing view is named with every place it was looked for: the two homes
    above fail differently, and "no such file" against one of two paths is a
    message that sends a reader to the wrong one.
    """
    for folder in DATA_DIRS:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise SystemExit("%s: not in %s" % (name, os.pathsep.join(DATA_DIRS)))

#: The manuscript's text width, in inches.  sn-jnl gives \linewidth = 372 pt.
LINEWIDTH_PT = 372.0
WIDTH = LINEWIDTH_PT / 72.27

#: Okabe--Ito, which stays separable in the eight per cent of readers with a
#: colour deficiency and in greyscale print.  Line styles carry the same
#: distinction independently, so no figure here relies on hue alone.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#5A5A5A"
LIGHT = "#BBBBBB"

#: Shared limits.  The two panels are read against different scales because
#: the two populations start on different sides of the problem: the source
#: population begins near one and can only be spent, the cohort begins low and
#: is what the budget is being spent on.
SRC_YLIM = (0.965, 1.0005)
COHORT_YLIM = (0.80, 0.96)
#: The extreme arrangements leave both scales, and are given their own.
WIDE_YLIM = (0.70, 1.005)

SRC_LABEL = "Accuracy on the source population"
COHORT_LABEL = "Accuracy on the outlier cohort"
SRC_TITLE = "Preservation"
COHORT_TITLE = "Adaptation"


def setup():
    """The one rcParams block every figure in the group uses."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "legend.handlelength": 2.1,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.4,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "savefig.transparent": False,
        "path.simplify": False,
    })


def read_traces(name):
    """``(rounds, {armid: {'src': [...], 'cohort': [...]}})`` from ``data/``.

    The reader is deliberately strict: the rounds must be 0..N with no gap,
    because a figure that silently plotted a short arm beside a full one would
    draw a difference that is about which run finished rather than about the
    training.
    """
    path = locate(name)
    with open(path, newline="") as handle:
        table = list(csv.DictReader(handle))
    rounds = [int(row["round"]) for row in table]
    if rounds != list(range(len(table))):
        raise SystemExit("%s: rounds must run 0..N without a gap" % name)
    arms = []
    for column in table[0]:
        if column.endswith("_src"):
            arms.append(column[:-4])
    series = {}
    for arm in arms:
        if arm + "_cohort" not in table[0]:
            raise SystemExit("%s: %s has no cohort column" % (name, arm))
        series[arm] = {
            "src": [float(row[arm + "_src"]) for row in table],
            "cohort": [float(row[arm + "_cohort"]) for row in table],
        }
    starts = {(series[a]["src"][0], series[a]["cohort"][0]) for a in arms}
    if len(starts) != 1:
        raise SystemExit("%s: round 0 must be one shared point" % name)
    return rounds, series


def read_rows(name):
    """A small CSV view as a list of dicts, unchanged."""
    with open(locate(name), newline="") as handle:
        return list(csv.DictReader(handle))


def panels(height=2.55, wide=False):
    """The two-panel canvas every figure in the group is drawn on."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, height))
    limits = WIDE_YLIM if wide else None
    for axis, title, ylabel, ylim in (
            (left, SRC_TITLE, SRC_LABEL, limits or SRC_YLIM),
            (right, COHORT_TITLE, COHORT_LABEL, limits or COHORT_YLIM)):
        axis.grid(True, color="0.90", linewidth=0.4, zorder=0)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.set_xlim(-2, 102)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_ylim(*ylim)
        axis.set_xlabel("Round")
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=4, loc="left")
    return figure, left, right


def draw(axis, rounds, values, colour, style="-", width=0.5, z=3, label=None):
    """One curve, vector, with the group's default weight."""
    return axis.plot(rounds, values, color=colour, linestyle=style,
                     linewidth=width, zorder=z, label=label,
                     solid_capstyle="round", rasterized=False)[0]


def mark_start(axis, value, text="shipped model", offset=(9, -2), va="top"):
    """Round 0, named on the axis rather than in the legend."""
    axis.scatter([0], [value], s=13, color=BLACK, zorder=8, linewidths=0)
    if text:
        axis.annotate(text, (0, value), textcoords="offset points",
                      xytext=offset, ha="left", va=va, fontsize=6.6, color=GREY)


def handles(entries):
    """Proxy handles so a legend can be built in the order the figure argues."""
    return [Line2D([], [], color=colour, linestyle=style, linewidth=1.0,
                   label=text) for text, colour, style in entries]


def save(figure, stem, rect=None):
    """Write ``<stem>.pdf`` beside this module.  Vector only, no raster pass."""
    if rect is None:
        figure.tight_layout(pad=0.4, w_pad=1.4)
    else:
        figure.tight_layout(pad=0.4, w_pad=1.4, rect=rect)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, stem + ".pdf")
    # No creation timestamp: two runs over the same CSVs must produce the same
    # bytes, so a regenerated figure that differs means the data differed.
    figure.savefig(path, metadata={"CreationDate": None})
    plt.close(figure)
    print("wrote %s" % os.path.basename(path))
    return path


def caption(stem, text):
    """Write the draft caption beside the PDF, one paragraph, hard-wrapped."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, stem + "_caption.txt")
    with open(path, "w") as handle:
        handle.write(" ".join(text.split()) + "\n")
    print("wrote %s" % os.path.basename(path))
    return path
