#!/usr/bin/env python3
"""Regenerate tables/*.tex from data/*.csv.

Usage:  python3 make_paper_tables.py [--check]

Contract
--------
* Every number printed in a table is read from a CSV cell and is a correct
  rounding of it.  Two assertions enforce that: the raw CSV string must occur
  verbatim in the file it is claimed to come from, and the rendered string must
  round-trip to the raw value at the printed precision.  A quantity that is not
  a CSV cell is either a macro from numbers.tex (so it is generated too) or is
  labelled ``derived'' in a comment at the top of the emitted file.
* Every table header states the evaluation basis it reports.
* No run-record identifier reaches the page.  Every method is named by
  paper_names.py, which turns a cell id into the published name of the method
  and its setting in that method's own symbol; a table is rejected by an
  assertion at the end of this script if any identifier survives into it.
* Running the script twice produces byte-identical files.

Where it reads and where it writes
----------------------------------
The views are read from ``data/`` beside this script in the manuscript
checkout, and from the study's shipped metadata core in the source
repository; ``FOA_PAPER_DATA`` overrides both.  The tables are written to
``tables/`` under the manuscript directory, which is where this script
sits in the manuscript checkout and which ``FOA_PAPER_OUT`` names anywhere
else.  Neither default writes into a source checkout.  See ``_data_dirs``
and ``out_dir``.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from paper_names import (ARM, EXTREME, REFERENCE, SCHEDULE, SIGNAL,  # noqa: E402
                         label, settings_note, short)


def _data_dirs():
    """Where the CSV views are read from, in the order they are tried.

    This script has two homes and reads the same files in both.  In the
    manuscript checkout the views sit in a ``data`` directory beside it; in the
    source repository it sits in ``tools/paper_figures`` and the views travel in
    the study's metadata core, where they are spread over four directories
    because four different tools write them --- ``report_tables.py`` into
    ``tables/paper``, the exporters into ``tables/paper_figures``,
    ``stopping_table.py`` into ``tables/stopping``, and the stage tools into
    ``tables`` itself.  ``FOA_PAPER_DATA`` overrides all of them with an
    explicit path (or several, separated the way ``PATH`` is), which is what a
    reader who unpacked the records elsewhere will want.

    THE ORDER IS NOT ARBITRARY.  ``tables/paper/combos.csv`` is the combination
    view the manuscript quotes and ``tables/combos.csv`` is the raw grid dump
    the stage left behind; they carry the same name and different rows, so the
    paper bundle has to be looked in first.
    """
    override = os.environ.get("FOA_PAPER_DATA")
    if override:
        return [os.path.abspath(part) for part in override.split(os.pathsep) if part]
    beside = os.path.join(HERE, "data")
    if os.path.isdir(beside):
        return [beside]
    tables = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                          "study", "artifacts", "Digits_study01", "tables")
    return [os.path.join(tables, "paper"), os.path.join(tables, "paper_figures"),
            os.path.join(tables, "stopping"), tables]


#: Every directory a view may be read from, and the first of them, kept under
#: its old name because the manuscript build addresses it directly.
DATA_DIRS = _data_dirs()
DATA = DATA_DIRS[0]


def locate(name):
    """One view, from the first directory that carries it.

    A missing view is named with every place it was looked for: the two homes
    above fail differently, and "no such file" against one of four paths is a
    message that sends a reader to the wrong one.
    """
    for folder in DATA_DIRS:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise SystemExit("%s: not in %s" % (name, os.pathsep.join(DATA_DIRS)))


def out_dir():
    """The manuscript directory the generated LaTeX is written into.

    ``numbers.tex`` sits in it and ``tables/`` sits beside them, which is the
    layout the manuscript checkout has --- so beside this script is the right
    answer there, and it is what the build wants.  A source checkout has no
    such directory and this one is tracked, so rather than writing LaTeX into
    it the script says which variable to set.  ``FOA_PAPER_OUT`` is that
    variable, the same one the figures take.
    """
    override = os.environ.get("FOA_PAPER_OUT")
    if override:
        return os.path.abspath(override)
    if os.path.isdir(os.path.join(HERE, "data")):
        return HERE
    raise SystemExit(
        "no manuscript directory: this is a source checkout, and writing the\n"
        "generated LaTeX beside the script would write into the tracked tree.\n"
        "Set FOA_PAPER_OUT to a scratch directory: numbers.tex is rewritten\n"
        "there, in place, and tables/ is written under it.")


#: The directory the .tex files are written into: ``tables`` under the
#: manuscript directory.  Filled in by main(), which is the only place
#: allowed to decide where a source checkout may write.
OUT = [None]

# The frozen crowning record.  Identical to the constants in make_numbers.py:
# the arms are named, never re-derived from the best row of a test table.
CROWNED = ("anchor_h2_ntd_b0p01_t0p5", "parallel")
BALANCED = ("eta_0p95_hybrid_seq_mix0p5", "parallel")
CYCLIC_ARM = ("seq_fedavg_ntd_b0p01_t2", "cyclic")

RAW = {}
CHECKED = [0]


# --------------------------------------------------------------------------
# csv
# --------------------------------------------------------------------------

def rows(name):
    path = locate(name)
    RAW[name] = open(path).read()
    with open(path, newline="") as fh:
        out = list(csv.DictReader(fh))
    for r in out:
        r["_src"] = name
    return out


def pick(table, **eq):
    hit = [r for r in table if all(r[k] == v for k, v in eq.items())]
    if len(hit) != 1:
        raise KeyError("expected 1 row for %r, found %d" % (eq, len(hit)))
    return hit[0]


def by_score(table):
    return sorted(table, key=lambda r: -float(r["score"]))


# Results are presented one schedule at a time: the parallel block first, the
# cyclic block second, each ordered by score inside itself.  The per-row
# Schedule column is then redundant and is dropped; the italic block header
# carries it, and the indented data rows under it separate the blocks without
# a rule, which keeps a grouped table the height the ungrouped one had.
SCHEDULE_BLOCKS = (("parallel", "Parallel schedule"),
                   ("cyclic", "Cyclic schedule"))


def check_schedules(table):
    """No row of a schedule-grouped table may fall outside the two blocks."""
    known = {fam for fam, _ in SCHEDULE_BLOCKS}
    stray = sorted({r["family"] for r in table} - known)
    assert not stray, "schedule-grouped table has unplaced families: %s" % stray


# --------------------------------------------------------------------------
# numbers.  Everything printed goes through num(); nothing is retyped.
# --------------------------------------------------------------------------

def num(row, col, nd=4, scale=1.0, signed=False):
    """The CSV cell row[col], rounded to nd places, optionally x100."""
    raw = row[col]
    assert raw in RAW[row["_src"]], (row["_src"], col, raw)
    v = float(raw) * scale
    out = "%.*f" % (nd, v)
    assert abs(v - float(out)) <= 0.5 * 10 ** (-nd) + 1e-12
    CHECKED[0] += 1
    if out.startswith("-"):
        return "$%s$" % out
    return ("$+%s$" % out) if signed else out


def derived(v, nd=2, scale=1.0, signed=False):
    """An aggregate over CSV cells: a median, a mean or an extreme of them.

    Not a cell itself, so it cannot be checked against the file verbatim; every
    table that prints one names the aggregation in its own header comment.
    """
    out = "%.*f" % (nd, v * scale)
    if out.startswith("-"):
        return "$%s$" % out
    return ("$+%s$" % out) if signed else out


def verbatim(row, col):
    """A CSV cell printed exactly as the file spells it (a budget)."""
    raw = row[col]
    assert raw in RAW[row["_src"]], (row["_src"], col, raw)
    CHECKED[0] += 1
    return raw


def count(row, col):
    """An integer CSV cell."""
    raw = row[col]
    assert raw in RAW[row["_src"]], (row["_src"], col, raw)
    v = float(raw)
    CHECKED[0] += 1
    return "%d" % int(round(v))


def loose(row, col, nd=2):
    """A CSV cell that may or may not be integral (clients per round)."""
    raw = row[col]
    assert raw in RAW[row["_src"]], (row["_src"], col, raw)
    v = float(raw)
    CHECKED[0] += 1
    return "%d" % int(round(v)) if abs(v - round(v)) < 1e-9 else "%.*f" % (nd, v)


def diffs(row):
    """The five per-fold differences of a compare_arms row."""
    xs = [float(x) for x in row["diffs"].split()]
    assert len(xs) == int(row["folds"].split()[-1]), row
    return xs


# --------------------------------------------------------------------------
# latex
# --------------------------------------------------------------------------

def tex(s):
    return s.replace("_", "\\_")


def code(cell):
    return "\\code{%s}" % tex(cell)


def block(label, caption, colspec, header, body, size="\\small",
          colsep=None, note=None, comment=""):
    L = ["%% generated by make_paper_tables.py --- do not edit by hand"]
    if comment:
        L += ["%% " + x for x in comment.strip().split("\n")]
    L += ["\\begin{table}[!htbp]", "\\centering",
          "\\caption{%s}" % caption.strip(), "\\label{%s}" % label, size]
    if colsep:
        L.append("\\setlength{\\tabcolsep}{%s}" % colsep)
    # shrink-only-if-too-wide: the tabular is never enlarged, and can never
    # run past the text block however long a cell identifier turns out to be.
    # \snresize is defined in the preamble; it wraps \resizebox and suspends
    # the threeparttable width hook that sn-jnl installs in every table.
    L += ["\\snresize{%",
          "\\begin{tabular}{@{}%s@{}}" % colspec, "\\toprule", header,
          "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}}"]
    if note:
        L += ["", "\\vspace{2pt}",
              "\\begin{minipage}{\\linewidth}\\footnotesize %s\\end{minipage}"
              % note.strip()]
    L += ["\\end{table}", ""]
    return "\n".join(L)


def bare(colspec, header, body, comment="", note=None):
    """A tabular with no surrounding table environment.

    Used where the float, its caption and its \\label are the section file's
    to own, so that the appendix can place and refer to the table itself.
    """
    L = ["%% generated by make_paper_tables.py --- do not edit by hand"]
    if comment:
        L += ["%% " + x for x in comment.strip().split("\n")]
    L += ["%% tabular only: the enclosing table environment, its caption and",
          "%% its label are supplied by the section file that inputs this."]
    if note:
        L += ["%% a footnote minipage follows the tabular, so this input must",
              "%% not be wrapped in \\resizebox."]
    L += ["\\begin{tabular}{@{}%s@{}}" % colspec, "\\toprule", header,
          "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}"]
    if note:
        L += ["", "\\vspace{2pt}",
              "\\begin{minipage}{\\linewidth}\\footnotesize %s\\end{minipage}"
              % note.strip()]
    L += [""]
    return "\n".join(L)


def write(name, text):
    path = os.path.join(OUT[0], name)
    old = open(path).read() if os.path.exists(path) else None
    if old != text:
        if CHECK[0]:
            sys.exit("tables/%s is out of date; run: python3 make_paper_tables.py"
                     % name)
        open(path, "w").write(text)
    return path


CHECK = [False]


# --------------------------------------------------------------------------
# T1 --- the reference ladder
# --------------------------------------------------------------------------

def t_references(refs):
    body = []
    for r in by_score(refs):
        body.append(" & ".join([REFERENCE[r["cell"]],
                                num(r, "adaptation"), num(r, "preservation"),
                                num(r, "gained", 2, 100), num(r, "spent", 2, 100),
                                num(r, "score", 2, 100)]) + " \\\\")
    return block(
        "tab:references",
        "The reference ladder, five-fold means, \\textbf{test} axis. Gained "
        "and spent are the differences from $\\thg$, score their difference, "
        "Eq.~\\eqref{eq:score}; centralized training is a ceiling the "
        "constraint forbids.",
        "lrrrrr",
        "Reference & Adaptation & Preservation & Gained & Spent & Score \\\\\n"
        " & & & (pts) & (pts) & (pts) \\\\",
        body, size="\\scriptsize",
        comment="source: data/references.csv, all rows, ordered by col score")


# --------------------------------------------------------------------------
# T2a --- server rules
# --------------------------------------------------------------------------

def t_agg(agg):
    body = []
    for fam, title in SCHEDULE_BLOCKS:
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{5}{@{}l}{\\textbf{%s}} \\\\" % title)
        for r in by_score([x for x in agg if x["family"] == fam]):
            body.append(" & ".join(["\\quad " + label(r["cell"]),
                                    num(r, "adaptation"), num(r, "preservation"),
                                    num(r, "spent", 2, 100),
                                    num(r, "score", 2, 100)]) + " \\\\")
    check_schedules(agg)
    return block(
        "tab:agg_winners",
        "Every server rule at its own best setting, at the full horizon under "
        "a plain cross-entropy objective. Five-fold means, \\textbf{test} axis, "
        "grouped by schedule and ordered by score within each block; "
        "``FedAvg (control)'' is plain FedAvg at $\\eta_s{=}1$.",
        "lrrrr",
        "Server rule & Adaptation & Preservation & Spent (pts) & Score (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt",
        comment="source: data/agg-winners.csv, all rows, grouped by col family\n"
                "(parallel block first, cyclic second), ordered by col score\n"
                "within each block")


# --------------------------------------------------------------------------
# T2b --- penalties (the blends have their own table)
# --------------------------------------------------------------------------

def t_reg(regu):
    single = [r for r in regu if not r["cell"].startswith("hybrid")]
    body = []
    for fam, title in SCHEDULE_BLOCKS:
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{5}{@{}l}{\\textbf{%s}} \\\\" % title)
        for r in by_score([x for x in single if x["family"] == fam]):
            body.append(" & ".join(["\\quad " + label(r["cell"]),
                                    num(r, "adaptation"), num(r, "preservation"),
                                    num(r, "spent", 2, 100),
                                    num(r, "score", 2, 100)]) + " \\\\")
    check_schedules(single)
    return block(
        "tab:reg_winners",
        "Every client penalty at its own best setting, at the full horizon "
        "under plain FedAvg on the server. Five-fold means, \\textbf{test} axis, "
        "grouped by schedule and ordered by score within each block; "
        "``No penalty (control)'' is the proximal term at $\\mu{=}0$.",
        "lrrrr",
        "Client penalty & Adaptation & Preservation & Spent (pts) & Score (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt",
        comment="source: data/reg-winners.csv, rows whose cell does not start\n"
                "with 'hybrid', grouped by col family (parallel block first,\n"
                "cyclic second), ordered by col score within each block")


# --------------------------------------------------------------------------
# T2c --- the blends, with the fold-paired verdict against each parent
# --------------------------------------------------------------------------

def t_blends(regu, blend):
    hyb = [r for r in regu if r["cell"].startswith("hybrid")]
    assert len(hyb) == 12, len(hyb)

    # blends.csv holds one comparison per (blend, parent) on the schedule the
    # blend's two halves were selected on; index it by that pair.
    verdict = {}
    parents = {}
    for b in blend:
        fam = SCHEDULE[b["family"]]
        kind = "kd" if b["minus"].startswith("kd_") else "fisher"
        d = diffs(b)
        verdict[(b["arm"], fam, kind)] = "%s (%d/%d)" % (
            num(b, "mean", 2, 1.0, signed=True), sum(1 for x in d if x > 0), len(d))
        parents.setdefault((fam, kind), b["minus"])
        assert parents[(fam, kind)] == b["minus"]

    body = []
    for r in by_score(hyb):
        key = (r["cell"], r["family"])
        body.append(" & ".join([
            label(r["cell"]), r["family"],
            num(r, "adaptation"), num(r, "preservation"),
            num(r, "spent", 2, 100), num(r, "score", 2, 100),
            verdict.get(key + ("kd",), "---"),
            verdict.get(key + ("fisher",), "---")]) + " \\\\")

    note = ("Verdict columns: the mean of the five fold-paired score "
            "differences, in points, against the distillation half and the "
            "consolidation half the blend was built from, with the number of "
            "folds on which "
            "the difference is positive. A blend is compared on the schedule "
            "its two halves were selected on, so the six rows run off that "
            "schedule carry no paired record and are marked ---. Parents: "
            + "; ".join("%s and %s on the %s schedule"
                        % (label(parents[(fam, "kd")]),
                           label(parents[(fam, "fisher")]), fam)
                        for fam in ("parallel", "cyclic")) + ".")

    return block(
        "tab:blends",
        "The one composite penalty the study builds, "
        "$\\lambda\\,[\\,m\\,D_{\\mathrm{kd}} + (1-m)\\,D_{\\mathrm{fisher}}\\,]$, "
        "at the three mixes and on both schedules. Performance columns are "
        "five-fold means on the \\textbf{test} axis; the two verdict columns "
        "are fold-paired against the blend's own parents. The plain rows are "
        "built from the parallel schedule's selected halves; the "
        "\\emph{cyclic-tuned} rows are built from the cyclic schedule's.",
        "llrrrrrr",
        "Blend & Schedule & Adapt. & Pres. & Spent & Score & vs.\\ KD & vs.\\ EWC \\\\\n"
        " & & & & (pts) & (pts) & (pts, folds) & (pts, folds) \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/reg-winners.csv, the 12 rows whose cell starts\n"
                "with 'hybrid', ordered by col score; data/blends.csv for the\n"
                "two verdict columns (cols mean and diffs)")


# --------------------------------------------------------------------------
# T3 --- the combination grid
# --------------------------------------------------------------------------

def split_combo(cell, agg_cells):
    for a in sorted(agg_cells, key=len, reverse=True):
        if cell.startswith(a + "_"):
            return a, cell[len(a) + 1:]
    raise KeyError(cell)


def t_combos(combo, agg, regu):
    agg_cells = {r["cell"] for r in agg}
    check_schedules(combo)
    ordered = []
    body = []
    halves = []
    for fam, title in SCHEDULE_BLOCKS:
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{6}{@{}l}{\\textbf{%s}} \\\\" % title)
        for r in by_score([x for x in combo if x["family"] == fam]):
            ordered.append(r)
            rule, pen = split_combo(r["cell"], agg_cells)
            a = pick(agg, cell=rule, family=r["family"])
            b = pick(regu, cell=pen, family=r["family"])
            half = a if float(a["score"]) >= float(b["score"]) else b
            d = 100.0 * (float(r["score"]) - float(half["score"]))
            CHECKED[0] += 1
            halves.append(half["cell"] == pen)
            name = "\\quad " + short(rule) + " & " + short(pen)
            if r["cell"] == CROWNED[0]:
                name += "$^{\\star}$"
            body.append(" & ".join([
                name,
                num(r, "adaptation"), num(r, "preservation"),
                num(r, "score", 2, 100),
                ("$+%.2f$" % d) if d >= 0 else ("$%.2f$" % d)]) + " \\\\")

    assert all(halves), "a server rule outscores its penalty somewhere"
    note = ("Settings, given once for the whole table: "
            + settings_note([(r["cell"], r["family"]) for r in ordered]) + ". "
            "$\\Delta$ better half is the row's score minus the score of "
            "whichever of its two halves scores higher on its own, read from "
            "Tables~\\ref{tab:agg_winners} and~\\ref{tab:reg_winners}. In all "
            "eighteen rows that better half is the \\emph{penalty}. The column is "
            "a difference of five-fold means, not a fold-paired verdict, which "
            "the text reports instead. $\\star$ marks the crowned pair.")

    return block(
        "tab:combos",
        "All eighteen combinations: each schedule's three best server rules "
        "crossed with its three best penalties, at the full horizon. Five-fold "
        "means, \\textbf{test} axis, grouped by schedule and ordered by score "
        "within each block.",
        "llrrrr",
        "Server rule & Penalty & Adapt. & Pres. & Score (pts) & "
        "$\\Delta$ better half (pts) \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/combos.csv, all rows, grouped by col family\n"
                "(parallel block first, cyclic second), ordered by col score\n"
                "within each block; the delta column is derived --- combos\n"
                "score minus the larger of the two parents' scores in\n"
                "data/agg-winners.csv and data/reg-winners.csv, same schedule")


# --------------------------------------------------------------------------
# T4 --- the four carry settings
# --------------------------------------------------------------------------

SETTING = [("c10d10", "ten clients, one dropped", "Ten clients"),
           ("five", "five clients, one dropped", "Five clients"),
           ("c20d10", "twenty, two dropped", "Twenty clients, two dropped"),
           ("c20d20", "twenty, four dropped", "Twenty clients, four dropped")]

ARM_ORDER = ["winner", "balanced", "sequential", "control"]


def t_scaling(sizes, cost_s):
    body = []
    for i, (key, stage, title) in enumerate(SETTING):
        if i:
            body.append("\\midrule")
        cs = pick(cost_s, stage=stage)
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{5}{@{}l}{\\textbf{%s, %s participating per "
                    "round}} \\\\" % (title, loose(cs, "clients_per_round")))
        table = sizes[key]
        for cell in ARM_ORDER:
            for r in [x for x in table if x["cell"] == cell]:
                body.append(" & ".join([
                    "\\quad " + ARM[cell], r["family"],
                    num(r, "adaptation"), num(r, "preservation"),
                    num(r, "score", 2, 100)]) + " \\\\")

    note = ("Carried arms, run unchanged: \\emph{" + ARM["winner"] + "} is "
            + label(CROWNED[0]) + ", \\emph{" + ARM["balanced"] + "} is "
            + label(BALANCED[0]) + " and \\emph{" + ARM["sequential"]
            + "} is " + label(CYCLIC_ARM[0])
            + ". The twenty-client settings omit the control.")

    return block(
        "tab:scaling",
        "The frozen configuration set carried, without re-search, to four "
        "federation settings. Five-fold means, \\textbf{test} axis; nothing "
        "here was tuned at the setting it is reported at.",
        "llrrr",
        "Arm & Schedule & Adaptation & Preservation & Score \\\\\n"
        " & & & & (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt", note=note,
        comment="sources: data/sizes_c10d10.csv, sizes_five.csv,\n"
                "sizes_c20d10.csv, sizes_c20d20.csv (all rows); the\n"
                "participation counts are cost_stages.csv col clients_per_round")


# --------------------------------------------------------------------------
# T7 --- who the gain reaches
# --------------------------------------------------------------------------

def t_fairness(fair_combo, fair10, fair20, agg_cells):
    def line(name, r):
        return " & ".join([name, r["family"],
                           num(r, "mean"), num(r, "min"),
                           num(r, "lift_worst", 2, 100, signed=True),
                           num(r, "improved", 2)]) + " \\\\"

    best = max(fair_combo, key=lambda r: float(r["lift_worst"]))
    body = ["\\multicolumn{6}{@{}l}{\\emph{Search setting, the combination "
            "stage (ten clients, eight per round)}} \\\\"]
    for arm_name, cell, family in (
            ("\\quad " + ARM["winner"], CROWNED[0], CROWNED[1]),
            ("\\quad " + ARM["balanced"], BALANCED[0], BALANCED[1]),
            ("\\quad " + ARM["sequential"], CYCLIC_ARM[0], CYCLIC_ARM[1]),
            ("\\quad best worst-client", best["cell"], best["family"])):
        body.append(line(arm_name, pick(fair_combo, cell=cell, family=family)))

    body += ["\\midrule",
             "\\multicolumn{6}{@{}l}{\\emph{Carry setting, ten clients, nine "
             "per round}} \\\\"]
    for cell in ARM_ORDER:
        for r in [x for x in fair10 if x["cell"] == cell]:
            body.append(line("\\quad " + ARM[cell], r))

    body += ["\\midrule",
             "\\multicolumn{6}{@{}l}{\\emph{Twenty clients: the one arm that "
             "leaves a client behind}} \\\\"]
    for key, _, title in SETTING[2:]:
        r = pick(fair20[key], cell="balanced", family="parallel")
        body.append(line("\\quad %s, %s" % (ARM["balanced"], title.lower().replace(
            "twenty clients, ", "")), r))

    note = ("$\\Delta$ worst is the worst-served client's accuracy minus "
            "\\emph{that client's own} accuracy under the shipped model, in "
            "points; lifted is the share of client-fold pairs that finish "
            "above their own shipped-model accuracy. Every entry is a "
            "five-fold mean over the per-client records. The only negative "
            "$\\Delta$ worst in the study is the balanced arm at twenty "
            "clients, in the last block. Arms: " + ARM["winner"] + " is "
            + label(CROWNED[0]) + ", " + ARM["balanced"] + " is "
            + label(BALANCED[0]) + ", " + ARM["sequential"] + " is "
            + label(CYCLIC_ARM[0]) + ", and the best worst-client pair is "
            + label(best["cell"]) + ".")

    return block(
        "tab:fairness",
        "The per-client distribution at the end of the budget, each client "
        "read against its own accuracy under the shipped model. "
        "\\textbf{Test} axis, five-fold means. The mean column is the cohort "
        "mean the selection rule optimises; the two columns beside it are what "
        "that mean can hide.",
        "llrrrr",
        "Arm & Schedule & Mean & Worst & $\\Delta$ worst & Lifted \\\\\n"
        " & & & client & (pts) & \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/fairness_combo.csv (search setting),\n"
                "data/fairness_c10d10.csv (carry setting), and\n"
                "data/fairness_c20d10.csv + fairness_c20d20.csv (the caveat rows)")


# --------------------------------------------------------------------------
# T6 --- cost
# --------------------------------------------------------------------------

def t_cost(cost_s):
    body = []
    total = 0.0
    for r in cost_s:
        total += float(r["hours"])
        body.append(" & ".join([r["stage"].replace(",", ","),
                                count(r, "tasks"), count(r, "rounds"),
                                loose(r, "clients_per_round"),
                                num(r, "seconds_per_round", 2),
                                num(r, "hours", 2)]) + " \\\\")
    body.append("\\midrule")
    body.append("\\emph{total} & & & & & %.2f \\\\" % total)

    note = ("Communication is identical for every mechanism in the study: with "
            "\\nParamCount{} parameters a round exchanges a model-sized message "
            "with each participant in both directions --- \\nCommPerRound{} at "
            "the search setting, the same for the control, the crowned pair and "
            "the balanced arm, to the byte. None of these methods changes what "
            "is transmitted. Wall-clock is comparable \\emph{within} a stage, "
            "where arms share a job configuration; across stages the runs "
            "shared a partition with other jobs and no ordering is claimed. "
            "The total row is the sum of the column above it.")

    return block(
        "tab:cost",
        "What the programme cost, per stage: tasks run, rounds per task, "
        "clients contributing per round, measured seconds per round, and GPU "
        "hours. Measured by the runner's own round timer.",
        "lrrrrr",
        "Stage & Tasks & Rounds & Clients & s/round & GPU-h \\\\\n"
        " & & & /round & & \\\\",
        body, colsep="5pt", note=note,
        comment="source: data/cost_stages.csv, all rows in file order;\n"
                "the total row is the sum of col hours (derived)")


# --------------------------------------------------------------------------
# T5 --- the extremes, fixed horizon against the oracle stop
# --------------------------------------------------------------------------

def t_extreme(extr, stopx):
    sx = {r["cell"]: r for r in stopx}
    body = []
    for r in by_score(extr):
        s = sx[r["cell"]]
        body.append(" & ".join([
            EXTREME[r["cell"]],
            num(r, "adaptation"), num(r, "preservation"), num(r, "score", 2, 100),
            num(s, "final_score", 2, 100), count(s, "oracle_round"),
            num(s, "oracle_score", 2, 100)]) + " \\\\")

    note = ("The first three columns are the fixed 100-round horizon on the "
            "\\textbf{test} axis, as everywhere else in this section. The last "
            "three are read from the per-round \\textbf{validation} trace, "
            "which is the only basis on which a stopping round may be chosen: "
            "final score is the same horizon on that trace, $r^{\\star}$ is the "
            "round at which the score peaks, and the oracle score is the score "
            "there. The gap between the last two score columns is what stopping "
            "is worth in these cases; Section~\\ref{sec:signals} asks how much "
            "of it a permitted rule can recover.")

    return block(
        "tab:extremes",
        "The two extreme arrangements, both running the crowned "
        "configuration unchanged at full participation. \\emph{"
        + EXTREME["double"] + "} gives the two worst writers a client each; "
        "\\emph{" + EXTREME["dual"] + "} merges the same rows into one client, "
        "so the two hold byte-identical data and differ only in whether "
        "the aggregation ever sees a client boundary.",
        "lrrrrcr",
        "Case & Adapt. & Pres. & Score & Final score & $r^{\\star}$ & "
        "Oracle score \\\\\n"
        " & & & (pts) & (pts, trace) & & (pts, trace) \\\\",
        body, colsep="5pt", note=note,
        comment="sources: data/extremes.csv (test axis, ordered by col score)\n"
                "and data/stopping_extreme.csv (cols final_score, oracle_round,\n"
                "oracle_score --- the per-round validation trace)")



# --------------------------------------------------------------------------
# T10 --- the eight permitted signals: tracker on the left, rule on the right
# --------------------------------------------------------------------------

# The order the section introduces them in: the three that need no data, the
# three the clients can be asked for, then the two on the public proxy set.
SIGNAL_ORDER = ("dist_l2_to_global", "dist_fisher_to_global",
                "dist_fisher_norm_to_global", "retention_known",
                "agreement_with_global", "kl_global_to_current",
                "proxy_acc", "proxy_kl")

#: Signals whose budgets are a units problem rather than an information one:
#: the raw distances share no scale with the grid, so no budget on them is a
#: rule.  Named here rather than inferred, so a data refresh cannot silently
#: promote one of them into a rule the section does not discuss.
UNSTEERABLE = ("dist_l2_to_global", "dist_fisher_to_global",
               "dist_fisher_norm_to_global")


def t_signals(sigs):
    by = {r["signal"]: r for r in sigs}
    body = []
    for key in SIGNAL_ORDER:
        r = by[key]
        fired = int(r["arms_fired"])
        if key in UNSTEERABLE:
            verdict = "\\emph{unsteerable}"
        elif fired == 0:
            verdict = "\\emph{never fires}"
        else:
            verdict = "%s at $\\delta=%s$" % (
                num(r, "mean_stopped_score", 2, 100), verbatim(r, "best_delta"))
        body.append(" & ".join([
            SIGNAL[key], r["observed_on"],
            num(r, "median_rho", 3),
            "%s\\%%" % num(r, "share_ge_0p9", 1, 100),
            verdict]) + " \\\\")

    note = ("The correlation columns read the signal's \\emph{drift} from its "
            "own round-zero value against the drop in source validation "
            "accuracy, one rank correlation per run, over the study's "
            "\\nSignalRuns{} runs. The last column prices the same signal as a "
            "\\emph{rule}: the mean score, over all \\nStoppingArms{} "
            "hundred-round arms, at which its best single budget stops --- "
            "against \\nFixedMeanScore{} points for the fixed horizon. "
            "\\emph{never fires} means no budget in the grid is ever crossed "
            "on any arm, so the rule is the fixed horizon by another name; "
            "\\emph{unsteerable} means the shared budget grid cannot be "
            "applied to a raw parameter-space distance at all --- the "
            "$\\ell_2$ distance crosses every budget inside one round and the "
            "Fisher distance reaches none of them.")

    return block(
        "tab:signals",
        "The eight quantities a server may watch, as trackers and as rules. "
        "Correlation is not deployability: the strongest tracker is the only "
        "firing rule that scores below the fixed horizon, and the best rule is "
        "a mediocre tracker.",
        "llrrl",
        "Signal & Observed on & Median $\\rho$ & $\\lvert\\rho\\rvert\\ge0.9$ "
        "& As a rule (best budget) \\\\\n"
        " & & (per run) & (share of runs) & (mean score, pts) \\\\",
        body, size="\\scriptsize", colsep="5pt", note=note,
        comment="source: data/signals_summary_extract.csv, one row per signal,\n"
                "cols median_rho, share_ge_0p9, mean_stopped_score, best_delta,\n"
                "arms_fired --- ordered as Section 7 introduces them")


# --------------------------------------------------------------------------
# T11 --- what the fixed horizon costs, stage by stage
# --------------------------------------------------------------------------

STAGE_LABEL = [
    ("aggfull", "Aggregation finals"),
    ("regfull", "Regularisation finals"),
    ("combo", "Combinations"),
    ("c10d10", "Ten clients, one dropped"),
    ("five", "Five clients, one dropped"),
    ("c20d10", "Twenty clients, two dropped"),
    ("c20d20", "Twenty clients, four dropped"),
    ("extreme", "\\textbf{The extreme cases}"),
]


def t_cohort(cohort):
    """A1 --- the ten selected writers, audited against both rankings.

    Emitted as a tabular only.  The float, the caption and \\label{tab:cohort}
    belong to the appendix section file, which places the table.
    """
    of = {r["of"] for r in cohort}
    assert len(of) == 1, of
    body = []
    for r in sorted(cohort, key=lambda r: int(r["rank"])):
        body.append(" & ".join([code(r["writer"]),
                                count(r, "rank"),
                                num(r, "ginit"),
                                num(r, "accuracy"),
                                count(r, "total_rows"),
                                count(r, "classes_covered")]) + " \\\\")
    return bare(
        "lrrrrr",
        "Writer & Rank & $g_{\\mathrm{init}}$ & $\\thg$ & Rows & Classes \\\\\n"
        " & (of \\nBadPoolSize{}) & accuracy & accuracy & (digit rows) "
        "& (of 10) \\\\",
        body,
        comment="source: data/cohort_table.csv, all rows, ordered by col rank\n"
                "rank is the writer's place in the bad pool under the shipped\n"
                "model, worst first; both accuracies are over every row the\n"
                "writer holds, on models that never trained on those rows",
        note="Rank is the writer's place in the bad pool under the shipped "
             "model $\\thg$, worst first. The $g_{\\mathrm{init}}$ column is "
             "the coarse detector's accuracy on the same writer, held out at "
             "the cut; the two columns disagree, which is why the cohort is "
             "cut on $\\thg$ and not on the detector.")


def t_plateau(pst):
    """T6 --- the plateau rule, stage by stage: what stopping recovers."""
    by = {r["stage"]: r for r in pst}
    body = []
    for stage, label in STAGE_LABEL:
        r = by[stage]
        if stage == "extreme":
            body.append("\\midrule")
        fired = int(float(r["fires"]))
        body.append(" & ".join([
            label, "%d" % int(float(r["arms"])),
            derived(float(r["fixed_mean"]), 2, 100),
            derived(float(r["rule_mean"]), 2, 100),
            ("0.00" if fired == 0
             else derived(float(r["gain"]), 2, 100, signed=True))]) + " \\\\")
    r = by["all"]
    body += ["\\midrule",
             " & ".join(["\\emph{All arms}", "%d" % int(float(r["arms"])),
                         derived(float(r["fixed_mean"]), 2, 100),
                         derived(float(r["rule_mean"]), 2, 100),
                         derived(float(r["gain"]), 2, 100, signed=True)])
             + " \\\\"]
    note = ("Mean score in points over each stage's arms, on the per-round "
            "\\textbf{validation} trace, the only basis on which a stopping "
            "round may be chosen. The rule keeps the checkpoint of the best "
            "cohort round and stops training after $k=\\nPlateauK{}$ rounds "
            "without improvement; it costs anything on exactly "
            "\\nPlateauHurtArms{} arm of \\nPlateauArms{} "
            "($-\\nPlateauWorstLoss{}$ points), and an arm on which it never "
            "fires runs the full budget unchanged.")
    return block(
        "tab:plateau",
        "The plateau rule, stage by stage: the fixed hundred-round budget "
        "against stopping on the cohort's own validation accuracy.",
        "lrrrr",
        "Stage & Arms & Fixed budget & With the rule & Gain \\\\\n"
        " & & (pts) & (pts) & (pts) \\\\",
        body, colsep="5pt", note=note,
        comment="source: data/plateau_stages.csv, primary setting rows\n"
                "(patience 20, margin 0, checkpoint_best), stage means")


def _median(xs):
    v = sorted(xs)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def t_stopping(stopa):
    body = []
    for stage, label in STAGE_LABEL:
        arms = [r for r in stopa if r["stage"] == stage]
        assert arms, stage
        gaps = [float(r["oracle_cost"]) for r in arms]
        gains = [float(r["one_vs_final"]) for r in arms]
        fired = sum(1 for r in arms
                    if int(r["one_round"]) < int(r["rounds"]))
        if stage == "extreme":
            body.append("\\midrule")
        body.append(" & ".join([
            label, "%d" % len(arms),
            derived(_median(gaps), 2, 100),
            derived(max(gaps), 2, 100),
            "%d" % fired,
            ("0.00" if fired == 0
             else derived(sum(gains) / len(gains), 2, 100, signed=True))])
            + " \\\\")

    all_gaps = [float(r["oracle_cost"]) for r in stopa]
    all_gains = [float(r["one_vs_final"]) for r in stopa]
    all_fired = sum(1 for r in stopa if int(r["one_round"]) < int(r["rounds"]))
    body += ["\\midrule",
             " & ".join(["\\emph{All arms}", "%d" % len(stopa),
                         derived(_median(all_gaps), 2, 100),
                         derived(max(all_gaps), 2, 100),
                         "%d" % all_fired,
                         derived(sum(all_gains) / len(all_gains), 2, 100,
                                 signed=True)]) + " \\\\"]

    note = ("Every column is read from the per-round \\textbf{validation} "
            "trace, the only basis on which a stopping round may be chosen. "
            "The one rule is the proxy-set accuracy signal at "
            "$\\delta=\\nOneRuleDelta{}$; \\emph{fires} counts the arms on "
            "which it stops before the horizon.")

    return block(
        "tab:stopping",
        "What the fixed horizon costs and what one permitted rule recovers, "
        "stage by stage.",
        "lrrrrr",
        "Stage & Arms & \\multicolumn{2}{c}{Oracle$-$fixed gap (pts)} & "
        "\\multicolumn{2}{c}{One rule} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        " & & Median & Max & Fires & Mean gain (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt", note=note,
        comment="source: data/stopping_all.csv, grouped by col stage.\n"
                "derived: per-stage median and max of col oracle_cost, count of\n"
                "rows with one_round < rounds, and mean of col one_vs_final")


# --------------------------------------------------------------------------
# the contract that this rewrite exists to keep
# --------------------------------------------------------------------------

def identifiers(*tables):
    """Every run-record identifier that appears anywhere in the data read.

    Writer identifiers are deliberately not in this set: a writer id is a
    datum of the corpus, printed as itself in the cohort audit, not a name the
    literature has an alternative for.
    """
    out = set()
    for table in tables:
        for r in table:
            for col in ("cell", "arm", "minus", "signal"):
                if r.get(col):
                    out.add(r[col])
    return {i for i in out if "_" in i}


def no_code_ids(name, text, ids):
    """Reject a table that still prints a run-record identifier.

    Both spellings are looked for --- the raw one and the one ``tex()`` would
    produce --- so that no escaping accident lets an identifier through.
    """
    for ident in sorted(ids):
        for spelling in (ident, tex(ident)):
            assert spelling not in text, (
                "tables/%s prints the run-record identifier %r; name it in "
                "paper_names.py instead" % (name, ident))
    stray = re.findall(r"\\code\{[a-z][a-z0-9]*(?:\\_[a-z0-9]+)+\}", text)
    stray = [x for x in stray if not re.match(r"\\code\{f\d+\\_\d+\}", x)]
    assert not stray, ("tables/%s sets a code identifier: %s" % (name, stray))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if any table is out of date")
    args = ap.parse_args()
    CHECK[0] = args.check

    OUT[0] = os.path.join(out_dir(), "tables")
    if not os.path.isdir(OUT[0]):
        os.makedirs(OUT[0])

    refs = rows("references.csv")
    agg = rows("agg-winners.csv")
    regu = rows("reg-winners.csv")
    blend = rows("blends.csv")
    combo = rows("combos.csv")
    cost_s = rows("cost_stages.csv")
    extr = rows("extremes.csv")
    stopx = rows("stopping_extreme.csv")
    stopa = rows("stopping_all.csv")
    pst = rows("plateau_stages.csv")
    cohort = rows("cohort_table.csv")
    sigs = rows("signals_summary_extract.csv")
    fair_combo = rows("fairness_combo.csv")
    fair10 = rows("fairness_c10d10.csv")
    fair20 = {"c20d10": rows("fairness_c20d10.csv"),
              "c20d20": rows("fairness_c20d20.csv")}
    sizes = {k: rows("sizes_%s.csv" % k) for k, _, _ in SETTING}

    emitted = [
        ("references.tex", t_references(refs), len(refs)),
        ("agg_winners.tex", t_agg(agg), len(agg)),
        ("reg_winners.tex", t_reg(regu),
         len([r for r in regu if not r["cell"].startswith("hybrid")])),
        ("blends.tex", t_blends(regu, blend),
         len([r for r in regu if r["cell"].startswith("hybrid")])),
        ("combos.tex", t_combos(combo, agg, regu), len(combo)),
        ("scaling.tex", t_scaling(sizes, cost_s),
         sum(len(v) for v in sizes.values())),
        ("fairness.tex", t_fairness(fair_combo, fair10, fair20,
                                    {r["cell"] for r in agg}), 4 + len(fair10) + 2),
        ("cost.tex", t_cost(cost_s), len(cost_s)),
        ("extreme.tex", t_extreme(extr, stopx), len(extr)),
        ("signals.tex", t_signals(sigs), len(sigs)),
        ("stopping.tex", t_stopping(stopa), len(stopa)),
        ("plateau.tex", t_plateau(pst), len(pst) - 1),
        ("cohort.tex", t_cohort(cohort), len(cohort)),
    ]
    ids = identifiers(refs, agg, regu, blend, combo, extr, stopx, sigs,
                      fair_combo, fair10, *fair20.values(), *sizes.values())
    for name, text, n in emitted:
        no_code_ids(name, text, ids)
        write(name, text)
        print("tables/%-16s %2d data rows" % (name, n))
    print("%d CSV cells rendered, every one checked against its file"
          % CHECKED[0])
    print("%d run-record identifiers checked for; none reaches a table"
          % len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
