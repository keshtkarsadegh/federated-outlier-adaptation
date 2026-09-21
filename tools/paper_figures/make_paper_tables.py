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
* A column that is a difference against a row of another table names that
  table, and every row it points at is a row that table prints.  ``tab:combos``
  reads six of its eighteen deltas against a COMPOSITE penalty, so
  ``tab:reg_winners`` carries the composite rows the grid crosses and the one
  the study selected instead of filtering every composite out; dropping them
  left six deltas that the page could not be used to reproduce, and a reader
  who recomputed them against the screened blend --- the only mixed penalty
  still printed --- got a different count of how many combinations clear their
  better component.  ``reg_printed`` decides that row set and an assertion in
  ``t_combos`` holds the delta column to it.
* No run-record identifier reaches the page.  Every method is named by
  paper_names.py, which turns a cell id into the published name of the method
  and its setting in that method's own symbol; a table is rejected by an
  assertion at the end of this script if any identifier survives into it.
* A printed number is rounded once, from the full-precision CSV value, and
  the columns of a row are rounded independently.  So the printed gained and
  spent of a row need not differ by exactly its printed score: the identity
  holds where the score was computed, which is at full precision.  numbers.tex
  rounds the same way, which is why no macro disagrees with the table beside
  it; a reader who subtracts two printed numbers may disagree with both.
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

from paper_names import (ARM, COMPOSITE, EXTREME, REFERENCE, SCHEDULE,  # noqa: E402
                         SIGNAL, label, params, settings_note, short,
                         short_mix)


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

#: The composite that LEADS tab:reg_winners on both schedules, named for the
#: same reason the three arms above are: the record is frozen, and a table
#: reports it rather than re-deriving it from its own best row.
#:
#: IT WAS NOT SELECTED, AND THE CONSTANT USED TO SAY IT WAS.  Nothing chose
#: this arm: every shortlist of the study was cut on the validation score, and
#: on the cyclic penalties this one stands third there while the arm the cross
#: actually carried is a different mix.  It leads the TEST ordering, which is a
#: post-selection reading, and the note tab:reg_winners prints has always said
#: so -- so a constant called BLEND_SELECTED contradicted the caption it fed,
#: and the emitter comment underneath it repeated the contradiction into the
#: provenance line of the .tex.
BLEND_TEST_LEADING = "hybrid_seq_mix0p75"

#: The superscript tab:reg_winners puts on a composite row, by the role that
#: earns the row its place in a table of penalties, in printing order.
BLEND_MARK = (("test-leading", "\\dagger"), ("crossed", "\\ddagger"))

#: What the two control rows of tab:agg_winners are, said where the rows are
#: printed.  The second one arms the accuracy-floor stop rule and is otherwise
#: the first, which a reader of a table headed "server rule" cannot be expected
#: to guess -- and "early stop" is a thing the protocol forbids, so a row whose
#: name is only that reads as a run that stopped.  None did: the rule fired on
#: no fold of either schedule, at either horizon, which `cost_arms.csv` records
#: as a full hundred rounds on both control arms and `tests/test_recipe.py`
#: holds it to.
#:
#: THE LAST SENTENCE IS THE ONE THAT HAD TO BE CHECKED.  It used to say the two
#: rows differ "under a different client-sampling seed" and stop there, which
#: invited the reader to price the difference at nothing.  The two rows are
#: emitted by consecutive blocks of one task file (`s10_agg_full2.txt`), at the
#: same hundred rounds, the same five folds, the same eight-of-ten
#: participation and the same run seeds 1--5; the sampler-seed block is
#: 704151--704155 for one and 704161--704165 for the other, and the stop flag
#: is the only other difference and never fired.  So the gap between them IS a
#: re-run of one configuration, and on the cyclic schedule it is
#: \nControlSeedSpread{} points -- larger than \nReplicateGap{}, the gap the
#: text calls a replicate gap, which is measured between two rows that differ
#: by more than a seed (different stage, task file, seed block and trainer
#: entry point).  Both numbers are macros so neither can be retyped.
CONTROL_NOTE = (
    "``FedAvg (control)'' is plain FedAvg at $\\eta_s{=}1$, the row every "
    "other row of this table is measured against. "
    "``FedAvg (control, stop rule armed)'' is that same configuration with the "
    "accuracy-floor stop rule switched on --- it ends a run the round the "
    "global accuracy first falls below the participating clients' own "
    "accuracy or below $0.90$ --- and it is a second control rather than a "
    "server rule. "
    "\\textbf{It never fired.} Both control arms ran the full hundred rounds on "
    "every fold of both schedules, so the pair is one configuration run twice: "
    "same task file, same horizon, same folds, same participation and the same "
    "run seeds, differing only in the client-sampling seed block. They "
    "disagree by \\nControlSeedSpread{} points on the cyclic schedule and "
    "\\nControlSeedSpreadPar{} on the parallel one, which is the run-to-run "
    "spread of a single configuration at this horizon and is the scale every "
    "margin in this paper should be read against. Nothing here reports a run "
    "that stopped early.")

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


#: Small counts in words, for the notes that state a population in prose.
#: The notes spell these out and the table cells print digits, which is the
#: usual split; a count here is always one a generator derived, never typed.
NUMBER_WORD = ("zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine", "ten", "eleven", "twelve", "thirteen",
               "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
               "nineteen", "twenty")


def spell(n):
    """A small count in words; larger ones fall back to digits."""
    return NUMBER_WORD[n] if 0 <= n < len(NUMBER_WORD) else "%d" % n


def count(row, col):
    """An integer CSV cell."""
    raw = row[col]
    assert raw in RAW[row["_src"]], (row["_src"], col, raw)
    v = float(raw)
    CHECKED[0] += 1
    return "%d" % int(round(v))


def grouped(row, col):
    """An integer CSV cell with LaTeX thin-space thousands: ``2\\,381``.

    numbers.tex groups every count it sets, so a footnote that prints one
    count through a macro and the next with ``count`` sets the same quantity
    two ways in one sentence and reads as two different quantities.
    """
    s = count(row, col)
    out = []
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return "\\,".join(out)


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
          colsep=None, note=None, comment="", stretch=None):
    L = ["%% generated by make_paper_tables.py --- do not edit by hand"]
    if comment:
        L += ["%% " + x for x in comment.strip().split("\n")]
    L += ["\\begin{table}[!htbp]", "\\centering",
          "\\caption{%s}" % caption.strip(), "\\label{%s}" % label, size]
    if colsep:
        L.append("\\setlength{\\tabcolsep}{%s}" % colsep)
    # \snresize only ever shrinks a tabular that is too WIDE; nothing here
    # shrinks one that is too TALL, and the tallest table in the paper ran
    # past the text block by two points.  A row-height factor below one is
    # the one knob that buys height without touching the type size, so the
    # table that needs it asks for it and no other table pays for it.
    if stretch:
        L.append("\\renewcommand{\\arraystretch}{%s}" % stretch)
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


#: The table helper, under a second name, because the function above needs a
#: local called `block` for the rows of one schedule and shadowing a helper is
#: how a table silently stops being a table.
block_env = block


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
        "(Eq.~\\eqref{eq:fg}) "
        "and spent are the differences from $\\thg$, score their difference, "
        "Eq.~\\eqref{eq:score}; centralized training is a ceiling the "
        "constraint forbids.",
        "lrrrrr",
        "Reference & Adaptation & Preservation & Gained & Spent & Score \\\\\n"
        " & & & (pts) & (pts) & (pts) \\\\",
        body, size="\\scriptsize",
        note=("Each column is rounded once from the stored value, so gained "
              "and spent as printed need not differ by exactly the score as "
              "printed; the identity holds at the precision the score was "
              "computed at. The same holds of every table here and of every "
              "number the text quotes."),
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
        "grouped by schedule and ordered by score within each block.",
        "lrrrr",
        "Server rule & Adaptation & Preservation & Spent (pts) & Score (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt", note=CONTROL_NOTE,
        comment="source: data/agg-winners.csv, all rows, grouped by col family\n"
                "(parallel block first, cyclic second), ordered by col score\n"
                "within each block")


# --------------------------------------------------------------------------
# T2b --- penalties, plus the composite rows another table reads;
# the construction sweep in full has its own table below
# --------------------------------------------------------------------------

def penalty_population(regu):
    """How many penalty cells the screen ranked, and how many finals there are.

    THE TWO COUNTS ARE NOT THE SAME COUNT, and two tables state one each.
    The screen ranked one cell per single penalty plus every setting of the
    composite the construction stage built -- that is the population
    ``selection_axis.csv`` carries and the population both of its rank
    columns are over.  The screened composite, which swept all three
    coefficients at once instead of the mix alone, was run afterwards and
    joined the finals without a place in that ordering.  So the finals are
    one larger than the ranking, per schedule, and an arm that scores below
    the screened composite stands one place higher in the ranking than it
    does in the finals table.

    Returned as (singles, mixes, finals) and derived from the view, because
    the sentence that states them sits beside the table that prints them.
    """
    per_family = {}
    for family, _ in SCHEDULE_BLOCKS:
        mine = [r["cell"] for r in regu if r["family"] == family]
        screened = [c for c in mine if c.startswith("blend_")]
        mixes = [c for c in mine if c.startswith("hybrid")]
        assert len(screened) == 1, (family, screened)
        per_family[family] = (len(mine) - len(mixes) - len(screened),
                              len(mixes), len(mine))
    assert len(set(per_family.values())) == 1, per_family
    singles, mixes, finals = per_family["parallel"]
    assert singles + mixes + 1 == finals, per_family
    return singles, mixes, finals


def reg_printed(regu, combo, agg_cells):
    """The rows tab:reg_winners carries, each with the roles that earn it one.

    THE COMPOSITE ROWS ARE NOT DECORATION.  tab:combos prints, for every one of
    its eighteen rows, the score minus the score of whichever component scores
    higher on its own, and its note sends the reader to this table to find that
    component.  Six of the eighteen cross a composite penalty, so a table of
    penalties that filtered every composite out left six deltas the page could
    not be used to reproduce --- and worse, left the screened blend standing as
    the only mixed penalty on it, so a reader recomputing against that instead
    got a count of how many combinations clear their better component that the
    text does not report.

    So the filter is not "no composites".  It is every single-method penalty,
    plus the composite rows another printed table reads: the one the grid
    crosses on that schedule, derived from the grid itself rather than assumed,
    and the test-leading one, which no selection chose and which leads the test
    ordering of both blocks.  The remaining composite rows are the construction
    sweep's own subject and are reported with it.
    """
    crossed = {(split_combo(r["cell"], agg_cells)[1], r["family"]) for r in combo}
    out = []
    for r in regu:
        if not r["cell"].startswith("hybrid"):
            out.append((r, ()))
            continue
        roles = [role for role, ok in
                 (("test-leading", r["cell"] == BLEND_TEST_LEADING),
                  ("crossed", (r["cell"], r["family"]) in crossed)) if ok]
        if roles:
            out.append((r, tuple(roles)))
    return out


#: The cell of tab:agg_winners that IS the no-penalty control: plain FedAvg,
#: no client penalty, at the same horizon.  A table of penalties borrows it
#: and marks the borrowed row, so no reader takes it for a row the
#: regularisation stage ran.
NO_PENALTY_CONTROL = "control_fedavg"
NO_PENALTY_MARK = "\\S"


def t_reg(reg_rows, agg, regu):
    """T3 --- every penalty at its best setting, with a control on both blocks.

    WHY THE PARALLEL BLOCK HAS TO BORROW ITS CONTROL.  The regularisation
    finals ran one cell per penalty family per schedule, at that family's own
    best setting.  The proximal family's best setting is mu = 0 on the cyclic
    schedule -- which IS the no-penalty control, and prints as one -- and
    mu = 1e-4 on the parallel schedule, which is not.  So the stage left the
    parallel block with no zero-penalty row at all, and a reader comparing the
    two blocks read the cyclic penalties against a control and the parallel
    ones against nothing at all.  The run that answers it exists and is
    already printed one table earlier: plain FedAvg, no penalty, the same
    hundred rounds, the same five folds and the same participation, is the
    control row of tab:agg_winners.  It is borrowed here, on BOTH schedules so
    that the two blocks are read the same way, and marked.  The assertion at
    the end is what stops a block losing it again.
    """
    singles, mixes, finals = penalty_population(regu)
    body = []
    controls = []
    for fam, title in SCHEDULE_BLOCKS:
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{5}{@{}l}{\\textbf{%s}} \\\\" % title)
        here = [x for x in reg_rows if x[0]["family"] == fam]
        for r, roles in sorted(here, key=lambda x: -float(x[0]["score"])):
            name = "\\quad " + label(r["cell"]) + "".join(
                "$^{%s}$" % mark for role, mark in BLEND_MARK if role in roles)
            body.append(" & ".join([name,
                                    num(r, "adaptation"), num(r, "preservation"),
                                    num(r, "spent", 2, 100),
                                    num(r, "score", 2, 100)]) + " \\\\")
        control = pick(agg, cell=NO_PENALTY_CONTROL, family=fam)
        controls.append(fam)
        body.append(" & ".join([
            "\\quad " + label(NO_PENALTY_CONTROL) + "$^{%s}$" % NO_PENALTY_MARK,
            num(control, "adaptation"), num(control, "preservation"),
            num(control, "spent", 2, 100),
            num(control, "score", 2, 100)]) + " \\\\")
    check_schedules([r for r, _ in reg_rows])
    assert sorted(controls) == sorted(fam for fam, _ in SCHEDULE_BLOCKS), (
        "a schedule block of tab:reg_winners carries no no-penalty control, so "
        "its penalties are read against nothing while the other block's are "
        "read against one: %s" % controls)
    # THE FOOTNOTE QUOTES FOUR NUMBERS IN PROSE, AND NOT ONE OF THEM IS
    # RETYPED.  A note that says the head of a block is a tie, and prints the
    # margin that makes it one, is a claim about the rows directly above it:
    # if the ordering moves and the sentence does not, the table disproves its
    # own footnote.  Each is differenced or read here, from the same rows the
    # body was built from, and rounded once.
    leads = {}
    for fam, _ in SCHEDULE_BLOCKS:
        here = sorted((float(r["score"]) for r, _ in reg_rows
                       if r["family"] == fam), reverse=True)
        leads[fam] = derived(here[0] - here[1], 2, 100)
    # The composite the cyclic validation ordering ranked third and this table
    # does not print: the row that makes "over the rows printed" true.
    crossed_par = [r["cell"] for r, roles in reg_rows
                   if r["family"] == "parallel" and "crossed" in roles]
    assert len(crossed_par) == 1, crossed_par
    unprinted = pick(regu, cell=crossed_par[0], family="cyclic")
    assert (unprinted["cell"], "cyclic") not in {(r["cell"], r["family"])
                                                 for r, _ in reg_rows}, (
        "the footnote says tab:reg_winners does not print the cyclic %s row, "
        "and it does" % unprinted["cell"])
    provenance = ("giving the \\emph{cyclic-tuned} composite at "
                  "$\\lambda{=}%s, T{=}%s$ and the \\emph{parallel-tuned} one "
                  "at $\\lambda{=}%s, T{=}%s$"
                  % (COMPOSITE["cyclic"] + COMPOSITE["parallel"]))
    note = ("$\\S$ The no-penalty control: plain FedAvg with no client penalty, "
            "at the same horizon, the same five folds and the same "
            "participation --- the row of Table~\\ref{tab:agg_winners} that "
            "every penalty here is measured against. It is borrowed on both "
            "schedules because this stage ran one cell per penalty family at "
            "that family's own best setting, and the proximal family's best "
            "setting is $\\mu{=}0$ on the cyclic schedule --- which is the "
            "control, and is the ``No penalty (control)'' row of that block "
            "--- but $\\mu{=}10^{-4}$ on the parallel one, which is not. The "
            "cyclic block therefore carries the same nominal configuration "
            "twice --- no penalty, plain FedAvg, a hundred rounds --- once "
            "from each stage, and they disagree by \\nReplicateGap{} points. "
            "That gap is not a pure replicate: the two runs come from "
            "different task files, different seed blocks and different "
            "trainer entry points. The pure one, two runs of one task file "
            "differing only in the client-sampling seed, is the pair of "
            "control rows of Table~\\ref{tab:agg_winners} and is "
            "\\nControlSeedSpread{} points. "
            "The composite penalties are the two the construction stage "
            "built, "
            "$\\lambda\\,[\\,m\\,D_{\\mathrm{kd}} + (1-m)\\,D_{\\mathrm{fisher}}\\,]$, "
            "one per schedule out of that schedule's own selected "
            "distillation and consolidation components: each inherited their "
            "$\\lambda$ and $T$ and swept the mix alone, "
            + provenance + ", and both were run under both schedules. "
            "The \\emph{screened} rows are the separate screen that "
            "swept all three coefficients together, which is why the two are "
            "named apart and carry different coefficients here. Of the "
            "composite rows, this table carries the two that other tables "
            "read: "
            "$\\ddagger$ is the composite that schedule's validation ordering "
            "selected --- $m{=}0.5$ on both schedules --- which is therefore "
            "the one Table~\\ref{tab:combos} crosses and reads its $\\Delta$ "
            "against (Table~\\ref{tab:selection_axis} prints the ordering). "
            "$\\dagger$ is the composite that leads this table on both "
            "schedules: the \\emph{test-leading} composite, a post-selection "
            "reading and not a selection, and the arm the extension of "
            "Section~\\ref{sec:results:jointtune} tunes. Its lead is $"
            + leads["parallel"] + "$ points over the next row on the parallel "
            "schedule and $" + leads["cyclic"] + "$ on the cyclic, both far "
            "inside the \\nControlSeedSpread{}-point spread of one "
            "configuration run twice, so the head of each block is a tie and "
            "not an ordering. The ordering is over the rows printed and not "
            "over every arm of the block: each schedule ranked "
            + spell(singles + mixes) + " penalty cells --- "
            + spell(singles) + " single penalties and " + spell(mixes)
            + " settings of the constructed composite --- and the "
            "\\emph{screened} row is a " + spell(finals) + "th final that "
            "ordering never held: it was run after the screen, so the two "
            "rank columns of Table~\\ref{tab:selection_axis} are over "
            + spell(singles + mixes) + " arms where this block's finals "
            "number " + spell(finals) + ". The composite rows here are the "
            "ones other tables read rather than every mix --- the cyclic "
            "composite at "
            "$\\lambda{=}" + COMPOSITE["cyclic"][0] + ", T{=}"
            + COMPOSITE["cyclic"][1] + ", m{=}0.5$ scores $"
            + num(unprinted, "score", 2, 100) + "$ and would stand third in "
            "that block; it is printed in "
            "Table~\\ref{tab:selection_axis}. The rest of the sweep is "
            "reported with the construction it belongs to.")
    return block(
        "tab:reg_winners",
        "Every client penalty at its own best setting, at the full horizon "
        "under plain FedAvg on the server, together with the composite rows "
        "the other tables read. Five-fold means, \\textbf{test} axis, "
        "grouped by schedule and ordered by score within each block over "
        "the rows printed; "
        "``No penalty (control)'' is the proximal term at $\\mu{=}0$, and the "
        "row marked $\\S$ at the foot of each block is the study's own "
        "no-penalty control, carried here so that both blocks are read "
        "against one.",
        "lrrrr",
        "Client penalty & Adaptation & Preservation & Spent (pts) & Score (pts) \\\\",
        body, size="\\scriptsize", colsep="5pt", note=note,
        comment="source: data/reg-winners.csv, every row whose cell does not\n"
                "start with 'hybrid', plus the 'hybrid' rows another table\n"
                "reads --- the penalty component of a data/combos.csv row\n"
                "on the same schedule, and the test-leading composite (mix 0.75,\n"
                "cyclic-tuned) --- grouped by col family (parallel block\n"
                "first, cyclic second), ordered by col score within each\n"
                "block. The last row of each block is the plain-FedAvg\n"
                "control row of data/agg-winners.csv for that family,\n"
                "which is the study's no-penalty control")


# --------------------------------------------------------------------------
# T2c --- the blends, with the fold-paired verdict against each parent
# --------------------------------------------------------------------------

def t_blends(regu, blend):
    hyb = [r for r in regu if r["cell"].startswith("hybrid")]
    assert len(hyb) == 12, len(hyb)

    # blends.csv holds one comparison per (blend, parent) on the schedule the
    # blend's two components were selected on; index it by that pair.
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
            "differences, in points, against the distillation component and "
            "the consolidation component the blend was built from, with the "
            "number of folds on which "
            "the difference is positive. A blend is compared on the schedule "
            "its two components were selected on, so the six rows run off "
            "that "
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
        "built from the parallel schedule's selected components; the "
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


def t_combos(combo, agg, regu, printed):
    agg_cells = {r["cell"] for r in agg}
    check_schedules(combo)
    ordered = []
    body = []
    pen_better = []
    for fam, title in SCHEDULE_BLOCKS:
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{6}{@{}l}{\\textbf{%s}} \\\\" % title)
        for r in by_score([x for x in combo if x["family"] == fam]):
            ordered.append(r)
            rule, pen = split_combo(r["cell"], agg_cells)
            a = pick(agg, cell=rule, family=r["family"])
            b = pick(regu, cell=pen, family=r["family"])
            comp = a if float(a["score"]) >= float(b["score"]) else b
            assert (comp["cell"], r["family"]) in printed, (
                "the delta for %s on the %s schedule is taken against %s, "
                "which neither tab:agg_winners nor tab:reg_winners prints"
                % (r["cell"], r["family"], comp["cell"]))
            d = 100.0 * (float(r["score"]) - float(comp["score"]))
            CHECKED[0] += 1
            pen_better.append(comp["cell"] == pen)
            # The penalty half carries its mix.  The settings note below
            # gives every coefficient once, but two rows of this table cross
            # two different composites and the note is not where a reader
            # should have to go to tell one row from another.
            name = "\\quad " + short(rule) + " & " + short_mix(pen)
            if r["cell"] == CROWNED[0]:
                name += "$^{\\star}$"
            body.append(" & ".join([
                name,
                num(r, "adaptation"), num(r, "preservation"),
                num(r, "score", 2, 100),
                ("$+%.2f$" % d) if d >= 0 else ("$%.2f$" % d)]) + " \\\\")

    assert all(pen_better), "a server rule outscores its penalty somewhere"

    # WHICH COMPOSITE A BLOCK CROSSES IS PART OF THE READING.  The two
    # composites are told apart by their coefficients alone, and each block
    # crosses the one its OWN validation ordering selected -- which on both
    # schedules is the one built from the other schedule's components.  A
    # settings list that prints "cyclic-tuned" and stops leaves the reader of
    # the parallel block to conclude the opposite, so each entry is annotated
    # with the schedule that selected it, derived from the crossing the block
    # actually prints rather than restated.
    settings = settings_note([(r["cell"], r["family"]) for r in ordered])
    for fam, _ in SCHEDULE_BLOCKS:
        crossed = sorted({split_combo(r["cell"], agg_cells)[1]
                          for r in ordered if r["family"] == fam
                          and split_combo(r["cell"], agg_cells)[1]
                          .startswith("hybrid")})
        assert len(crossed) == 1, (fam, crossed)
        entry = "%s %s" % (short(crossed[0]), params(crossed[0]))
        assert settings.count(entry) == 1, (entry, settings)
        settings = settings.replace(
            entry, entry + " (the composite the %s schedule's validation "
                           "ordering selected)" % fam)

    note = ("Settings, given once for the whole table: "
            + settings + ". "
            "The two composites are the two the construction stage built, "
            "one per schedule (Section~\\ref{sec:method:reg}); each "
            "schedule's block crosses the one its own validation ordering "
            "selected, which on both schedules is the composite built from "
            "the other schedule's components. "
            "$\\Delta$ vs.\\ better comp.\\ is the row's score minus the "
            "score of whichever of its two components, the rule alone or the "
            "penalty alone, scores higher on its own, read from "
            "Tables~\\ref{tab:agg_winners} and~\\ref{tab:reg_winners}. In all "
            "eighteen rows that better component is the \\emph{penalty}, and "
            "every "
            "one of the eighteen is printed in Table~\\ref{tab:reg_winners} "
            "--- the two composite penalties, one per schedule, that six of "
            "these rows cross are the rows marked $\\ddagger$ there. The "
            "column is "
            "a difference of five-fold means, not a fold-paired verdict, which "
            "the text reports instead. $\\star$ marks the crowned pair.")

    return block(
        "tab:combos",
        "All eighteen combinations: each schedule's three best server rules "
        "crossed with the three best penalty families on validation score, "
        "one cell per family, at the full horizon. Five-fold "
        "means, \\textbf{test} axis, grouped by schedule and ordered by score "
        "within each block.",
        "llrrrr",
        "Server rule & Penalty & Adapt. & Pres. & Score (pts) & "
        "$\\Delta$ vs.\\ better comp.\\ (pts) \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/combos.csv, all rows, grouped by col family\n"
                "(parallel block first, cyclic second), ordered by col score\n"
                "within each block; the delta column is derived --- combos\n"
                "score minus the larger of the two parents' scores in\n"
                "data/agg-winners.csv and data/reg-winners.csv, same schedule,\n"
                "and asserted to be a row one of those two tables prints")


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

    # THE TWO ROWS ARE TWO SERVER RULES.  label(BALANCED[0]) names the arm
    # correctly and still lets it be read as the control row with a penalty
    # added, which is the one comparison this table does not make: the control
    # is plain FedAvg at eta_s = 1 and the carry arm steps at 0.95.  So the
    # arm is spelled out and the difference said, pinned to the frozen record
    # below so a changed carry arm cannot leave the prose describing the old.
    assert BALANCED[0] == "eta_0p95_hybrid_seq_mix0p5", BALANCED
    note = ("Carried arms, run unchanged: \\emph{" + ARM["winner"] + "} is "
            + label(CROWNED[0]) + "; \\emph{" + ARM["balanced"] + "} is the "
            "server step at $\\eta_s{=}0.95$ with the cyclic-tuned KD+EWC "
            "composite at $m{=}0.5$, which is a \\emph{different server rule} "
            "from \\emph{" + ARM["control"] + "}, plain FedAvg at "
            "$\\eta_s{=}1$ and printed once per schedule --- the carried row "
            "and the parallel control row are two rules, not one rule with "
            "and without a penalty; and \\emph{" + ARM["sequential"]
            + "} is " + label(CYCLIC_ARM[0])
            + ". The twenty-client settings omit the controls. Each block "
            "is scored against the shipped model's own accuracy on that "
            "block's cohort and not against one shared figure --- "
            "\\nGZeroCohortAcc{} at ten clients, \\nGZeroCohortAccFive{} at "
            "five and \\nGZeroCohortAccTwenty{} at twenty under either "
            "dropout rate --- so every score printed here can be recomputed "
            "from the two accuracies beside it.")

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

#: Every block tab:fairness prints, in order: the key into the view bundle, the
#: cost_stages row whose participation the block header states, and the title.
#: The four carry settings are the four of tab:scaling, so the two tables list
#: the same arms at the same settings and a reader can join them row for row;
#: the search setting leads because it is the setting the arms were chosen at.
FAIRNESS_BLOCKS = (
    ("combo", "combinations",
     "Search setting, the combination stage (ten clients)"),
    ("c10d10", "ten clients, one dropped", "Carry setting, ten clients"),
    ("five", "five clients, one dropped", "Carry setting, five clients"),
    ("c20d10", "twenty, two dropped",
     "Carry setting, twenty clients, two dropped"),
    ("c20d20", "twenty, four dropped",
     "Carry setting, twenty clients, four dropped"),
)


def t_fairness(fair, sizes, combo, cost_s):
    """A1 --- the per-client distribution of every carried arm, every setting.

    THE MEAN COLUMN IS THE COHORT ACCURACY, NOT THE MEAN OVER CLIENTS.  It used
    to be the second, and the second is a different number: the clients hold
    between seventy and two hundred digit rows apiece, so an average over
    clients and an accuracy pooled over their rows do not agree, and the
    fairness table printed an adaptation for the crowned arm that disagreed
    with Tables 4 and 7 by up to a point at twenty clients.  Two tables of one
    arm that print two adaptations are a table a reader cannot join.  So this
    column is ``cohort`` --- ``final_evaluation.clients.accuracy``, the exact
    column ``report_tables.py`` reports as adaptation --- and the assertion
    below holds every row of it to the row the reporting table prints.  The
    per-client spread is still the point of the table and is still here: it is
    the three columns beside the mean.

    AND EVERY CARRIED ARM IS PRINTED AT EVERY SETTING.  The introducing
    sentence promised that and the table did not keep it: there was no
    five-client block at all and the twenty-client blocks carried one arm of
    the three, the one that came out negative.  Printing the negative case
    alone is the version of this table a reader has the least reason to trust.
    """
    def line(name, r):
        return " & ".join([name, r["family"],
                           num(r, "cohort"), num(r, "min"),
                           num(r, "lift_worst", 2, 100, signed=True),
                           num(r, "improved", 2)]) + " \\\\"

    def reported(key, cell, family):
        """The row the score table prints for the same arm, or None."""
        table = combo if key == "combo" else sizes[key]
        hit = [x for x in table if x["cell"] == cell and x["family"] == family]
        return hit[0] if len(hit) == 1 else None

    best = max(fair["combo"], key=lambda r: float(r["lift_worst"]))
    body = []
    listed = []
    shown = []
    for key, stage, title in FAIRNESS_BLOCKS:
        if body:
            body.append("\\midrule")
        cs = pick(cost_s, stage=stage)
        body.append("\\multicolumn{6}{@{}l}{\\emph{%s, %s participating per "
                    "round}} \\\\" % (title, loose(cs, "clients_per_round")))
        if key == "combo":
            here = [("\\quad " + ARM["winner"], CROWNED),
                    ("\\quad " + ARM["balanced"], BALANCED),
                    ("\\quad " + ARM["sequential"], CYCLIC_ARM),
                    ("\\quad best worst-client", (best["cell"], best["family"]))]
            rows = [(name, pick(fair[key], cell=cell, family=family))
                    for name, (cell, family) in here]
        else:
            rows = [("\\quad " + ARM[cell], r)
                    for cell in ARM_ORDER
                    for r in [x for x in fair[key] if x["cell"] == cell]]
        for name, r in rows:
            shown.append(r)
            # ONE ARM, ONE ADAPTATION.  The mean column of this table and the
            # adaptation column of the table that reports the same arm are the
            # same measurement of the same runs, and this is what says so.
            row = reported(key, r["cell"], r["family"])
            assert row is not None and abs(float(r["cohort"])
                                           - float(row["adaptation"])) < 5e-7, (
                "tab:fairness would print a cohort accuracy for %s/%s at the %s "
                "setting that the table reporting it does not print"
                % (r["cell"], r["family"], key))
            listed.append((key, r["cell"], r["family"]))
            body.append(line(name, r))

    # WHAT THE INTRODUCING SENTENCE PROMISES, CHECKED.  Every arm the score
    # table prints at a carry setting is an arm this table prints at that
    # setting; a record that never ran would be named here rather than dropped.
    missing = []
    for key, _, _ in FAIRNESS_BLOCKS:
        if key == "combo":
            continue
        for r in sizes[key]:
            if (key, r["cell"], r["family"]) not in listed:
                missing.append((key, r["cell"], r["family"]))
    assert not missing, (
        "tab:scaling reports these arms and tab:fairness does not list them, "
        "so the sentence that introduces it is false: %s" % missing)

    assert BALANCED[0] == "eta_0p95_hybrid_seq_mix0p5", BALANCED
    lifted = {k: num(pick(fair[k], cell="balanced"), "improved", 2)
              for k in ("c20d10", "c20d20")}
    assert all(float(r["improved"]) < 0.995 for r in shown), (
        "the note says the lifted column reaches 1.00 nowhere, and a row "
        "this table prints does")
    note = ("Mean is the cohort accuracy pooled over the cohort's test rows "
            "--- the same number Tables~\\ref{tab:combos} "
            "and~\\ref{tab:scaling} print as adaptation for the same arm, "
            "which is why the two tables can be read against each other. The "
            "three columns beside it are the per-client reading the mean "
            "hides: the worst-served client's own accuracy, that client's "
            "accuracy minus \\emph{its own} accuracy under the shipped model, "
            "in points, and the share of client-fold pairs that finish above "
            "their own shipped-model accuracy. Every entry is a five-fold "
            "mean over the per-client records, on the \\textbf{test} axis, and "
            "each block states its own participation in its header. "
            "The shipped-model reference $\\Delta$ worst is taken against is "
            "the five-fold mean of the worst-served client \\emph{of each "
            "fold}: \\nGZeroWorstFoldMeanFive{} at five clients, "
            "\\nGZeroWorstFoldMeanTen{} at ten and "
            "\\nGZeroWorstFoldMeanTwenty{} at twenty. It is not "
            "\\nGZeroClientMin{}, the figure quoted for the shipped model's "
            "worst client elsewhere: that is the worst client's own five-fold "
            "mean. Same writers, same rows, same model --- the minimum and "
            "the mean taken in the other order, and the worst-served client "
            "is not the same writer on every fold. "
            "In the ten-client carry setting every arm lifts its worst-served "
            "client, by between \\nFairWorstMinGain{} and "
            "\\nFairWorstMaxGain{} points. The only negative "
            "$\\Delta$ worst in the study is the " + ARM["balanced"] + " arm "
            "at twenty clients: \\nBalancedWorstTwentyTwo{} points "
            "with two of the twenty dropped and "
            "\\nBalancedWorstTwentyFour{} with four --- and in both the "
            "lifted column stands at $" + lifted["c20d10"] + "$ and $"
            + lifted["c20d20"] + "$, so between a third and two fifths of "
            "client--fold pairs finished \\emph{below} their own "
            "shipped-model accuracy, which is exactly what a cohort mean "
            "hides. The lifted column reaches $1.00$ nowhere in this table. "
            "Arms: " + ARM["winner"] + " is "
            + label(CROWNED[0]) + "; " + ARM["balanced"] + " is the server "
            "step at $\\eta_s{=}0.95$ with the cyclic-tuned KD+EWC composite "
            "at $m{=}0.5$; " + ARM["sequential"] + " is "
            + label(CYCLIC_ARM[0]) + "; " + ARM["control"] + " is plain "
            "FedAvg at $\\eta_s{=}1$, a different server rule from the "
            "server step and not the same rule without a penalty; and the "
            "best worst-client pair is "
            + label(best["cell"]) + ". The twenty-client settings ran no "
            "control.")

    return block(
        "tab:fairness",
        "The per-client distribution at the end of the budget, each client "
        "read against its own accuracy under the shipped model. Every carried "
        "arm at every setting it was carried to. "
        "\\textbf{Test} axis, five-fold means. The mean column is the cohort "
        "accuracy the selection rule optimises, the same one Tables~"
        "\\ref{tab:combos} and~\\ref{tab:scaling} report; the three columns "
        "beside it are what that mean can hide.",
        "llrrrr",
        "Arm & Schedule & Mean & Worst & $\\Delta$ worst & Lifted \\\\\n"
        " & & & client & (pts) & \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/fairness_combo.csv (search setting) and\n"
                "data/fairness_{c10d10,five,c20d10,c20d20}.csv (the four\n"
                "carry settings), every arm each one carries, in the arm order\n"
                "of tab:scaling; participation per block is cost_stages.csv\n"
                "col clients_per_round. The Mean column is col cohort, which\n"
                "is asserted equal to col adaptation of the table that reports\n"
                "the same arm (data/combos.csv, data/sizes_*.csv)")


# --------------------------------------------------------------------------
# A1 --- the axis the selecting ran on, beside the axis the tables report
# --------------------------------------------------------------------------

#: Each stage that selected, the title its block carries, and the shipped view
#: its test columns are joined from.  A stage cut its shortlist out of one
#: catalogue and is reported out of one table; pairing the two here is what
#: keeps the halves of a row two readings of one set of runs rather than two
#: numbers that happen to sit side by side.
SELECTION_STAGES = (
    ("aggregation", "Server rules: the shortlist the cross was built from", "agg"),
    ("regularisation", "Client penalties: the shortlist the cross was built from",
     "reg"),
    ("combination", "The cross: eighteen pairs, one crowned", "combo"),
)

#: How many arms of a block are printed.  A SHORTLIST OF THREE WHOSE RANKS READ
#: 1, 2 AND 5 CANNOT BE RECONCILED AGAINST A TABLE OF THREE ROWS: the arms the
#: rule stepped over have to be on the page, or the rule is invisible and the
#: ranks look like a mistake.  Five is the depth at which every shortlist of
#: this study is covered; anything the rule reached below five is printed as
#: well, because the promise is that a shortlisted arm is never off the block.
SELECTION_DEPTH = 5

#: The mark a printed arm carries for the role it played.
SELECTION_MARK = {"crowned": "\\dagger", "shortlist": "\\ddagger"}

#: The mark for an arm the paper carries PAST the selecting stages.
SELECTION_CARRY_MARK = "\\S"

#: The arms tab:scaling runs at the four federation settings, keyed the way
#: selection_axis.csv keys its rows.
#:
#: A CARRIED ARM IS NOT ALWAYS A SELECTED ONE, AND THE TABLE USED TO PRINT ONLY
#: THE SELECTED.  The role column of the view marks what a SELECTING stage did
#: -- crowned, or shortlisted -- and the balanced arm was neither: validation
#: put it last of its schedule's nine crossings, at rank nine, below the
#: depth this table prints.  So the one carried arm whose validation standing a
#: reader would most want to see, the arm Table~\ref{tab:scaling} runs at every
#: setting and the only one of the eighteen with a negative delta against its
#: better component, was the one arm the selection-axis table left out.  The
#: table now prints every arm with a carried role at whatever depth it sits.
SELECTION_CARRIED = (CROWNED, BALANCED, CYCLIC_ARM)

#: WHICH RULE CARRIED EACH MARKED ARM.  Three arms are run again at the four
#: federation settings and only one of them was ever selected; the other two
#: were carried by rules stated in the prose and nowhere in the view, so the
#: rule is written beside the arm here and printed in the note.  A reader can
#: then check each against the block it is printed in instead of taking the
#: sentence on trust.
SELECTION_CARRY_RULE = {
    CROWNED: "crowned by the cross on validation",
    BALANCED: "least preservation given up of the nine parallel crossings "
              "on validation",
    CYCLIC_ARM: "strongest cyclic pair on the crowning record's validation "
                "ranking",
}

#: Stages whose blocks print EVERY ranked arm instead of the top
#: SELECTION_DEPTH.  The cross is nine crossings per schedule, and the paper
#: argues about the arms at ranks 1, 2, 3 and 9 of them: a block cut at five
#: prints the claim and hides the arm it is made of.  Nine rows is also the
#: whole population, so the block header stops having a depth to state.
SELECTION_FULL = ("combination",)

#: Arms printed in their block at whatever depth they sit, because a claim
#: made elsewhere is about them.  The composite at m = 0.75 leads BOTH test
#: orderings of tab:reg_winners and no validation ordering selected it; that
#: sentence is checkable only when its validation rank is on the page in both
#: penalty blocks, and on the parallel one it sits at rank six, below the cut.
#: It carries the same dagger tab:reg_winners marks it with, which is free
#: here because the crowning dagger lives in the cross table and not this one.
SELECTION_SHOWN = ((BLEND_TEST_LEADING, "parallel"),
                   (BLEND_TEST_LEADING, "cyclic"))


def selection_marks(row, family):
    """Every mark a selection-axis row carries: role first, then carry."""
    marks = SELECTION_MARK[row["role"]] if row["role"] else ""
    if (row["cell"], family) in SELECTION_SHOWN:
        marks += BLEND_MARK[0][1]
    if (row["cell"], family) in SELECTION_CARRIED:
        marks += SELECTION_CARRY_MARK
    return marks


def selection_block(sel, stage, family):
    """One stage-by-schedule block of tab:selection_axis, as it is printed.

    Returned as (every ranked arm, the arms inside the depth, the marked arms
    below it) so that make_numbers counts the rows the PAGE prints and not the
    rows the view holds -- the two differ by the carried arms below the cut.
    """
    block = sorted([r for r in sel if r["stage"] == stage
                    and r["schedule"] == family],
                   key=lambda r: int(r["val_score_rank"]))
    assert block, (stage, family)
    depth = len(block) if stage in SELECTION_FULL else SELECTION_DEPTH
    carried = [r for r in block[depth:] if selection_marks(r, family)]
    return block, block[:depth], carried


SELECTION_BASIS = (
    "Basis, column by column: the four columns under "
    "\\textbf{validation} are the fold-mean cohort accuracy, the "
    "fold-mean source accuracy and the selection score of "
    "Eq.~\\eqref{eq:score} that each selection record was cut on, and "
    "the arm's place in that score's ordering --- which is the "
    "ordering that did the choosing. Both scores are taken against one "
    "pair of baselines, the shipped model's \\emph{test}-axis $A_0$ "
    "and $P_0$, on the validation side as well as the test side: that "
    "is what the selection records themselves did, so the validation "
    "score printed here is the number each shortlist was cut on and "
    "not a re-derivation of it. The four columns under "
    "\\textbf{test} are the row the reporting table prints for the "
    "same arm, and \\emph{Rank} there is that arm's place in the same "
    "population ordered by the test score. The two \\emph{Rank} columns "
    "share one denominator --- every arm that schedule's record ranked, "
    "which the block header states --- and it is neither the rows this "
    "table prints nor the rows the reporting table prints. "
    "The two orderings are not the same ordering. ")

SELECTION_HEAD = ("Arm & \\multicolumn{4}{c}{Validation (the axis that selected)} & "
                  "\\multicolumn{4}{c}{Test (the axis the tables report)} \\\\\n"
                  "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
                  " & Adapt. & Pres. & Score (pts) & Rank & "
                  "Adapt. & Pres. & Score (pts) & Rank \\\\")

SELECTION_SOURCE = ("source: data/selection_axis.csv, grouped by col stage then\n"
                    "by col schedule, ordered by col val_score_rank inside each\n"
                    "block; a selecting block is cut at the first five plus any\n"
                    "row a mark is owed to, the cross block prints all nine; the\n"
                    "four test columns are joined by (cell, family) onto\n"
                    "data/agg-winners.csv, data/reg-winners.csv and\n"
                    "data/combos.csv")


def selection_body(sel, reported, stages):
    """The rows of one selection-axis table, block by block.

    THE TABLE IS TWO TABLES AND WAS ONE.  Printing all nine crossings per
    schedule, and the test-leading composite in both penalty blocks, put the
    single float 47pt over a page; at an ``\\arraystretch`` tight enough to
    close that gap the rows touch.  So the selecting stages and the cross they
    were built into are two floats, which also frees the dagger: it marks the
    crowned pair in one table and the test-leading composite in the other, and
    neither table ever prints both meanings.
    """
    body = []
    for stage, title, which in stages:
        stage_rows = [r for r in sel if r["stage"] == stage]
        assert stage_rows, stage
        if body:
            body.append("\\midrule")
        body.append("\\rowcolor{blockband}\\multicolumn{9}{@{}l}{\\textbf{%s}} \\\\"
                    % title)
        for fam, fam_title in SCHEDULE_BLOCKS:
            block, within, extra = selection_block(stage_rows, stage, fam)
            shown = within + extra
            # BOTH RANK COLUMNS HAVE ONE DENOMINATOR AND THE HEADER SAYS
            # WHICH.  export_selection_axis ranks the record's population
            # twice, once on each axis, so the test rank is over the arms
            # that schedule's record ranked and not over the rows the
            # reporting table prints -- which is a larger set on the
            # penalties, where the screened composite joined the finals
            # after the screen had ranked these.  A reader checking a test
            # rank against Table 3 lands a place out without being told.
            ranked = count(block[0], "ranked")
            if len(within) == len(block):
                head = ("%s: all %s ranked arms, by validation score"
                        % (fam_title, ranked))
            else:
                head = ("%s: the top %s of %s ranked arms by validation score"
                        % (fam_title, len(within), ranked))
                if extra:
                    head += (", plus %s printed from below that depth"
                             % ("one arm" if len(extra) == 1
                                else "%d arms" % len(extra)))
            head += ("; both rank columns are over those %s ranked arms"
                     % ranked)
            body.append("\\multicolumn{9}{@{}l}{\\emph{%s}} \\\\" % head)
            for r in shown:
                row = pick(reported[which], cell=r["cell"], family=fam)
                # One set of runs, two readings of it.  A drift here would mean
                # the frozen record and the shipped view no longer describe the
                # same runs, which is exactly the failure this table would
                # otherwise hide behind two plausible-looking columns.
                assert abs(float(r["test_adaptation"])
                           - float(row["adaptation"])) < 5e-5, (stage, r["cell"])
                assert abs(float(r["test_score"])
                           - float(row["score"])) < 5e-5, (stage, r["cell"])
                mark = selection_marks(r, fam)
                name = "\\quad " + label(r["cell"])
                if mark:
                    # A composite row ends in its own maths, and the mark
                    # opens maths again: butted together they spell "$$" in
                    # the source.  A thin space is the join, and it is the
                    # space the mark would want after a number anyway.
                    name += ("\\," if name.endswith("$") else "")
                    name += "$^{%s}$" % mark
                body.append(" & ".join([
                    name,
                    num(r, "val_adaptation"), num(r, "val_preservation"),
                    num(r, "val_score", 2, 100), count(r, "val_score_rank"),
                    num(row, "adaptation"), num(row, "preservation"),
                    num(row, "score", 2, 100), count(r, "test_score_rank"),
                ]) + " \\\\")
    return body


def selection_rank_of(sel, cell, family, stage):
    """One arm's validation-score rank, read rather than written down."""
    row = next(r for r in sel if r["cell"] == cell
               and r["schedule"] == family and r["stage"] == stage)
    return int(row["val_score_rank"])


def t_selection_axis(sel, agg, regu, combo):
    """A1 --- the two selecting stages, on the axis that selected them.

    THE RANK COLUMNS ARE ON THE SCORE, AND THEY DID NOT USE TO BE.  Every
    shortlist of this study was cut by ``study_emit.ranked_by_trade`` --- the
    selection score of Eq. 2, what an arm gained on the cohort less what it
    spent on the source population, at w = 1.  The rank this table printed came
    from the records' ``rankings`` block instead, which
    ``study_emit.both_rankings`` writes ordered by ADAPTATION alone.  So the
    table ranked the arms on a quantity that chose nothing, its ranks
    reproduced the adaptation ordering exactly, and it printed no preservation
    on the validation side at all -- which is the half of the score that makes
    the ordering differ.  Both validation columns and both orderings are now
    in the view and both are printed, the score rank first because it is the
    one that selected.

    The test columns are read from the table that REPORTS the arm and not from
    the view, which carries a test figure of its own for the same runs.  They
    agree --- the two assertions inside ``selection_body`` are what say so ---
    and printing the reported one is what stops this table from becoming a
    second, slightly different copy of Tables 2 to 4.
    """
    reported = {"agg": agg, "reg": regu, "combo": combo}
    body = selection_body(sel, reported, SELECTION_STAGES[:2])
    check_schedules([{"family": r["schedule"]} for r in sel])

    # The claim the dagger makes is a claim about two ranks, so the two ranks
    # are read out of the view and written into the note. A composite that had
    # been selected somewhere would change these numbers and would change the
    # sentence with them, rather than leaving a stale sentence beside a moved
    # table.
    lead_parallel = selection_rank_of(sel, BLEND_TEST_LEADING, "parallel",
                                      "regularisation")
    lead_cyclic = selection_rank_of(sel, BLEND_TEST_LEADING, "cyclic",
                                    "regularisation")
    for family, rank in (("parallel", lead_parallel), ("cyclic", lead_cyclic)):
        row = next(r for r in sel if r["cell"] == BLEND_TEST_LEADING
                   and r["schedule"] == family
                   and r["stage"] == "regularisation")
        assert not row["role"], (family, row["role"])

    # WHICH POPULATION THE PENALTY RANKS ARE OVER, STATED RATHER THAN LEFT
    # TO BE INFERRED.  export_selection_axis ranks one population twice, so
    # the test rank is over the cells the screen ranked -- and the screened
    # composite is not one of them: it swept all three coefficients at once,
    # ran after the screen and joined the finals with no place in either
    # ordering.  A reader who checks a test rank here against
    # tab:reg_winners therefore lands one place out for every arm below the
    # screened composite, and both counts are read off the two views so that
    # the sentence saying so cannot go stale.
    ranked = {int(r["ranked"]) for r in sel if r["stage"] == "regularisation"}
    assert len(ranked) == 1, ranked
    ranked = ranked.pop()
    singles, mixes, finals = penalty_population(regu)
    assert ranked == singles + mixes, (ranked, singles, mixes)

    note = SELECTION_BASIS + (
        "On the penalty blocks that denominator is "
        + spell(ranked) + " --- " + spell(singles) + " single penalties and "
        + spell(mixes) + " settings of the constructed composite --- on both "
        "axes. The \\emph{screened} composite of "
        "Table~\\ref{tab:reg_winners} is outside it: that arm swept all "
        "three coefficients at once and ran after the screen, so it holds "
        "no place in either ordering and the finals number "
        + spell(finals) + " per schedule where these ranks are over "
        + spell(ranked) + ". A test rank here is therefore one place above "
        "the same arm's place among the finals whenever that arm scores "
        "below the screened composite. "
        "On the parallel schedule the rule the cross was built from is the "
        "one validation put first, and it is not the one that leads "
        "Table~\\ref{tab:agg_winners}. "
        "$\\ddagger$ marks an arm the stage shortlisted; on the penalty "
        "blocks the composite it marks is the composite that schedule "
        "selected, $m{=}0.5$ on both. "
        "$\\dagger$ marks the cyclic-tuned composite at $m{=}0.75$, the "
        "\\emph{test-leading} composite of Table~\\ref{tab:reg_winners}: it "
        "leads both test orderings and no validation ordering selected it. "
        "It is printed in \\emph{both} penalty blocks at whatever depth it "
        "sits --- validation rank %d on the cyclic schedule and %d on the "
        "parallel one --- because that is the claim, and a rank that is not "
        "on the page cannot be checked. A shortlist "
        "is the first three arms of the validation-score ordering that "
        "belong to three \\emph{different} methods --- a family's three "
        "slots are three ideas, not three settings of one, and the "
        "composite penalty competes for a slot of its own --- which is "
        "why a shortlist need not read $1,2,3$: on the cyclic penalties "
        "it reads $1,2,5$, the arms at 3 and 4 being further settings of "
        "a composite whose slot was already taken. The cross these two "
        "shortlists were built into is "
        "Table~\\ref{tab:selection_axis_cross}."
        % (lead_cyclic, lead_parallel))

    return block_env(
        "tab:selection_axis",
        "What the selecting saw, and what the tables report: the two "
        "selecting stages. Each stage's ranked arms at the top of the "
        "\\textbf{validation} ordering that chose them, beside the "
        "\\textbf{test} row that reports them. Five-fold means throughout; "
        "the two axes are two evaluations of one set of runs and must not be "
        "quoted against each other.",
        "lrrrrrrrr", SELECTION_HEAD, body,
        size="\\scriptsize", colsep="3pt", stretch="0.95", note=note,
        comment=SELECTION_SOURCE)


def t_selection_axis_cross(sel, combo):
    """A1b --- the cross, all nine crossings per schedule.

    EVERY CROSSING IS PRINTED AND FIVE OF THEM USED TO BE.  The paper argues
    about the arms at validation ranks 1, 2, 3 and 9 of this stage: the pair
    the cross crowned, the pair that leads the test ordering, the cyclic
    sibling, and the balanced arm that validation put last and Table 7 runs at
    every federation setting anyway.  A block cut at five printed the argument
    and hid one of the arms it is made of.
    """
    body = selection_body(sel, {"combo": combo}, SELECTION_STAGES[2:])

    # WHICH RULE CARRIED WHICH ARM, CHECKED RATHER THAN ASSERTED IN PROSE.
    # Two of the three carried arms were carried by rules that appear nowhere
    # in the records, so the note states them - and a note that states a rule
    # has to be a note the table can fail on.
    cross = [r for r in sel if r["stage"] == "combination"]
    parallel = [r for r in cross if r["schedule"] == "parallel"]
    balanced = pick(parallel, cell=BALANCED[0], schedule="parallel")
    assert float(balanced["val_preservation"]) == max(
        float(r["val_preservation"]) for r in parallel), \
        "the balanced arm is no longer the parallel crossing that gave up least"
    assert selection_rank_of(sel, CYCLIC_ARM[0], "cyclic", "combination") == 1, \
        "the cyclic sibling is no longer the strongest cyclic pair on validation"
    assert pick(cross, cell=CROWNED[0],
                schedule=CROWNED[1])["role"] == "crowned"
    # The note spells the balanced arm's composite, so the frozen record has
    # to be the arm the sentence describes -- the same pin tab:scaling and
    # tab:fairness put on the same sentence.
    assert BALANCED[0] == "eta_0p95_hybrid_seq_mix0p5", BALANCED

    note = SELECTION_BASIS + (
        "Every one of the nine crossings of each schedule is printed rather "
        "than a cut of them, because the arms this stage is argued about sit "
        "at ranks 1, 2, 3 and 9 of it. "
        "$\\dagger$ marks the pair the cross crowned, on validation, and "
        "therefore the arm every later stage carried; the cross cut no "
        "shortlist, it crowned once, and that arm leads the validation "
        "score ordering of all eighteen pairs. "
        "$\\S$ marks an arm carried past the selecting stages and run again "
        "at the four federation settings of Table~\\ref{tab:scaling}. Three "
        "arms carry it and only one of them was selected, so the rule that "
        "carried each of the other two is stated here and is checkable "
        "against the block above it: the crowned pair was carried because "
        "the cross crowned it; the \\emph{balanced} arm, the server step at "
        "$\\eta_s{=}0.95$ crossed with the cyclic-tuned composite at "
        "$m{=}0.5$, was "
        "carried as the crossing that gave up the least preservation of the "
        "nine on the parallel schedule, which is also why the score ordering "
        "puts it ninth; and the \\emph{cyclic sibling} was carried as the "
        "strongest cyclic pair on the crowning record's own validation "
        "ranking, which is rank one of its schedule --- the cross crowned "
        "nothing there, so it carries $\\S$ without a mark of selection. "
        "The two shortlists this cross was built from are "
        "Table~\\ref{tab:selection_axis}.")

    return block_env(
        "tab:selection_axis_cross",
        "What the selecting saw, and what the tables report: the cross. All "
        "nine crossings of each schedule in the \\textbf{validation} "
        "ordering that crowned one of them, beside the \\textbf{test} row "
        "that reports them. Five-fold means throughout; the two axes are two "
        "evaluations of one set of runs and must not be quoted against each "
        "other.",
        "lrrrrrrrr", SELECTION_HEAD, body,
        size="\\scriptsize", colsep="3pt", stretch="0.95", note=note,
        comment=SELECTION_SOURCE)


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
# T10 --- the eight permitted signals: tracker on the left, rule on the
# right, and the three references a rule column is worth nothing without
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


def t_signals(sigs, pst):
    """T10 --- every permitted signal twice: as a tracker, then as a rule.

    THE TWO SIDES OF THE TABLE ARE READ OVER DIFFERENT POPULATIONS, which is
    why the note says which is which.  The correlation columns are one rank
    correlation per run over the runs the signals pass covers; the four rule
    columns are means over the hundred-round stopping arms, a later and
    smaller population.  Both sit on the validation trace, the only basis a
    stopping round may be chosen on.  They are printed side by side because
    they disagree: the signal that orders the rounds of a run best is not the
    signal a server should stop on, and a section that reported the
    correlations alone would have picked the wrong one.

    THE LAST THREE ROWS ARE NOT SIGNALS.  A mean score means nothing without
    the horizon it is measured against, the rule that needs no proxy, and the
    oracle that prices what any stop could be worth; carrying the three here
    is what lets the subsection read the eight rows above them as worth
    something or not, rather than sending the reader to two other tables.
    """
    by = {r["signal"]: r for r in sigs}
    assert len(by) == len(sigs) == len(SIGNAL_ORDER), sorted(by)

    # The three references, all from the one row of plateau_stages.csv that
    # runs over every arm, so the horizon under the patience rule and the
    # horizon under a signal are the same 83 arms and the same mean.  Read
    # first because the fixed horizon is what every delta in the last column
    # is a delta from.
    ref = pick(pst, stage="all")
    fixed = num(ref, "fixed_mean", 2, 100)

    def against_fixed(score, stored, who):
        """The last column: the difference of the two numbers on the page.

        THE ONE COLUMN IN THIS FILE THAT IS NOT ROUNDED ONCE, AND IT IS
        DELIBERATE.  Everywhere else a difference is taken at full precision
        and rounded once, because a macro and a table have to agree; the
        contract at the top of this file says so and numbers.tex keeps the
        other half of it.  This column is different because both of its
        operands are printed in the same row of the same table: the score at
        the stopped round, and the fixed-horizon score three rows below it.
        Rounded once, three of the eight signal rows came out a hundredth away
        from the subtraction a reader does on the page -- 1.46 less 7.74
        printed as -6.27, 7.99 less 7.74 as +0.26, 8.12 less 7.74 as +0.39 --
        and a table whose own arithmetic does not close is a table a reader
        stops trusting at the first row they check.  So the delta is the
        difference of the printed values, the full-precision value it replaces
        is asserted to be within one hundredth of it, and the note says which
        it is.  No macro reads this column.
        """
        value = float(score) - float(fixed)
        assert abs(value - 100.0 * float(stored)) <= 0.01 + 1e-9, (
            "%s: the printed columns and the stored difference disagree by "
            "more than the rounding that separates them (%.4f vs %.4f)"
            % (who, value, 100.0 * float(stored)))
        CHECKED[0] += 1
        out = "%.2f" % value
        return ("$%s$" % out) if out.startswith("-") else ("$+%s$" % out)

    body = []
    for key in SIGNAL_ORDER:
        r = by[key]
        assert r["defined_on_all_arms"] == "1", (
            "%s is not defined on every stopping arm, so its rule columns are "
            "a mean over a different population than the rest of the column"
            % key)
        score = num(r, "mean_stopped_score", 2, 100)
        body.append(" & ".join([
            SIGNAL[key] + ("$^{\\dagger}$" if key in UNSTEERABLE else ""),
            verbatim(r, "observed_on"),
            num(r, "median_rho", 3),
            "%s\\%%" % num(r, "share_ge_0p9", 1, 100),
            verbatim(r, "best_delta"),
            count(r, "arms_fired"),
            score,
            against_fixed(score, r["mean_vs_fixed"], key)]) + " \\\\")

    dash = "---"
    rule = num(ref, "rule_mean", 2, 100)
    oracle = num(ref, "oracle_mean", 2, 100)
    gap = float(ref["oracle_mean"]) - float(ref["fixed_mean"])
    body += [
        "\\midrule",
        " & ".join(["\\emph{Fixed horizon}", "nothing", dash, dash, dash,
                    dash, fixed, dash]) + " \\\\",
        " & ".join(["\\emph{Patience rule} ($\\kappa=\\nPlateauK{}$)",
                    "cohort accuracy", dash, dash, dash,
                    count(ref, "fires"), rule,
                    against_fixed(rule, ref["gain"], "patience rule")]) + " \\\\",
        " & ".join(["\\emph{Oracle stop}", "the source population", dash,
                    dash, dash, dash, oracle,
                    against_fixed(oracle, gap, "oracle stop")]) + " \\\\"]

    # Six of the eight signals are recorded on the same runs; the two the
    # clients are asked for are missing from a few, and a median over a
    # different population is a thing the note has to say out loud.
    common = by["proxy_kl"]["n_runs"]
    assert sum(1 for r in sigs if r["n_runs"] == common) == 6, (
        "the run counts moved: %s" % sorted({r["n_runs"] for r in sigs}))

    note = ("Median $\\rho$ is the Spearman rank correlation of the signal's "
            "\\emph{drift} --- the signal signed so that forgetting makes it "
            "grow --- against the fall in source validation accuracy, one "
            "correlation per run; the column beside it is the share of those "
            "runs at $\\lvert\\rho\\rvert\\ge0.9$. Both are read over the "
            "runs of the signals pass that carry a source-validation series, "
            "which is \\nSignalCorrRuns{} of the \\nSignalRuns{} runs that "
            "pass covered --- itself the programme as it stood before its "
            "last two stages, and not the \\nProgrammeTasks{} tasks the whole "
            "programme ran. Six of the eight signals are carried by all "
            "\\nSignalCorrRuns{} of them; %s is carried by %s of them and %s "
            "by %s. The four columns on the right "
            "are means over the \\nStoppingArms{} hundred-round arms "
            "instead --- the budget the signal scores best at on the grid "
            "shared by all eight, the arms it stops before the horizon on, "
            "the mean score at the round it stops, and the difference from "
            "the fixed horizon. The one rule the study fixed for every "
            "arm before any of them ran is the proxy-set accuracy row of "
            "this table, at $\\delta=\\nOneRuleDelta{}$. $\\dagger$ The "
            "three "
            "parameter-space distances share no scale with that one grid: "
            "the $\\ell_2$ distance crosses every budget on it by the "
            "second round and the two Fisher distances reach almost none of "
            "them, so their rule columns report a units problem and not an "
            "information one. The last three rows are not signals: the "
            "horizon itself, the patience rule of "
            "Table~\\ref{tab:plateau}, which watches the cohort's own "
            "validation accuracy and needs no proxy at all, and the oracle "
            "stop, which reads the source population and is a bound on what "
            "stopping is worth rather than a method. "
            "$\\Delta$ vs.\\ fixed is the difference of the two numbers "
            "\\emph{as printed} --- the row's score less the fixed-horizon "
            "score three rows below it --- so every row of this column "
            "closes on the page. It is the one column in this paper that is "
            "not rounded once from the stored value; the two agree to within "
            "a hundredth of a point on every row, and the generator asserts "
            "it. No macro quotes this column."
            % (short("retention_known"),
               grouped(by["retention_known"], "n_runs"),
               short("agreement_with_global"),
               grouped(by["agreement_with_global"], "n_runs")))

    return block(
        "tab:signals",
        "The eight quantities a server may watch, each as a tracker of "
        "forgetting and as a stopping rule, with the fixed horizon, the "
        "patience rule and the oracle beside them. Every column is read from "
        "the per-round \\textbf{validation} trace, the only basis on which a "
        "stopping round may be chosen. Correlation is not deployability: of "
        "the five signals the shared budget grid can steer, the best tracker "
        "is the only one whose rule scores below the fixed horizon, and the "
        "best rule is the seventh tracker of the eight.",
        "llrrrrrr",
        "Signal & Reads & Median $\\rho$ & $\\lvert\\rho\\rvert\\ge0.9$ & "
        "Budget $\\delta$ & Fires & Score & $\\Delta$ vs.\\ fixed \\\\\n"
        " & & (per run) & (share of runs) & (best) & "
        "(of \\nStoppingArms{}) & (pts) & (pts) \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="sources: data/signals_summary_extract.csv, one row per\n"
                "signal in the order Section 7 introduces them, cols\n"
                "observed_on, median_rho, share_ge_0p9, best_delta,\n"
                "arms_fired, mean_stopped_score, mean_vs_fixed.  Col n_runs\n"
                "is read for the two run counts the note states and is not a\n"
                "column of this table;\n"
                "data/plateau_stages.csv, row all, for the three reference\n"
                "rows (cols fixed_mean, fires, rule_mean, gain, oracle_mean).\n"
                "derived: the oracle row's delta --- col oracle_mean minus\n"
                "col fixed_mean of that same row, differenced before it is\n"
                "rounded")


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


#: Which shipped-model accuracy each stage row of tab:plateau is read against.
#: The table stacks stages that ran on four different cohorts, and a note that
#: names two of them leaves a reader to assume the other rows share one of the
#: two.  The partition is the one report_tables.SETTING_COHORT makes from the
#: run-folder stems --- the search stages and the carry setting on the
#: ten-writer cohort, then five clients, twenty and the extremes --- written
#: out here so the note can be checked against the rows it describes.
STAGE_BASELINE = {
    "aggfull": "nGZeroCohortAcc",
    "regfull": "nGZeroCohortAcc",
    "combo": "nGZeroCohortAcc",
    "c10d10": "nGZeroCohortAcc",
    "five": "nGZeroCohortAccFive",
    "c20d10": "nGZeroCohortAccTwenty",
    "c20d20": "nGZeroCohortAccTwenty",
    "extreme": "nGZeroCohortAccExtreme",
}


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


def t_plateau(pst, prot):
    """T6 --- the plateau rule, stage by stage: what stopping recovers.

    THE GAIN COLUMN IS A FIT, AND THE NOTE NOW SAYS SO.  The rule's cell is a
    patience and a margin, and the published cell is the one the arms of this
    table chose; read a stage on its own and it is that stage's own arms doing
    the choosing.  So the column prices the rule where it was fitted, which is
    not what a column headed "gain" is taken to mean, and is exactly what the
    holdout table exists to correct.  The two numbers the note quotes are read
    out of the two views rather than written down, and the assertion below
    holds the in-sample one to the row this table prints.
    """
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

    # The holdout tool selects on one set of arms and reports on another; the
    # protocol that selects on the regularisation finals themselves returns,
    # by construction, this table's own gain for that stage -- which is what
    # makes "in-sample" a statement about this column rather than a hedge.
    # Read and checked, so the sentence cannot outlive the agreement.
    fitted = pick(prot, protocol="stage", selection_set="regfull arms")
    assert abs(float(fitted["sel_gain"]) - float(by["regfull"]["gain"])) < 5e-9, (
        "tab:plateau's regularisation gain is no longer the in-sample fit "
        "plateau_holdout_protocols.csv reports for the same arms")
    insample = num(by["regfull"], "gain", 2, 100, signed=True)
    held = num(pick(prot, protocol="stage-loo", heldout_set="regfull arms"),
               "held_gain", 2, 100, signed=True)

    note = ("Mean score in points over each stage's arms, on the per-round "
            "\\textbf{validation} trace, the only basis on which a stopping "
            "round may be chosen. The two level columns are means over "
            "stages whose settings are scored against four different "
            "shipped-model baselines --- \\nGZeroCohortAcc{} on the "
            "ten-writer cohort the search ran on and carried its winners "
            "onto, \\nGZeroCohortAccFive{} on the five-client stage, "
            "\\nGZeroCohortAccTwenty{} on both twenty-client stages and "
            "\\nGZeroCohortAccExtreme{} on the two extreme arrangements --- "
            "so a level is not comparable down the column; the gain column "
            "is a within-run difference, one arm stopped against the same "
            "arm run out, and is comparable down the column. "
            "The rule keeps the checkpoint of the best "
            "cohort round and stops training after "
            "$\\kappa=\\nPlateauK{}$ rounds "
            "without improvement; it costs anything on exactly "
            "\\nPlateauHurtArms{} arm of \\nPlateauArms{} "
            "($-\\nPlateauWorstLoss{}$ points), and an arm on which it never "
            "fires runs the full budget unchanged. "
            "The gain column is an \\emph{in-sample} fit: the rule's cell "
            "$(\\kappa,\\varepsilon)$ --- the patience and the margin an "
            "improvement has to clear --- is chosen on the stage's own arms, "
            "so each row prices the rule where it was fitted. "
            "Table~\\ref{tab:kholdout} holds that choice out, and the "
            "regularisation finals are where holding it out costs most: they "
            "read " + insample + " points here and " + held + " there, with "
            "the cell chosen on the other arms and applied to these "
            "unchanged.")
    for macro in sorted(set(STAGE_BASELINE.values())):
        assert "\\%s{}" % macro in note, macro
    return block(
        "tab:plateau",
        "The plateau rule, stage by stage: the fixed hundred-round budget "
        "against stopping on the cohort's own validation accuracy.",
        "lrrrr",
        "Stage & Arms & Fixed budget & With the rule & Gain \\\\\n"
        " & & (pts) & (pts) & (pts) \\\\",
        body, colsep="5pt", note=note,
        comment="sources: data/plateau_stages.csv, primary setting rows\n"
                "(patience 20, margin 0, checkpoint_best), stage means;\n"
                "data/plateau_holdout_protocols.csv for the two figures the\n"
                "note quotes --- the stage row selecting on the regularisation\n"
                "finals, col sel_gain, asserted equal to this table's own gain\n"
                "for that stage and printed from it, and the stage-loo row\n"
                "holding the same arms out, col held_gain")


#: How the holdout tool's set names read on the page.  The tool names its sets
#: after the run-folder stems it cut them from, and a stem is not a thing this
#: paper prints; STAGE_LABEL already says what each stage is called.
HOLDOUT_STAGE = {stem: name.replace("\\textbf{", "").replace("}", "")
                 for stem, name in STAGE_LABEL}

#: The protocol families of plateau_holdout_protocols.csv, the order the table
#: states them in, whether each row is printed on its own or the family is
#: summarised as one row with a mean and a spread, and the family's title.
#: EVERY FAMILY IS HERE, which is the point: the table used to print six rows
#: drawn from four of the six families, covering nineteen of the twenty-eight
#: protocols, and a reader who counted the rows had nowhere to look for the
#: other nine.
HOLDOUT_FAMILIES = (
    ("fold", "each", "Designated fold splits"),
    ("fold-loo", "summary", "Leave one fold out"),
    ("stage", "each", "Designated stage splits"),
    ("stage-loo", "each", "Leave one stage out"),
    ("random-half", "summary", "Random halves of the arms"),
    ("in-sample", "each", "In sample: the published setting, not a holdout"),
)


def _sd(xs):
    """Sample standard deviation, written out: a handful of values needs no
    library, and importing one into a standard-library-only generator to take
    a square root is how a dependency arrives."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _folds(digits):
    """``"123" -> "folds 1--3"``, and a comma list when they are not a run."""
    numbers = [int(d) for d in digits]
    if len(numbers) == 1:
        return "fold %d" % numbers[0]
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        return "folds %d--%d" % (numbers[0], numbers[-1])
    return "folds " + ", ".join("%d" % n for n in numbers)


def _holdout_name(text):
    """One selection- or held-out-set description, in the paper's own words.

    A REFUSAL RATHER THAN A PASS-THROUGH.  The alternative is to print whatever
    the tool wrote, which would put a run-folder stem on the page the first
    time a stage is added, silently.
    """
    if text == "same (in sample)":
        return "---"
    match = re.fullmatch(r"all arms, (all folds|folds? (\d+))", text)
    if match:
        return ("All arms, all folds" if match.group(2) is None
                else "All arms, " + _folds(match.group(2)))
    for stem, name in sorted(HOLDOUT_STAGE.items(), key=lambda kv: -len(kv[0])):
        if text == "%s arms" % stem:
            return name
        if text in ("all non-%s arms" % stem, "all but %s" % stem):
            lower = name[0].lower() + name[1:]
            # STAGE_LABEL names one stage "The extreme cases"; "all but the
            # the extreme cases" is what happens when a prefix is pasted on
            # without looking at what it is being pasted onto.
            return "All arms but " + (lower if lower.startswith("the ")
                                      else "the " + lower)
    raise AssertionError(
        "plateau_holdout_protocols.csv names a set this table cannot put into "
        "words: %r. Naming it here is what keeps a run-folder stem off the "
        "page." % text)


def t_kholdout(prot):
    """B4 --- the patience, chosen where it is not measured.

    THE COUNT HAD TO BECOME RECONCILABLE.  The tool runs twenty-eight
    protocols and the table printed six rows; the six expanded to nineteen of
    the twenty-eight, because two families were folded into one row each and
    the eight leave-one-stage-out protocols were not on the page at all.  A
    reader who took the caption at its word and counted got nineteen and had
    nowhere to look for the rest.  Every family is now on the table, each
    family header says how many protocols it stands for, and the assertion
    below refuses a table whose families do not add up to the file's own row
    count.
    """
    families = {}
    for r in prot:
        families.setdefault(r["protocol"], []).append(r)
    unplaced = sorted(set(families) - {name for name, _, _ in HOLDOUT_FAMILIES})
    assert not unplaced, (
        "plateau_holdout_protocols.csv carries protocol families this table "
        "does not place, so the count on the page cannot be reconciled with "
        "the file: %s" % unplaced)

    body, covered = [], 0
    for name, how, title in HOLDOUT_FAMILIES:
        here = families.get(name) or []
        assert here, name
        covered += len(here)
        if body:
            body.append("\\midrule")
        body.append("\\multicolumn{5}{@{}l}{\\emph{%s (%d of "
                    "\\nKholdoutProtocols{} protocols)}} \\\\"
                    % (title, len(here)))
        if how == "summary":
            gains = [float(r["held_gain"]) for r in here]
            chosen = {int(float(r["k"])) for r in here}
            held = {int(float(r["held_arms"])) for r in here}
            assert len(chosen) == 1 and len(held) == 1, (name, chosen, held)
            body.append(" & ".join([
                "\\quad each split, mean $\\pm$ sd",
                "the other half of the same cut" if name == "random-half"
                else "the fold that was held",
                "%d" % held.pop(),
                "%d on %d/%d" % (chosen.pop(), len(here), len(here)),
                "%s $\\pm$ %s" % (derived(sum(gains) / len(gains), 2, 100,
                                          signed=True),
                                  derived(_sd(gains), 2, 100))]) + " \\\\")
            continue
        for r in here:
            body.append(" & ".join([
                "\\quad " + _holdout_name(r["selection_set"]),
                _holdout_name(r["heldout_set"]),
                count(r, "held_arms"), count(r, "k"),
                num(r, "held_gain", 2, 100, signed=True)]) + " \\\\")

    assert covered == len(prot), (covered, len(prot))

    note = ("Every protocol the tool runs is on this table: "
            "\\nKholdoutProtocols{} of them in six families, with the two "
            "families whose splits are interchangeable summarised as one row "
            "apiece carrying the mean and the sample standard deviation of "
            "their held-out gains. The patience $\\kappa=\\nPlateauK{}$ is "
            "chosen "
            "by every one of the \\nKholdoutProtocols{}, at every margin and "
            "on every split. The designated stage split and the "
            "leave-one-stage-out of the same stage are one cut seen from its "
            "two sides, so the regularisation finals appear in both families "
            "and agree. The arm population is held fixed across every split: "
            "a fold protocol changes the traces an arm is read from, never "
            "which arms are in the table. The smallest held-out gain any "
            "protocol returns is \\nKholdoutMinHeldGain{} points and the "
            "largest is \\nKholdoutMaxHeldGain{}; the two three-arm settings "
            "return nothing because the rule never fires on them.")

    return block(
        "tab:kholdout",
        "Held-out selection of the patience $\\kappa$. The rule's cell is "
        "chosen on "
        "the selection set and applied unchanged to the held-out one; gains "
        "are the mean score over the held-out arms, plateau rule minus fixed "
        "horizon, in points on the \\textbf{validation} trace.",
        "llrrr",
        "Selection set & Held-out set & Arms & $\\kappa$ chosen & "
        "Held-out gain (pts) \\\\\n"
        " & & (held) & & \\\\",
        body, size="\\scriptsize", colsep="4pt", note=note,
        comment="source: data/plateau_holdout_protocols.csv, every row,\n"
                "grouped by col protocol in the order the tool runs them;\n"
                "the two summarised families print the mean and sample sd of\n"
                "col held_gain over their splits, which is derived, and every\n"
                "other row prints cols selection_set, heldout_set,\n"
                "held_arms, k and held_gain")


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
    prot = rows("plateau_holdout_protocols.csv")
    cohort = rows("cohort_table.csv")
    sigs = rows("signals_summary_extract.csv")
    sel = rows("selection_axis.csv")
    fair = {key: rows("fairness_%s.csv" % key)
            for key, _, _ in FAIRNESS_BLOCKS}
    sizes = {k: rows("sizes_%s.csv" % k) for k, _, _ in SETTING}

    # What Tables 2 and 3 print is what Table 4's delta column may point at.
    # Both tables are emitted from these two sets, so the assertion inside
    # t_combos cannot drift away from what is actually on the page.
    agg_cells = {r["cell"] for r in agg}
    reg_rows = reg_printed(regu, combo, agg_cells)
    printed = ({(r["cell"], r["family"]) for r in agg}
               | {(r["cell"], r["family"]) for r, _ in reg_rows})

    emitted = [
        ("references.tex", t_references(refs), len(refs)),
        ("agg_winners.tex", t_agg(agg), len(agg)),
        ("reg_winners.tex", t_reg(reg_rows, agg, regu), len(reg_rows) + 2),
        ("blends.tex", t_blends(regu, blend),
         len([r for r in regu if r["cell"].startswith("hybrid")])),
        ("combos.tex", t_combos(combo, agg, regu, printed), len(combo)),
        ("scaling.tex", t_scaling(sizes, cost_s),
         sum(len(v) for v in sizes.values())),
        ("fairness.tex", t_fairness(fair, sizes, combo, cost_s),
         4 + sum(len(sizes[k]) for k, _, _ in FAIRNESS_BLOCKS if k != "combo")),
        ("cost.tex", t_cost(cost_s), len(cost_s)),
        ("extreme.tex", t_extreme(extr, stopx), len(extr)),
        ("signals.tex", t_signals(sigs, pst), len(sigs)),
        ("stopping.tex", t_stopping(stopa), len(stopa)),
        ("plateau.tex", t_plateau(pst, prot), len(pst) - 1),
        ("kholdout.tex", t_kholdout(prot), len(prot)),
        ("cohort.tex", t_cohort(cohort), len(cohort)),
        ("selection_axis.tex", t_selection_axis(sel, agg, regu, combo),
         len([r for r in sel if r["stage"] != "combination"])),
        ("selection_axis_cross.tex", t_selection_axis_cross(sel, combo),
         len([r for r in sel if r["stage"] == "combination"])),
    ]
    ids = identifiers(refs, agg, regu, blend, combo, extr, stopx, sigs,
                      sel, *fair.values(), *sizes.values())
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
