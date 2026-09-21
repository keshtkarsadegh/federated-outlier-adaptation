#!/usr/bin/env python3
"""Regenerate the generated block of numbers.tex from data/*.csv.

Usage:  python3 make_numbers.py [--check]

Contract
--------
* numbers.tex carries two marker lines, ``% BEGIN GENERATED`` and
  ``% END GENERATED``.  This script rewrites *only* the text between them.
  Everything above BEGIN (the \\pub* block and the hand-maintained block) is
  copied through byte for byte.
* Every macro name found between the markers must have an entry in the
  registry built by :func:`build`.  A name without one is written back as
  ``\\TBD{unmapped}`` and reported on stdout, and the script exits non-zero.
* Every emitted definition carries a trailing comment naming the CSV file,
  the row and the column the value came from, so any number in the paper can
  be traced to a file without leaving the manuscript.
* A quantity is rounded ONCE, from the full-precision value, and a quantity
  that is a difference is differenced before it is rounded.  make_paper_tables
  rounds the same way, so no macro can disagree with the table beside it ---
  but a reader who subtracts two PRINTED numbers can land a digit away from
  both, and has.  See ``ROUND_ONCE``.
* Running the script twice produces a byte-identical file.

Where it reads and where it writes
----------------------------------
The views are read from ``data/`` beside this script in the manuscript
checkout, and from the study's shipped metadata core in the source
repository; ``FOA_PAPER_DATA`` overrides both.  ``numbers.tex`` is
rewritten in the manuscript directory, which is where this script sits in
the manuscript checkout and which ``FOA_PAPER_OUT`` names anywhere else.
Neither default writes into a source checkout.  See ``_data_dirs`` and
``out_dir``.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


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

BEGIN = "% BEGIN GENERATED"
END = "% END GENERATED"


# --------------------------------------------------------------------------
# csv helpers
# --------------------------------------------------------------------------

def rows(name):
    """All rows of the view <name> as dicts, in file order."""
    with open(locate(name), newline="") as fh:
        return list(csv.DictReader(fh))


def pick(table, **eq):
    """The single row of ``table`` matching every key=value pair."""
    hit = [r for r in table if all(r[k] == v for k, v in eq.items())]
    if len(hit) != 1:
        raise KeyError("expected 1 row for %r, found %d" % (eq, len(hit)))
    return hit[0]


def top(table, **eq):
    """Highest-``score`` row matching the filters (the crowning rule)."""
    hit = [r for r in table if all(r[k] == v for k, v in eq.items())]
    if not hit:
        raise KeyError("no row for %r" % (eq,))
    return max(hit, key=lambda r: float(r["score"]))


def f(row, col):
    return float(row[col])


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def acc(x, nd=4):
    """An accuracy on the 0-1 scale."""
    return "%.*f" % (nd, x)


def pts(x, nd=2):
    """A quantity already expressed in points."""
    return "%.*f" % (nd, x)


def dpts(x, nd=2):
    """A 0-1 difference rendered in points."""
    return "%.*f" % (nd, 100.0 * x)


def signed(x, nd=2):
    """A 0-1 difference in points, negatives set as a real minus sign."""
    v = "%.*f" % (nd, 100.0 * x)
    return "$%s$" % v if v.startswith("-") else v


def power(x):
    """A protocol constant the field writes as a power of ten.

    The learning rate and the weight decay are quoted as $10^{-3}$ and
    $10^{-4}$ everywhere the literature quotes them, and the view carries them
    as 0.001 and 0.0001 because that is what the runner recorded.  A constant
    that is not a clean power is written out rather than forced into one: a
    wrong exponent is a worse failure than an ugly number.
    """
    e = math.log10(float(x))
    if abs(e - round(e)) > 1e-12:
        return ("%g" % float(x)).replace("-", "$-$")
    return "$10^{%d}$" % int(round(e))


def word(n):
    n = int(round(n))
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def group(n):
    """1663370 -> ``1\\,663\\,370`` (LaTeX thin-space thousands)."""
    s = "%d" % int(round(n))
    out = []
    while len(s) > 3:
        out.insert(0, s[-3:])
        s = s[:-3]
    out.insert(0, s)
    return "\\,".join(out)


def dial(row, col):
    """A grid coordinate, spelled the way the grid spells it: 0.1, 0.05, 2.

    Not a measurement and not rounded: a coefficient the search was offered
    reaches the page as the number the record carries, so a reader can find
    the cell the sentence describes by the digits it prints.
    """
    return "%g" % float(row[col])


def code(cell):
    return "\\code{%s}" % cell.replace("_", "\\_")


# --------------------------------------------------------------------------
# cell -> prose.  Only the hyperparameter label is looked up here; every
# NUMBER in the manuscript comes from a CSV.  Unknown cells fall back to the
# verbatim cell name, so a changed winner can never silently acquire the old
# winner's description.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# The frozen record.  The study crowned its arms on validation and every later
# stage reads that record; the CSVs in data/ are the TEST tables, whose score
# ordering is close to but not identical with the validation one.  The three
# arms below are therefore named, not re-derived from a "best row" of a test
# table, so that a later data refresh can never silently re-crown anything.
# --------------------------------------------------------------------------

#: What every rounded quantity in this file does with a difference, written
#: down because four of these macros have been recomputed off the printed page
#: and reported as wrong.  A difference is taken at FULL PRECISION and rounded
#: once, here and in make_paper_tables.py alike; it is never the difference of
#: two numbers that were rounded first.  The two orders disagree in the last
#: digit often enough to matter --- the carry margin is 2.77 from the stored
#: scores and 2.78 from the printed ones --- and the stored one is the answer.
ROUND_ONCE = ("rounded once from the full-precision value; subtracting the"
              " PRINTED values instead can move the last digit")

CROWNED = ("anchor_h2_ntd_b0p01_t0p5", "parallel")     # the crowned pair
BALANCED = ("eta_0p95_hybrid_seq_mix0p5", "parallel")  # the balanced arm
CYCLIC_ARM = ("seq_fedavg_ntd_b0p01_t2", "cyclic")     # the cyclic arm
AGG_CROWNED_CONC = "anchor_h2"                         # the crowned parallel rule

FROZEN = "the frozen crowning record (validation basis), not re-derived here"


def suffix(cell, key):
    """``suffix('logit_l2_lam0p003', 'lam') -> 0.003`` (``p`` is the point)."""
    m = re.search(re.escape(key) + r"([0-9]+p[0-9]+|[0-9]+)", cell)
    if not m:
        raise KeyError("no %r coefficient in %r" % (key, cell))
    return float(m.group(1).replace("p", "."))


PROSE = {
    # server rules --- the literature's own name for the rule, with its
    # setting in the method's own symbol, exactly as paper_names.py prints it
    # in the tables.  No run-record identifier may reach a page.
    "eta_0p95": "the server step at $\\eta_s = 0.95$",
    "anchor_h2": "the server anchor at a half-life of $h = 2R$",
    "weight_q0": "uniform client weighting ($q = 0$)",
    "median": "the coordinate-wise median",
    "trimmed_t3": "the trimmed mean at $t = 3$",
    "seq_fedavg": "cyclic FedAvg",
    "seq_delta_capped": "the capped cyclic rule",
    "seq_delta_scaled": "the scaled cyclic rule",
    "seq_mix_r0p3": "cyclic mixing at $r = 0.3$",
    "control_fedavg": "plain FedAvg",
    # penalties
    "logit_l2_lam0p001": "logit matching at $\\lambda = 10^{-3}$",
    "logit_l2_lam0p003": "logit matching at $\\lambda = 3{\\times}10^{-3}$",
    "feature_l2_lam0p001": "feature matching at $\\lambda = 10^{-3}$",
    "feature_l2_lam0p01": "feature matching at $\\lambda = 10^{-2}$",
    "ntd_b0p01_t0p5": "FedNTD at $\\beta = 0.01$, $\\tau = 0.5$",
    "ntd_b0p01_t2": "FedNTD at $\\beta = 0.01$, $\\tau = 2$",
    "kd_T0p25_a0p9": "distillation at $T = 0.25$, $\\alpha = 0.9$",
    "kd_T2_a0p99": "distillation at $T = 2$, $\\alpha = 0.99$",
    "fisher_lam0p1": "EWC at $\\lambda = 0.1$",
    "fisher_lam8": "EWC at $\\lambda = 8$",
    "fisher_scaled_lam0p1": "loss-capped EWC at $\\lambda = 0.1$",
    "fisher_scaled_lam50": "loss-capped EWC at $\\lambda = 50$",
    "param_l2_mu0": "FedProx at $\\mu = 0$",
    "param_l2_mu0p0001": "FedProx at $\\mu = 10^{-4}$",
    "hybrid_mix0p25": "the KD+EWC composite at $m = 0.25$",
    "hybrid_mix0p5": "the KD+EWC composite at $m = 0.5$",
    "hybrid_mix0p75": "the KD+EWC composite at $m = 0.75$",
    "hybrid_seq_mix0p25": "the cyclic-tuned KD+EWC composite at $m = 0.25$",
    "hybrid_seq_mix0p5": "the cyclic-tuned KD+EWC composite at $m = 0.5$",
    "hybrid_seq_mix0p75": "the cyclic-tuned KD+EWC composite at $m = 0.75$",
    "blend_lam0p1_T0p5_mix0p25":
        "the screened KD+EWC blend at $\\lambda = 0.1$, $T = 0.5$, $m = 0.25$",
    "blend_lam0p1_T0p25_mix0p5":
        "the screened KD+EWC blend at $\\lambda = 0.1$, $T = 0.25$, $m = 0.5$",
}

# trimmed_t<k> keeps k updates out of n; the grid it was drawn from is
# {0.1, 0.2, 0.3, 0.4, 0.5} and floor(fraction * n) = k identifies the label.
TRIM_FRACTION = {"trimmed_t1": 0.2, "trimmed_t2": 0.3, "trimmed_t3": 0.4}


def prose(cell, agg_cells=()):
    """Human phrase for a cell; combination cells split on their rule prefix."""
    if cell in PROSE:
        return PROSE[cell]
    for a in sorted(agg_cells, key=len, reverse=True):
        if cell.startswith(a + "_"):
            return "%s with %s" % (prose(a), prose(cell[len(a) + 1:]))
    return code(cell)


def split_combo(cell, agg_cells):
    """``anchor_h2_ntd_b0p01_t0p5`` -> ``('anchor_h2', 'ntd_b0p01_t0p5')``."""
    for a in sorted(agg_cells, key=len, reverse=True):
        if cell.startswith(a + "_"):
            return a, cell[len(a) + 1:]
    raise KeyError("no server rule prefix in %r" % cell)


def cap(s):
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

def build():
    """Return {macro name: (value, provenance comment)}."""
    reg = {}

    def put(name, value, src):
        reg[name] = (value, src)

    refs = rows("references.csv")
    agg = rows("agg-winners.csv")
    regu = rows("reg-winners.csv")
    combo = rows("combos.csv")
    combo_folds = rows("combos_folds.csv")
    extr = rows("extremes.csv")
    cost_a = rows("cost_arms.csv")
    cost_s = rows("cost_stages.csv")
    cohort = rows("cohort_table.csv")
    recipe = {r["setting"]: r for r in rows("recipe.csv")}
    cohort_comp = rows("cohort_composition.csv")
    decouple = rows("decouple_example.csv")
    fair = rows("fairness_c10d10.csv")
    fairc = rows("fairness_c10d10_clients.csv")
    fair_combo = rows("fairness_combo.csv")
    fair20 = {"c20d10": rows("fairness_c20d10.csv"),
              "c20d20": rows("fairness_c20d20.csv")}
    comp = rows("composition.csv")
    blend = rows("blends.csv")
    sel = rows("selection_axis.csv")
    stopx = rows("stopping_extreme.csv")
    stopa = rows("stopping_all.csv")
    sigs = rows("signals_summary_extract.csv")
    sigx = {r["key"]: r["value"] for r in rows("signals_extras_extract.csv")}
    sizes = {"five": rows("sizes_five.csv"),
             "c10d10": rows("sizes_c10d10.csv"),
             "c20d10": rows("sizes_c20d10.csv"),
             "c20d20": rows("sizes_c20d20.csv")}
    fair5 = rows("fairness_five.csv")
    prot = rows("plateau_holdout_protocols.csv")
    prefilter = {r["cohort"]: r for r in rows("prefilter_coverage.csv")}

    agg_cells = {r["cell"] for r in agg}

    # ---- the two shipped-model accuracies -------------------------------
    # score = (adaptation - A0) - (P0 - preservation); the CSVs carry
    # gained = adaptation - A0 and spent = P0 - preservation, so both
    # constants are recoverable from any row and must agree across rows.
    a0s = [f(r, "adaptation") - f(r, "gained") for r in refs]
    p0s = [f(r, "preservation") + f(r, "spent") for r in refs]
    assert max(a0s) - min(a0s) < 1e-9 and max(p0s) - min(p0s) < 1e-9
    A0, P0 = a0s[0], p0s[0]

    SRC_A0 = "references.csv, every row: adaptation - gained"
    SRC_P0 = "references.csv, every row: preservation + spent"

    # ---- Section 6.1, the starting line ---------------------------------
    put("nGZeroSrcAcc", acc(P0), SRC_P0)
    put("nGZeroSrcPct", "%.1f\\,\\%%" % (100 * P0), SRC_P0 + ", as a percentage")
    put("nGZeroCohortPct", "%.1f\\,\\%%" % (100 * A0), SRC_A0 + ", as a percentage")
    put("nGZeroCohortAcc", acc(A0), SRC_A0)

    # ---- the same model, on the three other cohorts ---------------------
    # A0 IS NOT ONE NUMBER.  Every setting is scored against the shipped
    # model's accuracy on its OWN cohort, so the blocks of tab:scaling and
    # the extreme arrangements each carry a baseline of their own; a score
    # read against the ten-writer figure would be the distance between two
    # populations reported as something an arm did.  Each is recovered the
    # way A0 is - a view's adaptation less its gained - from a view whose
    # rows were scored on that cohort's book, and asserted constant over
    # those rows, so a view that mixed two books cannot pass through here
    # as one baseline.  The twenty-writer cohort is one cohort at two
    # dropout rates and both its views are read, for the same reason.
    def cohort_a0(macro, views):
        seen = [f(r, "adaptation") - f(r, "gained")
                for _, table in views for r in table]
        assert seen and max(seen) - min(seen) < 1e-9, (macro, seen)
        put(macro, acc(seen[0]),
            "%s, every row: adaptation - gained"
            % ", ".join(name for name, _ in views))

    cohort_a0("nGZeroCohortAccFive", [("sizes_five.csv", sizes["five"])])
    cohort_a0("nGZeroCohortAccTwenty",
              [("sizes_c20d10.csv", sizes["c20d10"]),
               ("sizes_c20d20.csv", sizes["c20d20"])])
    cohort_a0("nGZeroCohortAccExtreme", [("extremes.csv", extr)])
    put("nStartingGap", dpts(P0 - A0, 1), "derived: P0 - A0 (%s; %s)" % (SRC_P0, SRC_A0))

    wsa = rows("weight_sensitivity_agg.csv")
    wsr = rows("weight_sensitivity_reg.csv")
    words = {0:"none",1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",
             7:"seven",8:"eight",9:"nine",10:"ten"}
    for tag, tbl in (("Agg", wsa), ("Reg", wsr)):
        put("nSens%sMethods" % tag, words.get(len(tbl), str(len(tbl))),
            "weight_sensitivity_%s.csv, row count" % tag.lower())
        for w, wn in ((2, "Two"), (3, "Three"), (5, "Five")):
            n = sum(1 for r in tbl if r["w=%d" % w] != r["w=1"])
            put("nSens%sW%s" % (tag, wn), words.get(n, str(n)),
                "weight_sensitivity_%s.csv, rows where col w=%d differs from col w=1"
                % (tag.lower(), w))

    ctlp2 = pick(agg, cell="control_fedavg", family="parallel")
    best_par = max((f(r, "score") for r in agg if r["family"] == "parallel"))
    put("nAggTestBestGain", pts(100 * (best_par - f(ctlp2, "score"))),
        "agg-winners.csv, parallel block: max score - control_fedavg score, in points")
    put("nReplicateGap", pts(100 * (f(pick(agg, cell="control_fedavg", family="cyclic"), "score")
                                    - f(pick(regu, cell="param_l2_mu0", family="cyclic"), "score"))),
        "agg-winners.csv control_fedavg/cyclic score minus reg-winners.csv"
        " param_l2_mu0/cyclic score, in points: same nominal configuration run twice")
    put("nIsoScore", pts(100 * f(pick(refs, cell="isolated (from g-0)"), "score")),
        "references.csv, row 'isolated (from g-0)', col score, in points; "
        + ROUND_ONCE)
    put("nFedBestScore", pts(100 * max(f(r, "score")
                                       for view in (agg, regu, combo) for r in view)),
        "max col score over agg-winners.csv, reg-winners.csv, combos.csv, in points")
    esr = rows("extreme_stop_rounds.csv")
    dual = pick(esr, arm="dual")
    doub = pick(esr, arm="double")
    put("nDualOracleScore", pts(100 * f(dual, "oracle_score")),
        "extreme_stop_rounds.csv, row dual (the merged pair), col oracle_score, in points")
    put("nDoubleOracleScore", pts(100 * f(doub, "oracle_score")),
        "extreme_stop_rounds.csv, row double (two writers), col oracle_score, in points")
    put("nDualOracleAdapt", acc(f(dual, "oracle_cohort")),
        "extreme_stop_rounds.csv, row dual, col oracle_cohort")
    put("nDualOraclePres", acc(f(dual, "oracle_src")),
        "extreme_stop_rounds.csv, row dual, col oracle_src")
    put("nDoubleOracleAdapt", acc(f(doub, "oracle_cohort")),
        "extreme_stop_rounds.csv, row double, col oracle_cohort")
    put("nDoubleOraclePres", acc(f(doub, "oracle_src")),
        "extreme_stop_rounds.csv, row double, col oracle_src")
    put("nDoubleSatRound", "%d" % int(f(doub, "oracle_round")),
        "extreme_stop_rounds.csv, row double, col oracle_round")
    put("nDualFinalScore", pts(100 * f(dual, "final_score")),
        "extreme_stop_rounds.csv, row dual, col final_score (round-100 trace), in points")
    put("nDoubleFinalScore", pts(100 * f(doub, "final_score")),
        "extreme_stop_rounds.csv, row double, col final_score (round-100 trace), in points")

    # The accuracy pair each of the four extreme scores above is made of, so a
    # reader can close the score arithmetic on the page rather than take the
    # scores on trust.  The oracle pair is in extreme_stop_rounds.csv beside
    # its score; the round-100 pair is NOT - that view carries final_score and
    # not the two columns behind it - and stopping_extreme.csv carries both,
    # for the same arms on the same per-round validation trace.  That the two
    # views agree on final_score to the bit is asserted rather than assumed,
    # because a pair read off a different horizon would still look like one.
    for arm, tag in (("dual", "Dual"), ("double", "Double")):
        trace = pick(stopx, cell=arm)
        assert abs(f(trace, "final_score")
                   - f(pick(esr, arm=arm), "final_score")) < 1e-9, arm
        put("n%sFinalAdapt" % tag, acc(f(trace, "final_adaptation")),
            "stopping_extreme.csv, row %s, col final_adaptation"
            " (round-100 trace)" % arm)
        put("n%sFinalPres" % tag, acc(f(trace, "final_preservation")),
            "stopping_extreme.csv, row %s, col final_preservation"
            " (round-100 trace)" % arm)
    put("nLTwoRuleScore", pts(100 * float(next(r for r in sigs
                                               if r["signal"] == "dist_l2_to_global")["mean_stopped_score"])),
        "signals_summary_extract.csv, row dist_l2_to_global, col mean_stopped_score, in points")
    put("nBlendClearMin", pts(min(float(r["mean"]) for r in blend
                                  if r["all_positive"] == "1")),
        "blends.csv, rows with all_positive=1, min col mean, in points")

    pst = rows("plateau_stages.csv")
    pex = rows("plateau_extremes.csv")
    parm = rows("plateau_arms.csv")
    pall = pick(pst, stage="all")
    put("nPlateauK", "%d" % int(float(pall["patience"])),
        "plateau_stages.csv, row all, col patience")
    put("nPlateauAllMean", dpts(f(pall, "rule_mean")),
        "plateau_stages.csv, row all, col rule_mean, in points")
    put("nPlateauAllGain", dpts(f(pall, "gain")),
        "plateau_stages.csv, row all, col gain, in points")
    put("nPlateauOracleMean", dpts(f(pall, "oracle_mean")),
        "plateau_stages.csv, row all, col oracle_mean, in points")
    put("nPlateauFixedMean", dpts(f(pall, "fixed_mean")),
        "plateau_stages.csv, row all, col fixed_mean, in points")
    put("nPlateauArms", "%d" % int(float(pall["arms"])),
        "plateau_stages.csv, row all, col arms")
    put("nPlateauAllFires", "%d" % int(float(pall["fires"])),
        "plateau_stages.csv, row all, col fires: the arms the patience rule"
        " stops before the horizon")
    put("nPlateauOracleGain", dpts(f(pall, "oracle_mean") - f(pall, "fixed_mean")),
        "derived: plateau_stages.csv row all, col oracle_mean minus col"
        " fixed_mean, in points; " + ROUND_ONCE)
    pext = pick(pst, stage="extreme")
    hurt = [r for r in parm if float(r["gain"]) < 0]
    put("nPlateauHurtArms", {0: "no", 1: "one", 2: "two"}.get(len(hurt), str(len(hurt))),
        "plateau_arms.csv, count of rows with col gain < 0")
    put("nPlateauWorstLoss", dpts(-min(float(r["gain"]) for r in parm)),
        "plateau_arms.csv, most negative col gain, sign flipped, in points")
    pd_ = pick(pex, cell="dual")
    pb = pick(pex, cell="double")
    put("nDualPlateauFire", "%d" % int(float(pd_["fire_round"])),
        "plateau_extremes.csv, row dual, col fire_round")
    put("nDoublePlateauFire", "%d" % int(float(pb["fire_round"])),
        "plateau_extremes.csv, row double, col fire_round")

    isoc = rows("isolated_clients.csv")
    percl = {}
    for r in isoc:
        percl.setdefault(r["client"], []).append(r)
    g0means = {c: sum(f(r, "g0_own_test") for r in rs) / len(rs)
               for c, rs in percl.items()}
    isosrc = {c: sum(f(r, "iso_g0_src_test") for r in rs) / len(rs)
              for c, rs in percl.items()}
    put("nGZeroClientMin", acc(min(g0means.values()), 2),
        "isolated_clients.csv, per-client fold-mean of g0_own_test, minimum")
    put("nGZeroClientMax", acc(max(g0means.values()), 2),
        "isolated_clients.csv, per-client fold-mean of g0_own_test, maximum")
    put("nIsoWorstSrc", acc(min(isosrc.values())),
        "isolated_clients.csv, per-client fold-mean of iso_g0_src_test, minimum")

    r = pick(refs, cell="isolated (scratch)")
    put("nIsolatedScratchOwn", acc(f(r, "adaptation")),
        "references.csv, row 'isolated (scratch)', col adaptation")
    put("nIsolatedScratchSrc", acc(f(r, "preservation")),
        "references.csv, row 'isolated (scratch)', col preservation")
    r = pick(refs, cell="isolated (from g-0)")
    put("nIsolatedGlobalOwn", acc(f(r, "adaptation")),
        "references.csv, row 'isolated (from g-0)', col adaptation")
    put("nIsolatedGlobalSrc", acc(f(r, "preservation")),
        "references.csv, row 'isolated (from g-0)', col preservation")
    r = pick(refs, cell="centralized (from g-0)")
    put("nCentralAdapt", acc(f(r, "adaptation")),
        "references.csv, row 'centralized (from g-0)', col adaptation")
    put("nCentralPres", acc(f(r, "preservation")),
        "references.csv, row 'centralized (from g-0)', col preservation")

    # the parallel-schedule control, the run Figure 2 draws
    ctlp = pick(agg, cell="control_fedavg", family="parallel")
    put("nFedAvgAdapt", acc(f(ctlp, "adaptation")),
        "agg-winners.csv, row control_fedavg/concurrent, col adaptation")
    put("nFedAvgPres", acc(f(ctlp, "preservation")),
        "agg-winners.csv, row control_fedavg/concurrent, col preservation")

    # ---- Section 6.2, the trimmed row -----------------------------------
    trim = max((r for r in agg if r["cell"].startswith("trimmed_")),
               key=lambda r: float(r["score"]))
    frac = TRIM_FRACTION[trim["cell"]]
    n_part = int(round(f(pick(cost_a, arm="c10d10/control"), "clients_per_round")))
    k = int(trim["cell"].split("_t")[1])

    # ---- Section 6.2, the best server rules -----------------------------
    # The parallel rule the record crowned is the anchor (AGG_CROWNED_CONC);
    # it is the rule that enters the combination stage and the crowned pair.
    # On the test table below, eta_0p95 scores higher; the selection rule runs
    # on validation, so the arm is named from the record and not re-derived.
    r = pick(agg, cell=AGG_CROWNED_CONC, family="parallel")
    put("nAggBestConc", prose(r["cell"]),
        "agg-winners.csv, row %s/parallel, col cell -- %s" % (r["cell"], FROZEN))
    put("nAggBestConcAdapt", acc(f(r, "adaptation")),
        "agg-winners.csv, row %s/parallel, col adaptation" % r["cell"])
    put("nAggBestConcPres", acc(f(r, "preservation")),
        "agg-winners.csv, row %s/parallel, col preservation" % r["cell"])
    eta = max((x for x in agg if x["cell"].startswith("eta_")),
              key=lambda x: float(x["score"]))
    put("nAggEtaBest", "%g" % suffix(eta["cell"], "eta_"),
        "agg-winners.csv, row %s, col cell: the server step coefficient" % eta["cell"])
    ctlp = pick(agg, cell="control_fedavg", family="parallel")
    put("nAggBestGainOverFedAvg", dpts(f(r, "score") - f(ctlp, "score")),
        "derived: agg-winners.csv, row %s/parallel minus row control_fedavg/parallel,"
        " col score (same schedule), in points" % r["cell"])
    r = top(agg, family="cyclic")
    put("nAggBestSeq", prose(r["cell"]),
        "agg-winners.csv, top-score cyclic row (%s), col cell" % r["cell"])

    # ---- Section 6.3, the best penalties (blends excluded; they are the
    #      subject of Section 6.4 and are reported separately) ------------
    single = [r for r in regu if not r["cell"].startswith("hybrid")]
    r = top(single, family="parallel")
    put("nRegBestConc", prose(r["cell"]),
        "reg-winners.csv, top-score parallel non-blend row (%s), col cell" % r["cell"])
    put("nRegBestConcAdapt", acc(f(r, "adaptation")),
        "reg-winners.csv, row %s/parallel, col adaptation" % r["cell"])
    put("nRegBestConcPres", acc(f(r, "preservation")),
        "reg-winners.csv, row %s/parallel, col preservation" % r["cell"])
    cyc = sorted((r for r in single if r["family"] == "cyclic"),
                 key=lambda r: -float(r["score"]))
    put("nRegBestSeqA", prose(cyc[0]["cell"]),
        "reg-winners.csv, 1st cyclic non-blend row by score (%s), col cell" % cyc[0]["cell"])
    put("nRegBestSeqAdapt", acc(f(cyc[0], "adaptation")),
        "reg-winners.csv, row %s/cyclic, col adaptation" % cyc[0]["cell"])
    put("nRegBestSeqPres", acc(f(cyc[0], "preservation")),
        "reg-winners.csv, row %s/cyclic, col preservation" % cyc[0]["cell"])
    conc = top(single, family="parallel")
    put("nRegBestSpent", pts(100 * f(cyc[0], "spent")),
        "reg-winners.csv, row %s/cyclic, col spent (= P0 - preservation), in points"
        % cyc[0]["cell"])

    # the exact no-penalty control of the penalty family
    nopen = pick(regu, cell="param_l2_mu0", family="cyclic")
    put("nControlSpent", pts(100 * f(ctlp, "spent")),
        "agg-winners.csv, row control_fedavg/concurrent, col spent"
        " (= P0 - preservation), in points; same run as nFedAvgAdapt/Pres")

    tiny = pick(regu, cell="param_l2_mu0p0001", family="parallel")
    # NOTE: reg-winners.csv carries param_l2_mu0 on the CYCLIC schedule only, so
    # the no-penalty reference here is cross-schedule and the quantity is a
    # PRESERVATION difference (as the sentence states), not a score difference.

    # ---- Section 6.4, the blend -----------------------------------------
    blends = [r for r in regu if r["cell"].startswith("hybrid")]
    bb = max(blends, key=lambda r: float(r["score"]))
    put("nBlendBestAdapt", acc(f(bb, "adaptation")),
        "reg-winners.csv, top-score blend row (%s/%s), col adaptation"
        % (bb["cell"], bb["family"]))
    put("nBlendBestPres", acc(f(bb, "preservation")),
        "reg-winners.csv, top-score blend row (%s/%s), col preservation"
        % (bb["cell"], bb["family"]))
    put("nBlendBestSpent", pts(100 * f(bb, "spent")),
        "reg-winners.csv, row %s/%s, col spent (= P0 - preservation), in points"
        % (bb["cell"], bb["family"]))
    # THE SCREEN THAT SWEPT ALL THREE COEFFICIENTS DID NOT BEAT THE
    # CONSTRUCTION THAT SWEPT ONE.  tab:reg_winners carries both kinds of
    # composite, and the sentence that reads them wants the gap rather than
    # two printed scores a reader has to subtract -- which is how a difference
    # lands a digit away from both numbers it was taken between.  The
    # construction row is the test-leading composite, the cyclic-tuned one at
    # m = 0.75 that leads both blocks of that table and that no validation
    # ordering selected; the screened row is the single blend_* cell
    # reg-winners.csv carries for that schedule.
    from make_paper_tables import BLEND_TEST_LEADING  # noqa: E402
    screened = {}
    for family in ("parallel", "cyclic"):
        hit = [r for r in regu if r["cell"].startswith("blend_")
               and r["family"] == family]
        assert len(hit) == 1, (family, [r["cell"] for r in hit])
        screened[family] = hit[0]
    for macro, family in (("nBlendScreenGapPar", "parallel"),
                          ("nBlendScreenGapCyc", "cyclic")):
        lead = pick(regu, cell=BLEND_TEST_LEADING, family=family)
        put(macro, dpts(f(lead, "score") - f(screened[family], "score")),
            "reg-winners.csv, row %s/%s col score minus row %s/%s col score,"
            " in points: the test-leading composite of tab:reg_winners less"
            " the screened composite of the same schedule; %s"
            % (BLEND_TEST_LEADING, family, screened[family]["cell"], family,
               ROUND_ONCE))

    # AND THE SCREEN DID BEAT WHAT THE CYCLIC SELECTION ACTUALLY TOOK.  Which
    # composite that was is read off the cross rather than typed: the penalty
    # half of the cyclic combinations is the composite the cyclic validation
    # ordering carried, and it is the parallel-tuned one at m = 0.5 run under
    # the cyclic schedule, not the cyclic-tuned one the block leads with.
    crossed = {(split_combo(r["cell"], agg_cells)[1], r["family"])
               for r in combo}
    took = sorted(cell for cell, family in crossed
                  if family == "cyclic" and cell.startswith("hybrid"))
    assert len(took) == 1, took
    taken = pick(regu, cell=took[0], family="cyclic")
    put("nBlendScreenVsSelectedCyc",
        dpts(f(screened["cyclic"], "score") - f(taken, "score")),
        "reg-winners.csv, row %s/cyclic col score minus row %s/cyclic col"
        " score, in points: the screened composite less the composite the"
        " cyclic validation ordering selected, which is read off combos.csv"
        " as the penalty half of that schedule cross; %s"
        % (screened["cyclic"]["cell"], taken["cell"], ROUND_ONCE))

    kd = [r for r in blend if r["minus"].startswith("kd_")]
    assert len(kd) == 6
    put("nBlendVsKdMax", pts(max(f(r, "mean") for r in kd)),
        "blends.csv, col mean, maximum over the %d rows whose col minus is a"
        " distillation parent, in points" % len(kd))
    ewc = [r for r in blend if not r["minus"].startswith("kd_")]
    assert len(ewc) == 6
    put("nBlendKdFoldWins", "%s of %s"
        % (word(sum(1 for r in kd if int(r["all_positive"]))), word(len(kd))),
        "blends.csv, col all_positive over the %d rows whose col minus is a"
        " distillation parent" % len(kd))
    put("nBlendEwcFoldWins", "%s of %s"
        % (word(sum(1 for r in ewc if int(r["all_positive"]))), word(len(ewc))),
        "blends.csv, col all_positive over the %d rows whose col minus is a"
        " consolidation parent" % len(ewc))

    # ---- Section 6.5, the combinations ----------------------------------
    win = pick(combo, cell=CROWNED[0], family=CROWNED[1])
    put("nComboWinner", prose(win["cell"], agg_cells),
        "combos.csv, row %s/%s, col cell -- the crowned pair, %s"
        % (CROWNED[0], CROWNED[1], FROZEN))
    put("nComboWinnerAdapt", acc(f(win, "adaptation")),
        "combos.csv, row %s, col adaptation" % win["cell"])
    put("nComboWinnerPres", acc(f(win, "preservation")),
        "combos.csv, row %s, col preservation" % win["cell"])
    # THE DENOMINATOR IS THE SENTENCE'S OWN.  The text reads "closing X of the
    # shipped model's gap", and the gap it names is the one \nStartingGap
    # reports: P0 - A0, the shipped model's cohort gap.  This used to divide by
    # the CENTRALIZED ceiling's gain instead, which is a different question
    # with a different answer (89% against 64%), so both are emitted now and
    # the one the sentence means is the one the sentence uses.
    cen = pick(refs, cell="centralized (from g-0)")
    put("nComboWinnerHeadroom", "%d\\%%" % round(100 * f(win, "gained") / (P0 - A0)),
        "derived: combos.csv row %s col gained / the shipped model's cohort"
        " gap P0 - A0 (%s; %s), as a percentage" % (win["cell"], SRC_P0, SRC_A0))
    put("nComboWinnerHeadroomCeiling",
        "%d\\%%" % round(100 * f(win, "gained") / f(cen, "gained")),
        "derived: combos.csv row %s col gained / references.csv row"
        " 'centralized (from g-0)' col gained -- the same gain read against the"
        " ceiling rather than against the gap; no sentence uses it"
        % win["cell"])

    put("nComboWinnerSpent", pts(100 * f(win, "spent")),
        "combos.csv, row %s, col spent (= P0 - preservation), in points; %s"
        % (win["cell"], ROUND_ONCE))

    bal = pick(combo, cell=BALANCED[0], family=BALANCED[1])
    put("nComboBalanced", prose(bal["cell"], agg_cells),
        "combos.csv, row %s/%s, col cell -- the balanced arm, %s"
        % (BALANCED[0], BALANCED[1], FROZEN))
    put("nComboBalancedAdapt", acc(f(bal, "adaptation")),
        "combos.csv, row %s, col adaptation" % bal["cell"])
    put("nComboBalancedPres", acc(f(bal, "preservation")),
        "combos.csv, row %s, col preservation" % bal["cell"])

    seqarm = pick(combo, cell=CYCLIC_ARM[0], family=CYCLIC_ARM[1])
    put("nComboSeqAdapt", acc(f(seqarm, "adaptation")),
        "combos.csv, row %s/%s, col adaptation -- the cyclic arm, %s"
        % (CYCLIC_ARM[0], CYCLIC_ARM[1], FROZEN))
    put("nComboSeqPres", acc(f(seqarm, "preservation")),
        "combos.csv, row %s/%s, col preservation -- the cyclic arm, %s"
        % (CYCLIC_ARM[0], CYCLIC_ARM[1], FROZEN))

    # THE JOINT-TUNING EXTENSION'S OWN WINNER, Section 4.7.  The section says
    # which half-life the search chose and then stops, because the winning
    # configuration is printed nowhere: tab:jointtune reports the arms' scores
    # and not their dials, so a sentence naming the chosen coefficients had no
    # source on the page and would have been retyped off a CSV by hand.  Each
    # dial is read off rank 1 of its schedule in the screen's ranked view.
    #
    # TWO FILES, ONE WINNER.  The dials live in the screen's view and the arm
    # the table reports lives in the table's own; nothing but the assertion
    # below ties them together, and without it a re-run that moved the winner
    # would leave the sentence describing one cell beside a table reporting
    # another.
    jt_screen = rows("extension_combo_screen_selected.csv")
    jt_table = rows("extension_combo_tune_selected.csv")
    for fam, tag in (("concurrent", "Par"), ("sequential", "Cyc")):
        won = pick([r for r in jt_screen if r["family"] == fam], rank="1")
        shown = pick(jt_table, family=fam, role="tuned pair")
        assert won["cell"] == shown["arm"], (
            "the joint-tuning macros would describe %s and tab:jointtune "
            "reports %s on the %s schedule" % (won["cell"], shown["arm"], fam))
        for col, name, what in (
                ("c_ewc", "Ewc", "the consolidation coefficient"),
                ("c_kd", "Kd", "the distillation coefficient"),
                ("T", "T", "the distillation temperature"),
                ("m", "Mix", "the blend weight")):
            put("nJointSel%s%s" % (name, tag), dial(won, col),
                "extension_combo_screen_selected.csv, row family=%s rank=1"
                " (%s), col %s -- %s the joint search chose on that schedule;"
                " asserted equal to the 'tuned pair' arm of"
                " extension_combo_tune_selected.csv" % (fam, won["cell"], col,
                                                        what))

    # every selected penalty against every selected server rule, fold-paired
    put("nCompositionMinGap", pts(min(f(r, "mean") for r in comp)),
        "composition.csv, col mean, minimum over all %d pairings, in points" % len(comp))
    put("nCompositionMaxGap", pts(max(f(r, "mean") for r in comp)),
        "composition.csv, col mean, maximum over all %d pairings, in points" % len(comp))

    # how consistent that one-sided comparison is, per schedule: a pairing
    # counts only if the penalty beats the rule on every one of the five folds.
    for fam, name in (("concurrent", "nCompositionConcFolds"),
                      ("sequential", "nCompositionSeqFolds")):
        sub = [r for r in comp if r["family"] == fam]
        allpos = sum(1 for r in sub if int(r["all_positive"]))
        put(name, "%s of %s" % (word(allpos), word(len(sub))),
            "composition.csv, rows with col family = %s: col all_positive"
            " summed over the %d pairings of that schedule" % (fam, len(sub)))

    # a combination against its better component.  combos.csv carries means
    # only, so the gain is recomputed from the two parent tables; the
    # fold-paired verdict of the same comparison is in combos_folds.csv,
    # counted below.
    def comp_gain(cell, family):
        row = pick(combo, cell=cell, family=family)
        rule, pen = split_combo(cell, agg_cells)
        a = pick(agg, cell=rule, family=family)
        b = pick(regu, cell=pen, family=family)
        parent = a if f(a, "score") >= f(b, "score") else b
        return (f(row, "score") - f(parent, "score"),
                "derived: combos.csv, row %s/%s, col score, minus the better of"
                " agg-winners.csv row %s/%s and reg-winners.csv row %s/%s"
                " (the better component is %s), in points"
                % (cell, family, rule, family, pen, family, parent["cell"]))

    for macro, cell in (("nNtdAnchorGain", "anchor_h2_ntd_b0p01_t0p5"),
                        ("nNtdEtaGain", "eta_0p95_ntd_b0p01_t0p5")):
        g, src = comp_gain(cell, "parallel")
        put(macro, dpts(g), src)

    # the composition question read on five-fold means: how many of the cross
    # arms clear the better of the two components they are built from.
    put("nComboArms", word(len(combo)),
        "combos.csv, number of rows -- the cross of three rules with three"
        " penalties inside each of the two schedules")
    put("nComboBeatMeans",
        word(sum(1 for r in combo
                 if comp_gain(r["cell"], r["family"])[0] > 0)),
        "derived: combos.csv, rows whose score exceeds the better of its two"
        " components in agg-winners.csv and reg-winners.csv, counted")

    # the same question paired within folds.  combos_folds.csv carries, for
    # every cross arm, the five within-fold score differences against BOTH of
    # its components; a row counts only when every one of those ten
    # differences is positive, which is what col beats_both_folds records.
    # A mean gain that
    # changes sign across folds is not a gain, so this count and the means
    # count above are deliberately kept side by side.
    put("nComboBeatFolds",
        word(sum(1 for r in combo_folds if int(r["beats_both_folds"]))),
        "combos_folds.csv, rows with col beats_both_folds = 1, counted -- the"
        " within-fold difference is positive on all five folds against BOTH"
        " components (cols diffs_vs_agg, diffs_vs_reg)")

    # the same means count split by schedule, so the text can report the cross
    # one schedule at a time in the order Table~\ref{tab:combos} now presents
    # them.  combos.csv spells the schedules parallel/cyclic and
    # combos_folds.csv spells the same two concurrent/sequential; the two are
    # matched pairing for pairing before either is counted.
    for fam, ffam, suffix_ in (("parallel", "concurrent", "Conc"),
                               ("cyclic", "sequential", "Seq")):
        sub = [r for r in combo if r["family"] == fam]
        subf = [r for r in combo_folds if r["family"] == ffam]
        assert len(sub) == len(subf), (fam, len(sub), len(subf))
        put("nComboBeatMeans" + suffix_,
            "%s of %s" % (word(sum(1 for r in sub
                                   if comp_gain(r["cell"], r["family"])[0] > 0)),
                          word(len(sub))),
            "derived: combos.csv, rows with col family = %s whose score exceeds"
            " the better of its two components in agg-winners.csv and"
            " reg-winners.csv, counted over the %d pairings of that schedule"
            % (fam, len(sub)))

    # ---- Section 6.6, the carry settings --------------------------------
    n_by_file = {"five": 5, "c10d10": 10, "c20d10": 20, "c20d20": 20}
    listed = sorted({n_by_file[k] for k in sizes})

    ctl5 = [r for r in sizes["five"] if r["cell"] == "control"]
    n5 = int(round(f(pick(cost_s, stage="five clients, one dropped"), "clients_per_round")))

    delta = 0.0
    for cell in ("sequential", "winner", "balanced"):
        a = pick(sizes["c20d10"], cell=cell)
        b = pick(sizes["c20d20"], cell=cell)
        for col in ("adaptation", "preservation"):
            delta = max(delta, abs(f(a, col) - f(b, col)))
    put("nDropoutMaxDelta", dpts(delta, 1),
        "derived: max |sizes_c20d10.csv - sizes_c20d20.csv| over rows"
        " sequential/winner/balanced and cols adaptation, preservation, in points")

    w5 = pick(sizes["five"], cell="winner")
    w20 = pick(sizes["c20d10"], cell="winner")
    floor20 = min(f(r, "preservation")
                  for k in ("c20d10", "c20d20") for r in sizes[k])

    balp = [f(pick(sizes[k], cell="balanced"), "preservation") for k in sizes]
    gap = max(f(pick(sizes[k], cell="winner"), "adaptation")
              - f(pick(sizes[k], cell="balanced"), "adaptation") for k in sizes)

    # the carried arms are the three frozen ones; 'control' is plain FedAvg
    carried = [(k, r) for k in sizes for r in sizes[k] if r["cell"] != "control"]
    put("nCarrySpentMin", pts(100 * min(f(r, "spent") for _, r in carried)),
        "sizes_*.csv, col spent (= P0 - preservation), minimum over the %d"
        " non-control rows of the four settings, in points" % len(carried))
    put("nCarrySpentMax", pts(100 * max(f(r, "spent") for _, r in carried)),
        "sizes_*.csv, col spent (= P0 - preservation), maximum over the %d"
        " non-control rows of the four settings, in points" % len(carried))

    def worst_control(k):
        """The control row that spends most, i.e. the worse of the schedules."""
        c = [r for r in sizes[k] if r["cell"] == "control"]
        return max(c, key=lambda r: f(r, "spent")) if c else None

    # THE TWO CONTROLS CAN PRINT ONE VALUE AND THE COMMENT HAS TO SAY SO.  At
    # ten clients the cyclic control spends 2.8234 points and the parallel one
    # 2.8181: the cyclic row is the worse of the two, the macro is its value,
    # and both round to 2.82.  A reader who checks the macro by subtracting
    # the PRINTED preservations of tab:scaling instead gets 2.82 and 2.83 and
    # concludes that the value is the parallel row and the comment on it
    # wrong.  Both spends now go into the comment at the precision that tells
    # them apart, which is the precision the choice was made at.
    for macro, k in (("nControlSpentTen", "c10d10"), ("nControlSpentFive", "five")):
        c = worst_control(k)
        other = [r for r in sizes[k] if r["cell"] == "control" and r is not c]
        assert len(other) == 1, (k, len(other))
        put(macro, pts(100 * f(c, "spent")),
            "sizes_%s.csv, row control, col spent (= P0 - preservation):"
            " the worse (larger) of the parallel and cyclic schedules, here"
            " %s at %s points against %s at %s, so the macro is the %s row"
            " -- the two can round to one printed value, and subtracting the"
            " PRINTED preservations of tab:scaling separates them instead and"
            " points at the wrong one; %s"
            % (k, c["family"], pts(100 * f(c, "spent"), 4),
               other[0]["family"], pts(100 * f(other[0], "spent"), 4),
               c["family"], ROUND_ONCE))

    # margin over the control; the twenty-client settings carry no control row
    margins = []
    for k in sizes:
        ctls = [r for r in sizes[k] if r["cell"] == "control"]
        if not ctls:
            continue
        best_ctl = max(f(r, "score") for r in ctls)
        for _, r in [(k, r) for r in sizes[k] if r["cell"] != "control"]:
            margins.append((f(r, "score") - best_ctl, k, r["cell"]))
    m, mk, mc = min(margins)
    put("nCarryMinMargin", dpts(m),
        "derived: sizes_*.csv, col score, minimum over every non-control row of"
        " (arm - the better-scoring control of its setting); attained by"
        " sizes_%s.csv row %s. The two twenty-client settings carry no control"
        " row and are excluded. In points, %s." % (mk, mc, ROUND_ONCE))

    # ---- Section 6.7, fairness ------------------------------------------
    # how far the worst-served client of each arm moved, in points
    put("nFairWorstMinGain", dpts(min(f(r, "lift_worst") for r in fair)),
        "fairness_c10d10.csv, col lift_worst, minimum over the %d arm rows,"
        " in points" % len(fair))
    put("nFairWorstMaxGain", dpts(max(f(r, "lift_worst") for r in fair)),
        "fairness_c10d10.csv, col lift_worst, maximum over the %d arm rows,"
        " in points" % len(fair))

    # THE SAME RANGE OVER THE CARRIED ARMS ALONE.  The two macros above run
    # over every row of the ten-client block because the sentence they serve
    # is about every arm the block prints, controls included.  A sentence
    # about what the carrying bought cannot include them: plain FedAvg is the
    # thing a carried arm is measured against, it is the row that lifts its
    # worst client least, and folding it in makes the range read wider at the
    # bottom than any carried arm went.  'control' is the one cell of the
    # block that is not a carried arm - the same rule the carry settings are
    # read by above - so the three left are the three of tab:scaling.
    fair_carried = [r for r in fair if r["cell"] != "control"]
    assert len(fair_carried) == 3, sorted(r["cell"] for r in fair_carried)
    CARRIED_SRC = ("fairness_c10d10.csv, col lift_worst, %s over the %d"
                   " non-control rows of the ten-client carry block"
                   " (cells %s), in points")
    cells = ", ".join(sorted(r["cell"] for r in fair_carried))
    put("nFairWorstMinGainCarried",
        dpts(min(f(r, "lift_worst") for r in fair_carried)),
        CARRIED_SRC % ("minimum", len(fair_carried), cells))
    put("nFairWorstMaxGainCarried",
        dpts(max(f(r, "lift_worst") for r in fair_carried)),
        CARRIED_SRC % ("maximum", len(fair_carried), cells))

    for macro, k in (("nBalancedWorstTwentyTwo", "c20d10"),
                     ("nBalancedWorstTwentyFour", "c20d20")):
        r = pick(fair20[k], cell="balanced", family="parallel")
        put(macro, signed(f(r, "lift_worst")),
            "fairness_%s.csv, row balanced/parallel, col lift_worst, in points"
            " (negative: below the shipped model)" % k)

    # ---- Section 6.7, cost ----------------------------------------------
    ca = pick(cost_a, arm="c10d10/winner")
    put("nCommPerRound", "%.1f\\,MB" % f(ca, "mb_per_round"),
        "cost_arms.csv, row c10d10/winner, col mb_per_round")
    put("nParamCount", group(f(ca, "param_count")),
        "cost_arms.csv, row c10d10/winner, col param_count")

    # ---- Section 6.8, the extremes --------------------------------------
    # The per-round decision table.  Its final_* columns are the same fixed
    # horizon as extremes.csv read on the per-round (validation) trace, so the
    # fixed-horizon and oracle numbers quoted together below share one basis.
    sx = {r["cell"]: r for r in stopx}
    ex = {r["cell"]: r for r in extr}

    # The fixed-horizon numbers of Section 6.8 are read from extremes.csv, the
    # TEST table the section's own Table~\ref{tab:extremes} prints, so the
    # prose and the table agree.  stopping_extreme.csv carries the same
    # horizon on the per-round VALIDATION trace (cols final_*), which differs:
    # single 0.7375 / -10.66, double 0.9373.  Both appear side by side in that
    # table, and the trace pair is what Section 7 argues from.
    put("nDualSatRound", "%d" % int(f(sx["dual"], "oracle_round")),
        "stopping_extreme.csv, row dual, col oracle_round")

    # ---- Section 7, stopping and the permitted signals -------------------
    # Two files carry this section.  stopping_all.csv is one row per
    # hundred-round arm: the fixed horizon, the oracle stop and the one rule
    # the whole study fixes in advance, all on the per-round VALIDATION trace,
    # which is the only basis on which a stopping round may be chosen.
    # signals_summary_extract.csv is one row per permitted signal: how its
    # drift tracks forgetting within a run, and what it delivers as a rule at
    # its own best budget over the same arms.
    SIG = {r["signal"]: r for r in sigs}
    S7 = "stopping_all.csv, all %d arms" % len(stopa)

    def mean(col, sel=None):
        chosen = [r for r in stopa if sel is None or sel(r)]
        return sum(f(r, col) for r in chosen) / len(chosen)

    def median(values):
        v = sorted(values)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    stages = []
    for r in stopa:
        if r["stage"] not in stages:
            stages.append(r["stage"])
    gap_median = {s: median([f(r, "oracle_cost") for r in stopa
                             if r["stage"] == s]) for s in stages}
    extreme_gaps = sorted(f(r, "oracle_cost") for r in stopa
                          if r["stage"] == "extreme")
    main_medians = [gap_median[s] for s in stages if s != "extreme"]

    put("nStoppingArms", "%d" % len(stopa),
        "stopping_all.csv, number of data rows")
    put("nFixedMeanScore", dpts(mean("final_score")),
        "derived: mean of col final_score over %s, in points" % S7)
    put("nOracleMeanScore", dpts(mean("oracle_score")),
        "derived: mean of col oracle_score over %s, in points" % S7)
    put("nOracleMeanGain", dpts(mean("oracle_score") - mean("final_score")),
        "derived: mean of col oracle_score minus mean of col final_score over"
        " %s, in points; %s" % (S7, ROUND_ONCE))
    put("nOneRuleMeanScore", dpts(mean("one_score")),
        "derived: mean of col one_score over %s, in points" % S7)
    put("nOneRuleMeanGain", dpts(mean("one_vs_final")),
        "derived: mean of col one_vs_final over %s, in points" % S7)

    deltas = {r["one_delta"] for r in stopa}
    assert len(deltas) == 1, deltas
    signals = {r["one_signal"] for r in stopa}
    assert len(signals) == 1, signals
    put("nOneRuleDelta", "%g" % float(deltas.pop()),
        "stopping_all.csv, col one_delta (identical on every row)")

    put("nKlRuleScore", dpts(f(SIG["kl_global_to_current"],
                               "mean_stopped_score")),
        "signals_summary_extract.csv, row kl_global_to_current, col"
        " mean_stopped_score, in points")
    put("nProxyKlRuleScore", dpts(f(SIG["proxy_kl"], "mean_stopped_score")),
        "signals_summary_extract.csv, row proxy_kl, col mean_stopped_score,"
        " in points")



    # ---- Section 7, the correlation table --------------------------------
    # median_rho is the per-run median rank correlation of the signal's DRIFT
    # against the source-validation drop; the extractor applies the signal's
    # own direction so that every entry is 'forgetting makes it grow'.
    put("nSignalCount", word(len(sigs)),
        "signals_summary_extract.csv, number of data rows --- the signals the"
        " runner records on every arm")
    # THREE POPULATIONS, THREE NAMES, AND THEY ARE NOT THE SAME POPULATION.
    # nSignalRuns is every run the signals pass walked - the programme as it
    # stood before its last two stages.  nSignalCorrRuns is the subset of those
    # that carry a source-validation series, and so the only runs a rank
    # correlation against the fall in source accuracy can be computed on; it is
    # the number the signals table's note prints.  nProgrammeTasks is a count
    # of a different thing entirely: run FOLDERS over every stage of the whole
    # programme, screens included, which is neither a subset nor a superset of
    # a run count because a folder that computes both schedules stores two
    # results (nProgrammePayloads counts those).  All three used to be printed
    # as "runs".
    put("nSignalRuns", group(float(sigx["study_runs"])),
        "signals_extras_extract.csv, key study_runs --- every run the signals"
        " pass covered (the programme before its last two stages)")
    put("nSignalCorrRuns", group(float(sigx["signal_runs_with_source_val"])),
        "signals_extras_extract.csv, key signal_runs_with_source_val --- the"
        " runs of that pass which carry a source-validation series, and so the"
        " population every rho in the table is a median over")
    for macro, signal in (("KL", "kl_global_to_current"),
                          ("ProxyKL", "proxy_kl"),
                          ("LTwo", "dist_l2_to_global"),
                          ("Agreement", "agreement_with_global"),
                          ("Fisher", "dist_fisher_to_global"),
                          ("FisherNorm", "dist_fisher_norm_to_global"),
                          ("ProxyAcc", "proxy_acc"),
                          ("Retention", "retention_known")):
        row = SIG[signal]
        put("nSignal%sRho" % macro, "%.3f" % f(row, "median_rho"),
            "signals_summary_extract.csv, row %s, col median_rho" % signal)
        put("nSignal%sShare" % macro,
            "%.1f\\%%" % (100.0 * f(row, "share_ge_0p9")),
            "signals_summary_extract.csv, row %s, col share_ge_0p9" % signal)
        # EVERY CELL OF THE SIGNALS TABLE, AS A MACRO. Section 7 reads the
        # table across, not down, and the four columns on its right are the
        # ones an argument is made from - which budget the signal scored best
        # at, how many arms it stopped, what that cost and what it bought.
        # A number the prose has to retype from a table is a number that stops
        # agreeing with the table on the next data refresh.
        put("nSignal%sDelta" % macro, "%g" % f(row, "best_delta"),
            "signals_summary_extract.csv, row %s, col best_delta: the budget"
            " on the shared grid at which that signal scores best" % signal)
        put("nSignal%sFires" % macro, "%d" % int(f(row, "arms_fired")),
            "signals_summary_extract.csv, row %s, col arms_fired: the arms of"
            " nStoppingArms on which the rule stops before the horizon" % signal)
        put("nSignal%sScore" % macro, dpts(f(row, "mean_stopped_score")),
            "signals_summary_extract.csv, row %s, col mean_stopped_score, in"
            " points" % signal)
        put("nSignal%sVsFixed" % macro, signed(f(row, "mean_vs_fixed")),
            "signals_summary_extract.csv, row %s, col mean_vs_fixed, in"
            " points. The table prints the difference of its two PRINTED"
            " columns instead, so the two can differ by a hundredth; this is"
            " the stored difference" % signal)

    # ---- Section 5, the protocol ----------------------------------------
    put("nBadPoolSize", group(float(cohort[0]["of"])),
        "cohort_table.csv, col 'of' (identical on every row)")
    cohort_clients = sum(1 for r in cohort_comp if r["cohort"] == "cohort10")
    put("nRounds", "%d" % int(f(pick(cost_s, stage="regularisation finals"), "rounds")),
        "cost_stages.csv, row 'regularisation finals', col rounds")

    # The client recipe.  These are configuration and not measurements, which
    # is why they lived in the hand-maintained block for as long as they did;
    # recipe.csv reads them back out of the stored payloads, so the paragraph
    # that states them now has a file behind it like every other number here.
    put("nLearningRate", power(recipe["learning_rate"]["value"]),
        "recipe.csv, row learning_rate, col value (%s payloads, all agreeing)"
        % recipe["learning_rate"]["payloads"])
    put("nWeightDecay", power(recipe["weight_decay"]["value"]),
        "recipe.csv, row weight_decay, col value (%s payloads, all agreeing)"
        % recipe["weight_decay"]["payloads"])

    # The participation the search setting ran at, as the fraction NOT drawn.
    # Both halves are read: the per-round count off the stage that crowned the
    # pair, the cohort size off the cohort itself.  Writing 0.2 here would be
    # writing down a ratio that two other views already fix.
    per_round = f(pick(cost_s, stage="combinations"), "clients_per_round")
    put("nDropoutFraction", ("%g" % (1.0 - per_round / cohort_clients)),
        "derived: 1 - cost_stages.csv row 'combinations' col clients_per_round"
        " / cohort_composition.csv count of rows with cohort = cohort10")

    # What the programme cost, summed over its stages rather than restated.
    # docs/FAIRNESS_AND_COST.md prints the same two totals under the same
    # table and tests/test_recipe.py holds the prose to this sum.
    put("nProgrammeTasks", group(sum(f(r, "tasks") for r in cost_s)),
        "cost_stages.csv, col tasks summed over every stage --- run FOLDERS"
        " over the whole programme, screens included; not a run count and not"
        " comparable with nSignalRuns")
    put("nProgrammeGpuHours", "%.1f" % sum(f(r, "hours") for r in cost_s),
        "cost_stages.csv, col hours summed over every stage, in GPU-hours; "
        + ROUND_ONCE)

    # The decoupling example. The extract flags exactly one row: the writer
    # the detector ranked worst of the whole bad pool, which the shipped
    # model then scored at the top of that pool.  Both accuracies are read
    # from the flagged row, never restated by hand.
    flagged = [r for r in decouple if r["chosen"] == "yes"]
    if len(flagged) != 1:
        raise KeyError("decouple_example.csv must flag exactly one row, "
                       "found %d" % len(flagged))
    dec = flagged[0]
    put("nDecoupleGZero", acc(f(dec, "g0_acc")),
        "decouple_example.csv, row %s (chosen), col g0_acc" % dec["writer"])

    # ---- the two shipped-model worst-client figures, told apart ----------
    # THEY ARE THE SAME WRITERS, THE SAME ROWS AND THE SAME MODEL, and they
    # differ by the order the minimum and the mean are taken in.
    # nGZeroClientMin is the smallest of the ten clients' own five-fold means:
    # one writer, its whole record.  The three below are the five-fold mean of
    # the worst-served client OF EACH FOLD, which is the reference the fairness
    # table's delta-worst column is a delta from - and the worst-served client
    # is not the same writer on every fold, so the mean of the minima sits
    # below the minimum of the means.  Quoting either against the other as if
    # they were one quantity is what these three names exist to stop.
    for macro, view in (("nGZeroWorstFoldMeanFive", fair5),
                        ("nGZeroWorstFoldMeanTen", fair),
                        ("nGZeroWorstFoldMeanTwenty", fair20["c20d10"])):
        values = {r["g0_min"] for r in view}
        assert len(values) == 1, (macro, sorted(values))
        put(macro, acc(float(values.pop())),
            "fairness view of that cohort size, col g0_min (identical on every"
            " row): the five-fold mean of the worst-served client of each fold,"
            " under the shipped model, on the test rows")

    # ---- what a replicate of one configuration costs ---------------------
    # The two control rows of tab:agg_winners are one configuration run twice
    # out of consecutive blocks of one task file: same horizon, same folds,
    # same participation, same run seeds, differing in the client-sampling seed
    # block and in a stop flag that never fired.  So the gap between them is
    # the run-to-run spread of a hundred-round arm, which is the scale every
    # margin in the paper is read against.  nReplicateGap is NOT that: the two
    # rows it differences come from two different stages, task files, seed
    # blocks and trainer entry points, and only agree on what they nominally
    # configure.
    for macro, fam in (("nControlSeedSpread", "cyclic"),
                       ("nControlSeedSpreadPar", "parallel")):
        plain = pick(agg, cell="control_fedavg", family=fam)
        armed = pick(agg, cell="control_fedavg_earlystop", family=fam)
        put(macro, dpts(f(plain, "score") - f(armed, "score")),
            "agg-winners.csv, row control_fedavg/%s col score minus row"
            " control_fedavg_earlystop/%s col score, in points: one"
            " configuration run twice under two client-sampling seed blocks;"
            " %s" % (fam, fam, ROUND_ONCE))

    # ---- how deep tab:selection_axis prints ------------------------------
    from make_paper_tables import (SCHEDULE_BLOCKS, SELECTION_DEPTH,  # noqa: E402
                                   SELECTION_STAGES, selection_block)
    put("nSelectionDepth", word(SELECTION_DEPTH),
        "make_paper_tables.SELECTION_DEPTH: how many arms of each"
        " selecting block of tab:selection_axis prints, in words; the"
        " cross blocks of tab:selection_axis_cross print all nine")

    # ---- how much of that table moves between the two axes ---------------
    # THE CLAIM IS ABOUT THE PAGE, NOT ABOUT THE VIEW.  The table exists to
    # show that the ordering which selected and the ordering which reports are
    # two different orderings, and "most arms move" is worth nothing to a
    # reader who cannot see how many arms were on offer.  So both counts are
    # built the way the table builds its blocks -- grouped by stage and by
    # schedule, ordered on the validation score rank, cut at SELECTION_DEPTH
    # plus any arm carried forward from below that depth -- and not over every
    # row of selection_axis.csv, which carries sixty.  A count taken over the
    # view would describe a population the page does not print.
    printed = []
    for stage, _, _ in SELECTION_STAGES:
        for family, _ in SCHEDULE_BLOCKS:
            _, within, carried = selection_block(sel, stage, family)
            printed += within + carried
    blocks = len(SELECTION_STAGES) * len(SCHEDULE_BLOCKS)
    # The blocks times the depth is the floor, not the count: an arm the paper
    # carries onward is printed from wherever validation left it, and one of
    # the three sits below the cut.  Asserted as a floor rather than assumed
    # equal, so that the sentence quoting this macro cannot go a row short.
    assert len(printed) >= blocks * SELECTION_DEPTH, len(printed)
    put("nSelectionRankTotal", "%d" % len(printed),
        "selection_axis.csv, the rows tab:selection_axis and"
        " tab:selection_axis_cross print between them: the first %d by col"
        " val_score_rank of each of the %d selecting stage-by-schedule"
        " blocks, plus any row below that depth that col role marks as"
        " carried forward, that tab:scaling runs again, or that leads both"
        " test orderings, plus every row of the two cross blocks, counted"
        % (SELECTION_DEPTH, blocks - 2))
    still = [r for r in printed
             if int(r["val_score_rank"]) == int(r["test_score_rank"])]
    put("nSelectionRankMoved", "%d" % (len(printed) - len(still)),
        "selection_axis.csv, of those %d printed rows the ones whose col"
        " val_score_rank differs from col test_score_rank, counted; the %d"
        " that do not move are %s"
        % (len(printed), len(still),
           ", ".join("%s/%s" % (r["cell"], r["schedule"]) for r in still)))

    # ---- the blend that clears its distillation parent on every fold -----
    clearing = [f(r, "mean") for r in blend if r["all_positive"] == "1"]
    put("nBlendClearMax", pts(max(clearing)),
        "blends.csv, rows with all_positive=1, max col mean, in points."
        " nBlendVsKdMax is a different quantity: the max over ALL six"
        " distillation-parent rows, clearing on every fold or not")

    # ---- the held-out selection of the patience --------------------------
    # One macro per quantity the appendix table states, so the count on the
    # page and the count in the file cannot come apart by retyping.
    put("nKholdoutProtocols", word(len(prot)),
        "plateau_holdout_protocols.csv, number of data rows: every"
        " selection/holdout protocol the tool runs, in words")
    put("nKholdoutMinHeldGain", dpts(min(f(r, "held_gain") for r in prot)),
        "plateau_holdout_protocols.csv, col held_gain, minimum over every"
        " protocol, in points")
    put("nKholdoutMaxHeldGain", dpts(max(f(r, "held_gain") for r in prot)),
        "plateau_holdout_protocols.csv, col held_gain, maximum over every"
        " protocol, in points")
    for macro, held in (("nKholdoutAggfullHeldGain", "aggfull arms"),
                        ("nKholdoutRegfullHeldGain", "regfull arms"),
                        ("nKholdoutComboHeldGain", "combo arms")):
        hit = [r for r in prot if r["heldout_set"] == held
               and r["protocol"] == "stage-loo"]
        assert len(hit) == 1, (macro, len(hit))
        put(macro, dpts(f(hit[0], "held_gain")),
            "plateau_holdout_protocols.csv, the stage-loo row whose col"
            " heldout_set is %r, col held_gain, in points. This is the"
            " HELD-OUT gain; plateau_stages.csv's col gain for the same stage"
            " is the in-sample one. For regfull and combo the two are"
            " different numbers; for aggfull they coincide, because the"
            " protocol that held that stage out chose the same (k, eps)"
            " the whole-programme fit chose" % held)

    # ---- the pre-filter, and the room it ran with ------------------------
    # Coverage is 1.0 and cannot be anything else: the cohorts are cut out of
    # the pool, so a writer the pre-filter dropped is a writer the shipped
    # model was never asked to score.  The depth macros are the number the
    # records CAN answer - how narrow a pre-filter would have kept the same
    # cohort - and they are what a reader should be given.
    any_row = prefilter["ten"]
    put("nPrefilterPoolSize", group(f(any_row, "prefilter_pool")),
        "prefilter_coverage.csv, col prefilter_pool: writers the coarse"
        " detector kept, of col ranked_writers")
    put("nPrefilterRankedWriters", group(f(any_row, "ranked_writers")),
        "prefilter_coverage.csv, col ranked_writers: writers the coarse"
        " detector scored and ranked")
    put("nPrefilterSeparated", group(f(any_row, "ranked_above_ties")),
        "prefilter_coverage.csv, col ranked_above_ties: writers the detector"
        " scored strictly below its top value, and so actually separated; the"
        " rest of the pool is inside one tie and is ordered by writer id")
    put("nPrefilterTiedWriters", group(f(any_row, "writers_at_top_score")),
        "prefilter_coverage.csv, col writers_at_top_score: writers the coarse"
        " detector scored at its top value, all tied")
    for macro, key in (("Five", "five"), ("Ten", "ten"), ("Twenty", "twenty")):
        row = prefilter[key]
        put("nPrefilterCoverage%s" % macro,
            "%.0f\\,\\%%" % (100 * f(row, "coverage")),
            "prefilter_coverage.csv, row %s, col coverage, as a percentage:"
            " the share of that cohort the pre-filter kept. It is 100%% by"
            " construction - the cohort is cut out of the pool" % key)
        put("nPrefilterDepth%s" % macro,
            "%.1f\\,\\%%" % (100 * f(row, "depth_fraction")),
            "prefilter_coverage.csv, row %s, col depth_fraction, as a"
            " percentage: the deepest place a member of that cohort takes in"
            " the detector's own ranking, so the narrowest pre-filter that"
            " would still have kept the whole cohort" % key)

    # ---- the three run populations, told apart ---------------------------
    put("nRegFinalsArms", "%d" % len(regu),
        "reg-winners.csv, number of data rows: the arms the regularisation"
        " finals reported, both schedules and the two blend finals together")
    put("nProgrammePayloads", group(sum(f(r, "payloads") for r in cost_s)),
        "cost_stages.csv, col payloads summed over every stage: stored result"
        " files, which exceed the tasks because a task that computes both"
        " schedules writes two")

    folds = [c for c in cohort[0] if re.fullmatch(r"f\d+_val", c)]
    per_fold = [sum(float(r[c]) for r in cohort) for c in folds]

    return reg


# --------------------------------------------------------------------------
# rewrite
# --------------------------------------------------------------------------

NAME_RE = re.compile(r"\\newcommand\{\\(n[A-Za-z]+)\}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="do not write; fail if numbers.tex is out of date")
    args = ap.parse_args()

    tex = os.path.join(out_dir(), "numbers.tex")
    if not os.path.isfile(tex):
        # Only the block between the markers is rewritten, so the file to
        # rewrite has to exist.  A source checkout has no manuscript to copy
        # one from, which used to make this script unrunnable outside the
        # paper tree; the template shipped beside it is that starting point.
        seed = os.path.join(HERE, "numbers.template.tex")
        if not os.path.isfile(seed):
            sys.exit("%s: not there, and no numbers.template.tex beside %s to\n"
                     "seed it from.  Copy a numbers.tex into that directory."
                     % (tex, HERE))
        os.makedirs(os.path.dirname(tex) or ".", exist_ok=True)
        shutil.copyfile(seed, tex)
        print("seeded %s from numbers.template.tex" % tex)
    text = open(tex).read()
    if BEGIN not in text or END not in text:
        sys.exit("%s has no %s / %s markers" % (tex, BEGIN, END))
    head, rest = text.split("\n" + BEGIN + "\n", 1)
    block, tail = rest.split("\n" + END, 1)

    names = NAME_RE.findall(block)
    if len(names) != len(set(names)):
        sys.exit("duplicate macro names inside the generated block")

    reg = build()
    unmapped = [n for n in names if n not in reg]
    orphan = [n for n in sorted(reg) if n not in names]

    out = [BEGIN, ""]
    for n in names:
        if n in reg:
            value, src = reg[n]
            out.append("\\newcommand{\\%s}{%s}" % (n, value))
            out.append("  %% %s" % src)
        else:
            out.append("\\newcommand{\\%s}{\\TBD{unmapped}}" % n)
            out.append("  %% no mapping in make_numbers.py")
    out.append("")
    new = head + "\n" + "\n".join(out) + END + tail

    if args.check:
        if new != text:
            sys.exit("%s is out of date; run: python3 make_numbers.py" % tex)
    elif new != text:
        open(tex, "w").write(new)

    print("generated block: %d macros, %d filled, %d unmapped"
          % (len(names), len(names) - len(unmapped), len(unmapped)))
    if orphan:
        print("registry entries with no macro between the markers:")
        for n in orphan:
            print("  %s" % n)
    if unmapped:
        print("UNMAPPED (emitted as \\TBD{unmapped}):")
        for n in unmapped:
            print("  %s" % n)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
