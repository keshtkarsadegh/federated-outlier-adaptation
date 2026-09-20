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
    cohort_comp = rows("cohort_composition.csv")
    decouple = rows("decouple_example.csv")
    fair = rows("fairness_c10d10.csv")
    fairc = rows("fairness_c10d10_clients.csv")
    fair_combo = rows("fairness_combo.csv")
    fair20 = {"c20d10": rows("fairness_c20d10.csv"),
              "c20d20": rows("fairness_c20d20.csv")}
    comp = rows("composition.csv")
    blend = rows("blends.csv")
    stopx = rows("stopping_extreme.csv")
    stopa = rows("stopping_all.csv")
    sigs = rows("signals_summary_extract.csv")
    sigx = {r["key"]: r["value"] for r in rows("signals_extras_extract.csv")}
    sizes = {"five": rows("sizes_five.csv"),
             "c10d10": rows("sizes_c10d10.csv"),
             "c20d10": rows("sizes_c20d10.csv"),
             "c20d20": rows("sizes_c20d20.csv")}

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
    put("nDoubleSatRound", "%d" % int(f(doub, "oracle_round")),
        "extreme_stop_rounds.csv, row double, col oracle_round")
    put("nDualRuleRound", "%d" % int(f(dual, "rule_round")),
        "extreme_stop_rounds.csv, row dual, col rule_round")
    put("nDoubleRuleRound", "%d" % int(f(doub, "rule_round")),
        "extreme_stop_rounds.csv, row double, col rule_round")
    put("nDualRuleScore", pts(100 * f(dual, "rule_score")),
        "extreme_stop_rounds.csv, row dual, col rule_score, in points")
    put("nDoubleRuleScore", pts(100 * f(doub, "rule_score")),
        "extreme_stop_rounds.csv, row double, col rule_score, in points")
    put("nDualFinalScore", pts(100 * f(dual, "final_score")),
        "extreme_stop_rounds.csv, row dual, col final_score (round-100 trace), in points")
    put("nDoubleFinalScore", pts(100 * f(doub, "final_score")),
        "extreme_stop_rounds.csv, row double, col final_score (round-100 trace), in points")
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
    put("nPlateauFires", "%d" % int(float(pall["fires"])),
        "plateau_stages.csv, row all, col fires")
    put("nPlateauArms", "%d" % int(float(pall["arms"])),
        "plateau_stages.csv, row all, col arms")
    pext = pick(pst, stage="extreme")
    put("nPlateauExtremeGain", dpts(f(pext, "gain")),
        "plateau_stages.csv, row extreme, col gain, in points")
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

    worst = min(fairc, key=lambda r: f(r, "shipped"))
    put("nWorstWriterGap", dpts(P0 - f(worst, "shipped"), 1),
        "derived: P0 - fairness_c10d10_clients.csv, row %s, col shipped" % worst["client"])

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
    put("nTrimBestFrac", "a trim fraction of $%s$" % ("%g" % frac),
        "agg-winners.csv, row %s, col cell (label via TRIM_FRACTION)" % trim["cell"])
    n_part = int(round(f(pick(cost_a, arm="c10d10/control"), "clients_per_round")))
    k = int(trim["cell"].split("_t")[1])
    put("nTrimSurvivors", word(n_part - 2 * k),
        "derived: cost_arms.csv, row c10d10/control, col clients_per_round = %d,"
        " minus 2 x %d trimmed (agg-winners.csv, row %s)" % (n_part, k, trim["cell"]))
    put("nTrimBestAdapt", acc(f(trim, "adaptation")),
        "agg-winners.csv, row %s, col adaptation" % trim["cell"])
    put("nTrimBestPres", acc(f(trim, "preservation")),
        "agg-winners.csv, row %s, col preservation" % trim["cell"])
    put("nTrimBestCount", "%d" % k,
        "agg-winners.csv, row %s, col cell: the trimmed count t" % trim["cell"])

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
    put("nAggBestSeqAdapt", acc(f(r, "adaptation")),
        "agg-winners.csv, row %s/cyclic, col adaptation" % r["cell"])
    put("nAggBestSeqPres", acc(f(r, "preservation")),
        "agg-winners.csv, row %s/cyclic, col preservation" % r["cell"])

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
    put("nRegBestSeqB", prose(cyc[1]["cell"]),
        "reg-winners.csv, 2nd cyclic non-blend row by score (%s), col cell" % cyc[1]["cell"])
    put("nRegBestSeqAdapt", acc(f(cyc[0], "adaptation")),
        "reg-winners.csv, row %s/cyclic, col adaptation" % cyc[0]["cell"])
    put("nRegBestSeqPres", acc(f(cyc[0], "preservation")),
        "reg-winners.csv, row %s/cyclic, col preservation" % cyc[0]["cell"])
    put("nRegLogitLamSeq", "%g" % suffix(cyc[0]["cell"], "lam"),
        "reg-winners.csv, row %s/cyclic, col cell: the penalty coefficient" % cyc[0]["cell"])
    conc = top(single, family="parallel")
    put("nRegLogitLamConc", "%g" % suffix(conc["cell"], "lam"),
        "reg-winners.csv, row %s/parallel, col cell: the penalty coefficient" % conc["cell"])
    put("nRegBestSpent", pts(100 * f(cyc[0], "spent")),
        "reg-winners.csv, row %s/cyclic, col spent (= P0 - preservation), in points"
        % cyc[0]["cell"])

    # the exact no-penalty control of the penalty family
    nopen = pick(regu, cell="param_l2_mu0", family="cyclic")
    put("nControlSpent", pts(100 * f(ctlp, "spent")),
        "agg-winners.csv, row control_fedavg/concurrent, col spent"
        " (= P0 - preservation), in points; same run as nFedAvgAdapt/Pres")

    tiny = pick(regu, cell="param_l2_mu0p0001", family="parallel")
    put("nParamLBestConc", "%g" % suffix(tiny["cell"], "mu"),
        "reg-winners.csv, row %s/parallel, col cell: the penalty coefficient" % tiny["cell"])
    # NOTE: reg-winners.csv carries param_l2_mu0 on the CYCLIC schedule only, so
    # the no-penalty reference here is cross-schedule and the quantity is a
    # PRESERVATION difference (as the sentence states), not a score difference.
    put("nParamLTinyGain", dpts(f(tiny, "preservation") - f(nopen, "preservation")),
        "derived: reg-winners.csv, row param_l2_mu0p0001/parallel minus row"
        " param_l2_mu0/CYCLIC (the parallel no-penalty row is not in this file),"
        " col preservation, in points")

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
    kd = [r for r in blend if r["minus"].startswith("kd_")]
    assert len(kd) == 6
    put("nBlendVsKdMin", pts(min(f(r, "mean") for r in kd)),
        "blends.csv, col mean, minimum over the %d rows whose col minus is a"
        " distillation parent, in points" % len(kd))
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
    put("nComboBalancedGap", pts(100 * f(bal, "spent")),
        "combos.csv, row %s, col spent (= P0 - preservation), in points" % bal["cell"])
    put("nComboBalancedCost", dpts(f(win, "adaptation") - f(bal, "adaptation")),
        "derived: combos.csv, rows %s minus %s, col adaptation, in points"
        % (win["cell"], bal["cell"]))
    put("nComboCells", "%d" % int(f(pick(cost_s, stage="combinations"), "tasks")),
        "cost_stages.csv, row 'combinations', col tasks")

    seqarm = pick(combo, cell=CYCLIC_ARM[0], family=CYCLIC_ARM[1])
    put("nComboSeqAdapt", acc(f(seqarm, "adaptation")),
        "combos.csv, row %s/%s, col adaptation -- the cyclic arm, %s"
        % (CYCLIC_ARM[0], CYCLIC_ARM[1], FROZEN))
    put("nComboSeqPres", acc(f(seqarm, "preservation")),
        "combos.csv, row %s/%s, col preservation -- the cyclic arm, %s"
        % (CYCLIC_ARM[0], CYCLIC_ARM[1], FROZEN))

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
    put("nSizeList", "%s and %s" % (", ".join(word(n) for n in listed[:-1]), word(listed[-1])),
        "derived: the sizes_*.csv inventory (%s)" % ", ".join(sorted(sizes)))

    ctl5 = [r for r in sizes["five"] if r["cell"] == "control"]
    put("nFiveControlPres", acc(sum(f(r, "preservation") for r in ctl5) / len(ctl5)),
        "sizes_five.csv, rows control (cyclic, parallel), col preservation, mean")
    put("nFiveBalancedPres", acc(f(pick(sizes["five"], cell="balanced"), "preservation")),
        "sizes_five.csv, row balanced, col preservation")
    n5 = int(round(f(pick(cost_s, stage="five clients, one dropped"), "clients_per_round")))
    put("nFiveTrimSurvivors", word(n5 - 2 * int(math.floor(frac * n5))),
        "derived: cost_stages.csv, row 'five clients, one dropped',"
        " col clients_per_round = %d, at trim fraction %g" % (n5, frac))

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
    put("nFiveWinnerAdapt", acc(f(w5, "adaptation")),
        "sizes_five.csv, row winner, col adaptation")
    w20 = pick(sizes["c20d10"], cell="winner")
    put("nTwentyWinnerDrop", dpts(f(win, "adaptation") - f(w20, "adaptation"), 1),
        "derived: combos.csv row %s minus sizes_c20d10.csv row winner,"
        " col adaptation, in points" % win["cell"])
    floor20 = min(f(r, "preservation")
                  for k in ("c20d10", "c20d20") for r in sizes[k])
    put("nTwentyPresFloor", acc(floor20, 3),
        "sizes_c20d10.csv + sizes_c20d20.csv, min over all rows, col preservation")

    balp = [f(pick(sizes[k], cell="balanced"), "preservation") for k in sizes]
    put("nBalancedPresMin", acc(min(balp)),
        "sizes_*.csv, row balanced, col preservation, minimum over the four settings")
    put("nBalancedPresMax", acc(max(balp)),
        "sizes_*.csv, row balanced, col preservation, maximum over the four settings")
    gap = max(f(pick(sizes[k], cell="winner"), "adaptation")
              - f(pick(sizes[k], cell="balanced"), "adaptation") for k in sizes)
    put("nBalancedAdaptGap", dpts(gap, 2),
        "derived: sizes_*.csv, row winner minus row balanced, col adaptation,"
        " maximum over the four settings, in points")

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

    for macro, k in (("nControlSpentTen", "c10d10"), ("nControlSpentFive", "five")):
        c = worst_control(k)
        put(macro, pts(100 * f(c, "spent")),
            "sizes_%s.csv, row control, col spent (= P0 - preservation):"
            " the worse (larger) of the parallel and cyclic schedules,"
            " here %s, in points; %s" % (k, c["family"], ROUND_ONCE))

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
    def cov(cell, family):
        xs = [f(r, "accuracy") for r in fairc
              if r["cell"] == cell and r["family"] == family]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
        return sd / m, len(xs)

    def improved(cell, family):
        xs = [f(r, "delta") for r in fairc
              if r["cell"] == cell and r["family"] == family]
        return sum(1 for x in xs if x > 0), len(xs)

    for macro, cell, family, label in (("Control", "control", "parallel", "control"),
                                       ("Winner", "winner", "parallel", "winner"),
                                       ("Balanced", "balanced", "parallel", "balanced")):
        row = pick(fair, cell=cell, family=family)
        put("nFair%sMean" % macro, acc(f(row, "mean")),
            "fairness_c10d10.csv, row %s/%s, col mean" % (cell, family))
        put("nFair%sWorst" % macro, acc(f(row, "min")),
            "fairness_c10d10.csv, row %s/%s, col min" % (cell, family))
        c, n = cov(cell, family)
        put("nFair%sCoV" % macro, "%.3f" % c,
            "derived: fairness_c10d10_clients.csv, rows %s/%s, col accuracy,"
            " population sd / mean over the %d clients" % (cell, family, n))
        i, n = improved(cell, family)
        put("nFair%sImproved" % macro, "%d of %d" % (i, n),
            "fairness_c10d10_clients.csv, rows %s/%s, count of col delta > 0"
            % (cell, family))

    i, n = improved("control", "parallel")
    put("nFairControlLeftBehind", word(n - i),
        "fairness_c10d10_clients.csv, rows control/parallel, count of col delta <= 0")
    put("nWorstWriterId", code(worst["client"]),
        "fairness_c10d10_clients.csv, row with the smallest col shipped, col client")
    put("nWorstWriterShipped", acc(f(worst, "shipped")),
        "fairness_c10d10_clients.csv, row %s, col shipped" % worst["client"])

    # how far the worst-served client of each arm moved, in points
    put("nFairWorstMinGain", dpts(min(f(r, "lift_worst") for r in fair)),
        "fairness_c10d10.csv, col lift_worst, minimum over the %d arm rows,"
        " in points" % len(fair))
    put("nFairWorstMaxGain", dpts(max(f(r, "lift_worst") for r in fair)),
        "fairness_c10d10.csv, col lift_worst, maximum over the %d arm rows,"
        " in points" % len(fair))

    fbp = max(fair_combo, key=lambda r: f(r, "lift_worst"))
    crown_f = pick(fair_combo, cell=CROWNED[0], family=CROWNED[1])
    put("nFairBestPair", prose(fbp["cell"], agg_cells),
        "fairness_combo.csv, largest-lift_worst row (%s/%s), col cell"
        % (fbp["cell"], fbp["family"]))
    put("nFairBestPairWorstGain", dpts(f(fbp, "lift_worst") - f(crown_f, "lift_worst")),
        "derived: fairness_combo.csv, col lift_worst, row %s minus the crowned"
        " row %s, in points" % (fbp["cell"], CROWNED[0]))
    put("nFairBestPairMeanCost", pts(abs(100 * (f(crown_f, "mean") - f(fbp, "mean")))),
        "derived: fairness_combo.csv, col mean, |crowned row %s - row %s|,"
        " in points (the crowned pair is in fact the LOWER of the two here,"
        " by this amount)" % (CROWNED[0], fbp["cell"]))

    for macro, k in (("nBalancedWorstTwentyTwo", "c20d10"),
                     ("nBalancedWorstTwentyFour", "c20d20")):
        r = pick(fair20[k], cell="balanced", family="parallel")
        put(macro, signed(f(r, "lift_worst")),
            "fairness_%s.csv, row balanced/parallel, col lift_worst, in points"
            " (negative: below the shipped model)" % k)

    # ---- Section 6.7, cost ----------------------------------------------
    ca = pick(cost_a, arm="c10d10/winner")
    put("nParticipants", word(f(ca, "clients_per_round")),
        "cost_arms.csv, row c10d10/winner, col clients_per_round")
    put("nCommPerRound", "%.1f\\,MB" % f(ca, "mb_per_round"),
        "cost_arms.csv, row c10d10/winner, col mb_per_round")
    put("nCommTotal", "%.2f\\,GB" % (f(ca, "mb_per_round") * f(ca, "rounds") / 1000.0),
        "derived: cost_arms.csv, row c10d10/winner, cols mb_per_round x rounds")
    put("nParamCount", group(f(ca, "param_count")),
        "cost_arms.csv, row c10d10/winner, col param_count")

    # ---- Section 6.8, the extremes --------------------------------------
    for macro, cell in (("Double", "double"), ("Dual", "dual")):
        row = pick(extr, cell=cell)
        put("n%sAdapt" % macro, acc(f(row, "adaptation")),
            "extremes.csv, row %s, col adaptation" % cell)
        put("n%sPres" % macro, acc(f(row, "preservation")),
            "extremes.csv, row %s, col preservation" % cell)

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
    put("nDoubleFinalPres", acc(f(ex["double"], "preservation")),
        "extremes.csv, row double, col preservation (test axis; the trace"
        " value is stopping_extreme.csv row double col final_preservation"
        " = %s)" % acc(f(sx["double"], "final_preservation")))
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

    for macro, stage in (("Extreme", "extreme"), ("Five", "five"),
                         ("Agg", "aggfull")):
        put("nOneRule%sGain" % macro,
            dpts(mean("one_vs_final", lambda r, s=stage: r["stage"] == s)),
            "derived: mean of col one_vs_final over stopping_all.csv rows with"
            " stage = %s, in points" % stage)

    worst = min(stopa, key=lambda r: f(r, "one_vs_final"))
    put("nOneRuleWorstArm", prose(worst["cell"], agg_cells),
        "stopping_all.csv, row with the lowest col one_vs_final (stage %s,"
        " cell %s, %s)" % (worst["stage"], worst["cell"], worst["family"]))
    put("nOneRuleWorstLoss", dpts(-f(worst, "one_vs_final")),
        "derived: minus stopping_all.csv col one_vs_final of that row, in"
        " points (the sentence reads it as a cost)")


    put("nKlRuleScore", dpts(f(SIG["kl_global_to_current"],
                               "mean_stopped_score")),
        "signals_summary_extract.csv, row kl_global_to_current, col"
        " mean_stopped_score, in points")
    put("nProxyKlRuleScore", dpts(f(SIG["proxy_kl"], "mean_stopped_score")),
        "signals_summary_extract.csv, row proxy_kl, col mean_stopped_score,"
        " in points")

    put("nOracleGapMainMin", dpts(min(main_medians)),
        "derived: smallest per-stage MEDIAN of col oracle_cost over the seven"
        " non-extreme stages of stopping_all.csv, in points")
    put("nOracleGapMainMax", dpts(max(main_medians)),
        "derived: largest per-stage MEDIAN of col oracle_cost over the seven"
        " non-extreme stages of stopping_all.csv, in points")
    put("nOracleGapExtremeMin", dpts(extreme_gaps[0]),
        "derived: smallest col oracle_cost among the stage = extreme rows of"
        " stopping_all.csv, in points")
    put("nOracleGapExtremeMax", dpts(extreme_gaps[-1]),
        "derived: largest col oracle_cost among the stage = extreme rows of"
        " stopping_all.csv, in points")

    put("nRetentionMaxDev", acc(float(sigx["retention_max_dev_extreme"])),
        "signals_extras_extract.csv, key retention_max_dev_extreme --- the"
        " largest departure of retention_known from 1.0 over the three"
        " extreme arms")
    put("nRetentionSourceFall", dpts(float(sigx["source_max_fall_extreme"])),
        "signals_extras_extract.csv, key source_max_fall_extreme, in points")

    # ---- Section 7, the correlation table --------------------------------
    # median_rho is the per-run median rank correlation of the signal's DRIFT
    # against the source-validation drop; the extractor applies the signal's
    # own direction so that every entry is 'forgetting makes it grow'.
    put("nSignalCount", word(len(sigs)),
        "signals_summary_extract.csv, number of data rows --- the signals the"
        " runner records on every arm")
    put("nSignalRuns", group(float(sigx["study_runs"])),
        "signals_extras_extract.csv, key study_runs --- the run count the"
        " signals module reported for the study (of which %s carry a"
        " source-validation correlation)" % sigx["signal_runs_with_source_val"])
    put("nSignalCorrRuns", group(float(sigx["signal_runs_with_source_val"])),
        "signals_extras_extract.csv, key signal_runs_with_source_val --- the"
        " runs of that pass which carry a source-validation correlation, and"
        " so the population every rho in the table is a median over")
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

    # ---- Section 5, the protocol ----------------------------------------
    put("nBadPoolSize", group(float(cohort[0]["of"])),
        "cohort_table.csv, col 'of' (identical on every row)")
    put("nCohortSize", word(sum(1 for r in cohort_comp if r["cohort"] == "cohort10")),
        "cohort_composition.csv, count of rows with cohort = cohort10")
    put("nRounds", "%d" % int(f(pick(cost_s, stage="regularisation finals"), "rounds")),
        "cost_stages.csv, row 'regularisation finals', col rounds")

    # The decoupling example. The extract flags exactly one row: the writer
    # the detector ranked worst of the whole bad pool, which the shipped
    # model then scored at the top of that pool.  Both accuracies are read
    # from the flagged row, never restated by hand.
    flagged = [r for r in decouple if r["chosen"] == "yes"]
    if len(flagged) != 1:
        raise KeyError("decouple_example.csv must flag exactly one row, "
                       "found %d" % len(flagged))
    dec = flagged[0]
    put("nDecoupleGInit", acc(f(dec, "g_init_acc")),
        "decouple_example.csv, row %s (chosen), col g_init_acc" % dec["writer"])
    put("nDecoupleGZero", acc(f(dec, "g0_acc")),
        "decouple_example.csv, row %s (chosen), col g0_acc" % dec["writer"])

    folds = [c for c in cohort[0] if re.fullmatch(r"f\d+_val", c)]
    per_fold = [sum(float(r[c]) for r in cohort) for c in folds]
    put("nValRowsPerFold", "%d" % round(sum(per_fold) / len(per_fold)),
        "cohort_table.csv, cols %s summed over the cohort writers, mean over folds"
        % ", ".join(folds))

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
