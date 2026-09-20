#!/usr/bin/env python3
"""The single authoritative translation from run-record identifiers to the
names the literature uses.

Every table and every figure of the manuscript prints method names from this
module and from nowhere else.  A cell identifier such as ``fedadam_lr0p001_
tau1em06`` is a key into the study's own run records; it is not a name a
reader of the field recognises, and no comparable published paper prints one.
This module turns each identifier into the published name of the method plus
its setting in the method's own symbol, e.g.

    fedadam_lr0p001_tau1em06  ->  FedAdam ($\\eta{=}10^{-3},\\tau{=}10^{-6}$)

Three renderings are offered, and each has one job:

``label(cell)``
    The full literature name with its setting.  Used wherever a row names one
    method and has room for its parameters --- the single-method tables.

``short(cell)``
    The same name with the parameters dropped.  Used where two names share a
    row (the combination table), where the axis of a figure is a name, and
    anywhere the setting is stated once in a caption or a footnote instead.

``params(cell)``
    The setting alone, as maths, for a table whose row already names the
    method --- the per-weight sensitivity grids.  Falls back to ``short`` for
    a cell that has no free parameter.

Parameters are read out of the identifier by rule, never tabulated by hand:
``lam0p003`` is $3\\times10^{-3}$ and ``tau1em06`` is $10^{-6}$ because of how
the runner spells numbers, so a new cell from the same grid translates without
this file being touched.

Standard library only; imported by make_paper_tables.py,
make_sensitivity_tables.py and the figure scripts.
"""

from __future__ import annotations

import re

__all__ = ["label", "short", "params", "has_params", "settings_note", "run_label",
           "REFERENCE", "SIGNAL", "SIGNAL_SHORT", "ARM", "ARM_SHORT",
           "EXTREME", "SCHEDULE", "family_label", "UntranslatedIdentifier"]


class UntranslatedIdentifier(KeyError):
    """An identifier the map does not cover.

    Raised rather than falling back to the raw identifier: a code id must
    never reach a page of the manuscript, so a gap in the map is a build
    failure, not a silently ugly table cell.
    """


# ---------------------------------------------------------------------------
# numbers: how the runner spells them, and how the field prints them
# ---------------------------------------------------------------------------

def _value(tok):
    """The number a parameter token encodes.

    ``0p95`` is 0.95, ``1em06`` is 1e-6, ``8`` is 8.  These are the runner's
    two spellings and there are no others.
    """
    m = re.fullmatch(r"(\d+)em(\d+)", tok)
    if m:
        return float(m.group(1)) * 10.0 ** (-int(m.group(2)))
    m = re.fullmatch(r"(\d+)(?:p(\d+))?", tok)
    if not m:
        raise UntranslatedIdentifier("unparsable parameter token %r" % tok)
    return float(m.group(1) + ("." + m.group(2) if m.group(2) else ""))


def _dec(tok):
    """A plain decimal: 0.95, 0.3, 2, 0.01."""
    v = _value(tok)
    if v == int(v):
        return "%d" % int(v)
    return ("%.10f" % v).rstrip("0")


def _sci(tok):
    """A coefficient, printed the way a coefficient is printed.

    Below 0.1 the field writes a power of ten ($10^{-4}$, $3\\times10^{-3}$);
    at 0.1 and above it writes the decimal.  The threshold is the convention,
    not a property of these particular grids.
    """
    v = _value(tok)
    if v == 0:
        return "0"
    if v >= 0.1:
        return _dec(tok)
    exp = 0
    m = v
    while m < 1.0 - 1e-12:
        m *= 10.0
        exp -= 1
    m = round(m, 6)
    if abs(m - 1.0) < 1e-9:
        return "10^{%d}" % exp
    mant = "%d" % int(m) if abs(m - int(m)) < 1e-9 else ("%g" % m)
    return "%s{\\times}10^{%d}" % (mant, exp)


def _math(parts):
    """Wrap ``k{=}v`` fragments as one inline formula."""
    return "$" + ",".join(parts) + "$"


# ---------------------------------------------------------------------------
# the rules.  Each entry is (regex, name, short name, parameter renderer).
# The renderer receives the regex's groups and returns the maths, or None for
# a method with no free parameter.
# ---------------------------------------------------------------------------

#: The coefficients each composite parentage inherited from the halves it was
#: built out of, as the construction stage recorded them --- ``parallel`` from
#: ``tables/p14_hybrid_construction.json`` (the distillation winner at T = 0.25
#: and alpha = 0.9) and ``cyclic`` from its ``_sequential`` counterpart (T = 2,
#: alpha = 0.99).  Lambda is that record's ``lam``, which is (1-alpha)/alpha
#: and so is not a round number; it is printed to three significant figures,
#: because alpha is the dial the sweep turned and lambda is what it implies.
#: The two entries are restated here rather than read at import time: this
#: module turns identifiers into names and opens no files.
COMPOSITE = {"parallel": ("0.111", "0.25"), "cyclic": ("0.0101", "2")}


def _rules():
    R = []

    def rule(pattern, name, shortname, render=None, suffix=None):
        R.append((re.compile(r"\A" + pattern + r"\Z"), name, shortname,
                  render, suffix))

    # -- server rules, parallel schedule ------------------------------------
    rule(r"control_fedavg", "FedAvg (control)", "FedAvg (control)")
    rule(r"control_fedavg_earlystop", "FedAvg + early stop", "FedAvg + early stop")
    rule(r"eta_([0-9p]+)", "FedAvg", "Server step",
         lambda g: _math([r"\eta_s{=}%s" % _dec(g[0])]))
    rule(r"anchor_h([0-9p]+)", "Server anchor", "Anchor",
         lambda g: _math([r"h{=}%sR" % _dec(g[0])]))
    rule(r"weight_q([0-9p]+)",
         lambda g: ("Uniform weighting" if _value(g[0]) == 0
                    else "Size weighting"),
         lambda g: ("Uniform weights" if _value(g[0]) == 0
                    else "Size weights"),
         lambda g: _math([r"q{=}%s" % _dec(g[0])]))
    rule(r"trimmed_t([0-9p]+)", "Trimmed mean", "Trimmed mean",
         lambda g: _math([r"t{=}%s" % _dec(g[0])]))
    rule(r"median", "Coordinate median", "Coordinate median")
    rule(r"fedavgm_b([0-9pem]+)", "FedAvgM", "FedAvgM",
         lambda g: _math([r"\beta{=}%s" % _dec(g[0])]))
    rule(r"fedadam_lr([0-9pem]+)_tau([0-9pem]+)", "FedAdam", "FedAdam",
         lambda g: _math([r"\eta{=}%s" % _sci(g[0]), r"\tau{=}%s" % _sci(g[1])]))
    rule(r"fedyogi_lr([0-9pem]+)_tau([0-9pem]+)", "FedYogi", "FedYogi",
         lambda g: _math([r"\eta{=}%s" % _sci(g[0]), r"\tau{=}%s" % _sci(g[1])]))

    # -- server rules, cyclic schedule --------------------------------------
    rule(r"seq_fedavg", "Cyclic FedAvg", "Cyclic FedAvg")
    rule(r"seq_delta_capped", r"Cyclic, capped $\delta$", "Cyclic capped")
    rule(r"seq_delta_scaled", r"Cyclic, scaled $\delta$", "Cyclic scaled")
    rule(r"seq_order_shuffle", "Cyclic, shuffled order", "Cyclic shuffled")
    rule(r"seq_equal", "Cyclic, equal mix", "Cyclic equal")
    rule(r"seq_incremental", "Cyclic, incremental mix", "Cyclic incremental")
    rule(r"seq_mix_r([0-9p]+)", "Cyclic mixing", "Cyclic mixing",
         lambda g: _math([r"r{=}%s" % _dec(g[0])]))

    # -- client penalties ---------------------------------------------------
    rule(r"param_l2_mu0",
         "No penalty (control)", "No penalty")
    rule(r"param_l2_mu([0-9pem]+)", "FedProx", "FedProx",
         lambda g: _math([r"\mu{=}%s" % _sci(g[0])]))
    rule(r"fisher_lam([0-9pem]+)", "EWC", "EWC",
         lambda g: _math([r"\lambda{=}%s" % _dec(g[0])]))
    rule(r"fisher_scaled_lam([0-9pem]+)", "EWC, loss-capped", "EWC capped",
         lambda g: _math([r"\lambda{=}%s" % _dec(g[0])]))
    rule(r"logit_l2_lam([0-9pem]+)", "Logit matching", "Logit matching",
         lambda g: _math([r"\lambda{=}%s" % _sci(g[0])]))
    rule(r"feature_l2_lam([0-9pem]+)", "Feature matching", "Feature matching",
         lambda g: _math([r"\lambda{=}%s" % _sci(g[0])]))
    rule(r"kd_T([0-9p]+)_a([0-9p]+)", "KD", "KD",
         lambda g: _math([r"T{=}%s" % _dec(g[0]), r"\alpha{=}%s" % _dec(g[1])]))
    rule(r"ntd_b([0-9p]+)_t([0-9p]+)", "FedNTD", "FedNTD",
         lambda g: _math([r"\beta{=}%s" % _dec(g[0]), r"\tau{=}%s" % _dec(g[1])]))

    # -- the composite penalty, and the screen that swept the same three
    # -- coefficients on its own.  THREE DIFFERENT ARMS USED TO READ "KD+EWC
    # -- blend" ON THE PAGE.  The construction stage took a schedule's
    # -- selected distillation and consolidation halves, INHERITED their
    # -- lambda and T, and swept the mix alone; the screen swept lambda, T and
    # -- m together and is a different arm built a different way.  So the
    # -- composite is named for the construction and the screen for the
    # -- screen, and which schedule's halves a composite is built from stays
    # -- part of its identity: the two parentages never share a name, in the
    # -- full form or in the short one.
    # -- A composite cell spells its mix and nothing else, because the mix is
    # -- the only dial that stage turned, so its lambda and T reach the page
    # -- from COMPOSITE above or they reach it nowhere at all.
    rule(r"hybrid_mix([0-9p]+)", "KD+EWC composite", "KD+EWC composite",
         lambda g: _math([r"\lambda{=}%s" % COMPOSITE["parallel"][0],
                          r"T{=}%s" % COMPOSITE["parallel"][1],
                          r"m{=}%s" % _dec(g[0])]))
    rule(r"hybrid_seq_mix([0-9p]+)",
         "KD+EWC composite", "KD+EWC composite, cyclic-tuned",
         lambda g: _math([r"\lambda{=}%s" % COMPOSITE["cyclic"][0],
                          r"T{=}%s" % COMPOSITE["cyclic"][1],
                          r"m{=}%s" % _dec(g[0])]), "cyclic-tuned")
    # -- the screen carries all three coefficients in the identifier itself,
    # -- which is the whole difference between it and the rows above.
    rule(r"blend_lam([0-9pem]+)_T([0-9p]+)_mix([0-9p]+)",
         "KD+EWC screened", "KD+EWC screened",
         lambda g: _math([r"\lambda{=}%s" % _dec(g[0]),
                          r"T{=}%s" % _dec(g[1]),
                          r"m{=}%s" % _dec(g[2])]))

    # -- the EXTENSION: the leading pair, tuned jointly.  One identifier names
    # -- a whole arm - a server rule AND the penalty beside it - because that
    # -- is what was swept, so the name has to carry both halves.  The dials
    # -- are the owner's, not the trainer's: lambda_E and lambda_K are the two
    # -- coefficients and m the blend weight, and the trainer's own mix is a
    # -- function of all three, so printing that instead would name a number
    # -- nobody chose.  A cell with a server step is a parallel cell and one
    # -- without is cyclic; that is the whole of the schedule marking, and it
    # -- is why the two get different names rather than a shared one.
    rule(r"ctune_ewc([0-9pem]+)_kd([0-9pem]+)_T([0-9p]+)_mix([0-9p]+)"
         r"(?:_eta([0-9p]+))?",
         lambda g: ("FedAvg + KD+EWC blend" if g[4]
                    else r"Cyclic, capped $\delta$ + KD+EWC blend"),
         lambda g: ("Server step + blend (tuned)" if g[4]
                    else "Cyclic capped + blend (tuned)"),
         lambda g: _math(
             ([r"\eta_s{=}%s" % _dec(g[4])] if g[4] else [])
             + [r"\lambda_E{=}%s" % _sci(g[0]), r"\lambda_K{=}%s" % _sci(g[1]),
                r"T{=}%s" % _dec(g[2]), r"m{=}%s" % _dec(g[3])]))

    # -- the SECOND joint grid: the pair the study SELECTED, tuned jointly.
    # -- Same shape as the rule above and a name of its own, because it is a
    # -- different pair: the rule half here is the head of the aggregation
    # -- shortlist's own VALIDATION ranking, which on the parallel schedule is
    # -- a different rule from the one its test order leads with.  The rule
    # -- knob is the anchor's half-life rather than a server step, and it is
    # -- printed in R because that is the unit the cell stores - the
    # -- coefficient the trainer receives is a function of the horizon too, so
    # -- printing it would name a number that changes between two runs of one
    # -- cell.  A cell with a half-life is parallel and one without is cyclic.
    rule(r"ctunesel_ewc([0-9pem]+)_kd([0-9pem]+)_T([0-9p]+)_mix([0-9p]+)"
         r"(?:_h([0-9p]+))?",
         lambda g: ("Server anchor + KD+EWC blend" if g[4]
                    else "Cyclic FedAvg + KD+EWC blend"),
         lambda g: ("Anchor + blend (tuned)" if g[4]
                    else "Cyclic FedAvg + blend (tuned)"),
         lambda g: _math(
             ([r"h{=}%sR" % _dec(g[4])] if g[4] else [])
             + [r"\lambda_E{=}%s" % _sci(g[0]), r"\lambda_K{=}%s" % _sci(g[1]),
                r"T{=}%s" % _dec(g[2]), r"m{=}%s" % _dec(g[3])]))

    # -- the carried arms and the extreme federations -----------------------
    rule(r"winner", "Anchor + FedNTD (crowned)", "Crowned")
    rule(r"balanced", "FedAvg + KD/EWC blend", "Balanced")
    rule(r"sequential", "Cyclic + FedNTD", "Cyclic arm")
    rule(r"control", "FedAvg (control)", "Control")
    rule(r"double", "Two writers", "Two writers")
    rule(r"dual", "Merged pair", "Merged pair")
    return R


RULES = _rules()

#: Identifiers of the server rules, longest first, so that a combination
#: identifier ``<rule>_<penalty>`` splits at the right underscore.
_AGG_IDS = None


def _agg_ids():
    """Every server-rule identifier the rule table can name, longest first.

    A combination identifier is a server rule and a penalty joined by an
    underscore, and both halves contain underscores of their own, so the split
    has to be tried against the known rules.  The set is derived from the rule
    table rather than restated.
    """
    global _AGG_IDS
    if _AGG_IDS is None:
        _AGG_IDS = ["control_fedavg_earlystop", "control_fedavg", "median",
                    "eta_", "anchor_h", "weight_q", "trimmed_t", "fedavgm_b",
                    "fedadam_lr", "fedyogi_lr", "seq_fedavg", "seq_delta_capped",
                    "seq_delta_scaled", "seq_order_shuffle", "seq_equal",
                    "seq_incremental", "seq_mix_r"]
    return _AGG_IDS


def _match(cell):
    """(full name, short name, setting, suffix) for a cell, or None."""
    for rx, name, shortname, render, suffix in RULES:
        m = rx.match(cell)
        if m:
            g = m.groups()
            n = name(g) if callable(name) else name
            s = shortname(g) if callable(shortname) else shortname
            p = render(g) if render else None
            return n, s, p, suffix
    return None


def _split_combo(cell):
    """Split ``<server rule>_<penalty>``, or return None.

    Both halves must themselves be nameable; a bare penalty identifier that
    happens to start like a rule cannot be mistaken for a combination because
    the remainder would not parse.
    """
    for prefix in sorted(_agg_ids(), key=len, reverse=True):
        head = prefix.rstrip("_")
        for i in range(len(cell), 0, -1):
            if not cell[:i].startswith(head):
                continue
            if i < len(cell) and cell[i] == "_":
                a, b = cell[:i], cell[i + 1:]
                if _match(a) and _match(b):
                    return a, b
    return None


# ---------------------------------------------------------------------------
# the three renderings
# ---------------------------------------------------------------------------

def label(cell):
    """The full literature name with its setting."""
    if cell in REFERENCE:
        return REFERENCE[cell]
    if cell in SIGNAL:
        return SIGNAL[cell]
    hit = _match(cell)
    if hit:
        name, _, p, suffix = hit
        inside = ", ".join(x for x in (p, suffix) if x)
        return "%s (%s)" % (name, inside) if inside else name
    pair = _split_combo(cell)
    if pair:
        return "%s + %s" % (short(pair[0]), short(pair[1]))
    raise UntranslatedIdentifier(cell)


def short(cell):
    """The name with the parameters dropped."""
    if cell in REFERENCE:
        return REFERENCE[cell]
    if cell in SIGNAL_SHORT:
        return SIGNAL_SHORT[cell]
    hit = _match(cell)
    if hit:
        return hit[1]
    pair = _split_combo(cell)
    if pair:
        return "%s + %s" % (short(pair[0]), short(pair[1]))
    raise UntranslatedIdentifier(cell)


def params(cell):
    """The setting alone, for a row that already names the method."""
    if cell in REFERENCE:
        return REFERENCE[cell]
    if cell in SIGNAL:
        return SIGNAL[cell]
    hit = _match(cell)
    if hit:
        return hit[2] if hit[2] else hit[1]
    pair = _split_combo(cell)
    if pair:
        return "%s + %s" % (params(pair[0]), params(pair[1]))
    raise UntranslatedIdentifier(cell)


#: The stages of the programme, as the run records prefix an arm identifier.
STAGES = ("aggfull", "regfull", "combo", "c10d10", "c20d10", "c20d20",
          "five", "extreme")


def run_label(arm, how=None):
    """Name a run-record arm identifier, ``<stage>[_<family>]/<cell>``.

    The cost view indexes its rows this way.  The stage is dropped --- the
    table that prints such a row says which stage it is in --- and what
    remains is a cell identifier like any other.
    """
    how = how or label
    head, sep, tail = arm.rpartition("/")
    if not sep:
        return how(arm)
    for st in STAGES:
        if head == st:
            return how(tail)
        if head.startswith(st + "_"):
            return how(head[len(st) + 1:] + "_" + tail)
    return how(tail)


def has_params(cell):
    hit = _match(cell)
    return bool(hit and hit[2])


def settings_note(items):
    """``short name setting; short name setting; ...`` for a table footnote.

    Lets a combination table carry short names in its rows and state every
    setting once underneath, in the order the caller lists them, with each
    distinct method named once.

    An item is a cell identifier, or a ``(cell, tag)`` pair.  The tag ---
    in practice the row's schedule --- is printed only where it is needed to
    tell two entries apart, which happens whenever one method is carried at
    two different settings in the same table.
    """
    entries = []
    for it in items:
        cell, tag = it if isinstance(it, tuple) else (it, None)
        for part in (_split_combo(cell) or (cell,)):
            key = (short(part), params(part) if has_params(part) else None)
            if (key, tag) not in entries:
                entries.append((key, tag))

    ambiguous = {name for name, _ in (k for k, _ in entries)
                 if len({s for (n, s), _ in entries if n == name}) > 1}
    out = []
    for (name, setting), tag in entries:
        item = "%s %s" % (name, setting) if setting else name
        if name in ambiguous and tag:
            item += " (%s)" % tag
        if item not in out:
            out.append(item)
    return "; ".join(out)


# ---------------------------------------------------------------------------
# the identifiers that are words rather than grids
# ---------------------------------------------------------------------------

#: The reference ladder, spelled by the view that produces it.
REFERENCE = {
    "isolated (scratch)": "Isolated, from scratch",
    "isolated (from g-0)": "Isolated, fine-tuned from $\\thg$",
    "centralized (scratch)": "Centralized, from scratch",
    "centralized (from g-0)": "Centralized, from $\\thg$ \\emph{(ceiling)}",
}

#: The eight quantities a server may watch, in the manuscript's notation.
SIGNAL = {
    "dist_l2_to_global": "$\\lVert\\tht-\\thg\\rVert_2$",
    "dist_fisher_to_global": "Fisher-weighted distance",
    "dist_fisher_norm_to_global": "Fisher distance, normalised",
    "retention_known": "Retention on what $\\thg$ knew",
    "agreement_with_global": "Agreement with $\\thg$",
    "kl_global_to_current": "KL from $\\thg$ (clients)",
    "proxy_acc": "Proxy-set accuracy",
    "proxy_kl": "KL from $\\thg$ (proxy)",
}

#: The same eight, short enough for a figure axis (plain text, no macros).
SIGNAL_SHORT = {
    "dist_l2_to_global": "$\\ell_2$ distance",
    "dist_fisher_to_global": "Fisher distance",
    "dist_fisher_norm_to_global": "Fisher distance, normalised",
    "retention_known": "retention",
    "agreement_with_global": "agreement",
    "kl_global_to_current": "KL (clients)",
    "proxy_acc": "proxy accuracy",
    "proxy_kl": "KL (proxy)",
}

#: The carried arms and the extreme federations, as label()/short() give them.
ARM = {k: label(k) for k in ("winner", "balanced", "sequential", "control")}
ARM_SHORT = {k: short(k) for k in ("winner", "balanced", "sequential", "control")}
EXTREME = {k: label(k) for k in ("double", "dual")}

#: The runner's word for a round shape, and the manuscript's.
SCHEDULE = {"concurrent": "parallel", "sequential": "cyclic"}

#: The method families of the sensitivity grids, whose rows are a whole grid
#: row rather than one cell.
FAMILY = {
    "anchor": "Server anchor",
    "eta": "Server step",
    "fedadam": "FedAdam",
    "fedavgm": "FedAvgM",
    "fedyogi": "FedYogi",
    "median": "Coordinate median",
    "trimmed": "Trimmed mean",
    "weight_q": "Uniform weighting",
    "seq_delta_capped": "Cyclic, capped $\\delta$",
    "seq_delta_scaled": "Cyclic, scaled $\\delta$",
    "seq_equal": "Cyclic, equal mix",
    "seq_fedavg": "Cyclic FedAvg",
    "seq_incremental": "Cyclic, incremental mix",
    "seq_mix": "Cyclic mixing",
    "seq_order": "Cyclic, shuffled order",
    "feature_l2": "Feature matching",
    "fisher": "EWC",
    "fisher_scaled": "EWC, loss-capped",
    "kd": "KD",
    "logit_l2": "Logit matching",
    "ntd": "FedNTD",
    "param_l2": "FedProx",
}


def family_label(method):
    """Name a ``<schedule>/<family>`` row of a sensitivity grid.

    The control rows name a cell, not a family, so they are translated as
    cells; a cyclic family already says so in its own name and takes no
    schedule prefix; every other family gets one, because the regularisation
    grid runs the same family on both schedules.
    """
    sched, _, fam = method.partition("/")
    if sched == "control":
        return label(fam)
    if fam not in FAMILY:
        raise UntranslatedIdentifier(method)
    name = FAMILY[fam]
    if fam.startswith("seq"):
        return name
    if sched == "concurrent":
        return "Parallel: %s" % name
    if sched == "sequential":
        return "Cyclic: %s" % name
    raise UntranslatedIdentifier(method)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    import os
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    DATA = os.path.join(HERE, "data")
    seen = []
    for name in sorted(os.listdir(DATA)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(DATA, name), newline="") as fh:
            for r in csv.DictReader(fh):
                for col in ("cell", "arm", "minus", "method", "signal"):
                    if r.get(col) and (col, r[col]) not in seen:
                        seen.append((col, r[col]))
                if r.get("method"):
                    for k, v in r.items():
                        if k.startswith("w=") and ("cell", v) not in seen:
                            seen.append(("cell", v))
    bad = 0
    for col, ident in seen:
        try:
            if col == "method":
                out = family_label(ident)
            elif col == "signal":
                out = SIGNAL[ident]
            elif "/" in ident:
                out = run_label(ident)
            else:
                out = "%s  |  %s  |  %s" % (label(ident), short(ident),
                                            params(ident))
        except (UntranslatedIdentifier, KeyError):
            out = "*** UNTRANSLATED ***"
            bad += 1
        print("%-34s %s" % (ident, out))
    print("\n%d identifiers, %d untranslated" % (len(seen), bad))
    sys.exit(1 if bad else 0)
