"""
The regularisation-contribution screen: which penalty, at which strength.

Stage 6 asked what the *server* rule is worth.  Stage 7 asks the other half of
the question: what the **client-side penalty** is worth.  Every cell here runs
stage 6's screening protocol - cohort of twenty, sixteen per round, E=5, batch
64, the outlier fold book, the winner g-0 as the only initialisation, 25 rounds
- under plain FedAvg, and changes only the term added to the client's loss.

The anchor is the frozen g-0 in every cell.  That is the whole premise: the
penalty measures distance from the model the study is trying not to forget.

Both schedules, one task
------------------------
Each line runs ``--aggregation fedavg``, which is two families - the parallel
schedule and the cyclic one - in a single task.  A penalty that helps the server
average and a penalty that helps a sequential walk are not the same finding, so
selection is done per family and the two are reported separately.

The grids, and where the numbers come from
------------------------------------------
``param_l2``    FedProx's mu, in FedProx's own convention (``--fedprox-convention``),
                so the values mean what they mean in the paper.  ``mu = 0`` is an
                exact no-op - the trainer returns zero penalty for any
                ``lam <= 0`` - which makes it this stage's no-penalty control
                rather than a near-miss of one.
``fisher``      EWC's lambda, resolved finely through the region the published
                work actually uses.  The Fisher diagonal is normalised to mean 1
                (``normalise_fisher=True``), so ``lam`` is a quantity with a
                meaning rather than a number absorbing the Fisher's own scale -
                which is why this row does not need the eight decades an
                un-normalised penalty would.  The earlier work swept
                ``[0.2, 1, 4, 8, 10, 20, 100, 200]``; this row covers that span
                and resolves it.
``fisher_scaled`` the same values, under the dynamic cap this codebase's
                ``EWCTrainer`` applies.  Not a duplicate of ``fisher``: the cap
                pins the penalty at half the cross-entropy once the client has
                drifted far enough, so ``lam`` only acts before saturation.  The
                rows are matched deliberately - if they swept different values a
                difference between them could not be attributed to the cap.
``logit_l2``    a half-decade row at a **fixed** temperature.  The space is a
                plain MSE between logits and does not read ``T`` at all; it is
                recorded anyway so the provenance says what was intended.
                Placed below 1: the earlier work swept
                ``[1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]`` and measured adaptation
                collapsing to 0.85 by ``lam = 0.5``, so everything at and above
                1.0 is a frozen model.  Its own winner sat on its bottom edge,
                so this row goes one step further down than theirs.
``feature_l2``  the representation-matching row.  The earlier work *declared*
                ``[1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]`` for it and never ran it -
                there is no ``feature_alignment_grid_search`` in those results -
                so this knob has no prior measurement anywhere and the row is
                wide and evenly filled rather than sparse.
``kd``          Hinton distillation over six temperatures and seven alphas.
                The trainer's objective is ``CE + lam * T^2 * KL``, while the
                literature's is ``alpha * CE + (1 - alpha) * T^2 * KL``; the two
                have the same minimiser when ``lam = (1 - alpha) / alpha``, so
                the grid is written in alpha and converted - see
                :func:`kd_lam_of_alpha`.  ``T`` and ``alpha`` **multiply**: the
                effective weight is ``lam * T^2``, so much of a square grid moves
                along lines of constant penalty.  The top of the T row is gone:
                the earlier work measured preservation saturating by ``T = 8``,
                with ``T = 50`` buying 0.0003 over it and costing adaptation.
                Note that ``alpha`` was declared swept in that work and never
                was - every executed cell is ``alpha = 0.95`` bar one - so
                ``BEST_KD_ALPHA`` was the only value run, not a selected one.
``ntd``         not-true distillation: beta over five decades, tau over the
                range the NTD paper explores plus one step up to overlap the kd
                row.  Its ``beta`` and ``tau`` couple exactly as kd's ``lam`` and
                ``T`` do.  The earlier work declared
                ``beta in [0.3, 1, 3] x tau in [1, 3]`` and never ran it either,
                so the method now leading this study has no prior measurement.
``kd+fisher``   deliberately **not** in this screen.  A blend of two penalties is
                only worth pricing once each one's own strength is known, so its
                three cells are emitted afterwards from the two winners by
                ``tools/study_emit.py reg-hybrid``, once per schedule: the two
                families select different halves, so the blend is a different
                penalty in each and gets its own cell ids and its own seeds.
                Those three cells sweep ``mix`` and nothing else, which is why
                the blend now has a screen of its own - :func:`blend_cells`,
                emitted by ``tools/make_digits_p21.py`` as a separate stage
                rather than folded in here, so this file stays the table that
                ran as ``s16_reg_screen3.txt``.

Where the Fisher comes from
---------------------------
The Fisher-weighted spaces need the diagonal of the *shipped* model, and this
protocol never wrote one under the name the artefact resolver looks for
(``global_results/fisher_g0``).  What exists is the per-fold diagonal each
``global-train`` run wrote next to its own checkpoint, so the cells point at the
winning fold's directory explicitly.  That is the right object - the Fisher of
g-0 on the data g-0 was trained on - and it costs no recomputation.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Rounds the screen runs for; the same ranking horizon stage 6 uses.
SCREEN_ROUNDS = 25

#: Rounds the winners are re-run at.
FULL_ROUNDS = 100

#: The two schedules a task's single ``--aggregation fedavg`` covers.  Winners
#: are chosen per family: a penalty that helps the average and one that helps a
#: sequential walk are different findings.
FAMILIES = ("concurrent", "sequential")

#: Anchor of every cell: the frozen shipped model.
ANCHOR = "frozen"

#: FedProx mu, in FedProx's convention.  0 is the exact no-op control.
PARAM_L2_MUS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)

#: EWC lambda.  The Fisher is normalised to mean 1, so this is a comparable
#: coefficient rather than a number absorbing the Fisher's scale; the row is
#: therefore resolved through 1-1000 instead of spread over eight decades.  It
#: contains the earlier work's own grid, [0.2, 1, 4, 8, 10, 20, 100, 200], and
#: 8.0 explicitly - that study's reported setting.
FISHER_LAMS = (
    0.1, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
)

#: Logit-matching lambda at a fixed temperature.  Placed below 1.0, where the
#: earlier work's sweep shows the live region: by lam = 0.5 adaptation had
#: collapsed to 0.85, barely above doing nothing.  Their winner sat on their
#: bottom edge (1e-3), so this row adds a step below it.  0.1 is kept because it
#: is their reported setting.
LOGIT_L2_LAMS = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

#: Temperature recorded for the logit row.  The space does not read it.
LOGIT_L2_T = 2.0

#: Feature-matching lambda.  ReLU activations are order 1, so squared
#: differences are far smaller than the logit case and the useful region sits
#: higher, not lower.  No prior measurement exists for this knob - the earlier
#: work declared a grid for it and never ran it - so the row is wide and evenly
#: filled, and it contains that declared grid.
FEATURE_L2_LAMS = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: Distillation temperatures and alphas; alpha is converted to the trainer's lam.
#: The T row stops at 8: the earlier work measured preservation saturating there,
#: with T = 50 buying 0.0003 over T = 8 and costing adaptation.  It reaches 0.25
#: because the first screen's winner sat on the low edge - what was a boundary
#: extension is now part of the row.  The alpha row is unchanged and full: it has
#: never actually been swept, here or in the earlier work.
KD_TEMPERATURES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
KD_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)

#: Not-true-distillation beta and tau.  beta gains a step at the strong end so
#: that end is not an edge; tau gains 8.0 so this row overlaps the kd row, which
#: measures the same knob in the same units.
NTD_BETAS = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
NTD_TAUS = (0.5, 1.0, 2.0, 3.0, 4.0, 8.0)

#: Blend weights of the kd+fisher hybrid, emitted after the screen.
HYBRID_MIXES = (0.25, 0.5, 0.75)

# --------------------------------------------------------------------------- #
# The blend's own grid
# --------------------------------------------------------------------------- #
#: The kd+fisher screen, written in ITS PARENTS' units.
#:
#: The three cells ``reg-hybrid`` emits are not a search.  They take ``lam`` and
#: ``T`` from the KD winner and sweep ``mix`` alone, so the blend is the one
#: penalty in this study whose coefficients were never screened - it was added
#: after the screen closed and then led both schedules.  This row gives it the
#: treatment every other penalty had: its own grid, at the ranking horizon, in
#: both schedules.
#:
#: ``lam`` IS NOT A FREE THIRD KNOB.  The objective is
#: ``lam * (mix * KD + (1 - mix) * Fisher)``, so the two halves' coefficients are
#: locked in the ratio ``mix : (1 - mix)`` and one ``lam`` cannot carry both
#: parents' rows at once.  It can carry one of them exactly, and the Fisher half
#: is the one that needs it: the emitted blends run their Fisher term at
#: ``lam * (1 - mix)``, which is 0.083 where the concurrent selection chose
#: ``lambda = 8``, two decades below the row EWC was screened over.  So the
#: strength axis is written as the EWC lambda and converted - the same choice
#: the kd row makes in writing itself in alpha, and for the same reason.
BLEND_EWC_LAMS = FISHER_LAMS
BLEND_TEMPERATURES = KD_TEMPERATURES
BLEND_MIXES = HYBRID_MIXES

# --------------------------------------------------------------------------- #
# Boundary extensions
# --------------------------------------------------------------------------- #
# There are none.  The first screen ran narrower rows and appended four KD cells
# and four NTD cells after finding winners on their edges.  Those probes are now
# inside the rows themselves - KD reaches 0.25, NTD's beta and tau both gained a
# step - so keeping them as appended cells would only duplicate ids.
#
# The rows above are also placed against the earlier work's measurements rather
# than around a guess, which is what made the edges in the first place: three of
# the seven rows had their winner on a boundary because the row had been centred
# on the wrong decade.  If a winner lands on an edge again the remedy is the same
# as on the aggregation side - extend that row and re-emit - but nothing is
# carried forward from the superseded screen.


def kd_lam_of_alpha(alpha: float) -> float:
    """
    The trainer's ``lam`` for a literature ``alpha``.

    ``DistillationTrainer`` minimises ``alpha * CE + (1 - alpha) * T^2 * KL``;
    :class:`AnchoredTrainer` minimises ``CE + lam * T^2 * KL``.  Dividing the
    first by ``alpha`` gives the second with ``lam = (1 - alpha) / alpha``, so
    the two have the same minimiser and the same gradient direction and differ
    only by a constant factor on the effective learning rate.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1]; got {alpha!r}")
    return (1.0 - alpha) / alpha


def blend_lam_of_ewc(lam_ewc: float, mix: float) -> float:
    """
    The blend's ``lam`` that puts ``lam_ewc`` on its Fisher half.

    :class:`AnchoredTrainer` computes ``lam * (mix * KD + (1 - mix) * Fisher)``,
    so the Fisher term's coefficient is ``lam * (1 - mix)``.  Setting
    ``lam = lam_ewc / (1 - mix)`` makes that coefficient the EWC lambda the
    ``fisher`` row was screened over, and the KD term then rides at
    ``lam_ewc * mix / (1 - mix)``.

    Writing the grid in a parent's units and converting is what
    :func:`kd_lam_of_alpha` already does, for the same reason: a cell should
    name the quantity whose range was argued for, not the number the trainer
    happens to take.
    """
    if not 0.0 < mix < 1.0:
        raise ValueError(f"mix must lie in (0, 1); got {mix!r}")
    return lam_ewc / (1.0 - mix)


def combo_tune_lam_mix(c_ewc: float, c_kd: float, mix: float) -> tuple:
    """
    The ``(lam, mix)`` that puts named coefficients on BOTH halves at once.

    :func:`blend_lam_of_ewc` writes the strength in one parent's units and lets
    the other half ride wherever the ratio puts it, which is the right move when
    ``mix`` is the axis being searched.  It is the wrong move when the two
    coefficients are themselves the axes: there the intended penalty is

        m * c_kd * KD + (1 - m) * c_ewc * Fisher

    with ``m`` the blend weight the owner asked for, and :class:`AnchoredTrainer`
    computes ``lam * (mix * KD + (1 - mix) * Fisher)``.  Matching the two term by
    term gives ``lam * mix = m * c_kd`` and ``lam * (1 - mix) = (1 - m) * c_ewc``,
    hence

        lam = m * c_kd + (1 - m) * c_ewc
        mix = m * c_kd / lam

    So the trainer's ``mix`` is NOT the owner's ``m`` except by coincidence: it is
    the KD half's share of the total penalty weight, and it moves with the two
    coefficients as well as with ``m``.  A grid that wrote ``m`` straight into
    ``mix`` would be sweeping a different object than the one it named, and
    nothing in the emitted line would say so - the line would carry a legal
    coefficient and run.

    THIS IS AN EXTENSION HELPER.  Nothing in the core programme calls it; the
    blend's own screen is written in EWC's units and uses
    :func:`blend_lam_of_ewc`.
    """
    if not 0.0 < mix < 1.0:
        raise ValueError(f"mix must lie in (0, 1); got {mix!r}")
    if c_ewc <= 0.0 or c_kd <= 0.0:
        raise ValueError(
            f"both coefficients must be positive; got c_ewc={c_ewc!r}, "
            f"c_kd={c_kd!r}"
        )
    lam = mix * c_kd + (1.0 - mix) * c_ewc
    return lam, mix * c_kd / lam


def _fmt(value: float) -> str:
    """A float as a filename-safe token."""
    return f"{value:g}".replace(".", "p").replace("-", "m").replace("+", "")


def _cell(
    cell_id: str,
    method: str,
    space: str,
    note: str,
    trainer: str = "AnchoredTrainer",
    fedprox: bool = False,
    needs_fisher: bool = False,
    **hypers: Any,
) -> Dict[str, Any]:
    """One screening cell: the penalty, its coefficients, and what it is."""
    return {
        "id": cell_id,
        "method": method,
        "space": space,
        "trainer": trainer,
        "hypers": dict(hypers),
        "fedprox": fedprox,
        "needs_fisher": needs_fisher,
        "note": note,
    }


def param_l2_cells() -> List[Dict[str, Any]]:
    return [
        _cell(
            f"param_l2_mu{_fmt(mu)}", "param_l2", "param_l2",
            f"FedProx proximal term, mu={mu:g}"
            + (" - the exact no-penalty control" if mu == 0 else ""),
            fedprox=True, lam=mu,
        )
        for mu in PARAM_L2_MUS
    ]


def fisher_cells() -> List[Dict[str, Any]]:
    cells = []
    for space, method in (("fisher", "fisher"), ("fisher_scaled", "fisher_scaled")):
        for lam in FISHER_LAMS:
            cells.append(_cell(
                f"{method}_lam{_fmt(lam)}", method, space,
                f"Fisher-weighted distance to g-0, lambda={lam:g}"
                + (", with the dynamic cap" if space == "fisher_scaled" else ""),
                needs_fisher=True, lam=lam,
            ))
    return cells


def logit_l2_cells() -> List[Dict[str, Any]]:
    return [
        _cell(
            f"logit_l2_lam{_fmt(lam)}", "logit_l2", "logit_l2",
            f"MSE between student and anchor logits, lambda={lam:g}, T fixed at "
            f"{LOGIT_L2_T:g} (the space does not read T)",
            lam=lam, T=LOGIT_L2_T,
        )
        for lam in LOGIT_L2_LAMS
    ]


def feature_l2_cells() -> List[Dict[str, Any]]:
    return [
        _cell(
            f"feature_l2_lam{_fmt(lam)}", "feature_l2", "feature_l2",
            f"MSE between penultimate representations, lambda={lam:g}",
            lam=lam,
        )
        for lam in FEATURE_L2_LAMS
    ]


def kd_cells() -> List[Dict[str, Any]]:
    cells = []
    for temperature in KD_TEMPERATURES:
        for alpha in KD_ALPHAS:
            lam = kd_lam_of_alpha(alpha)
            cells.append(_cell(
                f"kd_T{_fmt(temperature)}_a{_fmt(alpha)}", "kd", "kd",
                f"Hinton distillation from g-0, T={temperature:g}, "
                f"alpha={alpha:g} (lam={lam:.4g})",
                lam=lam, T=temperature,
            ))
    return cells


def ntd_cells() -> List[Dict[str, Any]]:
    cells = []
    for beta in NTD_BETAS:
        for tau in NTD_TAUS:
            cells.append(_cell(
                f"ntd_b{_fmt(beta)}_t{_fmt(tau)}", "ntd", "ntd",
                f"not-true distillation, beta={beta:g}, tau={tau:g}",
                lam=beta, T=tau,
            ))
    return cells


def blend_cells() -> List[Dict[str, Any]]:
    """
    The kd+fisher blend's own grid.  **Not** part of :func:`screen_cells`.

    Deliberately a separate catalogue rather than an eighth row of the screen:
    ``s16_reg_screen3.txt`` has run, and a row added to the table it was emitted
    from would change every seed after it and stop that file regenerating.
    """
    cells = []
    for lam_ewc in BLEND_EWC_LAMS:
        for temperature in BLEND_TEMPERATURES:
            for mix in BLEND_MIXES:
                lam = blend_lam_of_ewc(lam_ewc, mix)
                cells.append(_cell(
                    f"blend_lam{_fmt(lam_ewc)}_T{_fmt(temperature)}"
                    f"_mix{_fmt(mix)}",
                    "kd+fisher", "kd+fisher",
                    f"KD blended with the Fisher distance to g-0, EWC half at "
                    f"lambda={lam_ewc:g}, T={temperature:g}, mix={mix:g} "
                    f"(lam={lam:.6g})",
                    needs_fisher=True, lam=lam, T=temperature, mix=mix,
                ))
    seen = set()
    for cell in cells:
        if cell["id"] in seen:  # pragma: no cover - guarded by a test
            raise ValueError(f"Duplicate blend cell id {cell['id']!r}")
        seen.add(cell["id"])
    return cells


# --------------------------------------------------------------------------- #
# EXTENSION - the best rule and the best penalty, tuned together
# --------------------------------------------------------------------------- #
# NOT PART OF THE CORE PROGRAMME.  Everything above is a stage the study runs
# and reports; this is an extension bolted on afterwards, and it is kept out of
# :func:`screen_cells`, out of the shortlists and out of the combination cross
# for the same reason :func:`blend_cells` is: those tables have run, and a cell
# added to a catalogue a shipped file was emitted from moves every seed after it
# and stops that file regenerating.  The extension reads the core's records; the
# core never reads the extension's.
#
# WHAT IT ASKS.  The combination stage crossed shortlists at ONE setting each: a
# pair is a rule at its own winning coefficients beside a penalty at its own, and
# nothing in that cross ever moved the two together.  Neither half was chosen in
# the other's presence, so "the two do not compose" is measured only at the point
# where each was best alone.  This screens the joint grid of the pair that leads
# each schedule, at the ranking horizon.

#: The EWC half's coefficient.  Around the two values the study's own selections
#: landed on - lambda = 8 concurrent, lambda = 0.1 sequential - as they appear
#: once the blend's screen is read: that screen chose lambda_ewc = 0.1 in BOTH
#: schedules, the bottom of the row, so the live region is at the floor of
#: REG_GRID_RANGES.md section 2 rather than in its middle.  The row therefore
#: brackets 0.1 rather than reaching up to 8.
CTUNE_EWC_COEFFS = (0.05, 0.1, 0.3)

#: The KD half's coefficient, in the trainer's units.  The kd row is written in
#: alpha and emits lam = (1 - alpha) / alpha, so its two winners - alpha = 0.9
#: and alpha = 0.99 - are lam = 0.111 and lam = 0.0101.  This row brackets the
#: first and reaches down toward the second.
CTUNE_KD_COEFFS = (0.05, 0.11, 0.2)

#: Distillation temperature.  The two the kd row's own winners sat at, and the
#: two the blend's screen chose: T = 0.25 (sequential) and T = 2 (concurrent).
#: Not the full six-point row - this grid pays for four other axes, and the two
#: values are the measured ones rather than a bracket around a guess.
CTUNE_TEMPERATURES = (0.25, 2.0)

#: The owner's blend weight m: the share of the penalty the KD half carries
#: BEFORE the two coefficients are applied.  The same three points the emitted
#: blends swept, so this axis can be read against them.
CTUNE_MIXES = HYBRID_MIXES

#: The parallel schedule's rule knob: the server step of ``con_delta_eta``, which
#: is what ``eta_0p95`` is a setting of.  0.95 is the winner and the row brackets
#: it; 1.0 is the plain full step, so the row also says what the rule buys at
#: all.  The cyclic schedule's winner - ``seq_delta_capped`` - has no
#: coefficient, which is why that half of the grid is a third the size.
CTUNE_SERVER_ETAS = (0.9, 0.95, 1.0)

#: Which server rule each schedule's leader is, and the flags it carries.  Read
#: against ``agg_cells``: ``eta_*`` cells are ``con_delta_eta`` at a server step,
#: ``seq_delta_capped`` is a rule with no coefficient at all.
CTUNE_RULES = {
    "concurrent": ("con_delta_eta", "eta"),
    "sequential": ("seq_delta_capped", None),
}


def combo_tune_id(family: str, c_ewc: float, c_kd: float, temperature: float,
                  mix: float, server_eta: float = None) -> str:
    """
    A cell id that names the OWNER'S dials, not the trainer's coefficients.

    ``lam`` and the trainer's ``mix`` are both functions of all three penalty
    dials, so an id built from them would be unreadable against the grid it came
    from and two different dial settings could not be told apart by eye.  The id
    names what was swept; :func:`combo_tune_lam_mix` says what the trainer gets.

    The schedule is in the id by construction rather than by a tag: the parallel
    cells carry a server step and the cyclic ones cannot, because the rule they
    run has no coefficient.  So an ``_eta`` suffix is exactly the parallel cells,
    and no two schedules can name one folder.
    """
    identifier = (f"ctune_ewc{_fmt(c_ewc)}_kd{_fmt(c_kd)}"
                  f"_T{_fmt(temperature)}_mix{_fmt(mix)}")
    if server_eta is not None:
        identifier += f"_eta{_fmt(server_eta)}"
    return identifier


def combo_tune_cells() -> List[Dict[str, Any]]:
    """
    The joint grid of the pair that leads each schedule.  **Extension only.**

    One cell is a whole arm - a server rule at a coefficient AND a penalty at
    three - so it carries the aggregation cell it runs under beside the penalty
    it adds, and the emitted line names one rule and therefore produces one
    family's result.  That is the combination stage's shape, not the screens':
    a ``--aggregation fedavg`` line would run both schedules under a rule that
    belongs to one of them.

    ``dials`` is the owner's grid, kept beside ``hypers`` rather than folded into
    it.  The boundary report is about the grid that was searched, and every one
    of ``hypers``' three numbers is a function of all four dials - so asking
    whether ``lam`` sat at the end of its row would answer a question nobody
    asked and miss the four that were.

    DEDUPLICATION.  Two dial settings collide when they hand the trainer the
    same ``(lam, mix, T)`` under the same rule, and nothing in the run records
    would say so - two folders, two seeds, one experiment.  The check is on the
    coefficients rather than on the dials for exactly that reason.  It finds
    nothing on this grid (see :func:`combo_tune_duplicates`), and it is here so
    that a widened row cannot quietly pay twice for one cell.
    """
    cells: List[Dict[str, Any]] = []
    seen: Dict[tuple, str] = {}
    for family in FAMILIES:
        rule, knob = CTUNE_RULES[family]
        etas = CTUNE_SERVER_ETAS if knob == "eta" else (None,)
        for c_ewc in CTUNE_EWC_COEFFS:
            for c_kd in CTUNE_KD_COEFFS:
                for temperature in CTUNE_TEMPERATURES:
                    for mix in CTUNE_MIXES:
                        lam, lam_mix = combo_tune_lam_mix(c_ewc, c_kd, mix)
                        for eta in etas:
                            key = (family, eta, round(lam, 12),
                                   round(lam_mix, 12), temperature)
                            if key in seen:
                                continue
                            cell_id = combo_tune_id(family, c_ewc, c_kd,
                                                    temperature, mix, eta)
                            seen[key] = cell_id
                            cell = _cell(
                                cell_id, "kd+fisher", "kd+fisher",
                                f"the {family} leader's rule and penalty tuned "
                                f"together: c_ewc={c_ewc:g}, c_kd={c_kd:g}, "
                                f"T={temperature:g}, m={mix:g}"
                                + (f", eta_s={eta:g}" if eta is not None else "")
                                + f" (lam={lam:.6g}, mix={lam_mix:.6g})",
                                needs_fisher=True,
                                lam=lam, T=temperature, mix=lam_mix,
                            )
                            cell["family"] = family
                            cell["dials"] = {
                                "c_ewc": c_ewc, "c_kd": c_kd,
                                "T": temperature, "m": mix,
                            }
                            flags: Dict[str, Any] = {}
                            if eta is not None:
                                cell["dials"]["server_eta"] = eta
                                flags["server_eta"] = eta
                            cell["agg"] = {
                                "id": ("eta_" + _fmt(eta)) if eta is not None
                                      else rule,
                                "path": family,
                                "rule": rule,
                                "flags": flags,
                                "note": (f"server step eta_s={eta:g}"
                                         if eta is not None
                                         else "cyclic capped delta form"),
                            }
                            cells.append(cell)
    return cells


def combo_tune_duplicates() -> int:
    """
    How many dial settings the deduplication removed.  Zero on this grid.

    Reported rather than assumed: the two halves' weights are ``m * c_kd`` and
    ``(1 - m) * c_ewc``, and a collision needs both to repeat, which no pair of
    rows here manages.  A widened row could, and then the number in the README
    would move with the grid instead of standing as a claim nobody rechecked.
    """
    dials = (len(CTUNE_EWC_COEFFS) * len(CTUNE_KD_COEFFS)
             * len(CTUNE_TEMPERATURES) * len(CTUNE_MIXES))
    total = dials * (len(CTUNE_SERVER_ETAS) + 1)
    return total - len(combo_tune_cells())


def combo_tune_by_family() -> Dict[str, List[Dict[str, Any]]]:
    """The extension's cells, split by the schedule whose leader they tune."""
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in FAMILIES}
    for cell in combo_tune_cells():
        grouped[cell["family"]].append(cell)
    return grouped


def control_cell() -> Dict[str, Any]:
    """
    Plain FedAvg with no penalty at all and no anchored trainer instantiated.

    ``param_l2_mu0`` is already an exact no-op, so this is the second, redundant
    control on purpose: if the two disagree by more than run-to-run noise, the
    penalty machinery is doing something even when it is switched off, and that
    is worth finding out from a cheap cell rather than from a paper reviewer.
    """
    return _cell(
        "control_none", "control", "none",
        "plain FedAvg, no penalty, BaseTrainer - the redundant no-op control",
        trainer="BaseTrainer",
    )


def screen_cells() -> List[Dict[str, Any]]:
    """Every screening cell.  Ids are unique by construction."""
    cells = (
        param_l2_cells() + fisher_cells() + logit_l2_cells()
        + feature_l2_cells() + kd_cells() + ntd_cells()
    )
    seen = set()
    for cell in cells:
        if cell["id"] in seen:  # pragma: no cover - guarded by a test
            raise ValueError(f"Duplicate screening cell id {cell['id']!r}")
        seen.add(cell["id"])
    return cells


def methods() -> List[str]:
    """The screened methods, in table order - the unit selection ranks within."""
    order: List[str] = []
    for cell in screen_cells():
        if cell["method"] not in order:
            order.append(cell["method"])
    return order


def cells_by_method() -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {name: [] for name in methods()}
    for cell in screen_cells():
        grouped[cell["method"]].append(cell)
    return grouped


#: Grid points deliberately not run, with the reason.
SKIPPED_CELLS = {
    "kd+fisher": (
        "not in THIS screen - a blend is only worth pricing once each penalty's "
        "own strength is known; emitted from the kd and fisher winners by "
        "tools/study_emit.py reg-hybrid, once per schedule, and screened over "
        "its own grid afterwards by tools/make_digits_p21.py"
    ),
    "fisher_lam0": (
        "param_l2_mu0 - any space at lam <= 0 returns zero penalty, so every "
        "method's zero cell is the same run"
    ),
    "logit_l2 T row": (
        "the logit_l2 space is a plain MSE and does not read T, so sweeping it "
        "would be five identical runs per lambda"
    ),
}
