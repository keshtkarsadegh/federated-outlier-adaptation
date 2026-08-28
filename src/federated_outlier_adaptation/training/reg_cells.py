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
``fisher``      EWC's lambda over eight decades.  EWC's own reported values sit
                in the thousands, so a grid that stops at 100 would only see one
                side of the optimum.
``fisher_scaled`` the same eight, under the dynamic cap this codebase's
                ``EWCTrainer`` applies.  Not a duplicate of ``fisher``: the cap
                changes the penalty, not just its scale.
``logit_l2``    a half-decade row at a **fixed** temperature.  The space is a
                plain MSE between logits and does not read ``T`` at all; it is
                recorded anyway so the provenance says what was intended.
``feature_l2``  a decade row.  The evidence for representation-matching on this
                task is thin, so the row is short and wide rather than dense.
``kd``          Hinton distillation over seven temperatures and seven alphas.
                The trainer's objective is ``CE + lam * T^2 * KL``, while the
                literature's is ``alpha * CE + (1 - alpha) * T^2 * KL``; the two
                have the same minimiser when ``lam = (1 - alpha) / alpha``, so
                the grid is written in alpha and converted - see
                :func:`kd_lam_of_alpha`.
``ntd``         not-true distillation: beta over four decades, tau over the
                range the NTD paper explores.
``kd+fisher``   deliberately **not** in the screen.  A blend of two penalties is
                only worth pricing once each one's own strength is known, so its
                three cells are emitted afterwards from the two winners by
                ``tools/emit_stage7_hybrid.py``.

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

#: EWC lambda, eight decades.
FISHER_LAMS = (0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6)

#: Logit-matching lambda, half-decade steps, at a fixed temperature.
LOGIT_L2_LAMS = (0.1, 0.316, 1.0, 3.16, 10.0)

#: Temperature recorded for the logit row.  The space does not read it.
LOGIT_L2_T = 2.0

#: Feature-matching lambda, decade steps.
FEATURE_L2_LAMS = (1e-2, 0.1, 1.0, 10.0, 1e2)

#: Distillation temperatures and alphas; alpha is converted to the trainer's lam.
KD_TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0)
KD_ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99)

#: Not-true-distillation beta and tau.
NTD_BETAS = (0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0)
NTD_TAUS = (0.5, 1.0, 2.0, 3.0, 4.0)

#: Blend weights of the kd+fisher hybrid, emitted after the screen.
HYBRID_MIXES = (0.25, 0.5, 0.75)

# --------------------------------------------------------------------------- #
# Boundary extensions, added after the screen found winners on a row's edge
# --------------------------------------------------------------------------- #
# These are APPENDED to the table rather than inserted into their rows, so every
# existing cell keeps its index - and therefore its sampler seed - and a screen
# that has already been submitted stays reproducible line for line.  Method
# grouping is by the cell's own ``method`` field, so the selector still sees each
# extension as a sibling of the row it extends.

#: The KD temperature row ran [1 .. 50] and its winner sat on the LOW end, so
#: the row is extended downwards.  A full T x alpha extension would be 2 x 7 =
#: 14 cells for what is a probe of one edge; alpha's optimum was interior and
#: agreed across both families at 0.99, so only the top of the alpha row comes
#: along - 0.99 because that is where the winner is, and 0.95 to catch it
#: drifting as T falls.
KD_EXT_TEMPERATURES = (0.5, 0.25)
KD_EXT_ALPHAS = (0.95, 0.99)

#: The NTD tau row ran [0.5 .. 4] and the concurrent winner sat on its LOW end.
#: Extended downwards at the two betas that won or came close there.
NTD_EXT_TAUS = (0.25,)
NTD_EXT_TAU_BETAS = (0.01, 0.001)

#: The NTD beta row ran [0.001 .. 10] and the sequential winner sat on its LOW
#: end.  Extended downwards at the two lowest taus, which is where that winner
#: lives.  Disjoint from the tau extension above: no (beta, tau) pair repeats.
NTD_EXT_BETAS = (0.0001,)
NTD_EXT_BETA_TAUS = (0.5, 1.0)


def kd_extension_cells() -> List[Dict[str, Any]]:
    """The KD temperature row's downward extension."""
    return [
        _cell(
            f"kd_T{_fmt(temperature)}_a{_fmt(alpha)}", "kd", "kd",
            f"Hinton distillation from g-0, T={temperature:g}, alpha={alpha:g} "
            f"(lam={kd_lam_of_alpha(alpha):.4g}) - boundary extension",
            lam=kd_lam_of_alpha(alpha), T=temperature,
        )
        for temperature in KD_EXT_TEMPERATURES
        for alpha in KD_EXT_ALPHAS
    ]


def ntd_extension_cells() -> List[Dict[str, Any]]:
    """The NTD tau and beta rows' downward extensions, deduplicated."""
    cells, seen = [], set()
    pairs = [(beta, tau) for tau in NTD_EXT_TAUS for beta in NTD_EXT_TAU_BETAS]
    pairs += [(beta, tau) for beta in NTD_EXT_BETAS for tau in NTD_EXT_BETA_TAUS]
    for beta, tau in pairs:
        if (beta, tau) in seen:
            continue
        seen.add((beta, tau))
        cells.append(_cell(
            f"ntd_b{_fmt(beta)}_t{_fmt(tau)}", "ntd", "ntd",
            f"not-true distillation, beta={beta:g}, tau={tau:g} - boundary extension",
            lam=beta, T=tau,
        ))
    return cells


def extension_cells() -> List[Dict[str, Any]]:
    """
    Every cell added after the screen, appended so existing indices never move.

    A screening seed is a function of a cell's position in this table, so
    inserting one in the middle would silently re-seed every cell after it - and
    a screen that has already run would no longer be reproducible from the table
    that describes it.  Appending costs nothing: the selector groups by the
    cell's ``method``, not by where it sits.
    """
    return kd_extension_cells() + ntd_extension_cells()


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
        + extension_cells()
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
        "not screened - a blend is only worth pricing once each penalty's own "
        "strength is known; emitted from the kd and fisher winners by "
        "tools/emit_stage7_hybrid.py"
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
