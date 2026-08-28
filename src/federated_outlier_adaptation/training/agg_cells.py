"""
The aggregation-contribution screen: which server rule, at which coefficients.

Stage 6 of the v6 protocol asks a question stage 5 cannot: *how much of the
adaptation, and how much of the forgetting, is the aggregation rule's doing?*
Stage 5 fixed the rule at plain FedAvg and varied nothing.  This module is the
list of everything the rule could have been, as a flat table of **cells**.

A cell is one server rule plus the coefficients it runs at, with an id that says
what it is.  The protocol around it is stage 5's, unchanged and deliberately so:
the same twenty writers, sixteen of twenty per round, E=5, batch 64, the same
outlier fold book, the same winner g-0 as the only initialisation.  One thing
varies between cells, so a difference between two of them is the rule.

Screening, not deciding
-----------------------
The screen runs at **25 rounds**, not 100.  Two hundred cells at the full
horizon is a week of GPU time to answer a question whose shape is visible much
earlier, so the screen ranks the cells cheaply and a second, small pass runs the
winners at the real horizon.  Nothing here is a reported number: a 25-round
result is a *ranking signal*, and treating it as a result would be reading a
race at the quarter mark.

Duplicates are skipped, and said so
-----------------------------------
Several of the obvious grid points are algebraically the same update as another,
and running both would spend GPU time reproducing run-to-run noise and then
report it as two settings.  Each one is recorded in :data:`SKIPPED_CELLS` with
the cell it duplicates, so the omission is a documented decision rather than a
gap someone has to rediscover.
"""

from __future__ import annotations

from typing import Any, Dict, List

#: Rounds the screen runs for.  A ranking horizon, not a reporting one.
SCREEN_ROUNDS = 25

#: Rounds the winners are re-run at, once the screen has ranked them.
FULL_ROUNDS = 100

#: The three paths a cell belongs to.  Winners are picked per path, because a
#: parallel rule and a cyclic schedule are not competing for the same slot.
PATHS = ("concurrent", "sequential", "control")


def _cell(cell_id: str, path: str, rule: str, note: str, **flags: Any) -> Dict[str, Any]:
    """One screening cell: an id, the rule, and the coefficients it runs at."""
    return {"id": cell_id, "path": path, "rule": rule, "flags": dict(flags), "note": note}


def _fmt(value: float) -> str:
    """A float as a filename-safe token: ``0.05`` -> ``0p05``, ``1e-04`` -> ``1e-04``."""
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m").replace("+", "")


# --------------------------------------------------------------------------- #
# Concurrent: the server step, the weighting, robustness, the anchor, FedOpt
# --------------------------------------------------------------------------- #
#: Server step sizes eta_s.  One rule, five coefficients - see
#: ``con_delta_eta`` on why this is not five functions.
ETAS = (0.1, 0.25, 0.5, 0.75, 1.0)

#: Client weightings p_k, screened at eta_s = 1.
WEIGHTINGS = ("uniform", "capped")

#: Fractions the trimmed mean drops at each end per coordinate.
TRIM_FRACTIONS = (0.1, 0.2)

#: The extension of that row, added after the screen found its winner sitting
#: on the high edge of it.  These are APPENDED to the table rather than
#: inserted into the trim row, so every existing cell keeps its index - and
#: therefore its sampler seed - and a screen that has already been submitted
#: stays reproducible line for line.  The method grouping is by id prefix, so
#: the selector still sees all four as siblings and evaluates the widened range.
TRIM_FRACTIONS_EXT = (0.3, 0.4)

#: Server-anchor pulls towards the frozen g-0.
ANCHOR_LAMBDAS = (0.01, 0.03, 0.1, 0.3, 1.0)

#: FedAvgM server momenta.  0 is the no-momentum control of its own row.
FEDAVGM_BETAS = (0.0, 0.5, 0.7, 0.9, 0.97, 0.99, 0.997)

#: FedAdam / FedYogi server learning rates: half-decade steps, 1e-4 to 10.
FEDOPT_LRS = (
    1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2, 3.16e-2, 1e-1, 3.16e-1, 1.0, 3.16, 10.0,
)

#: FedAdam / FedYogi adaptivity constants tau: decade steps, 1e-5 to 1.
FEDOPT_TAUS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

#: Sequential mixing weights alpha of ``seq_mix_alpha``.
MIX_ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def concurrent_cells() -> List[Dict[str, Any]]:
    """Every parallel-schedule cell."""
    cells: List[Dict[str, Any]] = []

    for eta in ETAS:
        cells.append(_cell(
            f"eta_{_fmt(eta)}", "concurrent", "con_delta_eta",
            f"server step eta_s={eta:g}, proportional weights",
            server_eta=eta,
        ))

    # Weighting is screened at eta_s = 1 only; proportional at eta_s = 1 is
    # already the last cell of the eta row, so it is not repeated here.
    for weighting in WEIGHTINGS:
        cells.append(_cell(
            f"weight_{weighting}", "concurrent", "con_delta_eta",
            f"client weighting p_k={weighting} at eta_s=1",
            server_eta=1.0, weighting=weighting,
        ))

    cells.append(_cell(
        "median", "concurrent", "con_delta_median",
        "coordinate-wise median of the client updates",
        server_eta=1.0,
    ))
    for fraction in TRIM_FRACTIONS:
        cells.append(_cell(
            f"trimmed_{_fmt(fraction)}", "concurrent", "con_delta_trimmed_mean",
            f"coordinate-wise trimmed mean, {fraction:g} dropped each end",
            server_eta=1.0, trim_frac=fraction,
        ))

    for lam in ANCHOR_LAMBDAS:
        cells.append(_cell(
            f"anchor_{_fmt(lam)}", "concurrent", "con_delta_anchor_lam",
            f"server anchor to g-0, lambda_s={lam:g}",
            server_eta=1.0, server_anchor=lam,
        ))

    for beta in FEDAVGM_BETAS:
        cells.append(_cell(
            f"fedavgm_b{_fmt(beta)}", "concurrent", "con_delta_fedavgm",
            f"FedAvgM, server momentum beta_s={beta:g}",
            server_beta=beta,
        ))

    for rule, prefix in (("con_delta_fedadam", "fedadam"), ("con_delta_fedyogi", "fedyogi")):
        for lr in FEDOPT_LRS:
            for tau in FEDOPT_TAUS:
                cells.append(_cell(
                    f"{prefix}_lr{_fmt(lr)}_tau{_fmt(tau)}", "concurrent", rule,
                    f"{prefix}, server lr={lr:g}, tau={tau:g}",
                    server_lr=lr, server_tau=tau,
                ))

    return cells


def sequential_cells() -> List[Dict[str, Any]]:
    """Every cyclic-schedule cell."""
    cells = [
        _cell("seq_fedavg", "sequential", "seq_fedavg_update",
              "cyclic FedAvg: theta <- (1 - n_k/N) theta + (n_k/N) theta_k"),
        _cell("seq_equal", "sequential", "seq_equal_update",
              "cyclic equal weighting: alpha = 1/(i+1)"),
        _cell("seq_incremental", "sequential", "seq_incremental_update",
              "cyclic progressive weighting: alpha = i/(K+i)"),
        _cell("seq_delta_scaled", "sequential", "seq_delta_scaled",
              "cyclic scaled delta form"),
        _cell("seq_delta_capped", "sequential", "seq_delta_capped",
              "cyclic capped delta form"),
    ]
    for alpha in MIX_ALPHAS:
        cells.append(_cell(
            f"seq_mix_{_fmt(alpha)}", "sequential", "seq_mix_alpha",
            f"cyclic mixing, alpha={alpha:g}",
            seq_mix_alpha=alpha,
        ))
    # Order is screened on cyclic FedAvg alone; 'fixed' is the seq_fedavg cell.
    cells.append(_cell(
        "seq_order_shuffle", "sequential", "seq_fedavg_update",
        "cyclic FedAvg, participants shuffled each round",
        client_order="shuffle",
    ))
    return cells


def control_cells() -> List[Dict[str, Any]]:
    """
    The two reference rows.

    Plain FedAvg exists from stage 5 at a hundred rounds, but a hundred-round
    number cannot be compared with a twenty-five-round one, so it is screened
    again here at the screen's own horizon.  That is the whole point of a
    control: it is measured the way the things it controls for are measured.
    """
    return [
        _cell("control_fedavg", "control", "fedavg",
              "plain FedAvg, both families, at the screening horizon"),
        _cell("control_fedavg_earlystop", "control", "fedavg",
              "plain FedAvg with the oracle stop rule armed",
              stop_when_global_below_clients=True),
    ]


#: Grid points deliberately not run, with the cell each duplicates.
SKIPPED_CELLS = {
    "weight_proportional": "eta_1 - proportional weights at eta_s=1 is that cell",
    "seq_delta_fedavg_update": (
        "seq_fedavg - the delta form is the same update written on deltas"
    ),
    "seq_delta_progressive_update": (
        "seq_incremental - theta + alpha(theta_k - theta) is "
        "(1-alpha) theta + alpha theta_k at the same alpha"
    ),
    "seq_fixed_ratio_update": "seq_mix_0p3 - the fixed ratio is alpha = 0.3",
    "seq_order_fixed": "seq_fedavg - 'fixed' is the default order",
    "con_delta_eta025": "eta_0p25 - the fixed rule at the swept value",
    "con_delta_eta05": "eta_0p5 - the fixed rule at the swept value",
    "con_delta_eta1": "eta_1 - the fixed rule at the swept value",
    "con_delta_anchor_lam01": "anchor_0p1 - the fixed rule at the swept value",
    "con_delta_anchor_lam05": "anchor_0p5 is not in the screened lambda row",
}


def extension_cells() -> List[Dict[str, Any]]:
    """
    Cells added after a screen, appended so existing indices never move.

    A screen's sampler seed is a function of a cell's position in this table, so
    inserting a cell in the middle would silently re-seed every cell after it -
    and a screen that has already run would no longer be reproducible from the
    table that describes it.  Appending costs nothing: the selector groups by id
    prefix, not by position.
    """
    return [
        _cell(
            f"trimmed_{_fmt(fraction)}", "concurrent", "con_delta_trimmed_mean",
            f"coordinate-wise trimmed mean, {fraction:g} dropped each end "
            "(boundary extension)",
            server_eta=1.0, trim_frac=fraction,
        )
        for fraction in TRIM_FRACTIONS_EXT
    ]


def screen_cells() -> List[Dict[str, Any]]:
    """Every screening cell, in path order.  Ids are unique by construction."""
    cells = (
        concurrent_cells() + sequential_cells() + control_cells()
        + extension_cells()
    )
    seen = set()
    for cell in cells:
        if cell["id"] in seen:  # pragma: no cover - guarded by a test
            raise ValueError(f"Duplicate screening cell id {cell['id']!r}")
        seen.add(cell["id"])
    return cells


def cells_by_path() -> Dict[str, List[Dict[str, Any]]]:
    """The cells grouped by the path their winner is picked within."""
    grouped: Dict[str, List[Dict[str, Any]]] = {path: [] for path in PATHS}
    for cell in screen_cells():
        grouped[cell["path"]].append(cell)
    return grouped
