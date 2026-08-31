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
#: Server step sizes eta_s.  One rule, several coefficients - see
#: ``con_delta_eta`` on why this is not several functions.
#:
#: Only downward from 1.0.  eta = 0.75 beat eta = 1.0 on BOTH axes on the first
#: screen, which is the full-size step overshooting rather than a trade, so the
#: over-relaxation the literature also tests is not swept here.
ETAS = (0.1, 0.3, 0.5, 0.6, 0.8, 0.95, 1.0)

#: Client weighting exponents q, screened at eta_s = 1: p_k proportional to
#: n_k ** q, so q = 1 is proportional and q = 0 uniform.
#:
#: The named schemes are points of this axis, not separate methods. `capped` at
#: the hardcoded 1/K clips every above-average client to exactly average, which
#: is within measurement noise of `uniform`; the two were reported as different
#: rules and were measuring one thing.  q = 1 is omitted because proportional
#: weights at eta_s = 1 is already the last cell of the eta row.
WEIGHT_EXPONENTS = (0.0, 0.25, 0.5, 0.75)

#: Clients the trimmed mean drops at EACH END per coordinate - a count, not a
#: fraction.  The fraction is quantised by the number of participants, so with
#: eight of them beta = 0.1 trims nobody and is the plain mean under another
#: name; and a fixed fraction discards one client of eight but four of twenty,
#: so it does not carry across federation sizes.  The emitter converts.
TRIM_COUNTS = (1, 2, 3)

#: Half-lives of the server anchor, as a multiple of the run length R.
#:
#: lambda_s removes a fraction of the CURRENT displacement from g-0 each round,
#: so the displacement decays geometrically and the coefficient is a half-life:
#: lambda_s = 1 - 2 ** (-1/h).  A fixed coefficient is therefore not a fixed
#: intervention - lambda_s = 0.05 halves the distance every quarter of a
#: 25-round run and every half of a 100-round one.  Sweeping the half-life
#: relative to R makes a setting mean the same thing at both horizons, and makes
#: a winner transfer to the other federation sizes.
#:
#: Nothing slower than 2R, where the anchor never acts inside the run.  Nothing
#: faster than R/8: at lambda_s = 1 the update collapses to g-0 plus one round
#: of learning, which does not preserve strongly, it fails to accumulate.
ANCHOR_HALFLIVES_R = (2.0, 1.0, 0.5, 0.25, 0.125)

#: FedAvgM server momenta.  0 is the no-momentum control of its own row.
#:
#: The momentum buffer is a SUM, so a constant update settles at Delta/(1-beta)
#: and beta multiplies the effective step by 1/(1-beta): beta = 0.9 with
#: eta = 1 is about eta = 10 without momentum, which is the whole of its
#: preservation collapse.  Above 0.9 the multiplier reaches 33x and 333x and the
#: buffer's memory outlasts the run, so the row stops there.
FEDAVGM_BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

#: FedAdam / FedYogi server learning rates: the grid of Reddi et al. (2021),
#: who introduced both.  Our first sweep ran half-decade steps from 1e-4 to 10;
#: lr = 10 is ten times beyond anything published and was the first FedYogi
#: winner, at the same adaptation as the cell now chosen and 2.2 points less
#: preservation.  In the tau-dominated regime behaviour is governed by eta/tau
#: rather than by either alone, so the extra resolution measured little.
FEDOPT_LRS = (0.001, 0.01, 0.1, 1.0)

#: FedAdam / FedYogi adaptivity constants tau.
#:
#: tau interpolates between two optimisers: where sqrt(v) >> tau every
#: coordinate moves +/- eta regardless of its gradient, and where tau >> sqrt(v)
#: the denominator is just tau and the rule is FedAvg with step eta/tau.  It is
#: the preservation knob of the pair.  Reddi et al. sweep down to 1e-8; we
#: stopped at 1e-5 and the boundary rule then spent three rounds crawling toward
#: the value that paper already recommends.
FEDOPT_TAUS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)

#: Retention of the cyclic schedule: the fraction of the model that began the
#: round which survives a full pass over the clients.
#:
#: seq_mix blends WEIGHTS - theta <- (1-alpha) theta + alpha theta_k - so an
#: earlier client's contribution is multiplied by (1-alpha) at every later
#: visit and r = (1-alpha)**K after a full cycle.  alpha's natural unit is
#: client visits, and there are K of them per round, so the same alpha erases
#: far more per round at twenty clients than at ten and would silently change
#: meaning when a winner is carried between sizes.  The emitter converts.
MIX_RETENTIONS = (0.7, 0.5, 0.3, 0.1, 0.03)

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
    for q in WEIGHT_EXPONENTS:
        cells.append(_cell(
            f"weight_q{_fmt(q)}", "concurrent", "con_delta_eta",
            f"client weighting p_k proportional to n_k**{q:g}, at eta_s=1",
            server_eta=1.0, weight_q=q,
        ))

    cells.append(_cell(
        "median", "concurrent", "con_delta_median",
        "coordinate-wise median of the client updates",
        server_eta=1.0,
    ))
    for count in TRIM_COUNTS:
        cells.append(_cell(
            f"trimmed_t{count}", "concurrent", "con_delta_trimmed_mean",
            f"coordinate-wise trimmed mean, {count} client(s) dropped each end",
            server_eta=1.0, trim_count=count,
        ))

    for halflife in ANCHOR_HALFLIVES_R:
        cells.append(_cell(
            f"anchor_h{_fmt(halflife)}", "concurrent", "con_delta_anchor_lam",
            f"server anchor to g-0, displacement halves every {halflife:g}R rounds",
            server_eta=1.0, anchor_halflife_r=halflife,
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
    for retention in MIX_RETENTIONS:
        cells.append(_cell(
            f"seq_mix_r{_fmt(retention)}", "sequential", "seq_mix_alpha",
            f"cyclic mixing, {retention:g} of the round's model survives a full cycle",
            mix_retention=retention,
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


def screen_cells() -> List[Dict[str, Any]]:
    """Every screening cell, in path order.  Ids are unique by construction."""
    cells = (
        concurrent_cells() + sequential_cells() + control_cells()
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
