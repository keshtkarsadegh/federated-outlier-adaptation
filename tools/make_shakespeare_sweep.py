"""
Stage A: does old-data retention recover when the protection is rescaled?

    python tools/make_shakespeare_sweep.py --jobs-dir "$FOA_PROJECT_DIR/jobs/v4"

Writes one task file: a login-node job, no GPU.

The question
------------
The three digit winners were carried to Shakespeare with their coefficients
untouched - that was the point, and it is what makes the transfer a transfer.
But a coefficient is a *rate*, and the two tasks are not the same size: a NIST
writer holds about a hundred images, a Shakespeare role holds thousands of
sequences, and the model is an LSTM rather than a CNN. A penalty tuned to hold a
CNN near g-0 over a hundred images may simply be too weak to hold an LSTM near
g-0 over thousands.

So this sweeps the **strength** of each winner's penalty around the transferred
value, and nothing else. Same aggregation rules, same horizon, same
participation, same fold, same evaluation. If retention recovers as the
coefficient rises, the transfer's weakness was a scale mismatch and not a
failure of the method - and that is a different sentence to write.

What is deliberately NOT here
-----------------------------
No selection. Ten runs, ten rows, and the table goes to the owner. A sweep that
picked its own winner would be a second screen wearing the first one's clothes,
and the whole value of the transfer claim is that nothing downstream of the
digit study chose these methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from federated_outlier_adaptation.training import agg_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import SHAKESPEARE_STUDY01 as CFG

import make_shakespeare_study as BASE

#: Seed block of this stage, clear of every range any study has emitted.
SEED_BLOCK = CFG.seed_base + 1000

#: The reference points, already run as ``s01_*``: the transferred coefficients.
#: They are the row every sweep row is read against, and they are not re-run.
REFERENCE = {
    "winner": ("trimmed_0p4", "feature_l2", {"lam": 0.1}),
    "balanced": ("anchor_0p03", "ntd", {"lam": 0.01, "T": 0.5}),
    "sequential": ("seq_order_shuffle", "feature_l2", {"lam": 0.01}),
}


def feature_cell(lam: float) -> Dict[str, Any]:
    """A feature-alignment penalty at an arbitrary strength."""
    return reg_cells._cell(
        f"feature_l2_lam{reg_cells._fmt(lam)}", "feature_l2", "feature_l2",
        f"MSE between penultimate representations, lambda={lam:g}",
        lam=lam,
    )


def ntd_cell(beta: float, tau: float = 0.5) -> Dict[str, Any]:
    """A not-true-distillation penalty at an arbitrary strength."""
    return reg_cells._cell(
        f"ntd_b{reg_cells._fmt(beta)}_t{reg_cells._fmt(tau)}", "ntd", "ntd",
        f"not-true distillation, beta={beta:g}, tau={tau:g}",
        lam=beta, T=tau,
    )


def anchor_cell(strength: float) -> Dict[str, Any]:
    """The server-side pull towards g-0, at an arbitrary strength."""
    return {
        "id": f"anchor_{reg_cells._fmt(strength)}",
        "path": "concurrent",
        "rule": "con_delta_anchor_lam",
        "flags": {"server_eta": 1.0, "server_anchor": strength},
        "note": f"server anchor to g-0, lambda_s={strength:g}",
    }


def _agg(cell_id: str) -> Dict[str, Any]:
    """A published aggregation cell, by id."""
    found = {cell["id"]: cell for cell in agg_cells.screen_cells()}
    return found[cell_id]


#: The sweep, as ``(family label, aggregation cell, penalty cell)``.
#:
#: Two axes are moved, one per winner-2 pair: the SERVER pull (``anchor``) and
#: the CLIENT penalty (``ntd`` beta). They are the two things holding that
#: configuration near g-0, and which of them is under-scaled is exactly the
#: question - so each is moved with the other held at its transferred value.
def sweep_cells() -> List[tuple]:
    rows: List[tuple] = []
    for lam in (0.316, 1.0, 3.16):
        rows.append(("winner", _agg("trimmed_0p4"), feature_cell(lam)))
    for strength in (0.1, 0.3):
        rows.append(("balanced", anchor_cell(strength), ntd_cell(0.01)))
    for beta in (0.03, 0.1):
        rows.append(("balanced", anchor_cell(0.03), ntd_cell(beta)))
    for lam in (0.316, 1.0, 3.16):
        rows.append(("sequential", _agg("seq_order_shuffle"), feature_cell(lam)))
    return rows


def lines() -> List[str]:
    out: List[str] = []
    for index, (_, agg, reg) in enumerate(sweep_cells()):
        extra = f"--aggregation {agg['rule']} --extended-aggregations"
        extra += " --outer-workers 1 --inner-workers 1"
        flags = SL._agg_flags(agg["flags"])
        if flags:
            extra += f" {flags}"
        penalty = SL._reg_set(reg)
        if penalty:
            extra += f" {penalty}"
        # The stage tag is "s01a", not "s01_a": these runs sit beside the s01
        # ones and must sort and glob as their own stage. BASE._run prefixes the
        # study tag with an underscore, so the join is corrected here rather
        # than by giving the study a second tag it does not have.
        parent = f"a_{agg['id']}_{reg['id']}"
        line = BASE._run(parent, reg["trainer"], extra).format(seed=SEED_BLOCK + index)
        out.append(line.replace(f" --parent {CFG.tag}_a_", f" --parent {CFG.tag}a_"))
    return out


def header(tasks: List[str]) -> List[str]:
    rows = sweep_cells()
    listed = [
        f"#   {label:10s} {agg['id']:18s} x {reg['id']}"
        for label, agg, reg in rows
    ]
    return [
        "# GENERATED, AND NOT AUTHORISED TO RUN.",
        "#",
        "# Shakespeare_study01 / stage A: STRENGTH SWEEP of the three winners.",
        "#",
        "# The digit winners transferred with their coefficients untouched, which",
        "# is what makes the transfer a transfer. But a coefficient is a rate,",
        "# and the tasks are not the same size: a NIST writer holds ~100 images,",
        "# a Shakespeare role holds thousands of sequences, and the model is an",
        "# LSTM. A penalty strong enough to hold a CNN near g-0 over a hundred",
        "# images may be too weak to hold an LSTM near g-0 over thousands.",
        "#",
        "# So this moves the PROTECTION STRENGTH and nothing else. Same rules,",
        "# same horizon (100 rounds), same participation (9 of 10), same fold 1,",
        "# same three-category evaluation.",
        "#",
        "# The reference row of each family already ran as s01_* and is NOT",
        "# re-run:",
        f"#   winner     trimmed_0p4        x feature_l2 lam=0.1",
        f"#   balanced   anchor_0p03        x ntd beta=0.01 tau=0.5",
        f"#   sequential seq_order_shuffle  x feature_l2 lam=0.01",
        "#",
        "# The sweep:",
    ] + listed + [
        "#",
        "# For the balanced pair BOTH of its holds are moved, one at a time: the",
        "# server pull (anchor) with beta at its transferred value, and the",
        "# client penalty (ntd beta) with the anchor at its transferred value.",
        "# Which of the two is under-scaled is the question.",
        "#",
        "# NO SELECTION. Ten runs, ten rows, and the table goes to the owner. A",
        "# sweep that picked its own winner would be a second screen wearing the",
        "# first one's clothes.",
        "#",
        f"# {len(tasks)} tasks. The steps are independent; any --array width.",
        "#",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--expect", type=int, default=10)
    args = parser.parse_args()

    tasks = lines()
    if len(tasks) != args.expect:
        print(f"FATAL: {len(tasks)} tasks, expected {args.expect}.")
        return 1
    out = Path(args.jobs_dir) / f"{CFG.tag}a_strength.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(header(tasks) + tasks) + "\n")
    print(f"wrote {out} with {len(tasks)} task lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
