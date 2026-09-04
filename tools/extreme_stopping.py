#!/usr/bin/env python3
"""
When the extreme cases should have stopped, and what the horizon cost them.

The two extreme arrangements are the smallest federations in this study: `dual`
is one client holding two writers' rows merged, and `double` is the same rows
with a client boundary between them, which makes it the only one of the pair
with a second update to average. What the server does at one client is not
aggregation - it is fine-tuning on the merged writer with a name borrowed from
federated learning - and the round series say so plainly. Both cases reach
their best trade in the first tens of rounds and then spend the rest of the
fixed hundred-round horizon taking the source model apart.

    python tools/extreme_stopping.py --root "$FOA_STUDY_DIR"
    python tools/extreme_stopping.py --root "$FOA_STUDY_DIR" \\
        --fig "$FOA_STUDY_DIR/figures/extreme_stopping.png"

THE STOPPING ROUND IS AN ORACLE, NOT A METHOD. It is the round that maximises
the study's own selection rule

    score = (adaptation - A0) - (P0 - preservation)

read off the VALIDATION columns - `pool_val_accuracies` against
`source_val_accuracies` - which is what a selection is allowed to see. Nothing
here proposes a stopping criterion and nothing here was used to choose one: the
number exists to say how much of the loss is the arrangement and how much is the
horizon. A0 and P0 come from the study's own shipped-model evaluations through
``report_tables.baselines``, so this tool and the reported tables measure against
the same two numbers rather than against two statements of them.

WHY RETENTION IS REPORTED BESIDE THE DROP. Every run carries a `retention_known`
signal, and on these cases it stays pinned at one while the source population
falls by up to seventeen points. A signal that reads 1.0 through a
collapse it is meant to detect is worth reporting as a finding, not as a
diagnostic: whatever `retention_known` measures on these runs, it is not what
the preservation column measures, and a reader who trusted it would have seen
nothing wrong at round 100.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_tables import baselines  # noqa: E402  - one definition of A0 and P0

from federated_outlier_adaptation.training.extreme_cells import (  # noqa: E402
    CASES as DEFINED_CASES,
)

#: In the order they are reported: worst trade first. `double` last because it
#: is the arrangement with a second update to average, and reading it after the
#: one that has none is what makes the mechanism visible.
#:
#: MEMBERSHIP IS THE CELL LIST'S, order is this file's. A reporting order is a
#: presentation choice and belongs here; which arrangements exist is a fact
#: about the stage and belongs where the stage is defined, so the two cannot
#: drift into naming different studies.
CASES = ("dual", "double")

#: What each arrangement is, for the table and the figure.
WHAT = {
    "dual": "one client holding both writers' rows merged",
    "double": "two clients: the cohort's worst two",
}

#: Reporting order and cell list must name the same study, or a case would be
#: reported that no stage defines - or one that does would be silently dropped.
assert set(CASES) == set(DEFINED_CASES), (CASES, DEFINED_CASES)

#: The round series every case stores. The two validation columns are what the
#: stopping rule reads; the other two are reported beside them.
SERIES = ("pool_val_accuracies", "source_val_accuracies",
          "retention_known", "dist_l2_to_global")

#: Rounds the trace table prints, before the oracle round is added to them. A
#: full hundred-row table per case buries the shape it exists to show.
MARKS = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100)


# --------------------------------------------------------------- pure helpers
def fold_mean(runs: Sequence[Sequence[float]]) -> List[float]:
    """
    The per-round mean across folds, truncated to the shortest fold.

    Truncated rather than padded or averaged over what is there: a round that
    only some folds reached is a mean over a different set of runs than the
    round before it, and plotting the two beside each other draws a step that
    is about which folds finished rather than about the training.
    """
    if not runs:
        raise ValueError("no folds to average")
    rounds = min(len(run) for run in runs)
    if rounds == 0:
        raise ValueError("a fold carries no rounds")
    return [st.mean(run[index] for run in runs) for index in range(rounds)]


def scores(adaptation: Sequence[float], preservation: Sequence[float],
           a0: float, p0: float) -> List[float]:
    """The study's selection rule, per round, on whatever columns it is given."""
    rounds = min(len(adaptation), len(preservation))
    return [(adaptation[i] - a0) - (p0 - preservation[i]) for i in range(rounds)]


def oracle_round(adaptation: Sequence[float], preservation: Sequence[float],
                 a0: float, p0: float) -> Tuple[int, float, float]:
    """
    The one-based round with the best score, that score, and the final one.

    Ties go to the EARLIEST round. A later round that merely equals an earlier
    one bought nothing and spent more of the source model getting there, so
    calling it the stopping point would overstate how long adaptation kept
    paying.
    """
    values = scores(adaptation, preservation, a0, p0)
    if not values:
        raise ValueError("no rounds to choose between")
    best = max(range(len(values)), key=lambda index: (values[index], -index))
    return best + 1, values[best], values[-1]


def retention_blindness(retention: Sequence[float],
                        preservation: Sequence[float]) -> Dict[str, float]:
    """
    How far the retention signal moved, against how far preservation fell.

    The deviation is measured from 1.0 rather than from the signal's own first
    round, because a retention signal is a claim about how much of the original
    behaviour survives and 1.0 is the claim that all of it did. Reported as a
    maximum over the run: a signal that dipped and recovered still noticed
    something, and one that never left 1.0 did not.
    """
    if not retention or not preservation:
        raise ValueError("both series are needed to compare them")
    return {
        "max_deviation": max(abs(1.0 - value) for value in retention),
        "final_deviation": abs(1.0 - retention[-1]),
        "source_drop": preservation[0] - preservation[-1],
    }


# ------------------------------------------------------------------- reading
def read_cases(root: Path, prefix: str) -> Dict[str, Dict[str, List[List[float]]]]:
    """``{case: {series: [per-fold list, ...]}}`` from the stored summaries."""
    found: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
        lambda: defaultdict(list))
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json",
                                 recursive=True)):
        folder = next(part for part in Path(path).parts
                      if part.startswith(prefix))
        case = folder[len(prefix):].split("_fold")[0]
        body = next(iter(json.loads(Path(path).read_text()).values()))
        for name in SERIES:
            values = [v for v in (body.get(name) or []) if v is not None]
            if values:
                found[case][name].append(values)
    return found


def traces(found, case: str) -> Optional[Dict[str, List[float]]]:
    """The four fold-mean series of one case, or None when it has not run."""
    stored = found.get(case)
    if not stored or not stored.get("pool_val_accuracies"):
        return None
    return {name: fold_mean(stored[name]) for name in SERIES if stored.get(name)}


# ------------------------------------------------------------------ printing
def report(case: str, mean: Dict[str, List[float]], folds: int,
           a0: float, p0: float) -> Dict[str, float]:
    """One case's trace table, its oracle stop, and its retention check."""
    adaptation = mean["pool_val_accuracies"]
    preservation = mean["source_val_accuracies"]
    rounds = min(len(adaptation), len(preservation))
    values = scores(adaptation, preservation, a0, p0)
    stop, stopped, final = oracle_round(adaptation, preservation, a0, p0)
    retention = mean.get("retention_known")
    distance = mean.get("dist_l2_to_global")

    print(f"\n{case.upper()} - {WHAT.get(case, '')}")
    print(f"  {folds} folds, {rounds} rounds, fold-mean traces on VALIDATION")
    header = f"{'round':>6}{'adapt':>10}{'source':>10}{'score':>9}"
    if retention:
        header += f"{'retention':>11}"
    if distance:
        header += f"{'l2 to g0':>10}"
    print("  " + header)
    print("  " + "-" * len(header))
    for index in sorted({m - 1 for m in MARKS if m <= rounds} | {stop - 1}):
        row = (f"{index + 1:>6}{adaptation[index]:>10.4f}"
               f"{preservation[index]:>10.4f}{values[index] * 100:>8.2f}p")
        if retention:
            row += f"{retention[index]:>11.4f}"
        if distance:
            row += f"{distance[index]:>10.3f}"
        print("  " + row + ("   <-- oracle stop" if index + 1 == stop else ""))

    print(f"  ORACLE STOP round {stop}: score {stopped * 100:>7.2f}p "
          f"(adapt {adaptation[stop - 1]:.4f}, source {preservation[stop - 1]:.4f})")
    print(f"  AT THE HORIZON round {rounds}: score {final * 100:>7.2f}p "
          f"(adapt {adaptation[rounds - 1]:.4f}, source {preservation[rounds - 1]:.4f})")
    print(f"  the horizon cost {(stopped - final) * 100:.2f} points of score, "
          f"and bought {(adaptation[rounds - 1] - adaptation[stop - 1]) * 100:+.2f} "
          "points of adaptation")

    summary = {"stop": stop, "stopped": stopped, "final": final,
               "rounds": rounds, "folds": folds}
    if retention:
        blind = retention_blindness(retention, preservation)
        summary.update(blind)
        print(f"  RETENTION IS BLIND TO IT: retention_known moves at most "
              f"{blind['max_deviation']:.4f} away from 1.0 "
              f"(ends {blind['final_deviation']:.4f} away) while the source "
              f"population falls {blind['source_drop'] * 100:.2f} points.")
    return summary


def figure(cases: Dict[str, Dict[str, List[float]]],
           stops: Dict[str, int], path: Path, a0: float, p0: float) -> None:
    """One panel per case: adaptation and preservation, with the oracle stop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [case for case in CASES if case in cases]
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 4.2),
                             sharey=True, squeeze=False)
    for axis, case in zip(axes[0], order):
        mean = cases[case]
        adaptation = mean["pool_val_accuracies"]
        preservation = mean["source_val_accuracies"]
        rounds = min(len(adaptation), len(preservation))
        stop = stops[case]
        axis.plot(range(1, rounds + 1), adaptation[:rounds], lw=2,
                  label="cohort (adaptation, val)")
        axis.plot(range(1, rounds + 1), preservation[:rounds], lw=2,
                  label="source population (val)")
        axis.axvline(stop, color="k", ls="--", lw=1)
        axis.annotate(
            f"oracle stop: round {stop}\n"
            f"{adaptation[stop - 1]:.3f} / {preservation[stop - 1]:.3f}",
            xy=(stop, preservation[stop - 1]), xytext=(stop + 6, 0.80),
            fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
        axis.set_title(f"{case}: {WHAT.get(case, '')}", fontsize=10)
        axis.set_xlabel("round")
        axis.set_ylim(0.70, 1.005)
        axis.grid(alpha=0.3)
    axes[0][0].set_ylabel("accuracy (fold mean)")
    axes[0][0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "Extreme cases: adaptation saturates within tens of rounds, and the "
        "fixed hundred-round horizon spends the rest on the source model",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"\n  -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--prefix", default="d01_extreme_",
                    help="Run-folder prefix the cases were written under.")
    ap.add_argument("--fig", type=Path, default=None,
                    help="Also write the per-case figure here.")
    args = ap.parse_args()

    a0, p0 = baselines(args.root)
    found = read_cases(args.root, args.prefix)
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print("basis: VALIDATION (pool_val_accuracies against source_val_accuracies), "
          "which is what a selection is allowed to read.")

    means, stops, summaries = {}, {}, {}
    for case in CASES:
        mean = traces(found, case)
        if mean is None:
            print(f"\n{case.upper()}\n  nothing on disk yet")
            continue
        folds = len(found[case]["pool_val_accuracies"])
        summaries[case] = report(case, mean, folds, a0, p0)
        means[case], stops[case] = mean, summaries[case]["stop"]

    if not summaries:
        print("\nno extreme case has run under this root.")
        return 1

    print("\nSTOPPED AGAINST FINAL")
    print(f"  {'case':<8}{'stop':>6}{'stopped':>10}{'final':>10}{'cost':>9}"
          f"{'retention off 1.0':>20}{'source drop':>14}")
    for case, row in summaries.items():
        print(f"  {case:<8}{row['stop']:>6}{row['stopped'] * 100:>9.2f}p"
              f"{row['final'] * 100:>9.2f}p{(row['stopped'] - row['final']) * 100:>8.2f}p"
              f"{row.get('max_deviation', float('nan')):>20.4f}"
              f"{row.get('source_drop', float('nan')) * 100:>13.2f}p")

    if args.fig:
        figure(means, stops, args.fig, a0, p0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
