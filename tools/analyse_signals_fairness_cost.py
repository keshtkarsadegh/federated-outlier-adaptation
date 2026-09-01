#!/usr/bin/env python3
"""
SUPERSEDED - see docs/STOPPING.md and the tools it names.

Its winner arm is ``d01_combo_trimmed_0p4_feature_l2_lam0p1_fold``, a cell of
the PRIOR programme that no folder under Digits_study01 carries, so the
fairness and cost halves resolve to nothing while the signals half still
reads whatever runs happen to be in the tree.

The three questions it asked are now answered separately and per stage:
``foa signals`` correlates each of the eight signals with both definitions of
forgetting, ``tools/stopping_table.py`` prices what a permitted signal would
have delivered, and ``tools/check_signals.py --all`` establishes that the
signals are present at all. ``docs/STOPPING.md`` reads all three. Kept as a
record, not repointed.

Signals, fairness and cost for Digits_study01, from the stored per-round series.

Nothing here retrains anything: every quantity is already written by the runner
into ``summary_0.json`` for every run of the study, once per round.

SIGNALS. The deployment cannot read the source population, so it cannot measure
forgetting. This asks whether any quantity it CAN read moves with forgetting.
For each run, each signal's *drift* from its own round-0 value is correlated
with the true source-accuracy *drop* from round 0, over the rounds of the run.
Correlating drifts rather than levels is what makes one budget mean the same
thing for signals on different scales.

FAIRNESS. The cohort accuracy is a mean over ten writers and a mean can rise
while its worst member does not move. Reported at the final round against each
writer's own accuracy under the shipped model.

COST. Seconds per round as measured by the runner, and bytes actually exchanged
per round: participants x parameters x 4 bytes x 2 directions.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

STUDY = Path(sys.argv[1])
OUT = STUDY / "tables" / "signals_fairness_cost.json"

SIGNALS = [
    ("dist_l2_to_global", "distance travelled from the shipped weights", "model only"),
    ("agreement_with_global", "agreement with the shipped model's predictions", "clients' held-out"),
    ("kl_global_to_current", "KL from the shipped model", "clients' held-out"),
    ("retention_known", "retention on what the shipped model got right", "clients' held-out"),
    ("proxy_acc", "accuracy on the public proxy set", "proxy"),
    ("proxy_kl", "KL from the shipped model on the proxy set", "proxy"),
]

FAMILIES = {
    "control": "d01_fl_global_fold",
    "winner": "d01_combo_trimmed_0p4_feature_l2_lam0p1_fold",
    "balanced": "d01_combo_anchor_0p03_ntd_b0p01_t0p5_fold",
}


def cells(path: Path):
    """Every measured cell of a summary file."""
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return
    for name, body in payload.items():
        if isinstance(body, dict) and "source_val_accuracies" in body:
            yield name, body


def spearman(a, b):
    """Rank correlation without a scipy dependency."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            share = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = share
            i = j + 1
        return r
    return pearson(rank(a), rank(b))


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return cov / (va ** 0.5 * vb ** 0.5)


# --------------------------------------------------------------------- gather
runs = []
for summary in sorted(STUDY.rglob("summary_0.json")):
    parent = summary.parents[2].name if len(summary.parents) > 2 else ""
    for cell, body in cells(summary):
        runs.append({"parent": parent, "cell": cell, "body": body})

print(f"runs with per-round series: {len(runs)}")

# -------------------------------------------------------------------- signals
per_signal = defaultdict(list)
usable = 0
for run in runs:
    body = run["body"]
    src = body.get("source_val_accuracies") or []
    if len(src) < 10 or src[0] is None:
        continue
    drop = [src[0] - x for x in src if x is not None]
    if max(drop) - min(drop) <= 0:
        continue
    usable += 1
    for key, _, _ in SIGNALS:
        series = body.get(key) or []
        if len(series) != len(src) or any(x is None for x in series):
            continue
        drift = [abs(x - series[0]) for x in series]
        rho = spearman(drift, drop)
        r = pearson(drift, drop)
        if rho is not None:
            per_signal[key].append((rho, r))

print(f"runs usable for the signal study: {usable}\n")
signal_rows = []
for key, label, where in SIGNALS:
    vals = per_signal.get(key, [])
    if not vals:
        signal_rows.append({"signal": key, "label": label, "observed_on": where,
                            "runs": 0, "spearman_median": None, "pearson_median": None})
        continue
    rhos = sorted(v[0] for v in vals)
    rs = sorted(v[1] for v in vals if v[1] is not None)
    signal_rows.append({
        "signal": key, "label": label, "observed_on": where, "runs": len(vals),
        "spearman_median": statistics.median(rhos),
        "spearman_min": rhos[0], "spearman_max": rhos[-1],
        "pearson_median": statistics.median(rs) if rs else None,
        "share_above_0p9": sum(1 for x in rhos if abs(x) >= 0.9) / len(rhos),
    })

print(f"{'signal':34s} {'observed on':18s} {'median rho':>11s} {'range':>18s} {'runs':>5s}")
print("-" * 92)
for row in sorted(signal_rows, key=lambda r: -(abs(r["spearman_median"] or 0))):
    if row["runs"] == 0:
        print(f"{row['signal']:34s} {row['observed_on']:18s} {'unavailable':>11s}")
        continue
    print(f"{row['signal']:34s} {row['observed_on']:18s} "
          f"{row['spearman_median']:>11.4f} "
          f"{row['spearman_min']:>8.3f}..{row['spearman_max']:<8.3f} {row['runs']:>5d}")

# ------------------------------------------------------------------- fairness
baseline = json.loads((STUDY / "tables" / "clients_acc_on_g0.json").read_text())["accuracies"]
fairness = {}
for family, prefix in FAMILIES.items():
    finals = defaultdict(list)
    for run in runs:
        if not run["parent"].startswith(prefix):
            continue
        series = run["body"].get("client_test_accuracies") or []
        if not series or not isinstance(series[-1], dict):
            continue
        for client, acc in series[-1].items():
            finals[client].append(acc)
    if not finals:
        continue
    mean_by_client = {c: statistics.fmean(v) for c, v in finals.items()}
    values = sorted(mean_by_client.values())
    improved = [c for c, a in mean_by_client.items() if a > baseline.get(c, 0)]
    mean = statistics.fmean(values)
    fairness[family] = {
        "clients": len(mean_by_client),
        "mean": mean,
        "worst": values[0],
        "best": values[-1],
        "cov": (statistics.pstdev(values) / mean) if mean else None,
        "improved_on_shipped": len(improved),
        "share_improved": len(improved) / len(mean_by_client),
        "worst_client": min(mean_by_client, key=mean_by_client.get),
        "per_client": {c: {"final": a, "shipped": baseline.get(c),
                           "delta": a - baseline.get(c, 0)}
                       for c, a in sorted(mean_by_client.items(),
                                          key=lambda kv: baseline.get(kv[0], 0))},
    }

print(f"\n{'arm':10s} {'mean':>8s} {'worst':>8s} {'CoV':>8s} {'improved':>10s}")
print("-" * 50)
for family, row in fairness.items():
    print(f"{family:10s} {row['mean']:8.4f} {row['worst']:8.4f} {row['cov']:8.4f} "
          f"{row['improved_on_shipped']:>4d}/{row['clients']:<5d}")

# ----------------------------------------------------------------------- cost
cost = {}
for family, prefix in FAMILIES.items():
    secs, comm, params = [], [], []
    for run in runs:
        if not run["parent"].startswith(prefix):
            continue
        body = run["body"]
        rs = [x for x in (body.get("round_seconds") or []) if x]
        if rs:
            secs.append(statistics.fmean(rs))
        cb = [x for x in (body.get("comm_bytes_per_round") or []) if x]
        if cb:
            comm.append(statistics.fmean(cb))
        if body.get("param_count"):
            params.append(body["param_count"])
    if not secs:
        continue
    rounds = 100
    cost[family] = {
        "runs": len(secs),
        "seconds_per_round": statistics.fmean(secs),
        "seconds_total": statistics.fmean(secs) * rounds,
        "mb_per_round": statistics.fmean(comm) / 1e6 if comm else None,
        "gb_total": statistics.fmean(comm) * rounds / 1e9 if comm else None,
        "param_count": params[0] if params else None,
    }

base = cost.get("control", {}).get("seconds_per_round")
for family, row in cost.items():
    row["overhead_vs_control"] = (row["seconds_per_round"] / base - 1) if base else None

print(f"\n{'arm':10s} {'s/round':>9s} {'s/run':>9s} {'MB/round':>10s} {'GB/run':>8s} {'overhead':>10s}")
print("-" * 62)
for family, row in cost.items():
    over = row["overhead_vs_control"]
    print(f"{family:10s} {row['seconds_per_round']:9.3f} {row['seconds_total']:9.1f} "
          f"{row['mb_per_round']:10.1f} {row['gb_total']:8.2f} "
          f"{(f'{over*100:+.1f}%' if over is not None else '-'):>10s}")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"signals": signal_rows, "fairness": fairness, "cost": cost,
                           "runs_scanned": len(runs), "runs_usable_for_signals": usable},
                          indent=2) + "\n")
print(f"\nwritten: {OUT}")
