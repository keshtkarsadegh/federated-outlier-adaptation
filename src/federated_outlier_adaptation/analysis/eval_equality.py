"""
Loader path against cache path: are the numbers the same, and how much faster?

The evaluation cache exists to make a round cheap, and it is only allowed to do
that if it changes nothing that is reported.  This module compares two result
trees produced by the same configuration - one run with ``--eval-path loader``,
one with ``--eval-path cache`` - and answers two questions per job:

*Are the series identical?*
    Every round-indexed series a run records is compared element by element and
    the largest absolute difference is reported.  For an identical pair of
    models the evaluation is deterministic, so the expected answer is exactly
    zero; a non-zero value that is tiny (order 1e-7) points at the batch size of
    the forward pass rather than at the cache, and anything larger points at a
    real difference.

*How much faster is it?*
    ``round_seconds`` and the run's wall clock are reported side by side,
    together with the ratio.

Run it as::

    python -m federated_outlier_adaptation.analysis.eval_equality \\
        --loader $P/results_speed/loader --cache $P/results_speed/cache

The comparison is keyed by the path of a result file relative to its root, so
the two trees must have been produced with the same parent folder, trainer and
seed - which is exactly what the two timing jobs do.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

#: Round-indexed series every run records; the ones that must match exactly.
COMPARED_SERIES = (
    "accuracies",
    "source_val_accuracies",
    "heldout_client_accuracies",
    "pool_val_accuracies",
    "pool_test_accuracies",
    "heldout_participant_accuracies",
    "retention_known",
    "agreement_with_global",
    "kl_global_to_current",
    "proxy_acc",
    "proxy_kl",
    "dist_l2_to_global",
    "dist_fisher_to_global",
    "dist_fisher_norm_to_global",
)

#: Per-client series: a mapping per round instead of a number per round.
COMPARED_MAPPINGS = (
    "client_accuracies",
    "client_heldout_accuracies",
    "client_val_accuracies",
    "client_test_accuracies",
)

#: Timing keys read from either the payload or its instrumentation block.
TIMING_KEYS = ("round_seconds",)


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return None


def result_files(root: Path) -> dict[str, dict]:
    """``{relative path: payload}`` of every per-job result file under a root."""
    root = Path(root)
    found: dict[str, dict] = {}
    for path in sorted(root.rglob("accuracies_*.json")):
        payload = _load(path)
        if isinstance(payload, dict):
            found[str(path.relative_to(root))] = payload
    return found


def _numbers(value: Any) -> list[Optional[float]]:
    """Flatten a series into plain floats; ``accuracies`` holds pairs."""
    flat: list[Optional[float]] = []
    for entry in value or []:
        if isinstance(entry, (list, tuple)):
            flat.extend(None if item is None else float(item) for item in entry)
        elif entry is None:
            flat.append(None)
        else:
            flat.append(float(entry))
    return flat


def compare_series(left: Any, right: Any) -> Optional[float]:
    """
    Largest absolute difference between two series, or ``None``.

    ``None`` means the two are not comparable at all - different lengths, or a
    value present on one side and absent on the other - which is a failure, not
    a rounding question, and is reported as such.
    """
    a, b = _numbers(left), _numbers(right)
    if len(a) != len(b):
        return None
    worst = 0.0
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if x is None or y is None:
            return None
        difference = abs(x - y)
        if math.isnan(difference):  # pragma: no cover - a NaN in a stored run
            return None
        worst = max(worst, difference)
    return worst


def compare_mappings(left: Any, right: Any) -> Optional[float]:
    """Largest absolute difference between two per-round per-client series."""
    left = left or []
    right = right or []
    if len(left) != len(right):
        return None
    worst = 0.0
    for a, b in zip(left, right):
        if set(a or {}) != set(b or {}):
            return None
        for key, value in (a or {}).items():
            worst = max(worst, abs(float(value) - float(b[key])))
    return worst


def _timing(payload: dict) -> dict[str, Any]:
    rounds = payload.get("round_seconds") or []
    total = float(sum(rounds)) if rounds else 0.0
    config = payload.get("config") or {}
    return {
        "rounds": len(rounds),
        "round_seconds_total": total,
        "round_seconds_mean": (total / len(rounds)) if rounds else None,
        "wall_seconds": config.get("wall_seconds"),
        "eval_path": (config.get("eval_path") or (payload.get("evaluation") or {}).get("eval_path")),
    }


def compare_runs(loader_payload: dict, cache_payload: dict) -> dict[str, Any]:
    """Differences and timings of one (loader, cache) pair of result files."""
    differences: dict[str, Optional[float]] = {}
    for key in COMPARED_SERIES:
        if key not in loader_payload and key not in cache_payload:
            continue
        differences[key] = compare_series(loader_payload.get(key), cache_payload.get(key))
    for key in COMPARED_MAPPINGS:
        if key not in loader_payload and key not in cache_payload:
            continue
        differences[key] = compare_mappings(loader_payload.get(key), cache_payload.get(key))

    comparable = [value for value in differences.values() if value is not None]
    loader_timing = _timing(loader_payload)
    cache_timing = _timing(cache_payload)
    speedup = None
    if cache_timing["round_seconds_mean"]:
        speedup = loader_timing["round_seconds_mean"] / cache_timing["round_seconds_mean"]
    return {
        "differences": differences,
        "max_abs_difference": max(comparable) if comparable else None,
        "incomparable": sorted(key for key, value in differences.items() if value is None),
        "loader": loader_timing,
        "cache": cache_timing,
        "round_seconds_speedup": speedup,
    }


def compare_roots(loader_root, cache_root) -> dict[str, Any]:
    """Compare every job the two result trees have in common."""
    loader_files = result_files(Path(loader_root))
    cache_files = result_files(Path(cache_root))
    shared = sorted(set(loader_files) & set(cache_files))

    jobs = {name: compare_runs(loader_files[name], cache_files[name]) for name in shared}
    worst = [job["max_abs_difference"] for job in jobs.values() if job["max_abs_difference"] is not None]
    loader_wall = sum(
        job["loader"]["round_seconds_total"] for job in jobs.values()
    )
    cache_wall = sum(job["cache"]["round_seconds_total"] for job in jobs.values())
    return {
        "loader_root": str(loader_root),
        "cache_root": str(cache_root),
        "jobs": jobs,
        "num_jobs": len(shared),
        "only_in_loader": sorted(set(loader_files) - set(cache_files)),
        "only_in_cache": sorted(set(cache_files) - set(loader_files)),
        "max_abs_difference": max(worst) if worst else None,
        "incomparable_jobs": sorted(name for name, job in jobs.items() if job["incomparable"]),
        "loader_round_seconds_total": loader_wall,
        "cache_round_seconds_total": cache_wall,
        "round_seconds_speedup": (loader_wall / cache_wall) if cache_wall else None,
    }


def format_table(summary: dict[str, Any]) -> str:
    """One line per job: rounds, mean seconds on both paths, speed-up, max diff."""
    header = (
        f"{'job':58s} {'rounds':>6s} {'loader s/round':>14s} "
        f"{'cache s/round':>13s} {'speedup':>8s} {'max|diff|':>10s}"
    )
    lines = [header, "-" * len(header)]
    for name, job in sorted(summary["jobs"].items()):
        loader_mean = job["loader"]["round_seconds_mean"]
        cache_mean = job["cache"]["round_seconds_mean"]
        speedup = job["round_seconds_speedup"]
        difference = job["max_abs_difference"]
        lines.append(
            f"{name[-58:]:58s} {job['loader']['rounds']:6d} "
            f"{(loader_mean if loader_mean is not None else float('nan')):14.3f} "
            f"{(cache_mean if cache_mean is not None else float('nan')):13.3f} "
            f"{(speedup if speedup is not None else float('nan')):8.2f} "
            f"{(difference if difference is not None else float('nan')):10.2e}"
        )
    lines.append("-" * len(header))
    total_speedup = summary["round_seconds_speedup"]
    lines.append(
        f"{'TOTAL':58s} {'':6s} "
        f"{summary['loader_round_seconds_total']:14.1f} "
        f"{summary['cache_round_seconds_total']:13.1f} "
        f"{(total_speedup if total_speedup is not None else float('nan')):8.2f} "
        f"{(summary['max_abs_difference'] if summary['max_abs_difference'] is not None else float('nan')):10.2e}"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a loader-path result tree with a cache-path one."
    )
    parser.add_argument("--loader", required=True, help="Result root of the --eval-path loader run.")
    parser.add_argument("--cache", required=True, help="Result root of the --eval-path cache run.")
    parser.add_argument("--out", default=None, help="Write the full comparison as JSON.")
    args = parser.parse_args(argv)

    summary = compare_roots(args.loader, args.cache)
    print(format_table(summary))
    print()
    print(
        json.dumps(
            {
                "num_jobs": summary["num_jobs"],
                "max_abs_difference": summary["max_abs_difference"],
                "incomparable_jobs": summary["incomparable_jobs"],
                "only_in_loader": summary["only_in_loader"],
                "only_in_cache": summary["only_in_cache"],
                "round_seconds_speedup": summary["round_seconds_speedup"],
            },
            indent=2,
        )
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
