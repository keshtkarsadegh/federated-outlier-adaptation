"""
Do the constraint-respecting signals tell us what the source data would?

Every run writes the signals of
:mod:`federated_outlier_adaptation.runners.forgetting_signals` next to its
accuracies.  This module reads a whole results tree and answers the three
questions that decide whether a signal is usable in place of the source split:

**(a) Does it track forgetting?**
    Pearson and Spearman correlation between each signal and the true
    forgetting, per run and pooled over all rounds of all runs, against both
    definitions of forgetting: the source validation drop and the source test
    drop.

**(b) Would stopping on it have worked?**
    For a signal ``s`` and a budget ``delta``, the simulated rule stops at the
    first round whose *drift* exceeds ``delta`` - the rise of a distance or the
    drop of an accuracy relative to the run's own round-0 value - and the
    resulting ``(adaptation, forgetting)`` pair is compared against two
    references: **oracle stopping** (the round with the highest source
    validation accuracy, which needs the data the constraint forbids) and the
    **fixed budget** (the last round, i.e. no stopping at all).

**(c) Would selecting on it have worked?**
    The configuration with the highest held-out adaptation among those whose
    signal stays within ``delta``, and the gap to the configuration an oracle
    would pick under a true-forgetting budget.

**Which runs?**
    Whatever the scanned root holds, unless a population manifest names them.
    The correlations and the selection pool across runs, so adding a stage to a
    study moves every median this module reports - which makes a pass taken
    before that stage unreproducible afterwards.  ``--population`` pins the
    pass to a named set of runs; see :class:`Population`.

Everything is read from stored JSON; nothing is retrained.  Outputs land in
``<root>/signals/``:

    signal_correlations.csv     per-run and pooled correlations
    signal_stopping.csv         one row per (run, signal, delta)
    signal_selection.csv        one row per (signal, delta) with the oracle gap
    signals_summary.json        the three tables plus the run inventory
    signals_pareto_<method>.png Pareto-style plot per method
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from federated_outlier_adaptation.runners.forgetting_signals import SIGNAL_KEYS

#: How a signal moves when the model forgets: ``+1`` it grows (a distance or a
#: divergence), ``-1`` it shrinks (an accuracy or an agreement rate).  The drift
#: of a signal is defined so that it grows with forgetting either way.
SIGNAL_DIRECTION: dict[str, int] = {
    "dist_l2_to_global": +1,
    "dist_fisher_to_global": +1,
    "dist_fisher_norm_to_global": +1,
    "retention_known": -1,
    "agreement_with_global": -1,
    "kl_global_to_current": +1,
    "proxy_acc": -1,
    "proxy_kl": +1,
}

#: Budgets the simulated stopping rules and the selection are swept over.  A
#: budget is read in the unit of the signal it is applied to: percentage points
#: for the accuracy-like signals, nats for the divergences, and parameter units
#: for the distances - which is why the grid spans three orders of magnitude
#: rather than being tuned per signal.  Override it with ``--deltas``.
DEFAULT_DELTAS: tuple[float, ...] = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

#: Rounds averaged at the end of a run, matching ``grid_search.selection``.
DEFAULT_WINDOW = 5

#: Default forgetting budget of the oracle selection, matching ``foa select``.
DEFAULT_EPS = 0.005


# ---------------------------------------------------------------- population
#: The manifest a study ships beside its records to say which runs one signals
#: pass was taken over.  Looked up by this name when ``--population`` is handed
#: a directory rather than a file.
POPULATION_FILE = "signals_population.txt"


@dataclass(frozen=True)
class Population:
    """
    The runs one pass is computed over, named rather than globbed.

    WHY A PASS HAS TO NAME ITS RUNS.  Every other reader in this repository is
    pointed at a *stage* - a folder prefix, a selection record, a task file -
    and is therefore blind to whatever else the root happens to hold.  This
    module pools over the whole tree, so its correlations are a property of the
    disk it ran against rather than of the study: a stage added afterwards
    re-weights every median and every share the manuscript prints, silently and
    without a line of code changing.  Naming the runs is what turns the pass
    back into a measurement that can be repeated.
    """

    names: frozenset
    path: Optional[Path] = None

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.names)

    @property
    def folders(self) -> frozenset:
        """The top-level run folders the named runs live under."""
        return frozenset(name.split("/", 1)[0] for name in self.names)


def load_population(path) -> Population:
    """
    Read a population manifest: one run name per line, ``#`` starts a comment.

    A run's name is its path under the study root - the run folder, the
    fold/seed directory, the scenario and the arm - which is exactly the
    identity ``signal_correlations.csv`` carries in its ``run`` column.  So a
    manifest can be read back off a pass that has already been taken, and a
    pass can be checked against the manifest it claims to be.
    """
    path = Path(path)
    if path.is_dir():
        path = path / POPULATION_FILE
    names = set()
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                names.add(line)
    if not names:
        raise ValueError(
            f"{path} names no run. A population of nothing is not a population."
        )
    return Population(frozenset(names), path)


def missing_runs(runs: Sequence["RunTrajectory"], population: Population) -> list:
    """Names the manifest asks for that the scanned tree did not hold."""
    return sorted(population.names - {run.name for run in runs})


# --------------------------------------------------------------- correlations
def _clean_pairs(xs: Sequence[Any], ys: Sequence[Any]) -> tuple[list[float], list[float]]:
    """Index-aligned pairs where both entries are finite numbers."""
    left: list[float] = []
    right: list[float] = []
    for x, y in zip(xs, ys):
        if isinstance(x, bool) or isinstance(y, bool):
            continue
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        left.append(float(x))
        right.append(float(y))
    return left, right


def pearson(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    """Pearson correlation, or ``None`` when it is undefined."""
    x, y = _clean_pairs(xs, ys)
    n = len(x)
    if n < 3:
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denominator <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def rank(values: Sequence[float]) -> list[float]:
    """Average ranks of ``values``, the ranking Spearman's rho is built on."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return ranks


def spearman(xs: Sequence[Any], ys: Sequence[Any]) -> Optional[float]:
    """Spearman rank correlation, or ``None`` when it is undefined."""
    x, y = _clean_pairs(xs, ys)
    if len(x) < 3:
        return None
    return pearson(rank(x), rank(y))


# ------------------------------------------------------------------ payloads
def _numeric(series) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for value in series or []:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append(None)
        elif not math.isfinite(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _tail_mean(series, window: int = DEFAULT_WINDOW) -> Optional[float]:
    values = [v for v in (series or []) if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values:
        return None
    tail = values[-window:]
    return sum(tail) / len(tail)


@dataclass
class RunTrajectory:
    """One run's round-by-round series, everything the analysis needs."""

    name: str
    source: str
    method: str
    config: dict = field(default_factory=dict)
    adaptation: list[Optional[float]] = field(default_factory=list)
    source_val: list[Optional[float]] = field(default_factory=list)
    source_test: list[Optional[float]] = field(default_factory=list)
    signals: dict[str, list[Optional[float]]] = field(default_factory=dict)

    @property
    def rounds(self) -> int:
        return len(self.adaptation)

    @property
    def usable(self) -> bool:
        """
        A trajectory needs two rounds and a source reference.

        Two is the smallest budget that has a decision to make at all - the
        stopping rules and the selection work from it - while the correlations
        need at least three points and skip a run that has fewer.
        """
        reference = _first(self.source_val)
        return self.rounds >= 2 and reference is not None

    def forgetting(self, on: str = "val") -> list[Optional[float]]:
        """
        True forgetting per round: the drop of a source split.

        Args:
            on: ``"val"`` (the default) uses the source validation split, the
                one a selection rule would read if it were allowed to;
                ``"test"`` uses the source test split, the reported number.
        """
        series = self.source_test if on == "test" else self.source_val
        reference = _first(series)
        if reference is None:
            return [None] * self.rounds
        return [
            None if value is None else reference - value
            for value in _pad(series, self.rounds)
        ]

    def drift(self, signal: str) -> list[Optional[float]]:
        """
        The signal's deviation from round 0, signed so forgetting makes it grow.

        A distance is reported as its rise and an accuracy as its drop, so a
        single budget ``delta`` means the same thing for every signal.  A signal
        that moves the *other* way in a round yields a negative drift there, and
        no budget fires on it.
        """
        series = _pad(self.signals.get(signal), self.rounds)
        reference = _first(series)
        if reference is None:
            return [None] * self.rounds
        sign = SIGNAL_DIRECTION.get(signal, 1)
        return [None if value is None else sign * (value - reference) for value in series]


def _first(series) -> Optional[float]:
    for value in series or []:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _pad(series, length: int) -> list[Optional[float]]:
    values = list(series or [])
    values = values[:length]
    return values + [None] * (length - len(values))


def _run_name(path: Path, root: Path, name: str) -> str:
    """
    A run's identity: where its file sits under the scanned root, plus the job.

    ``path.parent.name`` is the scenario folder - ``concurrent_delta`` or
    ``sequential_weights`` - and it is the same string under every arm of every
    stage.  A name built from it alone collapsed a whole study onto four labels:
    the per-run correlation rows could not be told apart, and the configuration
    the selection picked was reported as the name of a scenario folder.
    """
    try:
        parent = path.parent.relative_to(root)
    except ValueError:  # pragma: no cover - a file from outside the root
        parent = Path(path.parent.name)
    return f"{parent.as_posix()}/{name}"


def _method_of(config: dict, name: str) -> str:
    """Grouping key of a run: trainer plus scenario/metadata when recorded."""
    trainer = (config or {}).get("trainer")
    if not trainer:
        return name.split("/")[0] or "unknown"
    scenario = (config or {}).get("scenario")
    metadata = (config or {}).get("metadata")
    parts = [str(trainer)]
    if scenario:
        parts.append(str(scenario))
    if metadata:
        parts.append(str(metadata))
    return "_".join(parts)


def _trajectory_from_block(name: str, source: str, block: dict) -> Optional[RunTrajectory]:
    """Build a trajectory from a summary entry or a grid ``population`` block."""
    if not isinstance(block, dict):
        return None
    adaptation = _numeric(block.get("heldout_client_accuracies"))
    accuracies = block.get("accuracies") or []
    source_test = [
        float(pair[1])
        for pair in accuracies
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]
    if not adaptation:
        adaptation = [None] * len(source_test)
    signals = {key: _numeric(block.get(key)) for key in SIGNAL_KEYS}
    if not any(any(value is not None for value in series) for series in signals.values()):
        return None
    config = block.get("config") or {}
    return RunTrajectory(
        name=name,
        source=source,
        method=_method_of(config, name),
        config=config,
        adaptation=adaptation,
        source_val=_numeric(block.get("source_val_accuracies")),
        source_test=source_test,
        signals=signals,
    )


def _record_files(root: Path, population: Optional[Population], pattern: str):
    """
    Every stored record matching ``pattern``, in one order whatever is scanned.

    A restricted pass walks the folders its population names rather than the
    whole tree, which is what keeps it cheap on a root holding four times its
    runs.  Taking those folders in sorted order and sorting inside each
    reproduces the order a whole-tree walk gives, so no table depends on which
    of the two walks produced it.
    """
    if population is None:
        yield from sorted(root.rglob(pattern))
        return
    for folder in sorted(population.folders):
        directory = root / folder
        if directory.is_dir():
            yield from sorted(directory.rglob(pattern))


def collect_trajectories(
    root, population: Optional[Population] = None
) -> list[RunTrajectory]:
    """
    Read every run under ``root`` that carries the signal series.

    Both stored shapes are understood: ``summary_*.json`` written by the final
    and extreme runs, and ``config_points_*.json`` written by the sweeps, whose
    ``config`` blocks hold the same series under ``population``.

    Args:
        root: Results directory to scan, recursively.
        population: When given, the only runs read.  :class:`Population` says
            why a pass that pools over whatever is present is not repeatable.
    """
    root = Path(root)
    runs: list[RunTrajectory] = []

    for path in _record_files(root, population, "summary_*.json"):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            continue
        if not isinstance(data, dict):
            continue
        for name, block in data.items():
            run_name = _run_name(path, root, name)
            if population is not None and run_name not in population:
                continue
            if isinstance(block, dict) and "accuracies" not in block:
                # A summary entry carries every series except the round
                # accuracies themselves, which live in the per-job file next to
                # it.  Reading them back is what makes the source *test* drop
                # available as a second forgetting target.
                accuracies = _job_accuracies(path, name)
                if accuracies is not None:
                    block = {**block, "accuracies": accuracies}
            run = _trajectory_from_block(run_name, str(path), block)
            if run is not None:
                runs.append(run)

    for path in _record_files(root, population, "config_points_*.json"):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            continue
        configs = (data or {}).get("config", {})
        if not isinstance(configs, dict):
            continue
        accuracies = _sibling_accuracies(path)
        for name, block in configs.items():
            if not isinstance(block, dict):
                continue
            run_name = _run_name(path, root, name)
            if population is not None and run_name not in population:
                continue
            merged = dict(block.get("population") or {})
            merged["config"] = block
            if name in accuracies:
                merged.setdefault("accuracies", accuracies[name])
            run = _trajectory_from_block(run_name, str(path), merged)
            if run is not None:
                runs.append(run)

    return runs


def _job_accuracies(summary_path: Path, exp_name: str):
    """
    Round accuracies of one job, read from the file the driver wrote next to it.

    ``<parent>/summary_<i>.json`` is accompanied by
    ``<parent>/<exp_name>/accuracies_<i>.json``, which is the same payload plus
    the ``accuracies`` series.  Returns ``None`` when that file is absent.
    """
    index = summary_path.name[len("summary_") : -len(".json")]
    path = summary_path.parent / exp_name / f"accuracies_{index}.json"
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return None
    accuracies = payload.get("accuracies") if isinstance(payload, dict) else None
    return accuracies if isinstance(accuracies, list) else None


def _sibling_accuracies(config_path: Path) -> dict:
    """The ``accuracies_points_*.json`` written next to a sweep's config file."""
    index = config_path.name.replace("config_points_", "").replace(".json", "")
    sibling = config_path.parent / f"accuracies_points_{index}.json"
    if not sibling.is_file():
        return {}
    try:
        with open(sibling) as handle:
            data = json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return {}
    return data if isinstance(data, dict) else {}


# ------------------------------------------------------- (a) do they track it
#: The two definitions of true forgetting a signal is correlated against: the
#: source *validation* drop (what a selection rule would read if the constraint
#: allowed it) and the source *test* drop (the reported number).
FORGETTING_TARGETS = ("source_val", "source_test")


def correlation_table(runs: Iterable[RunTrajectory]) -> list[dict]:
    """
    Per-run and pooled correlations between each signal and true forgetting.

    Every signal is correlated against both definitions of forgetting, see
    :data:`FORGETTING_TARGETS`.

    Returns:
        list[dict]: rows with ``run``/``method``/``signal``/``target``/
        ``pearson``/``spearman``/``rounds``; the pooled rows carry
        ``run="__pooled__"``.
    """
    rows: list[dict] = []
    pooled: dict[tuple[str, str], tuple[list[float], list[float]]] = {
        (key, target): ([], []) for key in SIGNAL_KEYS for target in FORGETTING_TARGETS
    }

    for run in runs:
        if not run.usable:
            continue
        for target in FORGETTING_TARGETS:
            forgetting = run.forgetting(on="test" if target == "source_test" else "val")
            for signal in SIGNAL_KEYS:
                series = _pad(run.signals.get(signal), run.rounds)
                values, targets = _clean_pairs(series, forgetting)
                if len(values) < 3:
                    continue
                pooled[(signal, target)][0].extend(values)
                pooled[(signal, target)][1].extend(targets)
                rows.append(
                    {
                        "run": run.name,
                        "method": run.method,
                        "signal": signal,
                        "target": target,
                        "rounds": len(values),
                        "pearson": pearson(values, targets),
                        "spearman": spearman(values, targets),
                    }
                )

    for (signal, target), (values, targets) in pooled.items():
        if len(values) < 3:
            continue
        rows.append(
            {
                "run": "__pooled__",
                "method": "__all__",
                "signal": signal,
                "target": target,
                "rounds": len(values),
                "pearson": pearson(values, targets),
                "spearman": spearman(values, targets),
            }
        )
    return rows


# ---------------------------------------------------- (b) would stopping work
def stop_round(drift: Sequence[Optional[float]], delta: float) -> int:
    """
    First round whose drift exceeds ``delta``; the last round if none does.

    Round 0 is the untouched starting model and can never trigger, so the search
    starts at round 1.
    """
    for index in range(1, len(drift)):
        value = drift[index]
        if value is not None and value > delta:
            return index
    return max(len(drift) - 1, 0)


def oracle_round(run: RunTrajectory) -> int:
    """The round with the highest source validation accuracy (needs the data)."""
    best_index, best_value = 0, None
    for index, value in enumerate(_pad(run.source_val, run.rounds)):
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_index, best_value = index, value
    return best_index


def _at(run: RunTrajectory, index: int) -> dict:
    adaptation = _pad(run.adaptation, run.rounds)
    forgetting = run.forgetting()
    if not (0 <= index < run.rounds):  # pragma: no cover - guard
        return {"round": None, "adaptation": None, "forgetting": None}
    return {
        "round": index,
        "adaptation": adaptation[index],
        "forgetting": forgetting[index],
    }


def stopping_table(
    runs: Iterable[RunTrajectory], deltas: Sequence[float] = DEFAULT_DELTAS
) -> list[dict]:
    """
    Simulate every ``(signal, delta)`` stopping rule against the two references.

    Returns:
        list[dict]: one row per ``(run, signal, delta)``, carrying the stop
        round and its ``(adaptation, forgetting)`` pair next to the oracle stop
        and the fixed budget, plus the two gaps to the oracle.
    """
    rows: list[dict] = []
    for run in runs:
        if not run.usable:
            continue
        oracle = _at(run, oracle_round(run))
        fixed = _at(run, run.rounds - 1)
        for signal in SIGNAL_KEYS:
            drift = run.drift(signal)
            if all(value is None for value in drift):
                continue
            for delta in deltas:
                stopped = _at(run, stop_round(drift, delta))
                rows.append(
                    {
                        "run": run.name,
                        "method": run.method,
                        "signal": signal,
                        "delta": delta,
                        "stop_round": stopped["round"],
                        "adaptation": stopped["adaptation"],
                        "forgetting": stopped["forgetting"],
                        "oracle_round": oracle["round"],
                        "oracle_adaptation": oracle["adaptation"],
                        "oracle_forgetting": oracle["forgetting"],
                        "fixed_round": fixed["round"],
                        "fixed_adaptation": fixed["adaptation"],
                        "fixed_forgetting": fixed["forgetting"],
                        "adaptation_gap": _gap(oracle["adaptation"], stopped["adaptation"]),
                        "forgetting_gap": _gap(stopped["forgetting"], oracle["forgetting"]),
                        "rounds": run.rounds,
                    }
                )
    return rows


def _gap(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


# --------------------------------------------------- (c) would selection work
def selection_table(
    runs: Iterable[RunTrajectory],
    deltas: Sequence[float] = DEFAULT_DELTAS,
    eps: float = DEFAULT_EPS,
    window: int = DEFAULT_WINDOW,
) -> list[dict]:
    """
    Pick a configuration on each signal and measure the gap to the oracle pick.

    The signal-based rule maximises the held-out adaptation of the last
    ``window`` rounds subject to the signal's final drift staying within
    ``delta``.  The oracle rule is the one ``foa select`` implements: the same
    objective subject to the *true* forgetting staying within ``eps``.
    """
    candidates = [run for run in runs if run.usable]
    summaries = []
    for run in candidates:
        summaries.append(
            {
                "run": run,
                "adaptation": _tail_mean(run.adaptation, window),
                "forgetting": _tail_mean(run.forgetting(), window),
                "drift": {
                    signal: _tail_mean(run.drift(signal), window) for signal in SIGNAL_KEYS
                },
            }
        )

    feasible = [
        entry
        for entry in summaries
        if entry["adaptation"] is not None
        and entry["forgetting"] is not None
        and entry["forgetting"] <= eps
    ]
    oracle = max(feasible, key=lambda e: e["adaptation"], default=None)

    rows: list[dict] = []
    for signal in SIGNAL_KEYS:
        for delta in deltas:
            allowed = [
                entry
                for entry in summaries
                if entry["adaptation"] is not None
                and entry["drift"][signal] is not None
                and entry["drift"][signal] <= delta
            ]
            picked = max(allowed, key=lambda e: e["adaptation"], default=None)
            rows.append(
                {
                    "signal": signal,
                    "delta": delta,
                    "eps": eps,
                    "num_candidates": len(summaries),
                    "num_allowed": len(allowed),
                    "selected": None if picked is None else picked["run"].name,
                    "selected_adaptation": None if picked is None else picked["adaptation"],
                    "selected_forgetting": None if picked is None else picked["forgetting"],
                    "oracle": None if oracle is None else oracle["run"].name,
                    "oracle_adaptation": None if oracle is None else oracle["adaptation"],
                    "oracle_forgetting": None if oracle is None else oracle["forgetting"],
                    "adaptation_gap": _gap(
                        None if oracle is None else oracle["adaptation"],
                        None if picked is None else picked["adaptation"],
                    ),
                    "forgetting_excess": (
                        None
                        if picked is None or picked["forgetting"] is None
                        else picked["forgetting"] - eps
                    ),
                }
            )
    return rows


# --------------------------------------------------------------------- output
def write_csv(rows: Sequence[dict], path: Path, columns: Sequence[str]) -> Path:
    """Write ``rows`` as a CSV with a fixed column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return path


CORRELATION_COLUMNS = (
    "run",
    "method",
    "signal",
    "target",
    "rounds",
    "pearson",
    "spearman",
)
STOPPING_COLUMNS = (
    "run",
    "method",
    "signal",
    "delta",
    "stop_round",
    "adaptation",
    "forgetting",
    "oracle_round",
    "oracle_adaptation",
    "oracle_forgetting",
    "fixed_round",
    "fixed_adaptation",
    "fixed_forgetting",
    "adaptation_gap",
    "forgetting_gap",
    "rounds",
)
SELECTION_COLUMNS = (
    "signal",
    "delta",
    "eps",
    "num_candidates",
    "num_allowed",
    "selected",
    "selected_adaptation",
    "selected_forgetting",
    "oracle",
    "oracle_adaptation",
    "oracle_forgetting",
    "adaptation_gap",
    "forgetting_excess",
)


def plot_pareto(rows: Sequence[dict], out_dir: Path) -> list[Path]:
    """
    One ``(forgetting, adaptation)`` plot per method.

    Each signal contributes the curve its budget grid traces out; the oracle
    stop and the fixed budget are drawn as single reference markers.  Returns
    the paths written (empty when matplotlib is unavailable).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - headless without matplotlib
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, list[dict]] = {}
    for row in rows:
        methods.setdefault(str(row.get("method")), []).append(row)

    written: list[Path] = []
    for method, entries in sorted(methods.items()):
        figure, axes = plt.subplots(figsize=(6.5, 4.5))
        for signal in SIGNAL_KEYS:
            points = [
                (row["forgetting"], row["adaptation"])
                for row in entries
                if row["signal"] == signal
                and row["forgetting"] is not None
                and row["adaptation"] is not None
            ]
            if not points:
                continue
            points.sort()
            axes.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=signal,
            )
        oracle = [
            (row["oracle_forgetting"], row["oracle_adaptation"])
            for row in entries
            if row["oracle_forgetting"] is not None and row["oracle_adaptation"] is not None
        ]
        fixed = [
            (row["fixed_forgetting"], row["fixed_adaptation"])
            for row in entries
            if row["fixed_forgetting"] is not None and row["fixed_adaptation"] is not None
        ]
        if oracle:
            axes.scatter(
                [p[0] for p in oracle],
                [p[1] for p in oracle],
                marker="*",
                s=90,
                color="black",
                label="oracle stop",
                zorder=5,
            )
        if fixed:
            axes.scatter(
                [p[0] for p in fixed],
                [p[1] for p in fixed],
                marker="s",
                s=40,
                facecolors="none",
                edgecolors="black",
                label="fixed budget",
                zorder=5,
            )
        axes.set_xlabel("forgetting (source validation drop)")
        axes.set_ylabel("adaptation (clients' held-out accuracy)")
        axes.set_title(f"Stopping on constraint-respecting signals - {method}")
        axes.grid(alpha=0.3)
        axes.legend(fontsize=7, loc="best")
        figure.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in method)
        path = out_dir / f"signals_pareto_{safe}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def analyse(
    root,
    out_dir=None,
    deltas: Sequence[float] = DEFAULT_DELTAS,
    eps: float = DEFAULT_EPS,
    window: int = DEFAULT_WINDOW,
    plots: bool = True,
    population=None,
) -> dict[str, Any]:
    """
    Run the whole analysis over a results tree and write the tables.

    Args:
        root: Results directory to scan, recursively.
        out_dir: Where the tables and plots go; defaults to ``<root>/signals``.
        deltas: Budget grid of the stopping rules and of the selection.
        eps: Forgetting budget of the oracle selection.
        window: Final rounds averaged for the selection.
        plots: Whether to write the Pareto-style plots.
        population: A :class:`Population`, or the path of a manifest, or the
            directory one is shipped in.  Without it the pass pools over every
            run the root holds, which is a different measurement every time a
            stage lands - see :class:`Population`.

    Returns:
        dict: ``{"root", "num_runs", "correlations", "stopping", "selection",
        "plots"}``.

    Raises:
        FileNotFoundError: When a named population is not wholly on disk.  A
            pass over part of a population is a different pass, so it is
            refused rather than quietly taken.
    """
    root = Path(root)
    destination = Path(out_dir) if out_dir else root / "signals"
    destination.mkdir(parents=True, exist_ok=True)

    if population is not None and not isinstance(population, Population):
        population = load_population(population)

    runs = collect_trajectories(root, population)
    if population is not None:
        absent = missing_runs(runs, population)
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} of the {len(population)} runs {population.path} "
                f"names are not under {root}; the first is {absent[0]}. The "
                "population IS the measurement, so a pass over part of it is "
                "refused rather than quietly taken. Assemble the reviewer tree "
                "as docs/REPRODUCE.md section 10 says, or drop --population to "
                "pool over whatever this root holds and get a number of your "
                "own rather than this study's."
            )
    correlations = correlation_table(runs)
    stopping = stopping_table(runs, deltas=deltas)
    selection = selection_table(runs, deltas=deltas, eps=eps, window=window)

    write_csv(correlations, destination / "signal_correlations.csv", CORRELATION_COLUMNS)
    write_csv(stopping, destination / "signal_stopping.csv", STOPPING_COLUMNS)
    write_csv(selection, destination / "signal_selection.csv", SELECTION_COLUMNS)

    figures = plot_pareto(stopping, destination) if plots else []

    summary = {
        "root": str(root),
        "out_dir": str(destination),
        "deltas": list(deltas),
        "eps": eps,
        "window": window,
        "population": None if population is None else str(population.path),
        "population_runs": None if population is None else len(population),
        "num_runs": len(runs),
        "num_usable_runs": sum(1 for run in runs if run.usable),
        "runs": [
            {"name": run.name, "method": run.method, "rounds": run.rounds, "source": run.source}
            for run in runs
        ],
        "correlations": correlations,
        "stopping": stopping,
        "selection": selection,
        "plots": [str(path) for path in figures],
    }
    with open(destination / "signals_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from federated_outlier_adaptation import config

    parser = argparse.ArgumentParser(
        description="Correlate the constraint-respecting signals with true forgetting."
    )
    parser.add_argument("--root", default=None, help="Results folder to scan.")
    parser.add_argument("--out", default=None, help="Where to write the tables and plots.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--deltas", type=float, nargs="*", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--population",
        default=None,
        help="Manifest naming the runs this pass is taken over (or the study "
             "root that ships one).",
    )
    args = parser.parse_args(argv)

    summary = analyse(
        args.root or config.RESULTS_DIR,
        out_dir=args.out,
        deltas=tuple(args.deltas) if args.deltas else DEFAULT_DELTAS,
        eps=args.eps,
        window=args.window,
        plots=not args.no_plots,
        population=args.population,
    )
    print(
        json.dumps(
            {
                "root": summary["root"],
                "out_dir": summary["out_dir"],
                "population": summary["population"],
                "num_runs": summary["num_runs"],
                "num_usable_runs": summary["num_usable_runs"],
                "correlation_rows": len(summary["correlations"]),
                "stopping_rows": len(summary["stopping"]),
                "selection_rows": len(summary["selection"]),
                "plots": len(summary["plots"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
