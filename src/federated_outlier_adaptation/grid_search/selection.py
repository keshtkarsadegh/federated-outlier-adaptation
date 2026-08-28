"""
Validation-based, constrained model selection.

The legacy ranking of a sweep is a scalarisation,
``4 * clients_accuracy + 10 * global_accuracy^2``
(:mod:`federated_outlier_adaptation.analysis.top_selector`), which mixes
adaptation and forgetting into one number with hand-chosen coefficients.  It is
kept unchanged for continuity.

This module adds the selection rule the study actually argues for: **maximise
adaptation subject to a bound on forgetting**, decided on validation data only.

    maximise  mean over the last ``window`` rounds of the clients' held-out
              accuracy
    such that source_val_acc(theta_g) - mean source_val_acc <= eps

Selection is over **seed means**, not over single runs
------------------------------------------------------
A sweep stores one record per (configuration, family, seed).  Taking the argmax
over those records picks the luckiest seed of the luckiest configuration: with
seven lambdas, two families and two seeds that is an argmax over twenty-eight
noisy numbers, and the winner's advantage is partly the noise itself.  The
records are therefore grouped by (configuration, family) - the records of one
group differ only in the seed - and both axes are averaged within the group
before anything is compared.  The constraint is applied to the group's mean
forgetting and the argmax is taken over the group's mean adaptation, so a
configuration wins only if it wins on average.

The reported ``selected`` entry carries the group's means, with the
configuration's own name and its provenance block, so the downstream planner
still reads the selected ``lam``/``T`` out of it exactly as before.

``source_val_acc(theta_g)`` is the first entry of the source validation series,
which every run records *before* any client has trained, so the reference point
comes from the run itself and needs no extra evaluation.

Besides the winner, the full Pareto front of (adaptation, forgetting) is
reported, so the trade-off can be shown instead of asserted.

The same rule also decides the hyperparameters of the regularisation family's
final runs.  Each ``(space, anchor)`` sweep is selected *on its own*, at every
requested budget, and the winner's ``lam``/``T`` are written next to the sweep
as ``constrained_selection_eps<eps>.json`` so that
:mod:`federated_outlier_adaptation.training.matrix` can emit the finals with
the selected values instead of the trainer defaults.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

#: Rounds averaged at the end of a run.
DEFAULT_WINDOW = 5

#: Default forgetting budget on the source validation split.
DEFAULT_EPS = 0.005


@dataclass
class Record:
    """One configuration's summary numbers."""

    name: str
    source: str
    adaptation: Optional[float] = None
    source_reference: Optional[float] = None
    source_final: Optional[float] = None
    forgetting: Optional[float] = None
    pool_test: Optional[float] = None
    pool_val: Optional[float] = None
    rounds: int = 0
    config: dict = field(default_factory=dict)
    #: Seeds averaged into this entry; empty for a single run.
    seeds: list = field(default_factory=list)
    #: Number of records averaged into this entry; 1 for a single run.
    num_seeds: int = 1

    @property
    def complete(self) -> bool:
        return self.adaptation is not None and self.forgetting is not None


def group_key(record: "Record") -> tuple:
    """
    Identity of the (configuration, family) a record belongs to.

    The experiment name a sweep writes already contains the hyperparameters and
    the family (``anchored_lam1_T4_agg_<rule>_<metadata>_<scenario>``) and is
    identical across seeds, so it is the primary key.  The provenance block adds
    the family and the trainer keywords where it exists, which keeps two runs
    that happen to share a name but not a configuration apart.
    """
    config = _config_of(record)
    kwargs = config.get("trainer_kwargs")
    if not isinstance(kwargs, dict) or not kwargs:
        kwargs = config.get("hyperparameters") or {}
    parameters = tuple(sorted((str(k), repr(v)) for k, v in dict(kwargs).items()))
    return (
        record.name,
        config.get("scenario"),
        config.get("metadata"),
        config.get("agg_method_name"),
        parameters,
    )


def _mean(values) -> Optional[float]:
    numbers = [v for v in values if isinstance(v, (int, float))]
    return (sum(numbers) / len(numbers)) if numbers else None


def seed_mean_records(
    records: Iterable["Record"],
) -> list["Record"]:
    """
    Collapse the records of every (configuration, family) into their seed mean.

    Args:
        records: Records of one sweep, complete or not.

    Returns:
        list[Record]: One record per group, in first-appearance order.  Its
        ``adaptation``, ``forgetting`` and the two pool numbers are the means
        over the group's seeds; ``name`` and ``config`` are those of the group's
        first member, so the selected hyperparameters are still readable; the
        new ``seeds`` field lists the seeds that went into the mean.
    """
    groups: dict[tuple, list[Record]] = {}
    for record in records:
        groups.setdefault(group_key(record), []).append(record)

    collapsed: list[Record] = []
    for members in groups.values():
        head = members[0]
        seeds = [
            _config_of(member).get("seed")
            for member in members
            if _config_of(member).get("seed") is not None
        ]
        collapsed.append(
            Record(
                name=head.name,
                source=head.source,
                adaptation=_mean(m.adaptation for m in members),
                source_reference=_mean(m.source_reference for m in members),
                source_final=_mean(m.source_final for m in members),
                forgetting=_mean(m.forgetting for m in members),
                pool_test=_mean(m.pool_test for m in members),
                pool_val=_mean(m.pool_val for m in members),
                rounds=head.rounds,
                config=head.config,
                seeds=sorted(set(seeds)),
                num_seeds=len(members),
            )
        )
    return collapsed


def select_seed_mean(
    records: Iterable["Record"], eps: float
) -> tuple[Optional["Record"], list["Record"], list["Record"]]:
    """
    The constrained argmax over seed means.

    Args:
        records: Every record of the sweep.
        eps: Allowed drop of the source accuracy.

    Returns:
        ``(selected, feasible, groups)`` - the winning group, the groups within
        the budget and every group.
    """
    groups = seed_mean_records(records)
    feasible = [g for g in groups if g.forgetting is not None and g.forgetting <= eps]
    selected = max(feasible, key=lambda g: g.adaptation, default=None)
    return selected, feasible, groups


def _tail_mean(series, window: int = DEFAULT_WINDOW) -> Optional[float]:
    values = [v for v in (series or []) if isinstance(v, (int, float))]
    if not values:
        return None
    tail = values[-window:]
    return sum(tail) / len(tail)


def _first(series) -> Optional[float]:
    for value in series or []:
        if isinstance(value, (int, float)):
            return value
    return None


def _record_from_block(name: str, source: str, block: dict, window: int) -> Record:
    """Build a record from a payload/summary entry or a grid config block."""
    heldout = block.get("heldout_client_accuracies")
    source_val = block.get("source_val_accuracies")
    record = Record(name=name, source=source, config=block.get("config", {}) or {})
    record.adaptation = _tail_mean(heldout, window)
    record.pool_test = _tail_mean(block.get("pool_test_accuracies"), window)
    record.pool_val = _tail_mean(block.get("pool_val_accuracies"), window)
    record.source_reference = _first(source_val)
    record.source_final = _tail_mean(source_val, window)
    if record.source_reference is not None and record.source_final is not None:
        record.forgetting = record.source_reference - record.source_final
    record.rounds = len(block.get("accuracies") or heldout or [])
    return record


def collect_records(root, window: int = DEFAULT_WINDOW) -> list[Record]:
    """
    Scan a results tree for configurations that carry the new evaluation series.

    Both shapes are read: ``summary_*.json`` written by the final/extreme runs
    and ``config_points_*.json`` written by the sweeps (whose ``config`` blocks
    hold the same series under ``population``).

    Args:
        root: Results directory to scan, recursively.
        window: Number of final rounds averaged.

    Returns:
        list[Record]: One entry per configuration found, in path order.
    """
    root = Path(root)
    records: list[Record] = []

    for path in sorted(root.rglob("summary_*.json")):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            continue
        if not isinstance(data, dict):
            continue
        for name, block in data.items():
            if not isinstance(block, dict):
                continue
            record = _record_from_block(
                f"{path.parent.name}/{name}", str(path), block, window
            )
            if record.complete:
                records.append(record)

    for path in sorted(root.rglob("config_points_*.json")):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            continue
        configs = (data or {}).get("config", {})
        if not isinstance(configs, dict):
            continue
        for name, block in configs.items():
            if not isinstance(block, dict):
                continue
            population = block.get("population") or {}
            merged = dict(population)
            merged["config"] = block
            record = _record_from_block(
                f"{path.parent.name}/{name}", str(path), merged, window
            )
            if record.complete:
                records.append(record)

    return records


#: Differences below this are treated as ties, so that accumulated
#: floating-point noise in an averaged series cannot fake a Pareto-optimal point.
TOLERANCE = 1e-9


def pareto_front(records: Iterable[Record], tolerance: float = TOLERANCE) -> list[Record]:
    """
    Configurations that are not dominated on (adaptation up, forgetting down).

    Args:
        records: Candidates; incomplete ones are ignored.
        tolerance: Differences smaller than this count as equal.

    Returns:
        list[Record]: The front, ordered by increasing forgetting.
    """
    complete = [r for r in records if r.complete]

    def dominates(a: Record, b: Record) -> bool:
        no_worse = (
            a.adaptation >= b.adaptation - tolerance
            and a.forgetting <= b.forgetting + tolerance
        )
        strictly_better = (
            a.adaptation > b.adaptation + tolerance
            or a.forgetting < b.forgetting - tolerance
        )
        return no_worse and strictly_better

    front = [
        candidate
        for candidate in complete
        if not any(other is not candidate and dominates(other, candidate) for other in complete)
    ]
    return sorted(front, key=lambda r: (r.forgetting, -r.adaptation))


def select_constrained(
    root,
    eps: float = DEFAULT_EPS,
    on: str = "val",
    window: int = DEFAULT_WINDOW,
    out_dir=None,
) -> dict[str, Any]:
    """
    Pick the most adaptive configuration whose forgetting stays within ``eps``.

    The candidates are the **seed means** of each (configuration, family); see
    the module docstring.

    Args:
        root: Results directory to scan.
        eps: Allowed drop of the source accuracy relative to the run's own
            round-0 reference.
        on: ``"val"`` constrains on the source validation split (the default and
            the honest choice); ``"test"`` constrains on the source test split.
        window: Number of final rounds averaged.
        out_dir: Where to write ``constrained_selection.json`` and
            ``pareto_front.csv``.  Defaults to ``root``.

    Returns:
        dict: ``{"eps", "on", "window", "selected", "front", "records"}``.
    """
    records = collect_records(root, window=window)
    selected, feasible, groups = select_seed_mean(records, eps)
    front = pareto_front(groups)

    result = {
        "eps": eps,
        "on": on,
        "window": window,
        "root": str(root),
        "num_records": len(records),
        "num_groups": len(groups),
        "num_feasible": len(feasible),
        "selected": asdict(selected) if selected is not None else None,
        "front": [asdict(r) for r in front],
        "groups": [asdict(r) for r in groups],
        "records": [asdict(r) for r in records],
    }

    destination = Path(out_dir) if out_dir else Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    with open(destination / "constrained_selection.json", "w") as handle:
        json.dump(result, handle, indent=2)
    write_front_csv(front, destination / "pareto_front.csv")
    return result


# --------------------------------------------------------------------------- #
# per-sweep selection of the regularisation family
# --------------------------------------------------------------------------- #
#: Output folders of the anchored sweeps.  The default name is
#: ``anchored_<space>_<anchor>_grid_search``; ``foa grid --parent`` may put a
#: prefix in front of it, so the marker is matched anywhere in the name and the
#: stored provenance decides whether a match really is such a sweep.
SWEEP_GLOB = "*anchored_*_grid_search"

#: Trainer whose sweeps the per-sweep selection applies to.
ANCHORED_TRAINER = "AnchoredTrainer"

#: Distance spaces and anchors of the family, mirrored from
#: ``trainers.anchored_trainer`` so that this module stays free of torch.
ANCHOR_SPACE_NAMES = (
    "param_l2",
    "fisher",
    "fisher_scaled",
    "logit_l2",
    "kd",
    "ntd",
    "feature_l2",
    "kd+fisher",
)
ANCHOR_KIND_NAMES = ("frozen", "current")

#: Hyperparameters of the family that a final run has to be told about.  The
#: distance space and its anchor identify the objective; ``lam`` and ``T`` are
#: what the sweep searched over.
#:
#: ``param_l2_convention`` is carried along for the same reason: it decides
#: whether ``lam`` means FedProx's ``mu`` or the per-tensor-mean quantity, and a
#: final that executed the selected ``lam`` under the *other* convention would
#: be running a different objective from the one that was measured - the
#: sweep/final mismatch, in a new place.
SELECTED_KEYS = ("lam", "T", "param_l2_convention")


def selection_filename(eps: float) -> str:
    """Name of the per-sweep selection file of one forgetting budget."""
    return f"constrained_selection_eps{eps:g}.json"


def _config_of(record: Record) -> dict:
    return record.config if isinstance(record.config, dict) else {}


def sweep_configuration(folder) -> Optional[tuple[str, str]]:
    """
    The ``(space, anchor)`` a stored sweep was run with.

    Read from the provenance ``config`` block, which records the trainer keyword
    arguments verbatim; the folder name is only consulted when no readable
    config block is present, because it flattens ``kd+fisher`` to ``kd_fisher``.
    """
    folder = Path(folder)
    for path in sorted(folder.rglob("config_points_*.json")):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):  # pragma: no cover - unreadable artefact
            continue
        for block in ((data or {}).get("config") or {}).values():
            if not isinstance(block, dict):
                continue
            kwargs = dict(block.get("trainer_kwargs") or {})
            space, anchor = kwargs.get("space"), kwargs.get("anchor")
            if space and anchor:
                return str(space), str(anchor)

    name = folder.name
    marker = "anchored_"
    if marker not in name or not name.endswith("_grid_search"):
        return None
    tag = name[name.rindex(marker) + len(marker) : -len("_grid_search")]
    for anchor in ANCHOR_KIND_NAMES:
        if not tag.endswith(f"_{anchor}"):
            continue
        flattened = tag[: -len(anchor) - 1]
        for space in ANCHOR_SPACE_NAMES:
            if space.replace("+", "_") == flattened:
                return space, anchor
    return None


def discover_sweeps(root) -> list[tuple[Path, str, str]]:
    """
    Every anchored sweep under ``root``, with the space and anchor it swept.

    Returns:
        list[tuple]: ``(folder, space, anchor)``, in path order.  Folders whose
        configuration cannot be determined are skipped.
    """
    found: list[tuple[Path, str, str]] = []
    for folder in sorted(Path(root).rglob(SWEEP_GLOB)):
        if not folder.is_dir():
            continue
        configuration = sweep_configuration(folder)
        if configuration is None:
            continue
        found.append((folder, configuration[0], configuration[1]))
    return found


def selected_hyperparameters(record: Optional[Record]) -> dict[str, Any]:
    """
    The values a selected configuration has to be reproduced with.

    Read from ``trainer_kwargs``, which the sweep engine fills with exactly the
    keys the grid varied - ``lam`` always, ``T`` only where the space has a
    temperature - so a space without a temperature does not carry the trainer's
    default ``T`` onto the command line as if it had been chosen.  The recorded
    ``hyperparameters`` are only consulted for a run that stored no keyword
    arguments at all.
    """
    if record is None:
        return {}
    config = _config_of(record)
    kwargs = config.get("trainer_kwargs")
    if not isinstance(kwargs, dict) or not kwargs:
        kwargs = config.get("hyperparameters") or {}
    return {key: kwargs[key] for key in SELECTED_KEYS if kwargs.get(key) is not None}


def select_for_sweep(
    folder,
    eps: float = DEFAULT_EPS,
    on: str = "val",
    window: int = DEFAULT_WINDOW,
    space: Optional[str] = None,
    anchor: Optional[str] = None,
    write: bool = True,
) -> dict[str, Any]:
    """
    Run the constrained selection over a single sweep folder.

    Args:
        folder: One ``anchored_<space>_<anchor>_grid_search`` directory.
        eps: Allowed drop of the source accuracy, as in :func:`select_constrained`.
        on: Source split the constraint is read on.
        window: Number of final rounds averaged.
        space, anchor: Configuration of the sweep; read from the stored
            provenance when not given.
        write: Whether to store ``constrained_selection_eps<eps>.json`` next to
            the sweep, which is what makes the planner able to emit the finals.

    Returns:
        dict: the selection payload, including the selected ``hyperparameters``.
        The candidates are the seed means of each (configuration, family), so a
        sweep run with two seeds selects on two-seed averages and not on the
        better of the two runs.
    """
    folder = Path(folder)
    if space is None or anchor is None:
        configuration = sweep_configuration(folder) or (space, anchor)
        space, anchor = configuration

    records = collect_records(folder, window=window)
    selected, feasible, groups = select_seed_mean(records, eps)

    payload = {
        "eps": eps,
        "on": on,
        "window": window,
        "folder": str(folder),
        "space": space,
        "anchor": anchor,
        "trainer": ANCHORED_TRAINER,
        "num_records": len(records),
        "num_groups": len(groups),
        "num_feasible": len(feasible),
        "hyperparameters": selected_hyperparameters(selected),
        "selected": asdict(selected) if selected is not None else None,
        "front": [asdict(r) for r in pareto_front(groups)],
        "groups": [asdict(r) for r in groups],
    }
    if write:
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / selection_filename(eps), "w") as handle:
            json.dump(payload, handle, indent=2)
    return payload


def load_selection(folder, eps: float) -> Optional[dict[str, Any]]:
    """The stored selection of one sweep and budget, or ``None`` when absent."""
    path = Path(folder) / selection_filename(eps)
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return None
    return payload if isinstance(payload, dict) else None


def collect_selections(
    root, eps_values: Iterable[float]
) -> list[dict[str, Any]]:
    """
    Every stored per-sweep selection under ``root``, one per (sweep, budget).

    Only selections that actually name a configuration are returned, so a budget
    that excluded every point of a sweep produces no final run rather than a run
    with the trainer defaults.
    """
    payloads: list[dict[str, Any]] = []
    for folder, space, anchor in discover_sweeps(root):
        for eps in eps_values:
            payload = load_selection(folder, eps)
            if payload is None or payload.get("selected") is None:
                continue
            payload.setdefault("space", space)
            payload.setdefault("anchor", anchor)
            payload["eps"] = payload.get("eps", eps)
            payloads.append(payload)
    return payloads


def write_front_csv(records: Iterable[Record], path) -> Path:
    """Write the (adaptation, forgetting) front as a small CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "name",
        "num_seeds",
        "adaptation",
        "forgetting",
        "pool_val",
        "pool_test",
        "source_reference",
        "source_final",
        "rounds",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({column: row.get(column) for column in columns})
    return path
