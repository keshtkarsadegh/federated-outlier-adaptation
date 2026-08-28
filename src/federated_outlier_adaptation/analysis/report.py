"""
The statistics and report layer: from a results tree to tables and figures.

Every phase of this study writes JSON next to its numbers, and every one of
those files carries the provenance ``config`` block that says how it was
produced.  That is enough to rebuild the whole evaluation protocol without
retraining anything, and this module does exactly that:

    foa report --root results_v2 --out report/

**What it reads.**  ``summary_*.json`` and ``accuracies_*.json`` written by the
final, baseline and extreme runs (including their ``seed_<n>`` and provider
subfolders), ``config_points_*.json`` written by the sweeps, and the two
reference artefacts of each provider - the global model's metrics and the
combined model's metrics.  The ``config`` block identifies trainer,
hyperparameters, aggregation rule, scenario, metadata, seed, provider, client
population and parent, so a run is placed without parsing its path (the path is
used only for the parent folder, which the driver does not record).

**What it computes.**  At the fixed budget ``R`` (the last stored round):

===========================  =================================================
``A_src``                    source test accuracy of the adapted model
``F``                        ``A_src(theta_g) - A_src``, in percentage points
``A_new`` in-sample/val/test the clients' own data, and the two halves of the
                             40% split they never train on
``G``                        ``A_new_test - A_new_test(theta_g)``, in points
                             on the arm's *own* client set
normalised gain              ``G`` over the head-room to the combined model
rounds-to-target             first round within 95% of the final gain
worst-round forgetting       the largest drop over the whole trajectory
last-window std              stability of the adaptation curve at the end
fairness                     mean, worst decile, CoV and fraction improved over
                             the per-client test accuracies, when tracked
cost                         mean round seconds, wall clock, communication per
                             round and up to the target round
signals                      the final drift of every constraint-respecting
                             signal, and its correlation with true forgetting
===========================  =================================================

**The reference of ``F`` and ``G`` is always the shipped model.**  ``F`` is read
against ``theta_g``'s source test accuracy - the frozen ``global_metrics.json``
of the provider - and ``G`` against ``theta_g``'s accuracy on the client set the
arm actually adapts to, which is round 0 of any run of that client set that
started from ``theta_g``.  For a run that *is* initialised from ``theta_g`` this
is its own round 0 and nothing changes; for a run that is not - ``init=scratch``,
a locally fine-tuned model - round 0 is a different model altogether and reading
against it would report the distance travelled from that model rather than the
forgetting and the adaptation the study is about.  Such runs are reference
points, not methods: they are marked ``kind="reference"``, reported in the
references table with their corrected ``F``/``G`` and never enter the Pareto
front, the budget selection or the paired tests.

**Methods are ranked per group, not per class.**  ``AnchoredTrainer`` is not one
method but a family: the penalty space and the anchor decide what its objective
is, ``lam`` and ``T`` only how strong it is.  Every table that ranks methods -
the budget selection, the Pareto float, the paired tests, the trade-off figure -
therefore groups by ``anchored:<space>/<anchor>`` rather than by the class, and
``regularisation_family`` reports the family member by member.

**Client sets are part of a setting.**  The participants differ between studies -
the five selected writers, the 5% pool, a single writer, a replicated pair - so
the client set identifies a setting next to the communication scheme and the
round size.  Two arms are only compared, only pooled over seeds and only read
against the same ``theta_g`` accuracy when they adapt to the same clients; the
head-room of the normalised gain is likewise reported only where the combined
model covers exactly that client set.

**What it reports.**  Seeds are aggregated into mean, standard deviation and a
95% t-interval with the count ``n``.  Each arm is compared against the FedAvg
baseline of *its own setting* - same provider, scenario, metadata, pool, round
size and policy - seed by seed, with a paired t-test, a Wilcoxon signed-rank
test and Cohen's ``d_z``; the p-values are Holm-corrected within each table and
marked with the usual stars.  The trade-off itself is reported as the Pareto
front of ``(F, G)`` per method family and per aggregation hypothesis, and as the
adaptation reachable under a forgetting budget measured on validation data.

**Partial results are the normal case.**  A results root that is still filling
up produces the same tables with smaller ``n``; comparisons without a baseline,
or with fewer than two seed-matched pairs, are skipped and counted in
``report/index.md`` instead of failing.

Outputs::

    report/tables/*.csv     every table, machine-readable
    report/tables/*.tex     the same tables as booktabs floats to \\input
    report/figures/*.pdf    vector figures, one colour-blind-safe palette
    report/index.md         what each table and figure says, with the numbers
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from federated_outlier_adaptation.analysis import statistics as stats
from federated_outlier_adaptation.analysis.forgetting_signals import (
    SIGNAL_DIRECTION,
    pearson,
    spearman,
)

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
#: Aggregation rules that are FedAvg, in one form or another, most preferred
#: first.  The delta and the weight form of the same rule compute the same
#: update (README, "Algebraically identical pairs") and the unit server step of
#: the extended family is FedAvg by construction, so all of them can serve as
#: the baseline of a setting; the order only decides which one is picked when a
#: setting happens to hold several.
FEDAVG_RULES: tuple[str, ...] = (
    "con_weighted_cw",
    "con_delta_weighted_cgd",
    "con_delta_eta1",
    "seq_fedavg_update",
    "seq_delta_fedavg_update",
)

#: Which family a trainer belongs to.  The families are the axes the paper
#: compares along, not one row per class.
TRAINER_FAMILY = {
    "BaseTrainer": "none",
    "EWCTrainer": "fisher",
    "CFProxTrainer": "proximal",
    "DistillationTrainer": "distillation",
    "DistillationEWCTrainer": "distillation+fisher",
    "CFLogitConsistencyTrainer": "logit",
    "CFAlignedFeatureTrainer": "logit",
    "FeatureAlignmentTrainer": "feature",
    "FreezeTrainer": "freezing",
    "DistillationFreezeTrainer": "freezing",
    "NTDTrainer": "not-true distillation",
    "AnchoredTrainer": "anchored family",
}

#: Trainers that implement a *family* of objectives rather than one, with the
#: hyperparameters that name the variant.  ``AnchoredTrainer`` is the anchored
#: family: the penalty space and the anchor decide what the objective is - a
#: Fisher penalty towards the frozen shipped model and a not-true-distillation
#: penalty towards the current global model are different methods - while
#: ``lam`` and ``T`` are the strength of one method.  Grouping them together
#: would report the family by its best member and hide the other fourteen.
GROUP_HYPERPARAMETERS: dict[str, tuple[str, ...]] = {
    "AnchoredTrainer": ("space", "anchor"),
}

#: Prefix of the group label of a split trainer, so ``anchored:kd/frozen`` reads
#: as one name in a table of trainer names.
GROUP_PREFIX: dict[str, str] = {"AnchoredTrainer": "anchored"}

#: Aggregation hypotheses of the README, by the rule that tests them.
HYPOTHESIS_OF_RULE = {
    "con_delta_eta025": "H1 server step",
    "con_delta_eta05": "H1 server step",
    "con_delta_eta1": "H1 server step",
    "con_delta_median": "H3 robust",
    "con_delta_trimmed_mean": "H3 robust",
    "con_delta_anchor_lam01": "H4 anchoring",
    "con_delta_anchor_lam05": "H4 anchoring",
    "con_delta_fedavgm": "H5 server optimiser",
    "con_delta_fedadam": "H5 server optimiser",
    "con_delta_fedyogi": "H5 server optimiser",
    "seq_fixed_ratio_update": "H6 cyclic schedule",
    "seq_equal_update": "H6 cyclic schedule",
    "seq_incremental_update": "H6 cyclic schedule",
    "seq_delta_progressive_update": "H6 cyclic schedule",
}

#: Forgetting budgets of the adaptation-under-budget table, in points.
DEFAULT_BUDGETS: tuple[float, ...] = (0.25, 0.5, 1.0)

#: The setting the headline tables and the Pareto float are restricted to.
#: Matched as a substring of the setting label, so it covers the four
#: communication schemes of the same client population and round size.  A
#: selection that mixes settings is not a like-for-like comparison: it would put
#: one trainer's best arm from a ten-client uniform round next to another's from
#: a five-client worst-first round.  The cross-setting tables are still written,
#: as ``budget_selection_all`` and as the full ``pareto`` CSV.
MAIN_SETTING = "pool0.05 m10 uniform"

#: Runs that are reference points rather than methods: everything whose round 0
#: is not the shipped model ``theta_g`` - ``init=scratch``, local fine-tuning.
#: They keep their corrected ``F``/``G`` in the references table and are kept out
#: of every table that ranks methods against each other.
REFERENCE_KIND = "reference"

#: Parent folders and payload keys that mark a locally fine-tuned model.
LOCAL_FINETUNE_MARKERS: tuple[str, ...] = ("local_finetune", "local_finetuning")

#: Fraction of the final gain that defines the "target" a run is timed to.
TARGET_FRACTION = 0.95

#: Rounds averaged for the end-of-run stability measure.
STABILITY_WINDOW = 10

#: Colour-blind-safe categorical palette, in the order the slots are assigned.
#: Validated for the lightness band, the chroma floor, adjacent-pair CVD
#: separation and the normal-vision floor on a white surface; every series also
#: carries a marker or a line style, so identity never rests on colour alone.
PALETTE = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1)))
INK = "#1a1a19"
MUTED = "#6b6a66"
GRID = "#dcdbd6"

#: Metrics carried through the seed aggregation, with their label, unit and the
#: number of decimals a table prints them with.
METRICS: dict[str, tuple[str, str, int]] = {
    "a_src": ("A_src", "acc", 4),
    "forgetting": ("F", "pt", 3),
    "forgetting_val": ("F (val)", "pt", 3),
    "worst_forgetting": ("worst F", "pt", 3),
    "a_new_insample": ("A_new in-sample", "acc", 4),
    "a_new_val": ("A_new val", "acc", 4),
    "a_new_test": ("A_new test", "acc", 4),
    "gain": ("G", "pt", 3),
    "normalised_gain": ("G / head-room", "-", 3),
    "rounds_to_target": ("rounds to 95% G", "rounds", 1),
    "last_window_std": ("last-10 std", "pt", 3),
    "fairness_mean": ("client mean", "acc", 4),
    "fairness_worst_decile": ("worst decile", "acc", 4),
    "fairness_cov": ("client CoV", "-", 4),
    "fairness_improved": ("fraction improved", "-", 3),
    "mean_round_seconds": ("s / round", "s", 1),
    "total_wall_seconds": ("wall clock", "s", 0),
    "mean_comm_bytes": ("MB / round", "MB", 2),
    "comm_bytes_to_target": ("MB to target", "MB", 2),
}

#: Metrics the paired comparison against the FedAvg baseline is run on.
COMPARED_METRICS = ("a_new_test", "gain", "forgetting", "a_src")

#: Bytes per megabyte, for the cost table.
MEGABYTE = 1024.0 * 1024.0


# --------------------------------------------------------------------------- #
# reading the tree
# --------------------------------------------------------------------------- #
def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - unreadable artefact
        return None


def _first(series) -> Optional[float]:
    for value in series or []:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _last(series) -> Optional[float]:
    for value in reversed(list(series or [])):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _numbers(series) -> list[Optional[float]]:
    out: list[Optional[float]] = []
    for value in series or []:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append(None)
        elif not math.isfinite(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _source_test(accuracies) -> list[float]:
    return [
        float(pair[1])
        for pair in accuracies or []
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]


def _clients_insample(accuracies) -> list[float]:
    return [
        float(pair[0])
        for pair in accuracies or []
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    ]


def job_file(summary_path: Path, experiment: str, index: str) -> Optional[Path]:
    """
    The per-job file a summary entry refers to, in either stored layout.

    The driver writes ``<scenario>_<metadata>/summary_<i>.json`` next to
    ``<scenario>_<metadata>/<experiment>/accuracies_<i>.json``; the
    provider-adaptation entry point writes its summary one level higher, next to
    the ``<scenario>_<metadata>`` folder.  Recognising both matters twice over:
    the round accuracies are only in the per-job file, and a per-job file that
    is *not* recognised as belonging to a summary would be read a second time as
    an orphan and count its run twice.
    """
    direct = summary_path.parent / experiment / f"accuracies_{index}.json"
    if direct.is_file():
        return direct
    nested = sorted(summary_path.parent.glob(f"*/{experiment}/accuracies_{index}.json"))
    return nested[0] if nested else None


def parent_of(path: Path) -> str:
    """
    The experiment folder a run belongs to.

    The driver leaves ``config.parent_name`` unset, so the parent is read from
    the path: ``<parent>_<Trainer>_grid_search`` for the finals, the sweep's own
    folder name for a grid.  The ``seed_<n>`` and ``<scenario>_<metadata>``
    components below it are stripped.
    """
    for candidate in path.parents:
        if candidate.name.endswith("_grid_search"):
            return candidate.name
    return path.parent.name


def provider_of(config: Mapping[str, Any], path: Path) -> str:
    """Dataset a run belongs to: the recorded provider, else the path."""
    provider = (config or {}).get("provider")
    if provider:
        return str(provider)
    for part in path.parts:
        if part in ("shakespeare", "cifar10"):
            return part
    return "nist"


@dataclass
class RunRecord:
    """One stored run: what produced it, what it measured, what that means."""

    path: str
    parent: str
    experiment: str
    kind: str
    provider: str
    trainer: str
    scenario: Optional[str] = None
    metadata: Optional[str] = None
    aggregation: Optional[str] = None
    seed: Optional[int] = None
    init: str = "global"
    global_name: str = "global"
    max_round: Optional[int] = None
    hyperparameters: dict = field(default_factory=dict)
    population: dict = field(default_factory=dict)
    rounds: int = 0
    metrics: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        return TRAINER_FAMILY.get(self.trainer, self.trainer)

    @property
    def group(self) -> str:
        """The method this run competes as; see :func:`method_group`."""
        return method_group(self.trainer, self.hyperparameters)

    @property
    def hypothesis(self) -> Optional[str]:
        return HYPOTHESIS_OF_RULE.get(self.aggregation or "")

    @property
    def is_fedavg(self) -> bool:
        return (self.aggregation or "") in FEDAVG_RULES

    @property
    def is_reference(self) -> bool:
        """Whether this run is a reference point rather than a method."""
        return self.kind == REFERENCE_KIND

    @property
    def label(self) -> str:
        """Short human-readable name of the arm this run belongs to."""
        parts = [self.trainer]
        if self.aggregation:
            parts.append(self.aggregation)
        signature = hyperparameter_signature(self.hyperparameters)
        if signature:
            parts.append(signature)
        return " / ".join(parts)


def method_group(trainer: str, hyperparameters: Mapping[str, Any]) -> str:
    """
    The method a run belongs to when the tables rank methods against each other.

    For most trainers this is the trainer: one class, one objective.  A trainer
    that implements a family - :data:`GROUP_HYPERPARAMETERS` - is split by the
    hyperparameters that name the variant, so ``anchored:kd/frozen`` and
    ``anchored:fisher_scaled/current`` compete for their own budget row instead
    of the family being represented by whichever member happened to win.
    """
    keys = GROUP_HYPERPARAMETERS.get(trainer)
    if not keys:
        return trainer
    parts = [
        str((hyperparameters or {}).get(key))
        for key in keys
        if (hyperparameters or {}).get(key) is not None
    ]
    if len(parts) != len(keys):
        return trainer
    return f"{GROUP_PREFIX.get(trainer, trainer)}:" + "/".join(parts)


def hyperparameter_signature(hyperparameters: Mapping[str, Any]) -> str:
    """Stable ``k=v`` rendering of the objective hyperparameters of a run."""
    interesting = {
        key: value
        for key, value in sorted((hyperparameters or {}).items())
        if key not in ("learning_rate", "weight_decay") and value is not None
    }
    return ",".join(f"{key}={value}" for key, value in interesting.items())


def client_set_key(clients: Any) -> Optional[tuple[int, str]]:
    """
    Identity of a client population: its size and a digest of its members.

    The participants are stored by name in every provenance block, so the set a
    run adapts to is known exactly.  Two runs share a client set when they
    adapt to the same writers - which is what decides whether their numbers can
    be compared, pooled over seeds, or read against the same accuracy of the
    shipped model.  A duplicated member is kept, so a pool that replicates one
    writer is not the same set as the single writer.
    """
    if not isinstance(clients, (list, tuple)) or not clients:
        return None
    names = sorted(str(name) for name in clients)
    digest = hashlib.sha1("\x00".join(names).encode("utf-8")).hexdigest()[:10]
    return (len(names), digest)


def _clients_of(config: Mapping[str, Any]) -> Optional[tuple[int, str]]:
    """
    The client set of one run, from its provenance block.

    ``single_outlier`` is the participant list when a study replaces the pool by
    a hand-picked one (it may repeat a writer); ``outlier_writers`` is the pool
    itself otherwise.
    """
    config = config or {}
    chosen = config.get("single_outlier")
    if not isinstance(chosen, (list, tuple)) or not chosen:
        chosen = config.get("outlier_writers")
    return client_set_key(chosen)


def _population_of(config: Mapping[str, Any], block: Mapping[str, Any]) -> dict:
    """Client-population description of a run, from the config and the payload."""
    return {
        "clients": _clients_of(config),
        "pool_frac": (config or {}).get("pool_frac"),
        "clients_per_round": (config or {}).get("clients_per_round"),
        "policy": (config or {}).get("policy") or block.get("policy") or "all",
        "participation": (
            (config or {}).get("participation")
            if (config or {}).get("participation") is not None
            else block.get("participation")
        ),
        "pool_size": block.get("pool_size"),
        "outliers_file": (config or {}).get("outliers_file"),
        "track_clients": bool((config or {}).get("track_clients")),
    }


def is_reference_run(config: Mapping[str, Any], path: Path, experiment: str) -> bool:
    """
    Whether a stored run is a reference point rather than a method.

    A reference point is anything whose round 0 is not the shipped model: a run
    initialised from random weights (``init=scratch``) and the locally
    fine-tuned models.  Both answer "what would happen without the provider's
    model", which is a reference the methods are read against, not a competitor
    in the tables that rank them.
    """
    if str((config or {}).get("init") or "global") != "global":
        return True
    haystack = f"{path}/{experiment}".lower()
    return any(marker in haystack for marker in LOCAL_FINETUNE_MARKERS)


def _record(
    path: Path,
    experiment: str,
    block: Mapping[str, Any],
    kind: str,
) -> Optional[RunRecord]:
    """Build a record from one payload block, or ``None`` when it has no rounds."""
    config = block.get("config") or {}
    accuracies = block.get("accuracies") or []
    source_test = _source_test(accuracies)
    pool_test = _numbers(block.get("pool_test_accuracies"))
    if not source_test and not any(value is not None for value in pool_test):
        return None

    provider = provider_of(config, path)
    hyperparameters = dict(config.get("trainer_kwargs") or {})
    for key, value in (config.get("hyperparameters") or {}).items():
        hyperparameters.setdefault(key, value)

    record = RunRecord(
        path=str(path),
        parent=parent_of(path),
        experiment=experiment,
        kind=REFERENCE_KIND if is_reference_run(config, path, experiment) else kind,
        provider=provider,
        trainer=str(config.get("trainer") or "unknown"),
        scenario=config.get("scenario"),
        metadata=config.get("metadata"),
        aggregation=config.get("agg_method_name") or block.get("agg_method_name"),
        seed=config.get("seed") if config.get("seed") is not None else block.get("seed"),
        init=str(config.get("init") or "global"),
        global_name=str(config.get("global_name") or "global"),
        max_round=config.get("max_round"),
        hyperparameters=hyperparameters,
        population=_population_of(config, block),
    )
    record.series = {
        "source_test": source_test,
        "source_val": _numbers(block.get("source_val_accuracies")),
        "pool_test": pool_test,
        "pool_val": _numbers(block.get("pool_val_accuracies")),
        "heldout": _numbers(block.get("heldout_client_accuracies")),
        "insample": _clients_insample(accuracies),
        "signals": {key: _numbers(block.get(key)) for key in SIGNAL_DIRECTION},
        "client_test_last": _last_client_accuracies(block),
    }
    record.rounds = max(
        len(source_test), len(pool_test), len(record.series["heldout"])
    )
    record.metrics = compute_metrics(record, block)
    return record


def iter_run_blocks(root: Path) -> Iterable[tuple[Path, str, dict, str]]:
    """
    Every stored run under ``root``, as ``(path, experiment, block, kind)``.

    Three shapes are read.  A ``summary_*.json`` holds one entry per aggregation
    rule with every series except the round accuracies, which live in the
    per-job ``accuracies_<i>.json`` next to it and are merged back in.  A
    per-job file whose summary is missing is read on its own, so a job array
    interrupted between the two writes is not lost.  A sweep's
    ``config_points_*.json`` holds the same series under ``population`` inside
    each configuration's provenance block.
    """
    covered: set[Path] = set()

    for path in sorted(root.rglob("summary_*.json")):
        data = _load_json(path)
        if not isinstance(data, dict):
            continue
        index = path.name[len("summary_") : -len(".json")]
        for experiment, block in data.items():
            if not isinstance(block, dict):
                continue
            job = job_file(path, experiment, index)
            if job is not None:
                covered.add(job)
                if "accuracies" not in block:
                    payload = _load_json(job)
                    accuracies = payload.get("accuracies") if isinstance(payload, dict) else None
                    if isinstance(accuracies, list):
                        block = {**block, "accuracies": accuracies}
            yield path, experiment, block, "final"

    for path in sorted(root.rglob("accuracies_*.json")):
        if path.name.startswith("accuracies_points_") or path in covered:
            continue
        block = _load_json(path)
        if isinstance(block, dict) and block.get("accuracies"):
            yield path, path.parent.name, block, "final"

    for path in sorted(root.rglob("config_points_*.json")):
        data = _load_json(path)
        configs = (data or {}).get("config") if isinstance(data, dict) else None
        if not isinstance(configs, dict):
            continue
        index = path.name[len("config_points_") : -len(".json")]
        sibling = _load_json(path.parent / f"accuracies_points_{index}.json")
        accuracies = sibling if isinstance(sibling, dict) else {}
        for experiment, config in configs.items():
            if not isinstance(config, dict):
                continue
            block = dict(config.get("population") or {})
            block["config"] = config
            if experiment in accuracies:
                block.setdefault("accuracies", accuracies[experiment])
            yield path, experiment, block, "sweep"


def collect_runs(
    root, references: "References", datasets: Optional[Sequence[str]] = None
) -> list[RunRecord]:
    """
    Every run under ``root``, optionally restricted to a set of providers.

    The differences against the shipped model are filled in once the whole tree
    is read: the accuracy of ``theta_g`` on a client set is taken from the runs
    of that client set which started from it, so it cannot be known while a
    single file is being parsed.
    """
    wanted = set(datasets) if datasets else None
    runs: list[RunRecord] = []
    for path, experiment, block, kind in iter_run_blocks(Path(root)):
        record = _record(path, experiment, block, kind)
        if record is None:
            continue
        if wanted is not None and record.provider not in wanted:
            continue
        runs.append(record)
    apply_shipped_reference(runs, references)
    return runs


# --------------------------------------------------------------------------- #
# reference points
# --------------------------------------------------------------------------- #
@dataclass
class ProviderReferences:
    """The fixed points a provider's federated numbers are read against."""

    provider: str
    root: str
    source_test_acc: Optional[float] = None
    combined_source_acc: Optional[float] = None
    combined_clients_acc: Optional[float] = None
    combined_clients: Optional[tuple[int, str]] = None
    local_finetune: list[dict] = field(default_factory=list)

    @property
    def oracle_clients_acc(self) -> Optional[float]:
        """
        Adaptation head-room: the combined model on the participants' data.

        The combined model is trained on the server writers *and* the
        participants at once, so it is the upper reference the federated runs
        are normalised against.
        """
        return self.combined_clients_acc

    def oracle_for(self, clients: Optional[tuple[int, str]]) -> Optional[float]:
        """
        The head-room of one client set, or ``None`` when there is none.

        The combined model was trained and scored on the *selected* participants
        of the provider, so it is only an upper reference for runs that adapt to
        exactly those clients.  Normalising a five-per-cent pool's gain by the
        head-room of five hand-picked writers would divide by a number measured
        on other data, so the study reports no normalised gain there.
        """
        if self.combined_clients_acc is None:
            return None
        if self.combined_clients is None or clients is None:
            # A tree written before the participant list was frozen next to the
            # artefacts cannot be checked; the head-room is reported as before.
            return self.combined_clients_acc
        return self.combined_clients_acc if clients == self.combined_clients else None


class References:
    """
    Reference points of every provider found under a results root.

    The published dataset keeps its artefacts in the root itself and every
    other provider in its own subtree, exactly as ``providers/`` resolves them.
    """

    def __init__(self, root, datasets: Optional[Sequence[str]] = None):
        self.root = Path(root)
        self.providers: dict[str, ProviderReferences] = {}
        for provider in datasets or ("nist", "shakespeare", "cifar10"):
            sub = self.root if provider == "nist" else self.root / provider
            if not sub.is_dir():
                continue
            self.providers[provider] = self._read(provider, sub)

    @staticmethod
    def _accuracy(path: Path) -> Optional[float]:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            return None
        value = payload.get("test_accuracy")
        return float(value) if isinstance(value, (int, float)) else None

    @staticmethod
    def _selected_clients(path: Path) -> Optional[tuple[int, str]]:
        """The client set the combined model was scored on, when it is stored."""
        payload = _load_json(path)
        if isinstance(payload, dict):
            payload = payload.get("clients") or payload.get("selected")
        return client_set_key(payload)

    def _read(self, provider: str, sub: Path) -> ProviderReferences:
        references = ProviderReferences(
            provider=provider,
            root=str(sub),
            source_test_acc=self._accuracy(sub / "global_results" / "global_metrics.json"),
            combined_source_acc=self._accuracy(
                sub / "global_clients_results" / "global_clients_global_metrics.json"
            ),
            combined_clients_acc=self._accuracy(
                sub / "global_clients_results" / "global_clients_all_outliers_metrics.json"
            ),
            combined_clients=self._selected_clients(
                sub / "outliers" / "selected_outliers.json"
            ),
        )
        for path in sorted(sub.rglob("local_finetune.json")):
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            references.local_finetune.append(
                {
                    "path": str(path),
                    "trainer": payload.get("trainer"),
                    "seed": payload.get("seed"),
                    "provider": provider,
                    "global_name": str(
                        ((payload.get("config") or {}).get("global_name")) or "global"
                    ),
                    "client_set": client_set_key(payload.get("clients")),
                    "clients": len(payload.get("clients") or []),
                    "pool_insample_acc": payload.get("pool_insample_acc"),
                    "pool_val_acc": payload.get("pool_val_acc"),
                    "pool_test_acc": payload.get("pool_test_acc"),
                    "mean_source_test_acc": payload.get("mean_source_test_acc"),
                }
            )
        return references

    def get(self, provider: str) -> ProviderReferences:
        return self.providers.get(
            provider, ProviderReferences(provider=provider, root=str(self.root))
        )


# --------------------------------------------------------------------------- #
# per-run metrics
# --------------------------------------------------------------------------- #
def rounds_to_target(pool_test: Sequence[Optional[float]]) -> Optional[int]:
    """First round within :data:`TARGET_FRACTION` of the final adaptation gain."""
    values = [value for value in pool_test if value is not None]
    if len(values) < 2:
        return None
    reference, final = values[0], values[-1]
    total = final - reference
    if total <= 0:
        return None
    threshold = reference + TARGET_FRACTION * total
    for index, value in enumerate(pool_test):
        if value is not None and value >= threshold:
            return index
    return None  # pragma: no cover - the last round always reaches it


def _client_rounds(block: Mapping[str, Any]) -> list[dict]:
    """The non-empty per-client test-accuracy rounds of a payload."""
    series = block.get("client_test_accuracies") or []
    return [entry for entry in series if isinstance(entry, dict) and entry]


def _last_client_accuracies(block: Mapping[str, Any]) -> dict:
    """Per-client test accuracies of the final round, kept for the figures."""
    rounds = _client_rounds(block)
    return dict(rounds[-1]) if rounds else {}


def _instrumentation(block: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Timing and communication of a run, whichever shape it was stored in.

    The driver copies them into the payload itself; the sweep engine keeps them
    inside the provenance block under ``instrumentation``.
    """
    if block.get("round_seconds") is not None:
        return block
    nested = (block.get("config") or {}).get("instrumentation")
    return nested if isinstance(nested, dict) else {}


def fairness(block: Mapping[str, Any]) -> dict:
    """
    Spread of the per-client test accuracies at the last round.

    Only runs that tracked their clients carry the per-client series, so this
    returns an empty mapping for the published every-client-every-round loop and
    the fairness table simply has fewer rows.
    """
    rounds = _client_rounds(block)
    if not rounds:
        return {}
    first, last = rounds[0], rounds[-1]
    values = sorted(float(v) for v in last.values() if isinstance(v, (int, float)))
    if not values:
        return {}
    average = sum(values) / len(values)
    decile = max(1, len(values) // 10)
    improved = [
        client
        for client, value in last.items()
        if isinstance(value, (int, float))
        and isinstance(first.get(client), (int, float))
        and value > first[client]
    ]
    std = stats.sample_std(values)
    return {
        "fairness_mean": average,
        "fairness_worst_decile": sum(values[:decile]) / decile,
        "fairness_cov": (std / average) if std is not None and average else None,
        "fairness_improved": len(improved) / len(last) if last else None,
        "fairness_clients": len(values),
    }


def compute_metrics(record: RunRecord, block: Mapping[str, Any]) -> dict:
    """
    Every scalar the report prints for one run, at the fixed budget ``R``.

    Accuracies stay fractions; the differences ``F``, ``G``, the worst-round
    forgetting and the end-of-run standard deviation are percentage points,
    which is the unit the manuscript quotes them in.  The differences against
    the shipped model are filled in afterwards, by
    :func:`apply_shipped_reference`, because their reference is a property of
    the client set rather than of the single file this reads.
    """
    source_test = record.series["source_test"]
    source_val = record.series["source_val"]
    pool_test = record.series["pool_test"]
    pool_val = record.series["pool_val"]
    insample = record.series["insample"]

    metrics: dict[str, Any] = {}
    metrics["a_src"] = _last(source_test)
    metrics["a_new_insample"] = _last(insample)
    metrics["a_new_val"] = _last(pool_val)
    metrics["a_new_test"] = _last(pool_test)
    metrics["a_new_heldout"] = _last(record.series["heldout"])

    # Round 0 of every series, kept as the run's own starting point.  It is the
    # shipped model for a run that starts from it and something else otherwise,
    # so :func:`apply_shipped_reference` - not this function - decides which
    # number ``F`` and ``G`` are read against.
    metrics["round0_source_test"] = _first(source_test)
    metrics["round0_source_val"] = _first(source_val)
    metrics["round0_pool_test"] = _first(pool_test)
    for key in ("a_src_ref", "a_new_test_ref", "forgetting", "worst_forgetting",
                "forgetting_val", "gain", "normalised_gain"):
        metrics[key] = None

    target = rounds_to_target(pool_test)
    metrics["rounds_to_target"] = target
    window = [value for value in pool_test[-STABILITY_WINDOW:] if value is not None]
    std = stats.sample_std(window)
    metrics["last_window_std"] = std * 100.0 if std is not None else None

    metrics.update(fairness(block))

    instruments = _instrumentation(block)
    round_seconds = stats.finite(instruments.get("round_seconds") or [])
    metrics["mean_round_seconds"] = stats.mean(round_seconds)
    wall = (block.get("config") or {}).get("wall_seconds")
    metrics["total_wall_seconds"] = (
        float(wall) if isinstance(wall, (int, float)) else (sum(round_seconds) or None)
    )
    comm = stats.finite(instruments.get("comm_bytes_per_round") or [])
    metrics["mean_comm_bytes"] = (stats.mean(comm) / MEGABYTE) if comm else None
    metrics["comm_bytes_to_target"] = (
        sum(comm[:target]) / MEGABYTE if comm and target is not None else None
    )
    metrics["param_count"] = instruments.get("param_count")

    for signal, direction in SIGNAL_DIRECTION.items():
        series = record.series["signals"].get(signal) or []
        reference, final = _first(series), _last(series)
        metrics[f"drift_{signal}"] = (
            direction * (final - reference)
            if reference is not None and final is not None
            else None
        )
    return metrics


# --------------------------------------------------------------------------- #
# the shipped model as the reference of F and G
# --------------------------------------------------------------------------- #
@dataclass
class ShippedAccuracy:
    """What the provider's model ``theta_g`` scores, on one client set."""

    source_test: Optional[float] = None
    source_val: Optional[float] = None
    pool_test: Optional[float] = None
    #: How many global-initialised runs the pool accuracy was read from.
    n: int = 0
    #: Largest deviation between those runs, in points; a non-zero value means
    #: two runs of the same client set disagreed about ``theta_g`` itself.
    spread: float = 0.0


def _median(values: Sequence[float]) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def shipped_key(record: RunRecord) -> tuple:
    """The client set a ``theta_g`` accuracy belongs to."""
    return (record.provider, record.global_name, record.population.get("clients"))


class ShippedModel:
    """
    The accuracies of ``theta_g``, per provider and per client set.

    The source-side number is the frozen artefact of the provider, which is the
    same model on the same server test split for every run.  The participants'
    side has no artefact - the pool differs from study to study - so it is read
    off the runs that start from ``theta_g``: their round 0 *is* the shipped
    model on exactly those clients, and every such run of a client set must
    agree, which :attr:`spread` records.  The median is taken so that one
    truncated file cannot move the reference.
    """

    def __init__(self, runs: Sequence[RunRecord], references: "References"):
        pools: dict[tuple, list[float]] = {}
        source_val: dict[tuple, list[float]] = {}
        source_test: dict[str, list[float]] = {}
        for run in runs:
            if run.is_reference:
                continue
            pool = run.metrics.get("round0_pool_test")
            if isinstance(pool, (int, float)):
                pools.setdefault(shipped_key(run), []).append(float(pool))
            validation = run.metrics.get("round0_source_val")
            if isinstance(validation, (int, float)):
                source_val.setdefault(
                    (run.provider, run.global_name, run.seed), []
                ).append(float(validation))
            observed = run.metrics.get("round0_source_test")
            if isinstance(observed, (int, float)):
                source_test.setdefault(run.provider, []).append(float(observed))

        self._pool = {
            key: (
                _median(values),
                len(values),
                (max(values) - min(values)) * 100.0,
            )
            for key, values in pools.items()
        }
        self._source_val = {key: _median(values) for key, values in source_val.items()}
        self._observed_source_test = {
            provider: _median(values) for provider, values in source_test.items()
        }
        self._references = references

    @property
    def pool_spread(self) -> float:
        """Largest disagreement about ``theta_g`` inside a client set, in points."""
        return max((spread for _, _, spread in self._pool.values()), default=0.0)

    def pool_accuracy(
        self,
        provider: str,
        global_name: str,
        clients: Optional[tuple[int, str]],
    ) -> Optional[float]:
        """``theta_g``'s test accuracy on one client set, or ``None``."""
        pool, _, _ = self._pool.get((provider, global_name, clients), (None, 0, 0.0))
        return pool

    def source_test(self, provider: str) -> Optional[float]:
        """``theta_g``'s source test accuracy: the provider's frozen artefact."""
        stored = self._references.get(provider).source_test_acc
        if stored is not None:
            return stored
        return self._observed_source_test.get(provider)

    def source_test_deviation(self, provider: str) -> Optional[float]:
        """
        Points between the artefact and what the runs saw at round 0.

        Both numbers are ``theta_g`` on the server test split, so they have to
        agree; the report prints the deviation rather than assuming it.
        """
        stored = self._references.get(provider).source_test_acc
        observed = self._observed_source_test.get(provider)
        if stored is None or observed is None:
            return None
        return abs(stored - observed) * 100.0

    def source_val(self, record: RunRecord) -> Optional[float]:
        """
        ``theta_g``'s accuracy on the source validation split of a run.

        The split is built with the run's own loader seed, so the validation
        reference belongs to the seed rather than to the provider: a run that
        starts from ``theta_g`` reads its own round 0, and a run that does not
        borrows the round 0 of the runs that share its seed, falling back to the
        provider when that seed has none.
        """
        if not record.is_reference:
            own = record.metrics.get("round0_source_val")
            if isinstance(own, (int, float)):
                return float(own)
        for key in (
            (record.provider, record.global_name, record.seed),
            (record.provider, record.global_name, None),
        ):
            value = self._source_val.get(key)
            if value is not None:
                return value
        candidates = [
            value
            for (provider, global_name, _), value in self._source_val.items()
            if provider == record.provider
            and global_name == record.global_name
            and value is not None
        ]
        return _median(candidates)

    def for_run(self, record: RunRecord) -> ShippedAccuracy:
        """Everything ``theta_g`` scores that one run is read against."""
        pool, n, spread = self._pool.get(shipped_key(record), (None, 0, 0.0))
        return ShippedAccuracy(
            source_test=self.source_test(record.provider),
            source_val=self.source_val(record),
            pool_test=pool,
            n=n,
            spread=spread,
        )


def apply_shipped_reference(
    runs: Sequence[RunRecord], references: "References"
) -> ShippedModel:
    """
    Read ``F`` and ``G`` of every run against the shipped model.

    ``F`` is the drop from ``theta_g``'s source test accuracy and ``G`` the rise
    over ``theta_g``'s accuracy on the run's own client set, whatever the run
    itself started from.  For the runs that start from ``theta_g`` this is
    exactly their round 0 and nothing moves; for the others - ``init=scratch``,
    a locally fine-tuned model - it replaces a comparison against their own
    starting point, which measured how far they had travelled rather than what
    the provider gained or lost.

    The metrics are updated in place and the index is returned so the report can
    say which reference each number used.
    """
    shipped = ShippedModel(runs, references)
    for run in runs:
        theta = shipped.for_run(run)
        metrics = run.metrics
        metrics["a_src_ref"] = theta.source_test
        metrics["a_new_test_ref"] = theta.pool_test
        metrics["a_src_ref_n"] = theta.n

        a_src = metrics.get("a_src")
        metrics["forgetting"] = (
            (theta.source_test - a_src) * 100.0
            if theta.source_test is not None and a_src is not None
            else None
        )
        drops = [
            theta.source_test - value
            for value in run.series["source_test"]
            if value is not None
        ] if theta.source_test is not None else []
        metrics["worst_forgetting"] = max(drops) * 100.0 if drops else None

        final_val = _last(run.series["source_val"])
        metrics["forgetting_val"] = (
            (theta.source_val - final_val) * 100.0
            if theta.source_val is not None and final_val is not None
            else None
        )

        a_new = metrics.get("a_new_test")
        if a_new is not None and theta.pool_test is not None:
            gain = a_new - theta.pool_test
            metrics["gain"] = gain * 100.0
            oracle = references.get(run.provider).oracle_for(
                run.population.get("clients")
            )
            head_room = oracle - theta.pool_test if oracle is not None else None
            metrics["normalised_gain"] = (
                gain / head_room if head_room is not None and head_room > 0 else None
            )
        else:
            metrics["gain"] = None
            metrics["normalised_gain"] = None
    return shipped


# --------------------------------------------------------------------------- #
# arms: one configuration, several seeds
# --------------------------------------------------------------------------- #
def setting_key(record: RunRecord) -> tuple:
    """
    The experimental setting a run belongs to, seeds and method aside.

    Two runs share a setting when they could be compared directly: same
    dataset, same communication scheme, same client population and the same
    round budget.  The parent folder is deliberately *not* part of it, so the
    aggregation-hypothesis runs pair up with the FedAvg baselines that live
    under their own parent.

    The client set is part of it by name, not only by its size and policy: the
    studies of this results tree run the same grid over the five selected
    writers, over the five-per-cent pool and over a single writer, and those
    numbers describe different data.  Without the set itself they would share a
    setting whenever the sampler was left at its default, and their seeds would
    be pooled into one arm.
    """
    population = record.population
    return (
        record.provider,
        record.scenario,
        record.metadata,
        population.get("pool_frac"),
        population.get("clients_per_round"),
        population.get("policy"),
        population.get("participation"),
        record.init,
        record.global_name,
        record.max_round,
        population.get("clients"),
    )


def arm_key(record: RunRecord) -> tuple:
    """The setting plus the method, i.e. everything but the seed."""
    return setting_key(record) + (
        record.kind,
        record.parent,
        record.trainer,
        record.aggregation,
        hyperparameter_signature(record.hyperparameters),
    )


@dataclass
class Arm:
    """One configuration over its seeds."""

    key: tuple
    setting: tuple
    runs: list[RunRecord] = field(default_factory=list)

    @property
    def head(self) -> RunRecord:
        return self.runs[0]

    @property
    def provider(self) -> str:
        return self.head.provider

    @property
    def trainer(self) -> str:
        return self.head.trainer

    @property
    def aggregation(self) -> Optional[str]:
        return self.head.aggregation

    @property
    def parent(self) -> str:
        return self.head.parent

    @property
    def hyperparameters(self) -> dict:
        return self.head.hyperparameters

    @property
    def family(self) -> str:
        return self.head.family

    @property
    def group(self) -> str:
        return self.head.group

    @property
    def hypothesis(self) -> Optional[str]:
        return self.head.hypothesis

    @property
    def is_fedavg(self) -> bool:
        return self.head.is_fedavg

    @property
    def is_reference(self) -> bool:
        """A reference point, not a method: kept out of every ranking table."""
        return self.head.is_reference

    @property
    def is_baseline(self) -> bool:
        return self.trainer == "BaseTrainer" and self.is_fedavg and not self.is_reference

    @property
    def label(self) -> str:
        return self.head.label

    @property
    def seeds(self) -> list[Optional[int]]:
        return [run.seed for run in self.runs]

    def values(self, metric: str) -> list[Optional[float]]:
        return [run.metrics.get(metric) for run in self.runs]

    def by_seed(self, metric: str) -> dict[Any, float]:
        """Metric per seed; a repeated seed keeps its first observation."""
        out: dict[Any, float] = {}
        for run in self.runs:
            value = run.metrics.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                out.setdefault(run.seed, float(value))
        return out

    def summary(self, metric: str, level: float = stats.DEFAULT_LEVEL) -> stats.Summary:
        return stats.summarise(self.values(metric), level=level)


def group_arms(runs: Iterable[RunRecord]) -> list[Arm]:
    """Group runs into arms, keeping the order in which they were found."""
    arms: dict[tuple, Arm] = {}
    for record in runs:
        key = arm_key(record)
        arm = arms.get(key)
        if arm is None:
            arm = arms[key] = Arm(key=key, setting=setting_key(record))
        arm.runs.append(record)
    return list(arms.values())


def _preferred_baseline(candidates: Sequence[Arm]) -> Optional[Arm]:
    """Pick one baseline deterministically when a setting offers several."""
    if not candidates:
        return None
    order = {rule: index for index, rule in enumerate(FEDAVG_RULES)}
    return min(
        candidates,
        key=lambda arm: (order.get(arm.aggregation or "", len(order)), arm.parent, arm.label),
    )


def baseline_for(arm: Arm, arms: Sequence[Arm]) -> Optional[Arm]:
    """
    The FedAvg baseline an arm is measured against, inside its own setting.

    The study compares along two axes and the baseline follows the axis:

    - an arm running a **non-FedAvg aggregation rule** is compared against the
      FedAvg rule under the *same objective*, which is what isolates the rule;
    - an arm running a FedAvg rule is compared against plain FedAvg, i.e. the
      ``BaseTrainer`` arm of the same setting, which is what isolates the
      objective.

    ``BaseTrainer`` under FedAvg is the reference itself and has no baseline,
    and an arm is never its own.  A reference point neither has a baseline nor
    serves as one: it is not a method, and its round 0 is not ``theta_g``.
    """
    if arm.is_baseline or arm.is_reference:
        return None
    peers = [
        other
        for other in arms
        if other is not arm
        and not other.is_reference
        and other.setting == arm.setting
        and other.head.kind == arm.head.kind
    ]
    if not arm.is_fedavg:
        signature = hyperparameter_signature(arm.hyperparameters)
        same_objective = [
            other
            for other in peers
            if other.is_fedavg
            and other.trainer == arm.trainer
            and hyperparameter_signature(other.hyperparameters) == signature
        ]
        if same_objective:
            return _preferred_baseline(same_objective)
    return _preferred_baseline([other for other in peers if other.is_baseline])


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
def compare_arms(arm: Arm, baseline: Arm, metric: str) -> dict:
    """Seed-matched comparison of one metric between an arm and its baseline."""
    left, right = arm.by_seed(metric), baseline.by_seed(metric)
    seeds = [seed for seed in left if seed in right]
    a = [left[seed] for seed in seeds]
    b = [right[seed] for seed in seeds]
    t_test = stats.paired_t_test(a, b)
    wilcoxon = stats.wilcoxon_signed_rank(a, b)
    return {
        "metric": metric,
        "n_pairs": len(seeds),
        "seeds": seeds,
        "arm_mean": stats.mean(a),
        "baseline_mean": stats.mean(b),
        "difference": t_test["difference"],
        "t": t_test["t"],
        "p_t": t_test["p"],
        "p_wilcoxon": wilcoxon["p"],
        "wilcoxon_method": wilcoxon["method"],
        "cohens_d": stats.cohens_d_paired(a, b),
    }


def comparison_table(arms: Sequence[Arm], metrics: Sequence[str] = COMPARED_METRICS) -> list[dict]:
    """
    Every arm against the FedAvg baseline of its setting, Holm-corrected.

    The correction is applied *within each metric*, which is the family of tests
    a single table of the manuscript shows; the raw p-values are kept next to
    the adjusted ones so a different correction can be applied downstream.
    Reference points are dropped before the pairing, so nothing is ever tested
    against a model the provider never shipped.
    """
    arms = [arm for arm in arms if not arm.is_reference]
    rows: list[dict] = []
    for arm in arms:
        baseline = baseline_for(arm, arms)
        if baseline is None:
            continue
        for metric in metrics:
            comparison = compare_arms(arm, baseline, metric)
            if comparison["n_pairs"] < 2:
                continue
            rows.append(
                {
                    "provider": arm.provider,
                    "parent": arm.parent,
                    "setting": setting_label(arm.setting),
                    "group": arm.group,
                    "arm": arm.label,
                    "baseline": baseline.label,
                    "baseline_group": baseline.group,
                    **comparison,
                }
            )

    for metric in metrics:
        family = [row for row in rows if row["metric"] == metric]
        for column, adjusted in (("p_t", "p_t_holm"), ("p_wilcoxon", "p_wilcoxon_holm")):
            corrected = stats.holm([row[column] for row in family])
            for row, value in zip(family, corrected):
                row[adjusted] = value
                if adjusted == "p_t_holm":
                    row["marker"] = stats.significance_marker(value)
    return rows


def setting_label(setting: tuple) -> str:
    """
    Compact human-readable name of a setting tuple.

    The label carries the size of the client set and the initialisation when it
    is not the shipped model, so two settings that differ only in who
    participates - or in what the first round started from - never print under
    the same name.  Everything after the provider is a token the ``--setting``
    filter can match on.
    """
    provider, scenario, metadata, pool_frac, clients, policy, participation = setting[:7]
    init = setting[7] if len(setting) > 7 else None
    client_set = setting[10] if len(setting) > 10 else None
    parts = [str(provider)]
    if scenario:
        parts.append(str(scenario))
    if metadata:
        parts.append(str(metadata))
    if pool_frac is not None:
        parts.append(f"pool{pool_frac:g}")
    if clients is not None:
        parts.append(f"m{clients}")
    elif participation is not None and participation != 1.0:
        parts.append(f"C{participation:g}")
    if policy and policy != "all":
        parts.append(str(policy))
    if client_set is not None:
        parts.append(f"w{client_set[0]}")
    if init and str(init) != "global":
        parts.append(f"init {init}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Pareto fronts and forgetting budgets
# --------------------------------------------------------------------------- #
def pareto_front(points: Sequence[dict], x: str = "forgetting", y: str = "gain") -> list[dict]:
    """
    The points that no other point beats on both axes (``x`` down, ``y`` up).

    Ties are kept: two arms with the same pair are both on the front, which is
    what the plot should show.
    """
    usable = [
        point
        for point in points
        if isinstance(point.get(x), (int, float)) and isinstance(point.get(y), (int, float))
    ]

    def dominates(a: dict, b: dict) -> bool:
        return (
            a[x] <= b[x]
            and a[y] >= b[y]
            and (a[x] < b[x] or a[y] > b[y])
        )

    front = [
        point
        for point in usable
        if not any(other is not point and dominates(other, point) for other in usable)
    ]
    return sorted(front, key=lambda point: (point[x], -point[y]))


def arm_points(arms: Sequence[Arm], level: float = stats.DEFAULT_LEVEL) -> list[dict]:
    """One ``(forgetting, gain, adaptation)`` point per arm, averaged over seeds."""
    points: list[dict] = []
    for arm in arms:
        forgetting = arm.summary("forgetting", level)
        gain = arm.summary("gain", level)
        adaptation = arm.summary("a_new_test", level)
        validation = arm.summary("forgetting_val", level)
        points.append(
            {
                "provider": arm.provider,
                "parent": arm.parent,
                "setting": setting_label(arm.setting),
                "arm": arm.label,
                "trainer": arm.trainer,
                "group": arm.group,
                "aggregation": arm.aggregation,
                "family": arm.family,
                "hypothesis": arm.hypothesis or "",
                "kind": arm.head.kind,
                "hyperparameters": dict(arm.hyperparameters),
                "n": adaptation.n,
                "forgetting": forgetting.mean,
                "forgetting_ci": forgetting.ci,
                "forgetting_val": validation.mean,
                "gain": gain.mean,
                "gain_ci": gain.ci,
                "a_new_test": adaptation.mean,
                "a_new_test_ci": adaptation.ci,
            }
        )
    return points


#: Columns of the budget-selection tables, headline and cross-setting alike.
BUDGET_COLUMNS = (
    "provider",
    "trainer",
    "group",
    "family",
    "budget_pt",
    "num_candidates",
    "num_feasible",
    "selected",
    "setting",
    "a_new_test",
    "gain",
    "forgetting_val",
    "forgetting",
    "n",
)


def matches_setting(label: Optional[str], setting: Optional[str]) -> bool:
    """Whether a setting label matches a ``--setting`` filter, as a substring."""
    if not setting:
        return True
    return setting.strip().lower() in (label or "").lower()


def in_setting(points: Sequence[dict], setting: Optional[str]) -> list[dict]:
    """The points whose setting label matches the filter."""
    return [point for point in points if matches_setting(point.get("setting"), setting)]


def candidate_points(points: Sequence[dict], setting: Optional[str] = None) -> list[dict]:
    """The points a selection rule may choose from: one setting, no references."""
    return [
        point
        for point in in_setting(points, setting)
        if point.get("kind") != REFERENCE_KIND
    ]


def feasible_points(points: Sequence[dict], budget: float) -> list[dict]:
    """The points whose *validation* forgetting stays inside ``budget``."""
    return [
        point
        for point in points
        if isinstance(point.get("forgetting_val"), (int, float))
        and point["forgetting_val"] <= budget
        and isinstance(point.get("a_new_test"), (int, float))
    ]


def most_adaptive(points: Sequence[dict]) -> Optional[dict]:
    """The point with the highest reported test adaptation, or ``None``."""
    return max(points, key=lambda point: point["a_new_test"], default=None)


def budget_table(
    points: Sequence[dict],
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    setting: Optional[str] = None,
) -> list[dict]:
    """
    The most adaptive arm of each method under a forgetting budget.

    The constraint is read on the source *validation* drop - the split a
    selection rule is allowed to look at - and the objective is the reported
    test adaptation, so the budget never touches the number it selects on.

    ``setting`` restricts the candidates to one setting, matched as a substring
    of the setting label.  A selection that ranges over every setting answers a
    different question for every trainer - one wins with ten uniform clients per
    round, the next with five worst-first ones - so the headline table fixes the
    setting and the cross-setting version is written next to it.  Reference
    points are never candidates: they are not methods.

    One row per *method group* (:func:`method_group`), so a trainer that
    implements a family of objectives is selected once per member instead of
    once for the family.
    """
    rows: list[dict] = []
    points = candidate_points(points, setting)
    methods = sorted(
        {(point["provider"], point["group"]) for point in points if point.get("group")}
    )
    for provider, group in methods:
        candidates = [
            point
            for point in points
            if point["provider"] == provider and point["group"] == group
        ]
        trainer = candidates[0]["trainer"]
        for budget in budgets:
            feasible = feasible_points(candidates, budget)
            best = most_adaptive(feasible)
            rows.append(
                {
                    "provider": provider,
                    "trainer": trainer,
                    "group": group,
                    "family": TRAINER_FAMILY.get(trainer, trainer),
                    "budget_pt": budget,
                    "num_candidates": len(candidates),
                    "num_feasible": len(feasible),
                    "selected": None if best is None else best["arm"],
                    "setting": None if best is None else best["setting"],
                    "a_new_test": None if best is None else best["a_new_test"],
                    "gain": None if best is None else best["gain"],
                    "forgetting_val": None if best is None else best["forgetting_val"],
                    "forgetting": None if best is None else best["forgetting"],
                    "n": None if best is None else best["n"],
                }
            )
    return rows


#: Budget the regularisation-family table selects its representative under.
REGULARISATION_BUDGET = 0.5

#: Columns of the regularisation-family table.
REGULARISATION_COLUMNS = (
    "provider",
    "group",
    "space",
    "anchor",
    "budget_pt",
    "num_candidates",
    "num_feasible",
    "selected",
    "lam",
    "T",
    "setting",
    "a_new_test",
    "a_new_test_ci",
    "gain",
    "forgetting",
    "forgetting_val",
    "n",
    "on_setting_front",
)


def regularisation_rows(
    points: Sequence[dict],
    front: Sequence[dict] = (),
    budget: float = REGULARISATION_BUDGET,
    trainer: str = "AnchoredTrainer",
) -> list[dict]:
    """
    One row per member of a split family, at a single forgetting budget.

    The regularisation chapter asks what each *penalty* is worth, not what the
    best penalty is worth: one row per ``(space, anchor)``, holding the strength
    that member reaches furthest with inside the budget - its ``lam`` and, where
    the penalty has one, its temperature - the adaptation it reaches with its
    confidence interval, and whether it is on the Pareto front of the setting.
    Rows are ordered by that adaptation, so the table reads as a ranking; a
    member with nothing inside the budget keeps its row and sorts last.
    """
    keys = GROUP_HYPERPARAMETERS.get(trainer, ())
    on_front = {id(point) for point in front}
    rows: list[dict] = []
    groups = sorted(
        {
            (point["provider"], point["group"])
            for point in points
            if point.get("trainer") == trainer and point.get("group")
        }
    )
    for provider, group in groups:
        candidates = [
            point
            for point in points
            if point["provider"] == provider and point["group"] == group
        ]
        feasible = feasible_points(candidates, budget)
        best = most_adaptive(feasible)
        hyperparameters = (best or {}).get("hyperparameters") or {}
        named = dict(zip(keys, group.split(":", 1)[-1].split("/"))) if keys else {}
        rows.append(
            {
                "provider": provider,
                "group": group,
                "space": named.get("space"),
                "anchor": named.get("anchor"),
                "budget_pt": budget,
                "num_candidates": len(candidates),
                "num_feasible": len(feasible),
                "selected": None if best is None else best["arm"],
                "lam": hyperparameters.get("lam"),
                "T": hyperparameters.get("T"),
                "setting": None if best is None else best["setting"],
                "a_new_test": None if best is None else best["a_new_test"],
                "a_new_test_ci": None if best is None else best.get("a_new_test_ci"),
                "gain": None if best is None else best["gain"],
                "forgetting": None if best is None else best["forgetting"],
                "forgetting_val": None if best is None else best["forgetting_val"],
                "n": None if best is None else best["n"],
                "on_setting_front": False if best is None else id(best) in on_front,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["a_new_test"] is None,
            -(row["a_new_test"] or 0.0),
            row["group"],
        ),
    )


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #
def signal_table(runs: Sequence[RunRecord], signals_dir: Optional[Path] = None) -> list[dict]:
    """
    How well each constraint-respecting signal tracks true forgetting.

    When ``foa signals`` has already run over the same tree its pooled rows are
    reused verbatim - the two commands must not disagree - and the per-run
    correlations are recomputed from the trajectories otherwise.

    The correlation is against the drop from round 0, so only the runs that
    start from ``theta_g`` contribute: for a reference point that drop is the
    distance from another model and would not be forgetting at all.
    """
    runs = [run for run in runs if not run.is_reference]
    stored = _load_json(signals_dir / "signals_summary.json") if signals_dir else None
    if isinstance(stored, dict) and stored.get("correlations"):
        pooled = [
            row for row in stored["correlations"] if row.get("run") == "__pooled__"
        ]
        if pooled:
            return [{**row, "source": "foa signals"} for row in pooled]

    rows: list[dict] = []
    for signal in SIGNAL_DIRECTION:
        for target in ("source_val", "source_test"):
            values: list[float] = []
            targets: list[float] = []
            for run in runs:
                series = run.series["signals"].get(signal) or []
                reference = run.series["source_val" if target == "source_val" else "source_test"]
                start = _first(reference)
                if start is None:
                    continue
                for value, accuracy in zip(series, reference):
                    if value is None or accuracy is None:
                        continue
                    values.append(value)
                    targets.append(start - accuracy)
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
                    "source": "report",
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# writing tables
# --------------------------------------------------------------------------- #
def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    """Write ``rows`` as a CSV with a fixed column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return path


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


class Raw(str):
    """A cell that is already LaTeX and must survive :func:`latex_escape`."""


def latex_escape(text: Any) -> str:
    """Make a cell safe to paste into a LaTeX table; :class:`Raw` passes through."""
    if text is None:
        return "--"
    if isinstance(text, Raw):
        return str(text)
    return "".join(_LATEX_ESCAPES.get(character, character) for character in str(text))


def latex_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    headers: Optional[Sequence[str]],
    caption: str,
    label: str,
    alignment: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    A booktabs table the manuscript can ``\\input`` unchanged.

    The first column is left-aligned and the rest right-aligned unless
    ``alignment`` says otherwise; every body row carries exactly one cell per
    column, so the float compiles whatever the data looked like.

    Cells are escaped, because they are data; ``headers``, ``caption`` and
    ``notes`` are written verbatim, because they are authored here and carry
    the mathematics.  A cell that is deliberately LaTeX is wrapped in
    :class:`Raw`.  Without ``headers`` the column names are used, escaped.
    """
    if headers is not None and len(headers) != len(columns):  # pragma: no cover
        raise ValueError("headers and columns must have the same length")
    spec = alignment or ("l" + "r" * (len(columns) - 1))
    header_cells = (
        [str(header) for header in headers]
        if headers is not None
        else [latex_escape(column) for column in columns]
    )

    lines = [
        "% Generated by `foa report`; requires \\usepackage{booktabs}.",
        r"\begin{table}[t]",
        r"  \centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{spec}}}",
        r"    \toprule",
        "    " + " & ".join(header_cells) + r" \\",
        r"    \midrule",
    ]
    for row in rows:
        cells = [latex_escape(row.get(column)) for column in columns]
        lines.append("    " + " & ".join(cells) + r" \\")
    if not rows:
        lines.append("    " + " & ".join(["--"] * len(columns)) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}"]
    if notes:
        lines.append(f"  \\par\\smallskip\\footnotesize {notes}")
    lines += [r"\end{table}", ""]
    return "\n".join(lines)


def write_table(
    rows: Sequence[Mapping[str, Any]],
    out_dir: Path,
    name: str,
    columns: Sequence[str],
    headers: Optional[Sequence[str]] = None,
    caption: str = "",
    tex_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    tex_columns: Optional[Sequence[str]] = None,
    alignment: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """Write one table as CSV and, unless ``tex_columns`` is empty, as LaTeX."""
    out_dir = Path(out_dir)
    written = {
        "name": name,
        "rows": len(rows),
        "csv": str(write_csv(rows, out_dir / f"{name}.csv", columns)),
    }
    tex_columns = list(tex_columns if tex_columns is not None else columns)
    if tex_columns:
        source = tex_rows if tex_rows is not None else rows
        path = out_dir / f"{name}.tex"
        path.write_text(
            latex_table(
                source,
                tex_columns,
                headers,
                caption=caption or name.replace("_", " "),
                label=f"tab:{name}",
                alignment=alignment,
                notes=notes,
            )
        )
        written["tex"] = str(path)
    return written


def fmt(value: Any, digits: int = 3) -> str:
    """Format a number for a table cell; ``--`` for anything unavailable."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_summary(summary: stats.Summary, digits: int = 3) -> Raw:
    """``mean +- ci`` as a LaTeX cell, or just the mean when ``n = 1``."""
    if summary.mean is None:
        return Raw("--")
    if summary.ci is None:
        return Raw(f"{summary.mean:.{digits}f}")
    return Raw(f"{summary.mean:.{digits}f} $\\pm$ {summary.ci:.{digits}f}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _pyplot():
    """Matplotlib in a headless, paper-shaped configuration, or ``None``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - headless without matplotlib
        return None
    plt.rcParams.update(
        {
            "figure.figsize": (6.0, 3.8),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    return plt


def _slug(text: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in str(text)
    )


#: How far beyond the span of the points an axis follows a confidence interval.
#: A single arm whose three seeds disagreed wildly would otherwise set the scale
#: of the whole panel and hide every other arm; past this the interval is
#: clipped at the frame instead, and the figure's caption says so.
ERROR_BAR_REACH = 1.5


def _axis_limits(values: Sequence[float], errors: Sequence[float], pad: float = 0.08):
    """
    Axis range that shows every point, and every interval that is not extreme.

    Returns ``None`` when there is nothing to scale, so the caller can leave the
    automatic limits in place.
    """
    points = [value for value in values if value is not None]
    if not points:
        return None
    low, high = min(points), max(points)
    span = (high - low) or max(abs(high), 1.0)
    reach_low, reach_high = low - ERROR_BAR_REACH * span, high + ERROR_BAR_REACH * span
    for value, error in zip(values, errors):
        if value is None or not error:
            continue
        low = min(low, max(value - error, reach_low))
        high = max(high, min(value + error, reach_high))
    margin = pad * ((high - low) or max(abs(high), 1.0))
    return low - margin, high + margin


def figure_pareto(points: Sequence[dict], path: Path, group: str, title: str) -> Optional[Path]:
    """
    ``(F, G)`` scatter with the Pareto front drawn through it.

    One colour and one marker per group value, so the series are separable
    without colour; the front is the step line the non-dominated arms trace.

    Every group is drawn.  The colour cycles first and the marker only once the
    palette is exhausted, so the pairs stay unique well past the eight slots of
    the palette - a family split into its members needs more than eight series
    and dropping the rest would silently hide them.
    """
    plt = _pyplot()
    usable = [
        point
        for point in points
        if isinstance(point.get("forgetting"), (int, float))
        and isinstance(point.get("gain"), (int, float))
    ]
    if plt is None or not usable:
        return None

    groups: dict[str, list[dict]] = {}
    for point in usable:
        groups.setdefault(str(point.get(group) or "unlabelled"), []).append(point)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))

    figure, axes = plt.subplots()
    for index, (name, entries) in enumerate(ordered):
        axes.errorbar(
            [entry["forgetting"] for entry in entries],
            [entry["gain"] for entry in entries],
            xerr=[entry.get("forgetting_ci") or 0.0 for entry in entries],
            yerr=[entry.get("gain_ci") or 0.0 for entry in entries],
            fmt=MARKERS[(index // len(PALETTE)) % len(MARKERS)],
            color=PALETTE[index % len(PALETTE)],
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.6,
            capsize=1.5,
            linestyle="none",
            label=name,
        )

    front = pareto_front(usable)
    if len(front) > 1:
        axes.step(
            [point["forgetting"] for point in front],
            [point["gain"] for point in front],
            where="post",
            color=MUTED,
            linewidth=0.9,
            linestyle="--",
            zorder=0,
            label="Pareto front",
        )
    axes.axhline(0.0, color=GRID, linewidth=0.6, zorder=0)
    axes.axvline(0.0, color=GRID, linewidth=0.6, zorder=0)
    for setter, key in ((axes.set_xlim, "forgetting"), (axes.set_ylim, "gain")):
        limits = _axis_limits(
            [point[key] for point in usable],
            [point.get(f"{key}_ci") or 0.0 for point in usable],
        )
        if limits is not None:
            setter(*limits)
    axes.set_xlabel("forgetting F (points of source test accuracy)")
    axes.set_ylabel("adaptation gain G (points)")
    axes.set_title(title)
    # A split family carries many short labels; three narrow columns keep the
    # legend inside the panel instead of over the points.
    crowded = len(ordered) > len(PALETTE)
    axes.legend(loc="best", ncol=3 if crowded else 2, fontsize=6 if crowded else None)
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_arms(points: Sequence[dict], path: Path, title: str, metric: str = "gain") -> Optional[Path]:
    """Point plot with confidence intervals, one row per arm."""
    plt = _pyplot()
    usable = [point for point in points if isinstance(point.get(metric), (int, float))]
    if plt is None or not usable:
        return None
    usable = sorted(usable, key=lambda point: point[metric])
    height = max(2.0, 0.24 * len(usable) + 1.0)

    figure, axes = plt.subplots(figsize=(6.0, height))
    positions = list(range(len(usable)))
    axes.errorbar(
        [point[metric] for point in usable],
        positions,
        xerr=[point.get(f"{metric}_ci") or 0.0 for point in usable],
        fmt=MARKERS[0],
        color=PALETTE[0],
        markeredgecolor="white",
        markeredgewidth=0.5,
        elinewidth=0.7,
        capsize=2.0,
        linestyle="none",
    )
    axes.axvline(0.0, color=MUTED, linewidth=0.6, linestyle="--")
    limits = _axis_limits(
        [point[metric] for point in usable],
        [point.get(f"{metric}_ci") or 0.0 for point in usable],
    )
    if limits is not None:
        axes.set_xlim(*limits)
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{point['arm']} (n={point['n']})" for point in usable], fontsize=6.5
    )
    axes.set_xlabel(f"{METRICS.get(metric, (metric, '', 3))[0]} "
                    f"[{METRICS.get(metric, (metric, '', 3))[1]}], mean and 95% CI")
    axes.set_title(title)
    axes.grid(axis="y", visible=False)
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_learning_curves(arms: Sequence[Arm], path: Path, title: str) -> Optional[Path]:
    """
    Adaptation and source accuracy over rounds, with a band over the seeds.

    The band is the min-max envelope of the seeds rather than an interval: with
    three to five runs it is the honest statement of what was observed.
    """
    plt = _pyplot()
    usable = [arm for arm in arms if any(arm.head.series["pool_test"])]
    if plt is None or not usable:
        return None
    usable = sorted(usable, key=lambda arm: arm.label)[: len(PALETTE)]

    figure, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(6.0, 5.0), gridspec_kw={"hspace": 0.18}
    )
    nan = float("nan")
    for index, arm in enumerate(usable):
        colour = PALETTE[index % len(PALETTE)]
        style = LINESTYLES[index % len(LINESTYLES)]
        for axes, series in ((top, "pool_test"), (bottom, "source_test")):
            curves = [
                list(run.series[series])
                for run in arm.runs
                if any(value is not None for value in run.series[series])
            ]
            if not curves:
                continue
            rounds = list(range(min(len(curve) for curve in curves)))
            columns = [
                [curve[r] for curve in curves if curve[r] is not None] for r in rounds
            ]
            axes.plot(
                rounds,
                [sum(column) / len(column) if column else nan for column in columns],
                color=colour,
                linestyle=style,
                label=f"{arm.label} (n={len(curves)})" if series == "pool_test" else None,
            )
            if len(curves) > 1:
                axes.fill_between(
                    rounds,
                    [min(column) if column else nan for column in columns],
                    [max(column) if column else nan for column in columns],
                    color=colour,
                    alpha=0.14,
                    linewidth=0,
                )
    top.set_ylabel("A_new (pool test)")
    bottom.set_ylabel("A_src (source test)")
    bottom.set_xlabel("federated round")
    top.set_title(title)
    handles, labels = top.get_legend_handles_labels()
    if handles:
        top.legend(handles, labels, loc="best", ncol=1)
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_fairness(arms: Sequence[Arm], path: Path, title: str) -> Optional[Path]:
    """
    Empirical distribution of the per-client test accuracies at the last round.

    A cumulative curve rather than a box: with a handful of clients per pool the
    quartiles of a box plot are an interpolation, while the step function is the
    data.
    """
    plt = _pyplot()
    series: list[tuple[str, list[float]]] = []
    for arm in arms:
        values: list[float] = []
        for run in arm.runs:
            values += [
                float(value)
                for value in run.series.get("client_test_last", {}).values()
                if isinstance(value, (int, float))
            ]
        if values:
            series.append((arm.label, sorted(values)))
    if plt is None or not series:
        return None
    series = series[: len(PALETTE)]

    figure, axes = plt.subplots()
    for index, (label, values) in enumerate(series):
        fractions = [(i + 1) / len(values) for i in range(len(values))]
        axes.step(
            values,
            fractions,
            where="post",
            color=PALETTE[index % len(PALETTE)],
            linestyle=LINESTYLES[index % len(LINESTYLES)],
            label=f"{label} ({len(values)} clients)",
        )
    axes.set_xlabel("client test accuracy at the final round")
    axes.set_ylabel("fraction of clients at or below")
    axes.set_ylim(0.0, 1.02)
    axes.set_title(title)
    axes.legend(loc="best")
    figure.savefig(path)
    plt.close(figure)
    return path


def figure_signal_heatmap(rows: Sequence[dict], path: Path, title: str) -> Optional[Path]:
    """
    Correlation of every signal with both definitions of true forgetting.

    A diverging blue-to-red ramp with a neutral midpoint, because the sign of
    the correlation is the message; every cell also carries its number, so the
    figure never depends on reading a colour.
    """
    plt = _pyplot()
    usable = [row for row in rows if row.get("pearson") is not None]
    if plt is None or not usable:
        return None

    signals = [signal for signal in SIGNAL_DIRECTION if any(r["signal"] == signal for r in usable)]
    columns = [
        (target, statistic)
        for target in ("source_val", "source_test")
        for statistic in ("pearson", "spearman")
    ]
    grid = [
        [
            next(
                (
                    row[statistic]
                    for row in usable
                    if row["signal"] == signal and row["target"] == target
                ),
                None,
            )
            for target, statistic in columns
        ]
        for signal in signals
    ]

    figure, axes = plt.subplots(figsize=(4.6, 0.34 * len(signals) + 1.6))
    image = axes.imshow(
        [[0.0 if value is None else value for value in row] for row in grid],
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    axes.set_xticks(range(len(columns)))
    axes.set_xticklabels([f"{target}\n{statistic}" for target, statistic in columns], fontsize=7)
    axes.set_yticks(range(len(signals)))
    axes.set_yticklabels(signals, fontsize=7)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            axes.text(
                j,
                i,
                "--" if value is None else f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=INK if value is None or abs(value) < 0.55 else "white",
            )
    axes.grid(visible=False)
    axes.set_title(title)
    figure.colorbar(image, ax=axes, shrink=0.8, label="correlation with forgetting")
    figure.savefig(path)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
RUN_COLUMNS = (
    "provider",
    "parent",
    "experiment",
    "kind",
    "trainer",
    "group",
    "aggregation",
    "scenario",
    "metadata",
    "seed",
    "rounds",
    "hyperparameters",
    "policy",
    "clients_per_round",
    "pool_frac",
    "participation",
    "init",
    "client_set",
    *METRICS,
    # The two references ``F`` and ``G`` were read against, and the run's own
    # round 0 next to them, so the dump says which model each number is from.
    "a_src_ref",
    "a_new_test_ref",
    "round0_source_test",
    "round0_source_val",
    "round0_pool_test",
    *[f"drift_{signal}" for signal in SIGNAL_DIRECTION],
    "path",
)


def run_rows(runs: Sequence[RunRecord]) -> list[dict]:
    """The per-run dump every other table is derived from."""
    rows = []
    for run in runs:
        rows.append(
            {
                "provider": run.provider,
                "parent": run.parent,
                "experiment": run.experiment,
                "kind": run.kind,
                "trainer": run.trainer,
                "group": run.group,
                "aggregation": run.aggregation,
                "scenario": run.scenario,
                "metadata": run.metadata,
                "seed": run.seed,
                "rounds": run.rounds,
                "hyperparameters": hyperparameter_signature(run.hyperparameters),
                "policy": run.population.get("policy"),
                "clients_per_round": run.population.get("clients_per_round"),
                "pool_frac": run.population.get("pool_frac"),
                "participation": run.population.get("participation"),
                "init": run.init,
                "client_set": "/".join(
                    str(part) for part in (run.population.get("clients") or ())
                ),
                **{key: run.metrics.get(key) for key in METRICS},
                **{
                    key: run.metrics.get(key)
                    for key in (
                        "a_src_ref",
                        "a_new_test_ref",
                        "round0_source_test",
                        "round0_source_val",
                        "round0_pool_test",
                    )
                },
                **{
                    f"drift_{signal}": run.metrics.get(f"drift_{signal}")
                    for signal in SIGNAL_DIRECTION
                },
                "path": run.path,
            }
        )
    return rows


ARM_METRICS = (
    "a_new_test",
    "gain",
    "normalised_gain",
    "forgetting",
    "worst_forgetting",
    "a_src",
    "rounds_to_target",
    "last_window_std",
)


def arm_rows(arms: Sequence[Arm], level: float = stats.DEFAULT_LEVEL) -> list[dict]:
    """Seed-aggregated rows: mean, std, CI and ``n`` for every arm metric."""
    rows = []
    for arm in arms:
        row: dict[str, Any] = {
            "provider": arm.provider,
            "parent": arm.parent,
            "setting": setting_label(arm.setting),
            "arm": arm.label,
            "trainer": arm.trainer,
            "group": arm.group,
            "family": arm.family,
            "aggregation": arm.aggregation,
            "hypothesis": arm.hypothesis or "",
            "kind": arm.head.kind,
            "seeds": ",".join(str(seed) for seed in arm.seeds),
            "n": len(arm.runs),
        }
        for metric in ARM_METRICS:
            summary = arm.summary(metric, level)
            row[f"{metric}_n"] = summary.n
            row[f"{metric}_mean"] = summary.mean
            row[f"{metric}_std"] = summary.std
            row[f"{metric}_ci"] = summary.ci
        rows.append(row)
    return rows


COST_METRICS = (
    "mean_round_seconds",
    "total_wall_seconds",
    "mean_comm_bytes",
    "comm_bytes_to_target",
    "rounds_to_target",
)


def cost_rows(arms: Sequence[Arm], level: float = stats.DEFAULT_LEVEL) -> list[dict]:
    """Wall-clock and communication cost of every arm."""
    rows = []
    for arm in arms:
        row: dict[str, Any] = {
            "provider": arm.provider,
            "setting": setting_label(arm.setting),
            "arm": arm.label,
            "kind": arm.head.kind,
            "n": len(arm.runs),
            "param_count": arm.head.metrics.get("param_count"),
        }
        for metric in COST_METRICS:
            summary = arm.summary(metric, level)
            row[f"{metric}_mean"] = summary.mean
            row[f"{metric}_ci"] = summary.ci
        rows.append(row)
    return [row for row in rows if row.get("mean_round_seconds_mean") is not None]


FAIRNESS_METRICS = (
    "fairness_mean",
    "fairness_worst_decile",
    "fairness_cov",
    "fairness_improved",
)


def fairness_rows(arms: Sequence[Arm], level: float = stats.DEFAULT_LEVEL) -> list[dict]:
    """Fairness of the arms that tracked their clients."""
    rows = []
    for arm in arms:
        summaries = {metric: arm.summary(metric, level) for metric in FAIRNESS_METRICS}
        if summaries["fairness_mean"].mean is None:
            continue
        row: dict[str, Any] = {
            "provider": arm.provider,
            "setting": setting_label(arm.setting),
            "arm": arm.label,
            "kind": arm.head.kind,
            "n": len(arm.runs),
            "clients": arm.head.metrics.get("fairness_clients"),
        }
        for metric, summary in summaries.items():
            row[f"{metric}_mean"] = summary.mean
            row[f"{metric}_ci"] = summary.ci
        rows.append(row)
    return rows


REFERENCE_COLUMNS = (
    "provider",
    "reference",
    "detail",
    "value",
    "gain",
    "forgetting",
    "n",
)


def _points(new: Optional[float], old: Optional[float]) -> Optional[float]:
    """Difference of two accuracies in percentage points, when both are there."""
    if new is None or old is None:
        return None
    return (new - old) * 100.0


def reference_rows(
    references: References,
    shipped: Optional[ShippedModel] = None,
    reference_arms: Sequence["Arm"] = (),
    level: float = stats.DEFAULT_LEVEL,
) -> list[dict]:
    """
    The fixed points every federated number is read against.

    One row per provider artefact, one per stored local fine-tuning run and one
    per reference arm - the runs that did not start from ``theta_g``.  The
    reference arms are here *instead of* in the tables that rank methods: their
    ``F`` and ``G`` are the drop and the rise against the shipped model, which
    is what makes them readable next to the methods without letting them
    compete with them.
    """
    rows = []
    for provider, entry in sorted(references.providers.items()):
        rows.append(
            {
                "provider": provider,
                "reference": "global model theta_g",
                "detail": "source test accuracy",
                "value": entry.source_test_acc,
            }
        )
        rows.append(
            {
                "provider": provider,
                "reference": "combined model",
                "detail": "source test accuracy",
                "value": entry.combined_source_acc,
            }
        )
        rows.append(
            {
                "provider": provider,
                "reference": "combined model (pooled oracle)",
                "detail": "participants' data",
                "value": entry.combined_clients_acc,
            }
        )
        for finetune in entry.local_finetune:
            name = f"local fine-tuning ({finetune['trainer']}, seed {finetune['seed']})"
            pool = None
            if shipped is not None:
                pool = shipped.pool_accuracy(
                    provider,
                    finetune.get("global_name") or "global",
                    finetune.get("client_set"),
                )
            rows.append(
                {
                    "provider": provider,
                    "reference": name,
                    "detail": f"{finetune['clients']} clients, pool test",
                    "value": finetune["pool_test_acc"],
                    "gain": _points(finetune["pool_test_acc"], pool),
                }
            )
            rows.append(
                {
                    "provider": provider,
                    "reference": name,
                    "detail": "source test",
                    "value": finetune["mean_source_test_acc"],
                    "forgetting": _points(
                        entry.source_test_acc, finetune["mean_source_test_acc"]
                    ),
                }
            )

    for arm in sorted(reference_arms, key=lambda arm: (arm.provider, arm.label)):
        adaptation = arm.summary("a_new_test", level)
        rows.append(
            {
                "provider": arm.provider,
                "reference": f"{arm.label} [{arm.head.init} init]",
                "detail": f"{setting_label(arm.setting)}, pool test",
                "value": adaptation.mean,
                "gain": arm.summary("gain", level).mean,
                "forgetting": arm.summary("forgetting", level).mean,
                "n": adaptation.n,
            }
        )
    return [{key: row.get(key) for key in REFERENCE_COLUMNS} for row in rows]


def build_report(
    root,
    out_dir,
    datasets: Optional[Sequence[str]] = None,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    level: float = stats.DEFAULT_LEVEL,
    figures: bool = True,
    setting: Optional[str] = MAIN_SETTING,
) -> dict:
    """
    Read a results tree and write every table, figure and the index.

    Args:
        root: Results root to scan, recursively.
        out_dir: Report directory; ``tables/``, ``figures/`` and ``index.md``
            are created inside it.
        datasets: Providers to include; every one found by default.
        budgets: Forgetting budgets of the adaptation-under-budget table, in
            percentage points.
        level: Confidence level of every reported interval.
        figures: Whether to draw the figures.
        setting: Setting the headline budget table, the Pareto float and the
            Pareto figures are restricted to, matched as a substring of the
            setting label; ``None`` or an empty string keeps every setting in
            one table, which is not a like-for-like comparison.

    Returns:
        dict: the manifest that ``index.md`` is rendered from.
    """
    root = Path(root)
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    references = References(root, datasets)
    runs = collect_runs(root, references, datasets)
    shipped = ShippedModel(runs, references)
    arms = group_arms(runs)
    finals = [arm for arm in arms if arm.head.kind == "final"]
    reference_arms = [arm for arm in arms if arm.is_reference]
    points = arm_points(arms, level)
    final_points = [point for point in points if point["kind"] == "final"]
    headline_points = in_setting(final_points, setting)

    notes: list[str] = []
    tables: list[dict] = []
    written_figures: list[dict] = []

    # -- references ------------------------------------------------------- #
    references_table = reference_rows(references, shipped, reference_arms, level)
    tables.append(
        write_table(
            references_table,
            tables_dir,
            "references",
            columns=REFERENCE_COLUMNS,
            headers=(
                "Dataset",
                "Reference",
                "Measured on",
                "Accuracy",
                "$G$ [pt]",
                "$F$ [pt]",
                "n",
            ),
            alignment="lllrrrr",
            caption=(
                "Reference points every federated number is read against. $G$ and $F$ "
                "are measured against the shipped model $\\theta_g$, like every other "
                "table; these rows are reference points and take part in no ranking."
            ),
            tex_rows=[
                {
                    **row,
                    "value": fmt(row["value"], 4),
                    "gain": fmt(row["gain"], 3),
                    "forgetting": fmt(row["forgetting"], 3),
                    "n": fmt(row["n"], 0),
                }
                for row in references_table
            ],
        )
    )
    if not references.providers:
        notes.append("No provider artefacts were found under the results root.")
    if reference_arms:
        notes.append(
            f"{len(reference_arms)} arms are reference points rather than methods "
            "(their round 0 is not the shipped model); they are reported in the "
            "references table and take part in no Pareto front, budget selection "
            "or paired test."
        )
    for provider in sorted({run.provider for run in runs}):
        deviation = shipped.source_test_deviation(provider)
        if deviation is not None and deviation > 1e-6:
            notes.append(
                f"{provider}: the frozen source test accuracy of theta_g and the "
                f"round-0 value of the runs differ by {deviation:.4f} pt; the "
                "artefact is the reference."
            )
    if shipped.pool_spread > 1e-6:
        notes.append(
            "Two runs of the same client set disagreed about the shipped model's "
            f"accuracy on it by up to {shipped.pool_spread:.4f} pt; the median is "
            "the reference."
        )
    unreferenced = [
        run
        for run in runs
        if run.metrics.get("a_new_test") is not None
        and run.metrics.get("a_new_test_ref") is None
    ]
    if unreferenced:
        notes.append(
            f"{len(unreferenced)} runs report no adaptation gain: no run of their "
            "client set started from the shipped model, so its accuracy on those "
            "clients is unknown."
        )
    unmatched = [
        arm
        for arm in finals
        if arm.head.metrics.get("gain") is not None
        and arm.head.metrics.get("normalised_gain") is None
    ]
    if unmatched:
        notes.append(
            f"{len(unmatched)} final arms have no normalised gain: the combined "
            "model was scored on a different client set, so there is no head-room "
            "to divide by."
        )

    # -- per-run dump ----------------------------------------------------- #
    tables.append(
        write_table(
            run_rows(runs),
            tables_dir,
            "runs",
            columns=RUN_COLUMNS,
            tex_columns=(),
        )
    )

    # -- seed-aggregated arms --------------------------------------------- #
    aggregated = arm_rows(arms, level)
    tex_arms = [
        {
            "setting": row["setting"],
            "arm": row["arm"],
            "n": row["n"],
            "a_new_test": fmt_summary(
                stats.Summary(
                    n=row["a_new_test_n"],
                    mean=row["a_new_test_mean"],
                    std=row["a_new_test_std"],
                    ci=row["a_new_test_ci"],
                ),
                4,
            ),
            "gain": fmt_summary(
                stats.Summary(n=row["gain_n"], mean=row["gain_mean"], ci=row["gain_ci"]), 3
            ),
            "forgetting": fmt_summary(
                stats.Summary(
                    n=row["forgetting_n"], mean=row["forgetting_mean"], ci=row["forgetting_ci"]
                ),
                3,
            ),
            "rounds_to_target": fmt(row["rounds_to_target_mean"], 1),
        }
        for row in sorted(
            (row for row in aggregated if row["kind"] == "final"),
            key=lambda row: (row["setting"], row["arm"]),
        )
    ]
    tables.append(
        write_table(
            aggregated,
            tables_dir,
            "arms",
            columns=list(aggregated[0]) if aggregated else ["setting", "arm", "n"],
            tex_rows=tex_arms,
            tex_columns=("setting", "arm", "n", "a_new_test", "gain", "forgetting", "rounds_to_target"),
            headers=(
                "Setting",
                "Method",
                "n",
                "$A_\\mathrm{new}$ (test)",
                "$G$ [pt]",
                "$F$ [pt]",
                "rounds to 95\\% $G$",
            ),
            alignment="llrrrrr",
            caption=(
                "Adaptation and forgetting at the fixed round budget, mean and 95\\% "
                "confidence interval over the seeds."
            ),
            notes=(
                "Accuracies are fractions; $G$ and $F$ are percentage points. "
                "The float lists the final runs; the sweep configurations are in the CSV."
            ),
        )
    )

    # -- paired comparisons ------------------------------------------------ #
    comparisons = comparison_table(finals)
    tex_comparisons = [
        {
            "setting": row["setting"],
            "group": row["group"],
            "arm": row["arm"],
            "metric": METRICS.get(row["metric"], (row["metric"], "", 3))[0],
            "n_pairs": row["n_pairs"],
            "difference": f"{fmt(row['difference'], 4)}{row.get('marker', '')}",
            "p_t_holm": fmt(row.get("p_t_holm"), 4),
            "p_wilcoxon_holm": fmt(row.get("p_wilcoxon_holm"), 4),
            "cohens_d": fmt(row.get("cohens_d"), 2),
        }
        for row in comparisons
        if row["metric"] in ("a_new_test", "forgetting")
    ]
    tables.append(
        write_table(
            comparisons,
            tables_dir,
            "paired_vs_fedavg",
            columns=(
                "provider",
                "setting",
                "parent",
                "group",
                "arm",
                "baseline",
                "baseline_group",
                "metric",
                "n_pairs",
                "arm_mean",
                "baseline_mean",
                "difference",
                "t",
                "p_t",
                "p_t_holm",
                "p_wilcoxon",
                "p_wilcoxon_holm",
                "wilcoxon_method",
                "cohens_d",
                "marker",
            ),
            tex_rows=tex_comparisons,
            tex_columns=(
                "setting",
                "group",
                "arm",
                "metric",
                "n_pairs",
                "difference",
                "p_t_holm",
                "p_wilcoxon_holm",
                "cohens_d",
            ),
            headers=(
                "Setting",
                "Method group",
                "Method",
                "Metric",
                "$n$",
                "$\\Delta$ vs FedAvg",
                "$p$ (t, Holm)",
                "$p$ (W, Holm)",
                "$d_z$",
            ),
            alignment="llllrrrrr",
            caption=(
                "Seed-matched paired comparison against the FedAvg baseline of the same "
                "setting. Holm-corrected within each metric; $^{*}p<0.05$, $^{**}p<0.01$, "
                "$^{***}p<0.001$."
            ),
        )
    )
    if not comparisons:
        notes.append(
            "No paired comparison could be made: no setting held both a FedAvg "
            "baseline and another method with two or more shared seeds."
        )

    # -- Pareto ------------------------------------------------------------ #
    # The float shows the front of the headline setting - a front drawn over
    # several settings would put arms measured on different client sets and
    # different round sizes on one curve - and the CSV keeps every arm, with a
    # flag for each of the two fronts.
    front_all = pareto_front(final_points)
    front = pareto_front(headline_points)
    on_front = {id(point) for point in front_all}
    on_headline_front = {id(point) for point in front}
    tables.append(
        write_table(
            [
                {
                    **point,
                    "on_front": id(point) in on_front,
                    "on_setting_front": id(point) in on_headline_front,
                }
                for point in final_points
            ],
            tables_dir,
            "pareto",
            columns=(
                "provider",
                "setting",
                "parent",
                "arm",
                "trainer",
                "group",
                "aggregation",
                "family",
                "hypothesis",
                "n",
                "forgetting",
                "forgetting_ci",
                "gain",
                "gain_ci",
                "a_new_test",
                "on_front",
                "on_setting_front",
            ),
            tex_rows=[
                {
                    "setting": point["setting"],
                    "arm": point["arm"],
                    "group": point["group"],
                    "n": point["n"],
                    "forgetting": fmt(point["forgetting"], 3),
                    "gain": fmt(point["gain"], 3),
                    "a_new_test": fmt(point["a_new_test"], 4),
                }
                for point in front
            ],
            tex_columns=("setting", "arm", "group", "n", "forgetting", "gain", "a_new_test"),
            headers=(
                "Setting",
                "Method",
                "Method group",
                "n",
                "$F$ [pt]",
                "$G$ [pt]",
                "$A_\\mathrm{new}$ (test)",
            ),
            alignment="lllrrrr",
            caption=(
                "The Pareto-optimal configurations on (forgetting, adaptation gain)"
                + (f" in the {latex_escape(setting)} setting" if setting else "")
                + "."
            ),
            notes=(
                "The float is one setting; the CSV holds every arm, with "
                "\\texttt{on\\_front} over all settings and "
                "\\texttt{on\\_setting\\_front} inside this one."
            ),
        )
    )

    # -- forgetting budgets ------------------------------------------------ #
    budget_all = budget_table(final_points, budgets)
    tables.append(
        write_table(
            budget_all,
            tables_dir,
            "budget_selection_all",
            columns=BUDGET_COLUMNS,
            tex_columns=(),
        )
    )
    budget = budget_table(final_points, budgets, setting=setting)
    tables.append(
        write_table(
            budget,
            tables_dir,
            "budget_selection",
            columns=BUDGET_COLUMNS,
            tex_rows=[
                {
                    "provider": row["provider"],
                    "group": row["group"],
                    "budget_pt": fmt(row["budget_pt"], 2),
                    "num_feasible": f"{row['num_feasible']}/{row['num_candidates']}",
                    "selected": row["selected"] or "--",
                    "a_new_test": fmt(row["a_new_test"], 4),
                    "gain": fmt(row["gain"], 3),
                    "forgetting_val": fmt(row["forgetting_val"], 3),
                }
                for row in budget
            ],
            tex_columns=(
                "provider",
                "group",
                "budget_pt",
                "num_feasible",
                "selected",
                "a_new_test",
                "gain",
                "forgetting_val",
            ),
            headers=(
                "Dataset",
                "Method group",
                "$\\varepsilon$ [pt]",
                "feasible",
                "Selected configuration",
                "$A_\\mathrm{new}$ (test)",
                "$G$ [pt]",
                "$F$ (val) [pt]",
            ),
            alignment="llrrlrrr",
            caption=(
                "Adaptation reachable under a forgetting budget: the most adaptive arm "
                "whose validation forgetting stays within $\\varepsilon$"
                + (f", in the {latex_escape(setting)} setting" if setting else "")
                + "."
            ),
            notes=(
                "One setting, so the method groups are selected over the same clients "
                "and the same round size; the cross-setting selection is in "
                "\\texttt{budget\\_selection\\_all.csv}. A trainer that implements a "
                "family of objectives is one group per member."
                if setting
                else None
            ),
        )
    )
    if setting and not headline_points:
        notes.append(
            f"No final arm matched the headline setting '{setting}', so the budget "
            "table and the Pareto float are empty; the cross-setting tables are "
            "complete."
        )

    # -- the regularisation family, member by member ------------------------ #
    regularisation = regularisation_rows(
        candidate_points(headline_points), front, REGULARISATION_BUDGET
    )
    tables.append(
        write_table(
            regularisation,
            tables_dir,
            "regularisation_family",
            columns=REGULARISATION_COLUMNS,
            tex_rows=[
                {
                    "group": row["group"],
                    "lam": fmt(row["lam"], 3),
                    "T": fmt(row["T"], 1),
                    "a_new_test": fmt_summary(
                        stats.Summary(
                            n=row["n"] or 0,
                            mean=row["a_new_test"],
                            ci=row["a_new_test_ci"],
                        ),
                        4,
                    ),
                    "gain": fmt(row["gain"], 3),
                    "forgetting": fmt(row["forgetting"], 3),
                    "n": fmt(row["n"], 0),
                    "on_setting_front": "yes" if row["on_setting_front"] else "--",
                }
                for row in regularisation
            ],
            tex_columns=(
                "group",
                "lam",
                "T",
                "a_new_test",
                "gain",
                "forgetting",
                "n",
                "on_setting_front",
            ),
            headers=(
                "Penalty space / anchor",
                "$\\lambda$",
                "$T$",
                "$A_\\mathrm{new}$ (test)",
                "$G$ [pt]",
                "$F$ [pt]",
                "n",
                "front",
            ),
            alignment="lrrrrrrl",
            caption=(
                "The anchored family member by member: the strength each penalty "
                "reaches furthest with inside the "
                f"{REGULARISATION_BUDGET:g}~pt forgetting budget"
                + (f", in the {latex_escape(setting)} setting" if setting else "")
                + "."
            ),
            notes=(
                "One row per penalty space and anchor, ordered by the adaptation "
                "reached inside the budget; \\emph{front} marks the members on the "
                "Pareto front of this setting. $T$ is blank for a penalty without a "
                "temperature."
            ),
        )
    )

    # -- cost and fairness -------------------------------------------------- #
    costs = cost_rows(arms, level)
    tables.append(
        write_table(
            costs,
            tables_dir,
            "cost",
            columns=list(costs[0]) if costs else ["provider", "setting", "arm", "n"],
            tex_rows=[
                {
                    "setting": row["setting"],
                    "arm": row["arm"],
                    "n": row["n"],
                    "mean_round_seconds_mean": fmt(row.get("mean_round_seconds_mean"), 1),
                    "total_wall_seconds_mean": fmt(row.get("total_wall_seconds_mean"), 0),
                    "mean_comm_bytes_mean": fmt(row.get("mean_comm_bytes_mean"), 2),
                    "comm_bytes_to_target_mean": fmt(row.get("comm_bytes_to_target_mean"), 2),
                }
                for row in costs
                if row["kind"] == "final"
            ],
            tex_columns=(
                "setting",
                "arm",
                "n",
                "mean_round_seconds_mean",
                "total_wall_seconds_mean",
                "mean_comm_bytes_mean",
                "comm_bytes_to_target_mean",
            ),
            headers=("Setting", "Method", "n", "s / round", "wall [s]", "MB / round", "MB to target"),
            alignment="llrrrrr",
            caption="Wall-clock and communication cost per arm, averaged over the seeds.",
            notes="The float lists the final runs; the sweep configurations are in the CSV.",
        )
    )

    fairness_table = fairness_rows(arms, level)
    tables.append(
        write_table(
            fairness_table,
            tables_dir,
            "fairness",
            columns=list(fairness_table[0]) if fairness_table else ["provider", "setting", "arm", "n"],
            tex_rows=[
                {
                    "setting": row["setting"],
                    "arm": row["arm"],
                    "clients": row["clients"],
                    "fairness_mean_mean": fmt(row.get("fairness_mean_mean"), 4),
                    "fairness_worst_decile_mean": fmt(row.get("fairness_worst_decile_mean"), 4),
                    "fairness_cov_mean": fmt(row.get("fairness_cov_mean"), 4),
                    "fairness_improved_mean": fmt(row.get("fairness_improved_mean"), 3),
                }
                for row in fairness_table
                if row["kind"] == "final"
            ],
            tex_columns=(
                "setting",
                "arm",
                "clients",
                "fairness_mean_mean",
                "fairness_worst_decile_mean",
                "fairness_cov_mean",
                "fairness_improved_mean",
            ),
            headers=("Setting", "Method", "clients", "mean", "worst decile", "CoV", "improved"),
            alignment="llrrrrr",
            caption="Per-client test accuracy at the final round: level, tail and spread.",
            notes="The float lists the final runs; the sweep configurations are in the CSV.",
        )
    )
    if not fairness_table:
        notes.append(
            "No run tracked its clients, so the fairness table is empty; "
            "pass --track-clients to populate it."
        )

    # -- signals ------------------------------------------------------------ #
    signals = signal_table(runs, root / "signals")
    tables.append(
        write_table(
            signals,
            tables_dir,
            "signals",
            columns=("run", "method", "signal", "target", "rounds", "pearson", "spearman", "source"),
            tex_rows=[
                {
                    "signal": row["signal"],
                    "target": row["target"],
                    "rounds": row["rounds"],
                    "pearson": fmt(row.get("pearson"), 3),
                    "spearman": fmt(row.get("spearman"), 3),
                }
                for row in signals
            ],
            tex_columns=("signal", "target", "rounds", "pearson", "spearman"),
            headers=("Signal", "Forgetting target", "points", "Pearson", "Spearman"),
            alignment="llrrr",
            caption=(
                "Constraint-respecting signals against true forgetting, pooled over every "
                "round of every run."
            ),
        )
    )
    if not signals:
        notes.append("No run carried the constraint-respecting signals.")

    # -- coverage ------------------------------------------------------------ #
    single = [arm for arm in finals if len(arm.runs) == 1]
    if single:
        notes.append(
            f"{len(single)} of {len(finals)} final arms hold a single seed, so their "
            "confidence interval is reported as unavailable rather than as zero."
        )
    if not runs:
        notes.append("No run JSON was found under the results root.")

    # -- figures ------------------------------------------------------------ #
    if figures:
        written_figures = _draw_figures(
            final_points, headline_points, finals, signals, figures_dir, setting
        )

    manifest = {
        "root": str(root),
        "out_dir": str(out_dir),
        "datasets": sorted({run.provider for run in runs}),
        "num_runs": len(runs),
        "num_arms": len(arms),
        "num_final_arms": len(finals),
        "num_reference_arms": len(reference_arms),
        "num_seeds": len({run.seed for run in runs}),
        "level": level,
        "budgets": list(budgets),
        "setting": setting or "",
        "tables": tables,
        "figures": written_figures,
        "notes": notes,
        "references": references_table,
        "pareto_front": front,
        "budget_selection": budget,
        "regularisation_family": regularisation,
        "signals": signals,
        "comparisons": comparisons,
    }
    (out_dir / "index.md").write_text(render_index(manifest))
    with open(out_dir / "report.json", "w") as handle:
        json.dump(
            {key: value for key, value in manifest.items() if key != "comparisons"},
            handle,
            indent=2,
            default=str,
        )
    return manifest


def _draw_figures(
    points: Sequence[dict],
    headline_points: Sequence[dict],
    arms: Sequence[Arm],
    signals: Sequence[dict],
    figures_dir: Path,
    setting: Optional[str] = None,
) -> list[dict]:
    """
    Every figure of the report, skipping the ones with nothing to show.

    The two trade-off planes are drawn for the headline setting: a single plane
    holding arms from several client sets and round sizes reads as one curve
    while it is several.  The per-hypothesis panels and the learning curves are
    already inside one setting each.
    """
    written: list[dict] = []
    plane = list(headline_points) if headline_points else list(points)
    where = f" in the {setting} setting" if setting and headline_points else ""

    def record(path: Optional[Path], caption: str) -> None:
        if path is not None:
            written.append({"path": str(path), "caption": caption})

    record(
        figure_pareto(
            plane,
            figures_dir / "pareto_families.pdf",
            group="group",
            title="Adaptation against forgetting, by method group",
        ),
        f"Adaptation gain against forgetting for every arm{where}, coloured by method "
        "group - one series per trainer, and one per penalty space and anchor of the "
        "anchored family; bars are 95% intervals and are clipped where an arm's seeds "
        "disagreed far beyond the spread of the panel.",
    )
    hypothesis_points = [point for point in plane if point["hypothesis"]]
    record(
        figure_pareto(
            hypothesis_points,
            figures_dir / "pareto_hypotheses.pdf",
            group="hypothesis",
            title="Adaptation against forgetting, by aggregation hypothesis",
        ),
        f"The same plane{where} restricted to the aggregation hypotheses H1-H6.",
    )

    for hypothesis in sorted({point["hypothesis"] for point in hypothesis_points}):
        subset = [point for point in hypothesis_points if point["hypothesis"] == hypothesis]
        record(
            figure_arms(
                subset,
                figures_dir / f"hypothesis_{_slug(hypothesis)}.pdf",
                title=f"Adaptation gain under {hypothesis}",
            ),
            f"Adaptation gain with 95% confidence intervals for {hypothesis}.",
        )

    by_setting: dict[str, list[Arm]] = {}
    for arm in arms:
        by_setting.setdefault(setting_label(arm.setting), []).append(arm)
    for setting, members in sorted(by_setting.items(), key=lambda item: -len(item[1]))[:4]:
        record(
            figure_learning_curves(
                members,
                figures_dir / f"curves_{_slug(setting)}.pdf",
                title=f"Learning curves - {setting}",
            ),
            f"Adaptation and source accuracy over rounds for {setting}, band over the seeds.",
        )

    tracked = [arm for arm in arms if arm.head.metrics.get("fairness_clients")]
    record(
        figure_fairness(
            tracked,
            figures_dir / "fairness.pdf",
            title="Distribution of the per-client accuracies",
        ),
        "Empirical distribution of the per-client test accuracies at the final round.",
    )
    record(
        figure_signal_heatmap(
            signals,
            figures_dir / "signals_correlation.pdf",
            title="Signals against true forgetting",
        ),
        "Correlation of every constraint-respecting signal with the two forgetting targets.",
    )
    return written


def render_index(manifest: Mapping[str, Any]) -> str:
    """The human-readable summary of everything the report produced."""
    lines = [
        "# Experiment report",
        "",
        f"- results root: `{manifest['root']}`",
        f"- datasets: {', '.join(manifest['datasets']) or 'none found'}",
        f"- runs read: {manifest['num_runs']} "
        f"({manifest['num_final_arms']} final arms of {manifest['num_arms']} in total)",
        f"- confidence level: {manifest['level']:.0%} (Student's t, per arm)",
        f"- forgetting budgets: {', '.join(f'{b:g} pt' for b in manifest['budgets'])}",
        f"- headline setting: `{manifest.get('setting') or 'every setting'}`",
        "",
        "Accuracies are fractions. `F` (forgetting) and `G` (adaptation gain) are",
        "percentage points of accuracy, both measured against the provider's",
        "shipped model `theta_g` - never against a run's own starting point:",
        "",
        "- `F = A_src(theta_g) - A_src`, with `A_src(theta_g)` the frozen source",
        "  test accuracy of `theta_g` in `global_results/global_metrics.json`.",
        "- `G = A_new_test - A_new_test(theta_g)`, with `A_new_test(theta_g)` the",
        "  accuracy of `theta_g` on **that arm's own client set**, read off round 0",
        "  of the runs of that client set which started from `theta_g` (they all",
        "  share it; the median is used and any disagreement is reported above).",
        "- `F (val)` uses `theta_g` on the run's own source validation split,",
        "  which is built with the run's loader seed: its own round 0 for a run",
        "  that starts from `theta_g`, the round 0 of the runs sharing its seed",
        "  otherwise.",
        "",
        "For a run initialised from `theta_g` this is exactly its round 0, so",
        "nothing moves. For a run that is not - `init=scratch`, a locally",
        "fine-tuned model - round 0 is a different model, and reading against it",
        "would report the distance travelled from *that* model rather than the",
        "provider's loss and gain. Those runs are reference points, not methods:",
        "they appear in the references table with their corrected `F`/`G` and in",
        "no Pareto front, budget selection or paired test.",
        "",
        "The client set is part of a setting, so arms are only pooled over seeds,",
        "compared and normalised inside one set of participants; the normalised",
        "gain is reported only where the combined model covers exactly that set.",
        "",
        "The tables that rank methods do so per *method group*. For most",
        "trainers the group is the trainer; a trainer that implements a family",
        "of objectives is one group per member, so `AnchoredTrainer` appears as",
        "`anchored:<penalty space>/<anchor>` - the space and the anchor decide",
        "what the objective is, while `lam` and `T` are the strength of one",
        "objective. `regularisation_family.csv|tex` reports that family member by",
        "member at the 0.5 pt budget.",
        "",
        "The budget table and the Pareto float are one setting - the headline",
        "setting above - because a selection that ranges over settings picks a",
        "different experiment for every trainer. The cross-setting selection is",
        "in `budget_selection_all.csv` and every arm is in `pareto.csv`, where",
        "`on_front` is the front over all settings and `on_setting_front` the one",
        "inside the headline setting.",
        "",
        "The LaTeX tables are booktabs floats and need `\\usepackage{booktabs}`;",
        "`\\input` them directly.",
        "",
    ]

    if manifest["notes"]:
        lines += ["## Notes on coverage", ""]
        lines += [f"- {note}" for note in manifest["notes"]]
        lines.append("")

    lines += ["## Reference points", ""]
    for row in manifest["references"]:
        line = (
            f"- {row['provider']}: {row['reference']} - {row['detail']} = "
            f"{fmt(row['value'], 4)}"
        )
        if row.get("gain") is not None or row.get("forgetting") is not None:
            line += (
                f" (G = {fmt(row.get('gain'), 3)} pt, F = {fmt(row.get('forgetting'), 3)} pt)"
            )
        lines.append(line)
    lines.append("")

    lines += ["## Tables", ""]
    for table in manifest["tables"]:
        formats = ", ".join(
            f"`{Path(table[key]).name}`" for key in ("csv", "tex") if key in table
        )
        lines.append(f"- **{table['name']}** ({table['rows']} rows): {formats}")
    lines.append("")

    where = f" - {manifest['setting']}" if manifest.get("setting") else ""
    front = manifest["pareto_front"]
    lines += [f"## Pareto front (forgetting vs adaptation gain){where}", ""]
    if front:
        for point in front[:12]:
            lines.append(
                f"- {point['arm']} [{point['setting']}, n={point['n']}]: "
                f"F = {fmt(point['forgetting'], 3)} pt, G = {fmt(point['gain'], 3)} pt, "
                f"A_new(test) = {fmt(point['a_new_test'], 4)}"
            )
        if len(front) > 12:
            lines.append(f"- ... and {len(front) - 12} further non-dominated arms")
    else:
        lines.append("- no arm carried both a forgetting and a gain value")
    lines.append("")

    lines += [f"## Adaptation under a forgetting budget{where}", ""]
    if not manifest["budget_selection"]:
        lines.append("- no arm of this setting carried a validation forgetting value")
    for row in manifest["budget_selection"]:
        if row["selected"] is None:
            lines.append(
                f"- {row['provider']} / {row['group']} at eps = {row['budget_pt']:g} pt: "
                f"no configuration inside the budget ({row['num_candidates']} candidates)"
            )
            continue
        lines.append(
            f"- {row['provider']} / {row['group']} at eps = {row['budget_pt']:g} pt: "
            f"{row['selected']} reaches A_new(test) = {fmt(row['a_new_test'], 4)} "
            f"(G = {fmt(row['gain'], 3)} pt) with F(val) = "
            f"{fmt(row['forgetting_val'], 3)} pt, F = {fmt(row['forgetting'], 3)} pt, "
            f"n = {row['n']} ({row['num_feasible']}/{row['num_candidates']} feasible)"
        )
    lines.append("")

    regularisation = manifest.get("regularisation_family") or []
    if regularisation:
        lines += [
            f"## The anchored family under a {REGULARISATION_BUDGET:g} pt budget{where}",
            "",
        ]
        for row in regularisation:
            if row["selected"] is None:
                lines.append(
                    f"- {row['group']}: no configuration inside the budget "
                    f"({row['num_candidates']} candidates)"
                )
                continue
            strength = f"lam = {fmt(row['lam'], 3)}"
            if row["T"] is not None:
                strength += f", T = {fmt(row['T'], 1)}"
            lines.append(
                f"- {row['group']} ({strength}): A_new(test) = "
                f"{fmt(row['a_new_test'], 4)} +- {fmt(row['a_new_test_ci'], 4)}, "
                f"G = {fmt(row['gain'], 3)} pt, F = {fmt(row['forgetting'], 3)} pt, "
                f"n = {row['n']}"
                f"{', on the front' if row['on_setting_front'] else ''} "
                f"({row['num_feasible']}/{row['num_candidates']} feasible)"
            )
        lines.append("")

    significant = [
        row
        for row in manifest["comparisons"]
        if row.get("p_t_holm") is not None and row["p_t_holm"] < 0.05
    ]
    lines += ["## Paired comparisons against the FedAvg baseline", ""]
    lines.append(
        f"- {len(manifest['comparisons'])} seed-matched comparisons, "
        f"{len(significant)} significant after the Holm correction"
    )
    for row in significant[:12]:
        lines.append(
            f"- {row['arm']} vs {row['baseline']} [{row['setting']}], "
            f"{METRICS.get(row['metric'], (row['metric'], '', 3))[0]}: "
            f"delta = {fmt(row['difference'], 4)}{row.get('marker', '')}, "
            f"n = {row['n_pairs']}, d_z = {fmt(row.get('cohens_d'), 2)}"
        )
    lines.append("")

    lines += ["## Forgetting signals", ""]
    if manifest["signals"]:
        for row in manifest["signals"]:
            lines.append(
                f"- {row['signal']} vs {row['target']}: "
                f"Pearson {fmt(row.get('pearson'), 3)}, "
                f"Spearman {fmt(row.get('spearman'), 3)} "
                f"({row['rounds']} points, from {row.get('source')})"
            )
    else:
        lines.append("- no run carried the signal series")
    lines.append("")

    lines += ["## Figures", ""]
    if manifest["figures"]:
        for figure in manifest["figures"]:
            lines.append(f"- `{Path(figure['path']).name}` - {figure['caption']}")
    else:
        lines.append("- none (nothing to plot, or matplotlib unavailable)")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from federated_outlier_adaptation import config

    parser = argparse.ArgumentParser(
        description="Build the tables and figures of the experiment report."
    )
    parser.add_argument("--root", default=None, help="Results folder to scan.")
    parser.add_argument("--out", default=None, help="Report directory to write.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Providers to include.")
    parser.add_argument("--budgets", type=float, nargs="*", default=None)
    parser.add_argument("--level", type=float, default=stats.DEFAULT_LEVEL)
    parser.add_argument(
        "--setting",
        default=MAIN_SETTING,
        help=(
            "Setting of the headline tables, matched as a substring of the setting "
            f"label (default: {MAIN_SETTING!r}; pass an empty string for all)."
        ),
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    root = args.root or str(config.RESULTS_DIR)
    manifest = build_report(
        root,
        args.out or (Path(root) / "report"),
        datasets=args.datasets,
        budgets=tuple(args.budgets) if args.budgets else DEFAULT_BUDGETS,
        level=args.level,
        figures=not args.no_figures,
        setting=args.setting,
    )
    print(
        json.dumps(
            {
                "root": manifest["root"],
                "out_dir": manifest["out_dir"],
                "datasets": manifest["datasets"],
                "num_runs": manifest["num_runs"],
                "num_arms": manifest["num_arms"],
                "tables": len(manifest["tables"]),
                "figures": len(manifest["figures"]),
                "comparisons": len(manifest["comparisons"]),
                "notes": manifest["notes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
