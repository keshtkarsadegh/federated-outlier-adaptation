"""
matrix.py - experiment plans as job-array task lists.

Every large experiment of the study is a Cartesian product of trainers, seeds,
client populations and hyperparameters.  Rather than encoding those products in
shell scripts, a *plan* expands into one ``foa`` command per line::

    foa matrix --plan e1_seeds --out tasks_e1.txt
    sbatch --array=1-$(wc -l < tasks_e1.txt)%8 slurm/run_matrix.sbatch tasks_e1.txt

Every generated task carries ``--skip-existing``, so a job array that ran out
of wall-clock time can be resubmitted unchanged: tasks whose result file is
already on disk return immediately.

Plans
-----
``e1_seeds``        multi-seed repetition of the published final runs.
``e2_e5_grids``     the feature-alignment and layer-freezing sweeps.
``e2_e5_finals``    finals for those two trainers with the constants of
                    ``constants.py``; a placeholder until the sweeps have
                    selected the reported configuration.
``e6_pools``        prepares the larger client pools (CPU-cheap, run first).
``e6_population``   client population x participation x policy x trainer x seed.
``dual``            the truly duplicated client, seeds 1-3.
``regen_base``      baseline federated runs and the aggregation comparison.
``regen_grids``     every hyperparameter sweep.
``regen_finals``    finals for every trainer, seeds 1-5.
``regen_extreme``   the three extreme cases, seeds 1-3.
``fixed_cohort``    the published five-writer cohort, kept for continuity.
``pool``            builds the rule-based outlier pool of the main setting.
``pool_confirm_grids``  small KD/EWC sweeps repeated on the pool setting.
``aggregation_hypotheses``  H1-H7 of the aggregation redesign on the pool.
``regularisation_family``   every distance space x anchor, then the finals of
                    the constrained selection at three forgetting budgets.  The
                    finals need the lambda the selection picks, so the plan
                    emits them only when the selection files are already under
                    the results root given to the planner, and otherwise ends
                    with the single ``foa select --emit-finals`` line that
                    produces them.
``references``      local fine-tuning and a from-scratch federated run.
``lean_nist_sweeps``      the three regularisation sweeps of the lean protocol,
                    short ranges, one seed, no per-client tracking.
``lean_nist_finals_pre``  the finals that need no selection: FedAvg with and
                    without the early-stopping rule, and the selection-policy
                    study at half the round size.
``lean_nist_reg_finals``  the finals the constrained selection implies at the
                    single forgetting budget of the lean protocol; emitted by
                    ``foa select --emit-finals`` once the sweeps are done.
``lean_agg_hypotheses``   the aggregation redesign reduced to the ten arms that
                    carry a hypothesis, against the two trainer arms.
``lean_replica``    the whole lean protocol on one further dataset
                    (one provider: NIST).
``smoke``           cheap end-to-end checks of every code path (two rounds, one
                    local epoch); it produces no scientific result and writes
                    only under ``smoke_*`` folders.
``regen_all``       the plans above in dependency order (see ``REGEN_ORDER``).

The ``regen_*`` plans reproduce every published phase with the reproducible
pipeline.  They are meant to run against a separate results root
(``FOA_RESULTS_DIR``) holding copies of the frozen inputs - writer split,
teacher checkpoint, Fisher information and the selected-outlier list - so the
published ``results/`` tree is never written to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from federated_outlier_adaptation.constants import (
    BEST_FEATURE_ALIGN_BETA,
    FREEZE_SCOPE,
)

#: Trainers repeated over seeds in the E1 matrix.
E1_TRAINERS = (
    "BaseTrainer",
    "EWCTrainer",
    "CFProxTrainer",
    "DistillationTrainer",
    "CFLogitConsistencyTrainer",
    "DistillationEWCTrainer",
)

#: Every trainer of the package, used by the full regeneration plan.
ALL_TRAINERS = (
    "BaseTrainer",
    "EWCTrainer",
    "CFProxTrainer",
    "DistillationTrainer",
    "CFLogitConsistencyTrainer",
    "CFAlignedFeatureTrainer",
    "DistillationEWCTrainer",
    "FeatureAlignmentTrainer",
    "FreezeTrainer",
    "NTDTrainer",
)

#: Trainers compared across client populations in E6.
E6_TRAINERS = (
    "BaseTrainer",
    "DistillationTrainer",
    "EWCTrainer",
    "CFLogitConsistencyTrainer",
)

#: Client pools of E6: (k, mode).  ``("5", "sample")`` is the published list.
E6_POOLS = ((5, "sample"), (20, "lowest"), (50, "lowest"))

E6_PARTICIPATION = (0.2, 0.5, 1.0)
E6_SEEDS = (1, 2, 3)

E1_SEEDS = (1, 2, 3, 4, 5)
DUAL_SEEDS = (1, 2, 3)
EXTREME_SEEDS = (1, 2, 3)

#: Sweeps in the order the paper reports them.
GRID_METHODS = (
    "ewc",
    "prox",
    "kd",
    "logit",
    "aligned",
    "kd_ewc",
    "feature",
    "freeze",
    "ntd",
)
EXTREME_CASES = ("single", "double", "dual")

#: Assumed wall-clock cost of one task; used only for the planning estimate.
DEFAULT_MINUTES_PER_TASK = 12.0

#: The published dataset; every plan defaults to it, so the task lines of a
#: NIST plan are exactly what they were before the other datasets existed.
DEFAULT_PROVIDER = "nist"

#: Datasets the main plans are replicated over.
MAIN_PROVIDERS = ("nist",)

# --------------------------------------------------------------------------- #
# the pool setting (main configuration)
# --------------------------------------------------------------------------- #
#: Rule that defines the outlier pool of the main setting.
POOL_FRAC = 0.05

#: Clients drawn per round from the pool.
CLIENTS_PER_ROUND = (5, 10, 20)

#: Per-round selection policies compared on the pool.
POOL_POLICIES = ("uniform", "worst_first", "round_robin")

#: Trainers compared in the main setting.
POOL_TRAINERS = (
    "BaseTrainer",
    "DistillationTrainer",
    "EWCTrainer",
    "CFProxTrainer",
    "CFLogitConsistencyTrainer",
    "NTDTrainer",
    "FeatureAlignmentTrainer",
)

POOL_SEEDS = (1, 2, 3)
#: The FedAvg-equivalent baselines get two extra seeds.
POOL_BASELINE_SEEDS = (1, 2, 3, 4, 5)

#: Aggregation hypotheses H1-H7; see the README.
HYPOTHESIS_RULES = (
    # H1 server step size
    "con_delta_eta025",
    "con_delta_eta05",
    "con_delta_eta1",
    # H3 robust aggregators
    "con_delta_median",
    "con_delta_trimmed_mean",
    # H4 anchoring to the frozen global model
    "con_delta_anchor_lam01",
    "con_delta_anchor_lam05",
    # H5 server optimisers
    "con_delta_fedavgm",
    "con_delta_fedadam",
    "con_delta_fedyogi",
)

#: H2 weighting schemes, swept with the unit server step.
HYPOTHESIS_WEIGHTINGS = ("proportional", "uniform", "capped")

#: Trainers used for the aggregation hypotheses.
HYPOTHESIS_TRAINERS = ("BaseTrainer", "DistillationTrainer")

#: Distance spaces of the regularisation family, and their anchors.
ANCHOR_SPACES = (
    "param_l2",
    "fisher",
    "fisher_scaled",
    "logit_l2",
    "kd",
    "ntd",
    "feature_l2",
    "kd+fisher",
)
ANCHOR_KINDS = ("frozen", "current")
#: Sensitivity of the constrained selection.
SELECTION_EPS = (0.0025, 0.005, 0.01)
#: Seeds used inside the anchored grid.
GRID_SEEDS = (1, 2)

# --------------------------------------------------------------------------- #
# the smoke plan
# --------------------------------------------------------------------------- #
#: Rounds and local epochs of a smoke task.  Two rounds is the minimum that
#: exercises an aggregation step; one local epoch the minimum that exercises a
#: client update.  Nothing produced by this plan is a scientific result.
SMOKE_ROUNDS = 2
SMOKE_EPOCHS = 1

#: Worker counts of a smoke task; small enough to fit next to three siblings on
#: one shared GPU.
SMOKE_OUTER_WORKERS = 2
SMOKE_INNER_WORKERS = 2

#: Clients drawn per round in the smoke pool setting.
SMOKE_CLIENTS_PER_ROUND = 5

#: Seed of every smoke run, so a re-run lands on the same output paths.
SMOKE_SEED = 1

#: Prefix of every folder the smoke plan writes, so a smoke results root is
#: recognisable at a glance and can never collide with a real run.
SMOKE_PREFIX = "smoke"

#: Global model produced by the smoke federated pre-training.  It is *not*
#: ``global`` and not the published ``global_fl`` either, so neither the frozen
#: teacher nor an existing federated pre-training is ever overwritten.
SMOKE_GLOBAL_NAME = "smoke_global_fl"

#: Every trainer the smoke plan drives through the four (scenario, metadata)
#: jobs.  ``AnchoredTrainer`` needs its space and anchor, see ``_smoke_extra``.
SMOKE_TRAINERS = (
    "BaseTrainer",
    "DistillationTrainer",
    "EWCTrainer",
    "CFProxTrainer",
    "CFLogitConsistencyTrainer",
    "NTDTrainer",
    "FeatureAlignmentTrainer",
    "FreezeTrainer",
    "AnchoredTrainer",
)

#: Anchored-family configuration exercised by the smoke plan.
SMOKE_ANCHOR_SPACE = "kd"
SMOKE_ANCHOR_KIND = "current"

#: Distance space and anchor of the smoke sweep.
SMOKE_GRID_SPACE = "logit_l2"
SMOKE_GRID_ANCHOR = "frozen"

def _pool_args(clients: int, policy: str, seed: int, track: bool = True) -> str:
    """
    Pool source and per-round selection flags of the main setting.

    The pool is named by its rule rather than by a path, so a task line is
    portable across datasets and results roots: ``--pool-frac`` resolves
    through whichever provider the task runs on.

    ``track=False`` drops ``--track-clients``.  Per-client tracking is what the
    selection policies are studied with and what the per-client tables need, but
    a hyperparameter sweep only compares pooled numbers, so the lean sweeps
    leave it off.
    """
    return (
        f" --pool-frac {POOL_FRAC:g}"
        f" --clients-per-round {clients} --policy {policy}"
        f" --sampler-seed {seed}" + (" --track-clients" if track else "")
    )


def _pool_parent(prefix: str, clients: int, policy: str, variant: str) -> str:
    return f"{prefix}_m{clients}_{policy}_{variant.replace('+', '_')}"


def plan_pool(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """Build the rule-based outlier pool.  Run before every pool plan."""
    return [f"foa select-outliers --mode pool --pool-frac {POOL_FRAC:g}"]



# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _seed_args(seed: Optional[int]) -> str:
    return f" --seed {int(seed)}" if seed is not None else ""


def _provider_args(provider: str) -> str:
    """``--provider`` flag, omitted for the published dataset."""
    return "" if provider == DEFAULT_PROVIDER else f" --provider {provider}"


def provider_results_root(results_dir, provider: str = DEFAULT_PROVIDER) -> Path:
    """
    Where a dataset's artefacts live under a results root.

    The published dataset writes into the root itself; every other provider
    keeps its own subtree, exactly as ``providers/`` resolves it.
    """
    root = Path(results_dir)
    return root if provider == DEFAULT_PROVIDER else root / provider


def _pool_tag(k: int, mode: str) -> str:
    return f"k{int(k)}" if mode == "sample" else f"k{int(k)}{mode}"


def _participation_tag(participation: float) -> str:
    return f"c{participation:g}".replace(".", "")


def e6_policies(k: int, participation: float) -> tuple[str, ...]:
    """
    Policies evaluated for one (pool, participation) pair.

    Full participation leaves nothing to select, so it uses ``all``.  The
    cyclic policy only becomes interesting on the largest pool.
    """
    if participation >= 1.0:
        return ("all",)
    if int(k) == 50:
        return ("uniform", "worst_first", "round_robin")
    return ("uniform", "worst_first")


# --------------------------------------------------------------------------- #
# plans
# --------------------------------------------------------------------------- #
def plan_e1_seeds(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """Published final runs and the extreme double case, repeated over seeds."""
    tasks: list[str] = []
    for trainer in E1_TRAINERS:
        for seed in E1_SEEDS:
            tasks.append(
                f"foa final --trainer {trainer}{_seed_args(seed)} --skip-existing"
            )
    # Second knowledge-distillation variant with the softer temperature.
    for seed in E1_SEEDS:
        tasks.append(
            "foa final --trainer DistillationTrainer --parent final_result_T4"
            f"{_seed_args(seed)} --set T=4 alpha=0.95 --skip-existing"
        )
    for seed in E1_SEEDS:
        tasks.append(f"foa extreme --case double{_seed_args(seed)} --skip-existing")
    return tasks


def plan_e2_e5_grids(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """The two new sweeps: representation alignment and layer freezing."""
    return [
        "foa grid --method feature --skip-existing",
        "foa grid --method freeze --skip-existing",
    ]


def plan_e2_e5_finals(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Finals for the two new trainers with the values currently in constants.py.

    Placeholder: once the sweeps have run, replace the constants (or pass
    ``--set``) so these tasks use the selected configuration.
    """
    tasks: list[str] = []
    for seed in E1_SEEDS:
        tasks.append(
            "foa final --trainer FeatureAlignmentTrainer"
            f"{_seed_args(seed)} --set beta={BEST_FEATURE_ALIGN_BETA} --skip-existing"
        )
    for seed in E1_SEEDS:
        tasks.append(
            "foa final --trainer FreezeTrainer"
            f'{_seed_args(seed)} --set scope="{FREEZE_SCOPE}" --skip-existing'
        )
    return tasks


def plan_e6_pools(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """Build the larger client pools.  Run before ``e6_population``."""
    return [
        f"foa select-outliers --mode {mode} --k {k}"
        for k, mode in E6_POOLS
        if not (mode == "sample" and k == 5)
    ]


def plan_e6_population(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Selection-policy study on the pool setting.

    ``regen_finals`` already covers the default ``uniform`` policy, so this
    plan sweeps the two alternatives over the same grid of round sizes and
    trainers; together the two plans span the full product.
    """
    tasks: list[str] = []
    for policy in ("worst_first", "round_robin"):
        for clients in CLIENTS_PER_ROUND:
            parent = _pool_parent("pool", clients, policy, "fedavg")
            for trainer in POOL_TRAINERS:
                for seed in POOL_SEEDS:
                    tasks.append(
                        f"foa final --trainer {trainer} --parent {parent}"
                        f"{_pool_args(clients, policy, seed)}"
                        f" --aggregation fedavg{_seed_args(seed)} --skip-existing"
                    )
    return tasks


def plan_fixed_cohort(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Continuity ablation: the published five-writer cohort.

    Identical to the finals of the published pipeline - the frozen
    ``selected_outliers.json``, every client every round, every aggregation
    rule of each family - repeated over five seeds.
    """
    return [
        f"foa final --trainer {trainer}{_seed_args(seed)} --skip-existing"
        for trainer in ALL_TRAINERS
        for seed in E1_SEEDS
    ]


def plan_dual(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """The truly duplicated client: two participants over the same writer."""
    return [
        f"foa extreme --case dual{_seed_args(seed)} --skip-existing"
        for seed in DUAL_SEEDS
    ]


def plan_regen_base(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Baseline federated runs and the aggregation comparison.

    The baseline is run with and without the early-stopping rule, and the
    aggregation comparison once with the published method set and once with the
    server optimisers added.
    """
    return [
        "foa base-fl --skip-existing",
        "foa base-fl --parent prove_fl_stop --stop-when-global-below-clients --skip-existing",
        "foa all-aggs --skip-existing",
        "foa all-aggs --parent all_aggs_extended_fl --extended-aggregations --skip-existing",
    ]


def plan_regen_grids(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """Every hyperparameter sweep, including the three extreme cases."""
    tasks = [f"foa grid --method {method} --skip-existing" for method in GRID_METHODS]
    tasks += [
        f"foa grid --method extreme --case {case} --skip-existing"
        for case in EXTREME_CASES
    ]
    return tasks


def plan_regen_finals(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The main headline table: the pool setting with the default policy.

    Three round sizes x seven trainers, with the FedAvg-equivalent rule over
    five seeds, the capped/anchored rule over three, and one variant that adds
    the server optimisers.
    """
    tasks: list[str] = []
    policy = "uniform"

    for clients in CLIENTS_PER_ROUND:
        parent = _pool_parent("pool", clients, policy, "fedavg")
        for trainer in POOL_TRAINERS:
            for seed in POOL_BASELINE_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(clients, policy, seed)}"
                    f" --aggregation fedavg{_seed_args(seed)} --skip-existing"
                )

    for clients in CLIENTS_PER_ROUND:
        parent = _pool_parent("pool", clients, policy, "capped")
        for trainer in POOL_TRAINERS:
            for seed in POOL_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(clients, policy, seed)}"
                    f" --aggregation capped{_seed_args(seed)} --skip-existing"
                )

    clients = 10
    parent = _pool_parent("pool", clients, policy, "extended")
    for trainer in POOL_TRAINERS:
        for seed in POOL_SEEDS:
            tasks.append(
                f"foa final --trainer {trainer} --parent {parent}"
                f"{_pool_args(clients, policy, seed)}"
                f" --extended-aggregations{_seed_args(seed)} --skip-existing"
            )

    return tasks


def plan_pool_confirm_grids(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Small confirmation sweeps on the pool setting.

    The hyperparameter grids run on the cheap fixed cohort; these two sweeps
    check that the selected knowledge-distillation and EWC values still hold
    once the population is defined by the rule.
    """
    arguments = _pool_args(10, "uniform", 1)
    return [
        f"foa grid --method kd{arguments} --seed 1 --skip-existing",
        f"foa grid --method ewc{arguments} --seed 1 --skip-existing",
    ]


def plan_aggregation_hypotheses(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Hypotheses H1-H7 of the aggregation redesign, on the pool setting.

    H1/H3/H4/H5 are one task per rule, H2 sweeps the weighting of the unit
    server step, and H7 compares the two cyclic visiting orders.
    """
    tasks: list[str] = []
    clients, policy = 10, "uniform"

    for rule in HYPOTHESIS_RULES:
        parent = _pool_parent("hyp", clients, policy, rule)
        for trainer in HYPOTHESIS_TRAINERS:
            for seed in POOL_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(clients, policy, seed)}"
                    f" --aggregation {rule} --extended-aggregations"
                    f"{_seed_args(seed)} --skip-existing"
                )

    # H2: the same unit server step under the three weightings.
    for weighting in HYPOTHESIS_WEIGHTINGS:
        parent = _pool_parent("hyp", clients, policy, f"w_{weighting}")
        for trainer in HYPOTHESIS_TRAINERS:
            for seed in POOL_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(clients, policy, seed)}"
                    f" --aggregation con_delta_eta1 --extended-aggregations"
                    f" --weighting {weighting}{_seed_args(seed)} --skip-existing"
                )

    # H7: cyclic visiting order.
    for order in ("fixed", "shuffle"):
        parent = _pool_parent("hyp", clients, policy, f"order_{order}")
        for trainer in HYPOTHESIS_TRAINERS:
            for seed in POOL_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(clients, policy, seed)}"
                    f" --aggregation fedavg --client-order {order}"
                    f"{_seed_args(seed)} --skip-existing"
                )

    return tasks


#: Client count and policy of every task of the regularisation family, so the
#: sweeps and the finals they select from run on the same population.
REG_CLIENTS_PER_ROUND = 10
REG_POLICY = "uniform"

#: Task file the planner points ``foa select --emit-finals`` at when the sweeps
#: have not been selected yet.  It is written next to the results root, so a
#: second job array can be submitted straight from it.
REG_FINALS_TASK_FILE = "tasks_reg_finals.txt"


def reg_parent(
    space: str, anchor: str, eps: float, suffix: str = "", prefix: str = ""
) -> str:
    """
    Output folder of one regularisation final: space, anchor and budget.

    ``suffix`` separates the same selection executed under different aggregation
    rules.  Without it two rules would write into one folder and the second
    would be skipped by ``--skip-existing`` as though it had already run.
    ``prefix`` separates whole scenarios that share a results root - a cohort
    study and a population study of the same spaces, say.
    """
    name = f"reg_{space.replace('+', '_')}_{anchor}_eps{eps:g}".replace(".", "")
    if prefix:
        name = f"{prefix}{name}"
    return f"{name}_{suffix}" if suffix else name


def reg_final_task(
    space: str,
    anchor: str,
    eps: float,
    seed: int,
    hyperparameters: Optional[dict] = None,
    clients: int = REG_CLIENTS_PER_ROUND,
    policy: str = REG_POLICY,
    aggregation: str = "fedavg",
    extra_args: str = "",
    pool_args: Optional[str] = None,
    parent_suffix: str = "",
    parent_prefix: str = "",
) -> str:
    """
    One ``foa final`` line of the regularisation family.

    ``hyperparameters`` carries the values the constrained selection chose for
    this ``(space, anchor)`` at this budget - ``lam`` always, ``T`` for the
    softmax-based spaces, and the ``param_l2`` convention where the sweep set
    one.  Without them the line falls back to the trainer defaults, which is
    what the planner emitted before the selection existed.

    Args:
        aggregation: Rule or variant key the final is executed with.  A value
            selected under one rule and executed under another is the mismatch
            this study exists to remove, so the rule is explicit and the output
            folder carries it (see ``parent_suffix``).
        extra_args: Flags appended verbatim - the results root, the resolution
            and character set, the budget - so a plan can pin a setting the
            published defaults do not describe.
        pool_args: Replaces the ``--pool-frac`` population block entirely, for a
            plan that names its pool by path rather than by rule.
        parent_suffix: Appended to the output folder; give one per aggregation.
        parent_prefix: Prepended to the output folder; give one per scenario, so
            two scenarios can share a results root without colliding.
    """
    overrides = " ".join(
        f"{key}={_set_value(value)}"
        for key, value in sorted((hyperparameters or {}).items())
    )
    population = (
        _pool_args(clients, policy, seed)
        if pool_args is None
        else (f" {pool_args.strip()} --sampler-seed {seed}" if pool_args.strip() else "")
    )
    return (
        f"foa final --trainer AnchoredTrainer"
        f" --parent {reg_parent(space, anchor, eps, parent_suffix, parent_prefix)}"
        f"{population}"
        f"{' ' + extra_args.strip() if extra_args.strip() else ''}"
        f' --set space="{space}" anchor="{anchor}"'
        f"{' ' + overrides if overrides else ''}"
        f" --aggregation {aggregation}{_seed_args(seed)} --skip-existing"
    )


def _set_value(value) -> str:
    """Render a ``--set NAME=VALUE`` value the way the CLI parses it back."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return f"{value:g}" if isinstance(value, float) else str(value)


def regularisation_select_task(
    results_dir: Optional[Path] = None,
    eps_values: Sequence[float] = SELECTION_EPS,
    seeds: Sequence[int] = POOL_SEEDS,
) -> str:
    """
    The single ``foa select --emit-finals`` line that writes the finals.

    It runs the constrained selection over every stored ``(space, anchor)``
    sweep, at every budget, and writes the resulting ``foa final`` lines to
    :data:`REG_FINALS_TASK_FILE` - the task file of the second job array.
    """
    root = f" --root {results_dir}" if results_dir else ""
    out = (
        f"{Path(results_dir) / REG_FINALS_TASK_FILE}"
        if results_dir
        else REG_FINALS_TASK_FILE
    )
    return (
        f"foa select{root} --emit-finals"
        f" --eps {' '.join(f'{eps:g}' for eps in eps_values)}"
        f" --seeds {' '.join(str(int(seed)) for seed in seeds)}"
        f" --out {out}"
    )


def emit_regularisation_finals(
    root,
    eps_values: Sequence[float] = SELECTION_EPS,
    seeds: Sequence[int] = POOL_SEEDS,
    clients: int = REG_CLIENTS_PER_ROUND,
    policy: str = REG_POLICY,
    provider: str = DEFAULT_PROVIDER,
    aggregations: Sequence[str] = ("fedavg",),
    extra_args: str = "",
    pool_args: Optional[str] = None,
    parent_prefix: str = "",
) -> list[str]:
    """
    Final runs of the regularisation family, with the selected lambda and T.

    Reads the per-sweep selection files ``foa select --emit-finals`` wrote under
    ``root`` and turns each of them into one task per seed.  A sweep that no
    budget could select from contributes nothing, so an empty result means the
    selection has not run (or admitted nothing), never a run with the wrong
    hyperparameters.

    ``aggregations`` executes each selection under more than one rule - the
    plain FedAvg pair and, say, whichever aggregation an earlier phase found
    best - and gives each rule its own output folder, so the two never collide.
    ``extra_args`` and ``pool_args`` pin the setting the lines run in.
    """
    from federated_outlier_adaptation.grid_search.selection import collect_selections

    order = [
        (f"{eps:g}", space, anchor)
        for eps in eps_values
        for space in ANCHOR_SPACES
        for anchor in ANCHOR_KINDS
    ]
    ordering = {key: index for index, key in enumerate(order)}

    tasks: list[str] = []
    selections = collect_selections(root, eps_values)
    selections.sort(
        key=lambda payload: ordering.get(
            (
                f"{float(payload['eps']):g}",
                str(payload.get("space")),
                str(payload.get("anchor")),
            ),
            len(ordering),
        )
    )
    for payload in selections:
        space = str(payload.get("space"))
        anchor = str(payload.get("anchor"))
        eps = float(payload["eps"])
        hyperparameters = payload.get("hyperparameters") or {}
        for aggregation in aggregations:
            # One folder per rule.  With a single rule the suffix is empty and
            # the folder name is the one the planner has always emitted.
            suffix = "" if len(tuple(aggregations)) == 1 else str(aggregation)
            for seed in seeds:
                tasks.append(
                    with_provider(
                        reg_final_task(
                            space,
                            anchor,
                            eps,
                            int(seed),
                            hyperparameters,
                            clients,
                            policy,
                            aggregation=aggregation,
                            extra_args=extra_args,
                            pool_args=pool_args,
                            parent_suffix=suffix,
                            parent_prefix=parent_prefix,
                        ),
                        provider,
                    )
                )
    return tasks


def plan_regularisation_family(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The unified regularisation family on the pool setting.

    First every (space, anchor) sweep over the shared log-spaced lambda grid
    with two seeds, then the finals at three forgetting budgets.

    The finals must run with the lambda (and, for the softmax-based spaces, the
    temperature) that the constrained selection picked for their own sweep, and
    that value is only known once the sweep has finished.  The plan therefore
    has two shapes:

    - the results root already holds the per-sweep selection files, in which
      case the finals are emitted with the selected values;
    - it does not, in which case the plan ends with a single
      ``foa select --emit-finals`` line and emits **no** finals.  Running that
      line writes the finals into :data:`REG_FINALS_TASK_FILE`, which is
      submitted as a second job array.
    """
    tasks: list[str] = []
    for space in ANCHOR_SPACES:
        for anchor in ANCHOR_KINDS:
            for seed in GRID_SEEDS:
                tasks.append(
                    f"foa grid --method anchored --space {space} --anchor {anchor}"
                    f"{_pool_args(REG_CLIENTS_PER_ROUND, REG_POLICY, seed)}"
                    f" --seed {seed} --skip-existing"
                )

    finals = (
        emit_regularisation_finals(
            provider_results_root(results_dir, provider), provider=provider
        )
        if results_dir
        else []
    )
    if finals:
        return tasks + finals
    return tasks + [regularisation_select_task(results_dir)]


def plan_references(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Reference points the federated numbers are read against.

    Local fine-tuning (no aggregation at all) and a federated run that starts
    from a fresh model instead of the global one.
    """
    tasks = [
        f"foa local-finetune --trainer BaseTrainer --pool-frac {POOL_FRAC:g}"
        f"{_seed_args(seed)} --skip-existing"
        for seed in POOL_SEEDS
    ]
    parent = _pool_parent("pool_scratch", 10, "uniform", "fedavg")
    tasks += [
        f"foa final --trainer BaseTrainer --parent {parent}"
        f"{_pool_args(10, 'uniform', seed)}"
        f" --aggregation fedavg --init scratch{_seed_args(seed)} --skip-existing"
        for seed in POOL_SEEDS
    ]
    return tasks


def _smoke_budget() -> str:
    """Round / epoch / worker budget shared by every smoke training task."""
    return (
        f" --rounds {SMOKE_ROUNDS} --epochs {SMOKE_EPOCHS}"
        f" --outer-workers {SMOKE_OUTER_WORKERS} --inner-workers {SMOKE_INNER_WORKERS}"
    )


def _smoke_pool() -> str:
    """The main experimental setting, shrunk to the smoke round size."""
    return (
        f" --pool-frac {POOL_FRAC:g}"
        f" --clients-per-round {SMOKE_CLIENTS_PER_ROUND} --policy uniform --track-clients"
    )


def _smoke_extra(trainer: str) -> str:
    """Trainer-specific flags a smoke final needs on top of the pool setting."""
    if trainer == "AnchoredTrainer":
        return f' --set space="{SMOKE_ANCHOR_SPACE}" anchor="{SMOKE_ANCHOR_KIND}"'
    return ""


def _smoke_final(trainer: str, parent: str, options: str, provider: str = "") -> str:
    """One ``foa final`` smoke task on the pool setting."""
    tag = f" --provider {provider}" if provider else ""
    return (
        f"foa final{tag} --trainer {trainer} --parent {parent}"
        f"{_smoke_pool()}{options}"
        f" --seed {SMOKE_SEED}{_smoke_budget()} --skip-existing"
    )


def plan_smoke(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Cheap end-to-end checks of every code path the study uses.

    Two rounds and one local epoch per task, small worker counts, and the whole
    plan pinned to ``smoke_*`` output folders and to the ``smoke_global_fl``
    model name, so it can be pointed at any results root without touching a
    published artefact.  Point ``FOA_RESULTS_DIR`` at a scratch root all the
    same; the plan is meant to prove that the code runs, not to produce a
    number.

    Covered: rule-based pool selection, federated pre-training, local
    fine-tuning, every trainer through the four (scenario, metadata) jobs, two
    extended aggregation rules, the cyclic shuffle order, an extreme case, an
    anchored sweep, the two additional datasets, the constrained selection
    reading what the sweep produced, and the analysis of the
    constraint-respecting signals over the whole scratch root.

    ``select-outliers``, ``global-train-fl``, ``select`` and ``signals`` have
    neither a parent folder nor a resume flag; they are idempotent by
    construction (the first two reuse an existing artefact unless ``--force``
    is given, the last two only read) and are therefore the four tasks without
    ``--skip-existing``.  The two read-only tasks are appended last, so the
    index of every training task is unaffected by their presence.
    """
    tasks: list[str] = []

    # The population the finals draw from.  Deriving it costs seconds and the
    # runs resolve the same pool from the accuracy file if it is missing, so
    # this task carries no ordering constraint for the rest of the plan.
    tasks.append(f"foa select-outliers --mode pool --pool-frac {POOL_FRAC:g}")

    # An alternative starting point, written under its own model name so the
    # published global model is never touched.
    tasks.append(
        f"foa global-train-fl --rounds {SMOKE_ROUNDS} --local-epochs {SMOKE_EPOCHS}"
        f" --global-name {SMOKE_GLOBAL_NAME}"
    )

    # The no-federation reference point.
    tasks.append(
        f"foa local-finetune --trainer BaseTrainer --pool-frac {POOL_FRAC:g}"
        f" --epochs {SMOKE_EPOCHS} --parent {SMOKE_PREFIX}_local_finetune"
        f" --seed {SMOKE_SEED} --skip-existing"
    )

    # Every trainer, on the main setting with the FedAvg-equivalent rule.
    for trainer in SMOKE_TRAINERS:
        tasks.append(
            _smoke_final(
                trainer,
                f"{SMOKE_PREFIX}_final",
                f" --aggregation fedavg{_smoke_extra(trainer)}",
            )
        )

    # Two opt-in parallel rules: server-side anchoring (H4) and the robust
    # coordinate-wise median (H3).  Both live in the concurrent/delta family
    # only, so each of these tasks runs a single job.
    tasks.append(
        _smoke_final(
            "BaseTrainer",
            f"{SMOKE_PREFIX}_agg_anchor",
            " --aggregation con_delta_anchor_lam01 --extended-aggregations",
        )
    )
    tasks.append(
        _smoke_final(
            "BaseTrainer",
            f"{SMOKE_PREFIX}_agg_median",
            " --aggregation con_delta_median --extended-aggregations",
        )
    )

    # H7: the cyclic visiting order.
    tasks.append(
        _smoke_final(
            "BaseTrainer",
            f"{SMOKE_PREFIX}_cyclic_shuffle",
            " --aggregation fedavg --client-order shuffle",
        )
    )

    # The duplicated-client extreme case.
    tasks.append(
        f"foa extreme --case dual --parent {SMOKE_PREFIX}_extreme_dual"
        f" --seed {SMOKE_SEED}{_smoke_budget()} --skip-existing"
    )

    # One sweep through the grid engine, into its own folder.
    grid_parent = (
        f"{SMOKE_PREFIX}_grid_anchored_{SMOKE_GRID_SPACE}_{SMOKE_GRID_ANCHOR}"
        "_grid_search"
    )
    tasks.append(
        f"foa grid --method anchored --space {SMOKE_GRID_SPACE}"
        f" --anchor {SMOKE_GRID_ANCHOR} --parent {grid_parent}"
        f" --seed {SMOKE_SEED}{_smoke_budget()} --skip-existing"
    )


    # The constrained selection, reading the sweep that just ran, and the
    # signal analysis over everything the plan produced.  Both only read, so
    # they are appended last and carry no parent folder.
    root = f" --root {results_dir}" if results_dir else ""
    tasks.append(f"foa select{root} --eps 0.005")
    tasks.append(f"foa signals{root} --eps 0.005")

    return tasks


# --------------------------------------------------------------------------- #
# the lean plans
# --------------------------------------------------------------------------- #
#: Distance spaces of the lean sweep: the two knowledge-transfer objectives and
#: their combination with the Fisher penalty.  The remaining spaces of the full
#: family are covered by the replication plan on the other datasets.
LEAN_SWEEP_SPACES = ("kd", "ntd", "kd+fisher")

#: Penalty weights of a lean sweep: five points over four decades, i.e. the
#: published range with its two extremes dropped.  Those extremes are the ones
#: the constrained selection has never picked - ``1e-3`` is indistinguishable
#: from no penalty and ``1000`` freezes the client update - so removing them
#: costs no information and saves two sevenths of every sweep.
LEAN_LAMS = (1e-2, 1e-1, 1, 10, 100)

#: Temperatures of a lean sweep: one sharp and one soft teacher instead of four.
LEAN_TEMPERATURES = (2, 8)

#: Distance spaces the replication plan sweeps on the other datasets: one per
#: mechanism (output space, curvature, logits, parameters, non-true classes).
LEAN_REPLICA_SPACES = ("kd", "fisher", "logit_l2", "param_l2", "ntd")

#: Round size of the lean main setting and of the selection study.
LEAN_CLIENTS_PER_ROUND = 10
LEAN_SELECTION_CLIENTS = 5

#: Seeds of the lean plans.
LEAN_SEED = 1
LEAN_FINAL_SEEDS = (1, 2, 3, 4, 5)
LEAN_STUDY_SEEDS = (1, 2, 3)
LEAN_REPLICA_SEEDS = (1,)

#: Forgetting budget the lean regularisation finals are selected at.
LEAN_EPS = 0.005

#: Policies compared in the lean selection study.
LEAN_POLICIES = ("uniform", "worst_first")


#: The early-stopped FedAvg baseline, as a ``--set`` override.
EARLY_STOPPED = ' --set early_stopping=true'

#: Aggregation hypotheses of the lean plan: ``(tag, flags)``.  Each tag becomes
#: the output folder, so the arms never collide.
LEAN_HYPOTHESES = (
    ("con_delta_eta05", " --aggregation con_delta_eta05 --extended-aggregations"),
    ("w_uniform", " --aggregation con_delta_eta1 --extended-aggregations --weighting uniform"),
    ("w_capped", " --aggregation con_delta_eta1 --extended-aggregations --weighting capped"),
    ("con_delta_median", " --aggregation con_delta_median --extended-aggregations"),
    (
        "con_delta_trimmed_mean",
        " --aggregation con_delta_trimmed_mean --extended-aggregations",
    ),
    (
        "con_delta_anchor_lam01",
        " --aggregation con_delta_anchor_lam01 --extended-aggregations",
    ),
    ("con_delta_fedavgm", " --aggregation con_delta_fedavgm --extended-aggregations"),
    ("con_delta_fedadam", " --aggregation con_delta_fedadam --extended-aggregations"),
    # One task for the cyclic family: the `cyclic` variant key runs every
    # schedule of the two sequential families and skips the parallel ones.
    ("cyclic_all", " --aggregation cyclic"),
    ("order_shuffle", " --aggregation fedavg --client-order shuffle"),
)

#: The replication plan keeps the first six hypotheses, which are the ones the
#: NIST results single out as worth reproducing.
LEAN_REPLICA_HYPOTHESES = LEAN_HYPOTHESES[:6]


def _epochs_args(provider: str) -> str:
    """Local-epoch override of a provider, empty where the published one holds."""
    return ""


def _lean_trainers() -> tuple[tuple[str, str], ...]:
    """
    The two arms every lean comparison is made between.

    The early-stopped baseline and the distillation trainer share the same local
    stopping rule, so a difference between them is a difference of objective and
    not of local budget.
    """
    return (("BaseTrainer", EARLY_STOPPED), ("DistillationTrainer", ""))


def _lean_sweep_task(
    space: str,
    seed: int,
    provider: str,
    anchor: str = "frozen",
) -> str:
    """One lean anchored sweep: short ranges, no per-client tracking."""
    from federated_outlier_adaptation.grid_search.anchored import TEMPERATURE_SPACES

    lams = " ".join(f"{lam:g}" for lam in LEAN_LAMS)
    temperatures = (
        " --temperatures " + " ".join(f"{value:g}" for value in LEAN_TEMPERATURES)
        if space in TEMPERATURE_SPACES
        else ""
    )
    return (
        f"foa grid --method anchored --space {space} --anchor {anchor}"
        f"{_pool_args(LEAN_CLIENTS_PER_ROUND, REG_POLICY, seed, track=False)}"
        f" --lams {lams}{temperatures}"
        f"{_epochs_args(provider)} --seed {seed} --skip-existing"
    )


def plan_lean_nist_sweeps(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The regularisation sweeps of the lean protocol: three spaces, one anchor.

    The frozen anchor is the shipped global model, which is the only anchor the
    constraint of this study allows a server to keep; the moving anchor is an
    ablation the full ``regularisation_family`` plan still covers.  One seed is
    enough because a sweep only has to rank configurations - the seeds are spent
    on the finals the selection picks.
    """
    return [
        _lean_sweep_task(space, LEAN_SEED, provider) for space in LEAN_SWEEP_SPACES
    ]


def plan_lean_nist_finals_pre(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The finals that do not wait for a selection: the baselines and the policies.

    Three arms:

    - FedAvg with the published local budget, five seeds - the reference point;
    - FedAvg with the early-stopping rule of the regularised trainers, five
      seeds - so that the comparison isolates the objective;
    - the selection-policy study at half the round size, over both policies and
      both arms, three seeds.
    """
    tasks: list[str] = []
    epochs = _epochs_args(provider)

    for tag, extra in (("fedavg", ""), ("fedavg_es", EARLY_STOPPED)):
        parent = f"lean_m{LEAN_CLIENTS_PER_ROUND}_uniform_{tag}"
        for seed in LEAN_FINAL_SEEDS:
            tasks.append(
                f"foa final --trainer BaseTrainer --parent {parent}"
                f"{_pool_args(LEAN_CLIENTS_PER_ROUND, 'uniform', seed)}"
                f" --aggregation fedavg{extra}{epochs}"
                f"{_seed_args(seed)} --skip-existing"
            )

    for policy in LEAN_POLICIES:
        for trainer, overrides in _lean_trainers():
            parent = f"lean_sel_m{LEAN_SELECTION_CLIENTS}_{policy}_{_arm_tag(trainer)}"
            for seed in LEAN_STUDY_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(LEAN_SELECTION_CLIENTS, policy, seed)}"
                    f" --aggregation fedavg{overrides}{epochs}"
                    f"{_seed_args(seed)} --skip-existing"
                )
    return tasks


def _arm_tag(trainer: str) -> str:
    """Short output-folder tag of a lean arm."""
    return "fedavg_es" if trainer == "BaseTrainer" else "kd"


def plan_lean_nist_reg_finals(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    Finals of the lean sweeps, at the single forgetting budget of the protocol.

    Like ``regularisation_family``, the lambda (and temperature) must come from
    the constrained selection, so the plan emits the finals when the selection
    files are already under the results root and otherwise the single
    ``foa select --emit-finals`` line that writes them.
    """
    finals = (
        emit_regularisation_finals(
            provider_results_root(results_dir, provider),
            eps_values=(LEAN_EPS,),
            seeds=LEAN_STUDY_SEEDS,
            clients=LEAN_CLIENTS_PER_ROUND,
            policy=REG_POLICY,
            provider=provider,
        )
        if results_dir
        else []
    )
    if finals:
        return finals
    return [
        regularisation_select_task(
            results_dir, eps_values=(LEAN_EPS,), seeds=LEAN_STUDY_SEEDS
        )
    ]


def _hypothesis_tasks(
    hypotheses,
    seeds: Sequence[int],
    provider: str,
    prefix: str = "lean_hyp",
    trainers=None,
) -> list[str]:
    """One final per (hypothesis, arm, seed) on the lean main setting."""
    epochs = _epochs_args(provider)
    tasks: list[str] = []
    for tag, flags in hypotheses:
        for trainer, overrides in trainers or _lean_trainers():
            parent = f"{prefix}_m{LEAN_CLIENTS_PER_ROUND}_uniform_{tag}_{_arm_tag(trainer)}"
            for seed in seeds:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(LEAN_CLIENTS_PER_ROUND, 'uniform', seed)}"
                    f"{flags}{overrides}{epochs}"
                    f"{_seed_args(seed)} --skip-existing"
                )
    return tasks


def plan_lean_agg_hypotheses(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The aggregation redesign, reduced to the arms that carry a hypothesis.

    Ten rules - one server step size, two weightings, two robust aggregators,
    server anchoring, two server optimisers, the cyclic schedules and the
    shuffled visiting order - against the two arms, three seeds each.
    """
    return _hypothesis_tasks(LEAN_HYPOTHESES, LEAN_STUDY_SEEDS, provider)


def plan_lean_replica(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """
    The lean protocol replicated on one further dataset.

    Five one-seed sweeps over the short ranges, the finals the constrained
    selection implies at the single budget, the six aggregation hypotheses the
    NIST results single out, and the selection-policy study at half the round
    size.  Everything is emitted for ``provider``; ``foa matrix --provider`` adds
    the flag to every line and resolves the pool under that dataset's root.
    """
    tasks = [
        _lean_sweep_task(space, LEAN_SEED, provider) for space in LEAN_REPLICA_SPACES
    ]
    tasks += plan_lean_nist_reg_finals(results_dir, provider)
    tasks += _hypothesis_tasks(
        LEAN_REPLICA_HYPOTHESES,
        LEAN_REPLICA_SEEDS,
        provider,
        prefix="lean_replica_hyp",
        trainers=(("DistillationTrainer", ""),),
    )

    epochs = _epochs_args(provider)
    for policy in LEAN_POLICIES:
        for trainer, overrides in _lean_trainers():
            parent = f"lean_replica_sel_m{LEAN_SELECTION_CLIENTS}_{policy}_{_arm_tag(trainer)}"
            for seed in LEAN_REPLICA_SEEDS:
                tasks.append(
                    f"foa final --trainer {trainer} --parent {parent}"
                    f"{_pool_args(LEAN_SELECTION_CLIENTS, policy, seed)}"
                    f" --aggregation fedavg{overrides}{epochs}"
                    f"{_seed_args(seed)} --skip-existing"
                )
    return tasks


def plan_regen_extreme(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """The three extreme cases over three seeds."""
    return [
        f"foa extreme --case {case}{_seed_args(seed)} --skip-existing"
        for case in EXTREME_CASES
        for seed in EXTREME_SEEDS
    ]


#: Order in which ``regen_all`` chains its parts: the pool has to exist before
#: anything reads it, the baselines need nothing, the sweeps come before the
#: finals that consume their hyperparameters, and the extreme cases last.
REGEN_ORDER = (
    "pool",
    "regen_base",
    "regen_grids",
    "pool_confirm_grids",
    "regen_finals",
    "regen_extreme",
)


def plan_regen_all(
    results_dir: Optional[Path] = None, provider: str = DEFAULT_PROVIDER
) -> list[str]:
    """Full regeneration in dependency order; see :data:`REGEN_ORDER`."""
    tasks: list[str] = []
    for plan in REGEN_ORDER:
        tasks += PLANS[plan](results_dir, provider)
    return tasks


PLANS: dict[str, Callable[[Optional[Path]], list[str]]] = {
    "e1_seeds": plan_e1_seeds,
    "e2_e5_grids": plan_e2_e5_grids,
    "e2_e5_finals": plan_e2_e5_finals,
    "e6_pools": plan_e6_pools,
    "pool": plan_pool,
    "e6_population": plan_e6_population,
    "fixed_cohort": plan_fixed_cohort,
    "pool_confirm_grids": plan_pool_confirm_grids,
    "aggregation_hypotheses": plan_aggregation_hypotheses,
    "regularisation_family": plan_regularisation_family,
    "references": plan_references,
    "lean_nist_sweeps": plan_lean_nist_sweeps,
    "lean_nist_finals_pre": plan_lean_nist_finals_pre,
    "lean_nist_reg_finals": plan_lean_nist_reg_finals,
    "lean_agg_hypotheses": plan_lean_agg_hypotheses,
    "lean_replica": plan_lean_replica,
    "smoke": plan_smoke,
    "dual": plan_dual,
    "regen_base": plan_regen_base,
    "regen_grids": plan_regen_grids,
    "regen_finals": plan_regen_finals,
    "regen_extreme": plan_regen_extreme,
    "regen_all": plan_regen_all,
}


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def with_provider(task: str, provider: str) -> str:
    """
    Insert ``--provider`` right after the subcommand of a task line.

    A task that already names its dataset - the smoke plan deliberately visits
    all three - is left alone, so replicating a plan never produces a line with
    two conflicting ``--provider`` flags.
    """
    if provider == DEFAULT_PROVIDER or " --provider " in f"{task} ":
        return task
    parts = task.split(" ", 2)
    head = " ".join(parts[:2])
    tail = f" {parts[2]}" if len(parts) > 2 else ""
    return f"{head} --provider {provider}{tail}"


#: Fan-out of a sweep task.  The grid engine's own defaults (2 outer threads x
#: 8 inner processes) put sixteen CUDA contexts on one accelerator, and a sweep
#: trains at batch 512, so on a 40 GB A100 the sixteenth process runs out of
#: memory.  Six processes fit comfortably, so every generated ``foa grid`` line
#: carries these worker counts.  The engine defaults are deliberately left
#: alone: an existing command line keeps producing exactly what it produced.
GRID_OUTER_WORKERS = 2
GRID_INNER_WORKERS = 3


def with_grid_workers(task: str) -> str:
    """
    Pin the fan-out of a generated sweep task, unless the plan already set it.

    Only ``foa grid`` lines are touched; the finals run two by two at batch 64
    and fit as they are.
    """
    if not task.startswith("foa grid "):
        return task
    padded = f"{task} "
    if " --outer-workers " in padded or " --inner-workers " in padded:
        return task
    return f"{task} --outer-workers {GRID_OUTER_WORKERS} --inner-workers {GRID_INNER_WORKERS}"


def plan_tasks(
    plan: str,
    results_dir: Optional[Path] = None,
    provider: str = DEFAULT_PROVIDER,
) -> list[str]:
    """
    Expand a plan into its task lines.

    Args:
        plan: Plan name, see :data:`PLANS`.
        results_dir: Results root used to resolve client pool files.
        provider: Dataset the plan runs on.  The published dataset emits the
            task lines unchanged; any other one adds ``--provider`` to every
            task and resolves the pool files under that dataset's results root.

    Raises:
        KeyError: When the plan does not exist.
    """
    if plan not in PLANS:
        raise KeyError(f"Unknown plan {plan!r}; expected one of {sorted(PLANS)}")
    tasks = PLANS[plan](results_dir, provider)
    return [with_grid_workers(with_provider(task, provider)) for task in tasks]


def write_tasks(tasks: Sequence[str], path) -> Path:
    """Write one task per line, creating the parent directory if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        handle.write("\n".join(tasks) + "\n")
    return out


def estimate_gpu_hours(
    num_tasks: int, minutes_per_task: float = DEFAULT_MINUTES_PER_TASK
) -> float:
    """
    GPU hours a plan is expected to consume.

    The per-task cost is an *assumption* calibrated on one timed run, not a
    measurement of every task; sweeps and large pools run considerably longer
    than a single final run.
    """
    return float(num_tasks) * float(minutes_per_task) / 60.0


def plan_summary(
    plans: Optional[Iterable[str]] = None,
    results_dir: Optional[Path] = None,
    minutes_per_task: float = DEFAULT_MINUTES_PER_TASK,
    provider: str = DEFAULT_PROVIDER,
) -> list[tuple[str, int, float]]:
    """``(plan, task count, estimated GPU hours)`` for every requested plan."""
    names = list(plans) if plans is not None else list(PLANS)
    summary = []
    for name in names:
        count = len(plan_tasks(name, results_dir, provider))
        summary.append((name, count, estimate_gpu_hours(count, minutes_per_task)))
    return summary


def main():  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment task lists.")
    parser.add_argument("--plan", default=None, help="Plan to expand.")
    parser.add_argument("--out", default=None, help="Task file to write.")
    parser.add_argument(
        "--minutes-per-task", type=float, default=DEFAULT_MINUTES_PER_TASK
    )
    args = parser.parse_args()

    if args.plan is None:
        for name, count, hours in plan_summary(minutes_per_task=args.minutes_per_task):
            print(f"{name:16s} tasks={count:4d} estimated_gpu_hours={hours:7.1f}")
        return

    tasks = plan_tasks(args.plan)
    if args.out:
        write_tasks(tasks, args.out)
        print(f"Wrote {len(tasks)} tasks to {args.out}")
    else:
        print("\n".join(tasks))
    print(
        f"plan={args.plan} tasks={len(tasks)} "
        f"estimated_gpu_hours={estimate_gpu_hours(len(tasks), args.minutes_per_task):.1f}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
