"""
The aggregation screen's task lines, addressed at one study.

The protocol is the one every stage of this study shares - the cohort, its fold
book, the old book, g-0, nine of ten clients per round - and the only thing a
line varies is the **server rule** and the coefficients it runs at.  The rules
themselves come from :mod:`.agg_cells`, unchanged: the cell table is the
science, and a line is only its address.

Why this module exists at all
-----------------------------
The screen's line builder used to live in ``tools/make_stage6_screen.py``, which
hard-codes a results root and a sixteen-of-twenty sampler from a study that has
since been deleted.  Rather than edit a file whose outputs were already
submitted, this carries a config-driven builder over the same table.

Method grouping
---------------
The full-horizon stage that follows a screen runs **one line per method at its
best coefficients**, so "method" has to be a definition rather than an
intuition.  :func:`agg_method_of` groups the 169 cells into 18: a coefficient
row like ``eta_*`` is one method at several settings, while ``weight_uniform``
and ``weight_capped`` are two methods - a weighting rule is not a coefficient of
another weighting rule - and the two controls are likewise distinct, because a
control with the oracle stop rule armed is not a setting of the control without
it.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from federated_outlier_adaptation.training import agg_cells, five_cells, reg_cells

#: Every path in a task line goes through this, which the sbatch sets per study.
#: A study therefore cannot name another study's folder even by accident, and
#: the sbatch guard refuses a line that tries.
ROOT = "$FOA_STUDY_DIR"
BOOKS = f"{ROOT}/fold_books"
POOLS = f"{ROOT}/outliers"
TABLES = f"{ROOT}/tables"
OLD_FILE = f"{POOLS}/old_data.json"
OLD_BOOK = f"{BOOKS}/old_data.foldbook.npz"

#: The Fisher of the shipped model, for the penalties that need it.  The fold is
#: substituted by the runner, which reads it from the study's own selection
#: record - see ``study_phase.sbatch``.
FISHER = f"{ROOT}/g0_fold$G0_FOLD/global_results/fisher"


def fisher_dir(cfg=None) -> str:
    """
    Where a study's g-0 left its Fisher information.

    Two layouts, because two things differ.  A cross-validated study trains one
    g-0 per fold and the winner's fold number is only known at submit time, so
    its path carries ``$G0_FOLD`` and the runner substitutes it.  A study that
    trains a single g-0 has no fold to substitute - and a non-NIST provider
    writes under ``<results>/<provider>/``, one level deeper - so its path is
    fixed and complete.

    A study of the second kind that inherited the first kind's path would name
    ``$G0_FOLD`` and be refused by the runner for having no selection record to
    read it from, which is the right failure but an obscure way to learn this.
    """
    if cfg is None or getattr(cfg, "cross_validated", True):
        return FISHER
    provider = getattr(cfg, "provider", "nist")
    root = ROOT if provider == "nist" else f"{ROOT}/{provider}"
    return f"{root}/g0/global_results/fisher" if provider == "nist" \
        else f"{ROOT}/g0/{provider}/global_results/fisher"


def folds_of(cfg=None) -> tuple:
    """
    The folds a study's stages emit lines for.

    ``FOLDS`` is the five a book is always built with.  What a study *trains*
    on is its own: the digit study cross-validates, the transfer studies do not.
    Reading this rather than the constant is what lets one generator serve both.
    """
    return tuple(getattr(cfg, "folds", None) or FOLDS)

#: The two schedules a ``--aggregation fedavg`` task covers.
FAMILIES = reg_cells.FAMILIES

FOLDS = (1, 2, 3, 4, 5)
SCREEN_ROUNDS = agg_cells.SCREEN_ROUNDS
FULL_ROUNDS = agg_cells.FULL_ROUNDS
LOCAL_EPOCHS = 5
BATCH_SIZE = 64

#: How many of each method's best cells a later combination stage would cross.
TOP_K = 3


def setting(cfg) -> str:
    """
    The dataset flags every line of this study carries.

    NIST needs its resolution and its label subset named on every line, because
    one cache serves both and a line that omits them inherits a default that is
    wrong for this study.  Another provider names itself and supplies its own
    shape, so repeating image flags there would be noise at best and a
    contradiction at worst.
    """
    if getattr(cfg, "provider", "nist") != "nist":
        return f"--provider {cfg.provider}"
    return f"--resolution 28 --classes {cfg.classes}"


def cohort_file(cfg) -> str:
    """Where this study's cohort list lives."""
    return f"{POOLS}/{cfg.cohort_file_name}"


def cohort_book(cfg) -> str:
    """Where this study's cohort fold book lives."""
    return f"{BOOKS}/{cfg.cohort_book_name}.foldbook.npz"


#: A prefix here means "these cells are one method at several settings".  A cell
#: whose id matches none of them is a method on its own - see the module
#: docstring for why ``weight_`` and ``control_`` are deliberately absent.
AGG_METHOD_PREFIXES = (
    # "weight_q" before "eta_" is not an ordering accident: the weighting cells
    # are emitted on the eta rule, so without their own prefix each coefficient
    # would form a method of one and the selector would crown four winners for
    # one knob instead of choosing between them.
    "weight_q",
    "eta_", "median", "trimmed_", "anchor_", "fedavgm_",
    "fedadam_", "fedyogi_", "seq_mix_", "seq_order_",
)


def agg_method_of(cell: Dict[str, Any]) -> str:
    """The method a screening cell belongs to."""
    identifier = cell["id"]
    for prefix in AGG_METHOD_PREFIXES:
        if identifier.startswith(prefix):
            return prefix.rstrip("_")
    return identifier


def agg_methods() -> List[tuple]:
    """``(path, method)`` pairs of the aggregation screen, in table order."""
    seen: List[tuple] = []
    for cell in agg_cells.screen_cells():
        key = (cell["path"], agg_method_of(cell))
        if key not in seen:
            seen.append(key)
    return seen


#: The methods a full-horizon stage would run, one line per fold each.
AGG_METHODS = agg_methods()


def _base(cfg, parent: str, fold: int, rounds: int, seed: int, trainer: str) -> str:
    """The protocol every federated line of this study shares."""
    return (
        f"foa final --results-dir {ROOT} {setting(cfg)}"
        f" --model {getattr(cfg, 'model', 'fedavg_cnn')}"
        f" --trainer {trainer} --parent {parent}"
        f" --init global --global-name g0"
        f" --outliers-file {cohort_file(cfg)}"
        f" --fold-book {cohort_book(cfg)} --fold {fold}"
        f" --old-book {OLD_BOOK} --old-clients-file {OLD_FILE} --old-fold all"
        f" --policy uniform --clients-per-round {cfg.clients_per_round}"
        f" --sampler-seed {seed} --track-clients"
        f" --rounds {rounds} --epochs {LOCAL_EPOCHS} --batch-size {BATCH_SIZE}"
        f" --eval-batch-size 256 --save-final-model --seed {fold}"
    )


def _agg_flags(flags: Dict[str, Any]) -> str:
    """A cell's server coefficients, in a stable order and at full precision."""
    order = (
        ("server_eta", "--server-eta"), ("weighting", "--weighting"),
        ("trim_frac", "--trim-frac"), ("server_anchor", "--server-anchor"),
        ("server_beta", "--server-beta"), ("server_lr", "--server-lr"),
        ("server_tau", "--server-tau"), ("seq_mix_alpha", "--seq-mix-alpha"),
        ("client_order", "--client-order"),
    )
    parts = []
    for name, flag in order:
        if name in flags:
            value = flags[name]
            # repr for floats: {:g} is six significant figures, which is fine
            # for a label and wrong for a coefficient.
            parts.append(f"{flag} {value!r}" if isinstance(value, float)
                         else f"{flag} {value}")
    if flags.get("stop_when_global_below_clients"):
        parts.append("--stop-when-global-below-clients")
    return " ".join(parts)


def resolve_agg_flags(flags: Dict[str, Any], cfg, rounds: int) -> Dict[str, Any]:
    """
    Turn a cell's stage-independent settings into this stage's coefficients.

    THREE OF THE SERVER KNOBS ARE TIMESCALES WRITTEN AS COEFFICIENTS, and their
    meaning depends on quantities that change between stages - the number of
    rounds, or the number of participating clients. A cell that stored the
    coefficient would therefore be a different intervention on the screen than
    at the full horizon, and a different one again at another federation size.
    That is not hypothetical: it is why settings chosen on a quarter-length
    screen could not hold the run they were chosen for.

    So a cell stores the quantity whose meaning is stable and this turns it into
    the number the trainer wants. The emitted line still carries a concrete
    coefficient, so a task file remains readable and re-runnable on its own.

        anchor_halflife_r  ->  server_anchor   depends on the round count
        mix_retention      ->  seq_mix_alpha   depends on the client count
        trim_count         ->  trim_frac       depends on the client count
        weight_q           ->  weighting       depends on nothing; renamed only
    """
    out = dict(flags)
    clients = int(getattr(cfg, "clients_per_round", 0)) or 1

    if "anchor_halflife_r" in out:
        # lambda_s = 1 - 2 ** (-1/h), h the half-life in ROUNDS
        halflife = float(out.pop("anchor_halflife_r")) * float(rounds)
        out["server_anchor"] = 1.0 - 2.0 ** (-1.0 / halflife)

    if "mix_retention" in out:
        # r = (1 - alpha) ** K after a full pass, so alpha = 1 - r ** (1/K)
        retention = float(out.pop("mix_retention"))
        out["seq_mix_alpha"] = 1.0 - retention ** (1.0 / clients)

    if "trim_count" in out:
    # A COUNT DOES NOT TRANSFER DOWNWARD WITHOUT A LIMIT. Trimming t from each
    # end needs 2t < K, so t = 3 is meaningless where four clients participate:
    # it would discard everything and the server would silently fall back to the
    # plain mean. Clamping keeps the setting the strongest one the federation
    # can actually express, and keeps the arm comparable with the sizes where
    # the full count fits.
        # The trainer takes a fraction and floors it; the half step keeps
        # floating point from dropping floor(beta*K) to count - 1.
        count = min(int(out.pop("trim_count")), max((clients - 1) // 2, 0))
        out["trim_frac"] = (count + 0.5) / clients

    if "weight_q" in out:
        out["weighting"] = float(out.pop("weight_q"))

    return out



def agg_line(cfg, cell: Dict[str, Any], fold: int, rounds: int, seed: int) -> str:
    """One aggregation cell, at the screening or the full horizon."""
    prefix = f"{cfg.tag}_agg" if rounds == SCREEN_ROUNDS else f"{cfg.tag}_aggfull"
    line = _base(cfg, f"{prefix}_{cell['id']}_fold{fold}", fold, rounds, seed,
                 "BaseTrainer")
    line += f" --aggregation {cell['rule']} --extended-aggregations"
    line += " --outer-workers 2 --inner-workers 1"
    extra = _agg_flags(resolve_agg_flags(cell["flags"], cfg, rounds))
    return line + (f" {extra}" if extra else "")


# --------------------------------------------------------------------------- #
# the regularisation screen
# --------------------------------------------------------------------------- #
#: The anchor every penalty measures distance from: the frozen shipped model.
ANCHOR = "frozen"

#: The methods the full-horizon regularisation stage runs, per family.
REG_METHODS = reg_cells.methods()


def _reg_set(cell: Dict[str, Any], cfg=None) -> str:
    """The cell's penalty as ``--set NAME=VALUE`` tokens, in a stable order."""
    if cell["trainer"] != "AnchoredTrainer":
        return ""
    parts = [f"space={cell['space']}", f"anchor={ANCHOR}"]
    for name in ("lam", "T", "mix"):
        if name in cell["hypers"]:
            # repr, not {:g}: six significant figures is fine for a label and
            # wrong for a coefficient.
            parts.append(f"{name}={float(cell['hypers'][name])!r}")
    if cell["needs_fisher"]:
        parts.append(f"fisher_path={fisher_dir(cfg)}")
    return "--set " + " ".join(parts)


def reg_line(cfg, cell: Dict[str, Any], fold: int, rounds: int, seed: int,
             family: Optional[str] = None) -> str:
    """
    One regularisation cell; both families per task, as the stage always ran.

    Args:
        family: Which family's selection this line exists to serve.  The screen
            leaves it ``None`` - one line per cell, and the cell ids are unique.
            The **full-horizon** stage must set it, because the winners are
            chosen per (method, family) and the same cell often wins in both:
            without the tag two tasks would name one output folder, race each
            other into it, and leave the selector unable to say which family's
            result it had read.  The task still runs *both* families either way;
            the tag records which one the folder is read for.
    """
    prefix = f"{cfg.tag}_reg" if rounds == SCREEN_ROUNDS else f"{cfg.tag}_regfull"
    if family is not None:
        prefix = f"{prefix}_{family}"
    line = _base(cfg, f"{prefix}_{cell['id']}_fold{fold}", fold, rounds, seed,
                 cell["trainer"])
    line += " --aggregation fedavg --outer-workers 2 --inner-workers 1"
    if cell["fedprox"]:
        line += " --fedprox-convention"
    penalty = _reg_set(cell, cfg)
    return line + (f" {penalty}" if penalty else "")


def hybrid_line(cfg, cell: Dict[str, Any], fold: int, seed: int) -> str:
    """One kd+fisher blend at the full horizon."""
    return reg_line(cfg, cell, fold, FULL_ROUNDS, seed)


# --------------------------------------------------------------------------- #
# the cross, and the extreme cases the winner is run on
# --------------------------------------------------------------------------- #
#: How many of each side the combination stage crosses.
COMBO_TOP_K = TOP_K


def combo_line(cfg, agg: Dict[str, Any], reg: Dict[str, Any], fold: int,
               seed: int) -> str:
    """
    One server rule crossed with one client penalty, at the full horizon.

    Unlike the two screens, a combination names **one** aggregation rule, which
    lives in exactly one family - so a task produces one result rather than two,
    and the concurrent and sequential halves of the cross stay separate
    experiments with their own winners.
    """
    line = _base(
        cfg, f"{cfg.tag}_combo_{agg['id']}_{reg['id']}_fold{fold}", fold,
        FULL_ROUNDS, seed, reg["trainer"],
    )
    line += f" --aggregation {agg['rule']} --extended-aggregations"
    line += " --outer-workers 1 --inner-workers 1"
    if reg.get("fedprox"):
        line += " --fedprox-convention"
    # RESOLVE, do not emit what is stored. A cell may hold the stage-independent
    # form of a knob - a half-life in rounds, a retention over clients, a count
    # of clients - and _agg_flags has no flag for those names, so emitting the
    # stored dictionary drops the coefficient silently and the arm runs as
    # though the knob were never set.
    extra = _agg_flags(resolve_agg_flags(agg["flags"], cfg, FULL_ROUNDS))
    if extra:
        line += f" {extra}"
    penalty = _reg_set(reg, cfg)
    return line + (f" {penalty}" if penalty else "")


def extreme_line(cfg, case: str, clients_file: str, agg: Dict[str, Any],
                 reg: Dict[str, Any], fold: int, seed: int) -> str:
    """
    The winning combination on a one- or two-client federation.

    Full participation: dropping a client from a two-client federation is not a
    participation study, it is a coin flip on whether the round happens.
    """
    line = combo_line(cfg, agg, reg, fold, seed)
    line = line.replace(
        f" --parent {cfg.tag}_combo_{agg['id']}_{reg['id']}_fold{fold}",
        f" --parent {cfg.tag}_extreme_{case}_fold{fold}",
    )
    line = line.replace(
        f" --outliers-file {cohort_file(cfg)}", f" --outliers-file {clients_file}"
    )
    line = line.replace(
        f" --policy uniform --clients-per-round {cfg.clients_per_round}",
        " --policy all --participation 1.0",
    )
    return line


#: The smaller federation the winners are re-run on, and how many of it a round
#: draws: ``floor(0.9 * 5) = 4``, the study's own rule on a smaller population.
#: See :func:`~.study_config.participants`.
FIVE_CLIENTS = five_cells.COHORT_SIZE
FIVE_PER_ROUND = five_cells.clients_per_round(FIVE_CLIENTS)


def cohort_line(cfg, config: Dict[str, Any], clients_file: Optional[str],
                agg: Optional[Dict[str, Any]], reg: Optional[Dict[str, Any]],
                fold: int, seed: int, parent_tag: str, per_round: int,
                book: Optional[str] = None) -> str:
    """
    One selected configuration on a stated federation, at the reporting horizon.

    Everything a size or dropout stage varies is an argument here, and
    everything it does not vary comes from :func:`combo_line` and :func:`_base`
    unchanged - so a stage cannot silently move the anchor, the teacher, the
    horizon or the evaluation while claiming to move only the participation.

    A **smaller** cohort is narrowed by the clients file alone and keeps the
    ten-client book: the runner keeps only the writers the list names, and
    re-cutting a book for them would give those writers different rows than the
    ten-client runs gave them, turning a federation comparison into a split
    comparison.  A **larger** cohort has no such choice - the ten-client book
    holds no rows for writers it never covered - so it brings its own book, cut
    by the same 60/20/20 rule at the same seed, and ``book`` names it.

    Args:
        config: A cell of :data:`five_cells.CONFIGS`.
        clients_file: A narrowed client list, or ``None`` to federate the whole
            cohort - which is what a dropout stage does.
        agg: The aggregation cell, or ``None`` for the plain-FedAvg control.
        reg: The penalty cell, or ``None`` for the control, which carries none.
        parent_tag: The stage's own folder tag, so two stages running the same
            configuration cannot name one folder.
        per_round: Clients drawn per round, from
            :func:`~.study_config.participants`.
        book: A fold book to read instead of the study's own - required when the
            cohort holds writers the ten-client book never covered, and never
            used to re-cut writers it does cover.
    """
    parent = f"{cfg.tag}_{parent_tag}_{config['id']}_fold{fold}"
    if reg is None:
        # The control: no penalty, plain averaging, and therefore both
        # schedules in the one task, exactly as every fedavg line of this
        # study runs.
        line = _base(cfg, parent, fold, FULL_ROUNDS, seed, "BaseTrainer")
        line += " --aggregation fedavg --outer-workers 1 --inner-workers 1"
    else:
        # RESOLVE AGAINST THE RATE THIS STAGE RUNS AT. Two of the server knobs
        # are counted in clients, and this stage draws `per_round` of them, not
        # the study's own number. Resolving against the study default would
        # compute a trim fraction for a federation that is not the one running -
        # and the line would carry a coefficient that means something else.
        line = combo_line(
            dataclasses.replace(cfg, clients_per_round=per_round),
            agg, reg, fold, seed,
        )
        line = line.replace(
            f" --parent {cfg.tag}_combo_{agg['id']}_{reg['id']}_fold{fold}",
            f" --parent {parent}",
        )
    if clients_file is not None:
        line = line.replace(
            f" --outliers-file {cohort_file(cfg)}",
            f" --outliers-file {clients_file}",
        )
    if book is not None:
        line = line.replace(
            f" --fold-book {cohort_book(cfg)} --fold {fold}",
            f" --fold-book {book} --fold {fold}",
        )
    line = line.replace(
        f" --policy uniform --clients-per-round {cfg.clients_per_round}",
        f" --policy uniform --clients-per-round {per_round}",
    )
    return line


def five_line(cfg, config: Dict[str, Any], clients_file: str,
              agg: Optional[Dict[str, Any]], reg: Optional[Dict[str, Any]],
              fold: int, seed: int) -> str:
    """The five-client line: :func:`cohort_line` at this stage's tag and rate."""
    return cohort_line(cfg, config, clients_file, agg, reg, fold, seed,
                       "five", FIVE_PER_ROUND)


def counts(cfg) -> Dict[str, int]:
    """
    The screen's task counts, computed from the table it runs.

    These are the numbers an ``--array`` range is written from, so a generator
    that emits a different number must fail loudly rather than leave an array
    pointed at lines that are not there.
    """
    folds = len(FOLDS)
    return {
        "agg_screen": len(agg_cells.screen_cells()) * folds,
        "agg_full": len(AGG_METHODS) * folds,
        "reg_screen": len(reg_cells.screen_cells()) * folds,
        "reg_full": len(REG_METHODS) * len(FAMILIES) * folds,
        "combos": COMBO_TOP_K * COMBO_TOP_K * len(FAMILIES) * folds,
        "extreme": 3 * folds,
        "five": len(five_cells.CONFIGS) * folds,
        "drop20": len(five_cells.CONFIGS) * folds,
        "c20": (len(five_cells.CONFIGS) - 1) * folds * 2,
    }
