"""
Command-line interface.

Every phase of the pipeline is reachable through one entry point::

    foa <command> [options]
    python -m federated_outlier_adaptation.cli <command> [options]

``fal``, the entry point of the earlier repository, is installed as an alias of
``foa`` for one release, so existing scripts and task files keep working.

Commands (in pipeline order)::

    prepare-data     build the packed dataset cache from by_write.zip
                     (--resolution 28 --classes all builds the 62-class,
                      EMNIST-converted cache of the by-writer FEMNIST task)
    global-train     train the global model, Fisher information and outliers
    global-train-fl  obtain the global model by FedAvg over the server clients
    select-outliers  (re-)select the low-accuracy clients, or enrol a fixed cohort
    combined-train   train the combined global + outliers reference model
    base-fl          baseline federated runs
    all-aggs         baseline trainer against every aggregation method
    grid             hyperparameter sweeps
    final            final runs for one trainer
    extreme          minimal-client (extreme case) runs
    local-finetune   no-federation reference (each client on its own)
    matrix           write the task list of a multi-seed experiment plan
    select           constrained, validation-based configuration selection
    signals          analyse the constraint-respecting forgetting signals
    isolated-train   one private model per cohort client, scored three ways
    select-fold      pick the best fold of a CV training and persist it
    score-writers    score every writer on its held-out partition of a fold
    draw-old-data    draw the old-data population, seeded and documented
    evaluate-book    score a saved model on one part of one fold
    fold-book        materialise a cross-validation split and write it down
    outlier-figure   the selected clients' glyphs against the source means
    report           tables and figures of the whole evaluation protocol
    figures          regenerate grid-search rankings and heatmaps

Global options ``--results-dir`` and ``--data-dir`` override the configured
locations; they are applied to the environment before the package resolves any
path, so they affect every command consistently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

GRID_METHODS = {
    "ewc": ("federated_outlier_adaptation.grid_search.ewc", "ewc_grid_search"),
    "prox": ("federated_outlier_adaptation.grid_search.prox", "prox_grid_search"),
    "kd": ("federated_outlier_adaptation.grid_search.distillation", "distillation_grid_search"),
    "logit": (
        "federated_outlier_adaptation.grid_search.logit_consistency",
        "logit_consistency_grid_search",
    ),
    "aligned": (
        "federated_outlier_adaptation.grid_search.aligned_feature",
        "feature_aligned_grid_search",
    ),
    "feature": (
        "federated_outlier_adaptation.grid_search.feature_alignment",
        "feature_alignment_grid_search",
    ),
    "freeze": (
        "federated_outlier_adaptation.grid_search.freeze",
        "freeze_grid_search",
    ),
    "ntd": (
        "federated_outlier_adaptation.grid_search.ntd",
        "ntd_grid_search",
    ),
    "anchored": (
        "federated_outlier_adaptation.grid_search.anchored",
        "anchored_grid_search",
    ),
    "kd_ewc": (
        "federated_outlier_adaptation.grid_search.distillation_ewc",
        "distil_ewc_grid_search",
    ),
    "extreme": (
        "federated_outlier_adaptation.grid_search.extreme_distillation",
        "extreme_cases_grid_search",
    ),
}

EXTREME_CASES = ("single", "double", "dual")

#: Per-round client selection policies, mirrored from ``runners.client_sampler``
#: so that ``--help`` works without importing torch.
SELECTION_POLICIES = ("all", "uniform", "worst_first", "round_robin")

#: Selection modes of ``select-outliers``.
SELECTION_MODES = ("sample", "lowest", "pool", "worst")

#: Severity tags of a client pool, mirrored from ``outliers.client_accuracy``.
SEVERITIES = ("mild", "moderate", "severe")

#: Dataset providers, mirrored from ``providers.registry`` so that ``--help``
#: works without importing torch.
PROVIDERS = ("nist",)

#: Datasets ``prepare-data`` can build.  ``mnist`` is not a provider: it is the
#: server-side proxy set of the NIST provider and has no clients of its own.
PREPARABLE_DATASETS = PROVIDERS + ("mnist",)

#: Aggregation variant keys, mirrored from ``aggregation.selector``.
AGG_VARIANT_KEYS = ("all", "fedavg", "anchored", "capped", "cyclic")

#: Headline setting of ``report``, mirrored from ``analysis.report.MAIN_SETTING``
#: so that ``--help`` works without importing the analysis layer; the suites
#: check that the two agree.
MAIN_SETTING = "pool0.05 m10 uniform"

#: Folds of the cross-validation protocol, mirrored from ``config.FOLD_COUNT``.
FOLD_COUNT = 5

#: Source-sharing modes, mirrored from ``data.source_share``.
SOURCE_SHARE_MODES = ("off", "equal", "full")

#: Client weighting schemes, mirrored from ``aggregation.concurrent_methods``.
WEIGHTINGS = ("proportional", "uniform", "capped")

#: Cyclic visiting orders, mirrored from ``runners.sequential_runner``.
CLIENT_ORDERS = ("fixed", "shuffle")

#: Starting points of a federated run, mirrored from the runners.
INITIALISATIONS = ("global", "scratch")

#: Sources of the per-round evaluations, mirrored from ``utils.eval_cache``.
EVAL_PATHS = ("cache", "loader")

#: Resolutions of the NIST provider, mirrored from ``config``.
RESOLUTIONS = (28, 128)

#: Character sets of the NIST provider, mirrored from ``data.sd19_labels``.
CLASS_SETS = ("all", "digits")

#: Model topologies, mirrored from ``config.MODEL_NAMES``.
MODEL_NAMES = ("fedavg_cnn", "flexible_cnn")

#: Conventions of the ``param_l2`` penalty, mirrored from the anchored trainer.
PARAM_L2_CONVENTIONS = ("mean_per_tensor", "fedprox")

#: Distance spaces of the anchored trainer, mirrored to keep --help light.
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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_overrides(pairs: Optional[Sequence[str]], blob: Optional[str]) -> dict[str, Any]:
    """
    Build the trainer keyword overrides from ``--set K=V`` and ``--trainer-kwargs``.

    Values are parsed as JSON when possible (so ``T=4`` becomes an int and
    ``alpha=0.95`` a float) and fall back to plain strings.
    """
    overrides: dict[str, Any] = {}
    if blob:
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            raise ValueError("--trainer-kwargs must be a JSON object")
        overrides.update(parsed)
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects NAME=VALUE, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        overrides[key.strip()] = value
    return overrides


def _trainer_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """
    The trainer keyword overrides of one command line.

    ``--set``/``--trainer-kwargs`` verbatim, plus the ``param_l2`` convention
    when ``--fedprox-convention`` was given; an explicit
    ``--set param_l2_convention=...`` wins, so the flag is a shorthand and never
    overrides a value that was spelled out.
    """
    overrides = _parse_overrides(
        getattr(args, "overrides", None), getattr(args, "trainer_kwargs", None)
    )
    if getattr(args, "fedprox_convention", False):
        overrides.setdefault("param_l2_convention", "fedprox")
    return overrides


def _apply_global_paths(args: argparse.Namespace) -> None:
    """
    Push the global overrides into the environment.

    The directories, the NIST resolution, the character set and the model are
    resolved by :mod:`federated_outlier_adaptation.config` at import time of the
    provider, so they are exported here, before any command imports it.  A flag
    that was not given leaves the environment - and therefore the configured
    default - untouched.
    """
    if getattr(args, "results_dir", None):
        os.environ["FOA_RESULTS_DIR"] = str(args.results_dir)
    if getattr(args, "data_dir", None):
        os.environ["FOA_DATA_DIR"] = str(args.data_dir)
    if getattr(args, "cache_dir", None):
        os.environ["FOA_CACHE_DIR"] = str(args.cache_dir)
    if getattr(args, "resolution", None):
        os.environ["FOA_NIST_RESOLUTION"] = str(args.resolution)
    if getattr(args, "classes", None):
        os.environ["FOA_NIST_CLASSES"] = str(args.classes)
    if getattr(args, "model", None):
        os.environ["FOA_MODEL"] = str(args.model)
    if getattr(args, "require_trainable", False):
        os.environ["FOA_REQUIRE_TRAINABLE"] = "1"
    if getattr(args, "fold", None):
        os.environ["FOA_FOLD"] = str(args.fold)
    if getattr(args, "fold_book", None):
        os.environ["FOA_FOLD_BOOK"] = str(args.fold_book)


def _resolve_provider(args: argparse.Namespace):
    """Instantiate the dataset provider a command should run against."""
    from federated_outlier_adaptation.providers import get_provider

    return get_provider(getattr(args, "provider", "nist") or "nist")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        default="nist",
        choices=sorted(PROVIDERS),
        help="Dataset provider (default: nist, the published pipeline).",
    )
    parser.add_argument("--results-dir", default=None, help="Override the results root.")
    parser.add_argument("--data-dir", default=None, help="Override the data root.")
    parser.add_argument("--cache-dir", default=None, help="Override the dataset cache root.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        choices=RESOLUTIONS,
        help=(
            "NIST input resolution: 28 (the EMNIST conversion, 62 classes) or "
            "128 (the published digit pipeline). Default: the configured one."
        ),
    )
    parser.add_argument(
        "--classes",
        default=None,
        choices=sorted(CLASS_SETS),
        help=(
            "NIST character set: 'all' (62 classes) or 'digits' (10, the "
            "ablation). Default: the configured one."
        ),
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=range(1, FOLD_COUNT + 1),
        metavar="K",
        help=(
            f"Cross-validation fold, 1..{FOLD_COUNT}. Re-splits every client's "
            "own 60/20/20 deterministically and re-seeds the run, so folds are "
            "different draws over the same clients and the same shipped global "
            "model. Absent (the default) is the single published split, "
            "unchanged. The fold is part of the output path, so folds never "
            "overwrite each other."
        ),
    )
    parser.add_argument(
        "--fold-book",
        default=None,
        metavar="PATH",
        help=(
            "Read every client's split from this fold book instead of deriving "
            "it from a seed. With --fold K the rows the book recorded for fold "
            "K are the split, so one fold means the same thing in every process "
            "that reads it."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(MODEL_NAMES),
        help=(
            "Model topology: 'fedavg_cnn' (the reference CNN of McMahan et al.) "
            "or 'flexible_cnn' (the architecture ablation)."
        ),
    )


def _add_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="NAME=VALUE",
        help='Trainer hyperparameter overrides, e.g. --set T=4 alpha=0.95.',
    )
    parser.add_argument(
        "--trainer-kwargs",
        default=None,
        help='Trainer overrides as a JSON object, e.g. \'{"T": 4, "alpha": 0.95}\'.',
    )
    parser.add_argument(
        "--fedprox-convention",
        action="store_true",
        help=(
            "Measure the anchored 'param_l2' penalty as FedProx does, "
            "(mu/2)*||theta-theta_anchor||^2 summed over all parameters, so "
            "that lam is FedProx's mu. Off by default, which keeps the "
            "published per-tensor mean and its own lam scale."
        ),
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=None, help="Optional run seed (default: unseeded).")
    parser.add_argument(
        "--outliers-file", default=None, help="Alternative selected-client JSON list."
    )
    parser.add_argument("--outer-workers", type=int, default=None)
    parser.add_argument("--inner-workers", type=int, default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip jobs whose result file already exists (job-array resume).",
    )


def _add_budget_options(parser: argparse.ArgumentParser) -> None:
    """
    Round / epoch / batch-size overrides of one run.

    All three default to ``None``, which leaves the published budget in place
    (finals and extreme cases: 100 rounds, 100 local epochs, batch 64; sweeps:
    50 rounds, 50 local epochs, batch 512).  They exist for cheap end-to-end
    checks and for budget studies, and are recorded in the provenance ``config``
    block like every other run parameter.
    """
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        metavar="R",
        help="Federated rounds of the run (default: the published value).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="E",
        help="Local epochs per round (default: the published value).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="B",
        help="Client batch size (default: the published value).",
    )


#: ``--flag`` -> ``ServerState`` keyword of the swept server coefficients.
_SERVER_FLAGS = {
    "server_beta": "momentum_beta",
    "server_lr": "server_lr",
    "server_tau": "tau",
    "trim_frac": "trim_fraction",
    "server_anchor": "anchor_lambda",
}


def _server_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """
    Only the server coefficients that were actually given on the command line.

    An unset flag is absent from the mapping, so the aggregation rule keeps the
    published constant and a run that names none of them is unchanged.
    """
    given: dict[str, Any] = {}
    for flag, keyword in _SERVER_FLAGS.items():
        value = getattr(args, flag, None)
        if value is not None:
            given[keyword] = float(value)
    return given


def _budget_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Only the budget overrides that were actually given on the command line."""
    given: dict[str, Any] = {}
    for name in ("rounds", "epochs", "batch_size"):
        value = getattr(args, name, None)
        if value is not None:
            given[name] = value
    return given


def _add_eval_options(parser: argparse.ArgumentParser) -> None:
    """
    Where the per-round evaluations read their data from, and how often.

    ``--eval-path cache`` (the default) materialises every evaluation set once
    and scores it in one batched pass per round; ``loader`` keeps the published
    per-round ``DataLoader`` passes.  The two produce the same numbers - that is
    what the equality check verifies - and differ only in wall-clock time.
    """
    parser.add_argument(
        "--eval-path",
        default=None,
        choices=sorted(EVAL_PATHS),
        help=(
            "Source of the per-round evaluations: 'cache' (default) scores "
            "materialised tensors in one pass per set, 'loader' walks the "
            "published DataLoaders."
        ),
    )
    parser.add_argument(
        "--insample-every",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Measure the pool's in-sample metrics every N rounds and in the "
            "last round (default: 1, every round). The series keep their "
            "length; skipped rounds repeat the last measurement."
        ),
    )


def _add_population_options(parser: argparse.ArgumentParser) -> None:
    """Per-round client selection and tracking; the defaults change nothing."""
    parser.add_argument(
        "--participation",
        type=float,
        default=1.0,
        help="Fraction C of the client pool training per round (default: 1.0).",
    )
    parser.add_argument(
        "--pool-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Draw the participants from the provider's rule-based outlier pool "
            "(bottom fraction F). Overridden by an explicit --outliers-file."
        ),
    )
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=None,
        metavar="M",
        help="Absolute number of clients per round; takes precedence over --participation.",
    )
    parser.add_argument(
        "--policy",
        default="all",
        choices=sorted(SELECTION_POLICIES),
        help="Per-round client selection policy (default: all).",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed of the client sampler (defaults to the run seed).",
    )
    parser.add_argument(
        "--track-clients",
        action="store_true",
        help="Evaluate every pool client on its own data each round.",
    )


def _add_aggregation_options(parser: argparse.ArgumentParser) -> None:
    """Aggregation rule, client weighting, server step and cyclic order."""
    parser.add_argument(
        "--aggregation",
        default=None,
        help=(
            "Aggregation rule: a variant key (%s) or a concrete rule name. "
            "The default runs every rule of each family, as published."
        )
        % ", ".join(sorted(AGG_VARIANT_KEYS)),
    )
    parser.add_argument(
        "--weighting",
        default="proportional",
        choices=sorted(WEIGHTINGS),
        help="Client weights p_k of the extended parallel rules (default: proportional).",
    )
    parser.add_argument(
        "--server-eta",
        type=float,
        default=1.0,
        help="Server step size of the robust and anchored parallel rules.",
    )
    parser.add_argument(
        "--client-order",
        default="fixed",
        choices=sorted(CLIENT_ORDERS),
        help="Order of the participants in a cyclic round (default: fixed).",
    )
    parser.add_argument(
        "--seq-mix-alpha",
        type=float,
        default=None,
        metavar="A",
        help=(
            "Mixing weight of the cyclic rule 'seq_mix_alpha': "
            "theta <- (1-A) theta + A theta_k. Replaces the hard-coded "
            "0.7/0.3, 1/(i+1) and i/(K+i) coefficients with a swept scalar."
        ),
    )
    parser.add_argument(
        "--server-beta",
        type=float,
        default=None,
        metavar="B",
        help=(
            "FedAvgM server momentum (published grid 0.7, 0.9, 0.97, 0.99, "
            "0.997). Default: 0.9, the value the published runs used."
        ),
    )
    parser.add_argument(
        "--server-lr",
        type=float,
        default=None,
        metavar="ETA",
        help="FedAdam/FedYogi server learning rate (default: 1e-2).",
    )
    parser.add_argument(
        "--server-tau",
        type=float,
        default=None,
        metavar="TAU",
        help="FedAdam/FedYogi adaptivity constant (default: 1e-3).",
    )
    parser.add_argument(
        "--server-anchor",
        type=float,
        default=None,
        metavar="LAM",
        help=(
            "Pull of the parameterised server anchor 'con_delta_anchor_lam' "
            "towards the model the run started from: theta <- theta + eta_s "
            "Delta - LAM (theta - theta_g). Default: 0.1, the value the fixed "
            "rule con_delta_anchor_lam01 uses. LAM=0 is the plain step."
        ),
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Fraction the trimmed mean drops at each end per coordinate "
            "(default: 0.2; Yin et al. experiment at 0.1)."
        ),
    )
    parser.add_argument(
        "--init",
        default="global",
        choices=sorted(INITIALISATIONS),
        help=(
            "Start the federation from the global model (default) or from a "
            "freshly initialised one (scratch), as a reference point."
        ),
    )


def _add_extended_aggregations(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--extended-aggregations",
        action="store_true",
        help=(
            "Also sweep the opt-in server-optimiser rules (FedAvgM, FedAdam, "
            "FedYogi). Off by default, so the published method set is reproduced."
        ),
    )


def _add_global_name(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--global-name",
        default="global",
        help=(
            "Global model the run starts from: 'global' (default, centrally "
            "trained) or e.g. 'global_fl' for the federated pre-trained one. "
            "The teacher checkpoint and Fisher directory follow the name."
        ),
    )


def _add_stop_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stop-when-global-below-clients",
        action="store_true",
        help=(
            "Stop a run as soon as the global accuracy falls below the clients "
            "accuracy or below 0.90."
        ),
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_prepare_data(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation import config

    dataset = args.dataset or args.provider or "nist"

    if dataset != "mnist" and not args.zip:
        raise SystemExit(f"--zip is required to prepare {dataset}")

    if dataset == "mnist":
        from federated_outlier_adaptation.data.mnist import prepare

        summary = prepare(raw_dir=args.raw_dir or args.zip, out_dir=args.out)
        print(json.dumps(summary, indent=2))
        return 0

    resolution = args.resolution or config.nist_resolution()
    if resolution == 28:
        # The by-writer FEMNIST task: stream by_write.zip, apply the EMNIST
        # conversion and write the four packed files.  The labels come from the
        # by_class checksum log, so the second archive is never needed.
        from federated_outlier_adaptation.data.nist28 import build_cache as build_cache28

        summary = build_cache28(
            zip_path=args.zip,
            out_dir=args.out or config.NIST28_DIR,
            by_class_log=args.by_class_log,
            by_write_log=args.by_write_log,
            classes=args.classes or config.nist_classes(),
            size=28,
        )
        print(json.dumps(summary, indent=2))
        return 0

    from federated_outlier_adaptation.data.nist_cache import build_cache

    labels = args.labels or config.DIGITS_LABELS_JSON
    out_dir = args.out or config.CACHE_DIR
    summary = build_cache(
        zip_path=args.zip,
        labels_json=labels,
        out_dir=out_dir,
        size=args.size,
        probe=args.probe,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _maybe_write_pool(args: argparse.Namespace, provider) -> None:
    """Write the rule-based outlier pool when ``--pool-frac`` was given."""
    if getattr(args, "pool_frac", None) is None:
        return
    from federated_outlier_adaptation.outliers.selection import select_outlier_pool

    payload, path = select_outlier_pool(
        pool_frac=args.pool_frac, force=True, provider=provider
    )
    print(f"Outlier pool of {payload['size']} clients -> {path}")


def cmd_global_train(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.utils.seeding import set_run_seed

    # The run seed is the model initialisation and the training shuffle; the
    # split seed is a different thing and stays where it was, so a repeated
    # global training draws the same writers and a different initialisation.
    set_run_seed(args.seed)
    provider = _resolve_provider(args)

    # A named population - "train on exactly these clients" - is the study
    # pipeline's question, and `run_global_training` answers it for any
    # provider: it goes through `provider.dataset.build_dataset(writers, ...)`
    # and touches nothing dataset-specific. The provider-split entry point below
    # answers a different question (draw a global/local partition, then score
    # the held-out clients), and it has no way to be told which clients to use.
    # So the population decides the route, not the provider.
    if args.provider and args.provider != "nist" and args.population in ("source", None):
        from federated_outlier_adaptation.training.global_model import (
            run_provider_global_training,
        )

        run_provider_global_training(
            provider,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.split_seed,
            patience=args.patience,
            k_values=tuple(args.k_values) if args.k_values else (args.k,),
        )
        _maybe_write_pool(args, provider)
        return 0

    from federated_outlier_adaptation.training.global_model import run_global_training

    run_global_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.split_seed,
        k=args.k,
        skip_outlier_training=args.skip_outlier_training,
        patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
        population=args.population,
        writers_file=args.writers_file,
        provider=provider,
    )
    _maybe_write_pool(args, provider)
    return 0


def cmd_global_train_fl(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.federated_pretraining import (
        run_federated_pretraining,
    )

    metrics = run_federated_pretraining(
        rounds=args.rounds,
        participation=args.participation,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        global_name=args.global_name,
        force=args.force,
        provider=_resolve_provider(args),
    )
    print(
        json.dumps(
            {
                "test_accuracy": metrics.get("test_accuracy"),
                "rounds": metrics.get("rounds"),
                "model_path": metrics.get("model_path"),
            },
            indent=2,
        )
    )
    return 0


def cmd_select_outliers(args: argparse.Namespace) -> int:
    if args.mode == "pool":
        from federated_outlier_adaptation.outliers.selection import select_outlier_pool

        threshold = args.pool_max_acc if args.pool_max_acc is not None else args.pool_threshold
        payload, path = select_outlier_pool(
            pool_frac=args.pool_frac,
            pool_threshold=threshold,
            force=args.force,
            provider=_resolve_provider(args),
            severity=args.severity,
            write_table=not args.no_accuracy_table,
        )
        print(f"Wrote {path}")
        print(
            json.dumps(
                {
                    "rule": payload.get("rule"),
                    "severity": payload.get("severity"),
                    "threshold": payload.get("threshold"),
                    "frac": payload.get("frac"),
                    "size": payload.get("size"),
                    "eligible_clients": payload.get("eligible_clients"),
                    "min_accuracy": payload.get("min_accuracy"),
                    "max_accuracy": payload.get("max_accuracy"),
                    "mean_accuracy": payload.get("mean_accuracy"),
                    "population": payload.get("population"),
                    "clients": payload.get("clients"),
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "worst":
        from federated_outlier_adaptation.outliers.selection import select_cohort

        payload, path = select_cohort(
            k=args.k,
            force=args.force,
            provider=_resolve_provider(args),
            tag=args.tag,
            write_table=not args.no_accuracy_table,
            scores=args.scores,
            out=args.out,
        )
        print(f"Wrote {path}")
        print(
            json.dumps(
                {
                    "rule": payload.get("rule"),
                    "tag": payload.get("tag"),
                    "k": payload.get("k"),
                    "size": payload.get("size"),
                    "scored_clients": payload.get("scored_clients"),
                    "eligible_clients": payload.get("eligible_clients"),
                    "ineligible_clients": payload.get("ineligible_clients"),
                    "excluded_from_cohort": payload.get("excluded_from_cohort"),
                    "min_accuracy": payload.get("min_accuracy"),
                    "max_accuracy": payload.get("max_accuracy"),
                    "mean_accuracy": payload.get("mean_accuracy"),
                    "clients": payload.get("clients"),
                    "accuracies": payload.get("accuracies"),
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "lowest":
        from federated_outlier_adaptation.outliers.selection import select_lowest_pool

        writers, path = select_lowest_pool(
            k=args.k, force=args.force, provider=_resolve_provider(args)
        )
        print(f"Wrote {path}")
        print(json.dumps(writers, indent=2))
        return 0

    from federated_outlier_adaptation.training.global_model import GlobalTraining

    gt = GlobalTraining(provider=_resolve_provider(args))
    writers = gt.generate_selected_writers(
        seed=args.seed if args.seed is not None else 42,
        k=args.k,
        bottom_frac=args.bottom_frac,
        force_generate=args.force,
    )
    print(json.dumps(writers, indent=2))
    return 0


def cmd_combined_train(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.combined_model import run_combined_training

    run_combined_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        provider=_resolve_provider(args),
        k=args.k,
        clients_file=args.outliers_file,
        name=args.name,
        patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
    )
    return 0


def cmd_base_fl(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.base_fl import base_fl_training

    base_fl_training(
        outer_max_workers=args.outer_workers or 4,
        inner_max_workers=args.inner_workers or 1,
        seed=args.seed,
        trainer_kwargs=_trainer_overrides(args) or None,
        outliers_file=args.outliers_file,
        parent_name=args.parent,
        stop_when_global_below_clients=args.stop_when_global_below_clients,
        skip_existing=args.skip_existing,
        global_name=args.global_name,
        provider_name=args.provider,
    )
    return 0


def cmd_all_aggs(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.all_aggregations import all_aggregation_variants

    all_aggregation_variants(
        outer_max_workers=args.outer_workers or 1,
        inner_max_workers=args.inner_workers or 20,
        seed=args.seed,
        trainer_kwargs=_trainer_overrides(args) or None,
        outliers_file=args.outliers_file,
        skip_existing=args.skip_existing,
        extended_aggregations=args.extended_aggregations,
        global_name=args.global_name,
        parent_name=args.parent,
        provider_name=args.provider,
    )
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    import importlib

    module_name, func_name = GRID_METHODS[args.method]
    func = getattr(importlib.import_module(module_name), func_name)

    kwargs: dict[str, Any] = dict(
        outer_max_workers=args.outer_workers,
        inner_max_workers=args.inner_workers,
        seed=args.seed,
        trainer_kwargs=_trainer_overrides(args) or None,
        outliers_file=args.outliers_file,
        skip_existing=args.skip_existing,
        global_name=args.global_name,
        population=dict(
            participation=args.participation,
            clients_per_round=args.clients_per_round,
            policy=args.policy,
            sampler_seed=args.sampler_seed,
            track_clients=args.track_clients,
            pool_frac=args.pool_frac,
        ),
        provider_name=args.provider,
        folder=args.parent,
        eval_path=args.eval_path,
        insample_every=args.insample_every,
    )
    budget = _budget_kwargs(args)
    if "rounds" in budget:
        kwargs["max_round"] = budget["rounds"]
    if "epochs" in budget:
        kwargs["epochs"] = budget["epochs"]
    if "batch_size" in budget:
        kwargs["batch_size"] = budget["batch_size"]
    if args.method in ("kd", "extreme") and getattr(args, "capped_tasks", False):
        kwargs["capped_tasks"] = True
    if args.method == "extreme":
        kwargs["extreme_case"] = args.case
    if args.method == "anchored":
        kwargs["space"] = args.space
        kwargs["anchor"] = args.anchor
        if args.index is not None:
            kwargs["index"] = args.index
        if args.lams:
            kwargs["lams"] = list(args.lams)
        if args.temperatures:
            kwargs["temperatures"] = list(args.temperatures)
    func(**kwargs)
    return 0


def cmd_final(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.final_experiments import (
        generate_final_result_all_parallel,
    )

    generate_final_result_all_parallel(
        trainer_name=args.trainer,
        outer_max_workers=args.outer_workers or 2,
        inner_max_workers=args.inner_workers or 2,
        parent_name=args.parent,
        seed=args.seed,
        trainer_kwargs=_trainer_overrides(args) or None,
        outliers_file=args.outliers_file,
        participation=args.participation,
        policy=args.policy,
        sampler_seed=args.sampler_seed,
        track_clients=args.track_clients,
        stop_when_global_below_clients=args.stop_when_global_below_clients,
        skip_existing=args.skip_existing,
        extended_aggregations=args.extended_aggregations,
        global_name=args.global_name,
        clients_per_round=args.clients_per_round,
        weighting=args.weighting,
        server_eta=args.server_eta,
        server_kwargs=_server_kwargs(args) or None,
        client_order=args.client_order,
        seq_mix_alpha=args.seq_mix_alpha,
        source_share=args.source_share,
        source_share_cap=args.source_share_cap,
        save_final_model=args.save_final_model,
        aggregation=args.aggregation,
        init=args.init,
        provider_name=args.provider,
        pool_frac=args.pool_frac,
        eval_path=args.eval_path,
        insample_every=args.insample_every,
        old_book=args.old_book,
        old_clients_file=args.old_clients_file,
        old_folds=args.old_fold,
        final_eval_batch_size=args.eval_batch_size,
        source_share_multiplier=args.source_share_multiplier,
        val_blend_source=args.val_blend_source,
        **_budget_kwargs(args),
    )
    return 0


def cmd_extreme(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.extreme_cases import extreme_cases_final_training

    extreme_cases_final_training(
        extreme_case=args.case,
        single_outlier=list(args.clients) if args.clients else None,
        trainer_name=args.trainer,
        parent_name=args.parent,
        outer_max_workers=args.outer_workers or 4,
        inner_max_workers=args.inner_workers or 4,
        seed=args.seed,
        trainer_kwargs=_trainer_overrides(args) or None,
        outliers_file=args.outliers_file,
        participation=args.participation,
        policy=args.policy,
        sampler_seed=args.sampler_seed,
        track_clients=args.track_clients,
        stop_when_global_below_clients=args.stop_when_global_below_clients,
        skip_existing=args.skip_existing,
        global_name=args.global_name,
        clients_per_round=args.clients_per_round,
        weighting=args.weighting,
        server_eta=args.server_eta,
        server_kwargs=_server_kwargs(args) or None,
        client_order=args.client_order,
        seq_mix_alpha=args.seq_mix_alpha,
        aggregation=args.aggregation,
        init=args.init,
        provider_name=args.provider,
        pool_frac=args.pool_frac,
        eval_path=args.eval_path,
        insample_every=args.insample_every,
        **_budget_kwargs(args),
    )
    return 0


def _matrix_estimate(args, tasks: Sequence[str]) -> str:
    """
    Cost line printed under a task list.

    With ``--seconds-per-round`` the estimate is built from the measured unit
    cost and the runs each line expands into; without it the flat
    ``--minutes-per-task`` assumption is used, which is what the planner printed
    before the unit costs were measured.
    """
    from federated_outlier_adaptation.training.matrix import estimate_gpu_hours
    from federated_outlier_adaptation.training.tasks import plan_gpu_hours

    if args.seconds_per_round is None:
        hours = estimate_gpu_hours(len(tasks), minutes_per_task=args.minutes_per_task)
        return (
            f"tasks={len(tasks)} estimated_gpu_hours={hours:.1f} "
            f"(assumption: {args.minutes_per_task:g} min per task)"
        )
    estimate = plan_gpu_hours(
        tasks,
        seconds_per_round=args.seconds_per_round,
        sweep_seconds_per_round=args.sweep_seconds_per_round,
    )
    return (
        f"tasks={estimate['tasks']} runs={estimate['runs']} "
        f"estimated_gpu_hours={estimate['gpu_hours']:.1f} "
        f"(measured: {estimate['seconds_per_round']:g} s/round finals, "
        f"{estimate['sweep_seconds_per_round']:g} s/round sweeps; "
        f"{estimate['unpriced_tasks']} task(s) not priced)"
    )


def cmd_matrix_resume(args: argparse.Namespace) -> int:
    """Emit only the lines of a task file whose expected outputs are missing."""
    from pathlib import Path

    from federated_outlier_adaptation.training.matrix import write_tasks
    from federated_outlier_adaptation.training.tasks import missing_tasks

    if not args.source:
        print("--plan resume needs --from TASKFILE")
        return 1
    with open(args.source) as handle:
        lines = handle.read().splitlines()
    tasks = missing_tasks(lines, results_dir=args.results_dir)
    if args.out:
        write_tasks(tasks, args.out)
        print(f"Wrote {len(tasks)} tasks to {args.out}")
    else:
        for task in tasks:
            print(task)
    print(
        f"plan=resume source={Path(args.source).name} "
        f"lines={len([line for line in lines if line.strip()])} "
        f"remaining={len(tasks)} " + _matrix_estimate(args, tasks)
    )
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.matrix import (
        PLANS,
        plan_summary,
        plan_tasks,
        write_tasks,
    )

    if args.plan == "resume":
        return cmd_matrix_resume(args)

    if args.list or not args.plan:
        for name, count, hours in plan_summary(minutes_per_task=args.minutes_per_task):
            print(f"{name:24s} tasks={count:4d} estimated_gpu_hours={hours:7.1f}")
        print(f"(assumption: {args.minutes_per_task:g} min per task)")
        print("plan 'resume' re-emits the unfinished lines of --from TASKFILE")
        return 0

    if args.plan not in PLANS:
        print(f"Unknown plan {args.plan!r}; available: {', '.join(sorted(PLANS))}, resume")
        return 1

    tasks = plan_tasks(args.plan, results_dir=args.results_dir, provider=args.provider)
    if args.out:
        write_tasks(tasks, args.out)
        print(f"Wrote {len(tasks)} tasks to {args.out}")
    else:
        for task in tasks:
            print(task)
    print(f"plan={args.plan} " + _matrix_estimate(args, tasks))
    return 0


def cmd_local_finetune(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation.training.local_finetune import run_local_finetuning

    payload = run_local_finetuning(
        trainer_name=args.trainer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        outliers_file=args.outliers_file,
        pool_frac=args.pool_frac,
        trainer_kwargs=_trainer_overrides(args) or None,
        global_name=args.global_name,
        parent_name=args.parent,
        skip_existing=args.skip_existing,
        init=args.init,
        init_checkpoint=args.init_checkpoint,
        patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
        provider=_resolve_provider(args),
    )
    print(
        json.dumps(
            {
                "init": payload.get("init"),
                "init_checkpoint": payload.get("init_checkpoint"),
                "init_checkpoint_sha256": payload.get("init_checkpoint_sha256"),
                "pool_insample_acc": payload.get("pool_insample_acc"),
                "pool_val_acc": payload.get("pool_val_acc"),
                "pool_test_acc": payload.get("pool_test_acc"),
                "mean_source_test_acc": payload.get("mean_source_test_acc"),
            },
            indent=2,
        )
    )
    return 0


def cmd_select_emit_finals(args: argparse.Namespace, root: str) -> int:
    """
    Select every stored regularisation sweep and write the finals it implies.

    One constrained selection per ``(space, anchor)`` sweep and per budget, each
    stored next to its sweep, and one ``foa final`` task line per selection and
    seed.  The lines carry the selected ``lam`` (and ``T``), which is the whole
    point: without them the finals would run the trainer defaults.
    """
    from federated_outlier_adaptation.grid_search.selection import (
        discover_sweeps,
        select_for_sweep,
    )
    from federated_outlier_adaptation.training.matrix import (
        emit_regularisation_finals,
        provider_results_root,
        write_tasks,
    )

    scan_root = provider_results_root(root, args.provider)
    sweeps = discover_sweeps(scan_root)
    selections = []
    for folder, space, anchor in sweeps:
        for eps in args.eps:
            selections.append(
                select_for_sweep(
                    folder,
                    eps=eps,
                    on=args.on,
                    window=args.window,
                    space=space,
                    anchor=anchor,
                )
            )

    tasks = emit_regularisation_finals(
        scan_root,
        eps_values=args.eps,
        seeds=args.seeds,
        provider=args.provider,
        aggregations=tuple(args.final_aggregation),
        extra_args=args.final_args or "",
        pool_args=args.final_pool_args,
        parent_prefix=args.final_parent_prefix or "",
    )
    if args.out:
        write_tasks(tasks, args.out)
    else:
        for task in tasks:
            print(task)

    print(
        json.dumps(
            {
                "root": str(scan_root),
                "sweeps": len(sweeps),
                "eps": list(args.eps),
                "seeds": list(args.seeds),
                "selected": sum(1 for s in selections if s.get("selected")),
                "aggregations": list(args.final_aggregation),
                "tasks": len(tasks),
                "out": args.out,
            },
            indent=2,
        )
    )
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.grid_search.selection import select_constrained

    root = args.root or str(config.RESULTS_DIR)
    if args.emit_finals:
        return cmd_select_emit_finals(args, root)

    result = select_constrained(
        root, eps=args.eps[0], on=args.on, window=args.window, out_dir=args.out
    )
    selected = result.get("selected")
    print(
        json.dumps(
            {
                "eps": result["eps"],
                "on": result["on"],
                "num_records": result["num_records"],
                "num_feasible": result["num_feasible"],
                "front_size": len(result["front"]),
                "selected": None
                if selected is None
                else {
                    "name": selected["name"],
                    "adaptation": selected["adaptation"],
                    "forgetting": selected["forgetting"],
                },
            },
            indent=2,
        )
    )
    return 0


def cmd_signals(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.analysis.forgetting_signals import (
        DEFAULT_DELTAS,
        analyse,
    )

    root = args.root or str(config.RESULTS_DIR)
    summary = analyse(
        root,
        out_dir=args.out,
        deltas=tuple(args.deltas) if args.deltas else DEFAULT_DELTAS,
        eps=args.eps,
        window=args.window,
        plots=not args.no_plots,
    )
    print(
        json.dumps(
            {
                "root": summary["root"],
                "out_dir": summary["out_dir"],
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


def cmd_report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.analysis.report import (
        DEFAULT_BUDGETS,
        build_report,
    )

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
                "num_final_arms": manifest["num_final_arms"],
                "num_reference_arms": manifest["num_reference_arms"],
                "setting": manifest["setting"],
                "tables": len(manifest["tables"]),
                "figures": len(manifest["figures"]),
                "comparisons": len(manifest["comparisons"]),
                "notes": manifest["notes"],
            },
            indent=2,
        )
    )
    return 0


def _require_book_and_fold(args: argparse.Namespace, command: str) -> None:
    """
    Both are global settings, so a command that cannot work without them says so.

    ``--fold-book`` and ``--fold`` live on the common option group because every
    run may carry them; these two commands are meaningless without them, and an
    explicit message beats an ``AttributeError`` on ``None``.
    """
    if not getattr(args, "fold_book", None):
        raise SystemExit(f"{command} needs --fold-book PATH.")
    if not getattr(args, "fold", None):
        raise SystemExit(f"{command} needs --fold K.")


def _load_book(path):
    """Open a fold book, with a message that names what writes one."""
    from federated_outlier_adaptation.data.fold_book import FoldBook

    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"No fold book at {path}. Build it with 'foa fold-book' first."
        )
    return FoldBook.load(path)


def _load_checkpoint(provider, path):
    """A model of the provider's topology, with ``path``'s weights in it."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"No checkpoint at {path}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = provider.make_model()
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def cmd_select_fold(args: argparse.Namespace) -> int:
    """Pick the best fold of a cross-validated training and persist it."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.analysis.fold_selection import select_best_fold

    root = args.root or str(config.RESULTS_DIR)
    payload = select_best_fold(
        root,
        prefix=args.prefix,
        name=args.name,
        folds=tuple(args.folds),
        results_dir=args.out or root,
        evaluations=args.from_evaluations,
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "folds"}, indent=2))
    print(json.dumps({"folds": payload["folds"]}, indent=2))
    return 0


def cmd_score_writers(args: argparse.Namespace) -> int:
    """Score every writer on its held-out partition of one fold."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import score_writers, write_json

    _require_book_and_fold(args, "score-writers")
    provider = _resolve_provider(args)
    book = _load_book(args.fold_book)
    model = _load_checkpoint(provider, args.model_path)

    payload = score_writers(
        provider, model, book, fold=args.fold, batch_size=args.batch_size
    )
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = Path(args.out) if args.out else results_dir / "outliers" / "writer_scores.json"
    write_json(payload, out)

    # ...and the flat list every existing reader of the accuracy record expects.
    flat = out.with_name(args.accuracies_name)
    write_json(payload["scores"], flat)

    values = [value for entry in payload["scores"] for value in entry.values()]
    print(
        json.dumps(
            {
                "fold": payload["fold"],
                "writers": payload["writers"],
                "skipped": len(payload["skipped"]),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": (sum(values) / len(values)) if values else None,
                "scores": str(out),
                "accuracies": str(flat),
            },
            indent=2,
        )
    )
    return 0



def cmd_average_scores(args: argparse.Namespace) -> int:
    """Merge per-fold writer scores into one ranking."""
    import json as _json

    from federated_outlier_adaptation.outliers.scoring import (
        average_score_files,
        write_json,
    )

    payloads = []
    for path in args.inputs:
        with open(path) as handle:
            payloads.append(_json.load(handle))
    merged = average_score_files(payloads)

    out = Path(args.out)
    write_json(merged, out)
    flat = out.with_name(args.accuracies_name)
    write_json(merged["scores"], flat)
    print(_json.dumps({k: merged[k] for k in
                       ("rule", "folds", "writers", "fold_spread")}, indent=2))
    if merged["dropped_incomplete"]:
        print(f"dropped (not scored by every fold): {len(merged['dropped_incomplete'])}")
    return 0


def cmd_split_pools(args: argparse.Namespace) -> int:
    """Cut the population into a bad pool and a good pool by the coarse detector."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import (
        split_pools,
        write_json,
        write_pools_csv,
    )
    from federated_outlier_adaptation.outliers.selection import eligible_of
    from federated_outlier_adaptation.training.extreme_cells import accuracy_map

    provider = _resolve_provider(args)
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))

    scores_path = Path(args.scores) if args.scores else (
        results_dir / "outliers" / "clients_acc_on_global.json"
    )
    with open(scores_path) as handle:
        accuracies = accuracy_map(json.load(handle))

    payload = split_pools(
        accuracies,
        bad_fraction=args.bad_fraction,
        eligible=eligible_of(provider) if args.require_trainable else None,
    )
    payload["scores"] = str(scores_path)

    out = Path(args.out) if args.out else results_dir / "outliers" / "pools.json"
    write_json(payload, out)
    write_pools_csv(payload, Path(args.csv) if args.csv else out.with_suffix(".csv"))

    # The two lists, in the shape every reader of a pool file expects, so the
    # old-data draw can simply exclude the bad one.
    for name in ("bad", "good"):
        target = out.parent / f"pool_{name}.json"
        write_json(
            {"rule": payload["rule"], "pool": name, "size": len(payload[name]),
             "clients": payload[name],
             "accuracies": {w: payload["accuracies"][w] for w in payload[name]}},
            target,
        )
        print(f"Wrote {target}")

    print(json.dumps(
        {k: v for k, v in payload.items()
         if k not in ("bad", "good", "accuracies")}, indent=2,
    ))
    print(f"Wrote {out}")
    return 0


def cmd_score_pool(args: argparse.Namespace) -> int:
    """Score a model on every row of each writer in a pool."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import (
        score_pool,
        write_json,
        write_scores_csv,
    )
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    provider = _resolve_provider(args)
    model = _load_checkpoint(provider, args.model_path)
    clients, _ = load_client_pool(args.clients_file)

    payload = score_pool(provider, model, clients, batch_size=args.batch_size)
    payload["model"] = str(args.model_path)
    payload["clients_file"] = str(args.clients_file)

    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = Path(args.out) if args.out else (
        results_dir / "outliers" / "bad_scores_on_g0.json"
    )
    write_json(payload, out)
    write_scores_csv(payload, Path(args.csv) if args.csv else out.with_suffix(".csv"))

    if args.accuracies_name:
        # The flat ranking, in the shape the cohort cut reads.
        flat = out.with_name(args.accuracies_name)
        write_json(payload["scores"], flat)
        print(f"Wrote {flat}")

    print(json.dumps(
        {k: v for k, v in payload.items()
         if k not in ("scores", "accuracies", "samples", "rank", "ranking")},
        indent=2,
    ))
    print(f"Wrote {out}")
    return 0


def cmd_writer_counts(args: argparse.Namespace) -> int:
    """Bank how much of each class every writer holds - the selection's paper trail."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import (
        write_counts_csv,
        write_json,
        writer_counts,
    )

    provider = _resolve_provider(args)
    payload = writer_counts(provider.dataset)

    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = Path(args.out) if args.out else results_dir / "outliers" / "writer_counts.json"
    write_json(payload, out)
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    write_counts_csv(payload, csv_path)

    print(json.dumps(
        {k: v for k, v in payload.items()
         if k not in ("per_writer_total", "per_writer_per_class")},
        indent=2,
    ))
    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit a study chain from its manifest, or refuse having submitted none."""
    from federated_outlier_adaptation.submission import (
        ManifestError, load_manifest, submit,
    )

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 78

    result = submit(manifest, dry_run=not args.go)
    if not result.get("submitted") and result.get("problems"):
        return 78
    if not result.get("submitted") and result.get("failed_at"):
        return 1
    return 0


def cmd_select_eligible(args: argparse.Namespace) -> int:
    """Cut the eligible population from a writer-counts artefact."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.eligibility import (
        eligibility_record,
        load_totals,
        write_eligibility,
    )

    provider = _resolve_provider(args)
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    counts = Path(args.counts) if args.counts else results_dir / "outliers" / "writer_counts.json"
    if not counts.is_file():
        print(f"FATAL: no writer-counts artefact at {counts}.", file=sys.stderr)
        return 1
    try:
        totals = load_totals(counts)
    except ValueError as error:
        print(f"FATAL: {counts}: {error}", file=sys.stderr)
        return 1

    record = eligibility_record(totals, args.min_samples)
    if not record["clients"]:
        print(
            f"FATAL: no writer holds {args.min_samples} rows; the floor excludes "
            f"the whole population of {len(totals)}.",
            file=sys.stderr,
        )
        return 1
    record["source"] = str(counts)
    out = Path(args.out) if args.out else results_dir / "outliers" / "eligible.json"
    write_eligibility(record, out)
    print(json.dumps({k: v for k, v in record.items() if k != "clients"}, indent=2))
    print(f"Wrote {out}")
    return 0


def cmd_check_population(args: argparse.Namespace) -> int:
    """
    Assert what a chain stage must be able to assume about its own artefacts.

    Returns non-zero on the first stage that is wrong, so a ``%1`` chain stops
    where the mistake is instead of carrying it forward.
    """
    from federated_outlier_adaptation.outliers import checks

    clients = checks.load_clients(args.clients_file)
    problems = checks.check_size(clients, args.expect_size)

    subset_of = {Path(p).name: checks.load_clients(p) for p in (args.subset_of or [])}
    disjoint = {Path(p).name: checks.load_clients(p) for p in (args.disjoint_from or [])}
    problems += checks.check_membership(clients, subset_of, disjoint)

    if args.fold_book:
        from federated_outlier_adaptation.data.fold_book import FoldBook

        book = FoldBook.load(args.fold_book)
        folds = args.folds or [1]
        problems += checks.check_book(book, clients, folds)

    totals = None
    if args.counts and Path(args.counts).is_file():
        from federated_outlier_adaptation.outliers.eligibility import load_totals

        try:
            totals = load_totals(args.counts)
        except ValueError:
            totals = None

    label = args.label or Path(args.clients_file).name
    if problems:
        print(f"REFUSED: {label}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"{label}: {checks.describe(clients, totals)} - all checks pass")
    if args.show:
        print(json.dumps(clients, indent=2))
    return 0


def cmd_promote_model(args: argparse.Namespace) -> int:
    """
    Copy a trained model into the name the runs resolve, and record its checksum.

    ``--init global --global-name NAME`` looks for ``<results>/<provider>/NAME_model``,
    which is not where a training stage writes.  Promoting is what closes that
    gap, and doing it with a checksum is what makes the model every later run
    anchors on identified rather than assumed.
    """
    import hashlib
    import shutil

    source = Path(args.source)
    if not source.is_file():
        print(f"FATAL: no model at {source}.", file=sys.stderr)
        return 1
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    record = {
        "name": args.name,
        "source": str(source),
        "model": str(target),
        "sha256": digest,
        "rule": args.rule or "promoted from a single training run",
    }
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


def cmd_cohort_table(args: argparse.Namespace) -> int:
    """Bank the cohort with the evidence behind its selection."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import (
        cohort_table,
        write_cohort_table_csv,
        write_json,
    )
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.training.extreme_cells import accuracy_map

    provider = _resolve_provider(args)
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))

    clients, _ = load_client_pool(args.clients_file)
    scores_path = Path(args.scores) if args.scores else (
        results_dir / "outliers" / "clients_acc_on_global.json"
    )
    with open(scores_path) as handle:
        accuracies = accuracy_map(json.load(handle))
    counts_path = Path(args.counts) if args.counts else (
        results_dir / "outliers" / "writer_counts.json"
    )
    with open(counts_path) as handle:
        counts = json.load(handle)

    book = _load_book(args.fold_book) if args.fold_book else None

    extra: dict = {}
    for pair in args.extra_scores or []:
        if "=" not in pair:
            raise SystemExit(f"--extra-scores expects NAME=PATH, got {pair!r}")
        name, _, path = pair.partition("=")
        with open(path) as handle:
            extra[name.strip()] = accuracy_map(json.load(handle))

    payload = cohort_table(
        clients, accuracies, counts, book=book,
        extra_scores=extra or None, rule=args.rule,
    )
    payload["clients_file"] = str(args.clients_file)
    payload["scores"] = str(scores_path)
    payload["counts"] = str(counts_path)
    if args.fold_book:
        payload["fold_book"] = str(args.fold_book)

    out = Path(args.out) if args.out else results_dir / "tables" / "cohort_table.json"
    write_json(payload, out)
    csv_path = Path(args.csv) if args.csv else out.with_suffix(".csv")
    write_cohort_table_csv(payload, csv_path)

    print(json.dumps(
        {k: v for k, v in payload.items() if k != "rows"}, indent=2
    ))
    print(f"Wrote {out}")
    print(f"Wrote {csv_path}")
    return 0


def cmd_draw_cohort(args: argparse.Namespace) -> int:
    """Draw a cohort at random from a ranked population (the T2 and T3 studies)."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import draw_cohort, write_json
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.training.extreme_cells import accuracy_map

    provider = _resolve_provider(args)
    dataset = provider.dataset
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))

    scores_path = Path(args.scores) if args.scores else (
        results_dir / "outliers" / "clients_acc_on_global.json"
    )
    with open(scores_path) as handle:
        accuracies = accuracy_map(json.load(handle))

    splittable = None
    if args.require_trainable and hasattr(dataset, "trainable_writers"):
        splittable = dataset.trainable_writers()
    exclude, _ = load_client_pool(args.exclude_file) if args.exclude_file else ([], {})

    payload = draw_cohort(
        accuracies,
        size=args.k,
        seed=args.seed,
        pool=args.pool,
        splittable=splittable,
        exclude=exclude,
        best=args.best,
    )
    payload["tag"] = args.tag
    payload["scores"] = str(scores_path)
    out = Path(args.out) if args.out else results_dir / "outliers" / f"cohort_random{args.k}.json"
    write_json(payload, out)
    print(json.dumps({k: v for k, v in payload.items() if k != "accuracies"}, indent=2))
    print(f"Wrote {out}")
    return 0


def cmd_draw_old_data(args: argparse.Namespace) -> int:
    """Draw the old-data population from the writers the cohort does not take."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.scoring import draw_old_data, write_json
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    provider = _resolve_provider(args)
    dataset = provider.dataset
    counts = {writer: dataset.get_sample_count(writer) for writer in dataset.all_writers()}

    exclude, _ = load_client_pool(args.exclude_file) if args.exclude_file else ([], {})
    population = population_source = None
    if args.from_pool:
        population, _ = load_client_pool(args.from_pool)
        population_source = str(args.from_pool)
        if not population:
            print(f"FATAL: {args.from_pool} names no clients.", file=sys.stderr)
            return 1
    splittable = None
    if args.require_trainable and hasattr(dataset, "trainable_writers"):
        splittable = dataset.trainable_writers()

    payload = draw_old_data(
        counts,
        exclude=exclude,
        size=args.size,
        seed=args.seed,
        min_samples=args.min_samples,
        splittable=splittable,
        population=population,
        population_source=population_source,
    )
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = Path(args.out) if args.out else results_dir / "outliers" / "old_data.json"
    write_json(payload, out)
    print(json.dumps({k: v for k, v in payload.items() if k != "samples"}, indent=2))
    print(f"Wrote {out}")
    return 0


def cmd_evaluate_book(args: argparse.Namespace) -> int:
    """Score a saved model on one part of one fold of a book."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.training.evaluate import (
        evaluate_on_book,
        write_evaluation,
    )

    _require_book_and_fold(args, "evaluate-book")
    provider = _resolve_provider(args)
    book = _load_book(args.fold_book)
    model = _load_checkpoint(provider, args.model_path)
    writers = (
        load_client_pool(args.clients_file)[0] if args.clients_file else list(book.writers)
    )

    payload = evaluate_on_book(
        provider, model, book, fold=args.fold, writers=writers,
        part=args.part, batch_size=args.batch_size,
    )
    payload["tag"] = args.tag or f"fold{args.fold}_{args.part}"
    payload["model"] = str(args.model_path)
    payload["fold_book"] = str(args.fold_book)

    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = Path(args.out) if args.out else results_dir / "evaluations.json"
    write_evaluation(payload, out)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_writer"}, indent=2))
    return 0


def cmd_isolated_train(args: argparse.Namespace) -> int:
    """Train one private model per cohort client and score all three sets."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.outliers.selection import load_client_pool
    from federated_outlier_adaptation.training.isolated import (
        isolated_training,
        write_payload,
    )
    from federated_outlier_adaptation.utils.seeding import set_run_seed

    # THE ISOLATED ARM WAS THE ONE THAT DID NOT SEED. global-train and the
    # federated runner both call this; isolated-train did not, so the private
    # baselines drew a fresh initialisation and a fresh shuffle every run. They
    # are two of the four rungs every reported number is read against, which
    # made the comparison itself irreproducible while the arms being compared
    # were fine.
    set_run_seed(args.seed)

    _require_book_and_fold(args, "isolated-train")
    provider = _resolve_provider(args)
    cohort_book = _load_book(args.fold_book)
    old_book = _load_book(args.old_book)

    clients, _ = (
        load_client_pool(args.clients_file)
        if args.clients_file
        else (list(cohort_book.writers), {})
    )
    old_clients = (
        load_client_pool(args.old_clients_file)[0] if args.old_clients_file else None
    )

    payload = isolated_training(
        provider,
        clients=clients,
        cohort_book=cohort_book,
        fold=args.fold,
        old_book=old_book,
        old_fold=args.old_fold,
        old_clients=old_clients,
        only_client=args.only_client,
        pooled=args.pooled,
        init=args.init,
        init_checkpoint=args.init_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
        eval_batch_size=args.eval_batch_size,
    )
    payload["fold_book"] = str(args.fold_book)
    payload["old_book"] = str(args.old_book)

    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    if args.pooled:
        default_name = f"centralized_outliers_{args.init}_fold{args.fold}.json"
    elif args.only_client:
        default_name = f"isolated_{args.init}_{args.only_client}_fold{args.fold}.json"
    else:
        default_name = f"isolated_{args.init}_fold{args.fold}.json"
    out = Path(args.out) if args.out else results_dir / default_name
    write_payload(payload, out)
    keys = (
        ("stage", "init", "fold", "old_folds", "models", "pooled_train_rows",
         "pooled_val_rows", "pooled", "old_mean", "old_sd", "convergence",
         "union_test_samples", "old_test_samples")
        if args.pooled
        else ("init", "fold", "old_folds", "only_client", "trained", "skipped",
              "pooled", "union_test_samples", "old_test_samples")
    )
    print(json.dumps({key: payload[key] for key in keys}, indent=2))
    return 0


def cmd_fold_book(args: argparse.Namespace) -> int:
    """Materialise a cross-validation split and write it down."""
    import json as _json

    from federated_outlier_adaptation.data.fold_book import (
        build_fold_book,
        write_fold_book,
    )
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    provider = _resolve_provider(args)
    dataset = provider.dataset

    if args.writers:
        writers = list(args.writers)
    elif args.clients_file:
        writers, _ = load_client_pool(args.clients_file)
    else:
        writers = None                      # every writer of the dataset

    book = build_fold_book(
        dataset,
        writers=writers,
        folds=args.folds,
        seed=args.seed,
        train_rate=args.train_rate,
        eval_rate=args.eval_rate,
        tag=args.tag,
    )
    path = write_fold_book(book, args.out)

    summary = {
        "path": str(path),
        "tag": book.metadata.get("tag"),
        "folds": book.folds,
        "writers": len(book.writers),
        "rows": book.rows,
        "seed": book.metadata.get("seed"),
        "rates": [book.metadata.get("train_rate"), book.metadata.get("eval_rate")],
        "fold_counts": book.metadata.get("fold_counts"),
        "unsplittable": book.metadata.get("unsplittable"),
    }
    print(_json.dumps(summary, indent=2))
    return 0


def cmd_outlier_figure(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.analysis.outlier_gallery import render_outlier_gallery
    from federated_outlier_adaptation.outliers.selection import load_client_pool

    provider = _resolve_provider(args)
    if args.outliers_file:
        clients, _ = load_client_pool(args.outliers_file)
    elif args.pool_frac is not None:
        clients = list(provider.outlier_pool(args.pool_frac))
    else:
        clients = list(provider.selected_clients(k=args.k))
    if args.clients:
        clients = list(args.clients)

    source = None
    if args.source_clients_file:
        source, _ = load_client_pool(args.source_clients_file)
    if args.source_clients:
        source = list(args.source_clients)

    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))
    out = args.out or (results_dir / "outliers" / "outlier_gallery.png")
    summary = render_outlier_gallery(
        provider,
        clients=clients,
        out_path=out,
        source_clients=source,
        labels=args.labels,
        n_examples=args.n_examples,
        max_columns=args.max_columns,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.analysis.top_selector import top_6_configs
    from federated_outlier_adaptation.plotting.grid_heatmaps import plot_all_grid_heatmaps

    root = args.root or str(config.RESULTS_DIR)
    top_6_configs(root_folder=root)
    plot_all_grid_heatmaps(results_dir=root)
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foa",
        description=(
            "Knowledge-preserving adaptation to outlier clients in federated learning."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- prepare-data ---
    p = sub.add_parser("prepare-data", help="Build the packed dataset cache from by_write.zip.")
    _add_common(p)
    p.add_argument(
        "--zip",
        default=None,
        help="Path to the raw archive (never extracted); required except for --dataset mnist.",
    )
    p.add_argument("--labels", default=None, help="digits_labels.json manifest.")
    p.add_argument("--out", default=None, help="Cache output directory.")
    p.add_argument("--size", type=int, default=128, help="Target image edge length.")
    p.add_argument("--probe", type=int, default=2000, help="Images probed for bilevel packing.")
    p.add_argument(
        "--by-class-log",
        default=None,
        help=(
            "SD19 by_class_md5.log; carries the character label of every scan "
            "(--resolution 28). Downloaded next to the archive when absent."
        ),
    )
    p.add_argument(
        "--by-write-log",
        default=None,
        help="SD19 by_write_md5.log; joins the labels to the writer paths.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        choices=sorted(PREPARABLE_DATASETS),
        help="Dataset to prepare; defaults to --provider. 'mnist' builds the NIST proxy set.",
    )
    p.add_argument(
        "--raw-dir",
        default=None,
        help="MNIST only: directory holding the four idx .gz archives.",
    )
    p.set_defaults(func=cmd_prepare_data)

    # --- average-scores ---
    p = sub.add_parser(
        "average-scores",
        help="Merge per-fold writer scores into one ranking.",
    )
    _add_common(p)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One score file per fold.")
    p.add_argument("--out", required=True)
    p.add_argument("--accuracies-name", default="clients_acc_on_global.json",
                   help="Flat accuracy file written beside --out.")
    p.set_defaults(func=cmd_average_scores)

    # --- global-train ---
    p = sub.add_parser("global-train", help="Train the global model and its artefacts.")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--split-seed", type=int, default=42, help="Seed of the dataset split.")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Run seed: the model initialisation and the training shuffle. "
            "Independent of --split-seed, so several global models can be "
            "trained on one and the same writer split."
        ),
    )
    p.add_argument("--k", type=int, default=5, help="Number of outliers to select.")
    p.add_argument(
        "--k-values",
        type=int,
        nargs="*",
        default=None,
        help="Selection sizes for the additional datasets, e.g. --k-values 5 20 50.",
    )
    p.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience (non-NIST providers)."
    )
    p.add_argument(
        "--pool-frac",
        type=float,
        default=None,
        help="Also write the rule-based outlier pool of this bottom fraction.",
    )
    p.add_argument(
        "--require-trainable",
        action="store_true",
        help=(
            "Exclude clients that cannot form a local training split. A "
            "stratified 60/40 split cuts per label, so at 62 classes a writer "
            "holding one sample per class has neither a train nor a validation "
            "half and cannot participate at all. Off by default, which leaves "
            "every existing pool unchanged."
        ),
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop once the validation accuracy has not improved for N epochs, "
            "and keep the weights of the best validation epoch. Off by default, "
            "which runs every epoch and keeps the last one - the published "
            "behaviour."
        ),
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Epochs that always run before the patience rule may fire. "
            "Validation accuracy on a small split is noisy early on, so an "
            "opening plateau of a few epochs should not end the run."
        ),
    )
    p.add_argument(
        "--writers-file",
        default=None,
        help="With --population file: the client list to train on (g-0).",
    )
    p.add_argument(
        "--population",
        default="source",
        choices=("source", "all", "file"),
        help=(
            "Writers the model trains on. 'source' (default) is the published "
            "pipeline - the held-out writers stay held out and are scored "
            "afterwards. 'all' trains on every writer of the dataset: the "
            "centralised ceiling, and g-init of the v6 protocol. 'file' trains "
            "on exactly the writers of --writers-file, which is how g-0 is "
            "trained on the drawn old data. With 'all' or 'file' there is no "
            "held-out population, so the outlier phases are skipped and "
            "discovery runs separately against the model."
        ),
    )
    p.add_argument(
        "--skip-outlier-training",
        action="store_true",
        help=(
            "Stop after the outlier selection instead of training one personal "
            "model per selected writer. Off by default, so the published "
            "pipeline is unchanged. That step is the old Fig-4 baseline; where "
            "the reference ladder covers the same rung ('foa local-finetune', "
            "R1/R2) it is redundant and costs a further --epochs epochs per "
            "selected writer in every run."
        ),
    )
    p.set_defaults(func=cmd_global_train)

    # --- global-train-fl ---
    p = sub.add_parser(
        "global-train-fl", help="Obtain the global model by FedAvg over the server clients."
    )
    _add_common(p)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--participation", type=float, default=0.1)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--global-name", default="global_fl", help="Output model name.")
    p.add_argument("--force", action="store_true", help="Re-run even if the model exists.")
    p.set_defaults(func=cmd_global_train_fl)

    # --- select-outliers ---
    p = sub.add_parser("select-outliers", help="Select the low-accuracy clients.")
    _add_common(p)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bottom-frac", type=float, default=0.005)
    p.add_argument(
        "--mode",
        default="sample",
        choices=SELECTION_MODES,
        help=(
            "sample: the published random draw from the bottom fraction; "
            "lowest: the k lowest-accuracy clients, deterministic; "
            "pool: a rule-based population to sample participants from; "
            "worst: a fixed cohort of the k worst-scoring eligible clients, "
            "enrolled once for the whole of training."
        ),
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Label stored with a cohort (--mode worst), e.g. its scenario name.",
    )
    p.add_argument(
        "--pool-frac",
        type=float,
        default=0.05,
        help="Bottom fraction of writers forming the pool (--mode pool).",
    )
    p.add_argument(
        "--pool-threshold",
        type=float,
        default=None,
        help="Accuracy threshold rule for --mode pool; wins over --pool-frac.",
    )
    p.add_argument(
        "--pool-max-acc",
        type=float,
        default=None,
        metavar="A",
        help=(
            "Pool rule by accuracy: every held-out client scoring below A under "
            "the global model. The severity cuts are stated this way, because a "
            "threshold describes the accuracy distribution while a fraction "
            "describes the population size. Wins over --pool-threshold."
        ),
    )
    p.add_argument(
        "--severity",
        default=None,
        choices=sorted(SEVERITIES),
        help=(
            "Tag written into the pool file and into the provenance of every "
            "run drawn from it."
        ),
    )
    p.add_argument(
        "--scores",
        default=None,
        help=(
            "The ranking to cut the cohort from. Default: the root's "
            "outliers/clients_acc_on_global.json. Pass the SHIPPED model's "
            "scores when the coarse detector only decided who was worth "
            "looking at and the shipped model decides who the cohort is."
        ),
    )
    p.add_argument(
        "--no-accuracy-table",
        action="store_true",
        help=(
            "Skip the per-client accuracy CSV (--mode pool writes it by "
            "default; it is what the severity cuts are read off)."
        ),
    )
    p.add_argument(
        "--require-trainable",
        action="store_true",
        help=(
            "Exclude clients that cannot form a local training split. A "
            "stratified 60/40 split cuts per label, so at 62 classes a writer "
            "holding one sample per class has neither a train nor a validation "
            "half and cannot participate at all. Off by default, which leaves "
            "every existing pool unchanged."
        ),
    )
    p.add_argument("--force", action="store_true", help="Re-select even if the list exists.")
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Destination for the cohort file (--mode worst). Without it the "
            "cohort lands in the provider's own outliers directory, which for a "
            "non-NIST provider is <results>/<provider>/outliers - one level "
            "deeper than the rest of a study reads."
        ),
    )
    p.set_defaults(func=cmd_select_outliers)

    # --- combined-train ---
    p = sub.add_parser("combined-train", help="Train the combined global + outliers model.")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop once the validation accuracy has not improved for N epochs, "
            "and keep the weights of the best validation epoch. Off by default, "
            "which runs every epoch and keeps the last one - the published "
            "behaviour."
        ),
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Epochs that always run before the patience rule may fire. "
            "Validation accuracy on a small split is noisy early on, so an "
            "opening plateau of a few epochs should not end the run."
        ),
    )
    p.add_argument("--k", type=int, default=5, help="Participants when the provider resolves them.")
    p.add_argument(
        "--outliers-file",
        default=None,
        help=(
            "Client list or rule-based pool file to pool with the source "
            "writers. This is what makes the run the pooled-oracle ceiling of "
            "one client pool rather than of the published five writers."
        ),
    )
    p.add_argument(
        "--name",
        default="global_clients",
        help=(
            "Prefix of the two output folders (default: global_clients). Give "
            "one ceiling per pool its own name so they do not overwrite."
        ),
    )
    p.set_defaults(func=cmd_combined_train)

    # --- base-fl ---
    p = sub.add_parser("base-fl", help="Baseline federated runs.")
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_stop_option(p)
    _add_global_name(p)
    p.add_argument("--parent", default="prove_fl", help="Parent output folder prefix.")
    p.set_defaults(func=cmd_base_fl)

    # --- all-aggs ---
    p = sub.add_parser("all-aggs", help="Baseline trainer against every aggregation method.")
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_global_name(p)
    _add_extended_aggregations(p)
    p.add_argument(
        "--parent", default="all_aggs_fl", help="Parent output folder prefix."
    )
    p.set_defaults(func=cmd_all_aggs)

    # --- grid ---
    p = sub.add_parser("grid", help="Run a hyperparameter sweep.")
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_global_name(p)
    _add_population_options(p)
    _add_budget_options(p)
    _add_eval_options(p)
    p.add_argument("--method", required=True, choices=sorted(GRID_METHODS))
    p.add_argument(
        "--parent",
        default=None,
        help="Output folder of the sweep (default: the sweep's published folder).",
    )
    p.add_argument("--case", choices=EXTREME_CASES, default="single", help="Extreme case to sweep.")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Anchored sweeps only: file index of the written points, i.e. the "
            "<n> of accuracies_points_<n>.json. A second invocation with a "
            "different index adds its configurations to the same sweep instead "
            "of overwriting it - which is how a 'penalty off' arm joins a "
            "space's grid as one extra point."
        ),
    )
    p.add_argument(
        "--capped-tasks",
        action="store_true",
        help=(
            "Distillation sweeps only: use the published capped-rule task list "
            "instead of the FedAvg pair the finals run. Off by default, so a "
            "swept hyperparameter is selected under the rule it is executed with."
        ),
    )
    p.add_argument(
        "--space",
        default="kd",
        choices=sorted(ANCHOR_SPACE_NAMES),
        help="Distance space of the anchored sweep (--method anchored).",
    )
    p.add_argument(
        "--anchor",
        default="frozen",
        choices=("frozen", "current"),
        help="Anchor model of the anchored sweep (--method anchored).",
    )
    p.add_argument(
        "--lams",
        type=float,
        nargs="+",
        default=None,
        metavar="LAM",
        help=(
            "Penalty weights of the anchored sweep (--method anchored); "
            "default: the published range."
        ),
    )
    p.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "Temperatures of the anchored sweep (--method anchored, softmax "
            "spaces only); default: the published range."
        ),
    )
    p.set_defaults(func=cmd_grid)

    # --- final ---
    p = sub.add_parser("final", help="Final runs for one trainer.")
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_population_options(p)
    _add_stop_option(p)
    _add_global_name(p)
    _add_extended_aggregations(p)
    _add_aggregation_options(p)
    _add_budget_options(p)
    _add_eval_options(p)
    p.add_argument("--trainer", required=True, help="Trainer class name.")
    p.add_argument(
        "--source-share",
        default="off",
        choices=sorted(SOURCE_SHARE_MODES),
        help=(
            "Data-sharing upper bound: mix source-writer training data into "
            "every participating client's epoch. 'equal' draws a fresh sample "
            "the size of the client's own train split (a balanced 50/50 "
            "mixture); 'full' uses the whole source training pool every epoch. "
            "Off by default - the privacy constraint the rest of the study "
            "works under forbids it, and this arm exists to price it."
        ),
    )
    p.add_argument(
        "--source-share-cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Upper bound on the source images added per client-epoch. The "
            "'full' pool is two orders of magnitude larger than a client, so a "
            "documented cap is what keeps that arm affordable; it is recorded "
            "with the run."
        ),
    )
    p.add_argument(
        "--source-share-multiplier",
        type=float,
        default=1.0,
        metavar="K",
        help=(
            "How many times the client's own training volume the shared old "
            "data contributes per epoch, with --source-share equal. 1 (the "
            "default) is the balanced 50/50 mixture; the access-to-old-data "
            "study sweeps 1, 2, 4 and 8. Ignored by --source-share full, which "
            "is the whole pool by definition. Only the old book's TRAIN rows "
            "are ever shared - its validation and test rows are the "
            "preservation measurement."
        ),
    )
    p.add_argument(
        "--val-blend-source",
        type=float,
        default=0.0,
        metavar="RHO",
        help=(
            "MONITORING WITHOUT TRAINING: make the early-stopping criterion a "
            "blend of the client's own validation rows and the old book's "
            "fold-k VALIDATION rows, RHO being the old share of the blended "
            "set. No old row ever enters a gradient. 0.5 gives equal counts, "
            "so the accuracy over the blend is the arithmetic mean of the two "
            "accuracies. 0 (the default) is the published criterion, unchanged."
        ),
    )
    p.add_argument(
        "--save-final-model",
        action="store_true",
        help=(
            "Write the final server model of every family job to "
            "<run folder>/final_model.pt. Off by default: no existing run "
            "produces one, and the personalisation arm needs it as its starting "
            "point."
        ),
    )
    p.add_argument(
        "--old-book",
        default=None,
        metavar="PATH",
        help=(
            "Fold book of the old data. With it, the final server model is "
            "scored on EVERY fold of that book's test rows separately and the "
            "five accuracies are reported with their mean and standard "
            "deviation - the preservation number of the paper. The runner's "
            "per-round source series is kept, but it is measured on the "
            "pre-book source notion and is not that number."
        ),
    )
    p.add_argument(
        "--old-clients-file",
        default=None,
        metavar="PATH",
        help="Old-data writer list; default: everyone --old-book covers.",
    )
    p.add_argument(
        "--old-fold",
        default="all",
        metavar="K|all",
        help=(
            "Which folds of --old-book to score the final model on. Default "
            "'all': five partitions of one population, so the spread across "
            "them is the error bar on preservation and merging them would "
            "throw it away."
        ),
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Batch size of the final-model evaluations (default: 256).",
    )
    p.add_argument("--parent", default="final_result", help="Parent output folder prefix.")
    p.set_defaults(func=cmd_final)

    # --- extreme ---
    p = sub.add_parser("extreme", help="Minimal-client (extreme case) final runs.")
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_population_options(p)
    _add_stop_option(p)
    _add_global_name(p)
    _add_aggregation_options(p)
    _add_budget_options(p)
    _add_eval_options(p)
    p.add_argument("--case", choices=EXTREME_CASES, default="single")
    p.add_argument(
        "--clients",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Client ids of the extreme case, in place of the published "
            "defaults. Repeat an id to make it two participants holding "
            "identical data (the 'dual' case). The defaults in "
            "constants.EXTREME_CASE_CLIENTS are writers of the published "
            "128x128 digit setting; a different population needs its own ids, "
            "and naming ids that are absent is now an error rather than an "
            "empty run."
        ),
    )
    p.add_argument("--trainer", default="DistillationTrainer")
    p.add_argument("--parent", default="", help="Parent output folder prefix.")
    p.set_defaults(func=cmd_extreme)

    # --- matrix ---
    p = sub.add_parser("matrix", help="Generate the task list of an experiment plan.")
    _add_common(p)
    p.add_argument(
        "--plan",
        default=None,
        help=(
            "Plan name; use --list to see them all. The pseudo-plan 'resume' "
            "re-emits the unfinished lines of --from TASKFILE."
        ),
    )
    p.add_argument("--list", action="store_true", help="List every plan with its task count.")
    p.add_argument("--out", default=None, help="Write the tasks to this file.")
    p.add_argument(
        "--minutes-per-task",
        type=float,
        default=12.0,
        help="Assumed wall-clock minutes per task for the GPU-hour estimate.",
    )
    p.add_argument(
        "--seconds-per-round",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Measured seconds per federated round of a finals run. Given, the "
            "estimate counts the runs and rounds every task expands into "
            "instead of assuming a flat cost per task."
        ),
    )
    p.add_argument(
        "--sweep-seconds-per-round",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Measured seconds per round of a sweep run (batch 512, 50 local "
            "epochs); defaults to --seconds-per-round."
        ),
    )
    p.add_argument(
        "--from",
        dest="source",
        default=None,
        metavar="TASKFILE",
        help="Task file --plan resume filters down to its unfinished lines.",
    )
    p.set_defaults(func=cmd_matrix)

    # --- local-finetune ---
    p = sub.add_parser(
        "local-finetune", help="No-federation reference: each client fine-tunes alone."
    )
    _add_common(p)
    _add_run_options(p)
    _add_overrides(p)
    _add_global_name(p)
    p.add_argument(
        "--pool-frac",
        type=float,
        default=None,
        help="Use the provider's rule-based outlier pool of this bottom fraction.",
    )
    p.add_argument("--trainer", default="BaseTrainer", help="Trainer class name.")
    p.add_argument("--epochs", type=int, default=10, help="Local epochs per client.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--init",
        default="global",
        choices=sorted(INITIALISATIONS),
        help=(
            "Starting point of every client: 'global' (default) fine-tunes the "
            "global model - ladder rung R2 - and 'scratch' trains from a fresh "
            "initialisation on the client's own data alone - rung R1."
        ),
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop once the validation accuracy has not improved for N epochs, "
            "and keep the weights of the best validation epoch. Off by default, "
            "which runs every epoch and keeps the last one - the published "
            "behaviour."
        ),
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        metavar="M",
        help=(
            "Epochs that always run before the patience rule may fire. "
            "Validation accuracy on a small split is noisy early on, so an "
            "opening plateau of a few epochs should not end the run."
        ),
    )
    p.add_argument(
        "--init-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Start every client from this checkpoint instead of the named "
            "global model - the personalisation arm: point it at a federated "
            "run's final server model (written by 'foa final "
            "--save-final-model') and each client fine-tunes its own copy of "
            "what the federation produced. The same command without it is R2, "
            "the control that isolates what the federated start was worth. The "
            "path and its SHA-256 are recorded in the result."
        ),
    )
    p.add_argument(
        "--parent", default="local_finetune", help="Parent output folder prefix."
    )
    p.set_defaults(func=cmd_local_finetune)

    # --- select ---
    p = sub.add_parser("select", help="Constrained, validation-based configuration selection.")
    _add_common(p)
    p.add_argument("--root", default=None, help="Results folder to scan.")
    p.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[0.005],
        help=(
            "Allowed drop of the source accuracy (default: 0.005). Several "
            "budgets may be given; only --emit-finals uses more than the first."
        ),
    )
    p.add_argument(
        "--on",
        default="val",
        choices=("val", "test"),
        help="Source split the constraint is evaluated on (default: val).",
    )
    p.add_argument("--window", type=int, default=5, help="Final rounds averaged.")
    p.add_argument(
        "--emit-finals",
        action="store_true",
        help=(
            "Select every stored regularisation sweep at every --eps and write "
            "the 'foa final' task lines carrying the selected lam (and T)."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Seeds of the emitted final runs (default: 1 2 3).",
    )
    p.add_argument(
        "--final-aggregation",
        nargs="+",
        default=["fedavg"],
        metavar="RULE",
        help=(
            "Aggregation rules the emitted finals are executed with (default: "
            "fedavg). Naming more than one runs each selection under each rule, "
            "in its own output folder - e.g. the FedAvg pair and whichever rule "
            "the aggregation phase found best."
        ),
    )
    p.add_argument(
        "--final-args",
        default=None,
        metavar="FLAGS",
        help=(
            "Flags appended verbatim to every emitted final line, e.g. the "
            "results root, the resolution and character set and the budget."
        ),
    )
    p.add_argument(
        "--final-parent-prefix",
        default=None,
        metavar="PREFIX",
        help=(
            "Prepended to every emitted final's output folder, so two scenarios "
            "can share one results root without colliding, e.g. 'coh_'."
        ),
    )
    p.add_argument(
        "--final-pool-args",
        default=None,
        metavar="FLAGS",
        help=(
            "Replaces the --pool-frac population block of the emitted lines, "
            "for a plan that names its client pool by path. The sampler seed is "
            "added per line."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Task file with --emit-finals, otherwise the directory the "
            "selection files are written to."
        ),
    )
    p.set_defaults(func=cmd_select)

    # --- signals ---
    p = sub.add_parser(
        "signals",
        help="Analyse the constraint-respecting forgetting signals of stored runs.",
    )
    _add_common(p)
    p.add_argument("--root", default=None, help="Results folder to scan.")
    p.add_argument("--out", default=None, help="Where to write the tables and plots.")
    p.add_argument(
        "--eps",
        type=float,
        default=0.005,
        help="Forgetting budget of the oracle selection (default: 0.005).",
    )
    p.add_argument("--window", type=int, default=5, help="Final rounds averaged.")
    p.add_argument(
        "--deltas",
        type=float,
        nargs="*",
        default=None,
        help="Budget grid of the simulated stopping rules (default: a decade grid).",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip the Pareto-style plots.")
    p.set_defaults(func=cmd_signals)

    # --- report ---
    p = sub.add_parser(
        "report",
        help="Build the tables and figures of the experiment report from stored runs.",
    )
    _add_common(p)
    p.add_argument("--root", default=None, help="Results folder to scan.")
    p.add_argument(
        "--out",
        default=None,
        help="Report directory (default: <root>/report).",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        choices=sorted(PROVIDERS),
        help="Datasets to include (default: every one found).",
    )
    p.add_argument(
        "--budgets",
        type=float,
        nargs="*",
        default=None,
        metavar="PT",
        help="Forgetting budgets of the budget table, in points (default: 0.25 0.5 1).",
    )
    p.add_argument(
        "--level", type=float, default=0.95, help="Confidence level (default: 0.95)."
    )
    p.add_argument(
        "--setting",
        default=MAIN_SETTING,
        metavar="LABEL",
        help=(
            "Setting of the headline budget table, Pareto float and Pareto figures, "
            "matched as a substring of the setting label (default: "
            f"'{MAIN_SETTING}'). Pass an empty string to range over every setting; "
            "the cross-setting selection is always written to "
            "budget_selection_all.csv."
        ),
    )
    p.add_argument("--no-figures", action="store_true", help="Write the tables only.")
    p.set_defaults(func=cmd_report)

    # --- figures ---
    p = sub.add_parser(
        "isolated-train",
        help="Train one private model per cohort client; score own/union/old.",
    )
    _add_common(p)
    p.add_argument(
        "--clients-file", default=None,
        help="The cohort; default is everyone the cohort book covers.",
    )
    p.add_argument(
        "--old-book", required=True,
        help="Fold book of the old data, for the preservation evaluation.",
    )
    p.add_argument(
        "--old-fold", default="all", metavar="K|all",
        help=(
            "Which folds of the old-data book the preservation evaluation uses. "
            "'all' (the default) scores every fold separately and reports the "
            "five accuracies with their mean and standard deviation - the folds "
            "are five partitions of one population, so their spread is the error "
            "bar, and merging them would throw that away. An integer scores that "
            "one fold, kept for compatibility."
        ),
    )
    p.add_argument(
        "--only-client", default=None, metavar="ID",
        help=(
            "Train and score just this cohort writer, so the (writer, fold) "
            "cells can run as independent jobs. The union evaluation still "
            "spans every client of --clients-file: what one private model is "
            "worth to the whole group is meaningless against a group of one."
        ),
    )
    p.add_argument(
        "--pooled",
        action="store_true",
        help=(
            "CENTRALIZED ON THE OUTLIERS: train ONE model on the union of every "
            "client's fold-k train rows, early-stopped on their pooled fold-k "
            "validation rows, instead of one private model per client. The rung "
            "between isolation and federation - what a single model achieves "
            "once the privacy constraint is lifted and all twenty writers' data "
            "sits in one place, i.e. the ceiling the federated arm is trying to "
            "reach without moving any data. Writes "
            "centralized_outliers_<init>_fold<k>.json. Not combinable with "
            "--only-client."
        ),
    )
    p.add_argument("--old-clients-file", default=None)
    p.add_argument(
        "--init", default="global", choices=("global", "scratch"),
        help="Start every client from --init-checkpoint, or fresh.",
    )
    p.add_argument("--init-checkpoint", default=None, help="The shipped model.")
    p.add_argument("--epochs", type=int, default=100, help="Ceiling on local epochs.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument(
        "--early-stopping-patience", type=int, default=10, metavar="N",
        help="Per client: stop after N epochs without a validation improvement.",
    )
    p.add_argument("--min-epochs", type=int, default=20, metavar="M")
    p.add_argument(
        "--seed", type=int, default=None,
        help="Run seed: the initialisation and the training shuffle. The split "
             "is not drawn from it - that comes from the fold book.",
    )
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_isolated_train)

    p = sub.add_parser(
        "select-fold",
        help="Pick the best fold of a cross-validated training and persist it.",
    )
    _add_common(p)
    p.add_argument("--root", default=None, help="Where the per-fold folders live.")
    p.add_argument(
        "--prefix", required=True,
        help="Common prefix of the fold folders, e.g. 'ginit' for ginit_fold1.",
    )
    p.add_argument("--name", required=True, help="Name of the artefact to write.")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--out", default=None, help="Where to write it (default: --root).")
    p.add_argument(
        "--from-evaluations",
        default=None,
        metavar="PATH",
        help=(
            "Take the per-fold validation and test accuracies from an "
            "'evaluate-book' file instead of from what each training run "
            "reported. The recompute path: when the evaluation was wrong but "
            "the checkpoints were not, the folds are re-scored and re-selected "
            "with nothing retrained."
        ),
    )
    p.set_defaults(func=cmd_select_fold)

    p = sub.add_parser(
        "score-writers",
        help="Score every writer on its held-out partition of one fold.",
    )
    _add_common(p)
    p.add_argument("--model-path", required=True, help="Checkpoint to score with.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="Where the full record goes.")
    p.add_argument(
        "--accuracies-name", default="clients_acc_on_global.json",
        help="Flat {writer: accuracy} list written next to it, in the shape "
             "every existing reader of the accuracy record expects.",
    )
    p.set_defaults(func=cmd_score_writers)

    p = sub.add_parser(
        "draw-old-data",
        help="Draw the old-data population with a documented seed and rule.",
    )
    _add_common(p)
    p.add_argument("--size", type=int, default=100, help="Writers to draw.")
    p.add_argument("--seed", type=int, default=20260824, help="Seed of the draw.")
    p.add_argument(
        "--min-samples", type=int, default=100,
        help="Writers with fewer images than this are not eligible.",
    )
    p.add_argument(
        "--exclude-file", default=None,
        help="Client list whose writers the draw may not take - the cohort.",
    )
    p.add_argument(
        "--require-trainable", action="store_true",
        help="Draw only from writers that can form a local training split.",
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--from-pool",
        default=None,
        metavar="PATH",
        help=(
            "Draw from the clients named in this file - normally "
            "outliers/pool_good.json. THIS IS THE SAFE FORM: every other filter "
            "only narrows it, so the draw is a subset of the pool by "
            "construction. Without it the population is reconstructed from "
            "--min-samples and --exclude-file, which equals the good pool only "
            "when the pools were cut by the same eligibility rule - and when "
            "they were not, the draw silently takes clients that belong to "
            "neither pool."
        ),
    )
    p.set_defaults(func=cmd_draw_old_data)

    # --- draw-cohort ---
    p = sub.add_parser(
        "draw-cohort",
        help="Draw a cohort at random from a ranked population (T2/T3 studies).",
    )
    _add_common(p)
    p.add_argument("--k", type=int, required=True, help="Writers to draw.")
    p.add_argument("--seed", type=int, required=True, help="Seed of the draw.")
    p.add_argument(
        "--pool",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Draw from the N lowest-scoring eligible writers (T2, 'the 300 "
            "worst'). Absent draws from everyone eligible (T3), which may by "
            "chance include outliers - that is allowed, and is what makes T3 a "
            "study of an ordinary federation."
        ),
    )
    p.add_argument(
        "--scores",
        default=None,
        help="Writer accuracies; default: outliers/clients_acc_on_global.json.",
    )
    p.add_argument(
        "--require-trainable",
        action="store_true",
        help="Draw only from writers that can form a local training split.",
    )
    p.add_argument("--exclude-file", default=None, help="Writers to leave out.")
    p.add_argument(
        "--best",
        action="store_true",
        help=(
            "Draw from the TOP of the ranking instead of the bottom. The cohort "
            "always comes from the worst end; the reference population of the "
            "qualitative figure comes from the best, because 'a typical writer' "
            "means one the detector had no trouble with."
        ),
    )
    p.add_argument("--tag", default=None, help="Label stored with the cohort.")
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.set_defaults(func=cmd_draw_cohort)

    # --- writer-counts ---
    p = sub.add_parser(
        "writer-counts",
        help="Bank per-writer and per-class sample counts for the selected class set.",
    )
    _add_common(p)
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.add_argument("--csv", default=None, help="Destination CSV (default: next to --out).")
    p.set_defaults(func=cmd_writer_counts)

    # --- submit ---
    p = sub.add_parser(
        "submit",
        help="Submit a study chain from its manifest (preflight first, or nothing).",
    )
    p.add_argument("manifest", help="The chain manifest (TOML).")
    p.add_argument(
        "--go", action="store_true",
        help=(
            "Actually submit. Without it the invocations are printed and "
            "nothing is submitted."
        ),
    )
    p.set_defaults(func=cmd_submit)

    # --- speaker-table ---

    # --- select-eligible ---
    p = sub.add_parser(
        "select-eligible",
        help="Cut the eligible population from a writer-counts artefact.",
    )
    _add_common(p)
    p.add_argument(
        "--counts", default=None,
        help="The writer-counts JSON; default outliers/writer_counts.json.",
    )
    p.add_argument(
        "--min-samples", type=int, required=True, metavar="N",
        help=(
            "Rows a client needs to be eligible. The floor a per-client "
            "60/20/20 split has to survive; set it from the measured "
            "distribution, not from a guess."
        ),
    )
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.set_defaults(func=cmd_select_eligible)

    # --- check-population ---
    p = sub.add_parser(
        "check-population",
        help="Assert a client list's size, membership and fold-book coverage.",
    )
    _add_common(p)
    p.add_argument("--clients-file", required=True, help="The list under test.")
    p.add_argument("--expect-size", type=int, default=None, metavar="N")
    p.add_argument(
        "--subset-of", action="append", default=None, metavar="PATH",
        help="Every client must appear here. Repeatable.",
    )
    p.add_argument(
        "--disjoint-from", action="append", default=None, metavar="PATH",
        help="No client may appear here. Repeatable.",
    )
    # --fold-book comes from _add_common, which every subcommand shares.
    p.add_argument(
        "--folds", type=int, nargs="+", default=None, metavar="K",
        help="Folds that must split every client three ways (default: 1).",
    )
    p.add_argument("--counts", default=None, help="Writer counts, for the summary line.")
    p.add_argument("--label", default=None, help="Name this check reports under.")
    p.add_argument("--show", action="store_true", help="Print the client list.")
    p.set_defaults(func=cmd_check_population)

    # --- promote-model ---
    p = sub.add_parser(
        "promote-model",
        help="Copy a trained model to the name the runs resolve, with its checksum.",
    )
    _add_common(p)
    p.add_argument("--source", required=True, help="The trained checkpoint.")
    p.add_argument("--target", required=True, help="Where the runs look for it.")
    p.add_argument("--name", required=True, help="The global name, e.g. g0.")
    p.add_argument("--record", default=None, help="Destination for the selection record.")
    p.add_argument("--rule", default=None, help="How this model was chosen, for the record.")
    p.set_defaults(func=cmd_promote_model)

    # --- split-pools ---
    p = sub.add_parser(
        "split-pools",
        help="Cut the population into a bad pool and a good pool by the coarse detector.",
    )
    _add_common(p)
    p.add_argument(
        "--bad-fraction", type=float, default=0.30, metavar="F",
        help="Share of the ranking that is BAD; the cut is at ceil(F * n).",
    )
    p.add_argument(
        "--scores", default=None,
        help="The ranking to cut; default: outliers/clients_acc_on_global.json.",
    )
    p.add_argument(
        "--require-trainable", action="store_true",
        help="Restrict both pools to writers that can form a local training split.",
    )
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.add_argument("--csv", default=None, help="Destination CSV.")
    p.set_defaults(func=cmd_split_pools)

    # --- score-pool ---
    p = sub.add_parser(
        "score-pool",
        help="Score a model on every row of each writer in a pool.",
    )
    _add_common(p)
    p.add_argument("--model-path", required=True, help="The scoring model.")
    p.add_argument("--clients-file", required=True, help="The pool to score.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.add_argument("--csv", default=None, help="Destination CSV.")
    p.add_argument(
        "--accuracies-name", default=None, metavar="NAME",
        help="Also write the flat ranking under this name, next to --out.",
    )
    p.set_defaults(func=cmd_score_pool)

    # --- cohort-table ---
    p = sub.add_parser(
        "cohort-table",
        help="Bank the cohort with the accuracy, rank, rows and splits behind it.",
    )
    _add_common(p)
    p.add_argument("--clients-file", required=True, help="The cohort JSON.")
    p.add_argument(
        "--scores", default=None,
        help="Writer accuracies; default: outliers/clients_acc_on_global.json.",
    )
    p.add_argument(
        "--counts", default=None,
        help="Per-writer counts; default: outliers/writer_counts.json.",
    )
    p.add_argument(
        "--extra-scores",
        nargs="*",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Further rankings to carry as columns, e.g. "
            "ginit=outliers/clients_acc_on_global.json. A reader can then see "
            "whether the writers the shipped model finds hardest are the ones "
            "the coarse pass already flagged."
        ),
    )
    p.add_argument("--rule", default=None, help="How the cohort was cut, for the record.")
    p.add_argument("--out", default=None, help="Destination JSON.")
    p.add_argument("--csv", default=None, help="Destination CSV.")
    p.set_defaults(func=cmd_cohort_table)

    p = sub.add_parser(
        "evaluate-book",
        help="Score a saved model on one part of one fold of a fold book.",
    )
    _add_common(p)
    p.add_argument("--model-path", required=True)
    p.add_argument("--part", default="test", choices=("train", "val", "test"))
    p.add_argument(
        "--clients-file", default=None,
        help="Score on these writers; default is everyone the book covers.",
    )
    p.add_argument("--tag", default=None, help="Key this result is stored under.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate_book)

    p = sub.add_parser(
        "fold-book",
        help="Materialise a cross-validation split and write it down for reuse.",
    )
    _add_common(p)
    p.add_argument(
        "--out",
        required=True,
        help="Destination; '.foldbook.npz' is appended when it has no suffix.",
    )
    p.add_argument(
        "--clients-file",
        default=None,
        help=(
            "Cover the writers of this client list or pool file. Without it and "
            "without --writers the book covers every writer of the dataset."
        ),
    )
    p.add_argument(
        "--writers", nargs="+", default=None, metavar="ID", help="Cover exactly these writers."
    )
    p.add_argument("--folds", type=int, default=5, help="Folds to draw (default: 5).")
    p.add_argument("--seed", type=int, default=42, help="Base seed of the split.")
    p.add_argument("--train-rate", type=float, default=0.6)
    p.add_argument("--eval-rate", type=float, default=0.2)
    p.add_argument("--tag", default=None, help="Label stored with the book.")
    p.set_defaults(func=cmd_fold_book)

    p = sub.add_parser(
        "outlier-figure",
        help="Render the selected clients' glyphs against the source-population means.",
    )
    _add_common(p)
    p.add_argument(
        "--outliers-file", default=None, help="Client list or rule-based pool file."
    )
    p.add_argument(
        "--pool-frac",
        type=float,
        default=None,
        help="Draw the provider's rule-based pool of this bottom fraction instead.",
    )
    p.add_argument("--k", type=int, default=5, help="Selection size the provider resolves.")
    p.add_argument(
        "--clients",
        nargs="+",
        default=None,
        metavar="ID",
        help="Draw exactly these clients, whatever the selection holds.",
    )
    p.add_argument("--out", default=None, help="Destination PNG (a JSON sidecar joins it).")
    p.add_argument(
        "--n-examples", type=int, default=1, metavar="N", help="Rows drawn per client."
    )
    p.add_argument(
        "--max-columns",
        type=int,
        default=16,
        metavar="M",
        help=(
            "Upper bound on the columns; the classes the clients have most "
            "examples of are kept. 62 columns would be unreadable."
        ),
    )
    p.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        metavar="L",
        help="Draw exactly these class indices, in place of the automatic choice.",
    )
    p.add_argument(
        "--source-clients-file",
        default=None,
        help=(
            "JSON list of the REFERENCE writers the top row averages - what a "
            "typical writer looks like. Without it the legacy writer_split.json "
            "is used, which a study built on fold books never writes; naming "
            "the reference here also records it in the sidecar."
        ),
    )
    p.add_argument(
        "--source-clients",
        nargs="+",
        default=None,
        metavar="ID",
        help="Reference writers given directly, in place of --source-clients-file.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.set_defaults(func=cmd_outlier_figure)

    p = sub.add_parser("figures", help="Regenerate grid rankings and heatmaps.")
    _add_common(p)
    p.add_argument("--root", default=None, help="Results folder to scan.")
    p.set_defaults(func=cmd_figures)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_global_paths(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
