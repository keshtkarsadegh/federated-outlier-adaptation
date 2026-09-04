"""
extreme_cases.py - final runs for the PRIOR pipeline's minimal-client scenarios.

This is the earlier published study's extreme stage, kept so its runs stay
reproducible.  The live study's extreme cases are ``double`` and ``dual`` only,
are cut from its own cohort ranking, and live in ``training/extreme_cells.py``.

Purpose:
    Runs the best-performing trainer (knowledge distillation) with the client
    set restricted to a single low-accuracy writer, to probe stability and
    fairness in minimal-client federations.

Cases (``constants.EXTREME_CASE_CLIENTS`` / ``EXTREME_CASE_HYPERPARAMETERS``):
    single  ["f3503_07"]                 T=50,  alpha=0.98
    double  ["f3503_07", "f2307_62"]     T=8,   alpha=0.95
    dual    ["f3503_07", "f3503_07"]     T=8,   alpha=0.95

Note:
    In the published runs the client restriction was an intersection with the
    selected-outlier list, so a repeated id contributed a single participant
    and ``double``/``dual`` differed from ``single`` only in hyperparameters
    and output folder.  A repeated id now yields one participant per
    occurrence, and ``double`` uses two distinct writers, so both cases write
    into new folders (``double_writers_...``, ``dual_replicated_...``) and the
    stored single-participant results stay untouched.

Outputs:
    ``<results>/<parent_name>_<trainer>_grid_search/[seed_<n>/]<scenario>_<metadata>/``
"""

import argparse
import json
from typing import Any, Mapping, Optional, Sequence

from federated_outlier_adaptation.constants import (
    EXTREME_CASE_CLIENTS,
    EXTREME_CASE_HYPERPARAMETERS,
    EXTREME_CASE_TAGS,
)
from federated_outlier_adaptation.training.final_experiments import (
    generate_final_result_all_parallel,
)

DEFAULT_TRAINER = "DistillationTrainer"


def extreme_case_parent(extreme_case: str) -> str:
    """Default output folder prefix of an extreme case."""
    return f"{EXTREME_CASE_TAGS[extreme_case]}_final_results_final"


def extreme_cases_final_training(
    single_outlier: Optional[Sequence[str]] = None,
    parent_name: str = "",
    extreme_case: Optional[str] = None,
    trainer_name: str = DEFAULT_TRAINER,
    outer_max_workers: int = 4,
    inner_max_workers: int = 4,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    participation: float = 1.0,
    policy: str = "all",
    sampler_seed: Optional[int] = None,
    track_clients: bool = False,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    global_name: str = "global",
    clients_per_round: Optional[int] = None,
    weighting: str = "proportional",
    server_eta: float = 1.0,
    server_kwargs: Optional[Mapping[str, Any]] = None,
    client_order: str = "fixed",
    seq_mix_alpha: Optional[float] = None,
    aggregation: Optional[str] = None,
    init: str = "global",
    provider_name: str = "nist",
    pool_frac: Optional[float] = None,
    rounds: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
):
    """
    Run the four standard jobs for one extreme case.

    Args:
        single_outlier: Client ids to restrict to.  Defaults to the client set
            of ``extreme_case``; a repeated id yields one participant per
            occurrence.
        parent_name: Output folder prefix.  Defaults to
            ``<tag>_final_results_final`` with the tag of ``extreme_case``.
        extreme_case: ``"single"``, ``"double"`` or ``"dual"``.
        trainer_kwargs: Trainer overrides.  Defaults to the published
            hyperparameters of ``extreme_case``.
        participation / policy / sampler_seed / track_clients: Client
            population options, see :mod:`..runners.client_sampler`.
        stop_when_global_below_clients: Stop once the global model falls below
            the clients accuracy or below 0.90.
        skip_existing: Skip jobs whose summary file already exists.
        rounds / epochs / batch_size: Optional budget overrides; ``None`` keeps
            the published 100 rounds / 100 local epochs / batch 64.
    """
    if extreme_case:
        if single_outlier is None:
            single_outlier = EXTREME_CASE_CLIENTS[extreme_case]
        if trainer_kwargs is None:
            trainer_kwargs = EXTREME_CASE_HYPERPARAMETERS[extreme_case]
        if not parent_name:
            parent_name = extreme_case_parent(extreme_case)

    all_results = generate_final_result_all_parallel(
        trainer_name=trainer_name,
        outer_max_workers=outer_max_workers,
        inner_max_workers=inner_max_workers,
        parent_name=parent_name,
        single_outlier=list(single_outlier) if single_outlier else None,
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        participation=participation,
        policy=policy,
        sampler_seed=sampler_seed,
        track_clients=track_clients,
        stop_when_global_below_clients=stop_when_global_below_clients,
        skip_existing=skip_existing,
        global_name=global_name,
        clients_per_round=clients_per_round,
        weighting=weighting,
        server_eta=server_eta,
        server_kwargs=server_kwargs,
        client_order=client_order,
        seq_mix_alpha=seq_mix_alpha,
        aggregation=aggregation,
        init=init,
        provider_name=provider_name,
        pool_frac=pool_frac,
        rounds=rounds,
        epochs=epochs,
        batch_size=batch_size,
        eval_path=eval_path,
        insample_every=insample_every,
    )

    print("All jobs done:", all_results)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run the extreme-case final trainings.")
    parser.add_argument("--case", choices=sorted(EXTREME_CASE_CLIENTS), default="single")
    parser.add_argument("--trainer", type=str, default=DEFAULT_TRAINER)
    parser.add_argument("--parent_name", type=str, default="")
    parser.add_argument("--outer_max_workers", type=int, default=4)
    parser.add_argument("--inner_max_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trainer-kwargs", type=str, default=None)
    args = parser.parse_args()

    extreme_cases_final_training(
        extreme_case=args.case,
        trainer_name=args.trainer,
        parent_name=args.parent_name,
        outer_max_workers=args.outer_max_workers,
        inner_max_workers=args.inner_max_workers,
        seed=args.seed,
        trainer_kwargs=json.loads(args.trainer_kwargs) if args.trainer_kwargs else None,
    )


if __name__ == "__main__":
    main()
