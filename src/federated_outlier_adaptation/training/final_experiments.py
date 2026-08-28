"""
final_experiments.py - final runs with the selected hyperparameters.

Purpose:
    Runs one trainer over all four (scenario, metadata) configurations with the
    hyperparameters chosen by the grid searches, producing the per-trainer
    figures and tables of the paper.

Usage:
    python -m federated_outlier_adaptation.training.final_experiments \
        --trainer EWCTrainer --parent_name final_result

Outputs:
    ``<results>/<parent_name>_<trainer>_grid_search/[seed_<n>/]<scenario>_<metadata>/``
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Optional

from federated_outlier_adaptation.training.driver import final_training

JOBS = [
    dict(scenario="sequential", metadata="weights"),
    dict(scenario="sequential", metadata="delta"),
    dict(scenario="concurrent", metadata="weights"),
    dict(scenario="concurrent", metadata="delta"),
]


def budget_kwargs(
    rounds: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> dict[str, int]:
    """
    Translate the optional budget overrides into driver keyword arguments.

    Only the values that were actually given are returned, so an unset override
    leaves the driver's published default (``max_round=100``, ``epochs=100``,
    ``batch_size=64``) untouched.
    """
    given: dict[str, int] = {}
    if rounds is not None:
        given["max_round"] = int(rounds)
    if epochs is not None:
        given["epochs"] = int(epochs)
    if batch_size is not None:
        given["batch_size"] = int(batch_size)
    return given


def generate(
    trainer_name: str,
    scenario: str,
    metadata: str,
    index: int,
    agg_method_name: Optional[str],
    parent_name: str,
    inner_max_workers: int = 12,
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    participation: float = 1.0,
    policy: str = "all",
    sampler_seed: Optional[int] = None,
    track_clients: bool = False,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    extended_aggregations: bool = False,
    global_name: str = "global",
    clients_per_round: Optional[int] = None,
    weighting: str = "proportional",
    server_eta: float = 1.0,
    server_kwargs: Optional[Mapping[str, Any]] = None,
    client_order: str = "fixed",
    seq_mix_alpha: Optional[float] = None,
    source_share: str = "off",
    source_share_cap: Optional[int] = None,
    save_final_model: bool = False,
    aggregation: Optional[str] = None,
    init: str = "global",
    provider_name: str = "nist",
    pool_frac: Optional[float] = None,
    rounds: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    old_folds: Any = "all",
    final_eval_batch_size: int = 256,
    source_share_multiplier: float = 1.0,
    val_blend_source: float = 0.0,
):
    """
    Run a single final-training job with the specified configuration.

    Args:
        trainer_name: Trainer class to instantiate (e.g. ``"EWCTrainer"``).
        scenario: ``"sequential"`` or ``"concurrent"``.
        metadata: ``"weights"`` or ``"delta"``.
        index: Index suffix of the output files.
        agg_method_name: Aggregation method, or ``"none"``/``None`` for all.
        parent_name: Parent folder for the results.
        inner_max_workers: Processes in the inner pool.
        single_outlier: Optional client restriction (extreme cases); a repeated
            id yields one participant per occurrence.
        seed: Optional run seed.
        trainer_kwargs: Optional trainer constructor overrides.
        outliers_file: Optional alternative selected-client list.
        participation, policy, sampler_seed, track_clients: Client population
            options; the defaults reproduce the published loop.
        stop_when_global_below_clients: Stop once the global model falls below
            the clients accuracy or below 0.90.
        skip_existing: Skip the job when its summary file already exists.
        rounds, epochs, batch_size: Optional budget overrides.  ``None`` leaves
            the published budget (100 rounds, 100 local epochs, batch 64) in
            place, so the defaults reproduce the published runs.
    """
    if agg_method_name == "none":
        agg_method_name = None
    return final_training(
        **budget_kwargs(rounds=rounds, epochs=epochs, batch_size=batch_size),
        trainer_name=trainer_name,
        scenario=scenario,
        metadata=metadata,
        index=index,
        agg_method_name=agg_method_name,
        parent_name=parent_name,
        inner_max_workers=inner_max_workers,
        single_outlier=single_outlier,
        seed=seed,
        trainer_kwargs=trainer_kwargs,
        outliers_file=outliers_file,
        participation=participation,
        policy=policy,
        sampler_seed=sampler_seed,
        track_clients=track_clients,
        stop_when_global_below_clients=stop_when_global_below_clients,
        skip_existing=skip_existing,
        extended_aggregations=extended_aggregations,
        global_name=global_name,
        clients_per_round=clients_per_round,
        weighting=weighting,
        server_eta=server_eta,
        server_kwargs=server_kwargs,
        client_order=client_order,
        seq_mix_alpha=seq_mix_alpha,
        source_share=source_share,
        source_share_cap=source_share_cap,
        save_final_model=save_final_model,
        init=init,
        provider_name=provider_name,
        pool_frac=pool_frac,
        eval_path=eval_path,
        insample_every=insample_every,
        old_book=old_book,
        old_clients_file=old_clients_file,
        old_folds=old_folds,
        final_eval_batch_size=final_eval_batch_size,
        source_share_multiplier=source_share_multiplier,
        val_blend_source=val_blend_source,
    )


def generate_final_result_all_parallel(
    trainer_name: str = "EWCTrainer",
    outer_max_workers: int = 4,
    inner_max_workers: int = 8,
    parent_name: str = "final_result",
    single_outlier=None,
    seed: Optional[int] = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    outliers_file: Optional[str] = None,
    participation: float = 1.0,
    policy: str = "all",
    sampler_seed: Optional[int] = None,
    track_clients: bool = False,
    stop_when_global_below_clients: bool = False,
    skip_existing: bool = False,
    extended_aggregations: bool = False,
    global_name: str = "global",
    clients_per_round: Optional[int] = None,
    weighting: str = "proportional",
    server_eta: float = 1.0,
    server_kwargs: Optional[Mapping[str, Any]] = None,
    client_order: str = "fixed",
    seq_mix_alpha: Optional[float] = None,
    source_share: str = "off",
    source_share_cap: Optional[int] = None,
    save_final_model: bool = False,
    aggregation: Optional[str] = None,
    init: str = "global",
    provider_name: str = "nist",
    pool_frac: Optional[float] = None,
    rounds: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_path: Optional[str] = None,
    insample_every: int = 1,
    old_book: Optional[str] = None,
    old_clients_file: Optional[str] = None,
    old_folds: Any = "all",
    final_eval_batch_size: int = 256,
    source_share_multiplier: float = 1.0,
    val_blend_source: float = 0.0,
):
    """
    Run the four standard jobs (sequential/concurrent x weights/delta).

    Args:
        aggregation: ``None``/``"all"`` runs every rule of each family, which
            is the published behaviour.  A variant key (``fedavg``,
            ``anchored``, ``capped``) picks one comparable rule per family; a
            concrete rule name restricts the run to the families that have it.
            Give such runs their own ``parent_name`` so they do not overwrite
            the full-family results.
        rounds, epochs, batch_size: Optional budget overrides; ``None`` keeps
            the published 100 rounds / 100 local epochs / batch 64.

    Returns:
        list: Summary dict of every completed job.
    """
    from federated_outlier_adaptation.aggregation.selector import (
        SKIP_FAMILY,
        resolve_aggregation,
    )

    jobs = []
    for job in JOBS:
        resolved = resolve_aggregation(
            job["scenario"], job["metadata"], aggregation, extended=extended_aggregations
        )
        if resolved == SKIP_FAMILY:
            # The requested rule does not exist in this family (the extended
            # parallel rules only live in concurrent/delta), so the job is
            # simply not part of this run.
            continue
        jobs.append((job, resolved if resolved is not None else "none"))

    if not jobs:
        # Every family skipped the requested rule, i.e. the name does not exist
        # anywhere.  Without this the run would silently do nothing at all.
        raise ValueError(
            f"Aggregation rule {aggregation!r} exists in none of the four "
            "families; pass a variant key (all, fedavg, anchored, capped) or a "
            "rule name listed by the aggregation containers."
        )

    results = []
    with ThreadPoolExecutor(max_workers=outer_max_workers) as outer:
        futures = [
            outer.submit(
                generate,
                trainer_name=trainer_name,
                index=0,
                agg_method_name=resolved,
                parent_name=parent_name,
                inner_max_workers=inner_max_workers,
                single_outlier=single_outlier,
                seed=seed,
                trainer_kwargs=trainer_kwargs,
                outliers_file=outliers_file,
                participation=participation,
                policy=policy,
                sampler_seed=sampler_seed,
                track_clients=track_clients,
                stop_when_global_below_clients=stop_when_global_below_clients,
                skip_existing=skip_existing,
                extended_aggregations=extended_aggregations,
                global_name=global_name,
                clients_per_round=clients_per_round,
                weighting=weighting,
                server_eta=server_eta,
                server_kwargs=server_kwargs,
                client_order=client_order,
                seq_mix_alpha=seq_mix_alpha,
                source_share=source_share,
                source_share_cap=source_share_cap,
                save_final_model=save_final_model,
                init=init,
                provider_name=provider_name,
                pool_frac=pool_frac,
                rounds=rounds,
                epochs=epochs,
                batch_size=batch_size,
                eval_path=eval_path,
                insample_every=insample_every,
                old_book=old_book,
                old_clients_file=old_clients_file,
                old_folds=old_folds,
                final_eval_batch_size=final_eval_batch_size,
                source_share_multiplier=source_share_multiplier,
                val_blend_source=val_blend_source,
                **job,
            )
            for job, resolved in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the final experiments for one trainer.")
    parser.add_argument("--trainer", type=str, required=True, help="Trainer class name.")
    parser.add_argument("--outer_max_workers", type=int, default=4)
    parser.add_argument("--inner_max_workers", type=int, default=20)
    parser.add_argument("--parent_name", type=str, default="final_result")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--trainer-kwargs",
        type=str,
        default=None,
        help='JSON object of trainer overrides, e.g. \'{"T": 4, "alpha": 0.95}\'.',
    )
    args = parser.parse_args()

    all_results = generate_final_result_all_parallel(
        trainer_name=args.trainer,
        outer_max_workers=args.outer_max_workers,
        inner_max_workers=args.inner_max_workers,
        parent_name=args.parent_name,
        seed=args.seed,
        trainer_kwargs=json.loads(args.trainer_kwargs) if args.trainer_kwargs else None,
    )

    print("All jobs done:", all_results)


if __name__ == "__main__":
    main()
