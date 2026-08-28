"""
local_finetune.py - the no-federation reference point.

Every federated result needs two bracketing references: what a client reaches
on its own, and what the global model was worth before anything happened.  This
module produces the first one.  Each client of the pool fine-tunes a private
copy of the global model on its own training split; nothing is aggregated and
no client ever sees another client's update.

Reported per client and pooled over the pool:

    ``insample``  accuracy on the client's full data (the quantity the
                  federated runs report as the clients metric),
    ``val`` / ``test``  the two halves of the client's held-out 40% split,
    ``source``    the fine-tuned model's accuracy on the source test set, i.e.
                  how much of the global knowledge a purely local adaptation
                  destroys.

``init`` selects the starting point and therefore which rung of the reference
ladder the run is:

    ``global``  (default) each client starts from theta_g - rung **R2**, "local
                fine-tuning adapts but destroys what the model knew";
    ``scratch`` each client starts from a freshly initialised model and sees
                nothing but its own data - rung **R1**, "by its own data a
                client gets nothing".  The output folder gains a ``_scratch``
                component so the two rungs never overwrite each other.

``init_checkpoint`` replaces the starting model with an arbitrary checkpoint -
the final server model of a federated run - which turns the same command into
the **personalisation** arm: every client fine-tunes its own copy of what the
federation produced, and the same run from theta_g (R2) is the control that
isolates what the federated start was worth.

Output:
    ``<results>/local_finetune_<trainer>[_scratch][_seed<n>]/local_finetune.json``
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from federated_outlier_adaptation import config
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.outliers.selection import load_client_pool
from federated_outlier_adaptation.providers import default_provider
from federated_outlier_adaptation.runners.population import (
    SPLIT_SEED,
    _subset_loader,
    stratified_halves,
)
from federated_outlier_adaptation.trainers import artefacts
from federated_outlier_adaptation.trainers.registry import build_trainer
from federated_outlier_adaptation.training.convergence import convergence_kwargs
from federated_outlier_adaptation.utils import provenance
from federated_outlier_adaptation.utils.seeding import seed_suffix, set_run_seed

#: Starting points of a local reference run; mirrored by the runners.
INITIALISATIONS = ("global", "scratch")


def _pooled(entries):
    """Sample-count weighted mean of ``[(accuracy, count), ...]``."""
    total = sum(count for _, count in entries)
    if not total:
        return None
    return sum(accuracy * count for accuracy, count in entries) / total


def run_local_finetuning(
    trainer_name: str = "BaseTrainer",
    epochs: int = 10,
    batch_size: int = 64,
    seed: Optional[int] = None,
    outliers_file: Optional[str] = None,
    pool_frac: Optional[float] = None,
    k: int = 5,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    global_name: str = "global",
    parent_name: str = "local_finetune",
    provider=None,
    skip_existing: bool = False,
    init: str = "global",
    init_checkpoint: Optional[str] = None,
    patience: Optional[int] = None,
    min_epochs: int = 0,
) -> dict[str, Any]:
    """
    Fine-tune the global model separately on every pool client.

    Args:
        trainer_name: Trainer used for the local training.
        epochs: Local epochs per client.
        batch_size: Batch size of training and evaluation.
        seed: Optional run seed; adds a ``seed_<n>`` output component.
        outliers_file: Client pool; defaults to the provider's selection.
        pool_frac: Draw the clients from the provider's rule-based pool instead.
        k: Pool size when the provider resolves the selection itself.
        global_name: Global model to start from.
        parent_name: Output folder prefix.
        skip_existing: Return the stored result when it already exists.
        init: ``"global"`` (default) fine-tunes the global model, which is rung
            R2 of the reference ladder; ``"scratch"`` trains each client from a
            freshly initialised model, which is rung R1.
        patience: Non-improving epochs tolerated before a client's fine-tune
            stops.  Applied **per client**: each one converges on its own data,
            and a fixed epoch count for all of them is either too short for the
            larger writers or too long for the smaller.  ``None`` (the default)
            runs every epoch, as published.
        min_epochs: Epochs that always run before patience may fire.
        init_checkpoint: Start from this checkpoint instead of the named global
            model.  This is what makes the run a *personalisation* arm: point it
            at the final server model of a federated run and every client
            fine-tunes its own copy of what the federation produced, so the
            comparison against the same command from theta_g (R2) isolates what
            the federated start was worth.  Its path and SHA-256 are recorded.

    Returns:
        dict: Per-client and pooled metrics, with a provenance ``config`` block.
    """
    if init not in INITIALISATIONS:
        raise ValueError(f"Unknown initialisation {init!r}; expected one of {INITIALISATIONS}")

    provider = provider or default_provider()
    results_dir = Path(getattr(provider, "results_dir", config.RESULTS_DIR))

    out_dir = results_dir / f"{parent_name}_{trainer_name}"
    if init == "scratch":
        out_dir = results_dir / f"{parent_name}_{trainer_name}_scratch"
    suffix = seed_suffix(seed)
    if suffix:
        out_dir = out_dir / suffix
    out_file = out_dir / "local_finetune.json"
    if skip_existing and out_file.is_file():
        NistLogger.info(f"Skipping (exists): {out_file}")
        with open(out_file) as handle:
            return json.load(handle)

    set_run_seed(seed)
    timer = provenance.RunTimer()

    if outliers_file:
        clients, _ = load_client_pool(outliers_file)
    elif pool_frac is not None:
        clients = list(provider.outlier_pool(pool_frac))
    else:
        clients = list(provider.selected_clients(k=k))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global_model = provider.make_model()
    checkpoint_info: dict[str, Any] = {}
    if init == "global":
        if init_checkpoint:
            checkpoint = Path(init_checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"No starting checkpoint at {checkpoint}. A personalisation "
                    "run starts from a federated run's final server model, which "
                    "is only written when that run was given --save-final-model."
                )
            checkpoint_info = {
                "init_checkpoint": str(checkpoint),
                "init_checkpoint_sha256": provenance.sha256_of(checkpoint),
            }
            NistLogger.info(f"Personalising from {checkpoint}")
        else:
            checkpoint = Path(artefacts.global_model_path(provider, global_name))
        global_model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        # R1: the client has no shipped model, only its own data.  The freshly
        # initialised weights are drawn once, under the run seed set above, and
        # copied per client, so every client starts from the same untrained
        # model and the comparison to R2 differs in the starting point alone.
        NistLogger.info("Local reference from scratch (init=scratch): theta_g is not loaded.")
    global_model.to(device)

    trainer = build_trainer(
        trainer_name, trainer_kwargs, provider=provider, global_name=global_name
    )

    _, _, source_test_loader = provider.build_dataset(
        provider.global_client_ids(),
        train_rate=0.0,
        eval_rate=0.0,
        batch_size=batch_size,
        seed=SPLIT_SEED,
    )
    _, source_val_loader, _ = provider.build_dataset(
        provider.global_client_ids(),
        train_rate=0.6,
        eval_rate=0.4,
        batch_size=batch_size,
        seed=SPLIT_SEED,
    )

    per_client: dict[str, dict[str, Any]] = {}
    insample_pool, val_pool, test_pool, source_pool = [], [], [], []

    for client_id in clients:
        train_loader, eval_loader, _ = provider.build_dataset(
            client_id,
            train_rate=0.6,
            eval_rate=0.4,
            batch_size=batch_size,
            seed=SPLIT_SEED,
        )
        _, _, insample_loader = provider.build_dataset(
            client_id,
            train_rate=0.0,
            eval_rate=0.0,
            batch_size=batch_size,
            seed=SPLIT_SEED,
        )
        if train_loader is None or eval_loader is None:  # pragma: no cover
            continue

        val_index, test_index = stratified_halves(eval_loader.dataset, seed=SPLIT_SEED)
        val_loader = _subset_loader(eval_loader.dataset, val_index, batch_size)
        test_loader = _subset_loader(eval_loader.dataset, test_index, batch_size)

        trainer.set_model(copy.deepcopy(global_model))
        if patience is not None:
            # Per client: the stopping rule is about this client's own curve.
            trainer.early_stopping = True
        trainer.train(train_loader, eval_loader, epochs, patience, min_epochs)

        entry: dict[str, Any] = {
            "convergence": dict(getattr(trainer, "convergence", {}) or {}),
            "insample": float(trainer.evaluate(insample_loader)),
            "source_test": float(trainer.evaluate(source_test_loader)),
            "source_val": float(trainer.evaluate(source_val_loader)),
        }
        insample_pool.append((entry["insample"], len(insample_loader.dataset)))
        source_pool.append((entry["source_test"], 1))
        if val_loader is not None:
            entry["val"] = float(trainer.evaluate(val_loader))
            val_pool.append((entry["val"], len(val_loader.dataset)))
        if test_loader is not None:
            entry["test"] = float(trainer.evaluate(test_loader))
            test_pool.append((entry["test"], len(test_loader.dataset)))
        per_client[client_id] = entry
        NistLogger.info(f"Local fine-tuning {client_id}: {entry}")

    timer.stop()

    payload = {
        "trainer": trainer_name,
        "init": init,
        **checkpoint_info,
        "clients": clients,
        "epochs": epochs,
        "batch_size": batch_size,
        "per_client": per_client,
        "pool_insample_acc": _pooled(insample_pool),
        "pool_val_acc": _pooled(val_pool),
        "pool_test_acc": _pooled(test_pool),
        "mean_source_test_acc": _pooled(source_pool),
        "seed": seed,
        "config": provenance.build_run_config(
            trainer_name=trainer_name,
            trainer=trainer,
            trainer_kwargs=trainer_kwargs,
            scenario="local",
            metadata="none",
            agg_method_name="none",
            batch_size=batch_size,
            epochs=epochs,
            max_round=1,
            seed=seed,
            outliers_file=outliers_file or provider.outliers_file,
            selected_writers=clients,
            global_model_path=artefacts.global_model_path(provider, global_name),
            fisher_dir=artefacts.fisher_dir(provider, global_name),
            parent_name=parent_name,
            started_at=timer.started_at,
            finished_at=timer.finished_at,
            extra={
                "wall_seconds": timer.seconds,
                "global_name": global_name,
                "init": init,
                **convergence_kwargs(patience, min_epochs),
                **checkpoint_info,
            },
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as handle:
        json.dump(payload, handle, indent=2)
    NistLogger.info(f"Saved local fine-tuning reference to {out_file}")
    return payload


if __name__ == "__main__":  # pragma: no cover
    run_local_finetuning()
