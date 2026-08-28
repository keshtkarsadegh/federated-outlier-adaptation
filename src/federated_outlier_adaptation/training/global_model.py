import argparse
import copy
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn
from matplotlib import pyplot as plt

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.datasets import split_local_global_writers
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.outliers.client_accuracy import (
    accuracy_summary,
    select_outliers,
    selected_client_accuracies,
    write_pool,
    write_selected_clients,
)
from federated_outlier_adaptation.outliers.selection import get_low_acc_writers
from federated_outlier_adaptation.providers import default_provider, get_provider
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.training.fisher import compute_and_save_fisher_and_params
from federated_outlier_adaptation.training.outlier_models import outliers_train

"""
Global training pipeline for federated adaptive learning.

This module coordinates:
    - Global model training and evaluation
    - Client split generation (local vs global)
    - Outlier detection and training
    - Fisher information computation (for continual learning regularization)
    - Metrics logging and plotting

It serves as the entry point for preparing the global baseline and
supporting outlier-related experiments.

Everything is driven through a dataset provider.  With the default provider the
module reproduces the published NIST pipeline unchanged - same phases, same file
names, same results root.  ``run_provider_global_training`` is the additive
entry point used by the datasets that have no frozen artefacts to preserve
(LEAF Shakespeare, CIFAR-10); it writes into the provider's own results root.
"""

class GlobalTraining:
    """
    Global training manager for NIST federated adaptive learning experiments.

    This class coordinates the baseline training of a global model on NIST
    writers, manages experiment paths, and integrates additional steps such as:
        - Generating local/global writer splits
        - Logging and plotting metrics
        - Selecting low-performing (outlier) clients
        - Training with global + outlier clients
        - Computing Fisher information for continual learning

    Attributes:
        model (FlexibleCNN): The global CNN model.
        trainer (BaseTrainer): Training utility for running experiments.
        local_writers (list): IDs of local writers.
        global_writers (list): IDs of global writers.
        selected_writers (list): IDs of selected outlier writers.
        results_path (Path): Directory for experiment results.
        model_path (Path): Directory for saving checkpoints.
        fisher_path (Path): Directory for Fisher information files.
    """

    def __init__(self, provider=None, results_dir: Optional[Path] = None):
        self.metrics = None
        #: What the training loop did - the epoch it stopped at, the epoch it
        #: kept - filled by :meth:`train` when a convergence rule was given.
        self.convergence: dict = {}
        self.provider = provider or default_provider()
        self.results_dir = Path(results_dir) if results_dir else Path(
            getattr(self.provider, "results_dir", config.RESULTS_DIR)
        )
        self.model = self.provider.make_model()
        self.local_writers = None
        self.global_writers = None
        self.selected_writers = None
        self.trainer = BaseTrainer(provider=self.provider)
        self.name = "global"
        self.results_path = self.results_dir / f"{self.name}_results"
        self.model_path = self.results_dir / f"{self.name}_model"
        self.writer_split_path = self.results_dir / "writer_split.json"
        self.clients_acc_on_global_path = self.results_dir / "outliers" / "clients_acc_on_global.json"
        self.selected_outliers_path = self.results_dir / "outliers" / "selected_outliers.json"
        self.outliers_result_path = self.results_dir / "outliers" / "results"
        self.fisher_path = self.results_path / "fisher"
        self.path_init()




    def path_init(self):
        """
        Create required directories for results, models, and outliers.

        Ensures that the following exist:
            - results/global_results/
            - results/global_model/
            - results/outliers/
            - results/outliers/results/
        """

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "outliers").mkdir(parents=True, exist_ok=True)
        self.outliers_result_path.mkdir(parents=True, exist_ok=True)

    def train(self,trn_loader,evl_loader,epochs,load_model=False,patience=None,min_epochs=0):
        """
        Train or load the global model.

        If `load_model=True`, loads the model from disk. Otherwise trains from scratch
        using the provided train and validation loaders, then saves the model state.

        Args:
            trn_loader (DataLoader): Training dataset loader.
            evl_loader (DataLoader): Validation dataset loader.
            epochs (int): Number of epochs to train.
            load_model (bool): If True, load existing model instead of training.
            patience (int, optional): Non-improving epochs tolerated before the
                training stops.  ``None`` (the default) runs every epoch, which
                is the published behaviour.
            min_epochs (int): Epochs that always run before patience may fire.

        With a patience the weights of the best validation epoch are restored
        before the model is saved, so the checkpoint on disk and the accuracy
        reported for it are the same model.
        """

        if load_model:
            NistLogger.info(f"Loading model from {self.model_path}")
            self.model =self.trainer.load_model(self.model_path)
        else:

            self.trainer.set_model(self.model)
            if patience is None:
                self.trainer.train(
                    train_loader=trn_loader, eval_loader=evl_loader, epochs=epochs
                )
            else:
                _, self.convergence = _train_with_early_stopping(
                    self.trainer, trn_loader, evl_loader, epochs, patience, min_epochs
                )
            self.model=self.trainer.get_model()
            torch.save(self.model.state_dict(), self.model_path)
    def evaluate(self,tst_loader):
        """
        Evaluate the global model on a test set.

        Loads the saved state_dict and evaluates using BaseTrainer.

        Args:
            tst_loader (DataLoader): Test dataset loader.
        """

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.trainer.device))
        self.trainer.set_model(self.model)
        self.trainer.evaluate(tst_loader)

    def generate_split_writers(self, global_size=0.03, seed=42, force_generate=False):
        """
        Load the local/global writer split, generating it only when missing.

        ``results/writer_split.json`` is a frozen artefact of the published
        experiments.  It is reused whenever it exists so that every downstream
        phase keeps operating on exactly the same clients; it is regenerated
        only when the file is absent (or ``force_generate=True``), in which case
        a warning is emitted because a freshly drawn split cannot be expected to
        match the published one bit for bit.

        Args:
            global_size (float): Proportion of writers to assign to the global set.
            seed (int): Random seed of the pre-shuffle.
            force_generate (bool): Regenerate even if the split file exists.

        Returns:
            tuple: (local_writers, global_writers)
        """
        if self.writer_split_path.exists() and not force_generate:
            with open(self.writer_split_path, "r") as f:
                split = json.load(f)
            self.local_writers = split["local_writers"]
            self.global_writers = split["global_writers"]
            NistLogger.info(
                f"Loaded frozen writer split from {self.writer_split_path} "
                f"({len(self.local_writers)} local / {len(self.global_writers)} global)."
            )
            return self.local_writers, self.global_writers

        make_split = getattr(self.provider, "make_client_split", None)
        if make_split is not None:
            # Datasets that derive their split from the provider (LEAF
            # Shakespeare, CIFAR-10).  Deterministic for a given seed.
            local_writers, global_writers = make_split(seed=seed)
            NistLogger.info(
                f"Derived the {self.provider.name} client split "
                f"({len(local_writers)} local / {len(global_writers)} global)."
            )
        else:
            NistLogger.warning(
                "Regenerating the writer split; the published split cannot be "
                "reproduced bit for bit from the manifest order, so downstream "
                "numbers will differ from the paper."
            )
            local_writers, global_writers = split_local_global_writers(
                global_size=global_size,
                seed=seed,
                labels_json=self.provider.dataset.labels_json,
            )
        self.writer_split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.writer_split_path, "w") as f:
            json.dump({"local_writers": local_writers, "global_writers": global_writers}, f, indent=2)
        NistLogger.info(f"Local ({len(local_writers)}) and global ({len(global_writers)}) writers are stored.")
        self.global_writers = global_writers
        self.local_writers = local_writers
        adopt = getattr(self.provider, "set_client_split", None)
        if adopt is not None:
            # Keep a provider that had already cached the previous split in step.
            adopt(local_writers, global_writers)
        return local_writers, global_writers
    def save_metrics_plots(self):
        """
        Save training/validation/test metrics and accuracy plots.

        - Saves metrics as JSON
        - Saves PNG plot of train/validation accuracy curves and test accuracy line
        """

        # === Save Metrics as JSON ===
        self.metrics = {
            "train_accuracies": self.trainer.train_accuracies,
            "val_accuracies": self.trainer.val_accuracies,
            "test_accuracy": self.trainer.test_acc,
            # How long it trained and why it stopped; empty when the run used a
            # fixed epoch count, which is what every published run did.
            "convergence": dict(self.convergence),
        }
        with open(self.results_path/f"{self.name}_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        NistLogger.info(f"Saved metrics JSON to {self.results_path/f"{self.name}_metrics.json"}")

        # === Save Accuracy Plot ===
        plt.figure()
        plt.plot(self.trainer.train_accuracies, label="Train Accuracy")
        plt.plot(self.trainer.val_accuracies, label="Validation Accuracy")
        plt.axhline(y=self.trainer.test_acc, color='r', linestyle='--', label=f"Test Accuracy on Global Data : {self.trainer.test_acc:.4f}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_path/f"{self.name}_plots.png")
        NistLogger.info(f"Saved accuracy plot to {self.results_path/f"{self.name}_plots.png"}")
    def generate_selected_writers(self, seed=42, k=5, bottom_frac=0.005, force_generate=False):
        """
        Select underperforming (outlier) writers.

        Calls `get_low_acc_writers` using client accuracy statistics.

        Args:
            seed (int): Random seed.
            k (int): Number of outliers to sample.  ``k == 5`` writes the
                published ``selected_outliers.json``; any other ``k`` writes
                ``selected_outliers_k<k>.json`` so the frozen list is preserved.
            bottom_frac (float): Fraction of the worst clients to sample from.
            force_generate (bool): Re-select even if the list already exists.

        Returns:
            list: Selected outlier writer IDs.
        """
        target = (
            self.selected_outliers_path
            if k == 5
            else self.selected_outliers_path.with_name(f"selected_outliers_k{k}.json")
        )
        if target.exists() and not force_generate:
            with open(target, "r") as f:
                self.selected_writers = json.load(f)
            NistLogger.info(f"Loaded frozen outlier selection from {target}")
            return self.selected_writers

        self.selected_writers = get_low_acc_writers(
            self.clients_acc_on_global_path,
            target,
            self.outliers_result_path,
            seed=seed,
            k=k,
            bottom_frac=bottom_frac,
        )
        return self.selected_writers
    def get_outliers(self, force_generate=False, batch_size=64):
        """
        Compute client accuracies to identify potential outliers.

        Args:
            force_generate (bool): If True, regenerate client accuracy stats even if file exists.
            batch_size (int): Evaluation batch size; 64 is the published value.
        """


        select_outliers(
            self.trainer.model,
            self.local_writers,
            self.clients_acc_on_global_path,
            force_generate=force_generate,
            provider=self.provider,
            batch_size=batch_size,
        )

    def outliers_training(self,batch_size,epochs):
        """
        Train one personal model per selected outlier, the published Fig-4 baseline.

        Delegates to `outliers_train`.  A writer whose split holds too little
        data is skipped there with a warning rather than raising, so this tail
        step cannot destroy an already-finished global training.

        Args:
            batch_size (int): Batch size.
            epochs (int): Number of epochs.

        Returns:
            dict: The per-writer metrics `outliers_train` produced.
        """

        return outliers_train(self.global_writers, self.selected_writers, batch_size=batch_size, num_epochs=epochs, outliers_results_path=self.outliers_result_path, provider=self.provider)

    def generate_fisher(self, trn_loader, max_batches=None):
        """
        Compute Fisher information and save parameter snapshots.

        Runs `compute_and_save_fisher_and_params` using the provided training data.

        Args:
            trn_loader (DataLoader): Training dataset loader.
            max_batches (int, optional): Bound on the number of batches; ``None``
                (the default) traverses the whole loader, as every published run
                did.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        compute_and_save_fisher_and_params(
    model=self.model,
    dataloader=trn_loader,  # use global data
    criterion=nn.CrossEntropyLoss(),
    device=device,
    fisher_path=self.fisher_path,
    max_batches=max_batches,
)

def run_global_training(
    epochs: int = 100,
    batch_size: int = 64,
    seed: int = 42,
    load_model: bool = False,
    k: int = 5,
    provider=None,
    skip_outlier_training: bool = False,
    patience: Optional[int] = None,
    min_epochs: int = 0,
    population: str = "source",
    writers_file: Optional[str] = None,
):
    """
    Phase 1 of the pipeline: train the global model and derive its artefacts.

    Steps: writer split -> global model -> Fisher information -> evaluation and
    metrics -> per-client accuracies -> outlier selection -> per-outlier
    reference models.

    Args:
        patience: Non-improving epochs tolerated before the global training
            stops; ``None`` (the default) runs every epoch, as published.  With
            a patience the best validation epoch's weights are what gets saved.
        min_epochs: Epochs that always run before patience may fire.
        population: Which writers the model is trained on.  ``"source"`` (the
            default) is the published pipeline: the held-out writers stay held
            out and are scored afterwards.  ``"all"`` trains on **every** writer
            of the dataset - the centralised ceiling, and the starting point
            (g-init) of the v6 protocol - and skips the outlier phases, because
            with no held-out population there is nothing to score yet.
        skip_outlier_training: Stop after the outlier selection and do not train
            the per-outlier reference models.  ``False`` by default, so the
            published pipeline is unchanged.  The step is the old Fig-4
            baseline; where the reference ladder provides that rung instead
            (``foa local-finetune``, R1/R2), it is redundant here and costs a
            further ``epochs`` epochs per selected writer in every run.
    """
    if population not in ("source", "all", "file"):
        raise ValueError(
            f"Unknown population {population!r}; expected 'source', 'all' or 'file'"
        )
    if population == "file" and not writers_file:
        raise ValueError("population='file' needs --writers-file")

    gt = GlobalTraining(provider=provider)
    gt.generate_split_writers(seed=seed)
    dataset = gt.provider.dataset

    if population in ("all", "file"):
        # g-init trains on every writer - the centralised ceiling.  g-0 trains
        # on exactly the drawn old-data writers.  Either way the 60/20/20 of
        # each writer comes from the fold book when one is configured, so the
        # folds of this stage are the folds every later stage reuses.
        if population == "all":
            writers = gt.provider.all_client_ids()
            NistLogger.info(f"Training on the whole dataset: {len(writers)} writers.")
        else:
            from federated_outlier_adaptation.outliers.selection import load_client_pool

            writers, _ = load_client_pool(writers_file)
            NistLogger.info(
                f"Training on {len(writers)} writers from {writers_file}."
            )
            if not writers:
                raise ValueError(f"{writers_file} names no writers.")
        train_loader, eval_loader, test_loader = dataset.build_dataset(
            writers, train_rate=0.6, eval_rate=0.2, batch_size=batch_size, seed=seed
        )
    else:
        writers = gt.global_writers
        # 60/20/20, from ONE call, so the test set is genuinely held out.
        #
        # The published pipeline built its test loader with rates 0.0/0.0, which
        # in a stratified split means "put everything in test" - so the reported
        # test accuracy was measured on the writers' *whole* data, the 60 % the
        # model had just trained on included.  That is what made v5's theta_g
        # report a test accuracy above its own best validation accuracy.  A test
        # number has to be measured on rows the model did not train on.
        train_loader, eval_loader, test_loader = dataset.build_dataset(
            writers, train_rate=0.6, eval_rate=0.2, batch_size=batch_size, seed=seed
        )
    gt.train(
        trn_loader=train_loader,
        evl_loader=eval_loader,
        epochs=epochs,
        load_model=load_model,
        patience=patience,
        min_epochs=min_epochs,
    )
    gt.generate_fisher(train_loader)
    gt.evaluate(tst_loader=test_loader)
    gt.save_metrics_plots()
    if population in ("all", "file"):
        # Nothing is held out here, so there is no population to score and no
        # outliers to select; discovery is its own stage, run against this
        # model once it exists.
        NistLogger.info(
            f"population={population}: skipping the outlier phases - discovery "
            "runs separately, against this model."
        )
        return gt
    gt.get_outliers(force_generate=False)
    gt.generate_selected_writers(seed=seed, k=k)
    if skip_outlier_training:
        NistLogger.info(
            "Skipping the per-outlier reference models (--skip-outlier-training)."
        )
    else:
        gt.outliers_training(batch_size=batch_size, epochs=epochs)
    return gt


def _train_with_early_stopping(
    trainer, train_loader, eval_loader, epochs, patience, min_epochs=0
):
    """
    Train ``trainer`` and stop once validation accuracy stalls.

    Mirrors ``BaseTrainer.train`` - per-epoch train/validation accuracy, the best
    validation checkpoint is restored at the end - and adds the patience rule
    the regularised trainers already use.  ``patience=None`` runs all epochs, so
    the caller can reproduce the plain loop.

    Args:
        min_epochs: Epochs that always run before the patience rule may end the
            training.  See :mod:`federated_outlier_adaptation.training.convergence`.

    Returns:
        tuple[float, dict]: The best validation accuracy, and what the run did -
        the epoch it stopped at, the epoch it kept and whether patience fired.
    """
    from federated_outlier_adaptation.training.convergence import EarlyStopper

    trainer.train_accuracies.clear()
    trainer.val_accuracies.clear()

    stopper = EarlyStopper(patience=patience, min_epochs=min_epochs)
    criterion = trainer.criterion

    for epoch in range(epochs):
        trainer.model.train()
        correct, total = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
            trainer.optimizer.zero_grad()
            outputs = trainer.model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                NistLogger.error("Loss is NaN, aborting.")
                raise ValueError("Loss is NaN")
            loss.backward()
            trainer.optimizer.step()

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = correct / max(total, 1)
        trainer.train_accuracies.append(train_acc)

        trainer.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
                outputs = trainer.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)
        trainer.val_accuracies.append(val_acc)
        NistLogger.info(
            f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if stopper.update(epoch, val_acc, trainer.model):
            NistLogger.info(
                f"Early stopping (patience={patience}, min_epochs={min_epochs})."
            )
            break

    best = stopper.best()
    if best is not None:
        trainer.model = best
    trainer.convergence = stopper.metrics()
    return stopper.best_accuracy, stopper.metrics()


def run_provider_global_training(
    provider,
    epochs: int = 100,
    batch_size: int = 64,
    eval_batch_size: Optional[int] = None,
    seed: int = 42,
    patience: Optional[int] = 5,
    min_epochs: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    k_values: Sequence[int] = (5,),
    pool_fracs: Sequence[float] = (),
    fisher_max_batches: Optional[int] = None,
    force_client_accuracies: bool = False,
    force_split: bool = False,
    load_model: bool = False,
):
    """
    Global baseline for a dataset that has no frozen artefacts.

    Runs the same phases as :func:`run_global_training` - client split, pooled
    global model, Fisher information, evaluation, per-client accuracies,
    selection of the lowest-accuracy held-out clients - but writes into the
    provider's own results root and ranks the selection deterministically
    instead of sampling it.  The NIST pipeline is untouched.

    Args:
        provider: Dataset provider (``shakespeare``, ``cifar10``, ...).
        epochs: Upper bound on the number of pooled training epochs.
        batch_size: Training batch size.
        eval_batch_size: Batch size of the evaluation passes; defaults to four
            times ``batch_size``, which only affects throughput.
        seed: Split seed.
        patience: Early-stopping patience in epochs; ``None`` trains all epochs.
        learning_rate, weight_decay: Adam settings of the pooled training.
        k_values: Sizes of the selected-client lists to write.
        pool_fracs: Bottom fractions for which a rule-based outlier pool file is
            written.  Empty (the default) writes none.
        fisher_max_batches: Optional bound on the Fisher pass.
        force_client_accuracies: Recompute ``clients_acc_on_global.json`` even
            when it already exists.
        force_split: Redraw the client split even when ``writer_split.json``
            exists; needed after the client partition itself changed.
        load_model: Reuse the stored global model instead of training it again,
            so that a rerun can pick up after the training phase.

    Returns:
        A summary dict with the dataset statistics, the global accuracies and
        the selected clients.
    """
    eval_batch_size = eval_batch_size or batch_size * 4

    gt = GlobalTraining(provider=provider)
    gt.trainer = BaseTrainer(
        learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
    )
    local_clients, global_clients = gt.generate_split_writers(
        seed=seed, force_generate=force_split
    )

    NistLogger.info(
        f"[{provider.name}] {len(global_clients)} global / {len(local_clients)} local clients"
    )

    train_loader, eval_loader, _ = provider.build_training_dataset(
        global_clients, train_rate=0.6, eval_rate=0.4, batch_size=batch_size, seed=seed
    )
    fisher_loader, _, _ = provider.build_dataset(
        global_clients, train_rate=0.6, eval_rate=0.4, batch_size=batch_size, seed=seed
    )
    _, _, test_loader = provider.build_dataset(
        global_clients, train_rate=0.0, eval_rate=0.0, batch_size=eval_batch_size, seed=seed
    )

    if load_model and gt.model_path.is_file():
        NistLogger.info(f"[{provider.name}] loading global model from {gt.model_path}")
        gt.model = gt.trainer.load_model(gt.model_path)
    else:
        gt.trainer.set_model(gt.model)
        _train_with_early_stopping(
            gt.trainer, train_loader, eval_loader, epochs, patience, min_epochs
        )
        gt.model = gt.trainer.get_model()
        gt.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gt.model.state_dict(), gt.model_path)
        NistLogger.info(f"[{provider.name}] global model -> {gt.model_path}")

    gt.generate_fisher(fisher_loader, max_batches=fisher_max_batches)
    gt.evaluate(tst_loader=test_loader)
    gt.save_metrics_plots()

    gt.get_outliers(force_generate=force_client_accuracies, batch_size=eval_batch_size)

    eligible = None
    if hasattr(provider, "eligible_clients"):
        eligible = provider.eligible_clients()

    selections = {}
    for k in k_values:
        _, clients = write_selected_clients(
            gt.clients_acc_on_global_path, gt.results_dir / "outliers", k=int(k), eligible=eligible
        )
        selections[int(k)] = clients
    gt.selected_writers = selections.get(5, next(iter(selections.values()), []))

    pools = {}
    for frac in pool_fracs:
        path, payload = write_pool(
            gt.clients_acc_on_global_path,
            gt.results_dir / "outliers",
            frac=float(frac),
            eligible=eligible,
        )
        pools[f"{float(frac):g}"] = {
            "path": str(path),
            "size": payload["size"],
            "eligible_clients": payload["eligible_clients"],
            "min_accuracy": payload["min_accuracy"],
            "max_accuracy": payload["max_accuracy"],
            "mean_accuracy": payload["mean_accuracy"],
        }

    summary = {
        "provider": provider.name,
        "results_dir": str(gt.results_dir),
        "num_clients": len(provider.all_client_ids()),
        "num_global_clients": len(global_clients),
        "num_local_clients": len(local_clients),
        "global_pool_samples": sum(provider.sample_count(c) for c in global_clients),
        "epochs_run": len(gt.trainer.train_accuracies),
        "train_accuracy": gt.trainer.train_accuracies[-1] if gt.trainer.train_accuracies else None,
        "best_val_accuracy": max(gt.trainer.val_accuracies) if gt.trainer.val_accuracies else None,
        "global_test_accuracy": gt.trainer.test_acc,
        "held_out_accuracy": accuracy_summary(gt.clients_acc_on_global_path, eligible=eligible),
        "selected_clients": {str(k): v for k, v in selections.items()},
        "selected_client_accuracies": {
            str(k): selected_client_accuracies(gt.clients_acc_on_global_path, v)
            for k, v in selections.items()
        },
        "outlier_pools": pools,
    }
    with open(gt.results_path / "provider_global_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main(argv: Optional[Sequence[str]] = None):
    """
    Command-line entry point.

    Without arguments it runs the published NIST phase 1, exactly as before.
    """
    parser = argparse.ArgumentParser(description="Train the pooled global model of a dataset.")
    parser.add_argument("--provider", default="nist", help="nist | shakespeare | cifar10")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k", type=int, nargs="+", default=[5])
    parser.add_argument(
        "--pool-frac",
        type=float,
        nargs="*",
        default=[],
        help="Bottom fractions for which to write outlier_pool_frac<f>.json.",
    )
    parser.add_argument("--fisher-max-batches", type=int, default=None)
    parser.add_argument("--force-client-accuracies", action="store_true")
    parser.add_argument(
        "--force-split",
        action="store_true",
        help="Redraw the client split even when writer_split.json exists.",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Reuse the stored global model instead of training it again.",
    )
    args = parser.parse_args(argv)

    if args.provider.lower() == "nist":
        run_global_training(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
        return 0

    kwargs = {}
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    provider = get_provider(args.provider, **kwargs)

    summary = run_provider_global_training(
        provider,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        patience=args.patience,
        learning_rate=args.learning_rate,
        k_values=args.k,
        pool_fracs=args.pool_frac,
        fisher_max_batches=args.fisher_max_batches,
        force_client_accuracies=args.force_client_accuracies,
        force_split=args.force_split,
        load_model=args.load_model,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



