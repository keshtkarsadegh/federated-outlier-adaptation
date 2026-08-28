"""
base_trainer.py

Purpose:
    Provides a generic trainer class for federated adaptive learning experiments
    (used as a base for EWC, Distillation, Logit Consistency, Prox, etc.).

Capabilities:
    - Initialize and manage a CNN model for the configured dataset provider.
    - Train a client model with standard or custom loss functions.
    - Track per-epoch training and validation accuracy.
    - Select and retain the best validation model.
    - Evaluate trained models on test datasets.
    - Reload saved model weights.

Usage:
    trainer = BaseTrainer()
    trainer.train(train_loader, eval_loader, epochs=10)
    acc = trainer.evaluate(test_loader)
    model = trainer.get_model()
"""

import copy

import torch
from torch import nn, optim
from tqdm import tqdm

from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.providers import default_provider


class BaseTrainer:
    """
    Generic trainer for federated adaptive learning.

    Responsibilities:
        - Construct and manage the model supplied by the dataset provider.
        - Perform client-side training with Adam optimizer.
        - Track and log accuracy metrics.
        - Evaluate models on unseen test sets.
        - Provide hooks for specialized trainers (EWC, distillation, etc.)
          by supporting `custom_loss`.

    Attributes:
        device (str): Training device ('cuda' if available, else 'cpu').
        provider (DatasetProvider): Source of the model topology and artefacts.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        model (nn.Module): Active CNN model.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        criterion (nn.Module): Default loss function (CrossEntropy).
        custom_loss (callable, optional): Optional override for loss.
        test_acc (float): Last computed test accuracy.
        train_accuracies (list): History of training accuracies per epoch.
        val_accuracies (list): History of validation accuracies per epoch.
        early_stopping (bool): Whether a client update stops once the
            validation accuracy has not improved for ``patience`` epochs.
        patience (int): Number of non-improving epochs tolerated.

    Early stopping:
        The published baseline runs every local epoch and keeps the best
        validation model, which is what ``early_stopping=False`` (the default)
        does.  Every regularised trainer of this package instead stops as soon
        as the validation accuracy has not improved for ``patience`` epochs, so
        a comparison between them mixes two effects: the objective and the local
        budget.  ``early_stopping=True`` gives the baseline the *same* stopping
        rule - same best-validation model, same patience - so that an
        early-stopped FedAvg arm isolates the objective.  Nothing else changes;
        the flag is recorded in the provenance block like any other
        hyperparameter.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        provider=None,
        early_stopping: bool = False,
        patience: int = 5,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.provider = provider or default_provider()
        NistLogger.debug(f"device base trainer: {self.device}")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        #: Filled by :meth:`train`; empty until a run has happened.
        self.convergence: dict = {}

        self.model = self.provider.make_model()
        self.model.to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.custom_loss = None

        self.test_acc = None
        self.metrics = None
        self.train_accuracies = []
        self.val_accuracies = []

    def set_model(self, model):
        """
        Replace internal model with a copy of the provided one.

        Args:
            model (torch.nn.Module): Model to adopt and train.
        """
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def load_model(self, model_path):
        """
        Load model weights from a saved file.

        Args:
            model_path (str or Path): Path to model checkpoint.

        Returns:
            torch.nn.Module: The loaded model.
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return self.model

    def get_model(self):
        """
        Return the current managed model.

        Returns:
            torch.nn.Module: Active CNN model.
        """
        return self.model

    def train(self, train_loader, eval_loader, epochs=10, patience=None, min_epochs=0):
        """
        Train the model for a number of epochs.

        Args:
            train_loader (DataLoader): Training data loader.
            eval_loader (DataLoader): Validation data loader.
            epochs (int, optional): Number of epochs to train. Default=10.
            patience (int, optional): Non-improving epochs tolerated when
                ``early_stopping`` is on.  ``None`` uses the trainer's own
                ``patience``.
            min_epochs (int, optional): Epochs that always run, whatever the
                validation accuracy does.  0 (the default) keeps the published
                behaviour; a floor matters because validation accuracy on a
                small split is noisy early on and an opening plateau of two or
                three epochs means nothing.

        Process:
            - Runs training and validation loops.
            - Tracks accuracy history.
            - Selects best model by validation accuracy.
            - With ``early_stopping``, leaves the loop once the validation
              accuracy has not improved for ``patience`` epochs.
            - Restores best model at the end.
        """
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        from federated_outlier_adaptation.training.convergence import EarlyStopper

        limit = self.patience if patience is None else int(patience)
        stopper = EarlyStopper(
            patience=limit if self.early_stopping else None, min_epochs=min_epochs
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"[Training] Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = (
                    self.custom_loss(outputs, labels)
                    if self.custom_loss
                    else self.criterion(outputs, labels)
                )

                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss is NaN")

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            self.train_accuracies.append(train_acc)

            # === Validation ===
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            self.val_accuracies.append(val_acc)
            NistLogger.info(f"Epoch {epoch + 1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            improved = val_acc > stopper.best_accuracy
            should_stop = stopper.update(epoch, val_acc, self.model)
            if improved:
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            if should_stop:
                NistLogger.info(
                    f"Early stopping (patience={limit}, min_epochs={min_epochs})."
                )
                break

        best = stopper.best()
        if best is not None:
            self.model = best
        #: What the last :meth:`train` call did - which epoch it kept and why it
        #: stopped - for the metrics file and the provenance block.
        self.convergence = stopper.metrics()

    def evaluate(self, test_loader):
        """
        Evaluate current model on a test dataset.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            float: Test accuracy.
        """
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        self.test_acc = correct / total
        NistLogger.info(f"Test Accuracy: {self.test_acc:.4f}")
        return self.test_acc
