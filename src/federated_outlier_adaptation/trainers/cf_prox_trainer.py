"""FedProx-style trainer anchoring client weights to the global model."""

import copy

import torch
import torch.nn.functional as F

from federated_outlier_adaptation.constants import BEST_PROX_LAMBDA
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer


class CFProxTrainer(BaseTrainer):
    """
    Extension of BaseTrainer with FedProx-style regularization.

    Loss = CE(outputs, labels) + λ_prox * ||θ - θ_global||^2
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_prox=BEST_PROX_LAMBDA,
        global_model=None,
        global_model_path=None,
        provider=None,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.lambda_prox = lambda_prox
        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        # Save reference global state (anchor for prox)
        if global_model is not None:
            self._global_state = {
                k: v.detach().clone().to(self.device) for k, v in global_model.state_dict().items()
            }
        else:
            global_model = self.provider.make_teacher_model()
            global_model.load_state_dict(torch.load(global_model_path, map_location=self.device))
            global_model.to(self.device)
            global_model.eval()
            self._global_state = {
                k: v.detach().clone().to(self.device) for k, v in global_model.state_dict().items()
            }

    # ----------------------
    # Helpers
    # ----------------------
    def _prox_loss(self):
        """Compute proximal regularization against saved global weights."""
        if self._global_state is None:
            return 0.0
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self._global_state and param.requires_grad:
                ref = self._global_state[name]
                loss += F.mse_loss(param, ref, reduction="mean")
        return loss

    # ----------------------
    # Override train loop
    # ----------------------
    def train(self, train_loader, eval_loader, epochs=10):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model = None
        best_val_acc = -float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                ce_loss = self.criterion(outputs, labels)

                # Add proximal term
                prox = self._prox_loss() if self.lambda_prox > 0 else 0.0
                loss = ce_loss + self.lambda_prox * prox

                if torch.isnan(loss) or torch.isinf(loss):
                    NistLogger.error("Loss NaN/Inf, aborting.")
                    raise ValueError("Loss NaN/Inf")

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
            NistLogger.info(f"[Prox] Epoch {epoch + 1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")

        if best_model:
            self.model = best_model
