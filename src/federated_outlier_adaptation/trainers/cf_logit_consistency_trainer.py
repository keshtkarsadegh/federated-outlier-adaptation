"""Logit-consistency trainer anchoring client logits to the global model."""

import copy

import torch
from tqdm import tqdm

from federated_outlier_adaptation.constants import BEST_LOGIT_LAMBDA
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer


class CFLogitConsistencyTrainer(BaseTrainer):
    """
    Logit Consistency Regularization:
    Loss = CE(y, s(x)) + λ * || s(x) - s₀(x) ||²

    - s(x): current student logits
    - s₀(x): frozen snapshot logits from the initial model
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        lambda_consis=BEST_LOGIT_LAMBDA,
        global_model_path=None,
        provider=None,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.lambda_consis = lambda_consis
        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path
        # --- Snapshot of the initial student, frozen ---
        global_model = self.provider.make_teacher_model()
        global_model.load_state_dict(torch.load(global_model_path, map_location=self.device))

        self.initial_student = global_model.to(self.device)
        self.initial_student.eval()
        for p in self.initial_student.parameters():
            p.requires_grad = False

    def consistency_loss_(self, student_logits, images):
        """Compute L2 distance between current and initial student logits."""
        with torch.no_grad():
            init_logits = self.initial_student(images)
        return torch.nn.functional.mse_loss(student_logits, init_logits, reduction='mean')

    def custom_loss_fn(self, student_logits, labels, images):
        ce = self.criterion(student_logits, labels)
        consis = self.consistency_loss_(student_logits, images)
        total = ce + self.lambda_consis * consis
        return total, {'ce': ce.item(), 'consis': consis.item()}

    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            loss_parts_running = {'ce': 0.0, 'consis': 0.0}

            for images, labels in tqdm(train_loader, desc=f"[Consis] Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss, parts = self.custom_loss_fn(outputs, labels, images)
                if torch.isnan(loss):
                    raise ValueError("Loss became NaN")

                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                loss_parts_running['ce'] += parts['ce'] * batch_size
                loss_parts_running['consis'] += parts['consis'] * batch_size

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

            val_acc = correct / total if total > 0 else 0.0
            self.val_accuracies.append(val_acc)

            avg_loss = total_loss / max(total, 1)
            avg_parts = {k: v / max(total, 1) for k, v in loss_parts_running.items()}
            NistLogger.info(
                f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Loss={avg_loss:.4f} [ce={avg_parts['ce']:.4f}, consis={avg_parts['consis']:.4f}]"
            )

            if val_acc > best_val_acc:
                best_val_acc, bad, best_model = val_acc, 0, copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            else:
                bad += 1
                if bad >= patience:
                    NistLogger.info(f"Early stopping (patience={patience})")
                    break

        if best_model is not None:
            self.model = best_model
