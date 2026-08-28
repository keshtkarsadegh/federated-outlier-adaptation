"""
Not-True Distillation (FedNTD).

Lee et al., "Preservation of the Global Knowledge by Not-True Distillation in
Federated Learning", NeurIPS 2022.  A client keeps the global model's opinion
about the classes the sample does *not* belong to, and is free to fit the true
class as it likes:

    Loss = CE(student(x), y) + beta * tau^2 * KL( p_teacher^not-true || p_student^not-true )

Both logit vectors have the true-class entry masked out before the softmax at
temperature ``tau``, so only the relative ordering of the not-true classes is
transferred.  With ``beta = 0`` the objective is plain cross-entropy.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import BEST_NTD_BETA, BEST_NTD_TAU
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer


def not_true_log_softmax(logits: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Log-softmax over the not-true classes at temperature ``tau``.

    The true-class logit is replaced by ``-inf`` so that it receives zero
    probability and the remaining classes are renormalised among themselves.

    Args:
        logits: ``[N, C]`` pre-softmax outputs.
        labels: ``[N]`` ground-truth class indices.
        tau: Softmax temperature.

    Returns:
        torch.Tensor: ``[N, C]`` log-probabilities, ``-inf`` at the true class.
    """
    masked = logits / tau
    masked = masked.scatter(1, labels.view(-1, 1), float("-inf"))
    return F.log_softmax(masked, dim=1)


class NTDTrainer(BaseTrainer):
    """
    Cross-entropy plus not-true distillation against the frozen global model.

    Attributes:
        beta (float): Weight of the not-true distillation term.
        tau (float): Softmax temperature of both distributions.
        teacher_model (nn.Module): Frozen global model.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=BEST_NTD_BETA,
        tau=BEST_NTD_TAU,
        global_model_path=None,
        provider=None,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.beta = beta
        self.tau = tau

        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        teacher = self.provider.make_teacher_model()
        teacher.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model = teacher.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

    def not_true_loss(self, student_logits, labels, images):
        """KL divergence between teacher and student over the not-true classes."""
        if self.beta <= 0:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            teacher_logits = self.teacher_model(images)
            teacher_log_p = not_true_log_softmax(teacher_logits, labels, self.tau)
        student_log_p = not_true_log_softmax(student_logits, labels, self.tau)
        # KL(teacher || student); the masked entry is -inf in both, and
        # exp(-inf) = 0 makes it drop out of the sum.
        kl = F.kl_div(
            student_log_p, teacher_log_p, reduction="none", log_target=True
        ).nan_to_num(0.0).sum(dim=1)
        return kl.mean() * (self.tau ** 2)

    def custom_loss_fn(self, student_logits, labels, images):
        ce = self.criterion(student_logits, labels)
        ntd = self.not_true_loss(student_logits, labels, images)
        total = ce + self.beta * ntd
        return total, {"ce": ce.item(), "ntd": float(ntd.item())}

    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            loss_parts_running = {"ce": 0.0, "ntd": 0.0}

            for images, labels in tqdm(train_loader, desc=f"[NTD] Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss, parts = self.custom_loss_fn(outputs, labels, images)
                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss became NaN")

                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                loss_parts_running["ce"] += parts["ce"] * batch_size
                loss_parts_running["ntd"] += parts["ntd"] * batch_size

                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / max(total, 1)
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
                f"Loss={avg_loss:.4f} [ce={avg_parts['ce']:.4f}, ntd={avg_parts['ntd']:.4f}]"
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
