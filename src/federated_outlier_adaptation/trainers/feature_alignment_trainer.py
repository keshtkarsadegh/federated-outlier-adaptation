"""
Representation-level feature alignment against the frozen global model.

Unlike :class:`CFAlignedFeatureTrainer`, which matches pre-softmax *logits*,
this trainer matches the penultimate *representation* of the student and the
teacher::

    Loss = CE(student(x), y) + beta * MSE(h_s(x), h_t(x))

where ``h(x)`` is the post-``fc1`` ReLU activation exposed by every dataset
model through ``penultimate(x)`` (or ``forward(x, return_features=True)``).
"""

from __future__ import annotations

import copy
import inspect
from functools import lru_cache

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import BEST_FEATURE_ALIGN_BETA
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer


@lru_cache(maxsize=None)
def _accepts_return_features(model_cls) -> bool:
    """Whether a model class implements ``forward(x, return_features=...)``."""
    try:
        return "return_features" in inspect.signature(model_cls.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


def logits_and_features(model, images):
    """
    Return ``(logits, penultimate_features)`` for any provider model.

    Models that implement ``forward(x, return_features=True)`` are asked for
    both in one pass; models that only implement ``penultimate(x)`` are
    evaluated twice.
    """
    if _accepts_return_features(type(model)):
        return model(images, return_features=True)
    return model(images), model.penultimate(images)


class FeatureAlignmentTrainer(BaseTrainer):
    """
    Cross-entropy with an MSE penalty on the penultimate representation.

    Attributes:
        beta (float): Weight of the alignment term.
        teacher_model (nn.Module): Frozen global model providing ``h_t(x)``.
    """

    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta=BEST_FEATURE_ALIGN_BETA,
        global_model_path=None,
        provider=None,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.beta = beta

        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        teacher = self.provider.make_teacher_model()
        teacher.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model = teacher.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

    def alignment_loss_(self, student_features, images):
        """MSE between the student and the frozen teacher representation."""
        if self.beta <= 0:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            _, teacher_features = logits_and_features(self.teacher_model, images)
        return F.mse_loss(student_features, teacher_features.detach(), reduction="mean")

    def custom_loss_fn(self, student_logits, student_features, labels, images):
        ce = self.criterion(student_logits, labels)
        align = self.alignment_loss_(student_features, images)
        total = ce + self.beta * align
        return total, {"ce": ce.item(), "align": float(align.item())}

    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            loss_parts_running = {"ce": 0.0, "align": 0.0}

            for images, labels in tqdm(
                train_loader, desc=f"[FeatAlign] Epoch {epoch+1}/{epochs}"
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs, features = logits_and_features(self.model, images)

                loss, parts = self.custom_loss_fn(outputs, features, labels, images)
                if torch.isnan(loss):
                    raise ValueError("Loss became NaN")

                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                loss_parts_running["ce"] += parts["ce"] * batch_size
                loss_parts_running["align"] += parts["align"] * batch_size

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
                f"Loss={avg_loss:.4f} [ce={avg_parts['ce']:.4f}, align={avg_parts['align']:.4f}]"
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
