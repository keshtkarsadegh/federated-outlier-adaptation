"""
Selective layer freezing.

The client adapts only the top of the network; the frozen part keeps the
representation the global model learned.  Two scopes are supported:

    "conv"  the convolutional feature extractor (``model.features``)
    "body"  the feature extractor plus the first fully connected layer
            (``model.features`` and ``model.fc1``)

Freezing means three things, all of which survive :meth:`set_model` (the runners
hand a fresh deep copy of the global model to the trainer every round):

    - ``requires_grad`` is cleared on every parameter of the frozen part,
    - the optimizer is built over the trainable parameters only, so weight
      decay never touches the frozen weights,
    - normalisation layers inside the frozen part stay in ``eval`` mode while
      training, so their running statistics are not updated either.

:class:`DistillationFreezeTrainer` combines the same freezing with the
knowledge-distillation objective; the KD math itself is reused from
:class:`DistillationTrainer` rather than reimplemented.
"""

from __future__ import annotations

import copy

import torch
from torch import optim
from tqdm import tqdm

from federated_outlier_adaptation.constants import (
    BEST_KD_ALPHA,
    BEST_KD_T,
    FREEZE_SCOPE,
)
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.distillation_trainer import DistillationTrainer

#: Attribute names of the model that each scope freezes, in order.
FREEZE_SCOPES = {
    "conv": ("features",),
    "body": ("features", "fc1"),
}


class FreezeTrainer(BaseTrainer):
    """
    Cross-entropy training with part of the network frozen.

    Args:
        scope (str): ``"conv"`` or ``"body"``; defaults to
            ``constants.FREEZE_SCOPE``.
        learning_rate (float): Adam learning rate for the trainable part.
        weight_decay (float): Adam weight decay for the trainable part.
        provider: Dataset provider supplying the model topology.
    """

    def __init__(
        self,
        scope=FREEZE_SCOPE,
        learning_rate=1e-3,
        weight_decay=1e-4,
        provider=None,
    ):
        if scope not in FREEZE_SCOPES:
            raise ValueError(
                f"Unknown freeze scope {scope!r}; expected one of {sorted(FREEZE_SCOPES)}"
            )
        self.scope = scope
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self._apply_freeze()

    # ------------------------------------------------------------------ freeze
    def frozen_modules(self):
        """Sub-modules of the active model that the scope freezes."""
        modules = []
        for name in FREEZE_SCOPES[self.scope]:
            module = getattr(self.model, name, None)
            if module is None:
                raise AttributeError(
                    f"Model {type(self.model).__name__} has no attribute {name!r}; "
                    "freezing requires the standard 'features'/'fc1' accessors."
                )
            modules.append(module)
        return modules

    def frozen_parameter_names(self):
        """Names (as in ``state_dict``) of the parameters held constant."""
        names = []
        for attr in FREEZE_SCOPES[self.scope]:
            module = getattr(self.model, attr)
            names += [f"{attr}.{n}" for n, _ in module.named_parameters()]
        return names

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _apply_freeze(self):
        """Clear ``requires_grad`` on the frozen part and rebuild the optimizer."""
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        for module in self.frozen_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.optimizer = optim.Adam(
            self.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _train_mode(self):
        """Put the model in training mode while keeping the frozen part in eval."""
        self.model.train()
        for module in self.frozen_modules():
            module.eval()

    def set_model(self, model):
        """Adopt a copy of ``model`` and re-apply the freezing to the copy."""
        self.model = copy.deepcopy(model).to(self.device)
        self._apply_freeze()

    def load_model(self, model_path):
        """Load weights from disk and re-apply the freezing."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self._apply_freeze()
        return self.model

    # -------------------------------------------------------------------- loss
    def batch_loss(self, outputs, labels, images):
        """Objective of a single batch.  Cross-entropy for the plain variant."""
        return self.criterion(outputs, labels)

    # ------------------------------------------------------------------- train
    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self._train_mode()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(
                train_loader, desc=f"[Freeze:{self.scope}] Epoch {epoch+1}/{epochs}"
            ):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.batch_loss(outputs, labels, images)

                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss is NaN")

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
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
            NistLogger.info(
                f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc, bad, best_model = val_acc, 0, copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            else:
                bad += 1
                if bad >= patience:
                    NistLogger.info(f"Early stopping (patience={patience}).")
                    break

        if best_model is not None:
            self.model = best_model
            self._apply_freeze()


class DistillationFreezeTrainer(FreezeTrainer):
    """
    Knowledge distillation against the frozen global model, with layer freezing.

    The loss is exactly :meth:`DistillationTrainer.distillation_loss`; only the
    set of trainable parameters differs.
    """

    def __init__(
        self,
        scope=FREEZE_SCOPE,
        T=BEST_KD_T,
        alpha=BEST_KD_ALPHA,
        learning_rate=1e-3,
        weight_decay=1e-4,
        global_model_path=None,
        provider=None,
    ):
        super().__init__(
            scope=scope,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            provider=provider,
        )
        self.T = T
        self.alpha = alpha

        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        teacher = self.provider.make_teacher_model()
        teacher.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model = teacher.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

    def batch_loss(self, outputs, labels, images):
        return DistillationTrainer.distillation_loss(self, outputs, labels, images)
