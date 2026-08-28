"""Feature-alignment trainer matching client logits to the global model."""

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import BEST_FEATURE_BETA
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

class CFAlignedFeatureTrainer(BaseTrainer):
    """
    Loss = CE(student, labels) + β · MSE(logits_s, logits_t)

    - Teacher = frozen global model.
    - Feature alignment here = pre-softmax logits.
    """

    def __init__(
        self,
        global_model_path=None,
        beta=BEST_FEATURE_BETA,
        learning_rate=1e-3,
        weight_decay=1e-4,
        provider=None,
    ):
        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.beta = beta

        # Load frozen teacher
        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        self.teacher_model = self.provider.make_teacher_model()
        self.teacher_model.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    # --- Feature alignment loss ---
    def _feature_align_loss(self, student_logits, teacher_logits):
        if self.beta <= 0:
            return torch.zeros((), device=self.device)
        return F.mse_loss(student_logits, teacher_logits.detach())

    # --- Override train loop ---
    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()
        best_model, best_val, bad = None, -float("inf"), 0

        for ep in range(epochs):
            self.model.train()
            total, correct = 0, 0

            for images, labels in tqdm(train_loader, desc=f"[FeatAlign] Epoch {ep+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass
                student_logits = self.model(images)
                with torch.no_grad():
                    teacher_logits = self.teacher_model(images)

                # Loss = CE + β·MSE(logits_s, logits_t)
                ce_loss = self.criterion(student_logits, labels)
                feat_loss = self._feature_align_loss(student_logits, teacher_logits)
                loss = ce_loss + self.beta * feat_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    raise ValueError("Loss NaN/Inf")

                loss.backward()
                self.optimizer.step()

                # Training stats
                _, pred = student_logits.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / max(1, total)
            self.train_accuracies.append(train_acc)

            # === Validation ===
            self.model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    out = self.model(images)
                    _, pred = out.max(1)
                    v_correct += pred.eq(labels).sum().item()
                    v_total += labels.size(0)

            val_acc = v_correct / max(1, v_total)
            self.val_accuracies.append(val_acc)
            NistLogger.info(f"Epoch {ep+1} | Train {train_acc:.4f} | Val {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val:
                best_val, bad, best_model = val_acc, 0, copy.deepcopy(self.model)
                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            else:
                bad += 1
                if bad >= patience:
                    NistLogger.info(f"Early stopping (patience={patience}).")
                    break

        if best_model is not None:
            self.model = best_model
