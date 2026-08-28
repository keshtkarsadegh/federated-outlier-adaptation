"""Knowledge-distillation trainer using the frozen global model as teacher."""

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import BEST_KD_ALPHA, BEST_KD_T
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer


class DistillationTrainer(BaseTrainer):
    def __init__(
        self, global_model_path=None, T=BEST_KD_T, alpha=BEST_KD_ALPHA, provider=None
    ):  # extra weight when student is wrong
        super().__init__(provider=provider)
        self.T = T
        self.alpha = alpha

        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        self.teacher_model = self.provider.make_teacher_model()
        self.teacher_model.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model.to(self.device)
        self.teacher_model.eval()

    def distillation_loss(self, student_logits, labels, images):
        with torch.no_grad():
            teacher_logits = self.teacher_model(images)

        # per-sample CE
        ce = F.cross_entropy(student_logits, labels, reduction='none')

        # KD with temperature and log-targets
        log_p_s = F.log_softmax(student_logits / self.T, dim=1)
        log_p_t = F.log_softmax(teacher_logits / self.T, dim=1)
        kd_per_sample = F.kl_div(log_p_s, log_p_t,
                                 reduction='none', log_target=True).sum(dim=1) * (self.T ** 2)

        # combine per-sample, then reduce to scalar
        loss_vec = self.alpha * ce + (1 - self.alpha) * kd_per_sample
        return loss_vec.mean()

    def train(self, train_loader, eval_loader, epochs=10,patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0

            for images, labels in tqdm(train_loader, desc=f"[Distill-Train] Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.distillation_loss(outputs, labels, images)

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

            if val_acc > best_val_acc:
                best_val_acc, bad, best_model = val_acc, 0, copy.deepcopy(self.model)

                NistLogger.info(f"Saved best model with val_acc={val_acc:.4f}")
            else:
                bad += 1
                if bad >= patience:
                    NistLogger.info(f"Early stopping (patience={patience}).")
                    break

        if best_model:
            self.model = best_model


