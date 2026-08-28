"""Combined knowledge-distillation and EWC trainer."""

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import (
    BEST_EWC_LAMBDA,
    BEST_KD_ALPHA,
    BEST_KD_T,
)
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.ewc_trainer import load_fisher_and_params


class DistillationEWCTrainer(BaseTrainer):
    """
    If alpha == 1.0:
        Loss = CE + (lambda_scaled / 2) * Σ_i F_i * (θ_i − θ0_i)^2
        (identical to EWCTrainer.ewc_loss)
    Otherwise:
        Loss = α·CE + (1−α)·T²·KL + (lambda_scaled / 2) * Σ_i F_i * (θ_i − θ0_i)^2
    """
    def __init__(
        self,
        fisher_path=None,
        T=BEST_KD_T,
        alpha=BEST_KD_ALPHA,
        ewc_lambda=BEST_EWC_LAMBDA,
        provider=None,
        # match EWCTrainer
    ):
        super().__init__(provider=provider)
        self.T = T
        self.alpha = alpha
        self.ewc_lambda = ewc_lambda

        # Load Fisher + global params
        if not fisher_path:
            fisher_path = self.provider.fisher_dir
        self.fisher_path = fisher_path
        self.fisher, self.global_params = load_fisher_and_params(fisher_path, self.device)
        for k in list(self.fisher.keys()):
            self.fisher[k] = self.fisher[k].to(self.device)
        for k in list(self.global_params.keys()):
            if torch.is_tensor(self.global_params[k]):
                self.global_params[k] = self.global_params[k].to(self.device)

        # Teacher only if KD is used (alpha < 1.0)
        self.teacher_model = None
        if self.alpha < 1.0:
            self.teacher_model = self.provider.make_teacher_model()
            self.teacher_model.load_state_dict(self.global_params, strict=False)
            self.teacher_model.to(self.device).eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def _ewc_penalty_raw_all(self):
        """Σ_i F_i * (θ_i − θ0_i)^2 (no 0.5 here; exact param set as EWCTrainer)."""
        penalty = 0.0  # python float to mirror EWCTrainer init; will promote to tensor on first add
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            Fi = self.fisher[name]               # direct index (will error if missing, like EWCTrainer)
            ref = self.global_params[name]
            Fi = Fi.to(param.device)
            if torch.is_tensor(ref):
                ref = ref.to(param.device)
            penalty = penalty + (Fi * (param - ref).pow(2)).sum()
        return penalty

    def _task_loss(self, logits, labels, images=None):
        # If alpha==1.0 this is exactly CE (KD not used)
        ce = F.cross_entropy(logits, labels)
        if self.alpha >= 1.0 or self.teacher_model is None:
            return ce
        with torch.no_grad():
            t_logits = self.teacher_model(images)
        T = self.T
        sT_log = F.log_softmax(logits / T, dim=1)
        tT_log = F.log_softmax(t_logits / T, dim=1).detach()
        kd = F.kl_div(sT_log, tT_log, reduction="batchmean", log_target=True) * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kd

    def train(self, train_loader, eval_loader, epochs=10,patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            correct = total = 0

            for images, labels in tqdm(train_loader, desc=f"[EWC-Train] Epoch {epoch + 1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # match EWCTrainer

                logits = self.model(images)
                task_loss = self._task_loss(logits, labels, images)

                ewc_penalty_raw = self._ewc_penalty_raw_all()
                scaling_factor = task_loss.item() / (ewc_penalty_raw.item() + 1e-8)
                lambda_scaled = min(scaling_factor, self.ewc_lambda)
                loss = task_loss + (lambda_scaled / 2.0) * ewc_penalty_raw

                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss is NaN")

                loss.backward()
                self.optimizer.step()

                _, pred = logits.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total if total > 0 else 0.0
            self.train_accuracies.append(train_acc)

            # === Validation ===
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in eval_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total if total > 0 else 0.0
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
