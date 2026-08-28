"""
One trainer for the whole regularisation family.

Every regularised objective of this study has the same shape: cross-entropy on
the client's own data plus a penalty that keeps the client close to an anchor
model,

    Loss = CE(f_theta(x), y) + lam * D_space(theta ; anchor)

and the published trainers differ only in *where* the distance is measured and
*which* model is the anchor.  :class:`AnchoredTrainer` makes both explicit so
the family can be swept with one grid instead of six.

Spaces (``space``)
------------------
``param_l2``      the FedProx penalty of :class:`CFProxTrainer`; see
                  "The two ``param_l2`` conventions" below for what ``lam``
                  means in it.
``fisher``        Fisher-weighted parameter distance, ``0.5 * sum_i F_i
                  (theta_i - anchor_i)^2``.  ``F`` is normalised to mean 1 so
                  that ``lam`` is comparable across spaces.
``fisher_scaled`` the same penalty with the dynamic lambda cap of
                  :class:`EWCTrainer` (kept as an ablation of that rule).
``logit_l2``      MSE between student and anchor logits -
                  :class:`CFLogitConsistencyTrainer`.
``kd``            Hinton distillation, ``T^2 * KL(p_anchor^T || p_student^T)``,
                  written *without* the ``(1 - alpha)`` convention.
``ntd``           not-true distillation: the same KL restricted to the classes
                  the sample does not belong to.
``feature_l2``    MSE between the penultimate representations.
``kd+fisher``     hybrid; ``mix`` splits ``lam`` between the two terms.

Anchors (``anchor``)
--------------------
``frozen``   the global model on disk (default) - what every published trainer
             uses, so the defaults reproduce their behaviour.
``current``  the model the client received at the start of its local training.
             That is the true FedProx / drift-control formulation: in the
             cyclic scenario the anchor moves within a round.

Relation to the published trainers
----------------------------------
=========================================  ==========================================
published                                  equivalent configuration
=========================================  ==========================================
``CFProxTrainer(lambda_prox=l)``           ``space="param_l2", anchor="frozen", lam=l``
``CFLogitConsistencyTrainer(l)``           ``space="logit_l2", anchor="frozen", lam=l``
``CFAlignedFeatureTrainer(beta=b)``        ``space="logit_l2", lam=b`` (logit form)
``FeatureAlignmentTrainer(beta=b)``        ``space="feature_l2", lam=b``
``NTDTrainer(beta=b, tau=t)``              ``space="ntd", lam=b, T=t``
``EWCTrainer(ewc_lambda=l)``               ``space="fisher_scaled", lam=l``
``DistillationTrainer(T, alpha)``          ``space="kd", T=T, lam=(1-alpha)/alpha``
=========================================  ==========================================

The two ``param_l2`` conventions
-------------------------------
FedProx's proximal term is

    (mu / 2) * ||theta - theta_anchor||^2

with the norm taken over **all** parameters at once.  The published
implementation here is not that quantity: :meth:`_param_l2` adds up the
*per-tensor mean* squared error,

    sum_p mean_i (theta_p,i - anchor_p,i)^2 ,

so each tensor is divided by its own element count and the factor 1/2 is
absent.  For a network whose tensors hold n_p elements the two are related by

    lam * D_mean  =  lam * sum_p (1/n_p) ||Delta_p||^2 ,
    (mu/2) * ||Delta||^2 = (mu/2) * sum_p ||Delta_p||^2 ,

i.e. ``lam`` is larger than ``mu`` by roughly ``2 * n``, with ``n`` the typical
tensor size - five to six orders of magnitude for this model.  That is exactly
why the published sweep selected ``lam = 1000`` while FedProx's own grid stops
at ``mu = 1``: the two numbers are not the same quantity and were never
comparable.

Both conventions are available and neither is silently changed:

``param_l2_convention="mean_per_tensor"``
    the default and the published behaviour, bit for bit;
``param_l2_convention="fedprox"``
    ``0.5 * sum_p ||Delta_p||^2``, so ``lam`` **is** FedProx's ``mu`` and the
    published grid mu in {0.001, 0.01, 0.1, 1} means what it means in the paper.

The convention is recorded in the provenance of every run through the trainer's
keyword arguments, so a stored result always says which of the two it used.

The knowledge-distillation row holds *up to the alpha convention*:
``DistillationTrainer`` computes ``alpha * CE + (1 - alpha) * T^2 * KL`` while
this trainer computes ``CE + lam * T^2 * KL``.  With ``lam = (1-alpha)/alpha``
the two objectives differ by the constant factor ``alpha``, so they have the
same minimiser and the same gradient direction, and the effective learning rate
differs by that factor.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from federated_outlier_adaptation.constants import (
    BEST_ANCHOR_LAM,
    BEST_ANCHOR_MIX,
    BEST_ANCHOR_T,
    DEFAULT_ANCHOR,
    DEFAULT_ANCHOR_SPACE,
)
from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.ewc_trainer import load_fisher_and_params
from federated_outlier_adaptation.trainers.feature_alignment_trainer import (
    logits_and_features,
)
from federated_outlier_adaptation.trainers.ntd_trainer import not_true_log_softmax

#: Distance spaces the trainer understands.
ANCHOR_SPACES = (
    "param_l2",
    "fisher",
    "fisher_scaled",
    "logit_l2",
    "kd",
    "ntd",
    "feature_l2",
    "kd+fisher",
)

#: Anchor models the trainer understands.
ANCHORS = ("frozen", "current")

#: Conventions of the ``param_l2`` penalty; see the module docstring.
PARAM_L2_CONVENTIONS = ("mean_per_tensor", "fedprox")

#: Spaces that need a forward pass through the anchor model.
_OUTPUT_SPACES = ("logit_l2", "kd", "ntd", "feature_l2")

#: Spaces that need the Fisher information.
_FISHER_SPACES = ("fisher", "fisher_scaled", "kd+fisher")


class AnchoredTrainer(BaseTrainer):
    """
    Cross-entropy plus one anchored penalty; see the module docstring.

    Args:
        space: Distance space, one of :data:`ANCHOR_SPACES`.
        anchor: ``"frozen"`` (default) or ``"current"``.
        lam: Weight of the penalty.
        T: Softmax temperature of the ``kd`` and ``ntd`` spaces.
        mix: Split of ``lam`` in the hybrid space; ``mix`` goes to the first
            term (``kd``) and ``1 - mix`` to the second (``fisher``).
        normalise_fisher: Scale the Fisher diagonal to mean 1 so that ``lam``
            means the same thing as in the other spaces.  Turn it off to
            reproduce the raw ``EWCTrainer`` penalty.
        param_l2_convention: ``"mean_per_tensor"`` (default) is the published
            penalty; ``"fedprox"`` makes ``lam`` FedProx's ``mu``.  See the
            module docstring.
    """

    def __init__(
        self,
        space: str = DEFAULT_ANCHOR_SPACE,
        anchor: str = DEFAULT_ANCHOR,
        lam: float = BEST_ANCHOR_LAM,
        T: float = BEST_ANCHOR_T,
        mix: float = BEST_ANCHOR_MIX,
        normalise_fisher: bool = True,
        param_l2_convention: str = "mean_per_tensor",
        learning_rate=1e-3,
        weight_decay=1e-4,
        global_model_path=None,
        fisher_path=None,
        provider=None,
    ):
        if space not in ANCHOR_SPACES:
            raise ValueError(f"Unknown space {space!r}; expected one of {ANCHOR_SPACES}")
        if anchor not in ANCHORS:
            raise ValueError(f"Unknown anchor {anchor!r}; expected one of {ANCHORS}")
        if param_l2_convention not in PARAM_L2_CONVENTIONS:
            raise ValueError(
                f"Unknown param_l2 convention {param_l2_convention!r}; "
                f"expected one of {PARAM_L2_CONVENTIONS}"
            )

        super().__init__(
            learning_rate=learning_rate, weight_decay=weight_decay, provider=provider
        )
        self.space = space
        self.anchor = anchor
        self.lam = lam
        self.T = T
        self.mix = mix
        self.normalise_fisher = normalise_fisher
        self.param_l2_convention = param_l2_convention

        if not global_model_path:
            global_model_path = self.provider.global_model_path
        self.global_model_path = global_model_path

        # The frozen global model: the anchor itself when ``anchor="frozen"``,
        # and the network that produces the anchor outputs in every case where
        # the distance is measured on outputs.
        teacher = self.provider.make_teacher_model()
        teacher.load_state_dict(torch.load(global_model_path, map_location=self.device))
        self.teacher_model = teacher.to(self.device)
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        self.frozen_state = {
            name: value.detach().clone().to(self.device)
            for name, value in self.teacher_model.state_dict().items()
        }
        self.anchor_state = dict(self.frozen_state)

        self.fisher = None
        self.fisher_path = fisher_path
        if self._needs_fisher():
            if not fisher_path:
                fisher_path = self.provider.fisher_dir
            self.fisher_path = fisher_path
            fisher, _ = load_fisher_and_params(fisher_path, self.device)
            self.fisher = self._prepare_fisher(fisher)

    # ------------------------------------------------------------------ setup
    def _needs_fisher(self) -> bool:
        return self.space in _FISHER_SPACES

    def _prepare_fisher(self, fisher):
        """Optionally rescale the Fisher diagonal to mean 1."""
        if not self.normalise_fisher:
            return fisher
        total = sum(float(value.sum()) for value in fisher.values())
        count = sum(value.numel() for value in fisher.values())
        mean = (total / count) if count else 0.0
        if mean <= 0:  # pragma: no cover - degenerate Fisher
            return fisher
        return {name: value / mean for name, value in fisher.items()}

    def set_anchor(self, state_dict):
        """
        Point the penalty at a new anchor.

        Called by :meth:`set_model` with the model the client received, which
        for ``anchor="current"`` is the round's starting point.  With the
        default ``anchor="frozen"`` the call is a no-op, so the behaviour of
        the published objectives is unchanged.
        """
        if self.anchor != "current":
            return
        self.anchor_state = {
            name: value.detach().clone().to(self.device)
            for name, value in state_dict.items()
        }
        if self.space in _OUTPUT_SPACES or self.space == "kd+fisher":
            self.teacher_model.load_state_dict(state_dict)
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

    def set_model(self, model):
        """Adopt a copy of ``model`` and, for a moving anchor, anchor on it."""
        super().set_model(model)
        self.set_anchor(model.state_dict())

    # ------------------------------------------------------------------- loss
    def _param_l2(self):
        """
        Distance to the anchor in parameter space, in the configured convention.

        ``mean_per_tensor`` (the published one) sums the per-tensor mean squared
        error; ``fedprox`` returns ``0.5 * ||theta - theta_anchor||^2`` summed
        over every element, so ``lam`` is FedProx's ``mu``.  See the module
        docstring for why the two differ by orders of magnitude.
        """
        penalty = 0.0
        fedprox = self.param_l2_convention == "fedprox"
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or name not in self.anchor_state:
                continue
            if fedprox:
                penalty = penalty + ((parameter - self.anchor_state[name]) ** 2).sum()
            else:
                penalty = penalty + F.mse_loss(
                    parameter, self.anchor_state[name], reduction="mean"
                )
        return 0.5 * penalty if fedprox else penalty

    def _fisher_penalty(self):
        """``0.5 * sum_i F_i (theta_i - anchor_i)^2``."""
        penalty = 0.0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in self.fisher or name not in self.anchor_state:
                continue
            penalty = penalty + (
                self.fisher[name] * (parameter - self.anchor_state[name]) ** 2
            ).sum()
        return 0.5 * penalty

    def _output_penalty(self, space, student_logits, student_features, labels, images):
        """Distance measured on the network's outputs, in the given space."""
        with torch.no_grad():
            if space == "feature_l2":
                _, anchor_features = logits_and_features(self.teacher_model, images)
                anchor_logits = None
            else:
                anchor_logits = self.teacher_model(images)
                anchor_features = None

        if space == "feature_l2":
            return F.mse_loss(student_features, anchor_features.detach(), reduction="mean")
        if space == "logit_l2":
            return F.mse_loss(student_logits, anchor_logits.detach(), reduction="mean")
        if space == "kd":
            student_log_p = F.log_softmax(student_logits / self.T, dim=1)
            anchor_log_p = F.log_softmax(anchor_logits / self.T, dim=1)
            kl = F.kl_div(
                student_log_p, anchor_log_p, reduction="none", log_target=True
            ).sum(dim=1)
            return kl.mean() * (self.T ** 2)
        # ntd
        anchor_log_p = not_true_log_softmax(anchor_logits, labels, self.T)
        student_log_p = not_true_log_softmax(student_logits, labels, self.T)
        kl = F.kl_div(
            student_log_p, anchor_log_p, reduction="none", log_target=True
        ).nan_to_num(0.0).sum(dim=1)
        return kl.mean() * (self.T ** 2)

    def penalty(self, student_logits, student_features, labels, images):
        """The weighted penalty term ``lam * D_space``."""
        if self.lam <= 0:
            return torch.zeros((), device=self.device)

        if self.space == "param_l2":
            return self.lam * self._param_l2()
        if self.space == "fisher":
            return self.lam * self._fisher_penalty()
        if self.space == "fisher_scaled":
            # EWCTrainer caps lambda by the ratio of the two loss magnitudes.
            raw = self._fisher_penalty() * 2.0
            reference = F.cross_entropy(student_logits, labels)
            scaling = reference.item() / (raw.item() + 1e-8)
            capped = min(scaling, self.lam)
            return (capped / 2.0) * raw
        if self.space == "kd+fisher":
            kd_term = self._output_penalty("kd", student_logits, student_features, labels, images)
            fisher_term = self._fisher_penalty()
            return self.lam * (self.mix * kd_term + (1.0 - self.mix) * fisher_term)
        return self.lam * self._output_penalty(
            self.space, student_logits, student_features, labels, images
        )

    def custom_loss_fn(self, student_logits, student_features, labels, images):
        ce = self.criterion(student_logits, labels)
        term = self.penalty(student_logits, student_features, labels, images)
        total = ce + term
        return total, {"ce": ce.item(), "penalty": float(term.item())}

    # ------------------------------------------------------------------ train
    def train(self, train_loader, eval_loader, epochs=10, patience=5):
        self.train_accuracies.clear()
        self.val_accuracies.clear()

        needs_features = self.space == "feature_l2"
        best_model, best_val_acc, bad = None, -float("inf"), 0

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            running = {"ce": 0.0, "penalty": 0.0}

            for images, labels in tqdm(
                train_loader, desc=f"[Anchored:{self.space}] Epoch {epoch+1}/{epochs}"
            ):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                if needs_features:
                    outputs, features = logits_and_features(self.model, images)
                else:
                    outputs, features = self.model(images), None

                loss, parts = self.custom_loss_fn(outputs, features, labels, images)
                if torch.isnan(loss):
                    NistLogger.error("Loss is NaN, aborting.")
                    raise ValueError("Loss became NaN")

                loss.backward()
                self.optimizer.step()

                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                running["ce"] += parts["ce"] * batch_size
                running["penalty"] += parts["penalty"] * batch_size

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
            averaged = {k: v / max(total, 1) for k, v in running.items()}
            NistLogger.info(
                f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                f"Loss={avg_loss:.4f} [ce={averaged['ce']:.4f}, "
                f"penalty={averaged['penalty']:.4f}]"
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
