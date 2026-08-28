"""
When a centralised training stops, and which weights it keeps.

Every centralised phase of this study - the global model, the pooled oracle, a
client's local fine-tune - used to run a fixed number of epochs and keep
whatever the last one produced.  That makes the epoch count a hyperparameter
nobody tuned: too few and the model is undertrained, too many and the reported
number is a late, overfitted epoch rather than the best one the run found.

The rule here replaces it with the standard one:

* **stop** when the validation accuracy has not improved for ``patience``
  epochs, but never before ``min_epochs`` have run - the floor matters because
  validation accuracy on a small split is noisy early on and a plateau of two or
  three epochs at the start means nothing;
* **keep** the weights of the best validation epoch, restored before anything is
  evaluated or saved, so the reported test accuracy and the checkpoint on disk
  are the same model.

Both are **off by default** (``patience=None``), which runs every epoch and
keeps the published behaviour of each caller unchanged.

What is recorded
----------------
:meth:`EarlyStopper.metrics` returns the epoch the run stopped at, the epoch it
kept, that epoch's validation accuracy and whether the patience rule fired -
so a result says how long it trained and why it stopped, rather than only how
long it was allowed to.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional


class EarlyStopper:
    """
    The convergence rule of one training run.

    Args:
        patience: Non-improving epochs tolerated.  ``None`` disables stopping
            entirely; the best-epoch bookkeeping still runs, because keeping the
            best weights is worth doing whether or not the run stops early.
        min_epochs: Epochs that always run, whatever the validation accuracy
            does.  The patience counter is kept from the first epoch, but it
            cannot end the run before this floor.
        restore_best: Keep a copy of the best model and hand it back at the end.
    """

    def __init__(
        self,
        patience: Optional[int] = None,
        min_epochs: int = 0,
        restore_best: bool = True,
    ):
        self.patience = None if patience is None else max(0, int(patience))
        self.min_epochs = max(0, int(min_epochs))
        self.restore_best = bool(restore_best)

        self.best_accuracy: float = -float("inf")
        self.best_epoch: Optional[int] = None
        self.best_model = None
        self.epochs_run: int = 0
        self.stopped_early: bool = False
        self._bad: int = 0

    @property
    def enabled(self) -> bool:
        """Whether the patience rule can end a run at all."""
        return self.patience is not None

    def update(self, epoch: int, accuracy: float, model=None) -> bool:
        """
        Record one epoch and say whether training should stop after it.

        Args:
            epoch: Zero-based epoch index.
            accuracy: This epoch's validation accuracy.
            model: The model to copy when this epoch is the best so far.

        Returns:
            True when the run should stop now.
        """
        self.epochs_run = int(epoch) + 1
        if accuracy > self.best_accuracy:
            self.best_accuracy = float(accuracy)
            self.best_epoch = int(epoch)
            self._bad = 0
            if self.restore_best and model is not None:
                self.best_model = copy.deepcopy(model)
            return False

        self._bad += 1
        if not self.enabled or self.epochs_run < self.min_epochs:
            # Below the floor the counter still runs, but it cannot stop the run.
            return False
        if self._bad >= self.patience:
            self.stopped_early = True
            return True
        return False

    def best(self, fallback=None):
        """The best-validation model, or ``fallback`` when none was kept."""
        return self.best_model if self.best_model is not None else fallback

    def metrics(self) -> Dict[str, Any]:
        """What the run did, for the metrics file and the provenance block."""
        return {
            "early_stopping_patience": self.patience,
            "min_epochs": self.min_epochs,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_accuracy": (
                None if self.best_epoch is None else self.best_accuracy
            ),
            "stopped_early": self.stopped_early,
        }


def convergence_kwargs(
    patience: Optional[int] = None, min_epochs: int = 0
) -> Dict[str, Any]:
    """The two settings, for a provenance block that has no stopper to hand."""
    return {
        "early_stopping_patience": None if patience is None else int(patience),
        "min_epochs": int(min_epochs),
    }
