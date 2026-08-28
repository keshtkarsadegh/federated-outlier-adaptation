"""
Constraint-respecting forgetting signals.

The setting of this study forbids a running experiment from reading the source
population's data: once the global model has been shipped, the writers it was
trained on are gone.  Every number this repository reports on the source split -
``accuracies[r][1]``, ``source_val_accuracies`` - is therefore an *evaluation*
quantity.  A method that used it to decide when to stop or which configuration
to keep would be reading data it is not allowed to see, and the published stop
rule ``--stop-when-global-below-clients`` is kept only as such an oracle
reference.

This module records, every round and in both runners, the quantities a server
*may* compute, so that stopping and selection can be studied under the real
constraint.  All of them are new keys; nothing existing changes.

Parameter-space signals - no data at all
----------------------------------------
``dist_l2_to_global``
    ``||theta_t - theta_g||_2`` over the float entries of the state dict.
``dist_fisher_to_global``
    ``sum_i F_i (theta_t,i - theta_g,i)^2`` with the *shipped* diagonal Fisher
    information (``provider.fisher_dir``, i.e. the same file EWC uses), which the
    server owns because it produced it together with the global model.
``dist_fisher_norm_to_global``
    the same with ``F`` normalised to unit mass (``F / sum_i F_i``), so the value
    is a weighted mean square displacement and is comparable across datasets and
    across differently scaled Fisher estimates.

Client-side signals - the clients' own held-out data
----------------------------------------------------
Computed on the **validation half** of the pool's held-out split, i.e. data the
clients never train on and that the server is allowed to ask them about:

``retention_known``
    accuracy of ``theta_t`` restricted to the samples ``theta_g`` classified
    correctly - "how much of what the shipped model already knew is left".
``agreement_with_global``
    fraction of samples where ``argmax theta_t == argmax theta_g``.
``kl_global_to_current``
    mean ``KL(softmax theta_g || softmax theta_t)``.

Proxy-set signals - public data the server owns
-----------------------------------------------
``proxy_acc`` / ``proxy_kl``
    the accuracy of ``theta_t`` on the provider's proxy set and the same mean KL
    against ``theta_g``, when :meth:`DatasetProvider.proxy_loader` returns one.
    Providers without a proxy set report ``None`` for both.

Cost
----
``theta_g``'s predictions on both reference sets are computed **once per run**
and cached (predicted class, class probabilities and the entropy term of the
KL), so a round costs one extra forward pass over the pool's validation halves
and one over the proxy set - the same order as the evaluation passes the runners
already make every round.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.utils.eval_cache import POOL_VAL_SET, PROXY_SET

#: Keys the tracker contributes to a result payload, in report order.
SIGNAL_KEYS = (
    "dist_l2_to_global",
    "dist_fisher_to_global",
    "dist_fisher_norm_to_global",
    "retention_known",
    "agreement_with_global",
    "kl_global_to_current",
    "proxy_acc",
    "proxy_kl",
)

#: Floor of the current model's probabilities inside the KL, so a hard zero
#: cannot turn the divergence into an infinity.
KL_EPSILON = 1e-12


def float_state(model_or_state) -> dict[str, torch.Tensor]:
    """Float entries of a state dict, the parameters aggregation acts on."""
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    return {
        key: value.detach()
        for key, value in state.items()
        if torch.is_tensor(value) and value.dtype.is_floating_point
    }


def load_fisher(fisher_dir: Optional[Path]) -> Optional[dict[str, torch.Tensor]]:
    """
    Read ``fisher.pt`` from a Fisher directory, or ``None`` when it is absent.

    The file is the one the EWC objective already consumes; the tracker only
    reads it, so a run without Fisher information simply reports ``None`` for
    the two Fisher distances.
    """
    if fisher_dir is None:
        return None
    path = Path(fisher_dir) / "fisher.pt"
    if not path.is_file():
        return None
    try:
        fisher = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError):  # pragma: no cover - broken artefact
        NistLogger.info(f"Fisher information at {path} could not be read; distances disabled.")
        return None
    if not isinstance(fisher, dict):  # pragma: no cover - unexpected artefact
        return None
    return {key: value.detach().float() for key, value in fisher.items() if torch.is_tensor(value)}


class ReferencePredictions:
    """
    ``theta_g``'s answers on one evaluation set, computed once and kept.

    Attributes:
        predicted: ``[N]`` predicted class of the reference model.
        labels: ``[N]`` ground-truth labels.
        probabilities: ``[N, C]`` softmax of the reference model.
        negative_entropy: ``[N]`` the ``sum_c p_g log p_g`` term of the KL.
        correct: ``[N]`` boolean mask of the samples the reference got right.
    """

    __slots__ = ("predicted", "labels", "probabilities", "negative_entropy", "correct", "size")

    def __init__(self, predicted, labels, probabilities, negative_entropy):
        self.predicted = predicted
        self.labels = labels
        self.probabilities = probabilities
        self.negative_entropy = negative_entropy
        self.correct = predicted.eq(labels)
        self.size = int(predicted.numel())


def _forward_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenated logits and labels of one pass over ``loader``."""
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            output = model(inputs)
            if isinstance(output, (tuple, list)):  # models returning (logits, features)
                output = output[0]
            logits.append(output.detach().float().cpu())
            labels.append(torch.as_tensor(targets).detach().reshape(-1).cpu())
    if not logits:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(logits), torch.cat(labels).long()


def reference_from_logits(logits, labels) -> Optional[ReferencePredictions]:
    """Cache what the signals need from one set of reference logits."""
    if logits is None or logits.numel() == 0:
        return None
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    return ReferencePredictions(
        predicted=logits.argmax(dim=1),
        labels=torch.as_tensor(labels).reshape(-1).to(logits.device).long(),
        probabilities=probabilities,
        negative_entropy=(probabilities * log_probabilities).sum(dim=1),
    )


def reference_predictions(model, loader, device) -> Optional[ReferencePredictions]:
    """Run the reference model over ``loader`` and cache what the signals need."""
    if loader is None:
        return None
    logits, labels = _forward_logits(model, loader, device)
    return reference_from_logits(logits, labels)


def compare_logits(logits, reference: ReferencePredictions) -> dict:
    """
    Score one set of current logits against the cached reference answers.

    Returns:
        dict: ``accuracy``, ``retention_known``, ``agreement`` and ``kl``.  The
        retention entry is ``None`` when the reference model got nothing right.
    """
    if logits is None or logits.numel() == 0 or logits.shape[0] != reference.size:
        return {"accuracy": None, "retention_known": None, "agreement": None, "kl": None}

    log_probabilities = torch.log_softmax(logits, dim=1).clamp_min(math.log(KL_EPSILON))
    predicted = logits.argmax(dim=1)

    correct = predicted.eq(reference.labels)
    known = reference.correct
    retention = float(correct[known].float().mean()) if bool(known.any()) else None
    cross = (reference.probabilities * log_probabilities).sum(dim=1)
    return {
        "accuracy": float(correct.float().mean()),
        "retention_known": retention,
        "agreement": float(predicted.eq(reference.predicted).float().mean()),
        "kl": float((reference.negative_entropy - cross).mean()),
    }


def compare_to_reference(model, loader, reference: ReferencePredictions, device) -> dict:
    """Score the current model on ``loader`` against the cached reference answers."""
    logits, _ = _forward_logits(model, loader, device)
    return compare_logits(logits, reference)


class ForgettingSignalTracker:
    """
    Records the constraint-respecting signals of one run, round by round.

    Args:
        provider: Dataset provider; supplies the optional proxy set.
        reference_state: Float state dict of ``theta_g``, the model the run
            started from.
        reference_model: A model carrying those weights, used once to compute
            the cached reference predictions.
        device: Device the evaluation passes run on.
        fisher_dir: Directory holding ``fisher.pt``; ``None`` disables the two
            Fisher distances.
        batch_size: Batch size of the two evaluation loaders.
    """

    def __init__(
        self,
        provider,
        reference_state: dict[str, torch.Tensor],
        reference_model,
        device: str = "cpu",
        fisher_dir: Optional[Path] = None,
        batch_size: int = 64,
        cache=None,
    ):
        self.provider = provider
        self.device = device
        self.batch_size = int(batch_size)
        #: Materialised evaluation sets shared with the population, see
        #: :mod:`federated_outlier_adaptation.utils.eval_cache`.  When the pool's
        #: validation half and the proxy set are cached, a round's signals are
        #: read off the forward passes the pooled accuracies already needed.
        self.cache = cache
        self.reference_state = {
            key: value.detach().to("cpu", copy=True) for key, value in reference_state.items()
        }
        # The runners hand in their live ``global_model``, which aggregation
        # mutates in place; the reference answers must come from theta_g, so a
        # snapshot is taken here rather than a reference kept.
        self._reference_model = copy.deepcopy(reference_model).eval()
        self.fisher = load_fisher(fisher_dir)
        self.fisher_dir = str(fisher_dir) if fisher_dir is not None else None
        self._fisher_mass = None
        if self.fisher:
            total = sum(float(value.sum()) for value in self.fisher.values())
            self._fisher_mass = total if total > 0 else None

        self.heldout_loader = None
        self.heldout_reference: Optional[ReferencePredictions] = None
        self.proxy_loader = None
        self.proxy_reference: Optional[ReferencePredictions] = None
        self.proxy_set: dict = {}
        self._prepared = False

        self.series: dict[str, list] = {key: [] for key in SIGNAL_KEYS}
        self._prepare_proxy(self._reference_model)

    # -------------------------------------------------------------- reference
    def _cached(self, name: str) -> bool:
        return self.cache is not None and self.cache.has(name)

    def _reference_logits(self, name: str):
        """
        ``theta_g``'s logits on a cached set, computed once per run.

        Whatever model the cache was scoring is bound again afterwards, so
        anchoring the reference cannot disturb a round that is under way.
        """
        previous = self.cache.bound
        self.cache.bind(self._reference_model)
        logits = self.cache.logits(name).clone()
        labels = self.cache.labels(name).clone()
        self.cache.bind(previous)
        return logits, labels

    def _prepare_proxy(self, reference_model) -> None:
        """Build the proxy loader and its reference answers, if there is one."""
        try:
            loader = self.provider.proxy_loader(batch_size=self.batch_size)
        except (AttributeError, TypeError):  # pragma: no cover - exotic provider
            loader = None
        except (OSError, ValueError) as error:
            NistLogger.info(f"Proxy set unavailable: {error}")
            loader = None
        if loader is None:
            return
        try:
            self.proxy_set = dict(self.provider.proxy_info() or {})
        except (AttributeError, TypeError, OSError):  # pragma: no cover
            self.proxy_set = {}
        self.proxy_loader = loader
        if self.cache is not None and self.cache.add(PROXY_SET, loader.dataset):
            self.proxy_reference = reference_from_logits(*self._reference_logits(PROXY_SET))
        else:
            self.proxy_reference = reference_predictions(reference_model, loader, self.device)
        if self.proxy_reference is None:  # pragma: no cover - empty proxy set
            self.proxy_loader = None

    def prepare(self, population) -> None:
        """
        Anchor ``theta_g``'s answers on the pool's validation half.

        The reference set is exactly the set the pooled validation accuracy is
        measured on, in the same client order: with a cache the two share one
        forward pass per round, and without one the concatenated loader is built
        from the split loaders the population already caches.  Neither loader
        shuffles, so the cached answers stay aligned with every later pass.
        """
        if self._prepared:
            return
        self._prepared = True
        if population is None:
            return
        if self._cached(POOL_VAL_SET):
            self.heldout_reference = reference_from_logits(*self._reference_logits(POOL_VAL_SET))
            if self.heldout_reference is not None:
                return
        datasets = []
        for client_id in population.unique_clients:
            validation, _ = population.split_loaders(client_id, self.batch_size)
            if validation is not None:
                datasets.append(validation.dataset)
        if not datasets:
            return
        combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        self.heldout_loader = DataLoader(combined, batch_size=self.batch_size, shuffle=False)
        self.heldout_reference = reference_predictions(
            self._reference_model, self.heldout_loader, self.device
        )
        if self.heldout_reference is None:  # pragma: no cover - empty pool
            self.heldout_loader = None

    # ----------------------------------------------------------------- rounds
    def parameter_distances(self, model) -> dict:
        """The three parameter-space distances of the current model."""
        current = float_state(model)
        squared = 0.0
        fisher_weighted = 0.0
        for key, reference in self.reference_state.items():
            value = current.get(key)
            if value is None:
                continue
            delta = (value.detach().to("cpu", dtype=torch.float32) - reference.float()).pow(2)
            squared += float(delta.sum())
            if self.fisher is not None:
                weight = self.fisher.get(key)
                if weight is not None and weight.shape == delta.shape:
                    fisher_weighted += float((weight * delta).sum())
        if self.fisher is None:
            return {
                "dist_l2_to_global": math.sqrt(squared),
                "dist_fisher_to_global": None,
                "dist_fisher_norm_to_global": None,
            }
        normalised = (
            fisher_weighted / self._fisher_mass if self._fisher_mass is not None else None
        )
        return {
            "dist_l2_to_global": math.sqrt(squared),
            "dist_fisher_to_global": fisher_weighted,
            "dist_fisher_norm_to_global": normalised,
        }

    def _scores(self, name: str, loader, reference: ReferencePredictions, model) -> dict:
        """
        This round's comparison against the reference, from the cheapest source.

        A cached set is already being scored this round for the pooled accuracy,
        so the memoised logits are reused; otherwise the loader is walked.
        """
        if self._cached(name):
            return compare_logits(self.cache.logits(name), reference)
        return compare_to_reference(model, loader, reference, self.device)

    def record(self, model, population=None) -> dict:
        """
        Append this round's signals and return them.

        Args:
            model: The current global model ``theta_t``.
            population: The run's :class:`ClientPopulation`; passed on the first
                call so the held-out reference can be built.
        """
        if not self._prepared:
            self.prepare(population)
        values = dict(self.parameter_distances(model))
        values.update(
            {
                "retention_known": None,
                "agreement_with_global": None,
                "kl_global_to_current": None,
                "proxy_acc": None,
                "proxy_kl": None,
            }
        )
        if self.heldout_reference is not None:
            scores = self._scores(POOL_VAL_SET, self.heldout_loader, self.heldout_reference, model)
            values["retention_known"] = scores["retention_known"]
            values["agreement_with_global"] = scores["agreement"]
            values["kl_global_to_current"] = scores["kl"]
        if self.proxy_reference is not None:
            scores = self._scores(PROXY_SET, self.proxy_loader, self.proxy_reference, model)
            values["proxy_acc"] = scores["accuracy"]
            values["proxy_kl"] = scores["kl"]

        for key in SIGNAL_KEYS:
            self.series[key].append(values[key])
        return values

    # ----------------------------------------------------------------- report
    def info(self) -> dict:
        """The additive payload keys: the series plus a description block."""
        payload: dict[str, Any] = {key: list(values) for key, values in self.series.items()}
        payload["signals_info"] = {
            "reference": "the model the run started from (theta_g)",
            "heldout_samples": (
                self.heldout_reference.size if self.heldout_reference is not None else 0
            ),
            "fisher_dir": self.fisher_dir,
            "fisher_available": self.fisher is not None,
            "proxy_samples": (
                self.proxy_reference.size if self.proxy_reference is not None else 0
            ),
            "proxy_set": dict(self.proxy_set),
        }
        return payload


def empty_info() -> dict:
    """Payload keys of a run whose signals were not collected."""
    payload: dict[str, Any] = {key: [] for key in SIGNAL_KEYS}
    payload["signals_info"] = {"reference": None, "heldout_samples": 0, "proxy_samples": 0}
    return payload


def build_tracker(
    provider,
    model,
    device: str = "cpu",
    fisher_dir: Optional[Path] = None,
    batch_size: int = 64,
    cache=None,
) -> ForgettingSignalTracker:
    """Construct a tracker anchored at ``model``'s current weights."""
    return ForgettingSignalTracker(
        provider=provider,
        reference_state=float_state(model),
        reference_model=model,
        device=device,
        fisher_dir=fisher_dir,
        batch_size=batch_size,
        cache=cache,
    )


__all__ = [
    "PROXY_SET",
    "SIGNAL_KEYS",
    "ForgettingSignalTracker",
    "ReferencePredictions",
    "build_tracker",
    "compare_logits",
    "compare_to_reference",
    "empty_info",
    "float_state",
    "load_fisher",
    "reference_from_logits",
    "reference_predictions",
]
