"""
Evaluation tensors materialised once per run.

Every round of a federated run scores the current global model on the same
fixed sets: the source split, the pool's in-sample data, the pool's held-out
validation and test halves, every client on its own data, and the server-side
proxy set.  Those sets never change during a run, yet the published loop rebuilt
them batch by batch through single-process ``DataLoader``s, decoding one image
at a time from the packed cache on the CPU.  For the main setting that is around
55 000 images per round through a Python loop, and it dominates the wall clock
once the client updates run on an accelerator.

This module materialises each set **once**, in its compact storage form
(``uint8`` images, ``int64`` token windows), keeps it on the accelerator when
there is room for it, and turns a round's evaluation into one batched forward
pass per set.  The per-sample transform of the dataset is replaced by its
batched equivalent, which is chosen so that the produced float tensors are the
same ones the ``DataLoader`` produced:

``grayscale_u8``
    ``ToTensor`` followed by ``Normalize((0.5,), (0.5,))``, i.e. exactly the tail
    of :func:`~federated_outlier_adaptation.data.datasets.build_transform`.  The
    resize is applied while the image is still ``uint8``, by the very same
    ``Resize`` object, so it cannot drift either.
``identity``
    the payload is already what the model consumes (token windows, or a dataset
    the fallback path had to expand into float tensors).

A set that cannot be materialised - an exotic dataset, an augmenting one, a
transform that is not the project's - simply reports failure, and the caller
keeps using its ``DataLoader``.  Nothing in this module changes what is
computed; it only changes how often the data is decoded.

Placement
---------
The payload goes to the accelerator only when the free memory is at least
:data:`MEMORY_HEADROOM` times its size, so a shared GPU cannot be pushed into an
out-of-memory error by the evaluation cache.  Otherwise the tensors stay in
host memory and are copied batch by batch, which still removes the per-sample
decode and the loader overhead.

Reuse within a round
--------------------
:meth:`GpuEvalCache.bind` marks the start of a round and drops the memoised
logits.  Every consumer of a set - pooled accuracy, per-client accuracy, the
retention/agreement/KL signals - then reads the *same* forward pass, so a round
costs one pass per set no matter how many numbers are derived from it.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, Subset

#: Batch size of a cached evaluation pass.  Large batches are what makes the
#: pass cheap; the numbers are unaffected because evaluation has no batch
#: statistics (the models are in ``eval`` mode).
DEFAULT_EVAL_BATCH = 512

#: Free accelerator memory required, as a multiple of a payload's size, before
#: that payload is placed on the accelerator.
MEMORY_HEADROOM = 2.0

#: Where the per-round evaluations read their data from.
EVAL_PATHS = ("cache", "loader")

#: Default of ``--eval-path``.  The cache reproduces the loader numbers, so it
#: is the path a run takes unless the loader path is asked for explicitly.
DEFAULT_EVAL_PATH = "cache"

#: Names of the evaluation sets one run materialises.  The source split and the
#: pool's in-sample data carry the two published accuracy series; the validation
#: and test halves carry the held-out series and the forgetting signals; the
#: per-client set exists only when the run tracks its clients.
SOURCE_TEST_SET = "source_test"
SOURCE_VAL_SET = "source_val"
POOL_INSAMPLE_SET = "pool_insample"
POOL_CLIENTS_SET = "pool_clients"
POOL_VAL_SET = "pool_val"
POOL_TEST_SET = "pool_test"
PROXY_SET = "proxy"


class Decoder:
    """
    Batched equivalent of a dataset's per-sample transform.

    Two decoders are interchangeable when their ``key`` matches, which is what
    lets the materialiser concatenate the parts of a ``ConcatDataset``.
    """

    __slots__ = ("key", "function")

    def __init__(self, key: str, function: Callable[[torch.Tensor], torch.Tensor]):
        self.key = str(key)
        self.function = function

    def __call__(self, payload: torch.Tensor) -> torch.Tensor:
        return self.function(payload)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Decoder) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Decoder({self.key!r})"


def _as_is(payload: torch.Tensor) -> torch.Tensor:
    return payload


def _grayscale_u8(payload: torch.Tensor) -> torch.Tensor:
    """``ToTensor`` + ``Normalize((0.5,), (0.5,))`` of a ``[N, H, W]`` uint8 batch."""
    tensor = payload.unsqueeze(1).float().div(255.0)
    return tensor.sub_(0.5).div_(0.5)


#: The payload is already the model's input.
IDENTITY = Decoder("identity", _as_is)

#: Single-channel images stored as ``uint8``; the project's image transform.
GRAYSCALE_U8 = Decoder("grayscale_u8", _grayscale_u8)


# --------------------------------------------------------------------------- #
# materialisation
# --------------------------------------------------------------------------- #
def materialise(dataset) -> Optional[tuple[torch.Tensor, torch.Tensor, Decoder]]:
    """
    Expand a dataset into ``(payload, labels, decoder)``, or ``None``.

    ``Subset`` and ``ConcatDataset`` are unwrapped, a dataset that implements
    ``eval_payload()`` supplies its compact form, and anything else is expanded
    sample by sample into float tensors - one pass, still far cheaper than one
    pass per round.  ``None`` means "this dataset cannot be cached", which the
    callers answer by keeping their loader.
    """
    if dataset is None:
        return None

    if isinstance(dataset, Subset):
        inner = materialise(dataset.dataset)
        if inner is None:
            return None
        payload, labels, decoder = inner
        index = torch.as_tensor(list(dataset.indices), dtype=torch.long)
        return payload.index_select(0, index), labels.index_select(0, index), decoder

    if isinstance(dataset, ConcatDataset):
        parts = [materialise(part) for part in dataset.datasets]
        if any(part is None for part in parts):
            return None
        parts = [part for part in parts if part[0].shape[0]]
        if not parts:
            return None
        decoder = parts[0][2]
        if any(part[2] != decoder for part in parts):
            return None
        return (
            torch.cat([part[0] for part in parts]),
            torch.cat([part[1] for part in parts]),
            decoder,
        )

    hook = getattr(dataset, "eval_payload", None)
    if callable(hook):
        try:
            result = hook()
        except (OSError, ValueError, RuntimeError, KeyError):  # pragma: no cover - broken artefact
            result = None
        if result is not None:
            return result

    return expand_samples(dataset)


def expand_samples(dataset) -> Optional[tuple[torch.Tensor, torch.Tensor, Decoder]]:
    """
    Fallback materialisation: read every sample once and stack the results.

    Used for datasets without a compact form.  The tensors are exactly the ones
    the loader would have produced, so the numbers are unchanged; only the
    storage is float instead of the dataset's compact type.
    """
    try:
        size = len(dataset)
    except (TypeError, AttributeError):  # pragma: no cover - not a map dataset
        return None
    if size == 0:
        return None
    inputs: list[torch.Tensor] = []
    labels: list[int] = []
    try:
        for index in range(size):
            sample = dataset[index]
            inputs.append(torch.as_tensor(sample[0]))
            labels.append(int(sample[1]))
    except (OSError, ValueError, RuntimeError, TypeError):  # pragma: no cover - exotic dataset
        return None
    return torch.stack(inputs), torch.tensor(labels, dtype=torch.long), IDENTITY


class CachedSet:
    """One materialised evaluation set and the client segments inside it."""

    __slots__ = ("payload", "labels", "decoder", "segments", "on_device")

    def __init__(self, payload, labels, decoder, segments, on_device):
        self.payload = payload
        self.labels = labels
        self.decoder = decoder
        #: ``(client id, start, stop)`` triples, in the order the set was built.
        self.segments = list(segments)
        self.on_device = bool(on_device)

    @property
    def size(self) -> int:
        return int(self.payload.shape[0])


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
class GpuEvalCache:
    """
    The evaluation sets of one run, materialised once and scored in one pass.

    Args:
        device: Device the forward passes run on.
        batch_size: Batch size of a cached pass.
        headroom: Free-memory multiple required before a payload is placed on
            the accelerator.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = DEFAULT_EVAL_BATCH,
        headroom: float = MEMORY_HEADROOM,
    ):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.headroom = float(headroom)
        self.sets: dict[str, CachedSet] = {}
        self._model = None
        self._logits: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ build
    def add(self, name: str, dataset) -> bool:
        """Materialise one dataset as the set ``name``; ``False`` when it cannot be."""
        return self.add_segments(name, [(None, dataset)])

    def add_segments(self, name: str, segments: Sequence[tuple[Optional[str], Any]]) -> bool:
        """
        Materialise a set built from per-client parts.

        ``segments`` are ``(client id, dataset)`` pairs; the client ids let one
        pass produce both the pooled number and the per-client ones.  An empty
        or unmaterialisable part makes the whole set fail, so the caller falls
        back to its loaders for that set and for nothing else.
        """
        if name in self.sets:
            return True
        payloads: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        spans: list[tuple[Optional[str], int, int]] = []
        decoder: Optional[Decoder] = None
        offset = 0
        for client_id, dataset in segments:
            if dataset is None:
                continue
            part = materialise(dataset)
            if part is None:
                return False
            payload, part_labels, part_decoder = part
            if decoder is None:
                decoder = part_decoder
            elif part_decoder != decoder:
                return False
            count = int(payload.shape[0])
            if count == 0:
                continue
            payloads.append(payload)
            labels.append(part_labels)
            spans.append((client_id, offset, offset + count))
            offset += count
        if not payloads or decoder is None:
            return False

        stacked = payloads[0] if len(payloads) == 1 else torch.cat(payloads)
        stacked_labels = labels[0] if len(labels) == 1 else torch.cat(labels)
        stacked_labels = stacked_labels.to(torch.long)
        stacked, on_device = self._place(stacked)
        self.sets[name] = CachedSet(
            payload=stacked,
            labels=stacked_labels.to(self.device),
            decoder=decoder,
            segments=spans,
            on_device=on_device,
        )
        return True

    def _place(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Move a payload to the accelerator when there is comfortable room."""
        if self.device.type != "cuda":
            return tensor, False
        needed = tensor.numel() * tensor.element_size()
        try:
            # ``mem_get_info`` needs an indexed device; ``torch.device("cuda")`` has none.
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            free, _total = torch.cuda.mem_get_info(index)
        except (RuntimeError, AssertionError, AttributeError, ValueError):  # pragma: no cover - old driver
            free = 0
        if free < needed * self.headroom:
            return tensor.pin_memory() if tensor.is_cpu else tensor.cpu(), False
        try:
            return tensor.to(self.device), True
        except torch.cuda.OutOfMemoryError:  # pragma: no cover - racing allocation
            return tensor, False

    # ------------------------------------------------------------------ query
    def has(self, name: str) -> bool:
        return name in self.sets

    def size(self, name: str) -> int:
        entry = self.sets.get(name)
        return entry.size if entry is not None else 0

    def clients(self, name: str) -> list[str]:
        entry = self.sets.get(name)
        return [] if entry is None else [cid for cid, _, _ in entry.segments if cid is not None]

    def summary(self) -> dict:
        """Sizes and placement of every set, for the run's provenance block."""
        return {
            "sets": {name: entry.size for name, entry in sorted(self.sets.items())},
            "on_device": sorted(name for name, entry in self.sets.items() if entry.on_device),
            "device": str(self.device),
            "batch_size": self.batch_size,
        }

    # ----------------------------------------------------------------- rounds
    @property
    def bound(self):
        """The model the memoised passes belong to, or ``None``."""
        return self._model

    def bind(self, model) -> None:
        """Start a new round: score ``model`` and drop the memoised passes."""
        self._model = model
        self._logits.clear()

    def logits(self, name: str) -> torch.Tensor:
        """
        Logits of the bound model on ``name``, computed once per round.

        Every quantity derived from a set reads this tensor, so a round costs
        one forward pass per set.
        """
        cached = self._logits.get(name)
        if cached is not None:
            return cached
        entry = self.sets[name]
        model = self._model
        if model is None:  # pragma: no cover - guarded by the callers
            raise RuntimeError("No model is bound to the evaluation cache.")
        target = _model_device(model)
        model.eval()
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, entry.size, self.batch_size):
                payload = entry.payload[start : start + self.batch_size]
                if payload.device != target:
                    payload = payload.to(target, non_blocking=True)
                output = model(entry.decoder(payload))
                if isinstance(output, (tuple, list)):  # models returning (logits, features)
                    output = output[0]
                chunks.append(output.detach().float())
        result = torch.cat(chunks) if chunks else torch.empty(0)
        self._logits[name] = result
        return result

    def labels(self, name: str) -> torch.Tensor:
        """Ground-truth labels of a set, on the device the logits live on."""
        entry = self.sets[name]
        logits = self.logits(name)
        return entry.labels.to(logits.device)

    def counts(self, name: str) -> tuple[int, int]:
        """``(correct, total)`` of the bound model on a set."""
        logits = self.logits(name)
        if logits.numel() == 0:
            return 0, 0
        correct = int(logits.argmax(dim=1).eq(self.labels(name)).sum().item())
        return correct, int(logits.shape[0])

    def accuracy(self, name: str) -> Optional[float]:
        """Accuracy of the bound model on a set, or ``None`` when it is empty."""
        correct, total = self.counts(name)
        return (correct / total) if total else None

    def segment_accuracies(self, name: str) -> dict[str, float]:
        """
        ``{client: accuracy}`` from the one pass over ``name``.

        The per-client value is ``correct / count`` over that client's slice,
        i.e. exactly what a per-client loader would have produced.
        """
        entry = self.sets[name]
        logits = self.logits(name)
        labels = self.labels(name)
        correct = logits.argmax(dim=1).eq(labels)
        accuracies: dict[str, float] = {}
        for client_id, start, stop in entry.segments:
            if client_id is None or stop <= start:
                continue
            accuracies[client_id] = float(
                int(correct[start:stop].sum().item()) / (stop - start)
            )
        return accuracies

    def segment_sizes(self, name: str) -> dict[str, int]:
        """``{client: sample count}`` of a set's segments."""
        entry = self.sets.get(name)
        if entry is None:
            return {}
        return {
            client_id: stop - start
            for client_id, start, stop in entry.segments
            if client_id is not None
        }


def _model_device(model) -> torch.device:
    """Device a model's parameters live on; ``cpu`` for a parameterless module."""
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():  # pragma: no cover - parameterless model
        return buffer.device
    return torch.device("cpu")  # pragma: no cover - parameterless model


def resolve_eval_path(eval_path: Optional[str]) -> str:
    """Validate ``--eval-path``; ``None`` selects the default."""
    if eval_path is None:
        return DEFAULT_EVAL_PATH
    value = str(eval_path).lower()
    if value not in EVAL_PATHS:
        raise ValueError(f"Unknown eval path {eval_path!r}; expected one of {EVAL_PATHS}")
    return value


def make_cache(
    eval_path: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_EVAL_BATCH,
) -> Optional[GpuEvalCache]:
    """
    The evaluation cache of one run, or ``None`` on the loader path.

    ``eval_path="loader"`` reproduces the published per-round loader passes and
    is what the equality check compares against.
    """
    if resolve_eval_path(eval_path) == "loader":
        return None
    return GpuEvalCache(device=device, batch_size=batch_size)


__all__ = [
    "DEFAULT_EVAL_BATCH",
    "DEFAULT_EVAL_PATH",
    "EVAL_PATHS",
    "GRAYSCALE_U8",
    "IDENTITY",
    "MEMORY_HEADROOM",
    "POOL_CLIENTS_SET",
    "POOL_INSAMPLE_SET",
    "POOL_TEST_SET",
    "POOL_VAL_SET",
    "PROXY_SET",
    "SOURCE_TEST_SET",
    "SOURCE_VAL_SET",
    "CachedSet",
    "Decoder",
    "GpuEvalCache",
    "expand_samples",
    "make_cache",
    "materialise",
    "resolve_eval_path",
]
