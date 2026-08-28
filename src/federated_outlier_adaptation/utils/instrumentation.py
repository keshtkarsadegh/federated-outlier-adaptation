"""
Timing and communication-volume instrumentation for the federated runners.

All quantities produced here are reported as *new* keys of the run payload; the
existing accuracy structures are untouched.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional


def state_dict_bytes(state_dict: Mapping[str, Any]) -> int:
    """Number of bytes occupied by the floating-point tensors of a state dict."""
    total = 0
    for tensor in state_dict.values():
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if numel is None or element_size is None:
            continue
        total += int(numel()) * int(element_size())
    return total


def state_dict_params(state_dict: Mapping[str, Any]) -> int:
    """Number of scalar parameters in a state dict."""
    total = 0
    for tensor in state_dict.values():
        numel = getattr(tensor, "numel", None)
        if numel is not None:
            total += int(numel())
    return total


def round_communication_bytes(payload_bytes: int, num_clients: int) -> int:
    """
    Communication volume of one federated round.

    Each participating client receives the global model and returns its update,
    hence ``bytes x clients x 2``.
    """
    return int(payload_bytes) * int(num_clients) * 2


def communication_summary(
    bytes_up: Iterable[int], bytes_down: Iterable[int]
) -> dict[str, Any]:
    """
    Per-round and cumulative communication volume, split by direction.

    ``down`` is the global model shipped to the participants, ``up`` their
    returned updates.  The pre-existing ``comm_bytes_per_round`` is their sum
    and keeps its name and value.
    """
    up = [int(value) for value in bytes_up]
    down = [int(value) for value in bytes_down]
    return {
        "comm_bytes_up": up,
        "comm_bytes_down": down,
        "comm_bytes_up_total": sum(up),
        "comm_bytes_down_total": sum(down),
        "comm_bytes_total": sum(up) + sum(down),
    }


def summarise_timing(
    round_seconds: Iterable[float],
    client_seconds: Iterable[Iterable[float]],
) -> dict[str, Any]:
    """Aggregate per-round and per-client timings into a JSON-friendly dict."""
    rounds = [float(value) for value in round_seconds]
    clients = [[float(value) for value in per_round] for per_round in client_seconds]
    total = sum(rounds)
    return {
        "round_seconds": rounds,
        "client_seconds": clients,
        "total_seconds": total,
        "mean_round_seconds": (total / len(rounds)) if rounds else None,
    }


def device_name(device: str, gpu_name: Optional[str] = None) -> str:
    """Human-readable device label used in the payload (`cuda (NVIDIA A100)`)."""
    if gpu_name:
        return f"{device} ({gpu_name})"
    return device
