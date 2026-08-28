"""
Run provenance.

Every experiment writes a ``config`` block next to its numbers so that a result
file on disk fully describes how it was produced: trainer class and the
hyperparameters actually used, scenario/aggregation, batch size, epochs, rounds,
seed, outlier list, checksums of the frozen artefacts, library versions, git
commit, host and timestamps.

The block is purely additive - existing keys of ``summary_*.json``,
``accuracies_*.json`` and ``accuracies_points_*.json`` keep their names and
formats.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from federated_outlier_adaptation import config

# Constructor arguments that describe the training objective.  Anything a
# trainer exposes under one of these names is recorded.
_HYPERPARAMETER_ATTRS = (
    "T",
    "alpha",
    "ewc_lambda",
    "lambda_prox",
    "lambda_consis",
    "beta",
    "learning_rate",
    "weight_decay",
)


def utc_now() -> str:
    """ISO-8601 UTC timestamp used for the start/end markers."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(repo_root: Optional[Path] = None) -> str:
    """Return the current git commit, or ``"unknown"`` outside a checkout."""
    root = Path(repo_root or config.REPO_ROOT)
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else "unknown"


def sha256_of(path: Optional[Path]) -> Optional[str]:
    """Hex SHA-256 of a file, or ``None`` when it is missing/unreadable."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:  # pragma: no cover - unreadable artefact
        return None
    return digest.hexdigest()


def torch_versions() -> dict[str, Any]:
    """Version and device information of the active torch installation."""
    info: dict[str, Any] = {"torch": None, "cuda": None, "device": "cpu", "gpu_name": None}
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return info
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        info["device"] = "cuda"
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver hiccup
            info["gpu_name"] = None
    return info


def trainer_hyperparameters(trainer: Any) -> dict[str, Any]:
    """
    Collect the hyperparameters a trainer instance is actually running with.

    Reads the well-known objective attributes plus every keyword argument of the
    trainer's ``__init__`` that survives as an attribute of the same name.
    """
    values: dict[str, Any] = {}
    names = list(_HYPERPARAMETER_ATTRS)
    try:
        signature = inspect.signature(type(trainer).__init__)
        names += [p for p in signature.parameters if p != "self"]
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        pass
    for name in names:
        if name in values or not hasattr(trainer, name):
            continue
        value = getattr(trainer, name)
        if isinstance(value, (int, float, str, bool)) or value is None:
            values[name] = value
    return values


def _writer_ids(outliers_file: Optional[Path]) -> Optional[list[str]]:
    if outliers_file is None:
        return None
    path = Path(outliers_file)
    if not path.is_file():
        return None
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):  # pragma: no cover - corrupt artefact
        return None
    return data if isinstance(data, list) else None


def build_run_config(
    *,
    trainer_name: str,
    trainer: Any = None,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    scenario: Optional[str] = None,
    metadata: Optional[str] = None,
    agg_method_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    max_round: Optional[int] = None,
    seed: Optional[int] = None,
    outliers_file: Optional[Path] = None,
    selected_writers: Optional[Iterable[str]] = None,
    single_outlier: Any = None,
    global_model_path: Optional[Path] = None,
    fisher_dir: Optional[Path] = None,
    parent_name: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the ``config`` block stored next to a run's numbers."""
    versions = torch_versions()
    writers = list(selected_writers) if selected_writers is not None else _writer_ids(outliers_file)

    block: dict[str, Any] = {
        "trainer": trainer_name,
        "trainer_kwargs": dict(trainer_kwargs or {}),
        "hyperparameters": trainer_hyperparameters(trainer) if trainer is not None else {},
        "scenario": scenario,
        "metadata": metadata,
        "agg_method_name": agg_method_name,
        "parent_name": parent_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_round": max_round,
        "seed": seed,
        "single_outlier": single_outlier,
        "outliers_file": str(outliers_file) if outliers_file else None,
        "outlier_writers": writers,
        "global_model_path": str(global_model_path) if global_model_path else None,
        "global_model_sha256": sha256_of(global_model_path),
        "fisher_dir": str(fisher_dir) if fisher_dir else None,
        "results_dir": str(config.RESULTS_DIR),
        "data_dir": str(config.DATA_DIR),
        "torch_version": versions["torch"],
        "cuda_version": versions["cuda"],
        "device": versions["device"],
        "gpu_name": versions["gpu_name"],
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "started_at": started_at,
        "finished_at": finished_at or utc_now(),
    }
    if extra:
        block.update(dict(extra))
    return block


class RunTimer:
    """Small helper recording wall-clock start/end of a run."""

    def __init__(self) -> None:
        self.started_at = utc_now()
        self._t0 = time.perf_counter()
        self.finished_at: Optional[str] = None
        self.seconds: Optional[float] = None

    def stop(self) -> float:
        self.seconds = time.perf_counter() - self._t0
        self.finished_at = utc_now()
        return self.seconds
