"""
Central path configuration for the project.

Every module resolves its input and output locations through this module so that
the code can run unchanged from a laptop, a workstation or a cluster job.  All
locations can be overridden with environment variables:

    FOA_REPO_ROOT     repository root (default: inferred from this file)
    FOA_DATA_DIR      raw/derived dataset root      (default: <repo>/data)
    FOA_RESULTS_DIR   experiment output root        (default: <repo>/results)
    FOA_CACHE_DIR     packed dataset cache          (default: <data>/cache)
    FOA_LOG_DIR       log files                     (default: <results>/logs)

The variables were named ``FAL_*`` in the earlier repository.  Every name is
still read under that prefix, so an existing environment keeps working: the
``FOA_*`` name wins, and the ``FAL_*`` name is used when it is unset.

The defaults reproduce the layout that the published results were produced with,
so an unconfigured checkout behaves exactly like before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Current environment-variable prefix and the one of the earlier repository.
ENV_PREFIX = "FOA_"
LEGACY_ENV_PREFIX = "FAL_"


def env_value(name: str) -> Optional[str]:
    """
    Value of the setting ``name`` from the environment.

    ``FOA_<name>`` is read first; when it is unset the legacy ``FAL_<name>`` of
    the earlier repository is used, so existing job scripts keep working.
    """
    value = os.environ.get(ENV_PREFIX + name)
    if value is None or value == "":
        value = os.environ.get(LEGACY_ENV_PREFIX + name)
    return value


def env_name(name: str) -> str:
    """Full environment-variable name of the setting ``name``."""
    return ENV_PREFIX + name


def _env_path(name: str, default: Path) -> Path:
    value = env_value(name)
    return Path(value).expanduser().resolve() if value else default


# src/federated_outlier_adaptation/config.py -> repository root
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

REPO_ROOT: Path = _env_path("REPO_ROOT", _DEFAULT_REPO_ROOT)
DATA_DIR: Path = _env_path("DATA_DIR", REPO_ROOT / "data")
RESULTS_DIR: Path = _env_path("RESULTS_DIR", REPO_ROOT / "results")
CACHE_DIR: Path = _env_path("CACHE_DIR", DATA_DIR / "cache")
LOG_DIR: Path = _env_path("LOG_DIR", RESULTS_DIR / "logs")

# --- NIST dataset locations -------------------------------------------------
BY_WRITE_DIR: Path = DATA_DIR / "by_write"
DIGITS_LABELS_JSON: Path = BY_WRITE_DIR / "digits_labels.json"

# --- frozen artefacts produced by the global-training phase ------------------
WRITER_SPLIT_JSON: Path = RESULTS_DIR / "writer_split.json"
GLOBAL_MODEL_PATH: Path = RESULTS_DIR / "global_model"
GLOBAL_RESULTS_DIR: Path = RESULTS_DIR / "global_results"
FISHER_DIR: Path = GLOBAL_RESULTS_DIR / "fisher"
OUTLIERS_DIR: Path = RESULTS_DIR / "outliers"
SELECTED_OUTLIERS_JSON: Path = OUTLIERS_DIR / "selected_outliers.json"
CLIENTS_ACC_ON_GLOBAL_JSON: Path = OUTLIERS_DIR / "clients_acc_on_global.json"
GLOBAL_CLIENTS_RESULTS_DIR: Path = RESULTS_DIR / "global_clients_results"
GLOBAL_CLIENTS_MODEL_PATH: Path = RESULTS_DIR / "global_clients_model"

# --- cache artefacts ---------------------------------------------------------
CACHE_ARRAY_NAME = "nist_digits_u8.npy"
CACHE_INDEX_NAME = "nist_digits_index.json"

# --- NIST resolution, character set and model --------------------------------
# The published pipeline is 128x128, ten classes, FlexibleCNN.  The v4 setting
# is the by-writer FEMNIST task: 28x28 via the EMNIST conversion, 62 classes and
# the FedAvg CNN.  All three are settings of the same provider so that a task
# line does not have to name them; a job script exports them once, exactly like
# the directory overrides above.
#
#     FOA_NIST28_DIR        packed 28x28 cache        (default <data>/nist28)
#     FOA_NIST_RESOLUTION   28 | 128                  (default 128)
#     FOA_NIST_CLASSES      all | digits              (default digits)
#     FOA_MODEL             fedavg_cnn | flexible_cnn (default fedavg_cnn)

NIST28_DIR: Path = _env_path("NIST28_DIR", DATA_DIR / "nist28")

#: Resolutions the NIST provider can be run at.
NIST_RESOLUTIONS = (28, 128)

#: Model topologies selectable per run.
MODEL_NAMES = ("fedavg_cnn", "flexible_cnn")


def nist_resolution() -> int:
    """Configured NIST resolution; 128 (the published one) when unset."""
    raw = env_value("NIST_RESOLUTION")
    if not raw:
        return 128
    value = int(raw)
    if value not in NIST_RESOLUTIONS:
        raise ValueError(
            f"{env_name('NIST_RESOLUTION')}={raw!r}: expected one of {NIST_RESOLUTIONS}"
        )
    return value


def nist_classes() -> str:
    """Configured NIST character set; ``digits`` (the published one) when unset."""
    raw = (env_value("NIST_CLASSES") or "digits").lower()
    if raw not in ("all", "digits"):
        raise ValueError(f"{env_name('NIST_CLASSES')}={raw!r}: expected 'all' or 'digits'")
    return raw


#: Folds of the cross-validation protocol.
FOLD_COUNT = 5


def fold() -> Optional[int]:
    """
    The cross-validation fold this run belongs to, or ``None``.

    ``None`` - the default - means "the single split every published run used",
    and every seed, split and output path is then exactly what it was.  A fold
    of 1..:data:`FOLD_COUNT` re-splits every client's own samples and re-seeds
    the run, so five folds are five independent draws over the *same* client
    population and the same shipped global model.
    """
    raw = env_value("FOLD")
    if not raw:
        return None
    value = int(raw)
    if not 1 <= value <= FOLD_COUNT:
        raise ValueError(
            f"{env_name('FOLD')}={raw!r}: expected a fold in 1..{FOLD_COUNT}"
        )
    return value


def fold_book() -> Optional[Path]:
    """
    The fold book this run reads its splits from, or ``None``.

    ``FOA_FOLD_BOOK`` names a file written by
    :func:`~federated_outlier_adaptation.data.fold_book.write_fold_book`.  With
    it and ``FOA_FOLD`` set, every client loader is built from the book's
    recorded indices instead of being re-derived from a seed - which is what
    makes one split reusable across processes, machines and months.  Without it
    nothing changes.
    """
    raw = env_value("FOLD_BOOK")
    return Path(raw).expanduser() if raw else None


def require_trainable() -> bool:
    """
    Whether a pool must exclude clients that cannot form a training split.

    Off by default, so the client populations of every existing run are what
    they were.  See ``Nist28Dataset.trainable_writers``.
    """
    raw = (env_value("REQUIRE_TRAINABLE") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def model_name() -> str:
    """Configured model topology; ``fedavg_cnn`` when unset."""
    raw = (env_value("MODEL") or "fedavg_cnn").lower()
    if raw not in MODEL_NAMES:
        raise ValueError(f"{env_name('MODEL')}={raw!r}: expected one of {MODEL_NAMES}")
    return raw

# --- additional datasets -----------------------------------------------------
# Both live under ``FOA_DATA_DIR/<name>/`` by default and can be relocated
# individually.  They are additive: no NIST path depends on them.
#
#     FOA_SHAKESPEARE_DIR   LEAF Shakespeare root   (default <data>/shakespeare)
#     FOA_CIFAR10_DIR       CIFAR-10 root           (default <data>/cifar10)

SHAKESPEARE_DIR: Path = _env_path("SHAKESPEARE_DIR", DATA_DIR / "shakespeare")
SHAKESPEARE_RAW_TXT: Path = SHAKESPEARE_DIR / "pg100.txt"
SHAKESPEARE_RAW_ZIP: Path = SHAKESPEARE_DIR / "1994-01-100.zip"
SHAKESPEARE_NPZ_NAME = "shakespeare.npz"
SHAKESPEARE_INDEX_NAME = "shakespeare_index.json"
SHAKESPEARE_NPZ: Path = SHAKESPEARE_DIR / SHAKESPEARE_NPZ_NAME
SHAKESPEARE_INDEX_JSON: Path = SHAKESPEARE_DIR / SHAKESPEARE_INDEX_NAME
SHAKESPEARE_RESULTS_DIR: Path = RESULTS_DIR / "shakespeare"

# The MNIST proxy set of the NIST provider lives next to them and follows the
# same shape (one array file plus one JSON index):
#
#     FOA_MNIST_DIR         MNIST proxy root        (default <data>/mnist)

MNIST_DIR: Path = _env_path("MNIST_DIR", DATA_DIR / "mnist")
MNIST_NPZ_NAME = "mnist.npz"
MNIST_INDEX_NAME = "mnist_index.json"
MNIST_NPZ: Path = MNIST_DIR / MNIST_NPZ_NAME
MNIST_INDEX_JSON: Path = MNIST_DIR / MNIST_INDEX_NAME

CIFAR10_DIR: Path = _env_path("CIFAR10_DIR", DATA_DIR / "cifar10")
CIFAR10_RAW_ARCHIVE: Path = CIFAR10_DIR / "cifar-10-python.tar.gz"
CIFAR10_NPZ_NAME = "cifar10.npz"
CIFAR10_CLIENTS_NAME = "cifar10_clients.json"
CIFAR10_NPZ: Path = CIFAR10_DIR / CIFAR10_NPZ_NAME
CIFAR10_CLIENTS_JSON: Path = CIFAR10_DIR / CIFAR10_CLIENTS_NAME
CIFAR10_RESULTS_DIR: Path = RESULTS_DIR / "cifar10"

#: Results root of every provider that is not the default NIST one.
PROVIDER_RESULTS_DIRS = {
    "nist": RESULTS_DIR,
    "shakespeare": SHAKESPEARE_RESULTS_DIR,
    "cifar10": CIFAR10_RESULTS_DIR,
}


def provider_results_dir(name: str = "nist") -> Path:
    """Results root of a dataset provider (``RESULTS_DIR`` for NIST)."""
    return PROVIDER_RESULTS_DIRS.get((name or "nist").lower(), RESULTS_DIR)


def results_dir() -> Path:
    """Return the (possibly reconfigured) results root."""
    return RESULTS_DIR


def outliers_file(k: int = 5) -> Path:
    """
    Path of the selected-outlier list for a given ``k``.

    ``k == 5`` keeps the published file name so existing runs are unaffected.
    """
    if k == 5:
        return SELECTED_OUTLIERS_JSON
    return OUTLIERS_DIR / f"selected_outliers_k{k}.json"


def ensure_dirs() -> None:
    """Create the writable output directories if they do not exist yet."""
    for path in (RESULTS_DIR, LOG_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
