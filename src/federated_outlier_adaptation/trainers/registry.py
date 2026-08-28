"""
Name-based trainer lookup.

Experiments refer to trainers by class name (``"DistillationTrainer"``) so that
the name can travel through argument parsers and process pools.  The registry
walks the :mod:`federated_outlier_adaptation.trainers` package and returns
the matching class.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any, Mapping, Optional

from federated_outlier_adaptation.logging_utils import NistLogger
from federated_outlier_adaptation.trainers import artefacts

PACKAGE = "federated_outlier_adaptation.trainers"


def available_trainers() -> list[str]:
    """Names of every trainer class defined in the trainers package."""
    package = importlib.import_module(PACKAGE)
    found: list[str] = []
    for _, modname, is_pkg in pkgutil.walk_packages(package.__path__, prefix=PACKAGE + "."):
        if is_pkg:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:  # pragma: no cover - optional/broken module
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and name.endswith("Trainer"):
                found.append(name)
    return sorted(set(found))


def get_trainer_class(class_name: str):
    """
    Resolve a trainer class by name.

    Raises:
        ImportError: When no class of that name exists in the package.
    """
    package = importlib.import_module(PACKAGE)

    for _, modname, is_pkg in pkgutil.walk_packages(package.__path__, prefix=PACKAGE + "."):
        if is_pkg:
            continue
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if inspect.isclass(cls) and cls.__module__ == module.__name__:
            NistLogger.info("Found trainer class: {}".format(class_name))
            return cls

    raise ImportError(f"Class '{class_name}' not found under '{PACKAGE}'.")


def build_trainer(
    trainer_name: str,
    trainer_kwargs: Optional[Mapping[str, Any]] = None,
    provider: Any = None,
    global_name: str = artefacts.DEFAULT_GLOBAL_NAME,
):
    """
    Instantiate a trainer by name.

    Trainers keep all-default constructors, so ``build_trainer(name)`` reproduces
    the published configuration.  ``trainer_kwargs`` overrides individual
    hyperparameters without editing
    :mod:`federated_outlier_adaptation.constants`.

    Args:
        global_name: Name of the global model the run starts from.  For the
            default ``"global"`` nothing is injected and the trainers resolve
            their artefacts through the provider exactly as before.  For any
            other name the matching teacher checkpoint and Fisher directory are
            filled in for the trainers that accept them, so a run can be
            anchored on e.g. the federated pre-trained model.
    """
    cls = get_trainer_class(trainer_name)
    kwargs = dict(trainer_kwargs or {})
    parameters = inspect.signature(cls.__init__).parameters
    if provider is not None and "provider" in parameters:
        kwargs.setdefault("provider", provider)
    if provider is not None and not artefacts.is_default(global_name):
        if "global_model_path" in parameters:
            kwargs.setdefault(
                "global_model_path", str(artefacts.global_model_path(provider, global_name))
            )
        if "fisher_path" in parameters:
            kwargs.setdefault(
                "fisher_path", str(artefacts.fisher_dir(provider, global_name))
            )
    return cls(**kwargs)
