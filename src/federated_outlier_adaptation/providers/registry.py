"""
Provider registry.

``get_provider(name, **kwargs)`` is the single lookup every entry point uses to
choose a dataset.  ``"nist"`` (the default) returns the process-wide default
provider when called without arguments, exactly as before, so unconfigured runs
keep reproducing the published behaviour and output paths.

The dataset modules are imported lazily: loading the registry must not pull the
Shakespeare or CIFAR-10 arrays into a NIST run.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from federated_outlier_adaptation.providers.base import DatasetProvider


def _nist(**kwargs) -> DatasetProvider:
    from federated_outlier_adaptation.providers.nist import NistProvider, default_provider

    return NistProvider(**kwargs) if kwargs else default_provider()


def _shakespeare(**kwargs) -> DatasetProvider:
    from federated_outlier_adaptation.providers.shakespeare import ShakespeareProvider

    return ShakespeareProvider(**kwargs)


def _cifar10(**kwargs) -> DatasetProvider:
    from federated_outlier_adaptation.providers.cifar10 import Cifar10Provider

    return Cifar10Provider(**kwargs)


_FACTORIES: Dict[str, Callable[..., DatasetProvider]] = {
    "nist": _nist,
    "shakespeare": _shakespeare,
    "cifar10": _cifar10,
}


def available_providers() -> List[str]:
    """Names accepted by :func:`get_provider`."""
    return sorted(_FACTORIES)


def get_provider(name: str = "nist", **kwargs) -> DatasetProvider:
    """
    Look up a dataset provider by name.

    Args:
        name: ``"nist"``, ``"shakespeare"`` or ``"cifar10"``.  An empty value
            means NIST.
        **kwargs: Forwarded to the provider constructor (``results_dir``,
            ``data_dir``, ``cache_dir``, ``outliers_file``, ...).
    """
    key = (name or "nist").lower()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown dataset provider: {name!r} (known: {', '.join(available_providers())})"
        )
    return factory(**kwargs)
