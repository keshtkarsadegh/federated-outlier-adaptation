"""
Provider registry.

``get_provider(name, **kwargs)`` is the single lookup every entry point uses to
choose a dataset.  This study has one: NIST SD19 by writer.  The indirection is
kept because the runners, trainers and aggregation rules are written against
the provider interface rather than against NIST, which is what made the
dataset replaceable in the first place - and what a reader has to be able to
check.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from federated_outlier_adaptation.providers.base import DatasetProvider


def _nist(**kwargs) -> DatasetProvider:
    from federated_outlier_adaptation.providers.nist import NistProvider, default_provider

    return NistProvider(**kwargs) if kwargs else default_provider()


_FACTORIES: Dict[str, Callable[..., DatasetProvider]] = {
    "nist": _nist,
}


def available_providers() -> List[str]:
    """Names accepted by :func:`get_provider`."""
    return sorted(_FACTORIES)


def get_provider(name: str = "nist", **kwargs) -> DatasetProvider:
    """
    Look up a dataset provider by name.

    Args:
        name: ``"nist"``.  An empty value means NIST.
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
