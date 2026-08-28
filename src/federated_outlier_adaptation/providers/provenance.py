"""
Provider-specific provenance.

:func:`provider_config_block` returns the keys a result file needs in order to
identify the dataset it was produced from and the model it was trained with:
the provider name, its results and data roots, the SHA-256 of every prepared
dataset file, and the topology, resolution, character set and parameter count
of the model.  It is passed to
:func:`~federated_outlier_adaptation.utils.provenance.build_run_config`
through the existing ``extra`` argument, so nothing in ``utils/provenance.py``
has to change and NIST runs that do not pass it keep their exact ``config``
block.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from federated_outlier_adaptation.utils.provenance import sha256_of


def provider_config_block(provider: Any) -> Dict[str, Any]:
    """Provenance keys describing the dataset a run was produced from."""
    if provider is None:
        return {}

    files: Dict[str, str] = {}
    try:
        files = dict(provider.dataset_files())
    except (AttributeError, TypeError, OSError):  # pragma: no cover - exotic provider
        files = {}

    hashes = {}
    for key, path in files.items():
        hashes[key] = {"path": str(path), "sha256": sha256_of(Path(path))}

    # The server-side proxy set is part of how a run was observed, so its
    # description and content hash belong in the record.  Providers without one
    # report an empty mapping and the key stays empty.
    proxy: Dict[str, Any] = {}
    try:
        proxy = dict(provider.proxy_info() or {})
    except (AttributeError, TypeError, OSError, ValueError):  # pragma: no cover
        proxy = {}

    # Which topology was trained, and how large it is.  A provider that does
    # not describe its model reports an empty mapping and the key stays empty.
    model: Dict[str, Any] = {}
    try:
        model = dict(provider.model_info() or {})
    except (AttributeError, TypeError, ValueError):  # pragma: no cover
        model = {}

    results_dir = getattr(provider, "results_dir", None)
    data_dir = getattr(provider, "data_dir", None)
    return {
        "provider": getattr(provider, "name", None),
        "provider_results_dir": str(results_dir) if results_dir else None,
        "provider_data_dir": str(data_dir) if data_dir else None,
        "dataset_files": hashes,
        "proxy_set": proxy,
        "model_info": model,
    }
