"""
Resolving the frozen artefacts of a *named* global model.

Every phase can be run against an alternative starting model - for instance the
federated pre-trained ``global_fl`` model next to the centrally trained
``global`` one.  The runners already take a ``global_name`` and load
``<results>/<global_name>_model``; the regularised trainers need the matching
teacher checkpoint and Fisher directory, which this module derives with the
same naming convention:

    global_name   teacher checkpoint          Fisher directory
    ------------  --------------------------  ---------------------------------
    global        <results>/global_model      <results>/global_results/fisher
    global_fl     <results>/global_fl_model   <results>/global_results/fisher_fl
    <name>        <results>/<name>_model      <results>/global_results/fisher_<name>

For the default ``global`` the provider's own properties are returned verbatim,
so nothing about the published runs changes.
"""

from __future__ import annotations

from pathlib import Path

#: The name of the centrally trained model of the published pipeline.
DEFAULT_GLOBAL_NAME = "global"


def is_default(global_name) -> bool:
    """Whether ``global_name`` refers to the published global model."""
    return not global_name or global_name == DEFAULT_GLOBAL_NAME


def results_root(provider) -> Path:
    from federated_outlier_adaptation import config

    return Path(getattr(provider, "results_dir", config.RESULTS_DIR))


def fisher_tag(global_name: str) -> str:
    """Suffix of the Fisher directory belonging to a named global model."""
    prefix = DEFAULT_GLOBAL_NAME + "_"
    return global_name[len(prefix):] if global_name.startswith(prefix) else global_name


def global_model_path(provider, global_name: str = DEFAULT_GLOBAL_NAME) -> Path:
    """Teacher/anchor checkpoint of a named global model."""
    if is_default(global_name):
        return Path(provider.global_model_path)
    return results_root(provider) / f"{global_name}_model"


def fisher_dir(provider, global_name: str = DEFAULT_GLOBAL_NAME) -> Path:
    """Fisher directory belonging to a named global model."""
    if is_default(global_name):
        return Path(provider.fisher_dir)
    return results_root(provider) / "global_results" / f"fisher_{fisher_tag(global_name)}"
