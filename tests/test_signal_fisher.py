"""
The signal tracker reads the Fisher the penalty reads.

A task line points the penalty at a per-fold Fisher directory, because the only
curvature that matches a fold's shipped model is that fold's own. The tracker
used to ignore it and look under an artefact convention nothing writes, so it
recorded ``fisher_available: false`` and wrote nulls for the two Fisher-weighted
signals in every run of the study while the penalty worked correctly. These
tests pin the agreement between the two.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import (
    BaseSequentialRunner,
)

RUNNERS = [BaseConcurrentRunner, BaseSequentialRunner]


class _Trainer:
    def __init__(self, fisher_path=None):
        self.fisher_path = fisher_path


class _Provider:
    def __init__(self, fisher_dir):
        self.fisher_dir = fisher_dir


def _runner(cls, trainer, provider):
    """A bare instance: the resolver must not need a trained runner."""
    runner = cls.__new__(cls)
    runner.trainer = trainer
    runner.provider = provider
    return runner


@pytest.mark.parametrize("cls", RUNNERS)
def test_an_explicit_fisher_directory_wins(cls, tmp_path):
    explicit = tmp_path / "g0_fold3" / "global_results" / "fisher"
    explicit.mkdir(parents=True)
    convention = tmp_path / "global_results" / "fisher"
    convention.mkdir(parents=True)
    runner = _runner(cls, _Trainer(str(explicit)), _Provider(convention))
    assert runner._signal_fisher_dir("g0") == explicit


@pytest.mark.parametrize("cls", RUNNERS)
def test_without_one_it_falls_back_to_the_convention(cls, tmp_path):
    convention = tmp_path / "global_results" / "fisher"
    convention.mkdir(parents=True)
    runner = _runner(cls, _Trainer(None), _Provider(convention))
    assert runner._signal_fisher_dir("") == convention


@pytest.mark.parametrize("cls", RUNNERS)
def test_a_named_path_that_does_not_exist_is_not_used(cls, tmp_path):
    """
    A stale or mistyped path must not be preferred over the convention.

    Preferring it would swap one silent miss for another: the tracker would read
    nothing and say nothing, which is the failure this fix exists to end.
    """
    convention = tmp_path / "global_results" / "fisher"
    convention.mkdir(parents=True)
    runner = _runner(cls, _Trainer(str(tmp_path / "absent")), _Provider(convention))
    assert runner._signal_fisher_dir("") == convention


@pytest.mark.parametrize("cls", RUNNERS)
def test_a_trainer_without_the_attribute_is_tolerated(cls, tmp_path):
    convention = tmp_path / "global_results" / "fisher"
    convention.mkdir(parents=True)
    runner = _runner(cls, object(), _Provider(convention))
    assert runner._signal_fisher_dir("") == convention
