"""
The five-client study's design point, before anything runs on it.

Its whole value is that it differs from Study01 in ONE thing, so these tests
pin the things that must NOT differ - the old population, the classes, the
participation rule - as tightly as the one that must.
"""

from __future__ import annotations

import pytest

from federated_outlier_adaptation.training.study_config import (
    DIGITS_STUDY01,
    DIGITS_STUDY03,
    DROPOUT_RATE,
    STUDIES,
    config,
    participants,
)


def test_the_study_is_registered():
    assert config("Digits_study03") is DIGITS_STUDY03
    assert "Digits_study03" in STUDIES


def test_it_federates_five_clients_four_per_round():
    assert DIGITS_STUDY03.cohort_size == 5
    assert DIGITS_STUDY03.clients_per_round == 4
    assert DIGITS_STUDY03.clients_per_round == participants(5, DROPOUT_RATE)


def test_one_client_drops_and_that_is_the_same_rule_as_study01():
    """
    floor(0.9 x 5) = 4 and floor(0.9 x 10) = 9: one drops at either size, from
    one rule. If the two studies used different participation rules, a
    difference between them could be the rule rather than the size.
    """
    assert DIGITS_STUDY03.cohort_size - DIGITS_STUDY03.clients_per_round == 1
    assert DIGITS_STUDY01.cohort_size - DIGITS_STUDY01.clients_per_round == 1
    assert DIGITS_STUDY03.dropout == pytest.approx(0.2)
    assert DIGITS_STUDY01.dropout == pytest.approx(0.1)


def test_everything_except_the_federation_size_matches_study01():
    assert DIGITS_STUDY03.old_size == DIGITS_STUDY01.old_size == 200
    assert DIGITS_STUDY03.classes == DIGITS_STUDY01.classes == "digits"
    assert DIGITS_STUDY03.model == DIGITS_STUDY01.model
    assert DIGITS_STUDY03.cross_validated is DIGITS_STUDY01.cross_validated is True
    assert DIGITS_STUDY03.folds == DIGITS_STUDY01.folds == (1, 2, 3, 4, 5)


def test_a_client_is_one_writer():
    """A client is a writer, as in Study01: the cohort file names it that way."""
    assert DIGITS_STUDY03.cohort_file_name == "cohort_worst5.json"
    assert DIGITS_STUDY03.cohort_book_name == "cohort5"


def test_its_artefacts_cannot_collide_with_study01s():
    assert DIGITS_STUDY03.tag != DIGITS_STUDY01.tag
    assert DIGITS_STUDY03.cohort_file_name == "cohort_worst5.json"
    assert DIGITS_STUDY03.cohort_file_name != DIGITS_STUDY01.cohort_file_name
    assert DIGITS_STUDY03.cohort_book_name != DIGITS_STUDY01.cohort_book_name


def test_the_configuration_is_valid():
    DIGITS_STUDY03.check()


def test_describe_states_the_federation():
    described = DIGITS_STUDY03.describe()
    assert "5 outlier clients" in described
    assert "200 old writers" in described
    assert "4 of 5 per round" in described
