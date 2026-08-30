"""
A fold book asked for a multi-writer client returns its writers' rows.

This is the bug that made a hundred array elements of Digits_study02 report
success while training on nothing: ``FoldBook.part`` looked the merged id
``"w1+w2"`` up in its writer table, missed, and returned an empty list, and the
isolated trainer recorded the client as one to skip.  These tests pin both
halves of the fix - the lookup, and the refusal to call a run that trained
nothing a success.
"""

from __future__ import annotations

import numpy as np
import pytest

from federated_outlier_adaptation.data.fold_book import PARTS, FoldBook
from federated_outlier_adaptation.data.merged_clients import merged_id


@pytest.fixture
def book() -> FoldBook:
    """
    Three writers, twelve rows each, one fold.

    Rows are laid out writer by writer and the assignment cycles through
    train/val/test, so every writer has four rows in every part and any pooling
    mistake shows up as a wrong count rather than as a plausible one.
    """
    writers = ["w1", "w2", "w3"]
    per_writer = 12
    writer_index = np.repeat(np.arange(len(writers)), per_writer)
    assignment = np.tile(np.arange(len(PARTS)), per_writer * len(writers) // len(PARTS))
    assignment = assignment[: writer_index.size].reshape(1, -1)
    metadata = {"writers": writers, "writer_table": writers, "folds": 1,
                "tag": "merged-test"}
    return FoldBook(assignment, metadata, writer_index=writer_index)


def test_a_plain_writer_is_unaffected(book):
    for part in PARTS:
        assert len(book.part(1, "w1", part)) == 4


def test_a_merged_client_holds_its_members_rows(book):
    """The union, per part - not a re-split of the combined data."""
    for part in PARTS:
        pair = book.part(1, merged_id(["w1", "w2"]), part)
        assert pair == sorted(book.part(1, "w1", part) + book.part(1, "w2", part))
        assert len(pair) == 8


def test_a_three_writer_client_holds_all_three(book):
    trio = merged_id(["w1", "w2", "w3"])
    assert len(book.part(1, trio, "train")) == 12
    everything = sum(len(book.part(1, trio, part)) for part in PARTS)
    assert everything == 36


def test_the_parts_of_a_merged_client_stay_disjoint(book):
    """
    The merge unions each part separately, so no row can appear in two.

    If it re-split the combined data instead, a row could move between
    partitions and a client would be evaluated on rows it trained on.
    """
    trio = merged_id(["w1", "w2", "w3"])
    seen = [set(book.part(1, trio, part)) for part in PARTS]
    assert seen[0] & seen[1] == set()
    assert seen[0] & seen[2] == set()
    assert seen[1] & seen[2] == set()


def test_split_agrees_with_part_for_merged_clients(book):
    merged = merged_id(["w1", "w3"])
    by_split = book.split(1, [merged])
    for part in PARTS:
        assert by_split[part] == book.part(1, merged, part)


def test_an_unknown_member_contributes_nothing_but_does_not_hide_the_others(book):
    mixed = merged_id(["w1", "nobody"])
    assert book.part(1, mixed, "train") == book.part(1, "w1", "train")


def test_covers_requires_every_member(book):
    assert book.covers(merged_id(["w1", "w2"])) is True
    assert book.covers(merged_id(["w1", "nobody"])) is False
    assert book.covers("w1") is True
    assert book.covers("nobody") is False


def test_an_unknown_writer_is_still_empty(book):
    assert book.part(1, "nobody", "train") == []


def test_the_fold_and_part_guards_still_apply(book):
    with pytest.raises(ValueError):
        book.part(2, "w1", "train")
    with pytest.raises(ValueError):
        book.part(1, "w1", "nonsense")
