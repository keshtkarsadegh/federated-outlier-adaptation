"""Preprocessing round-trips of the LEAF Shakespeare pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest

from federated_outlier_adaptation.data import shakespeare as shk

from .conftest import ROLE_LINES, SECOND_PLAY_TITLE, TOY_PLAY_TITLE


def test_vocabulary_is_leafs_eighty_symbol_alphabet():
    assert len(shk.ALL_LETTERS) == 80
    assert shk.NUM_LETTERS == 80
    assert len(set(shk.ALL_LETTERS)) == 80
    for char in "\n !\"&'(),-.0123456789:;>?":
        assert char in shk.ALL_LETTERS
    assert shk.ALL_LETTERS.endswith("}")


def test_normalisation_folds_typographic_punctuation():
    folded = shk.normalise_text("“Quoth he,” she said—‘truly’.\r\n")
    assert folded == '"Quoth he," she said-\'truly\'.\n'
    tokens, dropped = shk.encode(folded)
    assert dropped == 0
    assert "".join(shk.ALL_LETTERS[t] for t in tokens) == folded


def test_out_of_vocabulary_characters_are_dropped():
    tokens, dropped = shk.encode("ab¶c")
    assert dropped == 1
    assert "".join(shk.ALL_LETTERS[t] for t in tokens) == "abc"


def test_parse_plays_finds_roles_and_skips_front_matter(toy_corpus_text):
    plays, discarded = parse(toy_corpus_text)
    assert [title for title, _ in plays] == [TOY_PLAY_TITLE, SECOND_PLAY_TITLE]
    assert list(plays[0][1]) == ["ALPHA", "BETA"]
    assert list(plays[1][1]) == ["GAMMA", "DELTA"]
    assert plays[0][1]["ALPHA"] == ROLE_LINES["ALPHA"]
    # The poem, the dramatis personae and the stage direction are not roles.
    assert all("THE SONNETS" not in title for title, _ in plays)
    assert discarded > 0


def parse(text):
    return shk.parse_plays(text)


def test_user_ids_follow_leafs_naming(toy_corpus_text):
    users, streams, stats = shk.build_users(toy_corpus_text)
    assert users == [
        "A_TRIAL_PLAY_ALPHA",
        "A_TRIAL_PLAY_BETA",
        "ANOTHER_TRIAL_PLAY_GAMMA",
        "ANOTHER_TRIAL_PLAY_DELTA",
    ]
    assert stats["num_plays"] == 2
    assert stats["num_users"] == 4
    assert stats["num_sequences"] == sum(s.size - 80 for s in streams)


def test_role_text_collapses_whitespace():
    text = shk.role_text(["one   two", "three"])
    assert text == "one two three "


def test_prepared_dataset_round_trips(shakespeare_dir):
    data = shk.ShakespeareData(
        shakespeare_dir / "shakespeare.npz", shakespeare_dir / "shakespeare_index.json"
    )
    assert data.seq_length == 80
    assert data.num_classes == 80
    assert len(data) == 4
    assert data.num_sequences == sum(data.sample_count(u) for u in data.users)

    expected = shk.role_text(ROLE_LINES["ALPHA"])
    expected = "".join(char for char in expected if char in shk.ALL_LETTERS)
    assert data.user_text("A_TRIAL_PLAY_ALPHA") == expected

    x, y = data.sequences("A_TRIAL_PLAY_ALPHA")
    tokens = data.user_tokens("A_TRIAL_PLAY_ALPHA")
    assert x.shape == (tokens.size - 80, 80)
    assert y.shape == (tokens.size - 80,)
    np.testing.assert_array_equal(x[0], tokens[:80])
    np.testing.assert_array_equal(y[:5], tokens[80:85])
    np.testing.assert_array_equal(x[3], tokens[3:83])


def test_all_sequences_match_the_per_user_offsets(shakespeare_dir):
    data = shk.ShakespeareData(
        shakespeare_dir / "shakespeare.npz", shakespeare_dir / "shakespeare_index.json"
    )
    assert data.X.shape == (data.num_sequences, 80)
    assert data.y.shape == (data.num_sequences,)
    # Windows never straddle two users.
    for user in data.users:
        offsets = data.window_offsets(user)
        tokens = data.user_tokens(user)
        assert offsets.size == tokens.size - 80


def test_index_records_the_statistics(shakespeare_dir):
    with open(shakespeare_dir / "shakespeare_index.json") as handle:
        index = json.load(handle)
    assert index["dataset"] == "shakespeare"
    assert index["num_classes"] == 80
    assert index["vocabulary"] == shk.ALL_LETTERS
    assert index["num_users"] == len(index["users"]) == len(index["counts"])
    assert index["num_sequences"] == sum(index["counts"])
    assert index["source_sha256"]


def test_preparation_is_deterministic(tmp_path, toy_corpus_text):
    raw = tmp_path / "corpus.txt"
    raw.write_text(toy_corpus_text, encoding="utf-8")
    first = tmp_path / "a"
    second = tmp_path / "b"
    shk.prepare(raw_path=raw, out_dir=first, log=lambda *a: None)
    shk.prepare(raw_path=raw, out_dir=second, log=lambda *a: None)

    assert (first / "shakespeare.npz").read_bytes() == (second / "shakespeare.npz").read_bytes()
    assert (first / "shakespeare_index.json").read_text() == (
        second / "shakespeare_index.json"
    ).read_text()


def test_zip_input_is_read_without_extraction(tmp_path, toy_corpus_text):
    import zipfile

    archive = tmp_path / "corpus.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("100.txt", toy_corpus_text)

    assert shk.read_raw_text(archive) == shk.normalise_text(toy_corpus_text)
    assert not (tmp_path / "100.txt").exists()


def test_min_samples_filter_drops_short_roles(toy_corpus_text):
    users, _, _ = shk.build_users(toy_corpus_text, min_samples=10_000)
    assert users == []


def test_build_dataset_respects_the_nist_rates(shakespeare_dir):
    data = shk.ShakespeareData(
        shakespeare_dir / "shakespeare.npz", shakespeare_dir / "shakespeare_index.json"
    )
    user = data.users[0]
    total = data.sample_count(user)

    train, val, test = data.build_dataset(user, train_rate=0.6, eval_rate=0.2, batch_size=8)
    assert len(train.dataset) + len(val.dataset) + len(test.dataset) == total
    assert abs(len(train.dataset) / total - 0.6) < 0.1

    train, val, test = data.build_dataset(user, train_rate=0.0, eval_rate=0.0, batch_size=8)
    assert train is None and val is None
    assert len(test.dataset) == total


def test_unknown_users_yield_no_loaders(shakespeare_dir):
    data = shk.ShakespeareData(
        shakespeare_dir / "shakespeare.npz", shakespeare_dir / "shakespeare_index.json"
    )
    assert data.build_dataset(["no_such_role"]) == (None, None, None)


@pytest.mark.parametrize("batch_size", [4, 16])
def test_loader_yields_long_tensors(shakespeare_dir, batch_size):
    import torch

    data = shk.ShakespeareData(
        shakespeare_dir / "shakespeare.npz", shakespeare_dir / "shakespeare_index.json"
    )
    loader, _, _ = data.build_dataset(data.users[0], batch_size=batch_size)
    x, y = next(iter(loader))
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    assert x.shape[1] == 80
    assert x.shape[0] == y.shape[0] <= batch_size
    assert int(y.max()) < 80
