"""
The transfer study: the digit winners, unchanged, on a different modality.

Two things have to be true for this study to mean anything, and neither is
visible in a result.

The first is that **nothing was retuned**. If a coefficient were quietly
adjusted to suit the LSTM, a good score here would say only that the method can
be tuned twice - which was never in doubt. So the emitted flags are compared
against the digit study's cell table directly, and any divergence fails.

The second is that the **split is the book's**. The digit study got that for
free because its dataset consults the fold book; Shakespeare's did not, so g-0
would have trained on a seeded cut and been scored on a booked one, and the two
overlap. A preservation number measured on rows the model trained on is not a
preservation number.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from federated_outlier_adaptation.training import agg_cells, five_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import (
    DIGITS_STUDY01,
    SHAKESPEARE_STUDY01 as CFG,
    participants,
)

TOOLS = str(Path(__file__).resolve().parents[2] / "tools")


@pytest.fixture()
def study():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_shakespeare_study

    return make_shakespeare_study


@pytest.fixture()
def lines(study):
    out = {}
    for name, (builder, _) in study.STAGES.items():
        out[name] = builder()
    return out


# --------------------------------------------------------------------------- #
# the design point
# --------------------------------------------------------------------------- #
def test_the_study_is_the_transfer_design_point():
    assert CFG.provider == "shakespeare"
    assert (CFG.cohort_size, CFG.old_size) == (10, 100)
    assert CFG.clients_per_round == participants(10, 0.1) == 9
    assert CFG.cross_validated is False
    assert CFG.folds == (1,)
    assert DIGITS_STUDY01.cross_validated is True
    assert DIGITS_STUDY01.folds == (1, 2, 3, 4, 5)


def test_the_two_studies_address_different_roots():
    assert CFG.tag != DIGITS_STUDY01.tag
    assert abs(CFG.seed_base - DIGITS_STUDY01.seed_base) >= 50_000


def test_the_dataset_flags_follow_the_provider():
    """Image flags on a text run would be noise at best and a contradiction at worst."""
    assert SL.setting(CFG) == "--provider shakespeare"
    assert SL.setting(DIGITS_STUDY01) == "--resolution 28 --classes digits"


def test_the_books_are_still_cut_with_five_folds(study):
    """
    Training uses fold 1; preservation is still the mean over five old-fold test
    partitions, which is the only reason that column is comparable to the digit
    study's.
    """
    assert study.FOLDS_BUILT == 5
    assert study.FOLD == 1
    for line in study.population_lines() + study.pools_lines() + study.cohort_lines():
        if line.startswith("foa fold-book "):
            assert " --folds 5 " in line, line


# --------------------------------------------------------------------------- #
# the transfer claim
# --------------------------------------------------------------------------- #
def test_the_winners_carry_the_digit_hyperparameters_unchanged(study):
    """
    The claim of the whole study. Every coefficient is compared with the digit
    study's own cell table; a retune anywhere fails here.
    """
    aggs = {c["id"]: c for c in agg_cells.screen_cells()}
    regs = {c["id"]: c for c in reg_cells.screen_cells()}
    emitted = [ln for ln in study.runs_lines() if ln.startswith("foa final ")]
    assert len(emitted) == 4, "three winners plus the plain-FedAvg baseline"

    for cell in five_cells.configs():
        if cell["regulariser"] is None:
            continue
        line = next(ln for ln in emitted if f" --parent s01_{cell['id']} " in ln)
        agg, reg = aggs[cell["aggregation"]], regs[cell["regulariser"]]
        assert f" --aggregation {agg['rule']} " in line
        for token in SL._agg_flags(agg["flags"]).split():
            assert token in line, (cell["id"], token)
        penalty = SL._reg_set(reg)
        if penalty:
            assert penalty in line, cell["id"]
        assert f" --trainer {reg['trainer']} " in line


def test_the_protocol_is_the_digit_studys(study):
    """Same horizon, same local budget, same batch - or it is not a transfer."""
    for line in [ln for ln in study.runs_lines() if ln.startswith("foa final ")]:
        assert " --rounds 100 " in line
        assert " --epochs 5 " in line
        assert " --batch-size 64 " in line
        assert " --clients-per-round 9 " in line
        assert " --old-fold all" in line
        assert " --init global --global-name g0" in line
        assert " --fold 1 " in line


def test_no_stage_needs_a_fisher(study, lines):
    """
    None of the three winners uses a Fisher-weighted penalty, so this study
    never reads $G0_FOLD - and a task that named it would be refused by the
    runner for a study that has no g0_selection to read it from.
    """
    for stage, body in lines.items():
        for line in body:
            assert "fisher_path" not in line, stage
            assert "G0_FOLD" not in line, stage


def test_the_control_runs_both_schedules_in_one_task(study):
    plain = [ln for ln in study.runs_lines() if " --parent s01_fl_global " in ln]
    assert len(plain) == 1
    assert " --aggregation fedavg " in plain[0]
    assert "--set " not in plain[0]
    assert " --trainer BaseTrainer " in plain[0]


def test_the_centralized_arm_runs_both_initialisations(study):
    arms = [ln for ln in study.runs_lines() if ln.startswith("foa isolated-train ")]
    assert len(arms) == 2
    scratch = next(ln for ln in arms if " --init scratch" in ln)
    assert "--init-checkpoint" not in scratch, (
        "a scratch arm that is handed the shipped model is not a scratch arm"
    )
    assert "--init-checkpoint" in next(ln for ln in arms if " --init global" in ln)


# --------------------------------------------------------------------------- #
# the task files
# --------------------------------------------------------------------------- #
def test_every_foa_line_parses(lines):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    seen = 0
    for stage, body in lines.items():
        for line in body:
            if not line.startswith("foa "):
                continue
            parser.parse_args(
                shlex.split(line.replace("$FOA_STUDY_DIR", "/study"))[1:]
            )
            seen += 1
    assert seen >= 20


def test_no_step_is_inline_python(lines, study):
    """
    The invariant that replaced six ``python -c`` one-liners.

    One of them read a key (``counts``) that no writer-counts artefact has ever
    had. Nothing could catch it: an inline one-liner is a string until it runs,
    so the suite never saw it and the chain died on a GPU one stage in. Every
    step is now a command whose behaviour a test can call directly.
    """
    for stage, body in lines.items():
        for line in body:
            assert line.startswith("foa "), (stage, line[:70])
            assert " -c " not in line, ("inline python is back", stage, line[:70])


def test_the_artefact_readers_are_commands_with_tests_behind_them(lines, study):
    """Each of the six former one-liners, by the command that replaced it."""
    joined = "\n".join(sum(lines.values(), []))
    assert joined.count("foa select-eligible ") == 1
    assert joined.count("foa check-population ") == 3
    assert joined.count("foa promote-model ") == 2
    # And they read the key the producer actually writes.
    from federated_outlier_adaptation.outliers.eligibility import TOTALS_KEY

    assert TOTALS_KEY == "per_writer_total"
    assert "counts.json" in joined, "the census artefact is still consumed"


def test_every_path_stays_inside_the_study(lines):
    """The runner refuses anything else, and a refusal in a %1 chain stops it."""
    flags = {
        "--results-dir", "--out", "--model-path", "--fold-book", "--clients-file",
        "--writers-file", "--exclude-file", "--old-book", "--old-clients-file",
        "--outliers-file", "--init-checkpoint", "--root", "--scores", "--csv",
    }
    for stage, body in lines.items():
        for line in body:
            if not line.startswith("foa "):
                continue
            words = line.split()
            for prev, word in zip(words, words[1:]):
                if prev in flags and not word.startswith("-"):
                    assert word.startswith("$FOA_STUDY_DIR"), (stage, prev, word)


def test_the_chained_stages_are_the_ones_that_read_their_predecessor(study):
    assert study.CHAINED == {"p01_population", "p03_pools", "p05_cohort"}
    # A stage that trains one model has nothing to chain.
    assert set(study.STAGES) - study.CHAINED == {"p02_ginit", "p04_g0", "p06_runs"}


def test_the_eligibility_floor_is_stated_and_applied(study):
    """One floor, named on every step that has to respect it."""
    assert study.MIN_SEQUENCES == 100
    cut = next(ln for ln in study.population_lines()
               if ln.startswith("foa select-eligible "))
    assert f" --min-samples {study.MIN_SEQUENCES}" in cut
    # The old-data draw must not admit a user the population step excluded.
    draw = next(ln for ln in study.pools_lines() if ln.startswith("foa draw-old-data "))
    assert f" --min-samples {study.MIN_SEQUENCES}" in draw
    assert f" --size {CFG.old_size}" in draw


def test_the_cohort_selection_names_its_destination(study):
    """
    Without --out the cohort lands in the provider's own outliers directory,
    which for a non-NIST provider is one level deeper than the rest of the
    chain reads - and nothing notices until a later stage cannot find it.
    """
    line = next(ln for ln in study.cohort_lines()
                if ln.startswith("foa select-outliers "))
    assert f" --out {study.COHORT_FILE}" in line
    assert " --require-trainable" in line


# --------------------------------------------------------------------------- #
# the cache facade the books are built on
# --------------------------------------------------------------------------- #
@pytest.fixture()
def data(shakespeare_provider):
    return shakespeare_provider.dataset


def test_the_row_space_is_dense_and_wholly_owned(data):
    """
    Every row belongs to exactly one user. Token offsets would leave gaps - the
    last 80 positions of each user start no window - and the book maps rows to
    writers by index, so a row owned by nobody would be attributed to whoever
    sat at index -1.
    """
    cache = data.cache
    assert len(cache) == data.num_sequences
    assert cache.writer_index.shape == (len(cache),)
    assert set(np.unique(cache.writer_index).tolist()) <= set(range(len(cache.writer_ids)))
    total = sum(data.sample_count(u) for u in data.users)
    assert len(cache) == total


def test_dense_rows_round_trip_to_the_windows_they_name(data):
    for user in data.users[:3]:
        rows = data.window_rows(user)
        assert len(rows) == data.sample_count(user)
        assert np.array_equal(data.offsets_of_rows(rows), data.window_offsets(user))
        assert np.array_equal(
            data.row_labels(rows), data.targets(data.window_offsets(user))
        )
        assert set(data.cache.writer_index[rows].tolist()) == {data.users.index(user)}


def test_writer_samples_is_lazy_but_complete(data):
    samples = data.writer_samples()
    assert len(samples) == len(data.users)
    assert data.users[0] in samples
    assert "nobody" not in samples
    pairs = samples[data.users[0]]
    assert len(pairs) == data.sample_count(data.users[0])
    assert all(isinstance(row, int) and isinstance(label, int) for row, label in pairs[:5])


def test_a_book_built_on_the_facade_splits_every_user(data, tmp_path):
    from federated_outlier_adaptation.data.fold_book import (
        FoldBook, build_fold_book, write_fold_book,
    )

    users = [u for u in data.users if data.sample_count(u) >= 20][:4]
    assert users, "the fixture corpus is too small to split"
    book = build_fold_book(data, writers=users, folds=3, seed=42,
                           train_rate=0.6, eval_rate=0.2, tag="t")
    path = write_fold_book(book, tmp_path / "b")
    reread = FoldBook.load(str(path))
    for user in users:
        assert reread.covers(user)
        rows = sum(len(reread.part(1, user, p)) for p in ("train", "val", "test"))
        assert rows == data.sample_count(user)


def test_the_book_is_the_split_when_one_is_configured(data, tmp_path, monkeypatch):
    """
    The defect this closes: g-0 training on a seeded cut while its preservation
    is measured on a booked one. The two overlap, and the number is then partly
    measured on rows the model fitted.
    """
    from federated_outlier_adaptation.data.fold_book import (
        FoldBook, build_fold_book, write_fold_book,
    )
    from federated_outlier_adaptation.data.shakespeare import ShakespeareData

    users = [u for u in data.users if data.sample_count(u) >= 20][:3]
    book = build_fold_book(data, writers=users, folds=3, seed=42,
                           train_rate=0.6, eval_rate=0.2, tag="t")
    path = write_fold_book(book, tmp_path / "b")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "1")

    fresh = ShakespeareData(npz_path=data.npz_path, index_path=data.index_path)
    loaders = fresh.build_dataset(users, batch_size=8, seed=42)
    sizes = [0 if l is None else len(l.dataset) for l in loaders]
    expected = FoldBook.load(str(path)).split(1, users)
    assert sizes == [len(expected[p]) for p in ("train", "val", "test")]

    offsets = [set(l.dataset.offsets.tolist()) for l in loaders if l is not None]
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            assert not offsets[i] & offsets[j], "the partitions overlap"


def test_without_a_book_the_seeded_split_still_works(data, monkeypatch):
    monkeypatch.delenv("FOA_FOLD_BOOK", raising=False)
    users = [u for u in data.users if data.sample_count(u) >= 20][:2]
    train, _, _ = data.build_dataset(users, batch_size=8, seed=42)
    assert train is not None and len(train.dataset) > 0


def test_the_feature_penalty_has_a_penultimate_to_align_on():
    """
    ``feature_l2`` is one of the three transferred winners. On the LEAF LSTM the
    penultimate representation is the final hidden state of the last layer -
    exactly what the head consumes - so the penalty means the same thing it did
    on the CNN.
    """
    import torch

    from federated_outlier_adaptation.models.char_lstm import CharLSTM

    model = CharLSTM(vocab=80, embed=8, hidden=256, layers=2)
    x = torch.randint(0, 80, (4, 80))
    features = model.penultimate(x)
    assert tuple(features.shape) == (4, 256)
    assert torch.allclose(model.fc(features), model(x), atol=1e-6)
