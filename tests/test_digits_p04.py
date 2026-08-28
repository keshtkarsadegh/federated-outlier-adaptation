"""
The digits study's populations, the table that defends them, and the figure.

Two of these are worth more than they look. The **cohort table** exists to
answer the objections a ranking cannot answer on its own, so its rank must be
over the whole population and its fold sizes must come from the book that will
actually be trained on. And the **gallery** had a real blocker - its reference
row read a file this study never writes - so the repoint is tested both ways:
that naming a reference works, and that failing to name one says so.
"""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.data.nist28 import Nist28Dataset
from federated_outlier_adaptation.outliers.scoring import (
    cohort_table,
    draw_cohort,
    write_cohort_table_csv,
    writer_counts,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def digits(cohort_cache):
    return Nist28Dataset(cohort_cache, classes="digits")


@pytest.fixture()
def scored(digits):
    """A ranking over every writer, worst first by construction."""
    writers = sorted(digits.writer_samples())
    return {writer: 0.80 + 0.01 * index for index, writer in enumerate(writers)}


# --------------------------------------------------------------------------- #
# the cohort table
# --------------------------------------------------------------------------- #
def test_the_rank_is_over_the_whole_population_not_the_cohort(digits, scored):
    """
    'rank 3 of 3,580' is the statement worth printing; 'rank 3 of 10' is not.

    A rank within the cohort would say nothing at all - the cohort is by
    definition the bottom of the ranking.
    """
    counts = writer_counts(digits)
    cohort = sorted(scored, key=lambda w: scored[w])[:3]
    table = cohort_table(cohort, scored, counts)

    assert [row["rank"] for row in table["rows"]] == [1, 2, 3]
    assert all(row["of"] == len(scored) for row in table["rows"])
    assert table["population"]["scored_writers"] == len(scored)


def test_each_row_carries_the_evidence_behind_the_selection(digits, scored):
    counts = writer_counts(digits)
    cohort = sorted(scored, key=lambda w: scored[w])[:3]
    table = cohort_table(cohort, scored, counts)

    samples = digits.writer_samples()
    for row in table["rows"]:
        assert row["accuracy"] == pytest.approx(scored[row["writer"]])
        assert row["total_rows"] == len(samples[row["writer"]])
        assert sum(row["per_class"].values()) == row["total_rows"]
        assert set(row["per_class"]) == {str(i) for i in range(10)}
        assert row["classes_covered"] == sum(1 for v in row["per_class"].values() if v)


def test_the_fold_sizes_come_from_the_book_that_will_be_trained_on(digits, scored):
    """
    The split sizes are the book's, not a recomputation.

    They are why the table is written after the book rather than before it: a
    table built earlier would silently drop this column.
    """
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    counts = writer_counts(digits)
    cohort = sorted(scored, key=lambda w: scored[w])[:2]
    book = build_fold_book(digits, writers=cohort, folds=5, seed=42)
    table = cohort_table(cohort, scored, counts, book=book)

    for row in table["rows"]:
        assert set(row["folds"]) == {"1", "2", "3", "4", "5"}
        for fold, parts in row["folds"].items():
            for part in ("train", "val", "test"):
                assert parts[part] == len(book.part(int(fold), row["writer"], part))
            assert sum(parts.values()) == row["total_rows"]


def test_without_a_book_the_table_simply_has_no_fold_column(digits, scored):
    counts = writer_counts(digits)
    cohort = sorted(scored, key=lambda w: scored[w])[:2]
    table = cohort_table(cohort, scored, counts)
    assert all("folds" not in row for row in table["rows"])


def test_the_csv_widens_for_the_folds_when_they_are_there(digits, scored, tmp_path):
    from federated_outlier_adaptation.data.fold_book import build_fold_book

    counts = writer_counts(digits)
    cohort = sorted(scored, key=lambda w: scored[w])[:2]
    book = build_fold_book(digits, writers=cohort, folds=5, seed=42)

    plain = write_cohort_table_csv(
        cohort_table(cohort, scored, counts), tmp_path / "plain.csv"
    )
    folded = write_cohort_table_csv(
        cohort_table(cohort, scored, counts, book=book), tmp_path / "folded.csv"
    )
    plain_head = next(csv.reader(open(plain)))
    folded_head = next(csv.reader(open(folded)))

    assert plain_head[:6] == ["writer", "accuracy", "rank", "of", "total_rows",
                              "classes_covered"]
    assert "class_9" in plain_head and "f1_train" not in plain_head
    assert folded_head[:len(plain_head)] == plain_head
    assert folded_head[-3:] == ["f5_train", "f5_val", "f5_test"]
    assert len(list(csv.reader(open(folded)))) == len(cohort) + 1


# --------------------------------------------------------------------------- #
# the reference population of the figure
# --------------------------------------------------------------------------- #
def test_the_best_end_draws_the_writers_the_detector_found_easy():
    """'A typical writer' means one the detector had no trouble with."""
    accuracies = {f"w{i:03d}": i / 100.0 for i in range(100)}
    worst = draw_cohort(accuracies, size=5, seed=7, pool=10)
    best = draw_cohort(accuracies, size=5, seed=7, pool=10, best=True)

    assert set(worst["clients"]) <= {f"w{i:03d}" for i in range(10)}
    assert set(best["clients"]) <= {f"w{i:03d}" for i in range(90, 100)}
    assert worst["end"] == "worst" and best["end"] == "best"
    assert not set(worst["clients"]) & set(best["clients"])


def test_the_best_end_draw_is_seeded_like_the_other():
    accuracies = {f"w{i:03d}": i / 100.0 for i in range(100)}
    first = draw_cohort(accuracies, size=5, seed=3, best=True)
    again = draw_cohort(accuracies, size=5, seed=3, best=True)
    assert first["clients"] == again["clients"]


# --------------------------------------------------------------------------- #
# the gallery repoint
# --------------------------------------------------------------------------- #
def test_a_named_reference_needs_no_writer_split(cohort_cache, tmp_path):
    """
    The blocker, gone.

    The top row used to come from ``provider.global_client_ids()``, which reads
    a ``writer_split.json`` this study never writes - so the figure could not be
    drawn at all. Naming the reference is now enough.
    """
    from federated_outlier_adaptation.analysis.outlier_gallery import (
        render_outlier_gallery,
    )
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())
    out = tmp_path / "gallery" / "cohort_vs_typical.png"

    summary = render_outlier_gallery(
        provider, clients=writers[:2], out_path=out,
        source_clients=writers[2:4], n_examples=2, max_columns=10,
    )
    assert out.is_file()
    sidecar = json.loads(out.with_suffix(".json").read_text())
    # the sidecar records what was compared against, so "typical" is stated
    assert sidecar["source_clients"] == 2
    assert summary["clients"] == writers[:2]
    assert len(summary["labels"]) <= 10


def test_without_a_reference_the_error_says_what_to_pass(cohort_cache, tmp_path):
    from federated_outlier_adaptation.analysis.outlier_gallery import (
        render_outlier_gallery,
    )
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "empty", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())
    with pytest.raises(ValueError, match="--source-clients-file"):
        render_outlier_gallery(
            provider, clients=writers[:1], out_path=tmp_path / "g.png"
        )


def test_the_legacy_split_is_still_honoured_when_it_exists(cohort_cache, tmp_path):
    """The published 62-class figure must be unchanged."""
    from federated_outlier_adaptation.analysis.outlier_gallery import _default_source
    from federated_outlier_adaptation.providers.nist import NistProvider

    results = tmp_path / "legacy"
    results.mkdir()
    provider = NistProvider(
        results_dir=results, cache_dir=cohort_cache, resolution=28, classes="digits"
    )
    writers = sorted(provider.dataset.writer_samples())
    (results / "writer_split.json").write_text(json.dumps(
        {"local_writers": writers[:2], "global_writers": writers[2:]}
    ))
    assert _default_source(provider) == writers[2:]


def test_the_reference_flags_are_on_the_command(tmp_path):
    from federated_outlier_adaptation.cli import build_parser

    args = build_parser().parse_args([
        "outlier-figure", "--resolution", "28", "--classes", "digits",
        "--outliers-file", "/c.json", "--source-clients-file", "/t.json",
        "--n-examples", "3", "--max-columns", "10",
    ])
    assert args.func.__name__ == "cmd_outlier_figure"
    assert args.source_clients_file == "/t.json"
    assert args.n_examples == 3 and args.max_columns == 10


# --------------------------------------------------------------------------- #
# the task file
# --------------------------------------------------------------------------- #
@pytest.fixture()
def p04(tools_path):
    import make_digits_p04

    return make_digits_p04


def test_every_line_is_digits_and_inside_the_study(p04):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for index, line in enumerate(p04.lines(), 1):
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.classes == "digits" and args.resolution == 28, index
        for attr in ("results_dir", "out", "csv", "clients_file", "fold_book",
                     "exclude_file", "outliers_file", "source_clients_file",
                     "scores", "counts"):
            value = getattr(args, attr, None)
            if isinstance(value, str) and ("/" in value or value.startswith("$")):
                assert value.startswith("$FOA_STUDY_DIR"), (index, attr, value)


def test_the_chain_is_cohort_old_books_table_reference_figure(p04):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    names = [
        parser.parse_args(shlex.split(line)[1:]).func.__name__ for line in p04.lines()
    ]
    assert names == [
        "cmd_select_outliers", "cmd_draw_old_data", "cmd_fold_book",
        "cmd_fold_book", "cmd_cohort_table", "cmd_draw_cohort",
        "cmd_outlier_figure",
    ]


def test_the_table_runs_after_the_book_it_reads(p04):
    """Its fold sizes do not exist until the book does."""
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    args = [parser.parse_args(shlex.split(line)[1:]) for line in p04.lines()]
    book_line = next(
        i for i, a in enumerate(args)
        if a.func.__name__ == "cmd_fold_book" and "cohort" in a.out
    )
    table_line = next(
        i for i, a in enumerate(args) if a.func.__name__ == "cmd_cohort_table"
    )
    assert table_line > book_line
    assert args[table_line].fold_book.endswith("cohort10.foldbook.npz")


def test_no_writer_is_both_an_outlier_and_old_data(p04):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    old = next(
        parser.parse_args(shlex.split(line)[1:]) for line in p04.lines()
        if " draw-old-data " in line
    )
    assert old.size == p04.OLD_SIZE and old.seed == p04.OLD_SEED
    assert old.require_trainable
    assert old.exclude_file.endswith(f"cohort_worst{p04.COHORT_SIZE}.json")


def test_the_figure_compares_the_cohort_against_drawn_typical_writers(p04):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    lines = p04.lines()
    reference = next(
        parser.parse_args(shlex.split(line)[1:]) for line in lines
        if " draw-cohort " in line
    )
    figure = next(
        parser.parse_args(shlex.split(line)[1:]) for line in lines
        if " outlier-figure " in line
    )
    assert reference.best is True and reference.k == p04.TYPICAL_SIZE
    assert reference.exclude_file.endswith(f"cohort_worst{p04.COHORT_SIZE}.json")
    assert figure.source_clients_file.endswith(f"typical{p04.TYPICAL_SIZE}.json")
    assert figure.outliers_file.endswith(f"cohort_worst{p04.COHORT_SIZE}.json")
    # ten digits is the whole alphabet here, so no column is hidden
    assert figure.max_columns == 10
    assert figure.out.endswith(".png")


def test_nothing_names_a_retired_root(p04):
    text = "\n".join(p04.lines()) + p04.readme(p04.lines())
    for dead in ("main_v6", "T1_20outliers", "T1_10outliers", "T2_", "T3_"):
        assert dead not in text, dead
