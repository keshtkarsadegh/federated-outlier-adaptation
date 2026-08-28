"""
The revised selection: a coarse cut by g-init, the real choice by g-0.

g-init trained on every writer, so its ranking is contaminated by memorisation -
a writer can score well because the model remembers it. That is survivable in a
cut and fatal in a selection. These tests pin the two halves of the split and,
above all, the property that makes the second half meaningful: **the model that
scores a writer never trained on it.**
"""

from __future__ import annotations

import csv
import json
import math
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.outliers.scoring import (
    cohort_table,
    score_pool,
    split_pools,
    write_cohort_table_csv,
    write_pools_csv,
    write_scores_csv,
    writer_counts,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p05(tools_path):
    import make_digits_p05v2

    return make_digits_p05v2


@pytest.fixture()
def parsed(p05):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    return [
        parser.parse_args(shlex.split(line)[1:]) if line.startswith("foa ") else None
        for line in p05.lines()
    ]


# --------------------------------------------------------------------------- #
# the coarse cut
# --------------------------------------------------------------------------- #
def test_the_cut_is_at_the_ceiling_so_the_bad_pool_is_never_short():
    accuracies = {f"w{i:04d}": i / 1000.0 for i in range(3580)}
    pools = split_pools(accuracies, bad_fraction=0.30)

    assert pools["cut_rank"] == math.ceil(0.30 * 3580) == 1074
    assert pools["bad_count"] == 1074 and pools["good_count"] == 2506
    assert pools["bad_count"] + pools["good_count"] == len(accuracies)


def test_the_two_pools_partition_the_population():
    accuracies = {f"w{i:03d}": (i % 37) / 37.0 for i in range(100)}
    pools = split_pools(accuracies, bad_fraction=0.30)

    assert set(pools["bad"]) | set(pools["good"]) == set(accuracies)
    assert not set(pools["bad"]) & set(pools["good"])
    # and the cut really is a cut: nothing in BAD scores above anything in GOOD
    assert max(accuracies[w] for w in pools["bad"]) <= min(
        accuracies[w] for w in pools["good"]
    )


def test_the_record_says_where_the_cut_fell():
    """A reader has to be able to see the boundary, not infer it."""
    accuracies = {f"w{i:03d}": i / 100.0 for i in range(100)}
    pools = split_pools(accuracies, bad_fraction=0.30)

    assert pools["accuracy_at_cut"] == pytest.approx(accuracies[pools["bad"][-1]])
    assert pools["first_good_accuracy"] == pytest.approx(
        accuracies[pools["good"][0]]
    )
    assert "30%" in pools["rule"]


def test_the_cut_can_be_restricted_to_splittable_writers():
    accuracies = {f"w{i:03d}": i / 50.0 for i in range(50)}
    eligible = [f"w{i:03d}" for i in range(0, 50, 2)]
    pools = split_pools(accuracies, bad_fraction=0.30, eligible=eligible)

    assert set(pools["bad"]) | set(pools["good"]) == set(eligible)
    assert pools["ranked_writers"] == 25
    assert pools["excluded_by_eligibility"] == 25


def test_an_impossible_fraction_is_refused():
    accuracies = {"a": 0.5, "b": 0.6}
    for fraction in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="bad fraction must lie"):
            split_pools(accuracies, bad_fraction=fraction)


def test_the_pools_csv_ranks_the_whole_population(tmp_path):
    accuracies = {f"w{i:03d}": i / 100.0 for i in range(100)}
    pools = split_pools(accuracies, bad_fraction=0.30)
    path = write_pools_csv(pools, tmp_path / "pools.csv")

    rows = list(csv.reader(open(path)))
    assert rows[0] == ["writer", "accuracy", "rank", "pool"]
    assert len(rows) == 101
    assert [r[3] for r in rows[1:31]] == ["bad"] * 30
    assert [int(r[2]) for r in rows[1:]] == list(range(1, 101))


# --------------------------------------------------------------------------- #
# the selection pass
# --------------------------------------------------------------------------- #
def test_the_pool_is_scored_on_every_row_it_holds(cohort_cache, tmp_path):
    """
    Not a held-out partition: g-0 has never seen these writers at all.

    There is nothing to hold out from, and using all of a writer's data makes
    the estimate as tight as that writer allows - which matters, because the
    writers being scored are the small, difficult ones.
    """
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())[:3]
    payload = score_pool(provider, provider.make_model(), writers, batch_size=8)

    samples = provider.dataset.writer_samples()
    assert payload["writers"] == 3
    assert payload["rows"] == sum(len(samples[w]) for w in writers)
    for writer in writers:
        assert payload["samples"][writer] == len(samples[writer])
        assert 0.0 <= payload["accuracies"][writer] <= 1.0


def test_the_ranking_is_worst_first_and_ranks_are_dense(cohort_cache, tmp_path):
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())
    payload = score_pool(provider, provider.make_model(), writers, batch_size=8)

    scores = [payload["accuracies"][w] for w in payload["ranking"]]
    assert scores == sorted(scores)
    assert [payload["rank"][w] for w in payload["ranking"]] == list(
        range(1, len(writers) + 1)
    )


def test_the_flat_ranking_is_the_shape_the_cohort_cut_reads(cohort_cache, tmp_path):
    """``scores`` must be readable by the same loader ``select-outliers`` uses."""
    from federated_outlier_adaptation.outliers.client_accuracy import (
        load_client_accuracies,
    )
    from federated_outlier_adaptation.outliers.scoring import write_json
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())
    payload = score_pool(provider, provider.make_model(), writers, batch_size=8)

    flat = write_json(payload["scores"], tmp_path / "flat.json")
    loaded = dict(load_client_accuracies(flat))
    assert loaded == pytest.approx(payload["accuracies"])


def test_the_scores_csv_carries_rows_and_rank(cohort_cache, tmp_path):
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    writers = sorted(provider.dataset.writer_samples())
    payload = score_pool(provider, provider.make_model(), writers, batch_size=8)
    path = write_scores_csv(payload, tmp_path / "s.csv")

    rows = list(csv.reader(open(path)))
    assert rows[0] == ["writer", "accuracy", "n_rows", "rank"]
    assert len(rows) == len(writers) + 1


# --------------------------------------------------------------------------- #
# the cohort is cut from the shipped model's ranking
# --------------------------------------------------------------------------- #
def test_the_cohort_is_cut_from_the_ranking_it_is_given(tmp_path):
    """
    ``--scores`` is what moves the selection off the coarse detector.

    Without it the cohort would be cut from g-init again, which is the whole
    thing this redesign exists to stop.
    """
    from federated_outlier_adaptation.outliers.selection import select_cohort

    outliers = tmp_path / "outliers"
    outliers.mkdir(parents=True)
    # the coarse detector says one thing...
    (outliers / "clients_acc_on_global.json").write_text(json.dumps(
        [{f"w{i}": 1.0 - i / 100.0} for i in range(20)]
    ))
    # ...and the shipped model, over the bad pool, says the reverse
    shipped = outliers / "bad_acc_on_g0.json"
    shipped.write_text(json.dumps([{f"w{i}": i / 100.0} for i in range(20)]))

    payload, _ = select_cohort(
        results_dir=tmp_path, k=3, force=True, scores=shipped
    )
    assert payload["clients"] == ["w0", "w1", "w2"]

    coarse, _ = select_cohort(results_dir=tmp_path, k=3, force=True)
    assert coarse["clients"] == ["w19", "w18", "w17"]


def test_the_table_ranks_within_the_pool_it_was_selected_from(cohort_cache):
    """
    'rank 3 of 1,074' is the statement worth printing.

    The cohort was chosen out of the bad pool, so the bad pool is the population
    the rank is relative to - not the whole 3,580.
    """
    from federated_outlier_adaptation.data.nist28 import Nist28Dataset

    dataset = Nist28Dataset(cohort_cache, classes="digits")
    writers = sorted(dataset.writer_samples())
    bad = {w: 0.5 + i / 100.0 for i, w in enumerate(writers[:4])}
    counts = writer_counts(dataset)

    table = cohort_table(writers[:2], bad, counts, rule="worst of the bad pool")
    assert [row["of"] for row in table["rows"]] == [len(bad), len(bad)]
    assert [row["rank"] for row in table["rows"]] == [1, 2]
    assert table["rule"] == "worst of the bad pool"


def test_the_table_carries_the_coarse_score_alongside(cohort_cache, tmp_path):
    """So a reader can see whether the two rankings agree."""
    from federated_outlier_adaptation.data.nist28 import Nist28Dataset

    dataset = Nist28Dataset(cohort_cache, classes="digits")
    writers = sorted(dataset.writer_samples())
    shipped = {w: 0.5 + i / 100.0 for i, w in enumerate(writers)}
    coarse = {w: 0.9 - i / 100.0 for i, w in enumerate(writers)}
    counts = writer_counts(dataset)

    table = cohort_table(
        writers[:2], shipped, counts, extra_scores={"ginit": coarse}
    )
    assert table["extra_columns"] == ["ginit"]
    for row in table["rows"]:
        assert row["ginit"] == pytest.approx(coarse[row["writer"]])
        assert row["accuracy"] == pytest.approx(shipped[row["writer"]])

    path = write_cohort_table_csv(table, tmp_path / "t.csv")
    head = next(csv.reader(open(path)))
    assert head[6] == "ginit"
    assert head.index("ginit") < head.index("class_0")


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def test_the_chain_is_archive_pools_draw_book_folds_winner_score_cohort(p05, parsed):
    names = [a.func.__name__ if a else "[shell]" for a in parsed]
    assert names == (
        ["[shell]", "cmd_split_pools", "cmd_draw_old_data", "cmd_fold_book"]
        + ["cmd_global_train"] * 5
        + ["cmd_select_fold", "cmd_score_pool", "cmd_select_outliers",
           "cmd_fold_book", "cmd_cohort_table", "cmd_draw_cohort",
           "cmd_outlier_figure"]
        + ["cmd_evaluate_book"] * 10
    )
    assert len(names) == 26


def test_every_foa_line_is_digits_and_inside_the_study(parsed):
    for index, args in enumerate(parsed, 1):
        if args is None:
            continue
        assert args.classes == "digits" and args.resolution == 28, index
        for attr in ("results_dir", "out", "csv", "clients_file", "fold_book",
                     "exclude_file", "outliers_file", "source_clients_file",
                     "scores", "counts", "model_path", "writers_file", "root"):
            value = getattr(args, attr, None)
            if isinstance(value, str) and ("/" in value or value.startswith("$")):
                assert value.startswith("$FOA_STUDY_DIR"), (index, attr, value)


def test_g0_never_trains_on_a_writer_it_will_later_score(parsed):
    """
    The property the whole redesign rests on.

    The old data is drawn with the BAD pool excluded, and the scoring pass runs
    over exactly that BAD pool - so no writer is ever scored by a model that
    trained on it.
    """
    draw = next(a for a in parsed if a and a.func.__name__ == "cmd_draw_old_data")
    score = next(a for a in parsed if a and a.func.__name__ == "cmd_score_pool")
    trains = [a for a in parsed if a and a.func.__name__ == "cmd_global_train"]

    assert draw.exclude_file.endswith("pool_bad.json")
    assert score.clients_file.endswith("pool_bad.json")
    assert all(t.writers_file.endswith("old_data.json") for t in trains)
    assert draw.out.endswith("old_data.json")


def test_the_cohort_is_cut_from_the_shipped_model_s_scores(parsed):
    cut = next(a for a in parsed if a and a.func.__name__ == "cmd_select_outliers")
    assert cut.mode == "worst" and cut.k == 10
    assert cut.scores.endswith("bad_acc_on_g0.json")
    assert cut.force
    # and must not rewrite the coarse detector's census on its way past
    assert cut.no_accuracy_table


def test_the_table_names_both_rankings(parsed):
    table = next(a for a in parsed if a and a.func.__name__ == "cmd_cohort_table")
    assert table.scores.endswith("bad_acc_on_g0.json")
    assert any(pair.startswith("ginit=") for pair in table.extra_scores)
    assert table.fold_book.endswith("cohort10.foldbook.npz")
    assert table.rule


def test_the_reference_writers_come_from_the_good_pool(parsed):
    reference = next(a for a in parsed if a and a.func.__name__ == "cmd_draw_cohort")
    assert reference.best is True and reference.k == 5
    assert reference.exclude_file.endswith("pool_bad.json")
    assert reference.scores.endswith("clients_acc_on_global.json")


def test_both_baselines_keep_their_filenames_and_tags(parsed):
    evals = [a for a in parsed if a and a.func.__name__ == "cmd_evaluate_book"]
    assert len(evals) == 10
    outs = {a.out.rsplit("/", 1)[-1] for a in evals}
    assert outs == {"g0_evaluations.json", "g0_perfold_evaluations.json"}
    assert len({a.tag for a in evals}) == 10


# --------------------------------------------------------------------------- #
# the archive
# --------------------------------------------------------------------------- #
def test_the_archive_moves_v1_and_keeps_what_still_stands(p05, tmp_path):
    """
    Moved, not deleted - and only the artefacts a v2 reader could mistake.

    The coarse detector and the census survive: they are still exactly what they
    were, and v2 reads both.
    """
    import subprocess

    study = tmp_path / "study"
    for folder in ("outliers", "fold_books", "tables/gallery", "g0_fold1",
                   "ginit_fold1"):
        (study / folder).mkdir(parents=True)
    survivors = [
        "ginit_model", "ginit_selection.json",
        "outliers/writer_scores.json", "outliers/clients_acc_on_global.json",
        "outliers/writer_counts.json",
        "fold_books/all_writers_digits.foldbook.npz",
    ]
    doomed = [
        "g0_model", "g0_selection.json", "g0_evaluations.json",
        "g0_perfold_evaluations.json", "outliers/old_data.json",
        "outliers/cohort_worst10.json", "outliers/typical5.json",
        "fold_books/old_data.foldbook.npz", "fold_books/cohort10.foldbook.npz",
        "tables/cohort_table.json", "tables/cohort_table.csv",
        "tables/gallery/cohort_vs_typical.png",
    ]
    for name in survivors + doomed:
        (study / name).touch()

    line = p05.lines()[0].replace("$FOA_STUDY_DIR", str(study))
    subprocess.run(["bash", "-c", line], check=True, capture_output=True)

    for name in survivors:
        assert (study / name).exists(), name
    for name in doomed:
        assert not (study / name).exists(), name
    assert (study / "v1_superseded" / "g0_model").exists()
    assert (study / "v1_superseded" / "outliers" / "old_data.json").exists()
    assert (study / "v1_superseded" / "tables" / "gallery").is_dir()
    assert (study / "g0_fold1").exists() is False


def test_the_archive_is_idempotent(p05, tmp_path):
    """Re-running a chain must not fail on a study that was already archived."""
    import subprocess

    study = tmp_path / "study"
    (study / "outliers").mkdir(parents=True)
    line = p05.lines()[0].replace("$FOA_STUDY_DIR", str(study))
    for _ in range(2):
        subprocess.run(["bash", "-c", line], check=True, capture_output=True)


def test_nothing_names_a_retired_root(p05):
    text = "\n".join(p05.lines()) + p05.readme(p05.lines())
    for dead in ("main_v6", "T1_20outliers", "T1_10outliers", "T2_", "T3_"):
        assert dead not in text, dead
