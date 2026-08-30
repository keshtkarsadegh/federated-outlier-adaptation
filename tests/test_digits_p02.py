"""
The digits study's detector stage, and the banking that documents it.

The 62-class programme is retired. This study is NIST SD19 digits 0-9 only,
which changes two things that matter and must be checked rather than assumed:
the model head is ten units (McMahan's network exactly), and the splittability
arithmetic moves, because fewer classes means more rows per class.
"""

from __future__ import annotations

import csv
import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.data.nist28 import Nist28Dataset

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


# --------------------------------------------------------------------------- #
# ten classes, and the head that follows from them
# --------------------------------------------------------------------------- #
def test_the_digit_class_set_gives_a_ten_unit_head(cohort_cache, tmp_path):
    """
    ``--classes digits`` has to reach the model, not just the label filter.

    A 62-unit head trained on ten labels would train perfectly well and be the
    wrong network, so this is checked at the provider rather than trusted.
    """
    from federated_outlier_adaptation.providers.nist import NistProvider

    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache,
        resolution=28, classes="digits",
    )
    assert provider.num_classes == 10
    model = provider.make_model()
    assert model.fc2.out_features == 10


def test_the_ten_class_network_is_mcmahan_s():
    from federated_outlier_adaptation.models.fedavg_cnn import FedAvgCNN

    total = sum(p.numel() for p in FedAvgCNN(num_classes=10).parameters())
    assert total == 1_663_370


def test_the_class_set_restricts_the_dataset(cohort_cache):
    """A writer's digit rows are its rows in 0-9 and nothing else."""
    everything = Nist28Dataset(cohort_cache, classes="all")
    digits = Nist28Dataset(cohort_cache, classes="digits")
    assert digits.num_classes == 10 and everything.num_classes == 62
    for writer, rows in digits.writer_samples().items():
        assert all(label < 10 for _, label in rows)


# --------------------------------------------------------------------------- #
# the banking step
# --------------------------------------------------------------------------- #
def test_the_counts_are_per_writer_and_per_class(cohort_cache):
    """
    The ranking says which writers are hard; this says what they had to work with.

    A reader will ask whether the worst writers were simply the ones with the
    least data, and a ranking on its own cannot answer that.
    """
    from federated_outlier_adaptation.outliers.scoring import writer_counts

    dataset = Nist28Dataset(cohort_cache, classes="digits")
    payload = writer_counts(dataset)

    samples = dataset.writer_samples()
    assert payload["num_classes"] == 10
    assert payload["writers"] == len(samples)
    assert payload["rows"] == sum(len(v) for v in samples.values())
    for writer, rows in samples.items():
        assert payload["per_writer_total"][writer] == len(rows)
        assert sum(payload["per_writer_per_class"][writer].values()) == len(rows)


def test_the_class_totals_add_up_to_the_rows(cohort_cache):
    from federated_outlier_adaptation.outliers.scoring import writer_counts

    payload = writer_counts(Nist28Dataset(cohort_cache, classes="digits"))
    assert sum(payload["class_totals"].values()) == payload["rows"]


def test_the_summary_reports_the_writers_missing_a_class(cohort_cache):
    """A writer cannot score on a class it has none of - the first thing to check."""
    from federated_outlier_adaptation.outliers.scoring import writer_counts

    payload = writer_counts(Nist28Dataset(cohort_cache, classes="digits"))
    summary = payload["summary"]
    assert summary["min_classes_covered"] <= 10
    assert summary["writers_with_all_classes"] <= payload["writers"]
    assert summary["min"] <= summary["median"] <= summary["max"]


def test_the_csv_has_one_row_per_writer_and_one_column_per_class(cohort_cache, tmp_path):
    from federated_outlier_adaptation.outliers.scoring import (
        write_counts_csv,
        writer_counts,
    )

    payload = writer_counts(Nist28Dataset(cohort_cache, classes="digits"))
    path = write_counts_csv(payload, tmp_path / "counts.csv")
    rows = list(csv.reader(open(path)))
    assert rows[0] == ["writer", "total", *[str(i) for i in range(10)]]
    assert len(rows) == payload["writers"] + 1
    for row in rows[1:]:
        assert int(row[1]) == sum(int(v) for v in row[2:])


def test_the_counts_command_is_wired_up(tmp_path):
    """
    The command exists, takes both destinations, and runs on the digit set.

    Not run end to end here: the 28x28 cache is resolved from
    ``FOA_NIST28_DIR`` at import time, so a fixture cache cannot be pointed at
    from inside a test. The payload and the CSV are covered directly above.
    """
    from federated_outlier_adaptation.cli import build_parser

    args = build_parser().parse_args([
        "writer-counts", "--results-dir", str(tmp_path),
        "--resolution", "28", "--classes", "digits",
        "--out", str(tmp_path / "c.json"), "--csv", str(tmp_path / "c.csv"),
    ])
    assert args.func.__name__ == "cmd_writer_counts"
    assert args.classes == "digits" and args.resolution == 28
    assert args.out.endswith("c.json") and args.csv.endswith("c.csv")


# --------------------------------------------------------------------------- #
# the task file
# --------------------------------------------------------------------------- #
@pytest.fixture()
def p02(tools_path):
    import make_digits_p02

    return make_digits_p02


def test_every_line_is_digits_and_inside_the_study(p02):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for index, line in enumerate(p02.lines(), 1):
        # $GINIT_FOLD is expanded by the shell at submit time.
        args = parser.parse_args(shlex.split(line.replace("$GINIT_FOLD", "3"))[1:])
        assert args.classes == "digits", index
        assert args.resolution == 28, index
        for attr in ("results_dir", "out", "csv", "model_path", "fold_book",
                     "root", "writers_file"):
            value = getattr(args, attr, None)
            if isinstance(value, str) and ("/" in value or value.startswith("$")):
                assert value.startswith("$FOA_STUDY_DIR"), (index, attr, value)


def test_nothing_names_a_retired_root(p02):
    """The 62-class study folders are deleted; a line naming one is a dead run."""
    text = "\n".join(p02.lines()) + p02.readme(p02.lines())
    for dead in ("main_v6", "T1_20outliers", "T1_10outliers", "T2_", "T3_"):
        assert dead not in text, dead


def test_the_chain_is_counts_book_folds_winner_scores(p02):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    names = [
        parser.parse_args(shlex.split(line.replace("$GINIT_FOLD", "3"))[1:]).func.__name__
        for line in p02.lines()
    ]
    assert names == (
        ["cmd_writer_counts", "cmd_fold_book"]
        + ["cmd_global_train"] * 5
        + ["cmd_select_fold"]
        + ["cmd_score_writers"] * 5
        + ["cmd_average_scores"]
    )


def test_ginit_trains_on_the_whole_digit_dataset(p02):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    trains = [
        parser.parse_args(shlex.split(line)[1:])
        for line in p02.lines() if " global-train " in line
    ]
    assert len(trains) == 5
    assert {t.fold for t in trains} == {1, 2, 3, 4, 5}
    for t in trains:
        # 'all' is every writer of the digit dataset, through the book
        assert t.population == "all"
        assert t.model == "fedavg_cnn"
        assert t.fold_book.endswith("all_writers_digits.foldbook.npz")
        assert (t.epochs, t.early_stopping_patience, t.min_epochs) == (100, 10, 20)


def test_every_fold_scores_its_own_held_out_rows(p02):
    """
    Each fold judges the writers on rows that fold did not train on.

    Every writer contributed training images to g-init, so scoring a writer on
    rows the scoring model trained on would measure memorisation rather than
    difficulty. Fold k therefore scores with fold k's model, on fold k's
    held-out rows, and the book is what still knows which rows those were.
    """
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    scorers = [
        parser.parse_args(shlex.split(line)[1:])
        for line in p02.lines() if " score-writers " in line
    ]
    assert len(scorers) == 5
    assert {s.fold for s in scorers} == {1, 2, 3, 4, 5}
    for s in scorers:
        # the model of that fold, never the crowned one
        assert s.model_path.endswith(f"ginit_fold{s.fold}/global_model")
        assert s.fold_book.endswith("all_writers_digits.foldbook.npz")
        assert s.out.endswith(f"writer_scores_fold{s.fold}.json")


def test_the_ranking_is_the_mean_and_no_fold_is_crowned(p02):
    """
    The defect this guards against.

    The five folds sit within six ten-thousandths of a point of one another, so
    crowning one and ranking with it lets floating-point error choose the
    study's clients: a rebuild crowned a different fold and re-derived a cohort
    sharing three writers of ten with the published one. Nothing in the chain
    may depend on which fold won.
    """
    from federated_outlier_adaptation.cli import build_parser

    lines = p02.lines()
    assert not any("$GINIT_FOLD" in line for line in lines)
    assert not any(line.rstrip().endswith("ginit_model") or "/ginit_model " in line
                   for line in lines if " score-writers " in line)

    args = build_parser().parse_args(shlex.split(lines[-1])[1:])
    assert args.func.__name__ == "cmd_average_scores"
    assert len(args.inputs) == 5
    for fold in (1, 2, 3, 4, 5):
        assert any(f"writer_scores_fold{fold}.json" in i for i in args.inputs)
    assert args.out.endswith("writer_scores.json")


def test_the_stated_survival_numbers_are_in_the_readme(p02):
    text = p02.readme(p02.lines())
    assert str(p02.DIGIT_WRITERS) in text
    assert f"{p02.DIGIT_ROWS:,}" in text
    assert "0" in text and "splittable" in text.lower()


# --------------------------------------------------------------------------- #
# the runner
# --------------------------------------------------------------------------- #
def test_the_runner_defaults_to_the_digits_study(tools_path):
    """
    The old default pointed at a folder that has been deleted.

    A default naming a missing study is worse than none: it turns "you forgot
    the export" into "your results went somewhere nobody looks".
    """
    import make_study_sbatch

    text = make_study_sbatch.sbatch_text()
    assert make_study_sbatch.DEFAULT_STUDY == "Digits_study01"
    assert "studies/Digits_study01" in text
    for dead in ("T1_10outliers", "T1_20outliers", "main_v6}"):
        assert dead not in text, dead


def test_the_runner_refuses_a_missing_study_root(tools_path):
    text = __import__("make_study_sbatch").sbatch_text()
    assert 'if [ ! -d "$FOA_STUDY_DIR" ]; then' in text
    assert "exit 78" in text
    # and it must not create the root it was told to check
    assert 'mkdir -p "$LOG_DIR" "$FOA_STUDY_DIR"' not in text


def test_the_runner_defaults_the_class_set_to_digits(tools_path):
    text = __import__("make_study_sbatch").sbatch_text()
    assert 'FOA_NIST_CLASSES:-digits' in text
