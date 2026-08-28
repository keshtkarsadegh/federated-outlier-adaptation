"""
P20: the twenty-client point of the scaling study, at both dropout levels.

The failure this file exists to prevent is a cohort that does not extend. If the
worst-20 is not the worst-10 plus the next ten writers down the same ranking,
the twenty-client point is not the ten-client point scaled up - it is a
different population, and every comparison drawn across the two sizes is
measuring that instead. Nothing downstream would say so: the tables would fill,
the numbers would differ, and the difference would be read as a size effect.

The second is the book. A larger cohort cannot reuse the ten-client book - it
holds no rows for writers it never covered - so this stage brings its own, and
the only thing that keeps it comparable is that it is cut by the same rule at
the same seed. A book with a different rule would move the rows under the ten
writers both sizes share.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from federated_outlier_adaptation.data.fold_book import FoldBook, write_fold_book
from federated_outlier_adaptation.training import five_cells
from federated_outlier_adaptation.training import study_config
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as CFG

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

TEN = [
    "f3642_03", "f2248_68", "f2297_69", "f2524_58", "f2145_89",
    "f2162_62", "f0619_33", "f2307_62", "f3151_13", "f2304_67",
]
NEXT_TEN = [
    "f2230_67", "f2325_86", "f3211_20", "f2157_61", "f0048_00",
    "f2169_69", "f2438_76", "f3200_41", "f2383_60", "f2422_85",
]
TWENTY = TEN + NEXT_TEN


@pytest.fixture()
def emit():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import study_emit

    return study_emit


@pytest.fixture()
def setup_tool():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_digits_p20

    return make_digits_p20


def _book(writers, path):
    rows = np.tile(np.array([0, 1, 2, 1, 2], dtype=np.int8), len(writers))
    book = FoldBook(
        np.vstack([rows for _ in range(5)]),
        {"tag": "test", "writers": list(writers), "folds": 5},
        writer_index=np.repeat(np.arange(len(writers)), 5),
    )
    write_fold_book(book, path)


@pytest.fixture()
def root(tmp_path):
    outliers = tmp_path / "outliers"
    outliers.mkdir(parents=True)
    (outliers / CFG.cohort_file_name).write_text(json.dumps({"clients": TEN}))
    (outliers / "cohort_worst20.json").write_text(json.dumps({"clients": TWENTY}))
    books = tmp_path / "fold_books"
    books.mkdir()
    _book(TEN, books / CFG.cohort_book_name)
    _book(TWENTY, books / "cohort20")
    return tmp_path


def run(emit, root, tmp_path, expect=30):
    out = tmp_path / "d01_p20_runs.txt"
    return emit.WHAT["c20"](CFG, root, out, expect), out


def body(out):
    return [ln for ln in out.read_text().splitlines()
            if ln and not ln.startswith("#")]


# --------------------------------------------------------------------------- #
# participation and the trimmed mean at the new sizes
# --------------------------------------------------------------------------- #
def test_the_two_participation_points():
    assert study_config.participants(20, 0.1) == 18   # 2 dropped
    assert study_config.participants(20, 0.2) == 16   # 4 dropped


@pytest.mark.parametrize(
    "fraction,participants,trim,survivors",
    [
        (0.4, 18, 7, 4),   # int(0.4 * 18) = 7 -> 18 - 14
        (0.4, 16, 6, 4),   # int(0.4 * 16) = 6 -> 16 - 12
        (0.3, 18, 5, 8), (0.2, 18, 3, 12), (0.1, 18, 1, 16),
        (0.3, 16, 4, 8), (0.2, 16, 3, 10), (0.1, 16, 1, 14),
    ],
)
def test_trim_survivors_match_the_server(fraction, participants, trim, survivors):
    """
    The winner's rule keeps FOUR updates at both points - out of 18 and out of
    16 - so the trimming gets harsher as the federation grows while the number
    averaged stays put. That belongs beside the number, not after it.
    """
    import torch

    from federated_outlier_adaptation.aggregation.concurrent_methods import (
        ServerState, con_delta_trimmed_mean,
    )

    assert int(fraction * participants) == trim
    assert five_cells.trim_survivors(fraction, participants) == survivors

    values = [-100.0] + [float(i) for i in range(participants - 2)] + [100.0]
    result = con_delta_trimmed_mean(
        {"w": torch.zeros(1)},
        [{"w": torch.tensor([v])} for v in values],
        [1] * participants,
        server_state=ServerState(eta=1.0, trim_fraction=fraction),
    )["w"].item()
    kept = sorted(values)[trim: participants - trim]
    assert len(kept) == survivors
    assert result == pytest.approx(sum(kept) / len(kept))


# --------------------------------------------------------------------------- #
# the cohort must extend
# --------------------------------------------------------------------------- #
def test_a_cohort_that_does_not_extend_the_ten_stops_it(emit, root, tmp_path):
    """
    Same ranking, same eligibility rule, so the first ten must come back
    identical. If they do not, the two sizes are different populations.
    """
    shuffled = [TWENTY[3]] + [w for w in TWENTY if w != TWENTY[3]]
    (root / "outliers" / "cohort_worst20.json").write_text(
        json.dumps({"clients": shuffled}))
    code, out = run(emit, root, tmp_path)
    assert code == 1
    assert not out.exists()


def test_a_cohort_of_the_wrong_size_stops_it(emit, root, tmp_path):
    (root / "outliers" / "cohort_worst20.json").write_text(
        json.dumps({"clients": TWENTY[:19]}))
    code, out = run(emit, root, tmp_path)
    assert code == 1
    assert not out.exists()


def test_a_missing_cohort_stops_it(emit, root, tmp_path):
    (root / "outliers" / "cohort_worst20.json").unlink()
    code, out = run(emit, root, tmp_path)
    assert code == 1
    assert not out.exists()


def test_a_book_that_misses_a_writer_stops_it(emit, root, tmp_path):
    """The ten-client book covers half of this cohort, which is not enough."""
    _book(TEN, root / "fold_books" / "cohort20")
    code, out = run(emit, root, tmp_path)
    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# the lines
# --------------------------------------------------------------------------- #
@pytest.fixture()
def lines(emit, root, tmp_path):
    code, out = run(emit, root, tmp_path)
    assert code == 0
    return body(out)


def test_thirty_tasks_three_configurations_five_folds_two_points(lines):
    assert len(lines) == 30
    assert SL.counts(CFG)["c20"] == 30
    for tag, drawn in (("d01_c20d10_", 18), ("d01_c20d20_", 16)):
        stage = [ln for ln in lines if f" --parent {tag}" in ln]
        assert len(stage) == 15
        assert all(f" --clients-per-round {drawn} " in ln for ln in stage)


def test_every_line_parses(lines):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for line in lines:
        tokens = shlex.split(line)
        assert tokens[0] == "foa"
        parser.parse_args(tokens[1:])


def test_winners_only_no_control(lines):
    """
    The control's job is answered by the ten-client pair; here the question is
    whether the winners hold at a larger size.
    """
    assert not [ln for ln in lines if "_control_" in ln]
    assert all(" --trainer AnchoredTrainer" in ln for ln in lines)
    for name in ("winner", "balanced", "sequential"):
        assert len([ln for ln in lines if f"_{name}_fold" in ln]) == 10


def test_every_line_reads_the_twenty_client_cohort_and_its_own_book(lines):
    for line in lines:
        assert f" --outliers-file {SL.POOLS}/cohort_worst20.json" in line
        assert f" --fold-book {SL.BOOKS}/cohort20.foldbook.npz --fold " in line
        assert "cohort10.foldbook.npz" not in line
        assert "cohort_worst10.json" not in line


def test_the_protocol_is_otherwise_the_studys(lines):
    for line in lines:
        assert " --init global --global-name g0" in line
        assert f" --old-book {SL.OLD_BOOK}" in line
        assert " --old-clients-file" in line and " --old-fold all" in line
        assert " --classes digits" in line and " --resolution 28" in line
        assert f" --rounds {SL.FULL_ROUNDS} " in line
        assert " --epochs 5 " in line and " --batch-size 64 " in line
        assert " --save-final-model" in line and " --track-clients" in line
        assert "anchor=frozen" in line


def test_parents_and_seeds_are_distinct_across_both_points(lines):
    parents = [ln.split(" --parent ")[1].split()[0] for ln in lines]
    seeds = [int(ln.split(" --sampler-seed ")[1].split()[0]) for ln in lines]
    assert len(set(parents)) == len(set(seeds)) == 30
    assert all(715000 < s < 716000 or 716000 < s < 717000 for s in seeds)
    assert {p.split("_")[1] for p in parents} == {"c20d10", "c20d20"}


def test_a_wrong_expect_refuses_to_leave_a_short_file(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, expect=40)
    assert code == 1


def test_the_records_state_both_points(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path)
    assert code == 0
    for name, drawn, rule in (
        ("p20_c20_d10.json", 18, "floor((1 - 0.1) * 20) = 18"),
        ("p20_c20_d20.json", 16, "floor((1 - 0.2) * 20) = 16"),
    ):
        record = json.loads((root / "tables" / name).read_text())
        assert record["clients"] == TWENTY
        assert record["clients_per_round"] == drawn
        assert record["participation_rule"] == rule
        assert record["fold_book"] == "cohort20.foldbook.npz"
        ids = {c["id"] for c in record["configurations"]}
        assert ids == {"winner", "balanced", "sequential"}
        winner = [c for c in record["configurations"] if c["id"] == "winner"][0]
        assert winner["trim_survivors"] == 4


# --------------------------------------------------------------------------- #
# the setup chain
# --------------------------------------------------------------------------- #
def test_the_setup_chain_is_ten_steps_in_dependency_order(setup_tool):
    tasks = setup_tool.lines()
    assert len(tasks) == 10
    assert tasks[0].startswith("foa select-outliers ") and "--k 20" in tasks[0]
    assert "cohort_worst20" in tasks[1] and "cohort_worst10" in tasks[1]
    assert tasks[2].startswith("foa fold-book ")
    assert "FoldBook.load" in tasks[3]
    assert all(t.startswith("foa evaluate-book ") for t in tasks[4:9])
    assert "study_emit.py c20" in tasks[9] and "--expect 30" in tasks[9]


def test_the_book_is_cut_by_the_studys_own_rule(setup_tool):
    """
    Same rule and same seed as every other book here. A different rule would
    move the rows under the ten writers both cohort sizes share, and the size
    comparison would be over two splits.
    """
    book = setup_tool.lines()[2]
    for token in ("--folds 5", "--seed 42", "--train-rate 0.6", "--eval-rate 0.2"):
        assert token in book
    assert "--clients-file $FOA_STUDY_DIR/outliers/cohort_worst20.json" in book


def test_the_baseline_scores_the_test_part_of_every_fold(setup_tool):
    evals = setup_tool.lines()[4:9]
    assert [f"--fold {k} " in e for k, e in zip((1, 2, 3, 4, 5), evals)] == [True] * 5
    for line in evals:
        assert " --part test" in line
        assert " --model-path $FOA_STUDY_DIR/g0_model" in line
        assert line.endswith("--out $FOA_STUDY_DIR/g0_cohort20_evaluations.json")


def test_every_setup_path_stays_inside_the_study(setup_tool):
    """
    The sbatch guard refuses a path-valued flag that does not, and a refusal at
    array element 1 of a %1 chain stops everything behind it.
    """
    flags = {
        "--results-dir", "--out", "--model-path", "--fold-book", "--clients-file",
        "--writers-file", "--exclude-file", "--old-book", "--old-clients-file",
        "--outliers-file", "--init-checkpoint", "--root", "--data-dir",
        "--cache-dir", "--scores",
    }
    for task in setup_tool.lines():
        words = task.split()
        for prev, word in zip(words, words[1:]):
            if prev in flags and not word.startswith("-"):
                assert word.startswith("$FOA_STUDY_DIR"), (prev, word)
