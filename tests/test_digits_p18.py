"""
The size and dropout stages, and the participation rule underneath them.

``participants(n, d) = floor((1 - d) * n)`` - the kept count floors, because a
fractional client is not trained on. One rule has to serve every stage: nine of
ten here, sixteen of twenty in the earlier programme, four of five at the
smaller size, eight of ten at the dropout point. A stage that wrote its own
number instead could drift from the rule and nothing in the output would say so,
which is why every emitter calls the function and none of them carries a
literal.

The extremes are the one documented exception, and it is a real one: at two
clients the rule keeps one, and dropping a client from a two-client federation
is not a participation study but a coin flip on whether the round happens. Those
cases run at full participation, which is what the extreme emitter already
states in so many words.

The other two failure modes: a cohort restated by hand instead of derived would
let a size stage federate writers the ten-client stages do not, and a book
re-cut for five writers would give them different rows than the ten-client runs
gave them - turning a participation comparison into a split comparison.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from federated_outlier_adaptation.data.fold_book import FoldBook, write_fold_book
from federated_outlier_adaptation.training import agg_cells, five_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training import study_config
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as CFG

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

#: The ranking the study's own artefacts carry, worst first.
RANKED = [
    "f3642_03", "f2248_68", "f2297_69", "f2524_58", "f2145_89",
    "f2162_62", "f0619_33", "f2307_62", "f3151_13", "f2304_67",
]
ACCURACIES = [0.54, 0.71, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]

#: ``(drawn, parent tag, cohort size, seed block)`` per stage.
STAGES = {
    "five": (4, "d01_five_", 5, 712000),
    "drop20": (8, "d01_drop20_", 10, 714000),
}


@pytest.fixture()
def emit():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import study_emit

    return study_emit


@pytest.fixture()
def root(tmp_path):
    """A study root carrying only what these emitters read."""
    outliers = tmp_path / "outliers"
    outliers.mkdir(parents=True)
    (outliers / "bad_acc_on_g0.json").write_text(json.dumps(
        [{w: a} for w, a in zip(RANKED, ACCURACIES)]))
    (outliers / "bad_scores_on_g0.json").write_text(json.dumps(
        {"ranking": RANKED, "accuracies": dict(zip(RANKED, ACCURACIES))}))
    (outliers / CFG.cohort_file_name).write_text(json.dumps(
        {"rule": "worst_k", "clients": RANKED}))

    # Ten writers, five rows each, three folds - enough to be a real book.
    assignment = np.tile(np.array([0, 1, 2, 1, 2], dtype=np.int8), len(RANKED))
    book = FoldBook(
        np.vstack([assignment for _ in range(3)]),
        {"tag": "test", "writers": RANKED, "folds": 3},
        writer_index=np.repeat(np.arange(len(RANKED)), 5),
    )
    books = tmp_path / "fold_books"
    books.mkdir()
    write_fold_book(book, books / CFG.cohort_book_name)
    return tmp_path


def run(emit, root, tmp_path, stage, expect=20):
    out = tmp_path / f"tasks_{stage}.txt"
    return emit.WHAT[stage](CFG, root, out, expect), out


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n,dropout,expected",
    [
        (10, 0.10, 9),    # the study's own participation
        (5, 0.10, 4),     # the five-client size stage
        (20, 0.20, 16),   # the earlier programme's participation
        (10, 0.20, 8),    # the dropout stage
    ],
)
def test_participants_floors_the_kept_count(n, dropout, expected):
    assert study_config.participants(n, dropout) == expected


def test_one_rule_serves_every_stage_this_programme_has_run():
    """
    Nine of ten, sixteen of twenty, four of five, eight of ten - all one
    expression. A stage that wrote its own number could drift from the rule with
    nothing in the output to say so.
    """
    rule = study_config.participants
    assert [rule(10, 0.1), rule(20, 0.2), rule(5, 0.1), rule(10, 0.2)] == [9, 16, 4, 8]


def test_the_extremes_are_a_real_exception_not_an_oversight():
    """
    At two clients the rule keeps one, which is not a participation study but a
    coin flip on whether the round happens; at one it keeps nobody and refuses.
    Both are why the extreme cases run at full participation instead - and the
    refusal names that exception rather than quietly rounding up.
    """
    assert study_config.participants(2, 0.1) == 1
    with pytest.raises(ValueError, match="FULL"):
        study_config.participants(1, 0.1)


def test_the_extreme_emitter_takes_the_exception_explicitly():
    """It says so in the line, not by omission."""
    agg = {"id": "x", "rule": "fedavg", "flags": {}, "path": "concurrent"}
    reg = {"id": "y", "trainer": "BaseTrainer", "fedprox": False}
    line = SL.extreme_line(CFG, "double", "/tmp/x.json", agg, reg, 1, 1)
    assert " --policy all --participation 1.0" in line
    assert "--clients-per-round" not in line


def test_the_float_cannot_steal_a_client():
    """
    A bare floor of a float product is short by one in real cases, and silently:
    the run just trains one client fewer than the rate says. Swept over every
    federation size to a hundred and every whole-percent rate, the rule must
    agree with exact integer arithmetic - repairing those cases and inventing
    none.
    """
    import math

    assert math.floor((1 - 0.8) * 10) == 1     # the arithmetic says 2
    assert study_config.participants(10, 0.8) == 2

    short = [(n, pct) for n in range(1, 101) for pct in range(1, 100)
             if math.floor((1 - pct / 100) * n) != ((100 - pct) * n) // 100]
    assert len(short) == 56, "the sweep's float shortfalls moved; recheck the guard"
    wrong = [(n, pct) for n in range(1, 101) for pct in range(1, 100)
             if ((100 - pct) * n) // 100 >= 1
             and study_config.participants(n, pct / 100) != ((100 - pct) * n) // 100]
    assert wrong == []


def test_a_rate_that_would_leave_nobody_is_refused():
    with pytest.raises(ValueError):
        study_config.participants(0, 0.1)
    with pytest.raises(ValueError):
        study_config.participants(10, 1.0)
    with pytest.raises(ValueError):
        study_config.participants(10, 0.95)


def test_the_study_config_is_the_rule_not_a_second_statement_of_it():
    assert CFG.clients_per_round == study_config.participants(CFG.cohort_size)
    assert five_cells.clients_per_round(5) == 4
    assert five_cells.clients_per_round(10, 0.2) == 8
    assert SL.FIVE_PER_ROUND == 4


@pytest.mark.parametrize(
    "fraction,participants,survivors",
    [
        (0.4, 8, 2), (0.3, 8, 4), (0.2, 8, 6), (0.1, 8, 8),
        (0.4, 4, 2), (0.3, 4, 2), (0.2, 4, 4), (0.1, 4, 4),
        # Nine is the table the ten-client stages were read against, and 0.1 is
        # not a trimmed mean there at all - it drops nothing.
        (0.4, 9, 3), (0.3, 9, 5), (0.2, 9, 7), (0.1, 9, 9),
    ],
)
def test_trim_survivors_match_the_server(fraction, participants, survivors):
    """The documented survivor count is the server's own expression."""
    import torch

    from federated_outlier_adaptation.aggregation.concurrent_methods import (
        ServerState, con_delta_trimmed_mean,
    )

    assert five_cells.trim_survivors(fraction, participants) == survivors

    values = [-100.0] + [1.0] * (participants - 2) + [100.0]
    result = con_delta_trimmed_mean(
        {"w": torch.zeros(1)},
        [{"w": torch.tensor([v])} for v in values],
        [1] * participants,
        server_state=ServerState(eta=1.0, trim_fraction=fraction),
    )["w"].item()
    trim = int(fraction * participants)
    kept = sorted(values)[trim: participants - trim] if trim else sorted(values)
    assert len(kept) == survivors
    assert result == pytest.approx(sum(kept) / len(kept))


# --------------------------------------------------------------------------- #
# the cohort is derived, and cross-checked
# --------------------------------------------------------------------------- #
def test_the_cohort_is_the_first_five_of_the_ranking(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, "five")
    assert code == 0
    listing = json.loads((root / "outliers" / "cohort_worst5.json").read_text())
    assert listing["clients"] == RANKED[:5]
    assert listing["accuracies"]["f3642_03"] == pytest.approx(0.54)


def test_a_ranking_that_disagrees_with_the_cohort_file_stops_it(emit, root, tmp_path):
    (root / "outliers" / CFG.cohort_file_name).write_text(json.dumps(
        {"rule": "worst_k", "clients": ["f2304_67"] + RANKED[:9]}))
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


def test_a_writer_the_book_never_split_stops_it(emit, root, tmp_path):
    """
    Narrowing by client list works only while the ten-client book covers every
    name on it; a writer it never split has no rows to train or score on.
    """
    outliers = root / "outliers"
    ranked = ["fZZZZ_99"] + RANKED
    accs = [0.1] + ACCURACIES
    (outliers / "bad_acc_on_g0.json").write_text(json.dumps(
        [{w: a} for w, a in zip(ranked, accs)]))
    (outliers / "bad_scores_on_g0.json").write_text(json.dumps({"ranking": ranked}))
    (outliers / CFG.cohort_file_name).write_text(json.dumps(
        {"rule": "worst_k", "clients": ranked}))
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


def test_the_dropout_stage_federates_the_whole_cohort(emit, root, tmp_path):
    """No narrowing: twenty percent dropout is about the draw, not the cohort."""
    code, out = run(emit, root, tmp_path, "drop20")
    assert code == 0
    lines = body(out)
    for line in lines:
        assert f" --outliers-file {SL.POOLS}/{CFG.cohort_file_name}" in line
        assert "cohort_worst5" not in line


def test_a_cohort_of_the_wrong_size_stops_the_dropout_stage(emit, root, tmp_path):
    (root / "outliers" / CFG.cohort_file_name).write_text(
        json.dumps({"clients": RANKED[:8]}))
    code, out = run(emit, root, tmp_path, "drop20")
    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# the lines
# --------------------------------------------------------------------------- #
def body(out):
    return [ln for ln in out.read_text().splitlines()
            if ln and not ln.startswith("#")]


@pytest.fixture()
def emitted(emit, root, tmp_path):
    made = {}
    for stage in STAGES:
        code, out = run(emit, root, tmp_path, stage)
        assert code == 0, stage
        made[stage] = body(out)
    return made


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_twenty_tasks_four_configurations_five_folds(emitted, stage):
    assert len(emitted[stage]) == 20
    assert len(five_cells.CONFIGS) * len(SL.FOLDS) == 20


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_every_line_parses(emitted, stage):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for line in emitted[stage]:
        tokens = shlex.split(line)
        assert tokens[0] == "foa"
        parser.parse_args(tokens[1:])


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_every_line_draws_the_ruled_number(emitted, stage):
    drawn = STAGES[stage][0]
    for line in emitted[stage]:
        assert f" --policy uniform --clients-per-round {drawn}" in line


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_every_line_reads_the_ten_client_book(emitted, stage):
    """Same rows as the ten-client runs, so a gap is about the federation."""
    for line in emitted[stage]:
        assert f" --fold-book {SL.cohort_book(CFG)} --fold " in line


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_the_protocol_is_otherwise_the_studys(emitted, stage):
    for line in emitted[stage]:
        assert " --init global --global-name g0" in line
        assert f" --old-book {SL.OLD_BOOK}" in line
        assert " --old-clients-file" in line and " --old-fold all" in line
        assert " --classes digits" in line and " --resolution 28" in line
        assert f" --rounds {SL.FULL_ROUNDS} " in line
        assert " --epochs 5 " in line and " --batch-size 64 " in line
        assert " --save-final-model" in line
        if "--set " in line:
            assert "anchor=frozen" in line


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_parents_and_seeds_are_distinct_and_tagged(emitted, stage):
    _, tag, _, block = STAGES[stage]
    lines = emitted[stage]
    parents = [line.split(" --parent ")[1].split()[0] for line in lines]
    seeds = [int(line.split(" --sampler-seed ")[1].split()[0]) for line in lines]
    assert len(set(parents)) == len(set(seeds)) == 20
    assert all(parent.startswith(tag) for parent in parents)
    assert all(block < seed < block + 1000 for seed in seeds)


def test_the_stages_share_no_parent_and_no_seed(emitted):
    """
    Both stages run the same four configurations. Without their own tags and
    their own seed blocks they would name one another's folders.
    """
    for field in (" --parent ", " --sampler-seed "):
        seen = [line.split(field)[1].split()[0]
                for lines in emitted.values() for line in lines]
        assert len(set(seen)) == 20 * len(STAGES)


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_the_control_is_plain_fedavg_with_no_penalty(emitted, stage):
    control = [line for line in emitted[stage] if "_control_" in line]
    assert len(control) == 5
    for line in control:
        assert " --trainer BaseTrainer" in line
        assert " --aggregation fedavg" in line
        assert "--set " not in line
        assert "--extended-aggregations" not in line


@pytest.mark.parametrize("stage", sorted(STAGES))
def test_the_selected_configurations_carry_their_rule_and_penalty(emitted, stage):
    for cell in five_cells.CONFIGS:
        if cell["regulariser"] is None:
            continue
        chosen = [ln for ln in emitted[stage] if f"_{cell['id']}_fold" in ln]
        assert len(chosen) == 5
        for line in chosen:
            assert " --extended-aggregations" in line
            assert "--set " in line
    winner = [ln for ln in emitted[stage] if "_winner_fold" in ln]
    assert all("--trim-frac 0.4" in ln for ln in winner)


def test_a_configuration_that_names_an_unknown_cell_stops_it(
        emit, root, tmp_path, monkeypatch):
    broken = five_cells.configs()
    broken[0]["aggregation"] = "no_such_rule"
    monkeypatch.setattr(five_cells, "configs", lambda: broken)
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


def test_a_rule_run_on_the_wrong_schedule_stops_it(emit, root, tmp_path, monkeypatch):
    """A concurrent rule reported as sequential is a claim about a loop it never ran on."""
    broken = five_cells.configs()
    broken[0]["family"] = "sequential"
    monkeypatch.setattr(five_cells, "configs", lambda: broken)
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


def test_a_wrong_expect_refuses_to_leave_a_short_file(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, "five", expect=19)
    assert code == 1


# --------------------------------------------------------------------------- #
# the records
# --------------------------------------------------------------------------- #
def test_the_five_client_record_states_the_rule(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, "five")
    assert code == 0
    record = json.loads((root / "tables" / "p18_five_client.json").read_text())
    assert record["clients"] == RANKED[:5]
    assert record["clients_per_round"] == 4
    assert record["participation_rule"] == "floor((1 - 0.1) * 5) = 4"
    survivors = {c["id"]: c["trim_survivors"] for c in record["configurations"]}
    assert survivors["winner"] == 2        # f=0.4 at K=4 keeps two updates
    assert survivors["balanced"] is None


def test_the_dropout_record_states_the_rule(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, "drop20")
    assert code == 0
    record = json.loads((root / "tables" / "p19_dropout20.json").read_text())
    assert record["clients_per_round"] == 8
    assert record["dropout"] == 0.2
    assert record["participation_rule"] == "floor((1 - 0.2) * 10) = 8"
    survivors = {c["id"]: c["trim_survivors"] for c in record["configurations"]}
    assert survivors["winner"] == 2
