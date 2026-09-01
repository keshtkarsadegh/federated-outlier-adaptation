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
#:
#: ``c10d10`` is the study's own setting - nine of ten, the participation every
#: screen and every final in this programme drew - given a stage of its own so
#: the carried arms have a row at the setting they were selected under. It runs
#: the same four arms as the other ten-client stage and reads the same book, so
#: everything parameterised over this table applies to it unchanged; only its
#: tag and its seed block separate the two.
STAGES = {
    "five": (4, "d01_five_", 5, 712000),
    "c10d10": (9, "d01_c10d10_", 10, 713000),
    "drop20": (8, "d01_drop20_", 10, 714000),
}


# --------------------------------------------------------------------------- #
# the selection records these stages read their configurations out of
# --------------------------------------------------------------------------- #
#: The cross, as ``p15_combination_grid.json`` records it, and the crowning it
#: produced.  Synthesised into the tmp study rather than pinned in source: which
#: pair won is a selection result, and a test that named one would be asserting
#: the same stale thing the emitters used to.
PAIRS = {
    "concurrent": [["trimmed_t3", "feature_l2_lam0p1"],
                   ["anchor_h1", "ntd_b0p01_t0p5"]],
    "sequential": [["seq_order_shuffle", "feature_l2_lam0p01"]],
}

#: ``ranked`` order is the crowning's own - gain less spend - so the first entry
#: of each family is that family's strongest pair.  The balanced arm is instead
#: the concurrent pair that gave up least preservation, which here is the second
#: one; the two arms therefore name two configurations rather than one twice.
PRESERVATION = [
    ("trimmed_t3_feature_l2_lam0p1", 0.980),
    ("anchor_h1_ntd_b0p01_t0p5", 0.991),
    ("seq_order_shuffle_feature_l2_lam0p01", 0.985),
]


def write_selection_records(root):
    """The cross and its crowning, in the shape ``study_emit`` writes them."""
    tables = Path(root) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "p15_combination_grid.json").write_text(json.dumps(
        {"rule": "top-3 aggregations x top-3 penalties, within each family",
         "pairs": PAIRS}))
    (tables / "p15_stage_winner.json").write_text(json.dumps({
        "rank_by": "test",
        "winner": PRESERVATION[0][0],
        "family": "concurrent",
        "aggregation": "trimmed_t3",
        "regulariser": "feature_l2_lam0p1",
        "ranked": [{"id": combo, "adaptation": 0.92, "preservation": value}
                   for combo, value in PRESERVATION],
    }))


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
    write_selection_records(tmp_path)
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
    assert len(five_cells.ARMS) * len(SL.FOLDS) == 20


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
def test_the_selected_configurations_carry_their_rule_and_penalty(emitted, root, stage):
    for cell in five_cells.configs(root):
        if cell["regulariser"] is None:
            continue
        chosen = [ln for ln in emitted[stage] if f"_{cell['id']}_fold" in ln]
        assert len(chosen) == 5
        for line in chosen:
            assert " --extended-aggregations" in line
            assert "--set " in line
    winner = [ln for ln in emitted[stage] if "_winner_fold" in ln]
    # The winner trims a COUNT of clients, so the fraction it asks for depends
    # on how many participate - and the count is clamped where the federation
    # is too small to express it. Derive the expectation rather than fix it.
    for line in winner:
        words = shlex.split(line)
        clients = int(words[words.index("--clients-per-round") + 1])
        count = min(3, max((clients - 1) // 2, 0))
        assert f"--trim-frac {(count + 0.5) / clients!r}" in line


def test_a_configuration_that_names_an_unknown_cell_stops_it(
        emit, root, tmp_path, monkeypatch):
    broken = five_cells.configs(root)
    broken[0]["aggregation"] = "no_such_rule"
    monkeypatch.setattr(five_cells, "configs", lambda _root: broken)
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


def test_a_rule_run_on_the_wrong_schedule_stops_it(emit, root, tmp_path, monkeypatch):
    """A concurrent rule reported as sequential is a claim about a loop it never ran on."""
    broken = five_cells.configs(root)
    broken[0]["family"] = "sequential"
    monkeypatch.setattr(five_cells, "configs", lambda _root: broken)
    code, out = run(emit, root, tmp_path, "five")
    assert code == 1
    assert not out.exists()


# --------------------------------------------------------------------------- #
# the configurations are read, not written down
# --------------------------------------------------------------------------- #
def test_the_arms_are_resolved_from_the_records(root):
    """
    Three arms out of the crowning, plus the control, in report order.

    winner is the strongest concurrent pair the crowning ranks; balanced is the
    concurrent pair that gave up least preservation, which is a different pair;
    sequential is the strongest sequential one. None of the three is named in
    the module.
    """
    cells = five_cells.configs(root)
    assert [c["id"] for c in cells] == [a["id"] for a in five_cells.ARMS]
    carried = {c["id"]: (c["family"], c["aggregation"], c["regulariser"])
               for c in cells}
    assert carried["winner"] == ("concurrent", "trimmed_t3", "feature_l2_lam0p1")
    assert carried["balanced"] == ("concurrent", "anchor_h1", "ntd_b0p01_t0p5")
    assert carried["sequential"] == (
        "sequential", "seq_order_shuffle", "feature_l2_lam0p01")
    assert carried["control"] == ("both", "fedavg", None)


def test_a_re_crowning_moves_the_arms_with_it(root):
    """The point of reading them: a new crowning is carried, not ignored."""
    record = json.loads((root / "tables" / "p15_stage_winner.json").read_text())
    record["ranked"] = list(reversed(record["ranked"]))
    (root / "tables" / "p15_stage_winner.json").write_text(json.dumps(record))
    carried = {c["id"]: c["aggregation"] for c in five_cells.configs(root)}
    assert carried["winner"] == "anchor_h1"
    assert carried["balanced"] == "trimmed_t3"


def test_a_missing_record_refuses_by_name_rather_than_guessing(tmp_path):
    """
    A guessed winner emits, trains and reports exactly like a chosen one.

    Both records are needed: the crowning says which pair won, and the cross
    says which schedule a pair belongs to - a combination id is two ids joined
    by an underscore and cannot be split without the lists it was built from.
    """
    with pytest.raises(SystemExit, match="p15_stage_winner.json"):
        five_cells.configs(tmp_path)
    write_selection_records(tmp_path)
    (tmp_path / "tables" / "p15_combination_grid.json").unlink()
    with pytest.raises(SystemExit, match="p15_combination_grid.json"):
        five_cells.configs(tmp_path)


def test_a_crowning_the_cross_does_not_list_is_refused(root):
    """Two records describing different stages is not something to average."""
    record = json.loads((root / "tables" / "p15_stage_winner.json").read_text())
    record["ranked"].append({"id": "made_up_pair", "adaptation": 0.99,
                             "preservation": 0.99})
    (root / "tables" / "p15_stage_winner.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="different stages"):
        five_cells.configs(root)


@pytest.mark.parametrize("missing", ["p15_stage_winner.json",
                                     "p15_combination_grid.json"])
def test_the_size_stage_stops_when_a_record_is_absent(
        emit, root, tmp_path, missing):
    (root / "tables" / missing).unlink()
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


def test_the_nine_of_ten_record_states_the_rule(emit, root, tmp_path):
    """
    The setting the whole study was selected under, recorded like any other.

    Its participation is not written into the stage - it is the same
    ``floor((1 - d) * n)`` every other stage applies, at the study's own rate -
    so the record has to say nine, and the trimmed winner has to keep what nine
    participants leave rather than what the five-client stage left.
    """
    code, _ = run(emit, root, tmp_path, "c10d10")
    assert code == 0
    record = json.loads((root / "tables" / "p21_c10_d10.json").read_text())
    assert record["clients"] == RANKED
    assert record["n_clients"] == 10
    assert record["dropout"] == 0.1
    assert record["clients_per_round"] == 9 == CFG.clients_per_round
    assert record["participation_rule"] == "floor((1 - 0.1) * 10) = 9"
    assert record["fold_book"] == Path(SL.cohort_book(CFG)).name
    ids = [c["id"] for c in record["configurations"]]
    assert ids == [arm["id"] for arm in five_cells.ARMS]
    survivors = {c["id"]: c["trim_survivors"] for c in record["configurations"]}
    assert survivors["winner"] == 3        # f=0.4 at K=9 keeps three updates
    assert survivors["balanced"] is None


def test_the_nine_of_ten_stage_federates_the_whole_cohort(emit, root, tmp_path):
    """No narrowing and no book of its own: only the participation is the point."""
    code, out = run(emit, root, tmp_path, "c10d10")
    assert code == 0
    for line in body(out):
        assert f" --outliers-file {SL.POOLS}/{CFG.cohort_file_name}" in line
        assert "cohort_worst5" not in line and "cohort20" not in line


def test_the_nine_of_ten_stage_carries_the_crowned_arms(emit, root, tmp_path):
    """
    The four arms are the crowning's, resolved at emission, control included.

    The control is what makes this stage readable beside the eight-of-ten one:
    without it a difference between the two could be the method or the draw.
    """
    code, out = run(emit, root, tmp_path, "c10d10")
    assert code == 0
    lines = body(out)
    for cell in five_cells.configs(root):
        chosen = [ln for ln in lines if f"_{cell['id']}_fold" in ln]
        assert len(chosen) == 5, cell["id"]
        for line in chosen:
            if cell["regulariser"] is None:
                assert " --aggregation fedavg" in line
                assert " --trainer BaseTrainer" in line
                assert "--set " not in line
                assert "--extended-aggregations" not in line
            else:
                assert " --extended-aggregations" in line
                assert "--set " in line


def test_the_nine_of_ten_stage_stops_on_a_cohort_of_the_wrong_size(
        emit, root, tmp_path):
    (root / "outliers" / CFG.cohort_file_name).write_text(
        json.dumps({"clients": RANKED[:8]}))
    code, out = run(emit, root, tmp_path, "c10d10")
    assert code == 1
    assert not out.exists()


@pytest.mark.parametrize("missing", ["p15_stage_winner.json",
                                     "p15_combination_grid.json"])
def test_the_nine_of_ten_stage_stops_when_a_record_is_absent(
        emit, root, tmp_path, missing):
    """A guessed winner emits, trains and reports exactly like a chosen one."""
    (root / "tables" / missing).unlink()
    code, out = run(emit, root, tmp_path, "c10d10")
    assert code == 1
    assert not out.exists()


def test_the_nine_of_ten_stage_seeds_in_a_block_of_its_own(emit, root, tmp_path):
    """
    Twelve, thirteen and fourteen thousand, and no overlap between them.

    The stage runs the same four configurations as the other two ten-client
    points, so without its own block it would draw the same client-sampling
    sequences they did and the three would be correlated with nothing in the
    output to say so.
    """
    spans = {}
    for stage, (_, _, _, block) in STAGES.items():
        code, out = run(emit, root, tmp_path, stage)
        assert code == 0
        seeds = [int(ln.split(" --sampler-seed ")[1].split()[0])
                 for ln in body(out)]
        assert all(block < seed < block + 1000 for seed in seeds), stage
        spans[stage] = set(seeds)
    assert spans["c10d10"] & (spans["five"] | spans["drop20"]) == set()
    assert min(spans["c10d10"]) == CFG.seed_base + 13001


def test_the_dropout_record_states_the_rule(emit, root, tmp_path):
    code, _ = run(emit, root, tmp_path, "drop20")
    assert code == 0
    record = json.loads((root / "tables" / "p19_dropout20.json").read_text())
    assert record["clients_per_round"] == 8
    assert record["dropout"] == 0.2
    assert record["participation_rule"] == "floor((1 - 0.2) * 10) = 8"
    survivors = {c["id"]: c["trim_survivors"] for c in record["configurations"]}
    assert survivors["winner"] == 2
