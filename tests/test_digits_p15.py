"""
P15: the cross, its winner, and the extreme cases the winner is run on.

Two failure modes drive these tests. A combination that pairs a rule chosen on
one schedule with a penalty chosen on the other would report the result as a
property of either, and nothing in the output would say otherwise. And a
generator dispatched by the wrong kind of introspection hands arguments to
functions that never asked for them - which is exactly how ``agg-top3`` came to
be called with a ``family`` it does not take.
"""

from __future__ import annotations

import inspect
import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import agg_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as CFG

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

RETIRED = (
    "v1_superseded", "main_v6", "studies/T", "T1_20outliers", "T1_10outliers",
    "T2_20outliers", "T2_10outliers", "T3_20normals", "T3_10normals",
    "all_writers_digits",
)


@pytest.fixture(autouse=True)
def _shipped_baselines(tmp_path):
    """
    Every synthetic study root carries g-0's own two accuracies.

    Selection is measured against them - what a run added on the new clients,
    less the source knowledge it spent - so a root without them is not a study
    root and the emitters refuse it rather than guess.
    """
    for name, accuracy in (("g0_perfold_evaluations.json", 0.8225),
                           ("g0_evaluations.json", 0.9986)):
        (tmp_path / name).write_text(json.dumps({
            str(fold): {"fold": fold, "part": "test", "accuracy": accuracy}
            for fold in (1, 2, 3, 4, 5)
        }))
    return tmp_path


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture()
def parser():
    from federated_outlier_adaptation.cli import build_parser

    return build_parser()


# --------------------------------------------------------------------------- #
# the dispatch bug
# --------------------------------------------------------------------------- #
def test_generators_are_dispatched_on_parameters_not_locals(emit):
    """
    ``__code__.co_varnames`` lists every local name in a function body.

    ``agg_top3`` has a ``for family in ...`` loop, so it *looked* like it took a
    ``family`` argument and was handed one - a TypeError at the top of a
    selection the owner was waiting on. Signatures, not code objects.
    """
    assert "family" in emit.agg_top3.__code__.co_varnames
    assert "family" not in inspect.signature(emit.agg_top3).parameters
    assert "family" in inspect.signature(emit.reg_patch).parameters


@pytest.mark.parametrize("what", ["agg-top3", "reg-top3"])
def test_a_selection_only_generator_needs_no_expect(emit, tmp_path, what):
    """It writes a json, so there is no line count to promise."""
    assert "expect" in inspect.signature(emit.WHAT[what]).parameters


def test_a_line_emitter_without_expect_refuses(emit, tmp_path):
    """
    A task file with no promised count is one an array can silently outrun.

    The default of 0 is what makes forgetting the flag an error rather than a
    file that happens to be the wrong length.
    """
    assert emit.emit(["a line"], tmp_path / "x.txt", 0, "test", []) == 1


# --------------------------------------------------------------------------- #
# what each generator reads and writes
# --------------------------------------------------------------------------- #
def test_the_derived_io_says_what_the_generators_actually_do(emit):
    """
    The relations the ordering check rests on, stated once and out loud.

    The one that matters is the last pair: the hybrid is built from the winners
    the full-horizon selection writes, which is why packing it first killed a
    stage.
    """
    io = {what: emit.table_io(what) for what in emit.WHAT}
    assert io["agg-full"]["writes"] == ("p11_agg_method_winners.json",)
    assert io["agg-top3"]["writes"] == ("p12_agg_top3.json",)
    assert io["reg-top3"]["writes"] == ("p14_reg_top3.json",)
    assert io["combos"]["reads"] == ("p12_agg_top3.json",
                                     "p14_hybrid_construction.json",
                                     "p14_reg_top3.json")
    assert io["reg-full"]["writes"] == ("p13_reg_method_winners.json",)
    assert io["reg-hybrid"]["reads"] == ("p13_reg_method_winners.json",)
    assert io["reg-hybrid"]["writes"] == ("p14_hybrid_construction.json",)


def test_the_derivation_covers_every_table_this_module_touches(emit):
    """
    The drift guard.

    ``table_io`` reads the module's syntax tree, so it cannot go stale against a
    declaration - but it could still go BLIND, if some later generator reached a
    table through a helper the walk does not follow or a name it cannot see. A
    blind spot here would not fail loudly; it would make the ordering check pass
    on a stage it never actually examined, which is the failure mode that put
    this whole thing in the repository. So every literal table name in the file
    must be accounted for by some generator's reads or writes.
    """
    import ast

    source = ast.parse(Path(emit.__file__).read_text())
    literals, dynamic = set(), 0
    for node in ast.walk(source):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "write_table" and len(node.args) >= 3):
            name = node.args[2]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                literals.add(name.value)
            else:
                dynamic += 1
        literal = emit._tables_literal(node)
        if literal is not None and literal != "BOUNDARY_HITS.txt":
            literals.add(literal)

    accounted = set()
    for what in emit.WHAT:
        io = emit.table_io(what)
        accounted |= set(io["reads"]) | set(io["writes"])
    assert literals - accounted == set(), sorted(literals - accounted)

    # The names the walk deliberately cannot see: the reg patch record, built
    # from an f-string over its method and family, and the size-study records
    # named through a spec dict. Nothing reads either, so they order nothing.
    # If this count moves, a computed name has been added and the blind spot
    # needs re-examining before the ordering check can be trusted over it.
    assert dynamic == 2


def test_the_boundary_log_is_not_mistaken_for_an_input(emit):
    """
    Every generator appends to tables/BOUNDARY_HITS.txt and none selects from
    it. Counted as an input it would make each generator look like a reader of
    a file no generator writes - noise in an ordering check, and the kind of
    noise that gets a check ignored.
    """
    for what in emit.WHAT:
        assert "BOUNDARY_HITS.txt" not in emit.table_io(what)["reads"]


def test_an_ordering_check_over_one_step_is_vacuous_not_broken(emit):
    """A single-step stage has nothing to order, and must not invent a fault."""
    single = ["$FOA_PYTHON t/study_emit.py reg-hybrid --root r --out o --expect 3"]
    assert emit.ordering_violations(single) == []
    assert emit.ordering_violations([]) == []


def test_lines_that_are_not_generators_are_ignored(emit):
    """
    A stage may pack a training line beside a generator. Those name no table,
    and a check that guessed at them would report faults that are not there.
    """
    assert emit.step_what("$FOA_PYTHON t/study_emit.py reg-full --root r") == "reg-full"
    assert emit.step_what("$FOA_PYTHON t/study_emit.py not-a-generator") is None


# --------------------------------------------------------------------------- #
# the top-three artefacts
# --------------------------------------------------------------------------- #
def _agg_result(root: Path, cell_id: str, fold: int, adaptation: float):
    run = (root / f"d01_aggfull_{cell_id}_fold{fold}_x_grid_search" / "s"
           / "concurrent_delta" / "job")
    run.mkdir(parents=True, exist_ok=True)
    (run / "accuracies_0.json").write_text(json.dumps({
        "scenario": "concurrent",
        "accuracies": [[adaptation, 0.5]] * 5,
        "pool_val_accuracies": [adaptation] * 5,
        "final_evaluation": {
            "clients": {"accuracy": adaptation - 0.01},
            "old": {"mean": 0.9, "sd": 0.0},
        },
    }))


def test_the_agg_top_three_are_three_distinct_methods(emit, tmp_path):
    """
    A patched method leaves BOTH its cells with full-horizon results - the
    superseded one is still on disk - so without this the replacement and the
    thing it replaced take two of three slots and get crossed against each
    other. A family's three slots are three ideas about aggregation.
    """
    trimmed = [c["id"] for c in agg_cells.screen_cells()
               if c["id"].startswith("trimmed_")]
    assert len(trimmed) >= 2
    # both trimmed cells score best of all, as a patched method's pair can
    for index, cell in enumerate(agg_cells.screen_cells()):
        if cell["path"] != "concurrent":
            continue
        score = 0.99 if cell["id"] in trimmed else 0.5 + index * 1e-4
        for fold in (1, 2):
            _agg_result(tmp_path, cell["id"], fold, score)
    for cell in agg_cells.screen_cells():
        if cell["path"] == "sequential":
            for fold in (1, 2):
                _agg_result(tmp_path, cell["id"], fold, 0.4)

    out = tmp_path / "top.json"
    assert emit.agg_top3(CFG, tmp_path, out, 0) == 0
    chosen = json.loads(out.read_text())
    by_id = {c["id"]: c for c in agg_cells.screen_cells()}
    methods = [SL.agg_method_of(by_id[i]) for i in chosen["concurrent"]]
    assert len(methods) == len(set(methods)) == 3
    assert sum(1 for i in chosen["concurrent"] if i in trimmed) == 1


def test_the_reg_top_three_let_the_hybrid_compete(emit, tmp_path):
    """
    The blend is not in the screening table - it is priced after both halves.

    Anything ranking full-horizon results has to be told it exists, or it could
    never take a slot it had earned.
    """
    (tmp_path / "tables").mkdir(parents=True)
    (tmp_path / "tables" / "p14_hybrid_construction.json").write_text(json.dumps({
        "kd_winner": "kd_T4_a0p9", "fisher_winner": "fisher_lam1000",
        "family": "concurrent", "mixes": [0.25, 0.5, 0.75],
        "lam": 0.1111, "T": 4.0,
    }))
    hybrids = emit.hybrid_cells(tmp_path)
    assert [c["id"] for c in hybrids] == [
        "hybrid_mix0p25", "hybrid_mix0p5", "hybrid_mix0p75"
    ]
    assert all(c["method"] == "kd+fisher" and c["needs_fisher"] for c in hybrids)


def test_hybrid_cells_are_empty_without_the_record(emit, tmp_path):
    assert emit.hybrid_cells(tmp_path) == []


# --------------------------------------------------------------------------- #
# the cross
# --------------------------------------------------------------------------- #
def _tops(root: Path, agg: dict, reg: dict):
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "tables" / "p12_agg_top3.json").write_text(json.dumps({"top": agg}))
    (root / "tables" / "p14_reg_top3.json").write_text(json.dumps({"top": reg}))


CONC_AGGS = ["anchor_h1", "median", "weight_q0"]
SEQ_AGGS = ["seq_delta_capped", "seq_equal", "seq_order_shuffle"]
CONC_REGS = ["kd_T4_a0p9", "fisher_lam1000", "ntd_b1_t2"]
SEQ_REGS = ["logit_l2_lam1", "feature_l2_lam1", "param_l2_mu0p1"]


def test_the_cross_is_three_by_three_per_family(emit, tmp_path, parser):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    assert emit.combos(CFG, tmp_path, out, 90) == 0
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == SL.counts(CFG)["combos"] == 90

    args = [parser.parse_args(shlex.split(l)[1:]) for l in lines]
    assert len({a.parent for a in args}) == 90
    assert len({a.sampler_seed for a in args}) == 90
    assert {a.fold for a in args} == {1, 2, 3, 4, 5}


def test_the_pairing_never_crosses_families(emit, tmp_path, parser):
    """
    A concurrent rule crossed with a penalty chosen on the cyclic loop would be
    reported as a property of either, and nothing in the output would say so.
    """
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]

    for line in lines:
        parent = line.split(" --parent ")[1].split()[0]
        stem = parent[len("d01_combo_"):].rsplit("_fold", 1)[0]
        conc = any(stem.startswith(a + "_") for a in CONC_AGGS)
        seq = any(stem.startswith(a + "_") for a in SEQ_AGGS)
        assert conc != seq
        if conc:
            assert any(stem.endswith("_" + r) for r in CONC_REGS), stem
        else:
            assert any(stem.endswith("_" + r) for r in SEQ_REGS), stem


def test_each_combination_names_one_family_s_rule(emit, tmp_path, parser):
    """One rule, one family, one result - not --aggregation fedavg."""
    from federated_outlier_adaptation.aggregation.selector import (
        SKIP_FAMILY,
        resolve_aggregation,
    )

    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]

    for line in lines:
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.aggregation != "fedavg"
        families = [
            (scenario, metadata)
            for scenario in ("concurrent", "sequential")
            for metadata in ("weights", "delta")
            if resolve_aggregation(scenario, metadata, args.aggregation,
                                   extended=True) != SKIP_FAMILY
        ]
        assert len(families) == 1, (args.aggregation, families)


def test_the_combination_protocol_is_the_full_horizon(emit, tmp_path, parser):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == 100 and args.epochs == 5 and args.batch_size == 64
        assert args.clients_per_round == 9 and args.policy == "uniform"
        assert args.classes == "digits" and args.old_fold == "all"
        assert args.init == "global" and args.global_name == "g0"
        assert args.outer_workers == 1


def test_the_fisher_path_is_only_where_the_penalty_needs_one(emit, tmp_path, parser):
    from federated_outlier_adaptation.cli import _trainer_overrides

    regs = ["fisher_lam1000", "kd_T4_a0p9", "logit_l2_lam1"]
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": regs, "sequential": regs})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)

    by_id = {c["id"]: c for c in reg_cells.screen_cells()}
    with_fisher = 0
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        stem = args.parent[len("d01_combo_"):].rsplit("_fold", 1)[0]
        reg = next(r for r in regs if stem.endswith("_" + r))
        overrides = _trainer_overrides(args)
        if by_id[reg]["needs_fisher"]:
            with_fisher += 1
            assert "g0_fold$G0_FOLD/global_results/fisher" in overrides["fisher_path"]
        else:
            assert "fisher_path" not in overrides
        assert overrides["anchor"] == "frozen"
    # one of three penalties needs it, on both families, over five folds
    assert with_fisher == 3 * 2 * 5


def test_the_cross_reads_the_artefacts_rather_than_restating_them(emit, tmp_path):
    """Hard-coding the winners here would let the two lists disagree."""
    with pytest.raises(SystemExit, match="run agg-top3 and reg-top3"):
        emit.combos(CFG, tmp_path, tmp_path / "p15.txt", 90)


def test_a_mislabelled_aggregation_is_refused(emit, tmp_path):
    """A sequential rule listed under concurrent means the lists disagree."""
    _tops(tmp_path, {"concurrent": ["seq_equal", "median", "weight_q0"],
                     "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    with pytest.raises(SystemExit, match="the two lists disagree"):
        emit.combos(CFG, tmp_path, tmp_path / "p15.txt", 90)


def test_the_cross_is_stamped_unauthorised(emit, tmp_path):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)
    assert "NOT AUTHORISED" in out.read_text()


def test_no_combination_line_names_a_retired_artefact(emit, tmp_path):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    out = tmp_path / "p15.txt"
    emit.combos(CFG, tmp_path, out, 90)
    text = out.read_text()
    for dead in RETIRED:
        assert dead not in text, dead


def test_a_cross_miscount_halts(emit, tmp_path):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    assert emit.combos(CFG, tmp_path, tmp_path / "p15.txt", 15) == 1


# --------------------------------------------------------------------------- #
# the winner
# --------------------------------------------------------------------------- #
def _combo_result(root: Path, combo_id: str, fold: int, adaptation: float,
                  preservation: float = 0.9):
    run = (root / f"d01_combo_{combo_id}_fold{fold}_x_grid_search" / "s"
           / "concurrent_delta" / "job")
    run.mkdir(parents=True, exist_ok=True)
    (run / "accuracies_0.json").write_text(json.dumps({
        "scenario": "concurrent",
        "accuracies": [[adaptation, 0.5]] * 5,
        "pool_val_accuracies": [adaptation] * 5,
        "final_evaluation": {
            "clients": {"accuracy": adaptation - 0.02},
            "old": {"mean": preservation, "sd": 0.0},
        },
    }))


def test_the_winner_is_crowned_on_validation_adaptation(emit, tmp_path, capsys):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    for agg in CONC_AGGS:
        for reg in CONC_REGS:
            score = 0.95 if (agg, reg) == ("median", "ntd_b1_t2") else 0.5
            for fold in (1, 2):
                _combo_result(tmp_path, f"{agg}_{reg}", fold, score)

    assert emit.stage_winner(CFG, tmp_path, tmp_path / "w.json", 0) == 0
    record = json.loads(
        (tmp_path / "tables" / "p15_stage_winner.json").read_text()
    )
    assert record["winner"] == "median_ntd_b1_t2"
    assert record["aggregation"] == "median" and record["regulariser"] == "ntd_b1_t2"
    assert record["family"] == "concurrent"


def test_the_winner_records_both_columns(emit, tmp_path):
    """
    A winner chosen on adaptation has spent something on preservation.

    A table that reports only the criterion hides the trade it made.
    """
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    for agg in CONC_AGGS:
        for reg in CONC_REGS:
            for fold in (1, 2):
                _combo_result(tmp_path, f"{agg}_{reg}", fold, 0.8, 0.77)
    emit.stage_winner(CFG, tmp_path, tmp_path / "w.json", 0)
    record = json.loads(
        (tmp_path / "tables" / "p15_stage_winner.json").read_text()
    )
    assert record["adaptation"] == pytest.approx(0.8)
    assert record["preservation"] == pytest.approx(0.77)
    assert record["adaptation_test"] == pytest.approx(0.78)
    # the crowning follows the owner's decision now, so the rule string says so
    assert "owner decision A" in record["rule"] and record["rank_by"] == "test"
    assert record["ranked"] and len(record["ranked"]) == record["considered"]


def test_the_protocol_letter_can_still_be_asked_for(emit, tmp_path):
    """``--rank-by val`` restores the letter, and the artefact says which ran."""
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    for agg in CONC_AGGS:
        for reg in CONC_REGS:
            for fold in (1, 2):
                _combo_result(tmp_path, f"{agg}_{reg}", fold, 0.8, 0.77)
    assert emit.stage_winner(CFG, tmp_path, tmp_path / "w.json", 0,
                             rank_by="val") == 0
    record = json.loads(
        (tmp_path / "tables" / "p15_stage_winner.json").read_text()
    )
    assert record["rank_by"] == "val"
    assert "VALIDATION" in record["rule"]


def test_crowning_nothing_is_refused(emit, tmp_path):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    assert emit.stage_winner(CFG, tmp_path, tmp_path / "w.json", 0) == 1


# --------------------------------------------------------------------------- #
# the extreme cases
# --------------------------------------------------------------------------- #
def _winner(root: Path, agg: str, reg: str, family: str = "concurrent"):
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "tables" / "p15_stage_winner.json").write_text(json.dumps(
        {"winner": f"{agg}_{reg}", "family": family,
         "aggregation": agg, "regulariser": reg}
    ))


def _cohort_and_scores(root: Path, worst_last: bool = True):
    clients = [f"w{i:02d}" for i in range(10)]
    (root / "outliers").mkdir(parents=True, exist_ok=True)
    (root / "outliers" / CFG.cohort_file_name).write_text(
        json.dumps({"clients": clients})
    )
    per_fold = {
        f"cohort_fold{fold}": {
            "per_writer": {c: 0.9 - 0.01 * i for i, c in enumerate(clients)}
        }
        for fold in range(1, 6)
    }
    (root / "g0_perfold_evaluations.json").write_text(json.dumps(per_fold))
    return clients


def test_the_extreme_cases_use_the_winner_and_this_studys_g0(emit, tmp_path, parser):
    """
    Ranked by g-0's own per-writer scores, not the coarse detector's.

    The detector ranks by a model common to every study; these are meant to be
    this study's hardest clients under its shipped model.
    """
    _winner(tmp_path, "median", "kd_T4_a0p9")
    clients = _cohort_and_scores(tmp_path)
    out = tmp_path / "p16.txt"
    assert emit.extreme(CFG, tmp_path, out, 15) == 0

    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == SL.counts(CFG)["extreme"] == 15

    worst, second = clients[-1], clients[-2]
    assert json.loads(
        (tmp_path / "outliers" / "extreme_single.json").read_text()) == [worst]
    assert json.loads(
        (tmp_path / "outliers" / "extreme_double.json").read_text()) == [worst, second]
    assert json.loads(
        (tmp_path / "outliers" / "extreme_dual.json").read_text()) == [f"{worst}+{second}"]
    assert (tmp_path / "tables" / "clients_acc_on_g0.json").is_file()

    # --aggregation carries the RULE the cell names, not the cell id
    rule = next(c["rule"] for c in agg_cells.screen_cells() if c["id"] == "median")
    assert rule == "con_delta_median"
    for line in lines:
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.aggregation == rule
        assert args.policy == "all" and args.participation == 1.0
        assert args.clients_per_round is None
        assert args.rounds == 100 and args.classes == "digits"
        assert args.outliers_file.endswith(
            ("extreme_single.json", "extreme_double.json", "extreme_dual.json")
        )


def test_the_extreme_file_is_stamped_and_counted(emit, tmp_path):
    _winner(tmp_path, "median", "kd_T4_a0p9")
    _cohort_and_scores(tmp_path)
    out = tmp_path / "p16.txt"
    emit.extreme(CFG, tmp_path, out, 15)
    assert "NOT AUTHORISED" in out.read_text()
    assert emit.extreme(CFG, tmp_path, tmp_path / "b.txt", 90) == 1


def test_the_extreme_stage_refuses_without_a_winner(emit, tmp_path):
    _cohort_and_scores(tmp_path)
    assert emit.extreme(CFG, tmp_path, tmp_path / "p16.txt", 15) == 1


def test_the_extreme_stage_refuses_an_unscored_cohort_writer(emit, tmp_path):
    _winner(tmp_path, "median", "kd_T4_a0p9")
    _cohort_and_scores(tmp_path)
    (tmp_path / "g0_perfold_evaluations.json").write_text(json.dumps(
        {"cohort_fold1": {"per_writer": {"w00": 0.5}}}
    ))
    assert emit.extreme(CFG, tmp_path, tmp_path / "p16.txt", 15) == 1


def test_double_and_dual_hold_the_same_writers(emit, tmp_path):
    """The controlled pair: same rows, different boundaries."""
    _winner(tmp_path, "median", "kd_T4_a0p9")
    _cohort_and_scores(tmp_path)
    emit.extreme(CFG, tmp_path, tmp_path / "p16.txt", 15)

    from federated_outlier_adaptation.data.merged_clients import members

    double = json.loads((tmp_path / "outliers" / "extreme_double.json").read_text())
    dual = json.loads((tmp_path / "outliers" / "extreme_dual.json").read_text())
    assert members(dual[0]) == double
    assert len(dual) == 1 and len(double) == 2


# --------------------------------------------------------------------------- #
# the selection rule, and the divergence it exists to handle
# --------------------------------------------------------------------------- #
def test_the_two_rules_are_named_and_dated(emit):
    """
    Selecting on test is a departure from the protocol letter, so the artefact
    says so in words rather than leaving a reader to infer it from a flag.
    """
    assert emit.RANK_BY == ("val", "test")
    assert "VALIDATION" in emit.RANK_RULES["val"]
    assert "owner decision A 2026-08-28" in emit.RANK_RULES["test"]
    assert emit.rank_key("val") == "adaptation"
    assert emit.rank_key("test") == "adaptation_test"
    with pytest.raises(SystemExit, match="Unknown --rank-by"):
        emit.rank_key("preservation")


def test_ties_are_broken_by_id_not_by_input_order(emit):
    """
    Not hypothetical: two aggregation methods came out at exactly 0.9107 on the
    test column. Without a rule their order is whatever the input happened to
    be, which is not a property of the data.
    """
    rows = [
        {"id": "zeta", "adaptation": {"mean": 0.5}, "adaptation_test": {"mean": 0.9}},
        {"id": "alpha", "adaptation": {"mean": 0.5}, "adaptation_test": {"mean": 0.9}},
    ]
    assert [r["id"] for r in emit.ranked_by(rows, "adaptation_test")] == ["alpha", "zeta"]
    assert [r["id"] for r in emit.ranked_by(rows[::-1], "adaptation_test")] == [
        "alpha", "zeta"
    ]


def test_the_two_columns_can_disagree_and_both_are_recorded(emit, tmp_path):
    """
    The val/test divergence is itself a finding, so an artefact naming one
    winner carries the other ordering beside it.
    """
    for index, cell in enumerate(agg_cells.screen_cells()):
        if cell["path"] != "concurrent":
            continue
        # validation likes the early cells; test likes the late ones
        run = (tmp_path / f"d01_aggfull_{cell['id']}_fold1_x_grid_search" / "s"
               / "concurrent_delta" / "job")
        run.mkdir(parents=True, exist_ok=True)
        (run / "accuracies_0.json").write_text(json.dumps({
            "scenario": "concurrent",
            "accuracies": [[0.5, 0.5]],
            "pool_val_accuracies": [0.9 - index * 1e-3],
            "final_evaluation": {
                "clients": {"accuracy": 0.5 + index * 1e-3},
                "old": {"mean": 0.9, "sd": 0.0},
            },
        }))
    for cell in agg_cells.screen_cells():
        if cell["path"] == "sequential":
            _agg_result(tmp_path, cell["id"], 1, 0.4)

    out = tmp_path / "top.json"
    assert emit.agg_top3(CFG, tmp_path, out, 0, rank_by="test") == 0
    record = json.loads(
        (tmp_path / "tables" / "p12_agg_top3.json").read_text()
    )
    assert record["rank_by"] == "test"
    assert "owner decision A" in record["rule"]
    both = record["rankings"]["concurrent"]
    assert set(both) == {"val", "test"}
    # the two orderings really are different here
    assert [r["id"] for r in both["val"]] != [r["id"] for r in both["test"]]
    # and the chosen three are the test ordering's
    chosen = json.loads(out.read_text())["concurrent"]
    by_id = {c["id"]: c for c in agg_cells.screen_cells()}
    test_first = [
        r["id"] for r in both["test"]
        if SL.agg_method_of(by_id[r["id"]]) not in set()
    ]
    assert chosen[0] == test_first[0]


def test_val_remains_the_default(emit, tmp_path):
    """The protocol letter is what you get when nobody says otherwise."""
    assert inspect.signature(emit.agg_top3).parameters["rank_by"].default == "val"
    assert inspect.signature(emit.reg_top3).parameters["rank_by"].default == "val"
    # the crowning is the one place the owner's decision is the default
    assert inspect.signature(emit.stage_winner).parameters["rank_by"].default == "test"


def test_the_winner_is_crowned_on_test_and_records_val(emit, tmp_path):
    _tops(tmp_path, {"concurrent": CONC_AGGS, "sequential": SEQ_AGGS},
          {"concurrent": CONC_REGS, "sequential": SEQ_REGS})
    for agg in CONC_AGGS:
        for reg in CONC_REGS:
            # this pair is best on validation and worst on test
            if (agg, reg) == ("anchor_h1", "kd_T4_a0p9"):
                val, test = 0.99, 0.10
            elif (agg, reg) == ("median", "ntd_b1_t2"):
                val, test = 0.20, 0.95
            else:
                val, test = 0.5, 0.5
            for fold in (1, 2):
                run = (tmp_path / f"d01_combo_{agg}_{reg}_fold{fold}_x_grid_search"
                       / "s" / "concurrent_delta" / "job")
                run.mkdir(parents=True, exist_ok=True)
                (run / "accuracies_0.json").write_text(json.dumps({
                    "scenario": "concurrent",
                    "accuracies": [[val, 0.5]],
                    "pool_val_accuracies": [val],
                    "final_evaluation": {
                        "clients": {"accuracy": test},
                        "old": {"mean": 0.9, "sd": 0.0},
                    },
                }))

    assert emit.stage_winner(CFG, tmp_path, tmp_path / "w.json", 0) == 0
    record = json.loads(
        (tmp_path / "tables" / "p15_stage_winner.json").read_text()
    )
    assert record["rank_by"] == "test"
    assert record["winner"] == "median_ntd_b1_t2"
    assert record["adaptation_test"] == pytest.approx(0.95)
    # the validation figure is recorded even though it did not decide
    assert record["adaptation"] == pytest.approx(0.20)
    assert "owner decision A" in record["rule"]
    assert set(record["rankings"]) == {"val", "test"}
    # and under the protocol letter a different combination would have won
    assert record["rankings"]["val"][0]["id"] == "anchor_h1_kd_T4_a0p9"
