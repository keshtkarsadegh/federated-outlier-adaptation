"""
P13: the regularisation screen, and the selections that follow it.

The screen is 700 runs whose only job is to rank penalties, so the tests are
about the ways a rank can be wrong without looking wrong: a line that does not
run the protocol it claims to, a penalty whose coefficients were rounded on the
way to the command line, a Fisher path that points at nothing, and a selection
whose output does not match the array written to consume it.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

#: The retired study roots. "T1_" on its own is a false positive here - a KD
#: cell id carries its temperature, as in ``kd_T1_a0p1`` - so the retired STUDY
#: names are matched rather than the letter.
RETIRED = (
    "v1_superseded", "main_v6", "studies/T", "T1_20outliers", "T1_10outliers",
    "T2_20outliers", "T2_10outliers", "T3_20normals", "T3_10normals",
    "all_writers_digits",
)

#: Every sampler-seed range this study has already used.
PRIOR_SEEDS = (
    set(range(30001, 30506))        # P09
    | set(range(702001, 703686))    # P11 screen
    | set(range(703691, 703706))    # P11 boundary extension
    | set(range(704001, 704176))    # P12 full
    | set(range(706001, 707256))    # the superseded P13 screen
)


@pytest.fixture(autouse=True)
def _shipped_baselines(tmp_path):
    """
    Every synthetic study root carries g-0's own two accuracies.

    Selection is measured against them - what a run ADDED on the new clients,
    less the source knowledge it SPENT - so a root without them is not a study
    root, and the emitters refuse it rather than guess. The real programme
    writes these in the selection stage, one record per fold.
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
def p13(tools_path):
    import make_digits_p13

    return make_digits_p13


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture()
def parsed(p13):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    # the screen FILE, not the whole table: the boundary extensions were
    # appended to the table afterwards and are submitted as their own file.
    return [
        parser.parse_args(shlex.split(line)[1:]) for line in p13.lines()
    ]


# --------------------------------------------------------------------------- #
# the table and the counts
# --------------------------------------------------------------------------- #


def test_the_hybrid_stays_out_of_the_screen():
    """
    A blend is only worth pricing once each half's own strength is known.

    Screening it would mean guessing three coefficients before any of them was
    measured - a three-dimensional grid for a question that only becomes
    meaningful after the two one-dimensional ones are settled.
    """
    assert "kd+fisher" not in {c["method"] for c in reg_cells.screen_cells()}
    assert reg_cells.HYBRID_MIXES == (0.25, 0.5, 0.75)


def test_the_full_horizon_count_is_per_family():
    """
    A penalty that helps a server average and one that helps a sequential walk
    are different claims, so each method is selected within each schedule.
    """
    assert SL.FAMILIES == ("concurrent", "sequential")
    assert SL.counts(DIGITS_STUDY01)["reg_full"] == 7 * 2 * 5 == 70


# --------------------------------------------------------------------------- #
# the 700 lines
# --------------------------------------------------------------------------- #
def test_the_screen_is_every_cell_at_every_fold(p13, parsed):
    assert len(parsed) == 700
    cells = {c["id"] for c in reg_cells.screen_cells()}
    seen = {(a.parent[len("d01_reg_"):].rsplit("_fold", 1)[0], a.fold) for a in parsed}
    assert len(seen) == 700
    assert {c for c, _ in seen} == cells


def test_every_line_runs_the_screen_protocol(parsed):
    for args in parsed:
        assert args.classes == "digits" and args.resolution == 28
        # the SEARCH rate, from search_clients_per_round - see the rate test
        assert args.clients_per_round == 8 and args.policy == "uniform"
        assert args.rounds == 25 and args.epochs == 5 and args.batch_size == 64
        assert args.aggregation == "fedavg"       # both families per task
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients
        assert args.old_fold == "all"


def test_the_screen_is_emitted_at_the_search_rate_not_the_study_rate(p13, parsed):
    """
    The rate is a constant, not a flag.

    This grid was searched at the study's own 9 of 10 because the
    ``--clients-per-round 8`` the aggregation grid had been given by hand was
    not given to it, and the mismatch reached the finals and the combinations
    before anyone read a rate off a line. The generator now takes
    ``search_clients_per_round`` from the config, so regeneration needs no flag
    - and the two rates are asserted to be DIFFERENT here, because a test that
    passed whichever field was read would not have caught the original error.
    """
    assert DIGITS_STUDY01.search_clients_per_round == 8
    assert DIGITS_STUDY01.clients_per_round == 9
    assert p13.CFG.clients_per_round == DIGITS_STUDY01.search_clients_per_round
    assert all(
        args.clients_per_round == DIGITS_STUDY01.search_clients_per_round
        for args in parsed
    )
    assert not any(
        args.clients_per_round == DIGITS_STUDY01.clients_per_round
        for args in parsed
    )


def test_every_line_reads_the_cohort_and_the_old_book(parsed):
    for args in parsed:
        assert args.outliers_file.endswith("outliers/cohort_worst10.json")
        assert args.fold_book.endswith("fold_books/cohort10.foldbook.npz")
        assert args.old_book.endswith("fold_books/old_data.foldbook.npz")
        assert args.old_clients_file.endswith("outliers/old_data.json")
        for attr in ("results_dir", "outliers_file", "fold_book", "old_book",
                     "old_clients_file"):
            value = getattr(args, attr, None)
            if isinstance(value, str):
                assert value.startswith("$FOA_STUDY_DIR"), (attr, value)


def test_every_penalty_is_anchored_on_the_frozen_g0(parsed):
    """The premise: the penalty measures distance from the shipped model."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    for args in parsed:
        assert args.trainer == "AnchoredTrainer"
        overrides = _trainer_overrides(args)
        assert overrides["anchor"] == "frozen"


def test_the_coefficients_reach_the_trainer_unrounded(parsed):
    """A label may round; a coefficient may not."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.screen_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_reg_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        assert overrides["space"] == cell["space"]
        for name, value in cell["hypers"].items():
            assert float(overrides[name]) == float(value), (cell["id"], name)


def test_the_fedprox_convention_is_on_exactly_the_param_l2_rows(parsed):
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.screen_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_reg_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        if cell["method"] == "param_l2":
            assert overrides["param_l2_convention"] == "fedprox"
        else:
            assert "param_l2_convention" not in overrides


def test_the_fisher_path_is_on_exactly_the_cells_that_need_it(parsed):
    """
    16 cells, 80 tasks - and nowhere else.

    A fisher_path on a cell that does not read one is harmless noise; a missing
    one on a cell that does is eighty array elements failing at startup.
    """
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.screen_cells()}
    with_fisher = 0
    for args in parsed:
        cell = cells[args.parent[len("d01_reg_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        if cell["needs_fisher"]:
            with_fisher += 1
            assert overrides["fisher_path"] == (
                "$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher"
            )
        else:
            assert "fisher_path" not in overrides
    assert with_fisher == 26 * 5 == 130


def test_the_lines_are_700_distinct_cells_with_700_distinct_seeds(parsed):
    assert len({a.parent for a in parsed}) == 700
    assert len({a.sampler_seed for a in parsed}) == 700


def test_no_seed_collides_with_any_earlier_stage(parsed):
    """P09, P11, its extension, P12 and the superseded P13 have all drawn."""
    seeds = {a.sampler_seed for a in parsed}
    assert not seeds & PRIOR_SEEDS
    assert min(seeds) > max(PRIOR_SEEDS)


def test_no_line_touches_an_archived_or_foreign_artefact(p13):
    text = "\n".join(p13.lines())
    for dead in RETIRED:
        assert dead not in text, dead


# --------------------------------------------------------------------------- #
# the Fisher, which must exist where the lines say it does
# --------------------------------------------------------------------------- #
def test_the_runner_derives_the_winning_fold_from_the_record(tools_path):
    """
    A number already written down should not also have to be remembered.

    A forgotten export would expand to an empty path component and fail all
    eighty Fisher tasks at startup, so the runner reads g0_selection.json - and
    refuses a task that needs the fold when it cannot.
    """
    import make_study_sbatch

    text = make_study_sbatch.sbatch_text()
    assert "g0_selection.json" in text
    assert "selected_fold" in text
    assert "export G0_FOLD" in text
    assert "exit 78" in text


def test_the_documented_winner_matches_what_the_lines_expect(p13):
    assert p13.G0_FOLD == 2
    text = p13.readme(p13.lines())
    assert "fold 2" in text or "fold {}".format(p13.G0_FOLD) in text
    assert "no export is required" in text.lower()


# --------------------------------------------------------------------------- #
# the selections
# --------------------------------------------------------------------------- #
def _result(root: Path, prefix: str, cell_id: str, fold: int, adaptation: float,
            family: str, preservation: float = 0.9):
    """One family's payload of one (cell, fold), where the driver writes it."""
    run = (root / f"{prefix}{cell_id}_fold{fold}_x_grid_search" / "fold1_seed_1"
           / f"{family}_delta" / f"base_agg_x_{family}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "accuracies_0.json").write_text(json.dumps({
        "scenario": family,
        "accuracies": [[adaptation, 0.5]] * 25,
        "pool_val_accuracies": [adaptation] * 25,
        "final_evaluation": {
            "clients": {"accuracy": adaptation - 0.01},
            "old": {"mean": preservation, "sd": 0.0},
        },
    }))


def test_the_selection_is_per_method_and_per_family(emit, tmp_path):
    cells = reg_cells.screen_cells()
    for index, cell in enumerate(cells):
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _result(tmp_path, "d01_reg_", cell["id"], fold,
                        0.5 + index * 1e-4, family)
    out = tmp_path / "p14.txt"
    expect = SL.counts(DIGITS_STUDY01)["reg_full"]
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, out, expect) == 0

    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == expect == 70
    record = json.loads(
        (tmp_path / "tables" / "p13_reg_method_winners.json").read_text()
    )
    assert len(record) == 7 * 2
    assert all(e["measured"] for e in record.values())


def test_the_full_horizon_parents_carry_their_family(emit, tmp_path):
    """
    Give every cell the same score, so the same cell wins in both families.

    That is the case that used to collide: two tasks naming one folder, racing
    into it, and leaving the selector unable to say which family it had read.
    """
    for cell in reg_cells.screen_cells():
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _result(tmp_path, "d01_reg_", cell["id"], fold, 0.5, family)
    out = tmp_path / "p14.txt"
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, out, 70) == 0
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    parents = [l.split(" --parent ")[1].split()[0] for l in lines]
    assert len(parents) == len(set(parents)) == 70
    for family in SL.FAMILIES:
        assert sum(family in p for p in parents) == 35


def test_a_full_run_still_trains_both_families(emit, tmp_path):
    """The tag says which family is READ, not which is RUN."""
    cell = reg_cells.screen_cells()[0]
    line = SL.reg_line(DIGITS_STUDY01, cell, 1, SL.FULL_ROUNDS, 1,
                       family="concurrent")
    assert "--aggregation fedavg" in line
    assert "d01_regfull_concurrent_" in line


def test_each_family_is_selected_from_its_own_folders(emit, tmp_path):
    """
    Reading a family's score out of the other family's folder would attribute a
    result to a selection that did not produce it.
    """
    cells = reg_cells.screen_cells()
    by_id = {c["id"]: c for c in cells}
    for cell in cells:
        for tagged in SL.FAMILIES:
            score = 0.4
            if tagged == "concurrent" and cell["method"] == "ntd":
                score = 0.95
            if tagged == "sequential" and cell["method"] == "kd":
                score = 0.95
            for family in SL.FAMILIES:
                _result(tmp_path, f"d01_regfull_{tagged}_", cell["id"], 1,
                        score, family)
    out = tmp_path / "top.json"
    assert emit.reg_top3(DIGITS_STUDY01, tmp_path, out, 0) == 0
    chosen = json.loads(out.read_text())
    assert by_id[chosen["concurrent"][0]]["method"] == "ntd"
    assert by_id[chosen["sequential"][0]]["method"] == "kd"
    for family in SL.FAMILIES:
        assert len({by_id[i]["method"] for i in chosen[family]}) == 3


def test_a_reg_miscount_halts(emit, tmp_path):
    for cell in reg_cells.screen_cells():
        for family in SL.FAMILIES:
            _result(tmp_path, "d01_reg_", cell["id"], 1, 0.5, family)
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p14.txt", 999) == 1


def test_the_emitted_p14_is_stamped_unauthorised(emit, tmp_path):
    for cell in reg_cells.screen_cells():
        for family in SL.FAMILIES:
            _result(tmp_path, "d01_reg_", cell["id"], 1, 0.5, family)
    out = tmp_path / "p14.txt"
    emit.reg_full(DIGITS_STUDY01, tmp_path, out, 70)
    text = out.read_text()
    assert "NOT YET AUTHORISED" in text
    assert "gated separately" in text


def test_the_emitted_p14_lines_are_the_full_horizon(emit, tmp_path):
    from federated_outlier_adaptation.cli import build_parser

    for cell in reg_cells.screen_cells():
        for family in SL.FAMILIES:
            _result(tmp_path, "d01_reg_", cell["id"], 1, 0.5, family)
    out = tmp_path / "p14.txt"
    emit.reg_full(DIGITS_STUDY01, tmp_path, out, 70)

    parser = build_parser()
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == SL.FULL_ROUNDS == 100
        # the finals must match the screen they were selected from, so they
        # take search_clients_per_round too
        assert args.clients_per_round == 8 and args.classes == "digits"
        assert args.old_fold == "all" and args.aggregation == "fedavg"


# --------------------------------------------------------------------------- #
# the hybrid
# --------------------------------------------------------------------------- #
def _selection(root: Path, kd_id: str, fisher_id: str):
    root.joinpath("tables").mkdir(parents=True, exist_ok=True)
    (root / "tables" / "p13_reg_method_winners.json").write_text(json.dumps({
        "kd/concurrent": {"winner": kd_id, "measured": True},
        "fisher/concurrent": {"winner": fisher_id, "measured": True},
    }))


def test_the_hybrid_is_built_from_the_two_winners(emit, tmp_path):
    _selection(tmp_path, "kd_T8_a0p3", "fisher_lam1000")
    out = tmp_path / "hybrid.txt"
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, out, 15) == 0

    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 3 * 5 == 15
    assert all("space=kd+fisher" in l for l in lines)
    assert all("fisher_path=" in l for l in lines)
    for mix in ("0.25", "0.5", "0.75"):
        assert sum(f"mix={mix}" in l for l in lines) == 5

    record = json.loads(
        (tmp_path / "tables" / "p14_hybrid_construction.json").read_text()
    )
    assert record["kd_winner"] == "kd_T8_a0p3"
    assert record["fisher_winner"] == "fisher_lam1000"
    # T and lam come from the KD half, so mix = 1 would reproduce it exactly
    kd = {c["id"]: c for c in reg_cells.screen_cells()}["kd_T8_a0p3"]
    assert record["T"] == kd["hypers"]["T"]
    assert record["lam"] == kd["hypers"]["lam"]


def test_the_hybrid_lines_are_the_full_horizon_and_parse(emit, tmp_path):
    from federated_outlier_adaptation.cli import build_parser

    _selection(tmp_path, "kd_T4_a0p9", "fisher_lam1000")
    out = tmp_path / "hybrid.txt"
    emit.reg_hybrid(DIGITS_STUDY01, tmp_path, out, 15)

    parser = build_parser()
    for line in out.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == 100 and args.classes == "digits"
        assert args.clients_per_round == 9 and args.old_fold == "all"


def test_the_hybrid_refuses_without_a_selection(emit, tmp_path):
    """It is built from winners; without them there is nothing to blend."""
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, tmp_path / "h.txt", 15) == 1


def test_the_hybrid_refuses_a_half_it_has_no_winner_for(emit, tmp_path):
    tmp_path.joinpath("tables").mkdir(parents=True)
    (tmp_path / "tables" / "p13_reg_method_winners.json").write_text(json.dumps(
        {"kd/concurrent": {"winner": "kd_T4_a0p9"}}
    ))
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, tmp_path / "h.txt", 15) == 1


def test_the_hybrid_is_stamped_unauthorised(emit, tmp_path):
    _selection(tmp_path, "kd_T4_a0p9", "fisher_lam1000")
    out = tmp_path / "hybrid.txt"
    emit.reg_hybrid(DIGITS_STUDY01, tmp_path, out, 15)
    assert "NOT YET AUTHORISED" in out.read_text()


def test_a_hybrid_miscount_halts(emit, tmp_path):
    _selection(tmp_path, "kd_T4_a0p9", "fisher_lam1000")
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, tmp_path / "h.txt", 5) == 1


# --------------------------------------------------------------------------- #
# a selection over nothing must not emit
# --------------------------------------------------------------------------- #
def test_reg_full_refuses_when_the_screen_measured_nothing(emit, tmp_path):
    """
    The screen failed once and this emitter ran against the empty result.

    Every winner came from the fallback - the first cell of each row, chosen by
    no evidence at all - and the file that came out was indistinguishable from a
    real one at a glance. That is what makes it dangerous: it would have run,
    produced numbers, and reported a grid's arbitrary first element as a
    selection.
    """
    out = tmp_path / "p14.txt"
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, out, 70) == 1
    assert not out.exists()
    assert not (tmp_path / "tables" / "p13_reg_method_winners.json").exists()


def test_agg_full_refuses_when_the_screen_measured_nothing(emit, tmp_path):
    out = tmp_path / "p12.txt"
    assert emit.agg_full(DIGITS_STUDY01, tmp_path, out, 90) == 1
    assert not out.exists()
    assert not (tmp_path / "tables" / "p11_agg_method_winners.json").exists()


def test_a_partial_screen_is_refused_unless_the_caller_says_otherwise(emit, tmp_path):
    """
    A partial selection is not obviously wrong, so it is not silently accepted.

    Some winners would be real and some fallbacks, and nothing in the emitted
    file would say which.
    """
    cells = reg_cells.screen_cells()
    kd_only = [c for c in cells if c["method"] == "kd"]
    for cell in kd_only:
        for family in SL.FAMILIES:
            _result(tmp_path, "d01_reg_", cell["id"], 1, 0.6, family)

    out = tmp_path / "p14.txt"
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, out, 70) == 1
    assert not out.exists()

    assert emit.reg_full(
        DIGITS_STUDY01, tmp_path, out, 70, allow_unmeasured=True
    ) == 0
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 70


def test_the_refusal_names_how_much_was_measured(emit, tmp_path, capsys):
    cells = [c for c in reg_cells.screen_cells() if c["method"] == "ntd"]
    for cell in cells:
        _result(tmp_path, "d01_reg_", cell["id"], 1, 0.6, "concurrent")
    capsys.readouterr()
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p14.txt", 70) == 1
    err = capsys.readouterr().err
    assert "measured 1 of 14" in err
    assert "--allow-unmeasured" in err


def test_the_hybrid_refuses_halves_that_were_never_measured(emit, tmp_path):
    """A blend of two fallbacks is not a blend of anything."""
    tmp_path.joinpath("tables").mkdir(parents=True)
    (tmp_path / "tables" / "p13_reg_method_winners.json").write_text(json.dumps({
        "kd/concurrent": {"winner": "kd_T4_a0p9", "measured": False},
        "fisher/concurrent": {"winner": "fisher_lam1000", "measured": False},
    }))
    out = tmp_path / "hybrid.txt"
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, out, 15) == 1
    assert not out.exists()


def test_the_hybrid_refuses_when_only_one_half_was_measured(emit, tmp_path):
    tmp_path.joinpath("tables").mkdir(parents=True)
    (tmp_path / "tables" / "p13_reg_method_winners.json").write_text(json.dumps({
        "kd/concurrent": {"winner": "kd_T4_a0p9", "measured": True},
        "fisher/concurrent": {"winner": "fisher_lam1000", "measured": False},
    }))
    assert emit.reg_hybrid(DIGITS_STUDY01, tmp_path, tmp_path / "h.txt", 15) == 1


def test_the_widened_ranges_are_what_boundary_detection_now_sees(emit):
    """
    T=1 was the low edge of [1..50]; it is interior to [0.25..50].

    If the extension confirms T=0.25 that is a genuine edge finding rather than
    an artefact of where the row happened to stop.
    """
    siblings = reg_cells.cells_by_method()["kd"]
    by_id = {c["id"]: c for c in siblings}

    assert not any(
        "T=1 is the LOW end" in h
        for h in emit.numeric_boundary(by_id["kd_T1_a0p99"], siblings, "hypers")
    )
    low = emit.numeric_boundary(by_id["kd_T0p25_a0p99"], siblings, "hypers")
    assert any("HIGH end" in h or "LOW end" in h for h in low)


# --------------------------------------------------------------------------- #
# the per-pair patch
# --------------------------------------------------------------------------- #
def _row_results(root: Path, method: str, family: str, best: str,
                 other_families: bool = True):
    """Screen results where ``best`` is the strongest cell of one row."""
    for cell in reg_cells.screen_cells():
        score = 0.9 if cell["id"] == best else 0.4
        families = SL.FAMILIES if other_families else (family,)
        for fam in families:
            for fold in (1, 2):
                _result(root, "d01_reg_", cell["id"], fold, score, fam)


def test_the_patch_is_five_lines_for_one_pair(emit, tmp_path):
    _row_results(tmp_path, "kd", "concurrent", "kd_T0p25_a0p99")
    out = tmp_path / "patch.txt"
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, out, 5,
                          method="kd", family="concurrent") == 0
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 5
    assert all("kd_T0p25_a0p99" in l for l in lines)
    assert all("d01_regfull_concurrent_" in l for l in lines)


def test_the_patch_carries_its_p14_siblings_seeds(emit, tmp_path):
    """A patched line and the line it replaces must be the same run."""
    from federated_outlier_adaptation.cli import build_parser

    for cell in reg_cells.screen_cells():
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _result(tmp_path, "d01_reg_", cell["id"], fold, 0.5, family)
    full = tmp_path / "p14.txt"
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, full, 70) == 0
    patch = tmp_path / "patch.txt"
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, patch, 5,
                          method="ntd", family="sequential") == 0

    parser = build_parser()
    p14 = [
        parser.parse_args(shlex.split(l)[1:])
        for l in full.read_text().splitlines() if l and not l.startswith("#")
    ]
    patched = [
        parser.parse_args(shlex.split(l)[1:])
        for l in patch.read_text().splitlines() if l and not l.startswith("#")
    ]
    sibling = [a for a in p14 if "regfull_sequential_ntd_" in a.parent]
    assert len(sibling) == len(patched) == 5
    assert {a.sampler_seed for a in patched} == {a.sampler_seed for a in sibling}
    assert {a.rounds for a in patched} == {100}


def test_the_patch_says_whether_the_winner_moved(emit, tmp_path, capsys):
    for cell in reg_cells.screen_cells():
        score = 0.9 if cell["id"] == "kd_T1_a0p99" else 0.4
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _result(tmp_path, "d01_reg_", cell["id"], fold, score, family)
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p14.txt", 70) == 0
    capsys.readouterr()

    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, tmp_path / "patch.txt", 5,
                          method="kd", family="concurrent") == 0
    out = capsys.readouterr().out
    assert "UNCHANGED" in out
    record = json.loads(
        (tmp_path / "tables" / "p13ext_kd_concurrent_reselection.json").read_text()
    )
    assert record["changed"] is False
    assert record["winner"] == record["previous_winner"] == "kd_T1_a0p99"
    # the kd row as it stands now: a re-selection ranks the current row,
    # which is the whole point of running one.
    assert record["cells_in_row"] == 42


def test_the_patch_reports_a_moved_winner(emit, tmp_path, capsys):
    """The first selection saw only the short row; the low end changes it."""
    for cell in reg_cells.screen_cells():
        if cell["id"].startswith(("kd_T0p5_", "kd_T0p25_")):
            continue
        score = 0.9 if cell["id"] == "kd_T1_a0p99" else 0.4
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _result(tmp_path, "d01_reg_", cell["id"], fold, score, family)
    assert emit.reg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p14.txt", 70,
                         allow_unmeasured=True) == 0
    capsys.readouterr()

    for family in SL.FAMILIES:
        for fold in (1, 2):
            _result(tmp_path, "d01_reg_", "kd_T0p25_a0p99", fold, 0.99, family)
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, tmp_path / "patch.txt", 5,
                          method="kd", family="concurrent") == 0

    out = capsys.readouterr().out
    assert "CHANGED" in out
    record = json.loads(
        (tmp_path / "tables" / "p13ext_kd_concurrent_reselection.json").read_text()
    )
    assert record["previous_winner"] == "kd_T1_a0p99"
    assert record["winner"] == "kd_T0p25_a0p99" and record["changed"] is True


def test_the_patch_is_stamped_unauthorised(emit, tmp_path):
    _row_results(tmp_path, "ntd", "concurrent", "ntd_b0p01_t0p5")
    out = tmp_path / "patch.txt"
    emit.reg_patch(DIGITS_STUDY01, tmp_path, out, 5,
                   method="ntd", family="concurrent")
    assert "NOT AUTHORISED" in out.read_text()


def test_the_patch_refuses_an_unmeasured_pair(emit, tmp_path):
    """A selection over nothing would emit a row's arbitrary first cell."""
    out = tmp_path / "patch.txt"
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, out, 5,
                          method="kd", family="concurrent") == 1
    assert not out.exists()


def test_the_patch_refuses_an_unknown_method_or_family(emit, tmp_path):
    _row_results(tmp_path, "kd", "concurrent", "kd_T1_a0p99")
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, tmp_path / "a.txt", 5,
                          method="nonsense", family="concurrent") == 1
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, tmp_path / "b.txt", 5,
                          method="kd", family="diagonal") == 1


def test_a_patch_miscount_halts(emit, tmp_path):
    _row_results(tmp_path, "kd", "concurrent", "kd_T1_a0p99")
    assert emit.reg_patch(DIGITS_STUDY01, tmp_path, tmp_path / "p.txt", 70,
                          method="kd", family="concurrent") == 1


def test_top3_ranks_a_patched_result_against_the_one_it_supersedes(emit, tmp_path):
    """
    reg-top3 needs no special handling for patches.

    It ranks whatever full-horizon folders exist, so a patched result simply
    competes with the one it replaces and the better one wins.
    """
    by_id = {c["id"]: c for c in reg_cells.screen_cells()}
    for cell in reg_cells.screen_cells():
        for tagged in SL.FAMILIES:
            for family in SL.FAMILIES:
                _result(tmp_path, f"d01_regfull_{tagged}_", cell["id"], 1, 0.5,
                        family)
    # the patched kd cell lands later and scores better
    for family in SL.FAMILIES:
        _result(tmp_path, "d01_regfull_concurrent_", "kd_T0p25_a0p99", 1, 0.99,
                family)

    out = tmp_path / "top.json"
    assert emit.reg_top3(DIGITS_STUDY01, tmp_path, out, 0) == 0
    chosen = json.loads(out.read_text())
    assert chosen["concurrent"][0] == "kd_T0p25_a0p99"
    assert by_id[chosen["concurrent"][0]]["method"] == "kd"


def test_the_screen_is_the_whole_re_ranged_table(p13):
    """
    140 cells, and the file is all of them.

    The superseded screen held its two edge probes in an appended tail; those
    probes are inside the kd and ntd rows now, so there is no tail to slice.
    """
    assert len(reg_cells.screen_cells()) == 140
    assert p13.SCREENED_CELLS == 140
    assert len(p13.lines()) == 700
    assert SL.counts(DIGITS_STUDY01)["reg_screen"] == 700
    assert SL.REG_METHODS == [
        "param_l2", "fisher", "fisher_scaled", "logit_l2", "feature_l2",
        "kd", "ntd",
    ]
