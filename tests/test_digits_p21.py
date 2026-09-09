"""
P21: the screen the kd+fisher blend never had.

The blend reached the finals with ``lam`` and ``T`` inherited from the KD winner
and only ``mix`` searched, and then held the best mean score in both schedules.
So the failures worth pinning here are the ones that would let a *second*
unsearched coefficient through: a grid narrower than the rows its parents were
screened over, a strength that does not reach the half it is named for, and a
seed window that quietly overlaps a stage already on disk - which would correlate
this screen with the one it is meant to be read against and say nothing.
"""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01

REPO = Path(__file__).resolve().parents[1]
TOOLS = str(REPO / "tools")
JOBS = REPO / "study" / "artifacts" / "Digits_study01" / "jobs"

#: The retired study roots, as P13 lists them. "T1_" alone is a false positive -
#: a blend cell id carries its temperature - so the retired STUDY names match.
RETIRED = (
    "v1_superseded", "main_v6", "studies/T", "T1_20outliers", "T1_10outliers",
    "T2_20outliers", "T2_10outliers", "T3_20normals", "T3_10normals",
    "all_writers_digits",
)


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p21(tools_path):
    import make_digits_p21

    return make_digits_p21


@pytest.fixture()
def parsed(p21):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    return [parser.parse_args(shlex.split(line)[1:]) for line in p21.lines()]


# --------------------------------------------------------------------------- #
# the grid: the parents' own rows, not a reduced set
# --------------------------------------------------------------------------- #
def test_the_strength_row_is_the_ewc_row_itself():
    """
    Not a copy of it. A row restated is a row that can drift from the one it
    claims to be, and the claim this stage makes is that the blend was screened
    over what EWC was screened over.
    """
    assert reg_cells.BLEND_EWC_LAMS is reg_cells.FISHER_LAMS
    assert reg_cells.BLEND_EWC_LAMS == (
        0.1, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
        1000.0,
    )


def test_the_temperature_row_is_the_kd_row_itself():
    assert reg_cells.BLEND_TEMPERATURES is reg_cells.KD_TEMPERATURES
    assert reg_cells.BLEND_TEMPERATURES == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def test_the_mixes_are_the_ones_the_emitted_blend_used():
    assert reg_cells.BLEND_MIXES is reg_cells.HYBRID_MIXES == (0.25, 0.5, 0.75)


def test_the_grid_is_thirteen_by_six_by_three(p21):
    cells = reg_cells.blend_cells()
    assert len(cells) == 13 * 6 * 3 == 234
    assert p21.SCREENED_CELLS == 234
    assert len({c["id"] for c in cells}) == 234
    assert len(p21.lines()) == 234 * len(SL.FOLDS) == 1170


def test_the_blend_is_still_not_in_the_screen_it_was_left_out_of():
    """
    P13's file has run. An eighth row added to the table it was emitted from
    would move every seed after it and stop s16_reg_screen3.txt regenerating.
    """
    assert len(reg_cells.screen_cells()) == 140
    assert "kd+fisher" not in {c["method"] for c in reg_cells.screen_cells()}
    ids = {c["id"] for c in reg_cells.screen_cells()}
    assert not (ids & {c["id"] for c in reg_cells.blend_cells()})


# --------------------------------------------------------------------------- #
# the conversion: a strength that reaches the half it is named for
# --------------------------------------------------------------------------- #
def test_the_named_lambda_is_what_the_fisher_half_actually_receives():
    """
    The whole point of writing the row in EWC's units. The trainer computes
    lam * (mix * KD + (1 - mix) * Fisher), so the Fisher term's coefficient is
    lam * (1 - mix), and that is the number the cell id names.
    """
    for cell in reg_cells.blend_cells():
        lam_ewc = float(re.match(
            r"blend_lam([0-9p]+)_", cell["id"]
        ).group(1).replace("p", "."))
        got = cell["hypers"]["lam"] * (1.0 - cell["hypers"]["mix"])
        assert got == pytest.approx(lam_ewc, rel=1e-12), cell["id"]


def test_the_conversion_refuses_a_mix_that_is_not_a_blend():
    """mix = 1 is the KD winner and mix = 0 divides by nothing."""
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            reg_cells.blend_lam_of_ewc(1.0, bad)


def test_alpha_is_not_a_knob_of_the_blend():
    """
    AnchoredTrainer has no alpha; kd is WRITTEN in alpha and converted. Inside
    the blend that coefficient is lam * mix, so gridding alpha would grid lam
    twice.
    """
    for cell in reg_cells.blend_cells():
        assert set(cell["hypers"]) == {"lam", "T", "mix"}
    source = (REPO / "src" / "federated_outlier_adaptation" / "trainers"
              / "anchored_trainer.py").read_text()
    signature = source[source.index("def __init__"):source.index("self.lam =")]
    assert "alpha" not in signature


# --------------------------------------------------------------------------- #
# the 1170 lines
# --------------------------------------------------------------------------- #
def test_the_screen_is_every_cell_at_every_fold(parsed):
    assert len(parsed) == 1170
    cells = {c["id"] for c in reg_cells.blend_cells()}
    seen = {(a.parent[len("d01_reg_"):].rsplit("_fold", 1)[0], a.fold)
            for a in parsed}
    assert len(seen) == 1170
    assert {c for c, _ in seen} == cells


def test_every_line_runs_the_screen_protocol(parsed):
    """P13's protocol, unchanged: only the penalty's coefficients move."""
    for args in parsed:
        assert args.classes == "digits" and args.resolution == 28
        assert args.clients_per_round == 8 and args.policy == "uniform"
        assert args.rounds == 25 and args.epochs == 5 and args.batch_size == 64
        assert args.aggregation == "fedavg"       # both families per task
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients
        assert args.old_fold == "all"
        assert args.trainer == "AnchoredTrainer"


def test_the_screen_is_emitted_at_the_search_rate_not_the_study_rate(p21, parsed):
    """The rate is a constant, not a flag - see P13's own test for why."""
    assert p21.CFG.clients_per_round == DIGITS_STUDY01.search_clients_per_round
    assert all(a.clients_per_round == 8 for a in parsed)
    assert not any(
        a.clients_per_round == DIGITS_STUDY01.clients_per_round for a in parsed
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


def test_every_cell_is_the_blend_anchored_on_the_frozen_g0(parsed):
    from federated_outlier_adaptation.cli import _trainer_overrides

    for args in parsed:
        overrides = _trainer_overrides(args)
        assert overrides["space"] == "kd+fisher"
        assert overrides["anchor"] == "frozen"
        assert "param_l2_convention" not in overrides


def test_the_coefficients_reach_the_trainer_unrounded(parsed):
    """A label may round; a coefficient may not - lam is 8/(1-0.75) here."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.blend_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_reg_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        for name, value in cell["hypers"].items():
            assert float(overrides[name]) == float(value), (cell["id"], name)


def test_every_line_names_the_fisher(parsed):
    """Every cell reads it; a missing path is 1170 elements failing at startup."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    for args in parsed:
        assert _trainer_overrides(args)["fisher_path"] == (
            "$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher"
        )


def test_the_lines_are_1170_distinct_cells_with_1170_distinct_seeds(parsed):
    assert len({a.parent for a in parsed}) == 1170
    assert len({a.sampler_seed for a in parsed}) == 1170


def test_no_line_touches_an_archived_or_foreign_artefact(p21):
    text = "\n".join(p21.lines())
    for dead in RETIRED:
        assert dead not in text, dead


# --------------------------------------------------------------------------- #
# the seeds, against everything already on disk
# --------------------------------------------------------------------------- #
def _shipped_seeds() -> set:
    """Every sampler seed in every shipped task file, read rather than listed."""
    seeds = set()
    for path in sorted(JOBS.rglob("*.txt")):
        if path.name == "s21_blend_screen.txt":
            continue
        for value in re.findall(r"--sampler-seed (\d+)", path.read_text()):
            seeds.add(int(value))
    return seeds


def test_no_seed_collides_with_any_other_stage(parsed):
    """
    Read off the files, not from a list. A hand-kept range is exactly what let
    the combination stage draw seeds the regularisation screen had used: two
    stages sharing a client-sampling sequence, correlated, with nothing saying
    so.
    """
    prior = _shipped_seeds()
    assert prior, "no shipped task file carried a sampler seed"
    assert not {a.sampler_seed for a in parsed} & prior


def test_the_seed_window_is_wide_enough_for_the_grid(p21, parsed):
    """
    234 cells span 2340, which is more than the thousand-wide block the scheme
    assumes, so the window has to be opened on width rather than on the next
    free thousand.
    """
    seeds = {a.sampler_seed for a in parsed}
    assert p21.SEED_OFFSET == 22000
    assert min(seeds) == 722001 and max(seeds) == 724335
    assert max(seeds) - min(seeds) > 1000


# --------------------------------------------------------------------------- #
# the shipped files
# --------------------------------------------------------------------------- #
def test_regenerating_reproduces_the_shipped_files_byte_for_byte(p21, tmp_path):
    """
    No flags. A task file that needs an argument remembered by hand is a task
    file nobody can regenerate.
    """
    assert p21.main.__module__ == "make_digits_p21"
    argv = sys.argv[:]
    sys.argv = ["make_digits_p21.py", "--jobs-dir", str(tmp_path)]
    try:
        assert p21.main() == 0
    finally:
        sys.argv = argv
    for name in ("s21_blend_screen.txt", "s21_blend_screen_README.md"):
        assert (tmp_path / name).read_text() == (JOBS / name).read_text(), name


def test_the_shipped_file_holds_1170_runnable_lines_that_parse():
    """The check docs/VERIFY.md runs over the whole jobs directory, here."""
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    text = (JOBS / "s21_blend_screen.txt").read_text()
    lines = [l for l in text.splitlines() if l.strip()
             and not l.strip().startswith("#")]
    assert len(lines) == 1170
    for line in lines:
        assert line.startswith("foa ")
        substituted = (line.replace("$FOA_STUDY_DIR", "/study")
                       .replace("$G0_FOLD", "4"))
        parser.parse_args(shlex.split(substituted)[1:])


def test_the_fisher_fold_is_read_from_the_record_not_remembered(p21):
    """
    The number this stage prints and the number the runner resolves must be the
    same one. They are not in P13: it carries a literal 2 and the record shipped
    beside it says 1, and nothing fails - the runner reads the record, the README
    says something else, and only a reader comparing the two would ever know.
    """
    import json

    record = json.loads(
        (REPO / "study" / "artifacts" / "Digits_study01" / "g0_selection.json")
        .read_text()
    )
    assert p21.G0_FOLD == record["selected_fold"]
    assert f"fold {p21.G0_FOLD}" in p21.readme(p21.lines())


def test_the_readme_places_the_inherited_values_and_the_conversion(p21):
    text = p21.readme(p21.lines())
    assert "0.111111" in text and "0.010101" in text
    assert "lam = lambda_ewc / (1 - mix)" in text
    assert "no export is required" in text.lower()
    assert "25 rounds, not 100" in text


# --------------------------------------------------------------------------- #
# P22: the finals the screen feeds
# --------------------------------------------------------------------------- #
#: The screen's winners re-run at the full horizon: one cell per schedule.
FULL = "s22_blend_full.txt"


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture(autouse=True)
def _shipped_baselines(tmp_path):
    """
    Every synthetic study root carries g-0's own two accuracies.

    The score is what a run ADDED on the cohort less the source knowledge it
    SPENT, so a root without them is not a study root and the emitter refuses it
    rather than guess. Written for every test, as P13's are: a fixture that
    applied only to the selection tests would be one more thing to remember.
    """
    import json

    for name, accuracy in (("g0_perfold_evaluations.json", 0.8225),
                           ("g0_evaluations.json", 0.9986)):
        (tmp_path / name).write_text(json.dumps({
            str(fold): {"fold": fold, "part": "test", "accuracy": accuracy}
            for fold in SL.FOLDS
        }))


def _screen_result(root: Path, cell_id: str, fold: int, family: str,
                   adaptation: float, preservation: float = 0.99):
    """One family's payload of one (cell, fold), where the driver writes it."""
    import json

    run = (root / f"d01_reg_{cell_id}_fold{fold}_x_grid_search" / "fold1_seed_1"
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


def _screened(root: Path, scores=None):
    """The whole blend screen on disk, at scores that separate the cells."""
    for index, cell in enumerate(reg_cells.blend_cells()):
        for family in SL.FAMILIES:
            for fold in (1, 2):
                _screen_result(root, cell["id"], fold, family,
                               (scores or {}).get((cell["id"], family),
                                                  0.5 + index * 1e-4))


def test_the_blend_has_an_emitter_of_its_own(emit):
    """
    ``reg-full`` cannot reach these cells and that is not an oversight.

    ``blend_cells()`` is deliberately outside ``screen_cells()`` - an eighth row
    would move every seed after it and stop the screen's own file regenerating -
    so the loop over the seven methods walks past the blend entirely. Without a
    mode of its own the penalty with 234 screened cells would reach the finals
    only through the three points ``reg-hybrid`` built from inherited ones.
    """
    import inspect

    assert emit.WHAT["blend-full"] is emit.blend_full
    assert "expect" in inspect.signature(emit.blend_full).parameters
    screened = {c["id"] for c in reg_cells.screen_cells()}
    assert not (screened & {c["id"] for c in reg_cells.blend_cells()})


def test_the_finals_are_one_winner_per_schedule(emit, tmp_path):
    """One penalty, two claims: a schedule chooses its own cell."""
    import json

    _screened(tmp_path)
    out = tmp_path / "p22.txt"
    expect = SL.counts(DIGITS_STUDY01)["blend_full"]
    assert expect == len(SL.FAMILIES) * len(SL.FOLDS) == 10
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, out, expect) == 0

    lines = [l for l in out.read_text().splitlines()
             if l and not l.startswith("#")]
    assert len(lines) == expect
    record = json.loads(
        (tmp_path / "tables" / "p21_blend_winners.json").read_text()
    )
    assert set(record) == {f"kd+fisher/{f}" for f in SL.FAMILIES}
    assert all(entry["measured"] for entry in record.values())
    assert all(entry["considered"] == len(reg_cells.blend_cells())
               for entry in record.values())


def test_the_selection_is_the_score_reg_full_selects_on(emit, tmp_path):
    """
    Not adaptation, and not preservation: the trade, at w = 1.

    Selecting on adaptation alone would crown the least constraining cell and
    selecting on preservation the most, and both failures look like a winner.
    The cell built here gives away more adaptation than it buys, so it wins on
    adaptation and must lose on the rule.
    """
    import json

    cells = reg_cells.blend_cells()
    greedy, traded = cells[0]["id"], cells[1]["id"]
    scores = {}
    for family in SL.FAMILIES:
        scores[(greedy, family)] = 0.90     # +8 points bought with -4 of source
        scores[(traded, family)] = 0.89
    _screened(tmp_path, scores)
    for family in SL.FAMILIES:
        for fold in (1, 2):
            _screen_result(tmp_path, greedy, fold, family, 0.90,
                           preservation=0.95)
            _screen_result(tmp_path, traded, fold, family, 0.89,
                           preservation=0.99)
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, tmp_path / "p22.txt",
                           10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p21_blend_winners.json").read_text()
    )
    assert {e["winner"] for e in record.values()} == {traded}


def test_a_selection_over_nothing_is_refused(emit, tmp_path):
    """A screen that never ran would make both winners the row's first cell."""
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, tmp_path / "p22.txt",
                           10) == 1
    assert not (tmp_path / "p22.txt").is_file()


def test_a_blend_miscount_halts(emit, tmp_path):
    _screened(tmp_path)
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, tmp_path / "p22.txt",
                           999) == 1


def test_the_finals_carry_their_schedule_in_the_parent(emit, tmp_path):
    """
    Give every cell one score, so both schedules choose the same cell.

    That is the case that collides: two tasks naming one folder, racing into it,
    and leaving the selector unable to say which schedule it had read.
    """
    _screened(tmp_path, {(c["id"], f): 0.5
                         for c in reg_cells.blend_cells() for f in SL.FAMILIES})
    out = tmp_path / "p22.txt"
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, out, 10) == 0
    parents = [l.split(" --parent ")[1].split()[0]
               for l in out.read_text().splitlines() if l.startswith("foa ")]
    assert len(parents) == len(set(parents)) == 10
    for family in SL.FAMILIES:
        assert sum(family in p for p in parents) == len(SL.FOLDS)


def test_the_emitted_finals_are_the_full_horizon_at_the_search_rate(emit,
                                                                    tmp_path):
    """The winners must be re-run under the conditions they were chosen under."""
    from federated_outlier_adaptation.cli import build_parser

    _screened(tmp_path)
    out = tmp_path / "p22.txt"
    assert emit.blend_full(DIGITS_STUDY01, tmp_path, out, 10) == 0
    parser = build_parser()
    seeds = set()
    for line in out.read_text().splitlines():
        if not line.startswith("foa "):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == SL.FULL_ROUNDS == 100
        assert args.clients_per_round == DIGITS_STUDY01.search_clients_per_round
        assert args.aggregation == "fedavg" and args.old_fold == "all"
        assert args.trainer == "AnchoredTrainer"
        seeds.add(args.sampler_seed)
    assert len(seeds) == 10


def test_the_finals_draw_a_block_of_their_own(emit, tmp_path):
    """
    Read off the shipped files, not from a list. A hand-kept range is what let
    the combination stage draw seeds the regularisation screen had used.
    """
    prior = set()
    for path in sorted(JOBS.rglob("*.txt")):
        if path.name == FULL:
            continue
        prior |= {int(v) for v in
                  re.findall(r"--sampler-seed (\d+)", path.read_text())}
    assert prior
    mine = {int(v) for v in re.findall(
        r"--sampler-seed (\d+)", (JOBS / FULL).read_text())}
    assert emit.BLEND_BLOCK == 26000
    assert mine == set(range(726001, 726006)) | set(range(726011, 726016))
    assert not mine & prior


def test_the_shipped_finals_are_ten_runnable_lines_that_parse():
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    lines = [l for l in (JOBS / FULL).read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert len(lines) == 10
    for line in lines:
        assert line.startswith("foa ")
        parser.parse_args(shlex.split(
            line.replace("$FOA_STUDY_DIR", "/study").replace("$G0_FOLD", "4")
        )[1:])


def test_the_shipped_finals_run_the_cells_the_record_names():
    """
    The file and the record are one selection or they are two claims.

    A task file that named a cell the record does not is a stage reporting a
    result under a selection that did not produce it.
    """
    import json

    record = json.loads(
        (REPO / "study" / "artifacts" / "Digits_study01" / "tables"
         / "p21_blend_winners.json").read_text()
    )
    text = (JOBS / FULL).read_text()
    cells = {c["id"]: c for c in reg_cells.blend_cells()}
    for key, entry in record.items():
        family = key.split("/")[1]
        assert entry["winner"] in cells
        assert f"d01_regfull_{family}_{entry['winner']}_fold1" in text
        assert f"#   {family:<11} {entry['winner']}" in text


def test_the_shipped_finals_are_not_the_blends_already_on_disk():
    """
    s18 and s19 swept mix at an inherited lam and T. These are new arms, and
    their folders have to say so or a selector would read one for the other.
    """
    text = (JOBS / FULL).read_text()
    assert "hybrid_mix" not in text and "hybrid_seq_mix" not in text
    assert "d01_regfull_concurrent_blend_" in text
    assert "d01_regfull_sequential_blend_" in text


def test_the_readme_states_the_rule_and_what_it_chose():
    text = (JOBS / "s22_blend_full_README.md").read_text()
    assert "(adaptation - A_0) - (P_0 - preservation)" in text
    assert "blend_lam0p1_T0p5_mix0p25" in text
    assert "blend_lam0p1_T0p25_mix0p5" in text
    assert "p21_blend_winners.json" in text
    for dead in RETIRED:
        assert dead not in text, dead
