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
