"""
P25/P26: the joint tuning of each schedule's SELECTED pair - **an extension**.

P23 tuned the pair that leads each schedule by TEST score. The study's own
aggregation shortlist was cut on VALIDATION - ``p12_agg_top3.json`` says so in
its own ``rank_by`` - and on the parallel schedule the two orderings disagree
about which rule comes first. So P23 answered the joint-grid question for a rule
the programme did not choose, and these two stages answer it for the rule it
did.

Three things have to hold and none of them fails loudly on its own.

The first is that this is still an extension. Two joint grids leaking into a
core catalogue would move seeds under files that have already run and put cells
the paper never screened into shortlists the paper reports from.

The second is that the two grids are ONE grid with one thing changed. The
penalty rows, the mapping onto the trainer's coefficients and the deduplication
are shared as code; if this stage started carrying a copy of them, a correction
would reach one grid and miss the other and the two would stop being comparable
without either failing.

The third is that the pair is READ. The rule half comes off the record, on the
basis the record names, and a stage that typed it instead would keep tuning
``anchor_h2`` long after a re-cut shortlist had stopped selecting it.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import agg_cells
from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01

REPO = Path(__file__).resolve().parents[1]
TOOLS = str(REPO / "tools")
STUDY = REPO / "study" / "artifacts" / "Digits_study01"
JOBS = STUDY / "jobs"
TABLES = STUDY / "tables"

SCREEN = "s25_combo_screen_selected.txt"
FULL = "s26_combo_full_selected.txt"

#: Every EXTENSION task file, P23's two included. The seed scheme's claim is
#: about where the extensions sit relative to the programme, so the programme
#: is what a window is measured against and these are subtracted.
EXTENSION_FILES = ("s23_combo_screen.txt", "s24_combo_full.txt", SCREEN, FULL)

#: The retired study roots, as P13, P21 and P23 list them.
RETIRED = (
    "v1_superseded", "main_v6", "studies/T", "T1_20outliers", "T1_10outliers",
    "T2_20outliers", "T2_10outliers", "T3_20normals", "T3_10normals",
    "all_writers_digits",
)

#: The shipped combination lines this stage copies every non-grid flag from.
SHIPPED_PAIR = {
    "concurrent": "d01_combo_anchor_h2_hybrid_seq_mix0p5_fold1",
    "sequential": "d01_combo_seq_fedavg_hybrid_mix0p5_fold1",
}


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p25(tools_path):
    import make_digits_p25

    return make_digits_p25


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture()
def parsed(p25):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    return [parser.parse_args(shlex.split(line)[1:]) for line in p25.lines()]


# --------------------------------------------------------------------------- #
# it is an extension, and the core cannot see it
# --------------------------------------------------------------------------- #
def test_the_second_grid_is_in_no_catalogue_the_core_reads():
    """
    The same failure P23 documents, with one more catalogue to stay out of.

    ``s16_reg_screen3.txt`` and ``s21_blend_screen.txt`` both seed as
    ``base + block + index * 10 + fold``, so a cell inserted into either
    catalogue moves every seed after it and stops that file regenerating. And
    this grid must stay out of P23's too: 216 cells added there would move
    1,080 seeds under a stage that has already run.
    """
    tuned = {cell["id"] for cell in reg_cells.combo_tune_selected_cells()}
    assert not (tuned & {c["id"] for c in reg_cells.screen_cells()})
    assert not (tuned & {c["id"] for c in reg_cells.blend_cells()})
    assert not (tuned & {c["id"] for c in reg_cells.combo_tune_cells()})
    assert len(reg_cells.screen_cells()) == 140
    assert len(reg_cells.blend_cells()) == 234
    assert len(reg_cells.combo_tune_cells()) == 216


def test_the_two_grids_cannot_read_each_others_runs():
    """
    Both are scanned back out of one results root by stem, so a stem that is a
    prefix of the other's would hand one selector 432 cells of two different
    pairs and let it crown one of them.

    The four stems are checked as strings rather than as folder names because
    that is how ``reg_selector.collect`` and ``study_record.counted`` use them.
    """
    stems = [f"d01_{tag}_" for tag in (SL.CTUNE_SCREEN_TAG, SL.CTUNE_FULL_TAG,
                                       SL.CTUNE_SEL_SCREEN_TAG,
                                       SL.CTUNE_SEL_FULL_TAG)]
    assert len(set(stems)) == 4
    for one in stems:
        for other in stems:
            if one is not other:
                assert not one.startswith(other), (one, other)


def test_the_counts_register_the_second_extension_and_say_that_it_is_one():
    counts = SL.counts(DIGITS_STUDY01)
    assert counts["combo_tune_selected_screen"] == 216 * 5 == 1080
    assert counts["combo_tune_selected_full"] == len(SL.FAMILIES) * 5 == 10
    assert SL.EXTENSION_COUNTS == (
        "combo_tune_screen", "combo_tune_full",
        "combo_tune_selected_screen", "combo_tune_selected_full")
    for name in SL.EXTENSION_COUNTS:
        assert name in counts


def test_every_shipped_file_says_it_is_an_extension():
    """A stage that is not part of the study has to say so where it is read."""
    for name in (SCREEN, "s25_combo_screen_selected_README.md"):
        text = (JOBS / name).read_text()
        assert "EXTENSION" in text or "extension" in text, name
    chain = REPO / "study" / "jobs" / "digits_s25_combo_screen_selected.chain.toml"
    assert "extension" in chain.read_text().lower()


# --------------------------------------------------------------------------- #
# the pair is read off the record, not typed
# --------------------------------------------------------------------------- #
def test_the_rule_half_is_the_head_of_the_records_own_ranking(p25):
    """
    On the basis the record names, and not on the other one.

    The record holds both orderings, and on the parallel schedule they disagree
    - which is the entire reason this stage exists. A stage that read the test
    order would be P23 again under a different name, and nothing would fail.
    """
    record = json.loads((TABLES / "p12_agg_top3.json").read_text())
    assert record["rank_by"] == "val"
    for family in SL.FAMILIES:
        head = record["rankings"][family]["val"][0]["id"]
        assert p25.THE_PAIR[family][0] == head, family
    assert (record["rankings"]["concurrent"]["val"][0]["id"]
            != record["rankings"]["concurrent"]["test"][0]["id"])


def test_the_two_grids_differ_in_the_rule_and_in_nothing_else(p25):
    """
    One grid with one thing changed, or the comparison is between two studies.

    The penalty half is P23's own, read out of its module; the four penalty rows
    are the same objects, not equal copies of them.
    """
    import make_digits_p23 as p23

    for family in SL.FAMILIES:
        assert p25.THE_PAIR[family][1] == p23.THE_PAIR[family][1]
        assert p25.THE_PAIR[family][0] != p23.THE_PAIR[family][0]
    assert p25.SCREENED_CELLS == p23.SCREENED_CELLS == 216


def test_the_grid_is_built_around_the_rule_the_record_selected(p25):
    """
    The record names a cell id and the grid names an ``--aggregation`` and a
    knob. Nothing makes those agree except the generator's own check, and a
    disagreement is silent: every line would still be legal and still run.
    """
    p25._check_the_grid_is_the_selected_pair()
    spec = reg_cells.CTUNE_SEL_RULES["concurrent"]
    ids = {spec.agg_id.format(reg_cells._fmt(v)) for v in spec.values}
    assert p25.THE_PAIR["concurrent"][0] in ids
    assert reg_cells.CTUNE_SEL_RULES["sequential"].agg_id == \
        p25.THE_PAIR["sequential"][0]


def test_a_record_that_selected_nothing_stops_the_stage(p25, tmp_path,
                                                        monkeypatch):
    """
    A shortlist with no ranking under its own basis is not a shortlist, and the
    stage refuses rather than falling back to whichever rule is first on disk.
    """
    (tmp_path / "p12_agg_top3.json").write_text(json.dumps(
        {"rank_by": "val", "rankings": {"concurrent": {"test": [{"id": "x"}]}}}))
    monkeypatch.setattr(p25, "_TABLES", tmp_path)
    with pytest.raises(SystemExit):
        p25.selected_rule("concurrent")


# --------------------------------------------------------------------------- #
# the grid, and the rule row that leaves the programme
# --------------------------------------------------------------------------- #
def test_the_penalty_rows_are_the_owners_and_are_shared_with_p23():
    assert reg_cells.CTUNE_EWC_COEFFS == (0.05, 0.1, 0.3)
    assert reg_cells.CTUNE_KD_COEFFS == (0.05, 0.11, 0.2)
    assert reg_cells.CTUNE_TEMPERATURES == (0.25, 2.0)
    assert reg_cells.CTUNE_MIXES == (0.25, 0.5, 0.75)
    assert reg_cells.CTUNE_MIXES is reg_cells.HYBRID_MIXES


def test_the_anchor_row_brackets_the_selection_and_leaves_the_screened_range():
    """
    The finding to state, not to smooth over.

    ``anchor_h2`` was selected at the TOP of ``ANCHOR_HALFLIVES_R``, which stops
    at 2R because beyond it the anchor never acts inside the run. So a bracket
    around the selection has no neighbour above inside anything the programme
    screened, and the one this grid uses steps outside it - to a WEAKER
    intervention than every value the screen tried, not a stronger one.
    """
    row = reg_cells.CTUNE_SEL_ANCHOR_HALFLIVES_R
    assert row == (1.0, 2.0, 4.0)
    assert max(agg_cells.ANCHOR_HALFLIVES_R) == 2.0
    assert 2.0 in row and 1.0 in agg_cells.ANCHOR_HALFLIVES_R
    assert 4.0 not in agg_cells.ANCHOR_HALFLIVES_R
    assert max(row) > max(agg_cells.ANCHOR_HALFLIVES_R)


def test_the_grid_is_one_hundred_and_sixty_two_plus_fifty_four(p25):
    cells = reg_cells.combo_tune_selected_cells()
    by_family = reg_cells.combo_tune_selected_by_family()
    assert len(by_family["concurrent"]) == 3 * 3 * 2 * 3 * 3 == 162
    assert len(by_family["sequential"]) == 3 * 3 * 2 * 3 == 54
    assert len(cells) == 216 == p25.SCREENED_CELLS
    assert len({c["id"] for c in cells}) == 216
    assert len(p25.lines()) == 216 * len(SL.FOLDS) == 1080


def test_the_mapping_puts_both_named_coefficients_on_their_own_half():
    """
    The whole correctness of either grid, asserted cell by cell on this one.

    The trainer computes lam * (mix * KD + (1 - mix) * Fisher), so the KD term's
    coefficient is lam * mix and the Fisher term's is lam * (1 - mix). Those must
    be m * c_kd and (1 - m) * c_ewc, or the grid is sweeping something other than
    what it names - and a line carrying the wrong coefficient is still legal,
    still parses and still runs.
    """
    for cell in reg_cells.combo_tune_selected_cells():
        dials, hypers = cell["dials"], cell["hypers"]
        lam, mix, m = hypers["lam"], hypers["mix"], dials["m"]
        assert lam * mix == pytest.approx(m * dials["c_kd"], rel=1e-12)
        assert lam * (1.0 - mix) == pytest.approx(
            (1.0 - m) * dials["c_ewc"], rel=1e-12)
        assert hypers["T"] == dials["T"]


def test_no_two_cells_hand_the_trainer_the_same_experiment(p25):
    """Deduplication is on the COEFFICIENTS, because the dials are what differ."""
    seen = set()
    for cell in reg_cells.combo_tune_selected_cells():
        key = (cell["family"],
               cell["agg"]["flags"].get("anchor_halflife_r"),
               round(cell["hypers"]["lam"], 12),
               round(cell["hypers"]["mix"], 12), cell["hypers"]["T"])
        assert key not in seen, cell["id"]
        seen.add(key)
    assert reg_cells.combo_tune_selected_duplicates() == 0 == p25.DUPLICATES


def test_the_cell_id_names_the_dials_and_the_schedule():
    """
    Not the trainer's coefficients, and not the anchor coefficient either: the
    half-life is what the cell stores, because the coefficient is a function of
    the horizon as well. The schedule rides on the ``_h`` suffix, which only the
    parallel rule can have.
    """
    for cell in reg_cells.combo_tune_selected_cells():
        dials = cell["dials"]
        expected = (f"ctunesel_ewc{reg_cells._fmt(dials['c_ewc'])}"
                    f"_kd{reg_cells._fmt(dials['c_kd'])}"
                    f"_T{reg_cells._fmt(dials['T'])}"
                    f"_mix{reg_cells._fmt(dials['m'])}")
        if cell["family"] == "concurrent":
            expected += f"_h{reg_cells._fmt(dials['anchor_halflife_r'])}"
        assert cell["id"] == expected
        assert ("_h" in cell["id"]) == (cell["family"] == "concurrent")
    assert "ctunesel_ewc0p1_kd0p11_T2_mix0p5_h2" in {
        c["id"] for c in reg_cells.combo_tune_selected_cells()
    }


def test_the_paper_can_name_every_cell():
    """A code id must never reach a page; a gap in the map is a build failure."""
    sys.path.insert(0, str(REPO / "tools" / "paper_figures"))
    import paper_names

    for cell in reg_cells.combo_tune_selected_cells():
        assert paper_names.label(cell["id"])
        assert paper_names.short(cell["id"])
        assert paper_names.params(cell["id"])
    parallel = "ctunesel_ewc0p1_kd0p11_T2_mix0p5_h2"
    cyclic = "ctunesel_ewc0p1_kd0p11_T2_mix0p5"
    assert r"h{=}2R" in paper_names.params(parallel)
    assert r"h{=}" not in paper_names.params(cyclic)
    assert paper_names.short(parallel) != paper_names.short(cyclic)
    # and neither name may collide with the first grid's
    assert paper_names.short(parallel) != paper_names.short(
        "ctune_ewc0p1_kd0p11_T2_mix0p5_eta0p95")


# --------------------------------------------------------------------------- #
# the 1080 lines
# --------------------------------------------------------------------------- #
def test_the_screen_is_every_cell_at_every_fold(parsed):
    assert len(parsed) == 1080
    cells = {c["id"] for c in reg_cells.combo_tune_selected_cells()}
    seen = {(a.parent[len("d01_ctunesel_"):].rsplit("_fold", 1)[0], a.fold)
            for a in parsed}
    assert len(seen) == 1080
    assert {c for c, _ in seen} == cells


def test_every_line_runs_the_screening_protocol_at_the_search_rate(p25, parsed):
    for args in parsed:
        assert args.classes == "digits" and args.resolution == 28
        assert args.clients_per_round == 8 and args.policy == "uniform"
        assert args.rounds == SL.SCREEN_ROUNDS == 25
        assert args.epochs == 5 and args.batch_size == 64
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients
        assert args.old_fold == "all"
        assert args.trainer == "AnchoredTrainer"
    assert p25.CFG.clients_per_round == DIGITS_STUDY01.search_clients_per_round
    assert not any(
        a.clients_per_round == DIGITS_STUDY01.clients_per_round for a in parsed
    )


def test_one_line_is_one_schedule(parsed):
    """
    A --aggregation fedavg line would run the cyclic family under a rule chosen
    for the parallel one. Each cell names a single rule, which lives in one
    family, so a task produces one result - the combination stage's shape.
    """
    cells = {c["id"]: c for c in reg_cells.combo_tune_selected_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_ctunesel_"):].rsplit("_fold", 1)[0]]
        assert args.aggregation == cell["agg"]["rule"]
        assert args.aggregation != "fedavg"
        if cell["family"] == "concurrent":
            assert args.aggregation == "con_delta_anchor_lam"
            assert args.server_eta == 1.0
        else:
            assert args.aggregation == "seq_fedavg_update"


def test_the_half_life_reaches_the_line_as_this_horizons_coefficient(parsed):
    """
    A cell stores h and the line carries lambda_s = 1 - 2 ** (-1/(h * rounds)).

    The point of storing the half-life is that one setting means one
    intervention at both horizons; the point of checking it here is that a cell
    which quietly stored the coefficient instead would emit a legal line that
    is a different rule on the screen than in the finals.
    """
    cells = {c["id"]: c for c in reg_cells.combo_tune_selected_cells()}
    seen = set()
    for args in parsed:
        cell = cells[args.parent[len("d01_ctunesel_"):].rsplit("_fold", 1)[0]]
        if cell["family"] != "concurrent":
            assert args.server_anchor in (None, 0.0), cell["id"]
            continue
        halflife = cell["dials"]["anchor_halflife_r"] * SL.SCREEN_ROUNDS
        assert args.server_anchor == pytest.approx(
            1.0 - 2.0 ** (-1.0 / halflife), rel=1e-12), cell["id"]
        seen.add(cell["dials"]["anchor_halflife_r"])
    assert seen == set(reg_cells.CTUNE_SEL_ANCHOR_HALFLIVES_R)


def test_the_coefficients_reach_the_trainer_unrounded(parsed):
    """A label may round; a coefficient may not - mix is 2/3 on some cells."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.combo_tune_selected_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_ctunesel_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        assert overrides["space"] == "kd+fisher"
        assert overrides["anchor"] == "frozen"
        for name, value in cell["hypers"].items():
            assert float(overrides[name]) == float(value), (cell["id"], name)
        assert overrides["fisher_path"] == (
            "$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher"
        )


def test_every_non_grid_flag_is_the_shipped_combination_lines(p25):
    """
    The claim the README makes, checked against the file it makes it about.

    Everything before ``--set`` must match the s20 line that pairs the same two
    halves, once the things this stage is allowed to move are blanked: the
    parent, the seed, the horizon and the anchor coefficient. ``--server-eta``
    is NOT blanked here - the selected parallel rule runs the plain full step
    and the shipped line says so, so a grid that dropped it would show up.
    """
    def normalise(line: str) -> str:
        head = line.split(" --set ")[0]
        # --fold and --seed move with the fold, and the shipped line is fold 1;
        # --seed is the fold by this study's convention. "--fold " with the
        # space is the fold flag and not --fold-book.
        for flag in ("--parent", "--sampler-seed", "--rounds", "--server-anchor",
                     "--fold", "--seed"):
            head = re.sub(rf"({re.escape(flag)}) \S+", r"\1 X", head)
        return head

    shipped = {}
    for line in (JOBS / "s20_combos4.txt").read_text().splitlines():
        for family, parent in SHIPPED_PAIR.items():
            if f" --parent {parent} " in line:
                shipped[family] = normalise(line)
    assert set(shipped) == set(SL.FAMILIES)
    assert "--server-eta 1.0" in shipped["concurrent"]

    cells = {c["id"]: c for c in reg_cells.combo_tune_selected_cells()}
    seen = set()
    for line in p25.lines():
        cell_id = line.split(" --parent d01_ctunesel_")[1].split(" ")[0]
        cell = cells[cell_id.rsplit("_fold", 1)[0]]
        assert normalise(line) == shipped[cell["family"]], cell["id"]
        seen.add(cell["family"])
    assert seen == set(SL.FAMILIES)


def test_no_line_touches_an_archived_or_foreign_artefact(p25):
    text = "\n".join(p25.lines())
    for dead in RETIRED:
        assert dead not in text, dead
    for line in p25.lines():
        for token in line.split():
            if token.startswith("/") or token.startswith("$FOA_"):
                assert token.startswith("$FOA_STUDY_DIR"), token


# --------------------------------------------------------------------------- #
# the seeds, against everything already on disk
# --------------------------------------------------------------------------- #
def _shipped_seeds(exclude) -> set:
    """Every sampler seed in every shipped task file but the named ones."""
    seeds = set()
    for path in sorted(JOBS.rglob("*.txt")):
        if path.name in exclude:
            continue
        for value in re.findall(r"--sampler-seed (\d+)", path.read_text()):
            seeds.add(int(value))
    return seeds


def test_no_seed_collides_with_any_other_stage(parsed):
    """
    Read off the files, not from a list - P23's reason, with P23's own two
    blocks now among the files to stay clear of.
    """
    prior = _shipped_seeds({SCREEN})
    assert prior, "no shipped task file carried a sampler seed"
    assert not {a.sampler_seed for a in parsed} & prior


def test_the_seed_window_is_wide_enough_and_outside_every_other_block(p25,
                                                                     parsed):
    seeds = {a.sampler_seed for a in parsed}
    assert p25.SEED_OFFSET == 70000
    assert min(seeds) == 770001 and max(seeds) == 772155
    assert max(seeds) - min(seeds) > 1000
    assert min(seeds) > max(_shipped_seeds(set(EXTENSION_FILES)))
    # and above P23's two blocks as well, which is the extra claim this stage
    # makes: its window is opened a ten-thousand past the first extension's.
    assert min(seeds) > max(_shipped_seeds({SCREEN, FULL}))


# --------------------------------------------------------------------------- #
# the shipped files
# --------------------------------------------------------------------------- #
def test_regenerating_reproduces_the_shipped_files_byte_for_byte(p25, tmp_path):
    """
    No flags. A task file that needs an argument remembered by hand is a task
    file nobody can regenerate.
    """
    assert p25.main.__module__ == "make_digits_p25"
    argv = sys.argv[:]
    sys.argv = ["make_digits_p25.py", "--jobs-dir", str(tmp_path)]
    try:
        assert p25.main() == 0
    finally:
        sys.argv = argv
    for name in (SCREEN, "s25_combo_screen_selected_README.md"):
        assert (tmp_path / name).read_text() == (JOBS / name).read_text(), name


def test_the_shipped_screen_holds_1080_runnable_lines_that_parse():
    """The check docs/VERIFY.md runs over the whole jobs directory, here."""
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    lines = [l for l in (JOBS / SCREEN).read_text().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    assert len(lines) == 1080
    for line in lines:
        assert line.startswith("foa ")
        parser.parse_args(shlex.split(
            line.replace("$FOA_STUDY_DIR", "/study").replace("$G0_FOLD", "4")
        )[1:])


def test_the_fisher_fold_is_read_from_the_record_not_remembered(p25):
    record = json.loads((STUDY / "g0_selection.json").read_text())
    assert p25.G0_FOLD == record["selected_fold"]
    assert f"fold {p25.G0_FOLD}" in p25.readme(p25.lines())


def test_the_readme_states_the_pair_the_basis_and_the_edge(p25):
    text = p25.readme(p25.lines())
    assert "lam = m * c_kd + (1 - m) * c_ewc" in text
    assert "mix = m * c_kd / lam" in text
    assert "p12_agg_top3.json" in text
    assert "anchor_h2" in text and "seq_fedavg" in text
    assert "hybrid_seq_mix0p75" in text
    assert "anchor_h2_hybrid_seq_mix0p5" in text
    assert "seq_fedavg_hybrid_mix0p5" in text
    # the h range note the stage exists to be honest about
    assert "top edge" in text
    assert "25 rounds, not 100" in text
    assert "extension" in text.lower()
    for dead in RETIRED:
        assert dead not in text, dead


# --------------------------------------------------------------------------- #
# P26: the finals the screen feeds
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _shipped_baselines(tmp_path):
    """
    Every synthetic study root carries g-0's own two accuracies.

    The score is what a run ADDED on the cohort less the source knowledge it
    SPENT, so a root without them is not a study root and the emitter refuses it
    rather than guess.
    """
    for name, accuracy in (("g0_perfold_evaluations.json", 0.8225),
                           ("g0_evaluations.json", 0.9986)):
        (tmp_path / name).write_text(json.dumps({
            str(fold): {"fold": fold, "part": "test", "accuracy": accuracy}
            for fold in SL.FOLDS
        }))


def _screen_result(root: Path, cell_id: str, fold: int, family: str,
                   adaptation: float, preservation: float = 0.99):
    """One (cell, fold) result where the driver writes it."""
    run = (root / f"d01_ctunesel_{cell_id}_fold{fold}_x_grid_search"
           / "fold1_seed_1" / f"{family}_delta" / f"base_agg_x_{family}")
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
    """The whole extension screen on disk, at scores that separate the cells."""
    for index, cell in enumerate(reg_cells.combo_tune_selected_cells()):
        family = cell["family"]
        for fold in (1, 2):
            _screen_result(root, cell["id"], fold, family,
                           (scores or {}).get(cell["id"], 0.5 + index * 1e-4))


def test_the_finals_are_one_winner_per_schedule(emit, tmp_path):
    _screened(tmp_path)
    out = tmp_path / "p26.txt"
    expect = SL.counts(DIGITS_STUDY01)["combo_tune_selected_full"]
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path, out,
                                         expect) == 0

    lines = [l for l in out.read_text().splitlines()
             if l and not l.startswith("#")]
    assert len(lines) == expect == 10
    record = json.loads(
        (tmp_path / "tables" / "p25_combo_tune_selected_winners.json").read_text()
    )
    assert record["extension"]["is_core_programme"] is False
    winners = {k: v for k, v in record.items() if k != "extension"}
    assert set(winners) == {f"combo-tune-selected/{f}" for f in SL.FAMILIES}
    assert all(entry["measured"] for entry in winners.values())
    assert winners["combo-tune-selected/concurrent"]["considered"] == 162
    assert winners["combo-tune-selected/sequential"]["considered"] == 54


def test_the_two_extensions_do_not_read_each_others_runs(emit, tmp_path):
    """
    The failure that has no error message.

    ``d01_ctunesel_*`` folders are on disk and P24's selector scans
    ``d01_ctune_``. If that prefix matched them, P24 would rank 432 cells of two
    different pairs against each other and crown one - and P24 has already run,
    so its record and its finals would silently stop describing what produced
    them.
    """
    _screened(tmp_path)
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 10) == 1
    assert not (tmp_path / "p24.txt").is_file()
    assert not (tmp_path / "tables" / "p23_combo_tune_winners.json").is_file()


def test_each_extension_has_exactly_one_selection_site(emit):
    """
    Two extensions, two records, two modes, and no third writer of either.

    ``table_io`` reads this module's own syntax tree, so the count is of the
    generators that actually call ``write_table`` with that name rather than of
    the ones a list here remembers. A second writer of one record would be two
    selections claiming one artefact, and the later one would win by running
    last.
    """
    records = {"p23_combo_tune_winners.json": "combo-tune-full",
               "p25_combo_tune_selected_winners.json":
                   "combo-tune-full-selected"}
    for name, owner in records.items():
        writers = [what for what in sorted(emit.WHAT)
                   if name in emit.table_io(what)["writes"]]
        assert writers == [owner], (name, writers)
        readers = [what for what in sorted(emit.WHAT)
                   if name in emit.table_io(what)["reads"]]
        assert readers == [], (name, readers)
    # and the two extension modes write nothing else and nothing in common
    written = {owner: set(emit.table_io(owner)["writes"])
               for owner in records.values()}
    assert not written["combo-tune-full"] & written["combo-tune-full-selected"]
    assert all(len(v) == 1 for v in written.values())


def test_a_schedule_is_ranked_only_against_its_own_cells(emit, tmp_path):
    """
    The cyclic half is 54 cells and the parallel one 162, and they are different
    arms - a cyclic cell has no half-life to move. Ranking them together would
    let one schedule's grid crown the other's winner.
    """
    best_cyclic = reg_cells.combo_tune_selected_by_family()["sequential"][7]["id"]
    _screened(tmp_path, {best_cyclic: 0.99})
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path,
                                         tmp_path / "p26.txt", 10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p25_combo_tune_selected_winners.json").read_text()
    )
    assert record["combo-tune-selected/sequential"]["winner"] == best_cyclic
    assert record["combo-tune-selected/concurrent"]["winner"] != best_cyclic
    assert "_h" in record["combo-tune-selected/concurrent"]["winner"]


def test_the_selection_is_the_score_every_other_final_selects_on(emit, tmp_path):
    """
    Not adaptation, and not preservation: the trade, at w = 1.

    An extension exists to be read against the programme, so a winner chosen by
    a different rule could not be. The greedy cell here gives away more
    preservation than it buys, so it wins on adaptation and must lose the rule.
    """
    by_family = reg_cells.combo_tune_selected_by_family()
    _screened(tmp_path)
    pairs = {}
    for family in SL.FAMILIES:
        greedy, traded = by_family[family][0]["id"], by_family[family][1]["id"]
        pairs[family] = traded
        for fold in (1, 2):
            _screen_result(tmp_path, greedy, fold, family, 0.90,
                           preservation=0.95)
            _screen_result(tmp_path, traded, fold, family, 0.89,
                           preservation=0.99)
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path,
                                         tmp_path / "p26.txt", 10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p25_combo_tune_selected_winners.json").read_text()
    )
    for family, traded in pairs.items():
        assert record[f"combo-tune-selected/{family}"]["winner"] == traded


def test_a_selection_over_nothing_is_refused(emit, tmp_path):
    """A screen that never ran would make both winners a row's first cell."""
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path,
                                         tmp_path / "p26.txt", 10) == 1
    assert not (tmp_path / "p26.txt").is_file()


def test_a_miscount_halts(emit, tmp_path):
    _screened(tmp_path)
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path,
                                         tmp_path / "p26.txt", 999) == 1


def test_the_finals_are_the_full_horizon_at_the_search_rate(emit, tmp_path):
    """The winners must be re-run under the conditions they were chosen under."""
    from federated_outlier_adaptation.cli import build_parser

    _screened(tmp_path)
    out = tmp_path / "p26.txt"
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path, out, 10) == 0
    parser = build_parser()
    seeds, parents = set(), []
    for line in out.read_text().splitlines():
        if not line.startswith("foa "):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == SL.FULL_ROUNDS == 100
        assert args.clients_per_round == DIGITS_STUDY01.search_clients_per_round
        assert args.aggregation in ("con_delta_anchor_lam", "seq_fedavg_update")
        assert args.trainer == "AnchoredTrainer" and args.old_fold == "all"
        seeds.add(args.sampler_seed)
        parents.append(args.parent)
    assert len(seeds) == 10
    assert len(set(parents)) == 10
    for family in SL.FAMILIES:
        assert sum(f"d01_ctuneselfull_{family}_" in p for p in parents) == 5


def test_the_half_life_is_resolved_at_the_horizon_the_finals_run(emit, tmp_path):
    """
    The reason a cell stores h and not the coefficient, checked where it pays.

    The screen emits lambda_s for 25 rounds and the finals for 100, from the one
    stored half-life - so the crowned cell is the same intervention in both. A
    cell that carried the coefficient would be four times as fast an anchor in
    the finals as in the screen that chose it.
    """
    from federated_outlier_adaptation.cli import build_parser

    _screened(tmp_path)
    out = tmp_path / "p26.txt"
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path, out, 10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p25_combo_tune_selected_winners.json").read_text()
    )
    halflife = record["combo-tune-selected/concurrent"]["dials"][
        "anchor_halflife_r"]
    parser = build_parser()
    for line in out.read_text().splitlines():
        if not line.startswith("foa ") or "con_delta_anchor_lam" not in line:
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.server_anchor == pytest.approx(
            1.0 - 2.0 ** (-1.0 / (halflife * SL.FULL_ROUNDS)), rel=1e-12)


def test_the_finals_draw_a_block_of_their_own(emit):
    assert emit.COMBO_TUNE_SEL_BLOCK == 74000
    assert emit.COMBO_TUNE_SEL_BLOCK != emit.COMBO_TUNE_BLOCK
    mine = {int(v) for v in re.findall(
        r"--sampler-seed (\d+)", (JOBS / FULL).read_text())}
    assert mine == set(range(774001, 774006)) | set(range(774011, 774016))


def test_the_boundary_report_is_on_the_dials_not_the_coefficients(emit,
                                                                  tmp_path):
    """
    lam and mix are functions of all four dials, so a boundary report on them
    would answer a question nobody asked and miss the four that were.
    """
    _screened(tmp_path)
    assert emit.combo_tune_full_selected(DIGITS_STUDY01, tmp_path,
                                         tmp_path / "p26.txt", 10) == 0
    hits = (tmp_path / "tables" / "BOUNDARY_HITS.txt").read_text()
    reported = [l for l in hits.splitlines()
                if "combo-tune-full-selected" in l]
    for line in reported:
        assert re.search(
            r": (c_ewc|c_kd|T|m|anchor_halflife_r)=", line), line
        assert " lam=" not in line and " mix=" not in line


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
    record = json.loads(
        (TABLES / "p25_combo_tune_selected_winners.json").read_text())
    assert record["extension"]["is_core_programme"] is False
    text = (JOBS / FULL).read_text()
    cells = {c["id"]: c for c in reg_cells.combo_tune_selected_cells()}
    for key, entry in record.items():
        if key == "extension":
            continue
        family = key.split("/")[1]
        assert entry["winner"] in cells
        assert cells[entry["winner"]]["family"] == family
        assert f"d01_ctuneselfull_{family}_{entry['winner']}_fold1" in text
        assert f"#   {family:<11} {entry['winner']}" in text


def test_the_shipped_finals_are_neither_the_cross_nor_the_first_extension():
    """
    Three stages, three sets of folders. A selector that read one for another
    would report an arm under a selection that did not produce it.
    """
    text = (JOBS / FULL).read_text()
    assert "d01_combo_" not in text
    assert "d01_ctunefull_" not in text
    assert "d01_ctuneselfull_concurrent_ctunesel_" in text
    assert "d01_ctuneselfull_sequential_ctunesel_" in text


def test_the_two_records_are_two_selections_and_not_one_made_twice():
    """
    Both ship, neither supersedes the other, and their keys cannot collide.

    P24's record is keyed ``combo-tune/<schedule>`` and P26's
    ``combo-tune-selected/<schedule>``; a reader that merged them would have to
    decide which arm belonged to which grid, and nothing in a shared key would
    let it.
    """
    first = json.loads((TABLES / "p23_combo_tune_winners.json").read_text())
    second = json.loads(
        (TABLES / "p25_combo_tune_selected_winners.json").read_text())
    assert not (set(first) - {"extension"}) & (set(second) - {"extension"})
    assert first["extension"]["screen"] == "s23_combo_screen.txt"
    assert second["extension"]["screen"] == "s25_combo_screen_selected.txt"
    for key, entry in second.items():
        if key != "extension":
            assert entry["winner"].startswith("ctunesel_")
    for key, entry in first.items():
        if key != "extension":
            assert entry["winner"].startswith("ctune_")


def test_the_finals_readme_states_the_rule_and_what_it_chose():
    text = (JOBS / "s26_combo_full_selected_README.md").read_text()
    assert "(adaptation - A_0) - (P_0 - preservation)" in text
    assert "p25_combo_tune_selected_winners.json" in text
    assert "p12_agg_top3.json" in text
    assert "extension" in text.lower()
    record = json.loads(
        (TABLES / "p25_combo_tune_selected_winners.json").read_text())
    for key, entry in record.items():
        if key != "extension":
            assert entry["winner"] in text
    for dead in RETIRED:
        assert dead not in text, dead
