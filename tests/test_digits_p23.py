"""
P23/P24: the joint tuning of each schedule's leading pair - **an extension**.

The study is complete without these two stages, and that is the first thing
worth pinning: an extension that leaked into a core catalogue would move seeds
under files that have already run, and would put cells the paper never screened
into shortlists the paper reports from. Nothing here may reach ``screen_cells``,
``reg_top3`` or the combination cross.

The second is the mapping. This grid names two coefficients and a blend weight,
while ``AnchoredTrainer`` takes one ``lam`` and one ``mix``, so the dials and the
trainer's arguments are different objects and the conversion between them is the
whole of the stage's correctness. A grid that wrote ``m`` straight into ``mix``
would sweep something else entirely and every emitted line would still be legal,
still parse and still run - which is exactly the class of failure that has to be
caught here rather than in a table six weeks later.
"""

from __future__ import annotations

import json
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

SCREEN = "s23_combo_screen.txt"
FULL = "s24_combo_full.txt"

#: The retired study roots, as P13 and P21 list them.
RETIRED = (
    "v1_superseded", "main_v6", "studies/T", "T1_20outliers", "T1_10outliers",
    "T2_20outliers", "T2_10outliers", "T3_20normals", "T3_10normals",
    "all_writers_digits",
)

#: The shipped combination lines this stage copies every non-grid flag from.
SHIPPED_PAIR = {
    "concurrent": "d01_combo_eta_0p95_hybrid_seq_mix0p5_fold1",
    "sequential": "d01_combo_seq_delta_capped_hybrid_mix0p5_fold1",
}


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p23(tools_path):
    import make_digits_p23

    return make_digits_p23


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture()
def parsed(p23):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    return [parser.parse_args(shlex.split(line)[1:]) for line in p23.lines()]


# --------------------------------------------------------------------------- #
# it is an extension, and the core cannot see it
# --------------------------------------------------------------------------- #
def test_the_extension_is_not_in_any_catalogue_the_core_reads():
    """
    The failure this prevents is not subtle and it is not recoverable.

    ``s16_reg_screen3.txt`` and ``s21_blend_screen.txt`` both seed as
    ``base + block + index * 10 + fold``, so a cell inserted into the catalogue
    either was emitted from moves every seed after it and stops that file
    regenerating - 700 and 1,170 finished runs no longer reproducible from the
    generator that claims to produce them.
    """
    tuned = {cell["id"] for cell in reg_cells.combo_tune_cells()}
    assert not (tuned & {c["id"] for c in reg_cells.screen_cells()})
    assert not (tuned & {c["id"] for c in reg_cells.blend_cells()})
    assert len(reg_cells.screen_cells()) == 140
    assert len(reg_cells.blend_cells()) == 234


def test_no_core_generator_reads_the_extensions_record(emit):
    """
    One direction only: the extension reads the programme's records, and the
    programme reads none of the extension's. Read off the module's own syntax
    tree, which is how ``table_io`` already answers this for every other mode.
    """
    written = "p23_combo_tune_winners.json"
    assert written in emit.table_io("combo-tune-full")["writes"]
    for what in sorted(emit.WHAT):
        if what == "combo-tune-full":
            continue
        assert written not in emit.table_io(what)["reads"], what
        assert written not in emit.table_io(what)["writes"], what


def test_the_counts_register_the_extension_and_say_that_it_is_one():
    counts = SL.counts(DIGITS_STUDY01)
    assert counts["combo_tune_screen"] == 216 * 5 == 1080
    assert counts["combo_tune_full"] == len(SL.FAMILIES) * 5 == 10
    assert SL.EXTENSION_COUNTS == ("combo_tune_screen", "combo_tune_full")
    for name in SL.EXTENSION_COUNTS:
        assert name in counts


def test_every_shipped_file_says_it_is_an_extension():
    """A stage that is not part of the study has to say so where it is read."""
    for name in (SCREEN, FULL, "s23_combo_screen_README.md",
                 "s24_combo_full_README.md"):
        text = (JOBS / name).read_text()
        assert "EXTENSION" in text or "extension" in text, name
    for stem in ("digits_s23_combo_screen", "digits_s24_combo_full"):
        chain = REPO / "study" / "jobs" / f"{stem}.chain.toml"
        assert "extension" in chain.read_text().lower(), stem


# --------------------------------------------------------------------------- #
# the grid, and the mapping onto the trainer's coefficients
# --------------------------------------------------------------------------- #
def test_the_grid_is_the_owners_and_has_not_been_altered():
    assert reg_cells.CTUNE_EWC_COEFFS == (0.05, 0.1, 0.3)
    assert reg_cells.CTUNE_KD_COEFFS == (0.05, 0.11, 0.2)
    assert reg_cells.CTUNE_TEMPERATURES == (0.25, 2.0)
    assert reg_cells.CTUNE_MIXES == (0.25, 0.5, 0.75)
    assert reg_cells.CTUNE_SERVER_ETAS == (0.9, 0.95, 1.0)
    assert reg_cells.CTUNE_MIXES is reg_cells.HYBRID_MIXES


def test_the_grid_is_one_hundred_and_sixty_two_plus_fifty_four(p23):
    cells = reg_cells.combo_tune_cells()
    by_family = reg_cells.combo_tune_by_family()
    assert len(by_family["concurrent"]) == 3 * 3 * 2 * 3 * 3 == 162
    assert len(by_family["sequential"]) == 3 * 3 * 2 * 3 == 54
    assert len(cells) == 216 == p23.SCREENED_CELLS
    assert len({c["id"] for c in cells}) == 216
    assert len(p23.lines()) == 216 * len(SL.FOLDS) == 1080


def test_the_mapping_puts_both_named_coefficients_on_their_own_half():
    """
    The whole correctness of the stage, asserted cell by cell.

    The trainer computes lam * (mix * KD + (1 - mix) * Fisher), so the KD term's
    coefficient is lam * mix and the Fisher term's is lam * (1 - mix). Those must
    be m * c_kd and (1 - m) * c_ewc, or the grid is sweeping something other than
    what it names - and a line carrying the wrong coefficient is still legal,
    still parses and still runs.
    """
    for cell in reg_cells.combo_tune_cells():
        dials, hypers = cell["dials"], cell["hypers"]
        lam, mix, m = hypers["lam"], hypers["mix"], dials["m"]
        assert lam * mix == pytest.approx(m * dials["c_kd"], rel=1e-12)
        assert lam * (1.0 - mix) == pytest.approx(
            (1.0 - m) * dials["c_ewc"], rel=1e-12)
        assert hypers["T"] == dials["T"]


def test_the_trainers_mix_is_not_the_owners_m_except_where_it_must_be():
    """
    The confusion the helper exists to prevent, pinned as a fact about the grid.

    The two coincide exactly when the two coefficients are equal, which on this
    grid is c_ewc = c_kd = 0.05 and nothing else. If they coincided everywhere,
    the conversion would be doing nothing and nobody would notice.
    """
    same, different = 0, 0
    for cell in reg_cells.combo_tune_cells():
        equal = cell["hypers"]["mix"] == pytest.approx(cell["dials"]["m"])
        assert equal == (cell["dials"]["c_ewc"] == cell["dials"]["c_kd"])
        same += equal
        different += not equal
    assert same and different


def test_the_helper_refuses_a_weight_that_is_not_a_blend():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            reg_cells.combo_tune_lam_mix(0.1, 0.11, bad)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            reg_cells.combo_tune_lam_mix(bad, 0.11, 0.5)
        with pytest.raises(ValueError):
            reg_cells.combo_tune_lam_mix(0.1, bad, 0.5)


def test_no_two_cells_hand_the_trainer_the_same_experiment(p23):
    """
    Deduplication is on the COEFFICIENTS, because the dials are what differ.

    Two dial settings that resolve to one (lam, mix, T) under one rule would be
    two folders, two seeds and one experiment, with nothing in the records to
    say so. This grid has none; the check is here so a widened row cannot
    quietly pay twice.
    """
    seen = set()
    for cell in reg_cells.combo_tune_cells():
        key = (cell["family"], cell["agg"]["flags"].get("server_eta"),
               round(cell["hypers"]["lam"], 12),
               round(cell["hypers"]["mix"], 12), cell["hypers"]["T"])
        assert key not in seen, cell["id"]
        seen.add(key)
    assert reg_cells.combo_tune_duplicates() == 0 == p23.DUPLICATES


def test_the_cell_id_names_the_dials_and_the_schedule():
    """
    Not the trainer's coefficients: lam and mix are functions of all three
    penalty dials, so an id built from them could not be read against the grid.
    The schedule rides on the eta suffix, which only the parallel rule can have.
    """
    for cell in reg_cells.combo_tune_cells():
        dials = cell["dials"]
        expected = (f"ctune_ewc{reg_cells._fmt(dials['c_ewc'])}"
                    f"_kd{reg_cells._fmt(dials['c_kd'])}"
                    f"_T{reg_cells._fmt(dials['T'])}"
                    f"_mix{reg_cells._fmt(dials['m'])}")
        if cell["family"] == "concurrent":
            expected += f"_eta{reg_cells._fmt(dials['server_eta'])}"
        assert cell["id"] == expected
        assert ("_eta" in cell["id"]) == (cell["family"] == "concurrent")
    assert "ctune_ewc0p1_kd0p11_T2_mix0p5" in {
        c["id"] for c in reg_cells.combo_tune_cells()
    }


def test_the_paper_can_name_every_cell():
    """A code id must never reach a page; a gap in the map is a build failure."""
    sys.path.insert(0, str(REPO / "tools" / "paper_figures"))
    import paper_names

    for cell in reg_cells.combo_tune_cells():
        assert paper_names.label(cell["id"])
        assert paper_names.short(cell["id"])
        assert paper_names.params(cell["id"])
    parallel = "ctune_ewc0p1_kd0p11_T2_mix0p5_eta0p95"
    cyclic = "ctune_ewc0p1_kd0p11_T2_mix0p5"
    assert r"\eta_s{=}0.95" in paper_names.params(parallel)
    assert r"\eta_s" not in paper_names.params(cyclic)
    assert paper_names.short(parallel) != paper_names.short(cyclic)


# --------------------------------------------------------------------------- #
# the 1080 lines
# --------------------------------------------------------------------------- #
def test_the_screen_is_every_cell_at_every_fold(parsed):
    assert len(parsed) == 1080
    cells = {c["id"] for c in reg_cells.combo_tune_cells()}
    seen = {(a.parent[len("d01_ctune_"):].rsplit("_fold", 1)[0], a.fold)
            for a in parsed}
    assert len(seen) == 1080
    assert {c for c, _ in seen} == cells


def test_every_line_runs_the_screening_protocol_at_the_search_rate(p23, parsed):
    for args in parsed:
        assert args.classes == "digits" and args.resolution == 28
        assert args.clients_per_round == 8 and args.policy == "uniform"
        assert args.rounds == SL.SCREEN_ROUNDS == 25
        assert args.epochs == 5 and args.batch_size == 64
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients
        assert args.old_fold == "all"
        assert args.trainer == "AnchoredTrainer"
    assert p23.CFG.clients_per_round == DIGITS_STUDY01.search_clients_per_round
    assert not any(
        a.clients_per_round == DIGITS_STUDY01.clients_per_round for a in parsed
    )


def test_one_line_is_one_schedule(parsed):
    """
    A --aggregation fedavg line would run the cyclic family under a rule chosen
    for the parallel one. Each cell names a single rule, which lives in one
    family, so a task produces one result - the combination stage's shape.
    """
    cells = {c["id"]: c for c in reg_cells.combo_tune_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_ctune_"):].rsplit("_fold", 1)[0]]
        assert args.aggregation == cell["agg"]["rule"]
        assert args.aggregation != "fedavg"
        if cell["family"] == "concurrent":
            assert args.aggregation == "con_delta_eta"
            assert args.server_eta == cell["dials"]["server_eta"]
        else:
            assert args.aggregation == "seq_delta_capped"


def test_the_coefficients_reach_the_trainer_unrounded(parsed):
    """A label may round; a coefficient may not - mix is 2/3 on some cells."""
    from federated_outlier_adaptation.cli import _trainer_overrides

    cells = {c["id"]: c for c in reg_cells.combo_tune_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_ctune_"):].rsplit("_fold", 1)[0]]
        overrides = _trainer_overrides(args)
        assert overrides["space"] == "kd+fisher"
        assert overrides["anchor"] == "frozen"
        for name, value in cell["hypers"].items():
            assert float(overrides[name]) == float(value), (cell["id"], name)
        assert overrides["fisher_path"] == (
            "$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher"
        )


def test_every_non_grid_flag_is_the_shipped_combination_lines(p23):
    """
    The claim the README makes, checked against the file it makes it about.

    Everything before ``--set`` must match the s20 line that pairs the same two
    halves, once the four things this stage is allowed to move are blanked: the
    parent, the seed, the horizon and the server step. If anything else drifted -
    a worker count, a book, the rate - the arm would not be the shipped pair with
    its coefficients moved, and the comparison the stage exists for would be
    against something else.
    """
    def normalise(line: str) -> str:
        head = line.split(" --set ")[0]
        # --fold and --seed move with the fold, and the shipped line is fold 1;
        # --seed is the fold by this study's convention. "--fold " with the
        # space is the fold flag and not --fold-book.
        for flag in ("--parent", "--sampler-seed", "--rounds", "--server-eta",
                     "--fold", "--seed"):
            head = re.sub(rf"({re.escape(flag)}) \S+", r"\1 X", head)
        return head

    shipped = {}
    for line in (JOBS / "s20_combos4.txt").read_text().splitlines():
        for family, parent in SHIPPED_PAIR.items():
            if f" --parent {parent} " in line:
                shipped[family] = normalise(line)
    assert set(shipped) == set(SL.FAMILIES)

    cells = {c["id"]: c for c in reg_cells.combo_tune_cells()}
    seen = set()
    for line in p23.lines():
        cell_id = line.split(" --parent d01_ctune_")[1].split(" ")[0]
        cell = cells[cell_id.rsplit("_fold", 1)[0]]
        assert normalise(line) == shipped[cell["family"]], cell["id"]
        seen.add(cell["family"])
    assert seen == set(SL.FAMILIES)


def test_no_line_touches_an_archived_or_foreign_artefact(p23):
    text = "\n".join(p23.lines())
    for dead in RETIRED:
        assert dead not in text, dead
    for line in p23.lines():
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
    Read off the files, not from a list. A hand-kept range is what let the
    combination stage draw seeds the regularisation screen had already used: two
    stages sharing a client-sampling sequence, correlated, with nothing saying
    so - and an extension correlated with the programme it is read against would
    be the same failure with a worse consequence.
    """
    prior = _shipped_seeds({SCREEN})
    assert prior, "no shipped task file carried a sampler seed"
    assert not {a.sampler_seed for a in parsed} & prior


def test_the_seed_window_is_wide_enough_and_outside_the_programmes(p23, parsed):
    seeds = {a.sampler_seed for a in parsed}
    assert p23.SEED_OFFSET == 60000
    assert min(seeds) == 760001 and max(seeds) == 762155
    assert max(seeds) - min(seeds) > 1000
    assert min(seeds) > max(_shipped_seeds({SCREEN, FULL}))


def test_the_screen_and_the_finals_do_not_share_a_seed():
    screen = {int(v) for v in re.findall(
        r"--sampler-seed (\d+)", (JOBS / SCREEN).read_text())}
    finals = {int(v) for v in re.findall(
        r"--sampler-seed (\d+)", (JOBS / FULL).read_text())}
    assert len(screen) == 1080 and len(finals) == 10
    assert not screen & finals
    assert not finals & _shipped_seeds({FULL})


# --------------------------------------------------------------------------- #
# the shipped files
# --------------------------------------------------------------------------- #
def test_regenerating_reproduces_the_shipped_files_byte_for_byte(p23, tmp_path):
    """
    No flags. A task file that needs an argument remembered by hand is a task
    file nobody can regenerate.
    """
    assert p23.main.__module__ == "make_digits_p23"
    argv = sys.argv[:]
    sys.argv = ["make_digits_p23.py", "--jobs-dir", str(tmp_path)]
    try:
        assert p23.main() == 0
    finally:
        sys.argv = argv
    for name in (SCREEN, "s23_combo_screen_README.md"):
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


def test_the_fisher_fold_is_read_from_the_record_not_remembered(p23):
    record = json.loads(
        (REPO / "study" / "artifacts" / "Digits_study01" / "g0_selection.json")
        .read_text()
    )
    assert p23.G0_FOLD == record["selected_fold"]
    assert f"fold {p23.G0_FOLD}" in p23.readme(p23.lines())


def test_the_readme_states_the_mapping_the_pair_and_the_horizon(p23):
    text = p23.readme(p23.lines())
    assert "lam = m * c_kd + (1 - m) * c_ewc" in text
    assert "mix = m * c_kd / lam" in text
    assert "eta_0p95" in text and "seq_delta_capped" in text
    assert "hybrid_seq_mix0p75" in text
    assert "eta_0p95_hybrid_seq_mix0p5" in text
    assert "seq_delta_capped_hybrid_mix0p5" in text
    assert "25 rounds, not 100" in text
    assert "extension" in text.lower()
    for dead in RETIRED:
        assert dead not in text, dead
# --------------------------------------------------------------------------- #
# P24: the finals the screen feeds
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
    run = (root / f"d01_ctune_{cell_id}_fold{fold}_x_grid_search"
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
    for index, cell in enumerate(reg_cells.combo_tune_cells()):
        family = cell["family"]
        for fold in (1, 2):
            _screen_result(root, cell["id"], fold, family,
                           (scores or {}).get(cell["id"], 0.5 + index * 1e-4))


def test_the_finals_are_one_winner_per_schedule(emit, tmp_path):
    _screened(tmp_path)
    out = tmp_path / "p24.txt"
    expect = SL.counts(DIGITS_STUDY01)["combo_tune_full"]
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path, out, expect) == 0

    lines = [l for l in out.read_text().splitlines()
             if l and not l.startswith("#")]
    assert len(lines) == expect == 10
    record = json.loads(
        (tmp_path / "tables" / "p23_combo_tune_winners.json").read_text()
    )
    assert record["extension"]["is_core_programme"] is False
    winners = {k: v for k, v in record.items() if k != "extension"}
    assert set(winners) == {f"combo-tune/{f}" for f in SL.FAMILIES}
    assert all(entry["measured"] for entry in winners.values())
    assert winners["combo-tune/concurrent"]["considered"] == 162
    assert winners["combo-tune/sequential"]["considered"] == 54


def test_a_schedule_is_ranked_only_against_its_own_cells(emit, tmp_path):
    """
    The cyclic half is 54 cells and the parallel one 162, and they are different
    arms - a cyclic cell has no server step to move. Ranking them together would
    let one schedule's grid crown the other's winner.
    """
    best_cyclic = reg_cells.combo_tune_by_family()["sequential"][7]["id"]
    _screened(tmp_path, {best_cyclic: 0.99})
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p23_combo_tune_winners.json").read_text()
    )
    assert record["combo-tune/sequential"]["winner"] == best_cyclic
    assert record["combo-tune/concurrent"]["winner"] != best_cyclic
    assert "_eta" in record["combo-tune/concurrent"]["winner"]


def test_the_selection_is_the_score_every_other_final_selects_on(emit, tmp_path):
    """
    Not adaptation, and not preservation: the trade, at w = 1.

    An extension exists to be read against the programme, so a winner chosen by
    a different rule could not be. The greedy cell here gives away more
    preservation than it buys, so it wins on adaptation and must lose the rule.
    """
    by_family = reg_cells.combo_tune_by_family()
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
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 10) == 0
    record = json.loads(
        (tmp_path / "tables" / "p23_combo_tune_winners.json").read_text()
    )
    for family, traded in pairs.items():
        assert record[f"combo-tune/{family}"]["winner"] == traded


def test_a_selection_over_nothing_is_refused(emit, tmp_path):
    """A screen that never ran would make both winners a row's first cell."""
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 10) == 1
    assert not (tmp_path / "p24.txt").is_file()


def test_a_miscount_halts(emit, tmp_path):
    _screened(tmp_path)
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 999) == 1


def test_the_finals_are_the_full_horizon_at_the_search_rate(emit, tmp_path):
    """The winners must be re-run under the conditions they were chosen under."""
    from federated_outlier_adaptation.cli import build_parser

    _screened(tmp_path)
    out = tmp_path / "p24.txt"
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path, out, 10) == 0
    parser = build_parser()
    seeds, parents = set(), []
    for line in out.read_text().splitlines():
        if not line.startswith("foa "):
            continue
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == SL.FULL_ROUNDS == 100
        assert args.clients_per_round == DIGITS_STUDY01.search_clients_per_round
        assert args.aggregation in ("con_delta_eta", "seq_delta_capped")
        assert args.trainer == "AnchoredTrainer" and args.old_fold == "all"
        seeds.add(args.sampler_seed)
        parents.append(args.parent)
    assert len(seeds) == 10
    assert len(set(parents)) == 10
    for family in SL.FAMILIES:
        assert sum(f"d01_ctunefull_{family}_" in p for p in parents) == 5


def test_the_finals_draw_a_block_of_their_own(emit):
    assert emit.COMBO_TUNE_BLOCK == 64000
    mine = {int(v) for v in re.findall(
        r"--sampler-seed (\d+)", (JOBS / FULL).read_text())}
    assert mine == set(range(764001, 764006)) | set(range(764011, 764016))


def test_the_boundary_report_is_on_the_dials_not_the_coefficients(emit,
                                                                  tmp_path):
    """
    lam and mix are functions of all four dials, so a boundary report on them
    would answer a question nobody asked and miss the four that were.
    """
    _screened(tmp_path)
    assert emit.combo_tune_full(DIGITS_STUDY01, tmp_path,
                                tmp_path / "p24.txt", 10) == 0
    hits = (tmp_path / "tables" / "BOUNDARY_HITS.txt").read_text()
    reported = [l for l in hits.splitlines() if "combo-tune-full" in l]
    for line in reported:
        assert re.search(r": (c_ewc|c_kd|T|m|server_eta)=", line), line
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
        (REPO / "study" / "artifacts" / "Digits_study01" / "tables"
         / "p23_combo_tune_winners.json").read_text()
    )
    assert record["extension"]["is_core_programme"] is False
    text = (JOBS / FULL).read_text()
    cells = {c["id"]: c for c in reg_cells.combo_tune_cells()}
    for key, entry in record.items():
        if key == "extension":
            continue
        family = key.split("/")[1]
        assert entry["winner"] in cells
        assert cells[entry["winner"]]["family"] == family
        assert f"d01_ctunefull_{family}_{entry['winner']}_fold1" in text
        assert f"#   {family:<11} {entry['winner']}" in text


def test_the_shipped_finals_are_not_the_combinations_already_on_disk():
    """
    s20 crossed two shortlists at one setting each. These are new arms, and
    their folders have to say so or a selector would read one for the other.
    """
    text = (JOBS / FULL).read_text()
    assert "d01_combo_" not in text
    assert "d01_ctunefull_concurrent_ctune_" in text
    assert "d01_ctunefull_sequential_ctune_" in text


def test_the_finals_readme_states_the_rule_and_what_it_chose():
    text = (JOBS / "s24_combo_full_README.md").read_text()
    assert "(adaptation - A_0) - (P_0 - preservation)" in text
    assert "p23_combo_tune_winners.json" in text
    assert "extension" in text.lower()
    record = json.loads(
        (REPO / "study" / "artifacts" / "Digits_study01" / "tables"
         / "p23_combo_tune_winners.json").read_text()
    )
    for key, entry in record.items():
        if key != "extension":
            assert entry["winner"] in text
    for dead in RETIRED:
        assert dead not in text, dead
