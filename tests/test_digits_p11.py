"""
P11: the aggregation screen, and the selection that follows it.

The screen is 845 runs whose only purpose is to rank, so the tests here are
about the two ways a rank can be wrong without looking wrong: a line that does
not run the protocol it claims to, and a selection whose output does not match
the array written to consume it.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import agg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import (
    DIGITS_STUDY01,
    STUDIES,
    config,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p11(tools_path):
    import make_digits_p11

    return make_digits_p11


@pytest.fixture()
def emit(tools_path):
    import study_emit

    return study_emit


@pytest.fixture()
def parsed(p11):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    # the screen FILE, not the whole table: the boundary extension was appended
    # to the table afterwards and is submitted as its own file.
    return [
        parser.parse_args(shlex.split(line)[1:]) for line in p11.screened_lines()
    ]


# --------------------------------------------------------------------------- #
# the revived configuration, and nothing else
# --------------------------------------------------------------------------- #
def test_no_study_of_the_deleted_programme_survived_the_reset():
    """
    The multi-study grid named folders that were deleted.

    Leaving those configurations in place would have kept five studies' worth of
    dead addresses one import away from a live pipeline. The registry has since
    gained the Shakespeare transfer study, which is a *new* design point with a
    root of its own - so the invariant is not "one study" but "no study that
    addresses a deleted root".
    """
    assert set(STUDIES) == {"Digits_study01", "Shakespeare_study01"}
    for dead in ("T1_20outliers100oldselection20", "T2_10outliers",
                 "T3_20normals", "main_v6"):
        with pytest.raises(KeyError, match="Unknown study"):
            config(dead)


def test_the_two_studies_cannot_collide_in_one_root():
    """
    Different tags and different seed bases, so a folder or a sampler seed can
    never be claimed by both.
    """
    tags = {cfg.tag for cfg in STUDIES.values()}
    bases = {cfg.seed_base for cfg in STUDIES.values()}
    assert len(tags) == len(bases) == len(STUDIES)
    # Far enough apart that no stage's offset can reach the next study's block.
    assert min(abs(a - b) for a in bases for b in bases if a != b) >= 50_000


def test_the_study_is_the_digit_design_point():
    cfg = DIGITS_STUDY01
    assert cfg.classes == "digits"
    assert (cfg.cohort_size, cfg.old_size, cfg.clients_per_round) == (10, 200, 9)
    assert cfg.dropout == pytest.approx(0.1)
    assert cfg.cohort_file_name == "cohort_worst10.json"
    assert cfg.cohort_book_name == "cohort10"


def test_the_seed_base_is_clear_of_every_earlier_stage(p11):
    """
    P09's seeds run 30001-30505, and the earlier stages' are hand-written.

    Two lines drawing the same participation pattern would be a coincidence
    nobody would notice and nobody could undo.
    """
    assert DIGITS_STUDY01.seed_base >= 700000
    first = DIGITS_STUDY01.seed_base + p11.SEED_OFFSET
    assert first > 100000


def test_the_dataset_flags_follow_the_configuration():
    assert SL.setting(DIGITS_STUDY01) == "--resolution 28 --classes digits"


# --------------------------------------------------------------------------- #
# the table and the grouping
# --------------------------------------------------------------------------- #
def test_the_screen_is_the_same_cell_table_as_always(p11):
    """
    The table is 171 now; the screen that ran is still its first 169.

    The two extra cells are the trim row's boundary extension, appended after
    the fact - see the extension tests below.
    """
    assert len(agg_cells.screen_cells()) == 171
    assert p11.SCREENED_CELLS == 169
    assert len(p11.screened_lines()) == 845
    assert SL.counts(DIGITS_STUDY01)["agg_screen"] == 855


def test_the_methods_are_the_eighteen():
    """
    A weighting rule is not a coefficient of another weighting rule, and a
    control with the oracle stop rule armed is not a setting of the one without.
    """
    methods = {method for _, method in SL.AGG_METHODS}
    assert len(SL.AGG_METHODS) == 18
    for name in ("weight_uniform", "weight_capped",
                 "control_fedavg", "control_fedavg_earlystop"):
        assert name in methods, name
    assert SL.counts(DIGITS_STUDY01)["agg_full"] == 90


def test_every_cell_belongs_to_exactly_one_method():
    from collections import Counter

    counted = Counter(
        (cell["path"], SL.agg_method_of(cell)) for cell in agg_cells.screen_cells()
    )
    assert set(counted) == set(SL.AGG_METHODS)
    assert sum(counted.values()) == 171
    # a wider row is not a new method
    assert counted[("concurrent", "trimmed")] == 4


# --------------------------------------------------------------------------- #
# the 845 lines
# --------------------------------------------------------------------------- #
def test_the_screen_is_every_cell_at_every_fold(p11, parsed):
    assert len(parsed) == 845
    cells = {c["id"] for c in agg_cells.screen_cells()[:p11.SCREENED_CELLS]}
    seen = {
        (a.parent[len("d01_agg_"):].rsplit("_fold", 1)[0], a.fold) for a in parsed
    }
    assert len(seen) == 845
    assert {c for c, _ in seen} == cells


def test_every_line_draws_nine_of_ten(parsed):
    for args in parsed:
        assert args.clients_per_round == 9
        assert args.policy == "uniform"


def test_every_line_is_the_digit_task_at_the_screening_horizon(parsed):
    for args in parsed:
        assert args.classes == "digits" and args.resolution == 28
        assert args.rounds == SL.SCREEN_ROUNDS == 25
        assert (args.epochs, args.batch_size) == (5, 64)
        assert args.trainer == "BaseTrainer"


def test_every_line_reads_the_cohort_and_the_old_book(parsed):
    for args in parsed:
        assert args.outliers_file.endswith("outliers/cohort_worst10.json")
        assert args.fold_book.endswith("fold_books/cohort10.foldbook.npz")
        assert args.old_book.endswith("fold_books/old_data.foldbook.npz")
        assert args.old_clients_file.endswith("outliers/old_data.json")


def test_every_line_starts_from_g0_and_keeps_its_model(parsed):
    for args in parsed:
        assert args.init == "global" and args.global_name == "g0"
        assert args.save_final_model and args.track_clients


def test_preservation_is_five_folds_on_every_line(parsed):
    """Five partitions of one population is what the error bar is made of."""
    for args in parsed:
        assert args.old_fold == "all"


def test_no_line_touches_an_archived_or_foreign_artefact(p11, parsed):
    text = "\n".join(p11.lines())
    # "T1_" alone would be a false positive on a KD-style cell id; match the
    # retired STUDY names rather than the letter.
    for dead in ("v1_superseded", "main_v6", "studies/T", "T1_20outliers",
                 "T1_10outliers", "T2_20outliers", "T2_10outliers",
                 "T3_20normals", "T3_10normals", "all_writers_digits"):
        assert dead not in text, dead
    for args in parsed:
        for attr in ("results_dir", "outliers_file", "fold_book", "old_book",
                     "old_clients_file"):
            value = getattr(args, attr, None)
            if isinstance(value, str):
                assert value.startswith("$FOA_STUDY_DIR"), (attr, value)


def test_the_lines_are_845_distinct_cells_with_845_distinct_seeds(parsed):
    assert len({a.parent for a in parsed}) == 845
    assert len({a.sampler_seed for a in parsed}) == 845


def test_no_seed_collides_with_the_federated_stage(parsed):
    """P09 used 30001-30505; a shared seed would repeat a draw silently."""
    p09 = set(range(30001, 30506))
    assert not {a.sampler_seed for a in parsed} & p09


def test_each_named_rule_resolves_to_its_own_family(parsed):
    """
    A named rule runs one family; only the controls span two, by definition.

    A cell resolving to the wrong schedule would be a different experiment under
    the same name.
    """
    from federated_outlier_adaptation.aggregation.selector import (
        AGG_VARIANTS,
        SKIP_FAMILY,
        resolve_aggregation,
    )

    cells = {c["id"]: c for c in agg_cells.screen_cells()}
    variants = 0
    for args in parsed:
        cell = cells[args.parent[len("d01_agg_"):].rsplit("_fold", 1)[0]]
        if args.aggregation in AGG_VARIANTS:
            assert cell["path"] == "control"
            variants += 1
            continue
        families = [
            (scenario, metadata)
            for scenario in ("concurrent", "sequential")
            for metadata in ("weights", "delta")
            if resolve_aggregation(scenario, metadata, args.aggregation,
                                   extended=True) != SKIP_FAMILY
        ]
        assert len(families) == 1, (cell["id"], families)
        assert families[0][0] == cell["path"]
    # the two control cells, five folds each
    assert variants == 10


def test_the_coefficients_round_trip_exactly(parsed, p11):
    """A label may round; a coefficient may not."""
    from federated_outlier_adaptation.cli import _server_kwargs

    cells = {c["id"]: c for c in agg_cells.screen_cells()}
    for args in parsed:
        cell = cells[args.parent[len("d01_agg_"):].rsplit("_fold", 1)[0]]
        kwargs = _server_kwargs(args)
        flags = cell["flags"]
        if "server_anchor" in flags:
            assert kwargs["anchor_lambda"] == flags["server_anchor"]
        if "server_lr" in flags:
            assert kwargs["server_lr"] == flags["server_lr"]
        if "server_tau" in flags:
            assert kwargs["tau"] == flags["server_tau"]
        if "server_eta" in flags:
            assert args.server_eta == flags["server_eta"]
        if "seq_mix_alpha" in flags:
            assert args.seq_mix_alpha == flags["seq_mix_alpha"]


# --------------------------------------------------------------------------- #
# the selection
# --------------------------------------------------------------------------- #
def _result(root: Path, prefix: str, cell_id: str, fold: int, adaptation: float,
            preservation: float = 0.9):
    run = (root / f"{prefix}{cell_id}_fold{fold}_x_grid_search" / "fold1_seed_1"
           / "concurrent_delta" / "base_agg_x")
    run.mkdir(parents=True, exist_ok=True)
    (run / "accuracies_0.json").write_text(json.dumps({
        "scenario": "concurrent",
        "accuracies": [[adaptation, 0.5]] * 25,
        "pool_val_accuracies": [adaptation] * 25,
        "final_evaluation": {
            "clients": {"accuracy": adaptation - 0.01},
            "old": {"mean": preservation, "sd": 0.0},
        },
    }))


def test_the_selection_keeps_every_method(emit, tmp_path):
    """
    No method filtering: eighteen methods, eighteen winners.

    Selecting one overall winner would leave the full stage comparing it against
    nothing.
    """
    cells = agg_cells.screen_cells()
    for index, cell in enumerate(cells):
        for fold in (1, 2):
            _result(tmp_path, "d01_agg_", cell["id"], fold, 0.5 + index * 1e-4)

    out = tmp_path / "p12.txt"
    expect = SL.counts(DIGITS_STUDY01)["agg_full"]
    assert emit.agg_full(DIGITS_STUDY01, tmp_path, out, expect) == 0

    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == expect == 90
    record = json.loads(
        (tmp_path / "tables" / "p11_agg_method_winners.json").read_text()
    )
    assert len(record) == 18 and all(e["measured"] for e in record.values())


def test_a_miscount_halts_rather_than_mismatching_the_array(emit, tmp_path):
    for cell in agg_cells.screen_cells():
        _result(tmp_path, "d01_agg_", cell["id"], 1, 0.5)
    assert emit.agg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p12.txt", 999) == 1


def test_the_emitted_p12_file_says_it_is_not_authorised(emit, tmp_path):
    """It exists so the file is ready for review, not so it can be submitted."""
    for cell in agg_cells.screen_cells():
        _result(tmp_path, "d01_agg_", cell["id"], 1, 0.5)
    out = tmp_path / "p12.txt"
    emit.agg_full(DIGITS_STUDY01, tmp_path, out, 90)
    text = out.read_text()
    assert "NOT YET AUTHORISED" in text
    assert "gated separately" in text


def test_the_emitted_p12_lines_are_the_full_horizon(emit, tmp_path):
    from federated_outlier_adaptation.cli import build_parser

    for cell in agg_cells.screen_cells():
        _result(tmp_path, "d01_agg_", cell["id"], 1, 0.5)
    out = tmp_path / "p12.txt"
    emit.agg_full(DIGITS_STUDY01, tmp_path, out, 90)

    parser = build_parser()
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    for line in lines:
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.rounds == SL.FULL_ROUNDS == 100
        assert args.clients_per_round == 9 and args.classes == "digits"
        assert args.old_fold == "all"
    assert len({shlex.split(l)[shlex.split(l).index("--parent") + 1] for l in lines}) == 90


def test_grid_edge_winners_are_flagged_and_do_not_halt(emit, tmp_path):
    """
    A winner at the end of its row means the optimum may be outside the range.

    Worth reporting; not a reason to stop, because the run is still the best of
    what was tried.
    """
    for cell in agg_cells.screen_cells():
        score = 0.9 if cell["id"] == "eta_0p1" else 0.4
        for fold in (1, 2):
            _result(tmp_path, "d01_agg_", cell["id"], fold, score)
    assert emit.agg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p12.txt", 90) == 0
    hits = (tmp_path / "tables" / "BOUNDARY_HITS.txt").read_text()
    assert "eta_0p1" in hits and "LOW end" in hits


def test_the_selector_is_still_parameterised_by_prefix(emit):
    """One selection rule, addressed at whichever study is being read."""
    prefixes = emit.prefixes(DIGITS_STUDY01)
    assert prefixes["agg_screen"] == "d01_agg_"
    assert prefixes["agg_full"] == "d01_aggfull_"
    # the regularisation path was added later and shares the same mechanism
    assert set(prefixes) >= {"agg_screen", "agg_full"}
    assert all(value.startswith(DIGITS_STUDY01.tag) for value in prefixes.values())


def test_the_readme_gives_the_exact_selection_command(p11):
    text = p11.readme(p11.lines())
    assert "tools/study_emit.py" in text and "agg-full" in text
    assert "--expect 90" in text
    assert "NOT AUTHORISED" in text


# --------------------------------------------------------------------------- #
# the boundary extension of the trim row
# --------------------------------------------------------------------------- #
def test_the_trim_row_is_now_four_wide():
    ids = [c["id"] for c in agg_cells.screen_cells() if c["id"].startswith("trimmed_")]
    assert ids == ["trimmed_0p1", "trimmed_0p2", "trimmed_0p3", "trimmed_0p4"]
    assert agg_cells.TRIM_FRACTIONS_EXT == (0.3, 0.4)


def test_the_new_cells_are_appended_so_no_existing_seed_moves(p11):
    """
    A screening seed is a function of the cell's index in the table.

    Inserting 0.3 and 0.4 into the trim row would have re-seeded every cell
    after it, and the screen that has already run would no longer be
    reproducible from the table describing it.
    """
    cells = agg_cells.screen_cells()
    assert len(cells) == 171
    # the originals keep their positions; the new ones are at the very end
    positions = {c["id"]: i for i, c in enumerate(cells)}
    assert positions["trimmed_0p1"] == 8 and positions["trimmed_0p2"] == 9
    assert positions["trimmed_0p3"] == 169 and positions["trimmed_0p4"] == 170
    assert [c["id"] for c in cells[:p11.SCREENED_CELLS]] == [
        c["id"] for c in cells[:p11.SCREENED_CELLS]
    ]


def test_the_screen_file_is_unchanged_by_the_extension(p11):
    """The 845 already-submitted lines must be exactly what they were."""
    screened = p11.screened_lines()
    assert len(screened) == 845
    assert screened == p11.lines()[:845]
    for line in screened:
        for new in ("trimmed_0p3", "trimmed_0p4"):
            assert new not in line


def test_the_extension_file_is_the_two_new_cells(p11):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    ext = p11.ext_lines()
    assert len(ext) == 10
    args = [parser.parse_args(shlex.split(line)[1:]) for line in ext]
    cells = {a.parent[len("d01_agg_"):].rsplit("_fold", 1)[0] for a in args}
    assert cells == {"trimmed_0p3", "trimmed_0p4"}
    assert {a.trim_frac for a in args} == {0.3, 0.4}
    assert {a.fold for a in args} == {1, 2, 3, 4, 5}


def test_the_extension_runs_the_screen_s_protocol(p11):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for line in p11.ext_lines():
        args = parser.parse_args(shlex.split(line)[1:])
        assert args.classes == "digits" and args.rounds == 25
        assert args.clients_per_round == 9 and args.policy == "uniform"
        assert args.init == "global" and args.global_name == "g0"
        assert args.old_fold == "all"
        assert args.aggregation == "con_delta_trimmed_mean"
        assert args.outliers_file.endswith("cohort_worst10.json")
        assert args.fold_book.endswith("cohort10.foldbook.npz")


def test_the_extension_seeds_are_fresh_and_continue_the_scheme(p11):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    screened = {
        parser.parse_args(shlex.split(l)[1:]).sampler_seed
        for l in p11.screened_lines()
    }
    ext = {
        parser.parse_args(shlex.split(l)[1:]).sampler_seed for l in p11.ext_lines()
    }
    assert len(ext) == 10
    assert not ext & screened
    assert min(ext) > max(screened)
    # and still clear of the federated stage
    assert not ext & set(range(30001, 30506))


def test_the_selector_sees_all_four_as_one_method():
    """Grouping is by id prefix, so position in the table is irrelevant."""
    trimmed = [
        c for c in agg_cells.screen_cells() if SL.agg_method_of(c) == "trimmed"
    ]
    assert len(trimmed) == 4
    # the method count is unchanged: a wider row is not a new method
    assert len(SL.AGG_METHODS) == 18
    assert SL.counts(DIGITS_STUDY01)["agg_full"] == 90


def test_the_widened_range_is_what_boundary_detection_now_sees(emit):
    """
    0.2 was the high edge of [0.1, 0.2]; it is interior to [0.1..0.4].

    If the extension confirms 0.4 as the winner that is a genuine edge finding,
    not an artefact of a short row.
    """
    cells = agg_cells.screen_cells()
    siblings = [c for c in cells if c["id"].startswith("trimmed_")]
    by_id = {c["id"]: c for c in siblings}

    assert emit.numeric_boundary(by_id["trimmed_0p2"], siblings, "flags") == []
    low = emit.numeric_boundary(by_id["trimmed_0p1"], siblings, "flags")
    high = emit.numeric_boundary(by_id["trimmed_0p4"], siblings, "flags")
    assert any("LOW end" in h for h in low)
    assert any("HIGH end" in h for h in high)


# --------------------------------------------------------------------------- #
# what the trimmed mean actually does at nine clients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraction,survivors", [(0.1, 9), (0.2, 7), (0.3, 5), (0.4, 3)])
def test_how_many_updates_survive_each_trim_fraction(fraction, survivors):
    """
    ``trim = int(f * K)``, and the rule only bites when that is at least one.

    At nine clients ``int(0.1 * 9)`` is ZERO, so trim_frac=0.1 is not a trimmed
    mean at all - it is the plain coordinate-wise mean. The row the boundary
    rule saw was effectively [no trimming, drop one].
    """
    import torch

    from federated_outlier_adaptation.aggregation.concurrent_methods import (
        ServerState,
        con_delta_trimmed_mean,
    )

    K = 9
    assert K - 2 * int(fraction * K) == survivors or int(fraction * K) == 0
    if int(fraction * K) == 0:
        assert survivors == K

    # a coordinate whose extremes are outliers: the surviving count decides the mean
    values = [-100.0] + [1.0] * 7 + [100.0]
    globals_ = {"w": torch.zeros(1)}
    clients = [{"w": torch.tensor([v])} for v in values]
    state = ServerState(eta=1.0, trim_fraction=fraction)
    result = con_delta_trimmed_mean(globals_, clients, [1] * K, server_state=state)
    trim = int(fraction * K)
    kept = sorted(values)[trim: K - trim] if trim else sorted(values)
    assert len(kept) == survivors
    assert result["w"].item() == pytest.approx(sum(kept) / len(kept), abs=1e-4)


def test_a_half_fraction_is_not_reachable():
    """int(0.5 * 9) = 4 would leave one update, which is not a mean of anything."""
    from federated_outlier_adaptation.aggregation.concurrent_methods import ServerState

    with pytest.raises(ValueError, match=r"trim_fraction must lie in \[0, 0.5\)"):
        ServerState(trim_fraction=0.5)


def test_the_documented_survivor_counts_match_the_implementation(p11):
    assert p11.TRIM_SURVIVORS == {0.1: 9, 0.2: 7, 0.3: 5, 0.4: 3}
    assert p11.CLIENTS == DIGITS_STUDY01.clients_per_round == 9
    text = p11.ext_readme(p11.ext_lines())
    assert "is not a trimmed mean" in text
    assert "0.4 is the top of the row that can exist here" in text


# --------------------------------------------------------------------------- #
# the patch
# --------------------------------------------------------------------------- #
def _trim_results(root: Path, best: str):
    """Screen results where ``best`` is the strongest trimmed cell."""
    for cell in agg_cells.screen_cells():
        score = 0.9 if cell["id"] == best else 0.4
        for fold in (1, 2):
            _result(root, "d01_agg_", cell["id"], fold, score)


def test_the_patch_is_five_lines_not_ninety(emit, tmp_path):
    """
    One row widened, so at most one method's winner can move.

    Re-running the other seventeen would spend GPU time reproducing results
    whose inputs did not change.
    """
    _trim_results(tmp_path, "trimmed_0p3")
    out = tmp_path / "patch.txt"
    assert emit.agg_trimmed_patch(DIGITS_STUDY01, tmp_path, out, 5) == 0
    lines = [l for l in out.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 5
    assert all("trimmed_0p3" in l for l in lines)


def test_the_patch_lines_are_the_full_horizon_with_p12_seeds(emit, tmp_path):
    """The patched lines and the ones they replace must be the same runs."""
    from federated_outlier_adaptation.cli import build_parser

    _trim_results(tmp_path, "trimmed_0p4")
    out = tmp_path / "patch.txt"
    emit.agg_trimmed_patch(DIGITS_STUDY01, tmp_path, out, 5)

    full = tmp_path / "p12.txt"
    emit.agg_full(DIGITS_STUDY01, tmp_path, full, 90)

    parser = build_parser()
    patch = [
        parser.parse_args(shlex.split(l)[1:])
        for l in out.read_text().splitlines() if l and not l.startswith("#")
    ]
    p12 = [
        parser.parse_args(shlex.split(l)[1:])
        for l in full.read_text().splitlines() if l and not l.startswith("#")
    ]
    trimmed_in_p12 = [a for a in p12 if "trimmed_" in a.parent]
    assert len(patch) == len(trimmed_in_p12) == 5
    assert {a.rounds for a in patch} == {100}
    assert {a.sampler_seed for a in patch} == {a.sampler_seed for a in trimmed_in_p12}
    assert {a.fold for a in patch} == {1, 2, 3, 4, 5}


def test_the_patch_says_whether_the_winner_moved(emit, tmp_path, capsys):
    _trim_results(tmp_path, "trimmed_0p2")
    emit.agg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p12.txt", 90)
    capsys.readouterr()

    emit.agg_trimmed_patch(DIGITS_STUDY01, tmp_path, tmp_path / "patch.txt", 5)
    out = capsys.readouterr().out
    assert "UNCHANGED" in out
    record = json.loads(
        (tmp_path / "tables" / "p11ext_trimmed_reselection.json").read_text()
    )
    assert record["changed"] is False
    assert record["winner"] == record["previous_winner"] == "trimmed_0p2"
    assert record["row"] == [0.1, 0.2, 0.3, 0.4]


def test_the_patch_reports_a_moved_winner(emit, tmp_path, capsys):
    """The first selection saw only the short row; the extension changes it."""
    for cell in agg_cells.screen_cells():
        if cell["id"].startswith("trimmed_0p3") or cell["id"].startswith("trimmed_0p4"):
            continue
        score = 0.9 if cell["id"] == "trimmed_0p2" else 0.4
        for fold in (1, 2):
            _result(tmp_path, "d01_agg_", cell["id"], fold, score)
    emit.agg_full(DIGITS_STUDY01, tmp_path, tmp_path / "p12.txt", 90)
    capsys.readouterr()

    # now the extension lands, and 0.4 wins
    for fold in (1, 2):
        _result(tmp_path, "d01_agg_", "trimmed_0p4", fold, 0.99)
    emit.agg_trimmed_patch(DIGITS_STUDY01, tmp_path, tmp_path / "patch.txt", 5)

    out = capsys.readouterr().out
    assert "CHANGED" in out
    record = json.loads(
        (tmp_path / "tables" / "p11ext_trimmed_reselection.json").read_text()
    )
    assert record["previous_winner"] == "trimmed_0p2"
    assert record["winner"] == "trimmed_0p4" and record["changed"] is True
    # 0.4 is the high edge of the widened row - a genuine finding, still flagged
    assert any("HIGH end" in h for h in record["boundary_hits"])


def test_the_patch_is_stamped_unauthorised(emit, tmp_path):
    _trim_results(tmp_path, "trimmed_0p3")
    out = tmp_path / "patch.txt"
    emit.agg_trimmed_patch(DIGITS_STUDY01, tmp_path, out, 5)
    assert "NOT AUTHORISED" in out.read_text()


def test_a_patch_miscount_halts(emit, tmp_path):
    _trim_results(tmp_path, "trimmed_0p3")
    assert emit.agg_trimmed_patch(
        DIGITS_STUDY01, tmp_path, tmp_path / "patch.txt", 90
    ) == 1
