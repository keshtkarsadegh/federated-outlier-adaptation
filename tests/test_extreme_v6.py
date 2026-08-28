"""
Stage 10: the extreme cases, and the merged client the third one needs.

``single`` and ``double`` are federations the pipeline already knows how to
build.  ``dual`` is not: it is one client whose data is two writers' data, and
the study had no word for that object.  Most of this file is about that word -
that a merged client's partitions are exact unions of its members', that nothing
leaks between partitions, and that everything downstream counts it as one
participant rather than two.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.data.merged_clients import (
    MERGE_SEPARATOR,
    expand,
    is_merged,
    members,
    merged_id,
    unknown_members,
)
from federated_outlier_adaptation.training.extreme_cells import (
    CASES,
    FULL_ROUNDS,
    WINNING_COMBO,
    accuracy_map,
    case_clients,
    case_writers,
    extreme_cells,
    rank_cohort,
)

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


# --------------------------------------------------------------------------- #
# The id is the definition
# --------------------------------------------------------------------------- #
def test_a_merged_id_carries_its_own_members():
    """No registry to keep in step: the id says what it means."""
    assert merged_id(["a", "b"]) == f"a{MERGE_SEPARATOR}b"
    assert is_merged("a+b") and not is_merged("a")
    assert members("a+b") == ["a", "b"]
    assert members("a") == ["a"]
    assert expand(["a+b", "c"]) == ["a", "b", "c"]


def test_a_single_writer_merges_to_itself():
    assert merged_id(["a"]) == "a"
    assert not is_merged(merged_id(["a"]))


def test_a_writer_id_may_not_contain_the_separator():
    with pytest.raises(ValueError, match="may not contain"):
        merged_id(["a+b", "c"])
    with pytest.raises(ValueError, match="at least one writer"):
        merged_id([])


def test_a_stray_separator_produces_no_empty_member():
    """An empty member would match no writer and contribute no rows, silently."""
    assert members("a++b") == ["a", "b"]
    assert members("a+") == ["a"]


def test_a_merged_id_is_valid_when_its_members_are():
    assert unknown_members(["a+b"], {"a", "b"}) == []
    assert unknown_members(["a+b"], {"a"}) == ["b"]
    assert unknown_members(["a+b", "c"], {"a", "b"}) == ["c"]


# --------------------------------------------------------------------------- #
# The merge is a union per partition
# --------------------------------------------------------------------------- #
def test_the_merged_partitions_are_exact_unions(stage5_setup, cohort_cache, tmp_path, monkeypatch):
    """
    Train with train, validation with validation, test with test.

    Not a re-split of the combined data: a re-split would move rows between
    partitions, and then 'dual' would no longer hold what 'double' holds, which
    is the only reason the pair is worth running.
    """
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "2")
    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache, resolution=28, classes="all"
    )

    first, second = cohort_writers[0], cohort_writers[1]
    loaders = provider.build_dataset(merged_id([first, second]), batch_size=4)
    for part, loader in zip(("train", "val", "test"), loaders):
        expected = sorted(
            cohort.part(2, first, part) + cohort.part(2, second, part)
        )
        assert sorted(loader.dataset.rows) == expected
        assert expected, "the test would be vacuous"


def test_nothing_leaks_between_the_merged_partitions(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "2")
    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache, resolution=28, classes="all"
    )

    merged = merged_id([cohort_writers[0], cohort_writers[1]])
    train, val, test = provider.build_dataset(merged, batch_size=4)
    parts = [set(loader.dataset.rows) for loader in (train, val, test)]
    assert not parts[0] & parts[1]
    assert not parts[0] & parts[2]
    assert not parts[1] & parts[2]


def test_the_merged_client_holds_exactly_what_the_two_hold(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    """'dual' and 'double' are the same rows; only the boundaries differ."""
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "1")
    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache, resolution=28, classes="all"
    )

    pair = cohort_writers[:2]
    merged_rows = set(provider.build_dataset(merged_id(pair), batch_size=4)[0].dataset.rows)
    separate = set()
    for writer in pair:
        separate |= set(provider.build_dataset(writer, batch_size=4)[0].dataset.rows)
    assert merged_rows == separate


def test_a_merged_client_weighs_as_much_as_its_members(stage5_setup):
    """Aggregation weights by held data, and a merged client holds both lots."""
    provider, cohort_writers, _, _, _ = stage5_setup
    pair = cohort_writers[:2]
    assert provider.sample_count(merged_id(pair)) == sum(
        provider.sample_count(writer) for writer in pair
    )


def test_the_pool_reads_the_book_for_a_merged_client(
    stage5_setup, cohort_cache, tmp_path, monkeypatch
):
    """The reported test rows of a merged client are the book's, unioned."""
    from federated_outlier_adaptation.data.fold_book import write_fold_book
    from federated_outlier_adaptation.providers.nist import NistProvider
    from federated_outlier_adaptation.runners.population import ClientPopulation

    _, cohort_writers, cohort, _, _ = stage5_setup
    path = write_fold_book(cohort, tmp_path / "cohort")
    monkeypatch.setenv("FOA_FOLD_BOOK", str(path))
    monkeypatch.setenv("FOA_FOLD", "3")
    provider = NistProvider(
        results_dir=tmp_path / "r", cache_dir=cohort_cache, resolution=28, classes="all"
    )

    pair = cohort_writers[:2]
    merged = merged_id(pair)
    population = ClientPopulation(provider=provider, selected=[merged], loader_seed=1)
    val_loader, test_loader = population.split_loaders(merged, batch_size=4)
    for part, loader in (("val", val_loader), ("test", test_loader)):
        expected = sorted(r for w in pair for r in cohort.part(3, w, part))
        assert sorted(loader.dataset.rows) == expected


def test_the_merged_client_is_one_row_of_the_per_client_column(stage5_setup):
    """One participant, one row - keyed by the merged id."""
    from federated_outlier_adaptation.training.evaluate import evaluate_on_book

    provider, cohort_writers, cohort, _, _ = stage5_setup
    pair = cohort_writers[:2]
    merged = merged_id(pair)

    result = evaluate_on_book(
        provider, provider.make_model(), cohort, fold=1, writers=[merged],
        part="test", batch_size=8,
    )
    assert list(result["per_writer"]) == [merged]
    assert result["writers"] == 1
    assert result["samples"] == sum(len(cohort.part(1, w, "test")) for w in pair)


def test_a_merged_participant_is_accepted_by_the_population(stage5_setup):
    from federated_outlier_adaptation.runners.population import resolve_participants

    provider, cohort_writers, _, _, _ = stage5_setup
    merged = merged_id(cohort_writers[:2])
    unique, participants = resolve_participants(provider, [], [merged])
    assert unique == [merged] and participants == [merged]


def test_a_merged_participant_with_an_unknown_member_is_refused(stage5_setup):
    from federated_outlier_adaptation.runners.population import (
        UnknownClientError,
        resolve_participants,
    )

    provider, cohort_writers, _, _, _ = stage5_setup
    bad = merged_id([cohort_writers[0], "no_such_writer"])
    with pytest.raises(UnknownClientError, match="no_such_writer"):
        resolve_participants(provider, [], [bad])


# --------------------------------------------------------------------------- #
# The three cases
# --------------------------------------------------------------------------- #
def test_the_ranking_is_worst_first_and_deterministic():
    ranked = rank_cohort({"a": 0.5, "b": 0.4, "c": 0.6}, ["a", "b", "c"])
    assert ranked == ["b", "a", "c"]
    # ties break on the writer id, so the ranking is a function of the data
    assert rank_cohort({"b": 0.4, "a": 0.4}, ["b", "a"]) == ["a", "b"]


def test_both_shapes_of_the_accuracy_file_are_read():
    """
    The published pipeline writes a list of one-key dicts, not a flat mapping.

    Guessing wrong would not fail loudly - it would rank the cohort by nothing
    and pick whichever writer sorted first.
    """
    as_list = [{"a": 0.5}, {"b": 0.4}]
    as_dict = {"a": 0.5, "b": 0.4}
    assert accuracy_map(as_list) == accuracy_map(as_dict) == {"a": 0.5, "b": 0.4}


def test_an_unscored_cohort_writer_stops_the_ranking():
    with pytest.raises(KeyError, match="No g-0 accuracy"):
        rank_cohort({"a": 0.5}, ["a", "b"])


def test_double_and_dual_take_the_same_writers():
    """The controlled pair: same rows, different boundaries."""
    ranked = ["w1", "w2", "w3"]
    assert case_writers(ranked, "double") == case_writers(ranked, "dual") == ["w1", "w2"]
    assert case_writers(ranked, "single") == ["w1"]

    assert case_clients(ranked, "double") == ["w1", "w2"]
    assert case_clients(ranked, "dual") == ["w1+w2"]
    assert case_clients(ranked, "single") == ["w1"]


def test_the_cases_differ_only_in_how_the_rows_are_held():
    ranked = ["w1", "w2", "w3"]
    cells = {cell["id"]: cell for cell in extreme_cells(ranked)}
    assert set(cells) == set(CASES)
    assert cells["double"]["writers"] == cells["dual"]["writers"]
    assert cells["double"]["participants"] == 2
    assert cells["dual"]["participants"] == 1


def test_too_few_writers_is_an_error():
    with pytest.raises(ValueError, match="needs 2 writer"):
        case_writers(["w1"], "double")
    with pytest.raises(ValueError, match="Unknown extreme case"):
        case_writers(["w1"], "triple")


# --------------------------------------------------------------------------- #
# The task file
# --------------------------------------------------------------------------- #
@pytest.fixture()
def stage10(tools_path):
    import make_stage10_extreme

    return make_stage10_extreme


def test_every_line_is_the_winning_method_with_this_case_s_clients(stage10):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    combo = stage10.winning_combo()
    cells = extreme_cells(["w1", "w2", "w3"])

    for cell in cells:
        for fold in (1, 5):
            args = parser.parse_args(
                shlex.split(stage10.task_line(cell, combo, fold, 100))[1:]
            )
            assert args.func.__name__ == "cmd_final"
            assert args.rounds == 100 and args.epochs == 5 and args.batch_size == 64
            assert args.fold == fold and args.seed == fold
            assert args.init == "global" and args.global_name == "g0"
            assert args.save_final_model
            assert args.old_fold == "all"
            # the winner's method, copied not restated
            assert args.aggregation == combo["agg"]["rule"] == "seq_delta_capped"
            assert args.parent == f"coh10_extreme_{cell['id']}_fold{fold}"
            assert args.outliers_file.endswith(f"extreme_{cell['case']}.json")


def test_the_winning_penalty_reaches_the_trainer(stage10):
    from federated_outlier_adaptation.cli import _trainer_overrides, build_parser

    parser = build_parser()
    combo = stage10.winning_combo()
    cell = extreme_cells(["w1", "w2"])[0]
    args = parser.parse_args(shlex.split(stage10.task_line(cell, combo, 1, 100))[1:])
    overrides = _trainer_overrides(args)

    assert overrides["space"] == "kd+fisher"
    assert overrides["anchor"] == "frozen"
    assert overrides["mix"] == pytest.approx(0.5)
    for name, value in combo["reg"]["hypers"].items():
        assert float(overrides[name]) == float(value)
    assert "fisher_path" in overrides and "$G0_FOLD" in overrides["fisher_path"]


def test_participation_is_full_and_the_sampler_is_off(stage10):
    """Dropping a fifth of a two-client federation is a coin flip, not a study."""
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    combo = stage10.winning_combo()
    for cell in extreme_cells(["w1", "w2", "w3"]):
        args = parser.parse_args(shlex.split(stage10.task_line(cell, combo, 2, 100))[1:])
        assert args.policy == "all"
        assert args.participation == pytest.approx(1.0)
        assert args.clients_per_round is None


def test_a_changed_stage_eight_line_stops_the_copy(stage10, monkeypatch):
    """
    Stage 10 copies stage 8's line; if it stops matching, it must not proceed.

    A silently failed substitution would leave this stage running stage 8 under
    a stage-10 name, which is the sort of thing nobody notices until the numbers
    are in a table.
    """
    combo = stage10.winning_combo()
    cell = extreme_cells(["w1", "w2"])[0]
    monkeypatch.setattr(
        stage10, "combo_task_line", lambda *a, **k: "foa final --parent something_else"
    )
    with pytest.raises(SystemExit, match="no longer contains"):
        stage10.task_line(cell, combo, 1, 100)


def test_the_client_lists_are_written_as_the_runs_read_them(stage10, tmp_path):
    cells = extreme_cells(["w1", "w2", "w3"])
    written = stage10.write_client_lists(tmp_path, cells, tmp_path / "outliers")
    assert set(written) == set(CASES)
    for cell in cells:
        payload = json.loads(Path(written[cell["case"]]).read_text())
        assert payload == cell["clients"]
    assert json.loads(Path(written["dual"]).read_text()) == ["w1+w2"]
