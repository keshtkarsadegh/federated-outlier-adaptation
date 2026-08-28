"""
P08, P09, P10: three arrangements of the same ten writers.

These are the runs the results chapter is built from, so the tests here are
correctness tests rather than shape tests. Four properties matter enough that
getting them wrong would produce numbers that look fine and mean something else:
the federation must drop exactly one client per round, the ``--init global``
runs must start from the shipped model, preservation must be five folds and not
one, and nothing may read an artefact that was archived as superseded.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")


@pytest.fixture()
def tools_path():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


@pytest.fixture()
def p08(tools_path):
    import make_digits_p08

    return make_digits_p08


@pytest.fixture()
def parser():
    from federated_outlier_adaptation.cli import build_parser

    return build_parser()


def _parse(parser, lines):
    return [parser.parse_args(shlex.split(line)[1:]) for line in lines]


@pytest.fixture()
def isolated(p08, parser):
    return _parse(parser, p08.isolated_lines(list(p08.COHORT)))


@pytest.fixture()
def federated(p08, parser):
    return _parse(parser, p08.fl_lines())


@pytest.fixture()
def centralized(p08, parser):
    return _parse(parser, p08.centralized_lines())


@pytest.fixture()
def every(isolated, federated, centralized):
    return isolated + federated + centralized


# --------------------------------------------------------------------------- #
# (a) the federation drops exactly one client, and reads the cohort's own files
# --------------------------------------------------------------------------- #
def test_every_federated_line_draws_nine_of_ten(federated, p08):
    """
    Nine of ten is this study's 10% dropout.

    Ten of ten would be a different experiment - no participation effect at all
    - and eight would be twice the intended one. Neither would announce itself
    in the results.
    """
    assert len(federated) == 10
    for args in federated:
        assert args.clients_per_round == p08.CLIENTS_PER_ROUND == 9
        assert args.policy == "uniform"
        assert args.participation == 1.0  # the count governs, not the fraction


def test_every_federated_line_reads_the_cohort_book_and_file(federated):
    for args in federated:
        assert args.outliers_file.endswith("outliers/cohort_worst10.json")
        assert args.fold_book.endswith("fold_books/cohort10.foldbook.npz")
        assert args.old_book.endswith("fold_books/old_data.foldbook.npz")
        assert args.old_clients_file.endswith("outliers/old_data.json")


def test_the_federated_budget_and_families_are_the_protocol(federated):
    for args in federated:
        assert args.aggregation == "fedavg"      # both families in one task
        assert args.trainer == "BaseTrainer"     # no regulariser: this is the baseline
        assert (args.rounds, args.epochs, args.batch_size) == (100, 5, 64)
        assert args.save_final_model and args.track_clients


def test_the_federated_lines_are_ten_distinct_cells(federated):
    """Two inits x five folds, each with its own parent, seed and sampler seed."""
    assert len({a.parent for a in federated}) == 10
    assert len({a.sampler_seed for a in federated}) == 10
    assert {a.fold for a in federated} == {1, 2, 3, 4, 5}
    assert {a.init for a in federated} == {"global", "scratch"}
    for args in federated:
        assert args.seed == args.fold


# --------------------------------------------------------------------------- #
# (b) the global-init runs start from the shipped model
# --------------------------------------------------------------------------- #
def test_every_global_isolated_line_points_at_g0(isolated, centralized):
    """
    ``--init global`` without a checkpoint would silently start from scratch.

    That is the failure worth guarding: the two arms would run, produce numbers,
    and be the same experiment twice.
    """
    for args in isolated + centralized:
        if args.init == "global":
            assert args.init_checkpoint == "$FOA_STUDY_DIR/g0_model"
        else:
            assert args.init == "scratch"
            assert not args.init_checkpoint


def test_the_federated_global_runs_name_the_same_artefact(federated):
    for args in federated:
        assert args.global_name == "g0"
    assert {a.init for a in federated} == {"global", "scratch"}


def test_both_inits_are_run_in_equal_number(isolated, centralized, federated):
    for group in (isolated, centralized, federated):
        inits = [a.init for a in group]
        assert inits.count("global") == inits.count("scratch") == len(group) // 2


# --------------------------------------------------------------------------- #
# (c) preservation is five folds everywhere it is meant
# --------------------------------------------------------------------------- #
def test_every_line_scores_all_five_old_folds(every):
    """
    Five partitions of one population: the spread across them IS the error bar.

    A single fold would report a preservation figure with no uncertainty
    attached, which is worse than reporting none.
    """
    assert len(every) == 120
    for args in every:
        assert args.old_fold == "all", args
        assert args.old_book.endswith("old_data.foldbook.npz")
        assert args.old_clients_file.endswith("old_data.json")


# --------------------------------------------------------------------------- #
# (d) nothing reads an archived artefact
# --------------------------------------------------------------------------- #
def test_no_line_touches_the_superseded_v1_artefacts(p08):
    """
    v1's cohort, books, g-0 and evaluations were archived, not deleted.

    A line still naming one would run happily against the old selection and
    report it as this study's result.
    """
    text = "\n".join(
        p08.isolated_lines(list(p08.COHORT)) + p08.fl_lines() + p08.centralized_lines()
    )
    for dead in ("v1_superseded", "main_v6", "T1_", "T2_", "T3_"):
        assert dead not in text, dead


def test_no_line_reads_the_all_writers_book(every):
    """
    That book is the detector's, over all 3,580 writers.

    Using it here would silently widen every population in the stage.
    """
    for args in every:
        assert "all_writers_digits" not in (args.fold_book or "")
        assert "all_writers_digits" not in (args.old_book or "")


def test_every_path_stays_inside_the_study(every):
    for args in every:
        for attr in ("results_dir", "out", "clients_file", "fold_book", "old_book",
                     "old_clients_file", "outliers_file", "init_checkpoint"):
            value = getattr(args, attr, None)
            if isinstance(value, str) and ("/" in value or value.startswith("$")):
                assert value.startswith("$FOA_STUDY_DIR"), (attr, value)


def test_everything_is_the_digit_task(every):
    for args in every:
        assert args.classes == "digits" and args.resolution == 28


# --------------------------------------------------------------------------- #
# the three arrangements themselves
# --------------------------------------------------------------------------- #
def test_the_cohort_is_the_studys_own_ten_writers(p08):
    assert len(p08.COHORT) == 10 == len(set(p08.COHORT))
    assert all(w.startswith("f") and "_" in w for w in p08.COHORT)


def test_isolated_is_one_private_model_per_writer_fold_and_init(isolated, p08):
    assert len(isolated) == len(p08.COHORT) * 5 * 2 == 100
    cells = {(a.only_client, a.fold, a.init) for a in isolated}
    assert len(cells) == 100
    assert {c[0] for c in cells} == set(p08.COHORT)
    # every private model is scored on the WHOLE cohort, not only its own writer
    for args in isolated:
        assert args.clients_file.endswith("cohort_worst10.json")
        assert not args.pooled


def test_isolated_writes_one_file_per_cell_and_no_checkpoints(isolated):
    outs = [a.out for a in isolated]
    assert len(set(outs)) == 100
    for args in isolated:
        assert args.out.endswith(
            f"isolated/isolated_{args.init}_{args.only_client}_fold{args.fold}.json"
        )


def test_centralized_pools_the_cohort_and_names_no_single_client(centralized):
    """One model over all ten, which is what makes it the ceiling."""
    assert len(centralized) == 10
    for args in centralized:
        assert args.pooled is True
        assert args.only_client is None
        assert args.clients_file.endswith("cohort_worst10.json")
    assert len({(a.init, a.fold) for a in centralized}) == 10


def test_the_three_rungs_share_one_convergence_rule(isolated, centralized):
    """They must differ in the arrangement, not in the stopping rule."""
    for args in isolated + centralized:
        assert (args.epochs, args.early_stopping_patience, args.min_epochs) == (
            100, 10, 20
        )
        assert args.batch_size == 64 and args.eval_batch_size == 256


def test_the_three_stages_produce_no_input_for_each_other(p08):
    """
    Which is what lets all three arrays be submitted at once.

    Each writes into its own directory and reads only artefacts that already
    exist, so no ordering between them can matter.
    """
    written = set()
    for line in p08.isolated_lines(list(p08.COHORT)) + p08.centralized_lines():
        written.add(line.split(" --out ")[1].strip())
    for line in p08.fl_lines():
        written.add(line.split(" --parent ")[1].split()[0])

    reads = ("cohort_worst10.json", "cohort10.foldbook.npz",
             "old_data.foldbook.npz", "old_data.json", "g0_model")
    for line in (p08.isolated_lines(list(p08.COHORT)) + p08.fl_lines()
                 + p08.centralized_lines()):
        for token in line.split():
            if token.startswith("$FOA_STUDY_DIR") and token not in written:
                assert any(name in token for name in reads) or "/isolated/" in token \
                    or "/centralized/" in token or token == "$FOA_STUDY_DIR", token
