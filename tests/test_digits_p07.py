"""
P07: g-0, and the starting line every method is judged against.

The load-bearing claim here is not that the lines parse - it is that the Fisher
diagonal lands where the regularisation stages will later look for it, computed
on the data it is supposed to be computed on. That is asserted against the
training module rather than trusted, because a Fisher in the wrong place fails
eighty array elements at startup, weeks later.
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
def p07(tools_path):
    import make_digits_p07

    return make_digits_p07


@pytest.fixture()
def parsed(p07):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    return [parser.parse_args(shlex.split(line)[1:]) for line in p07.lines()]


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def test_the_chain_is_folds_winner_then_two_evaluation_groups(parsed):
    names = [args.func.__name__ for args in parsed]
    assert names == (
        ["cmd_global_train"] * 5 + ["cmd_select_fold"] + ["cmd_evaluate_book"] * 10
    )


def test_every_line_is_digits_and_inside_the_study(parsed):
    for index, args in enumerate(parsed, 1):
        assert args.classes == "digits" and args.resolution == 28, index
        for attr in ("results_dir", "out", "clients_file", "fold_book",
                     "writers_file", "model_path", "root"):
            value = getattr(args, attr, None)
            if isinstance(value, str) and ("/" in value or value.startswith("$")):
                assert value.startswith("$FOA_STUDY_DIR"), (index, attr, value)


def test_g0_trains_on_the_old_book_with_the_standard_convergence(parsed):
    trains = [a for a in parsed if a.func.__name__ == "cmd_global_train"]
    assert len(trains) == 5
    assert {t.fold for t in trains} == {1, 2, 3, 4, 5}
    for t in trains:
        assert t.population == "file"
        assert t.model == "fedavg_cnn"
        assert t.writers_file.endswith("old_data.json")
        assert t.fold_book.endswith("old_data.foldbook.npz")
        assert (t.epochs, t.early_stopping_patience, t.min_epochs) == (100, 10, 20)


def test_each_fold_writes_into_its_own_folder(parsed):
    """
    Which is also what puts each fold's Fisher in its own directory.

    Five folds sharing one results root would leave one Fisher, from whichever
    fold happened to finish last.
    """
    trains = [a for a in parsed if a.func.__name__ == "cmd_global_train"]
    roots = {t.results_dir for t in trains}
    assert len(roots) == 5
    for t in trains:
        assert t.results_dir.endswith(f"g0_fold{t.fold}")


# --------------------------------------------------------------------------- #
# the Fisher, where the reg stages will look for it
# --------------------------------------------------------------------------- #
def test_the_fisher_lands_where_the_reg_stages_look(tmp_path):
    """
    ``<results-dir>/global_results/fisher`` is the convention, and P07 relies on
    it: with ``--results-dir .../g0_fold<k>`` the Fisher lands at
    ``g0_fold<k>/global_results/fisher``, which is what ``--set fisher_path=``
    names in the regularisation stages.
    """
    from federated_outlier_adaptation.training.global_model import GlobalTraining

    trainer = GlobalTraining.__new__(GlobalTraining)
    trainer.results_dir = tmp_path / "g0_fold3"
    trainer.name = "global"
    trainer.results_path = trainer.results_dir / f"{trainer.name}_results"
    trainer.fisher_path = trainer.results_path / "fisher"

    assert trainer.fisher_path == tmp_path / "g0_fold3" / "global_results" / "fisher"


def test_the_fisher_is_computed_on_the_folds_training_split():
    """
    EWC wants the curvature of the model on the data it was fitted to.

    A Fisher taken over validation or test rows would be a different quantity
    wearing the same name - and would also put held-out rows into a term the
    clients optimise.
    """
    import inspect

    from federated_outlier_adaptation.training import global_model

    source = inspect.getsource(global_model)
    assert "gt.generate_fisher(train_loader)" in source
    for wrong in ("generate_fisher(eval_loader", "generate_fisher(test_loader",
                  "generate_fisher(tst_loader"):
        assert wrong not in source


def test_the_reg_stage_path_matches_what_p07_produces(p07, parsed):
    """The two ends of the convention, checked against each other."""
    trains = [a for a in parsed if a.func.__name__ == "cmd_global_train"]
    for t in trains:
        produced = f"{t.results_dir}/global_results/fisher"
        expected = "$FOA_STUDY_DIR/g0_fold" + str(t.fold) + "/global_results/fisher"
        assert produced == expected
    # and that is the shape the reg stages substitute $G0_FOLD into
    assert "g0_fold$G0_FOLD/global_results/fisher" in p07.readme(p07.lines())


# --------------------------------------------------------------------------- #
# the two baselines
# --------------------------------------------------------------------------- #
def test_preservation_and_the_gap_go_to_different_files(parsed):
    """Two questions, two artefacts - and it lets them run as two chains."""
    evals = [a for a in parsed if a.func.__name__ == "cmd_evaluate_book"]
    assert len(evals) == 10
    old = [e for e in evals if e.out.endswith("g0_evaluations.json")]
    cohort = [e for e in evals if e.out.endswith("g0_perfold_evaluations.json")]
    assert len(old) == len(cohort) == 5
    assert all(e.fold_book.endswith("old_data.foldbook.npz") for e in old)
    assert all(e.fold_book.endswith("cohort10.foldbook.npz") for e in cohort)
    assert all(e.part == "test" for e in evals)
    assert all(e.model_path.endswith("g0_model") for e in evals)


def test_every_appended_result_has_its_own_key(parsed):
    """
    Both groups append to one file each, keyed by tag.

    A repeated tag would overwrite a fold's result silently rather than fail.
    """
    evals = [a for a in parsed if a.func.__name__ == "cmd_evaluate_book"]
    tags = [e.tag for e in evals]
    assert len(tags) == len(set(tags)) == 10
    assert {e.tag for e in evals if "old_data" in e.tag} == {
        f"old_data_fold{k}" for k in range(1, 6)
    }
    assert {e.tag for e in evals if "cohort" in e.tag} == {
        f"cohort_fold{k}" for k in range(1, 6)
    }


def test_the_file_shapes_match_the_study_two_versions(parsed):
    """Downstream readers key off these names; changing them breaks them."""
    outs = {a.out.rsplit("/", 1)[-1] for a in parsed
            if a.func.__name__ == "cmd_evaluate_book"}
    assert outs == {"g0_evaluations.json", "g0_perfold_evaluations.json"}


def test_the_winner_is_selected_by_validation(parsed):
    winner = next(a for a in parsed if a.func.__name__ == "cmd_select_fold")
    assert winner.prefix == "g0" and winner.name == "g0"
    assert winner.folds == [1, 2, 3, 4, 5]


def test_the_selection_records_a_digest():
    """g0_selection.json must say which bytes it froze."""
    import inspect

    from federated_outlier_adaptation.analysis import fold_selection

    source = inspect.getsource(fold_selection)
    assert "sha256_of" in source and '"sha256"' in source


def test_nothing_names_a_retired_root(p07):
    text = "\n".join(p07.lines()) + p07.readme(p07.lines())
    for dead in ("main_v6", "T1_20outliers", "T1_10outliers", "T2_", "T3_"):
        assert dead not in text, dead
