"""
Every setting is scored against its own cohort's shipped model.

The study's rule is ``score = (adaptation - A0) - (P0 - preservation)``. ``P0``
is one number - there is one source population - but ``A0`` is not: the
five-writer cohort, the ten-writer one the search ran on, the twenty-writer one
and the extreme pair are four different populations, and the shipped model is
not equally good on them. A view that subtracts the ten-writer cohort's 0.8225
from a twenty-client run reports the distance between two populations as
something the arm did, and does it silently, because the arithmetic still
closes.

So the shipped views are checked against the shipped evaluation books, row by
row: the score column of every carried setting, of both extreme arrangements,
and the per-client reference the fairness view prints beside them. These read
the artefacts in ``study/artifacts``, so they pass in a clone with no study
root assembled.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import report_tables  # noqa: E402

ARTIFACTS = REPO / "study" / "artifacts" / "Digits_study01"
VIEWS = ARTIFACTS / "tables"

#: The carried settings and the extreme arrangements, each with the cohort its
#: rows were cut from. Written out rather than read from ``SETTING_COHORT`` so
#: that a wrong entry there fails here instead of being confirmed by itself.
SETTINGS = (
    ("sizes_c10d10.csv", "c10"),
    ("sizes_five.csv", "c5"),
    ("sizes_c20d10.csv", "c20"),
    ("sizes_c20d20.csv", "c20"),
    ("extremes.csv", "extreme"),
)


def a0_of(cohort: str) -> float:
    """The shipped model's own accuracy on one cohort, from its own book."""
    book = json.loads((ARTIFACTS / report_tables.COHORT_BOOKS[cohort]).read_text())
    folds = list(book.values())
    return sum(f["accuracy"] for f in folds) / len(folds)


def p0() -> float:
    book = json.loads((ARTIFACTS / "g0_evaluations.json").read_text())
    folds = list(book.values())
    return sum(f["accuracy"] for f in folds) / len(folds)


def rows(name: str) -> list:
    return list(csv.DictReader((VIEWS / "paper" / name).open()))


# ------------------------------------------------- the books are all there
def test_every_cohort_this_study_scores_has_its_own_shipped_model_book():
    """A cohort without a book is a cohort that would borrow another's A0."""
    for cohort, name in report_tables.COHORT_BOOKS.items():
        assert (ARTIFACTS / name).is_file(), f"{cohort}: {name} is not shipped"


def test_the_four_cohorts_disagree():
    """
    The whole point of the per-setting baseline: these are four different
    numbers. If they ever coincide the fix has become untestable and the four
    books have collapsed into one.
    """
    values = {c: round(a0_of(c), 4) for c in report_tables.COHORT_BOOKS}
    assert len(set(values.values())) == 4, values
    # pinned, so a regenerated book that moves is a failure and not a diff
    assert values == {"c10": 0.8225, "c5": 0.8032, "c20": 0.8441, "extreme": 0.7590}


# ------------------------------------------------------ the score columns
@pytest.mark.parametrize("name,cohort", SETTINGS)
def test_each_row_scores_against_its_own_cohort(name: str, cohort: str):
    """(a) and (b): the arithmetic of every carried and extreme row."""
    a0, source = a0_of(cohort), p0()
    found = rows(name)
    assert found, name
    for row in found:
        adaptation = float(row["adaptation"])
        preservation = float(row["preservation"])
        expected = (adaptation - a0) - (source - preservation)
        assert float(row["score"]) == pytest.approx(expected, abs=5e-5), \
            f"{name}:{row['cell']}/{row['family']}"
        assert float(row["gained"]) == pytest.approx(adaptation - a0, abs=5e-5)
        assert float(row["spent"]) == pytest.approx(source - preservation, abs=5e-5)


@pytest.mark.parametrize("name,cohort", SETTINGS)
def test_no_row_scores_against_the_search_cohort_instead(name: str, cohort: str):
    """
    The defect this file exists for, stated as its own check.

    A row of a setting that is not the search one must NOT close against
    0.8225. Without this, dropping the per-setting lookup would still pass the
    test above on the ten-client rows and fail nowhere else.
    """
    if cohort == "c10":
        pytest.skip("the search setting's own cohort IS the ten-writer one")
    search = a0_of("c10")
    for row in rows(name):
        wrong = (float(row["adaptation"]) - search) - (p0() - float(row["preservation"]))
        assert float(row["score"]) != pytest.approx(wrong, abs=5e-5), \
            f"{name}:{row['cell']} is still scored against the search cohort"


# ------------------------------------------------------- the fairness view
@pytest.mark.parametrize("name,cohort", (
    ("fairness_c10d10.csv", "c10"),
    ("fairness_five.csv", "c5"),
    ("fairness_c20d10.csv", "c20"),
    ("fairness_c20d20.csv", "c20"),
))
def test_the_fairness_view_references_the_same_cohort(name: str, cohort: str):
    """
    (c) The per-client references agree with the setting's own A0.

    The fairness view reads a per-WRITER book and the score tables read the
    same book's per-FOLD accuracy, so ``g0_mean`` and ``A0`` are not the same
    average and do not have to agree to the digit - but they must come from
    the same book. This checks that ``g0_mean`` is the per-writer mean of THIS
    cohort's book and of no other, which is the property that makes the
    fairness table readable beside the score table.
    """
    def writer_mean(which: str) -> float:
        book = json.loads((ARTIFACTS / report_tables.COHORT_BOOKS[which]).read_text())
        per_writer: dict = {}
        for fold in book.values():
            for writer, accuracy in fold["per_writer"].items():
                per_writer.setdefault(writer, []).append(accuracy)
        return sum(sum(v) / len(v) for v in per_writer.values()) / len(per_writer)

    own = writer_mean(cohort)
    alien = [writer_mean(c) for c in report_tables.COHORT_BOOKS if c != cohort]
    for row in rows(name):
        assert float(row["g0_mean"]) == pytest.approx(own, abs=5e-5), \
            f"{name}:{row['cell']} does not reference the {cohort} book"
        for other in alien:
            assert float(row["g0_mean"]) != pytest.approx(other, abs=5e-5)


# --------------------------------------------------------- the lookup itself
def test_the_stem_to_cohort_table_covers_every_stage_that_is_reported():
    for stem in ("d01_five_", "five_", "c20d10_", "c20d20_", "c10d10_",
                 "extreme_", "aggfull_", "regfull_", "combo_"):
        assert report_tables.cohort_of(stem) in report_tables.COHORT_BOOKS


def test_twenty_four_dropped_is_not_read_as_twenty_two_dropped():
    """Longest-stem matching, which is the one way this table can go wrong."""
    assert report_tables.cohort_of("c20d20_") == "c20"
    assert report_tables.cohort_of("c10d10_") == "c10"


def test_an_unregistered_setting_is_refused_rather_than_defaulted():
    with pytest.raises(SystemExit):
        report_tables.cohort_of("d01_something_new_")


def test_a_root_without_the_cohorts_book_is_refused(tmp_path: Path):
    """
    Refusing is the point: the old code had no book to miss, because every
    setting borrowed the ten-writer one.
    """
    root = tmp_path / "study"
    root.mkdir()
    (root / "g0_perfold_evaluations.json").write_text(json.dumps(
        {"cohort_fold1": {"fold": 1, "accuracy": 0.8}}))
    (root / "g0_evaluations.json").write_text(json.dumps(
        {"old_fold1": {"fold": 1, "accuracy": 0.99}}))
    assert report_tables.baselines(root) == pytest.approx((0.8, 0.99))
    with pytest.raises(SystemExit):
        report_tables.baselines(root, "c20")


# --------------------------------------- the macros the manuscript quotes
#: The macro each cohort's shipped-model accuracy is quoted by, and the book
#: it has to agree with. Written out rather than derived from the macro names
#: so that a macro pointed at the wrong cohort fails here.
COHORT_MACRO = (("nGZeroCohortAcc", "c10"),
                ("nGZeroCohortAccFive", "c5"),
                ("nGZeroCohortAccTwenty", "c20"),
                ("nGZeroCohortAccExtreme", "extreme"))


def test_every_cohort_baseline_the_paper_quotes_is_its_own_book():
    """
    Four table notes now name a baseline, and a note that named the wrong one
    would say the arm beside it is scored against a population it never saw.

    The macros are built from the CSV views - adaptation less gained, the way
    A0 is recovered everywhere - and checked here against the evaluation books
    those views were scored on, which is the only place the two chains meet.
    """
    sys.path.insert(0, str(REPO / "tools" / "paper_figures"))
    import make_numbers

    registry = make_numbers.build()
    for macro, cohort in COHORT_MACRO:
        value, source = registry[macro]
        assert value == "%.4f" % a0_of(cohort), (macro, cohort, value)
        assert "adaptation - gained" in source, (macro, source)
    values = {macro: registry[macro][0] for macro, _ in COHORT_MACRO}
    assert len(set(values.values())) == 4, values


def test_the_twenty_client_macro_is_one_baseline_over_both_dropout_rates():
    """
    Two dropout rates, one cohort, one book. The macro reads both views and
    asserts they agree; if the views ever carried two baselines the assertion
    would fire inside the generator, so what is checked here is the other
    half - that both views really are named in the provenance comment, and so
    that a reader following it lands on both.
    """
    sys.path.insert(0, str(REPO / "tools" / "paper_figures"))
    import make_numbers

    _, source = make_numbers.build()["nGZeroCohortAccTwenty"]
    for view in ("sizes_c20d10.csv", "sizes_c20d20.csv"):
        assert view in source, source
