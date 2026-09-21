"""
The appendix tables, held to the claims they make about themselves.

Four of them said something the file behind them did not support, and each
failure was silent in the same way: the table still rendered, the numbers were
all real, and the only thing wrong was what they meant.

    tab:fairness       printed an unweighted mean over CLIENTS in a column
                       headed Mean, beside two tables printing an accuracy
                       pooled over ROWS for the same arm under the same name.
                       It promised every carried arm at every setting and
                       printed no five-client block and one arm of three at
                       twenty.
    tab:selection_axis printed a rank taken from the records' stored ordering,
                       which is ordered by adaptation, while every shortlist in
                       the study was cut on the selection score. Its ranks
                       reproduced the adaptation ordering exactly and it showed
                       no validation preservation at all.
    tab:reg_winners    printed a no-penalty control on the cyclic block, where
                       the proximal family's best cell happens to be mu = 0,
                       and none on the parallel block.
    tab:kholdout       printed six rows that expanded to nineteen of the
                       twenty-eight protocols its own caption counted.

None of the four can be caught by re-reading the generator, because the
generator did exactly what it was told. They are caught here, on the emitted
LaTeX and on the views behind it.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
FIGURES = TOOLS / "paper_figures"
STUDY = REPO / "study" / "artifacts" / "Digits_study01"
PAPER = STUDY / "tables" / "paper"

for entry in (str(TOOLS), str(FIGURES), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _rows(path: Path) -> list:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _emitted(tmp_path, monkeypatch):
    """The .tex the table generator writes, into a scratch directory."""
    import make_paper_tables

    monkeypatch.setenv("FOA_PAPER_OUT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["make_paper_tables.py"])
    make_paper_tables.main()
    return tmp_path / "tables"


def _body(path: Path) -> list:
    """The data rows of an emitted tabular, headers and rules dropped."""
    lines = path.read_text().splitlines()
    start = lines.index("\\midrule")
    end = lines.index("\\bottomrule")
    return [ln for ln in lines[start + 1:end]
            if ln.strip() and not ln.startswith("\\midrule")
            and "multicolumn" not in ln]


# ------------------------------------------------------- tab:fairness, mean
def test_the_fairness_mean_is_the_cohort_accuracy_the_score_tables_print():
    """
    One arm, one adaptation, wherever it appears.

    `fairness_*.csv` carries both readings of the same evaluation: `cohort` is
    `final_evaluation.clients.accuracy`, pooled over the cohort's rows, and
    `mean` is the unweighted mean over its clients. The writers hold between
    seventy and two hundred-odd digit rows apiece, so the two disagree - by
    up to a point at twenty clients, which is larger than most margins the
    paper reports. The table quotes the first; this is what says the first is
    the number the reporting tables print.
    """
    import make_paper_tables as tables

    for key, _, _ in tables.FAIRNESS_BLOCKS:
        reported = {(r["cell"], r["family"]): r for r in
                    _rows(PAPER / ("combos.csv" if key == "combo"
                                   else "sizes_%s.csv" % key))}
        for row in _rows(PAPER / ("fairness_%s.csv" % key)):
            arm = (row["cell"], row["family"])
            assert arm in reported, (key, arm)
            assert abs(float(row["cohort"])
                       - float(reported[arm]["adaptation"])) < 5e-7, (
                "%s at the %s setting: the fairness view says %s and the table "
                "that reports it says %s"
                % (arm, key, row["cohort"], reported[arm]["adaptation"]))
            # The reading the table does NOT print is still in the view, and
            # it is a different number - which is the whole reason for this.
            assert row["mean"] != row["cohort"] or row["clients"] == "1"


def test_the_fairness_table_lists_every_carried_arm_at_every_setting(
        tmp_path, monkeypatch):
    """
    Its introducing sentence promises that; the table has to keep it.

    The five-client block was absent altogether and the twenty-client blocks
    carried one arm of three - the one whose worst client came out below the
    shipped model. Printing the negative case alone is the version of this
    table a reader has the least reason to trust.
    """
    import make_paper_tables as tables

    out = _emitted(tmp_path, monkeypatch)
    printed = _body(out / "fairness.tex")
    text = (out / "fairness.tex").read_text()

    for key, _, _ in tables.FAIRNESS_BLOCKS:
        if key == "combo":
            continue
        for row in _rows(PAPER / ("sizes_%s.csv" % key)):
            name = tables.ARM[row["cell"]]
            assert any(name in line and row["family"] in line
                       for line in printed), (key, row["cell"], row["family"])

    carried = sum(len(_rows(PAPER / ("sizes_%s.csv" % key)))
                  for key, _, _ in tables.FAIRNESS_BLOCKS if key != "combo")
    assert len(printed) == carried + 4, (len(printed), carried)
    for key, _, _ in tables.FAIRNESS_BLOCKS:
        assert "participating per round" in text


def test_the_two_shipped_model_worst_client_figures_are_told_apart():
    """
    0.72 and 0.76 are the same writers, the same rows and the same model.

    One is the five-fold mean of the worst-served client OF EACH FOLD, which is
    what the fairness table's delta-worst column is a delta from; the other is
    the smallest of the ten clients' own five-fold means. The worst-served
    client is not the same writer on every fold, so a mean of minima sits below
    a minimum of means, and the paper had been quoting one of them as if it
    were the other.
    """
    import make_numbers

    registry = make_numbers.build()
    of_means = float(registry["nGZeroClientMin"][0])
    of_folds = float(registry["nGZeroWorstFoldMeanTen"][0])
    assert of_folds < of_means, (of_folds, of_means)
    for name in ("nGZeroWorstFoldMeanFive", "nGZeroWorstFoldMeanTen",
                 "nGZeroWorstFoldMeanTwenty"):
        assert "worst-served client of each fold" in registry[name][1], name
    assert "fold-mean of g0_own_test" in registry["nGZeroClientMin"][1]


# ------------------------------------------------- tab:selection_axis, ranks
def _blocks():
    """The selection-axis view, grouped by stage and schedule."""
    grouped = {}
    for row in _rows(PAPER / "selection_axis.csv"):
        grouped.setdefault((row["stage"], row["schedule"]), []).append(row)
    assert grouped, "the selection-axis view is empty"
    return grouped


def test_the_selection_ranks_are_computed_on_the_score():
    """
    Both rank columns, recomputed from the view's own numbers.

    `study_emit.ranked_by_trade` sorts on the selection score descending with
    ties broken by cell id, and that is the ordering that cut every shortlist
    in this study. The records store an ordering too, and it is not that one:
    `both_rankings` orders by adaptation. The table printed the stored one, so
    its ranks reproduced the adaptation ordering exactly and no reader could
    tell that a shortlist at ranks 1, 2 and 5 had been cut by anything.
    """
    for (stage, schedule), block in _blocks().items():
        for column, rank_column in (("val_score", "val_score_rank"),
                                    ("test_score", "test_score_rank")):
            order = sorted(block, key=lambda r: (-float(r[column]), r["cell"]))
            for place, row in enumerate(order, start=1):
                assert int(row[rank_column]) == place, (
                    stage, schedule, row["cell"], column,
                    row[rank_column], place)


def test_the_shortlist_is_the_top_three_methods_of_the_score_ordering():
    """
    The rule that makes a shortlist read 1, 2, 5 instead of 1, 2, 3.

    Both screens take one cell per METHOD - a family's three slots are three
    ideas about the problem, not three settings of one - and the composite
    penalty competes for a slot of its own. So the shortlist is the first three
    arms of the validation-SCORE ordering that belong to three different
    methods, and the arms the rule stepped over are further settings of a
    method already taken. Reconstructed here from the view alone, and the
    failure names the ranks the shortlist actually occupies.
    """
    for (stage, schedule), block in _blocks().items():
        marked = [r["cell"] for r in block if r["role"] == "shortlist"]
        if stage == "combination":
            assert not marked, (stage, schedule, marked)
            continue
        assert len(marked) == 3, (stage, schedule, marked)
        order = sorted(block, key=lambda r: int(r["val_score_rank"]))
        seen, derived = [], []
        for row in order:
            assert row["method"], (stage, row["cell"])
            if row["method"] in seen:
                continue
            seen.append(row["method"])
            derived.append(row["cell"])
            if len(derived) == 3:
                break
        ranks = {r["cell"]: int(r["val_score_rank"]) for r in block}
        assert derived == marked, (
            "%s/%s: the arms the stage carried forward are not the first three "
            "distinct methods of the validation-score ordering. Carried %s at "
            "ranks %s; the rule gives %s."
            % (stage, schedule, marked, [ranks[c] for c in marked], derived))


def test_the_crowned_pair_leads_the_validation_score_ordering():
    """The cross cut no shortlist. It crowned, once, and on the score."""
    rows = _rows(PAPER / "selection_axis.csv")
    cross = [r for r in rows if r["stage"] == "combination"]
    crowned = [r for r in cross if r["crowned"] == "yes"]
    assert len(crowned) == 1, crowned
    assert crowned[0]["role"] == "crowned"
    assert int(crowned[0]["val_score_rank"]) == 1, crowned[0]
    assert float(crowned[0]["val_score"]) == max(
        float(r["val_score"]) for r in cross), "crowned on its schedule only"


def test_the_selection_axis_table_prints_every_arm_it_carried(
        tmp_path, monkeypatch):
    """An arm the study carried is never below the depth the table prints."""
    import make_paper_tables as tables
    import paper_names

    out = _emitted(tmp_path, monkeypatch)
    printed = (out / "selection_axis.tex").read_text()
    for row in _rows(PAPER / "selection_axis.csv"):
        if not row["role"]:
            continue
        assert paper_names.label(row["cell"]) in printed, row["cell"]
    assert tables.SELECTION_DEPTH >= 3


# ------------------------------------------------------- tab:reg_winners
def test_both_schedule_blocks_print_a_no_penalty_control(
        tmp_path, monkeypatch):
    """
    Two blocks read against two different baselines is not a comparison.

    The regularisation finals ran one cell per penalty family at that family's
    own best setting. The proximal family's best setting is mu = 0 on the
    cyclic schedule, which IS the no-penalty control and prints as one, and
    mu = 1e-4 on the parallel schedule, which is not - so the parallel block
    had no zero-penalty row at all.
    """
    import make_paper_tables as tables
    import paper_names

    out = _emitted(tmp_path, monkeypatch)
    lines = (out / "reg_winners.tex").read_text().splitlines()
    control = paper_names.label(tables.NO_PENALTY_CONTROL)
    marked = [ln for ln in lines
              if control in ln and tables.NO_PENALTY_MARK in ln]
    assert len(marked) == 2, (
        "one schedule block of tab:reg_winners carries no no-penalty "
        "control: %s" % marked)

    agg = {(r["cell"], r["family"]): r
           for r in _rows(PAPER / "agg-winners.csv")}
    for family in ("parallel", "cyclic"):
        row = agg[(tables.NO_PENALTY_CONTROL, family)]
        assert any("%.2f" % (100 * float(row["score"])) in ln
                   for ln in marked), (family, row["score"])


# --------------------------------------------------------- tab:kholdout
def test_every_holdout_protocol_reaches_the_table(tmp_path, monkeypatch):
    """
    Twenty-eight protocols, and a page a reader can count them off.

    The table printed six rows. Two of them stood for five and ten protocols
    and said so; the eight leave-one-stage-out protocols were not on it at all,
    so the six expanded to nineteen and the caption's twenty-eight could not be
    reached from the page.
    """
    import make_paper_tables as tables

    out = _emitted(tmp_path, monkeypatch)
    text = (out / "kholdout.tex").read_text()
    prot = tables.rows("plateau_holdout_protocols.csv")

    families = {}
    for row in prot:
        families.setdefault(row["protocol"], []).append(row)
    assert set(families) == {name for name, _, _ in tables.HOLDOUT_FAMILIES}

    covered = 0
    for name, how, title in tables.HOLDOUT_FAMILIES:
        here = families[name]
        covered += len(here)
        assert "%s (%d of" % (title, len(here)) in text, title
        if how == "each":
            for row in here:
                assert tables._holdout_name(row["heldout_set"]) in text, row
    assert covered == len(prot) == 28, (covered, len(prot))

    # and the held-out gain of the regularisation finals is the one the prose
    # quotes, which is NOT the in-sample gain of the same stage.
    import make_numbers

    registry = make_numbers.build()
    held = registry["nKholdoutRegfullHeldGain"][0]
    in_sample = tables.pick(tables.rows("plateau_stages.csv"),
                            stage="regfull")["gain"]
    assert held != "%.2f" % (100 * float(in_sample)), (
        "the held-out and in-sample gains of the regularisation finals are "
        "the same number, so one of the two is being read from the wrong file")


# ------------------------------------------------------ the pre-filter view
def test_the_prefilter_view_is_rebuilt_from_the_shipped_records(tmp_path):
    """It reads `outliers/` only, so the clone can rebuild it byte for byte."""
    import export_prefilter_coverage

    export_prefilter_coverage.write_csv(
        tmp_path / "prefilter_coverage.csv",
        export_prefilter_coverage.rows_of(STUDY))
    assert (tmp_path / "prefilter_coverage.csv").read_bytes() == \
        (PAPER / "prefilter_coverage.csv").read_bytes()


def test_the_prefilter_coverage_is_a_hundred_percent_and_says_why():
    """
    The number is true, trivially, and the view has to carry the one that
    is not.

    The cohorts are cut out of the pool, so a writer the pre-filter dropped is
    a writer the shipped model was never asked to score and coverage cannot
    read anything but one. What the records CAN answer is how deep into the
    detector's own ranking the cohort's writers sit, which is the narrowest
    pre-filter that would have kept the same cohort.
    """
    rows = {r["cohort"]: r for r in _rows(PAPER / "prefilter_coverage.csv")}
    assert set(rows) == {"five", "ten", "twenty"}
    for name, row in rows.items():
        assert float(row["coverage"]) == 1.0, name
        assert int(row["kept_by_prefilter"]) == int(row["clients"]) == int(row["k"])
        assert 0 < float(row["depth_fraction"]) < float(row["pool_fraction"]), (
            "%s sits deeper in the detector's ranking than the pre-filter "
            "reached, which cannot happen if the cohort was cut from the pool"
            % name)
        # every cohort member is a writer the detector actually separated,
        # rather than one carried into the pool by the tie-break on writer id
        assert int(row["deepest_prefilter_rank"]) <= int(row["ranked_above_ties"]), name

    import make_numbers

    registry = make_numbers.build()
    assert registry["nPrefilterPoolSize"][0] == "1\\,074"
    assert "100" in registry["nPrefilterCoverageTen"][0]
    assert "by construction" in registry["nPrefilterCoverageTen"][1]


def test_the_selection_axis_view_carries_the_columns_its_tool_declares():
    """A renamed column reaches the table as a missing key, one build later."""
    import export_prefilter_coverage
    import export_selection_axis

    for name, columns in (("selection_axis.csv", export_selection_axis.COLUMNS),
                          ("prefilter_coverage.csv",
                           export_prefilter_coverage.COLUMNS)):
        with open(PAPER / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), name
