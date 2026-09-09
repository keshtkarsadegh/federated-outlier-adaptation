"""
Which arms reach a shipped view, and which are kept out of one.

Two stages arrived after the views were last cut, and they pull in opposite
directions. The blend's finals (`s22_blend_full.txt`) are **core** regularisation
arms: they run the same penalty family under the same rule at the same horizon,
they are selected by reg-full's own rule, and a reg-full view that did not carry
them would be reporting a stage that no longer exists. The joint-tuning stages
(`s23`, `s24`) are an **extension**: nothing the manuscript reports may read
them, and a core view that quietly grew two `ctune_` rows would put an arm the
programme never selected into a table the paper quotes.

Both failures are silent. A membership predicate is a prefix or a cell list,
never an announcement, so a stage that walks into the wrong glob does it without
a message and a stage that falls out of the right one leaves a table that still
renders. This file asserts the two directions against the shipped views
themselves, and against the one predicate that had to be told the difference.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = str(REPO / "tools")
STUDY = REPO / "study" / "artifacts" / "Digits_study01"
TABLES = STUDY / "tables"

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

#: The two arms `s22_blend_full.txt` ran, one per schedule, as
#: `tables/p21_blend_winners.json` names them.
BLEND_RECORD = TABLES / "p21_blend_winners.json"

#: The stem every folder of the extension carries, and therefore the string no
#: core view may contain.
EXTENSION_STEM = "ctune"

#: Every shipped view that reports the programme. The two extension views are
#: named beside them rather than globbed away, so a third one cannot be added
#: later and quietly excuse itself from the check below.
EXTENSION_VIEWS = ("extension_combo_tune.csv", "extension_combo_screen.csv")


def _rows(path: Path) -> list:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def core_views() -> list:
    """Every shipped CSV a core table or figure is read from."""
    found = sorted(TABLES.glob("*.csv"))
    for bundle in ("paper", "paper_figures", "stopping"):
        found += [path for path in sorted((TABLES / bundle).glob("*.csv"))
                  if path.name not in EXTENSION_VIEWS]
    return found


def blend_winners() -> dict:
    """``{schedule: cell}`` for the two arms the blend's screen crowned."""
    record = json.loads(BLEND_RECORD.read_text())
    return {key.split("/")[-1]: body["winner"]
            for key, body in record.items() if body.get("measured")}


# ------------------------------------------------- the blend arms are core
def test_the_blend_finals_are_rows_of_the_regularisation_winners():
    """
    They are reg-full arms and they sit in reg-full's table, one per schedule.

    `report_tables.py` reads `d01_regfull_*` by prefix, so nothing had to be
    told about these two - which is exactly why it is worth pinning. A prefix
    that stops matching does not raise; the table simply comes back two rows
    shorter than the stage that ran.
    """
    rows = {(row["cell"], row["family"]) for row in _rows(TABLES / "paper" / "reg-winners.csv")}
    winners = blend_winners()
    assert ("blend" in winners["concurrent"]) and ("blend" in winners["sequential"])
    assert (winners["concurrent"], "parallel") in rows
    assert (winners["sequential"], "cyclic") in rows


def test_the_blend_finals_are_arms_of_every_stopping_view():
    """
    The stopping bundle prices the fixed horizon arm by arm, so an arm missing
    from it is an arm nobody priced. `plateau_arms.csv` and `stopping_all.csv`
    are cut from one read of the runs and have to agree on the population.
    """
    winners = blend_winners()
    for name in ("stopping_regfull.csv", "stopping_all.csv", "plateau_arms.csv"):
        cells = {row["cell"] for row in _rows(TABLES / "stopping" / name)}
        assert set(winners.values()) <= cells, name


def test_the_blends_own_screen_is_in_the_regularisation_screen_view():
    """
    `s21_blend_screen.txt` screened 234 cells over the rows its two parents were
    screened over, in both schedules, so the screen view carries 468 rows it did
    not carry before. A screen ranks and does not report - this is the check
    that the ranking is readable at all, not a claim about its numbers.
    """
    rows = [row for row in _rows(TABLES / "paper" / "reg-screen.csv")
            if row["cell"].startswith("blend_")]
    assert len(rows) == 468
    assert {row["family"] for row in rows} == {"parallel", "cyclic"}


# ------------------------------------------ the extension arms are not core
def test_no_core_view_carries_an_extension_arm():
    """
    The extension's whole contract is that no shipped table reads it. Its run
    folders are `d01_ctune_*` and `d01_ctunefull_*`, which no core prefix
    matches - but `d01_combo_` is one underscore away from doing so, and a
    reader that globbed the stage rather than the prefix would put 216 screened
    cells into the combination table without a word.
    """
    offenders = [path.relative_to(REPO) for path in core_views()
                 if EXTENSION_STEM in path.read_text()]
    assert offenders == [], f"extension arms in a core view: {offenders}"


def test_the_extension_views_are_the_only_place_its_arms_appear():
    """The other half of the check above: they do ship, in files of their own."""
    for name in EXTENSION_VIEWS:
        assert EXTENSION_STEM in (TABLES / "paper" / name).read_text(), name


# ------------------------------------------- the record that had to be told
def test_the_study_record_counts_a_containing_stage_without_its_lodgers(tmp_path):
    """
    The one predicate that could not be left alone.

    `study_record.py` checks a stage's task count against the folders it wrote,
    and the blend writes under stems two earlier stages already own -
    `d01_reg_blend_*` inside `d01_reg_*`, `d01_regfull_<schedule>_blend_*`
    inside `d01_regfull_<schedule>_*`. Counted by prefix alone, the
    regularisation screen would be credited with the blend screen's 1,170
    folders and the count would stop being a check on anything.
    """
    import study_record

    for name in ("d01_reg_kd_T2_a0p99_fold1", "d01_reg_kd_T2_a0p99_fold2",
                 "d01_reg_blend_lam0p1_T0p5_mix0p25_fold1",
                 "d01_regfull_concurrent_kd_T2_a0p99_fold1",
                 "d01_regfull_concurrent_blend_lam0p1_T0p5_mix0p25_fold1"):
        (tmp_path / name).mkdir()

    assert study_record.counted(tmp_path, ("d01_reg_", "-d01_reg_blend_")) == 2
    assert study_record.counted(tmp_path, ("d01_reg_blend_",)) == 1
    assert study_record.counted(
        tmp_path, ("d01_regfull_concurrent_", "-d01_regfull_concurrent_blend_")) == 1


def test_every_stage_of_the_blend_and_the_extension_has_a_stem_of_its_own():
    """
    Four stages ran after the record was last generated, and a stage with no
    entry here reports a blank folder count - which reads as "this stage
    produced nothing", the opposite of what it means.
    """
    import study_record

    for stage in ("s21_blend_screen", "s22_blend_full",
                  "s23_combo_screen", "s24_combo_full"):
        assert stage in study_record.PREFIXES, stage
        assert (STUDY / "jobs" / f"{stage}.txt").is_file(), stage


# -------------------------------------------------- the extension's own views
def test_the_extension_views_carry_the_columns_their_tool_declares():
    """A renamed column reaches a reader as a missing key, one build later."""
    import export_extension_views as extension

    for name, columns in (("extension_combo_tune.csv", extension.COLUMNS),
                          ("extension_combo_screen.csv", extension.SCREEN_COLUMNS)):
        with open(TABLES / "paper" / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), f"{name}: header is not what its tool writes"


def test_the_horizon_view_is_five_arms_per_schedule():
    """
    A tuned pair is only worth reporting against what it is supposed to beat, so
    the row that would be quietly dropped is a baseline rather than the winner.
    """
    import export_extension_views as extension

    rows = _rows(TABLES / "paper" / "extension_combo_tune.csv")
    for family in ("concurrent", "sequential"):
        here = [row["role"] for row in rows if row["family"] == family]
        assert here == list(extension.ROLES), family
    assert all(int(row["folds"]) == 5 for row in rows)


def test_the_tuned_row_and_the_screens_first_rank_are_the_crowned_cell():
    """
    Both views are ranked by the rule that wrote
    `tables/p23_combo_tune_winners.json`, so the cell it names has to be the
    tuned row of one and rank 1 of the other. If it is not, one of the two is
    reporting a selection the record never made.
    """
    record = json.loads((TABLES / "p23_combo_tune_winners.json").read_text())
    crowned = {family: record[f"combo-tune/{family}"]["winner"]
               for family in ("concurrent", "sequential")}

    tuned = {row["family"]: row["arm"]
             for row in _rows(TABLES / "paper" / "extension_combo_tune.csv")
             if row["role"] == "tuned pair"}
    assert tuned == crowned

    top = {row["family"]: row["cell"]
           for row in _rows(TABLES / "paper" / "extension_combo_screen.csv")
           if row["rank"] == "1"}
    assert top == crowned


def test_the_screen_summary_reports_the_dials_and_where_they_sit():
    """
    Every winner of this grid sits at an edge in every dial but one, which is
    the finding to read first - so the boundary column is part of the view
    rather than a line of prose beside it.
    """
    rows = _rows(TABLES / "paper" / "extension_combo_screen.csv")
    assert len(rows) == 10
    for row in rows:
        assert row["c_ewc"] and row["c_kd"] and row["T"] and row["m"]
        # only the parallel cells have a server step to move
        assert bool(row["server_eta"]) == (row["family"] == "concurrent")
    crowned = [row for row in rows if row["rank"] == "1"]
    assert all(row["dials_at_edge"] for row in crowned)
