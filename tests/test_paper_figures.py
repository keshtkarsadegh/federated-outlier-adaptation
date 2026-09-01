"""
The figure pipeline: every view a figure reads is shipped, and shipped correctly.

The figures are the one part of the manuscript that used to be built by hand.
Now they are `tools/export_*.py` into `tables/paper_figures/` and `fig_*.py` out
of it, and each of the three joints between those pieces fails silently: a view
the exporter stopped writing still leaves the old CSV in the tree, a figure that
starts reading a new view finds nothing until somebody runs it, and a trace whose
arms disagree at round 0 draws a difference that is about the shipped model
rather than about the training.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
FIGURES = TOOLS / "paper_figures"
VIEWS = REPO / "study" / "artifacts" / "Digits_study01" / "tables" / "paper_figures"

for entry in (str(TOOLS), str(FIGURES), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def _rows(name: str) -> list:
    with open(VIEWS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def test_every_view_the_exporters_name_is_shipped():
    """A view outside the tree is a figure nobody can redraw from the clone."""
    import export_baseline_views
    import export_combo_folds  # noqa: F401  - imported for its side-free presence
    import export_traces

    expected = set(export_traces.VIEWS) | set(export_baseline_views.VIEWS)
    expected.add("combos_folds")
    shipped = {path.stem for path in VIEWS.glob("*.csv")}
    assert expected <= shipped, f"named by an exporter, not in the tree: {sorted(expected - shipped)}"
    assert shipped <= expected, f"in the tree, written by no exporter: {sorted(shipped - expected)}"


def test_every_csv_a_figure_reads_resolves():
    """
    Resolved through `figstyle` itself rather than by looking in one directory.

    The views the figures need are split across two bundles - one written by
    `report_tables.py`, one by the exporters - and a test that knew only about
    the second would pass while `fig_problem` could not find `references.csv`.
    """
    import re

    import figstyle

    wanted = set()
    for script in sorted(FIGURES.glob("fig_*.py")):
        wanted |= set(re.findall(r'"([a-z0-9_]+\.csv)"', script.read_text()))
    assert wanted, "no figure reads any view; the pattern has stopped matching"
    for name in sorted(wanted):
        figstyle.locate(name)          # raises SystemExit when it is not there


def test_round_zero_is_one_shared_point_in_every_trace():
    """
    Every arm of a figure starts at the shipped model, and the rounds have no gap.

    `figstyle.read_traces` refuses a view that breaks either rule, so a stale one
    fails at figure time with a build half-run. It is cheaper to fail here.
    """
    for path in sorted(VIEWS.glob("traces_*.csv")):
        table = _rows(path.name)
        rounds = [int(row["round"]) for row in table]
        assert rounds == list(range(len(table))), f"{path.name}: rounds skip"
        arms = [column[:-4] for column in table[0] if column.endswith("_src")]
        assert arms, f"{path.name}: no arm column"
        starts = {(table[0][f"{arm}_src"], table[0][f"{arm}_cohort"]) for arm in arms}
        assert len(starts) == 1, f"{path.name}: round 0 is not one shared point"
        for arm in arms:
            assert f"{arm}_cohort" in table[0], f"{path.name}: {arm} has no cohort column"


def test_the_shipped_views_carry_the_columns_their_tools_declare():
    """A renamed column reaches a figure as a missing key, one build later."""
    import export_baseline_views
    import export_combo_folds

    for name, columns in (
            ("g0_training.csv", export_baseline_views.G0_COLUMNS),
            ("isolated_clients.csv", export_baseline_views.ISOLATED_COLUMNS),
            ("combos_folds.csv", export_combo_folds.COLUMNS)):
        with open(VIEWS / name, newline="") as handle:
            header = next(csv.reader(handle))
        assert header == list(columns), f"{name}: header is not what its tool writes"
