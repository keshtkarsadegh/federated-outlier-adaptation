"""
The paper bundle: every view the manuscript quotes, from one command.

``report_tables.py --what all`` is what assembles the bundle. A view that is
reachable by name but missing from ``all`` is a table nobody regenerates,
because nobody runs the eight commands by hand - and the two newest views,
the carry settings and the extremes, are the ones a reader is most likely to
find only in a stale copy.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

SOURCE = (REPO / "tools" / "report_tables.py").read_text()


def _named_views() -> set:
    """Every value of --what except 'all', read from the parser itself."""
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not any(isinstance(k, ast.keyword) and k.arg == "choices" for k in node.keywords):
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                values = {ast.literal_eval(e) for e in keyword.value.elts}
                if "all" in values:
                    return values - {"all"}
    raise AssertionError("no --what choices found")


def _all_tuple() -> set:
    tree = ast.parse(SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "all":
                    return {ast.literal_eval(e) for e in value.elts}
    raise AssertionError("no 'all' mapping found")


def test_every_named_view_is_in_the_bundle():
    """A view outside 'all' is a table the bundle silently omits."""
    missing = _named_views() - _all_tuple()
    assert not missing, f"reachable by name but not emitted by --what all: {sorted(missing)}"


def test_the_two_newest_views_are_named_explicitly():
    """
    Pinned by name as well as by the set comparison above, because these two
    are the ones that were added last and the ones the carry-settings and
    extreme-case sections of the manuscript are built from.
    """
    bundle = _all_tuple()
    assert "sizes" in bundle
    assert "extremes" in bundle


def test_the_bundle_covers_both_grids_at_both_horizons():
    """A screen ranks and a final reports; the bundle has to carry both."""
    bundle = _all_tuple()
    for view in ("agg-screen", "agg-winners", "reg-screen", "reg-winners",
                 "combos", "references"):
        assert view in bundle, view
