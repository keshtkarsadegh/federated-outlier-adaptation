"""
The two seeding failures this study has actually had, and the false alarm.

One: a stage that fits a model without a seed, so two of the rungs everything
was compared against were not reproducible. Two: two stages drawing the same
sampler seeds, so their client sampling was identical and their results
correlated with nothing in the output to say so.

And the false alarm that nearly buried the second: a smoke file re-running four
lines of a screen, and a subset emitted to re-run the winners that moved, both
carry the SAME task as the file they came from. Flagging those as collisions
made the check cry wolf on 36 pairs, and a check that cries wolf gets waved
through by hand. A shared seed is only a collision when the two lines produce
different runs.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import check_seeds as cs  # noqa: E402


LINE = ("foa final --results-dir X --parent {parent} --fold 1 "
        "--rounds 25 --sampler-seed {seed} --seed 1")


def _file(tmp_path: Path, name: str, pairs) -> Path:
    """pairs: (seed, parent) tuples."""
    path = tmp_path / name
    path.write_text("# a header\n" + "\n".join(
        LINE.format(seed=s, parent=p) for s, p in pairs) + "\n")
    return path


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def test_each_seed_is_read_with_the_task_it_drives(tmp_path):
    path = _file(tmp_path, "a.txt", [(1, "run_a"), (2, "run_b")])
    assert cs.sampler_seeds(path) == {1: "run_a", 2: "run_b"}


def test_comments_and_blank_lines_draw_nothing(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("# --sampler-seed 999\n\n"
                    + LINE.format(seed=5, parent="run_a") + "\n")
    assert cs.sampler_seeds(path) == {5: "run_a"}


# --------------------------------------------------------------------------- #
# what counts as a collision
# --------------------------------------------------------------------------- #
def test_disjoint_files_do_not_collide():
    spans = {"a": {1: "x", 2: "y"}, "b": {10: "z"}}
    assert cs.collisions(spans) == []


def test_one_seed_driving_two_different_tasks_is_a_collision():
    """
    The real case: a 140-cell screen spans two 1000-wide blocks and the next
    stage's block sits inside it, so two unrelated runs share a sequence.
    """
    screen = {s: f"screen_{s}" for s in range(710001, 710176)}
    combos = {s: f"combo_{s}" for s in range(710001, 710176)}
    found = cs.collisions({"reg_screen.txt": screen, "combos.txt": combos})
    assert len(found) == 1
    assert "175 seed(s) drive DIFFERENT tasks" in found[0]
    assert "710001-710175" in found[0]


def test_the_same_task_in_two_files_is_not_a_collision():
    """
    A smoke file re-runs four lines of a screen; a subset re-runs the winners
    that moved. Both legitimately carry the same task, and the seed belongs to
    the task, not to the file.
    """
    screen = {s: f"run_{s}" for s in range(100, 110)}
    subset = {s: f"run_{s}" for s in (102, 105)}
    assert cs.collisions({"screen.txt": screen, "subset.txt": subset}) == []


def test_a_slot_whose_cell_changed_is_still_flagged():
    """
    A re-selected winner takes over its slot's seed under a NEW name. That is
    the same seed driving a different run, so it is reported - correctly. The
    resolution is to retire the superseded file, not to loosen the check.
    """
    old = {708021: "d01_regfull_concurrent_fisher_lam0p1_fold1"}
    new = {708021: "d01_regfull_concurrent_fisher_lam1_fold1"}
    assert len(cs.collisions({"old.txt": old, "new.txt": new})) == 1


def test_every_pair_is_checked_not_just_neighbours():
    spans = {"a": {1: "x"}, "b": {5: "y"}, "c": {1: "z"}}
    found = cs.collisions(spans)
    assert len(found) == 1 and "a and c" in found[0]


# --------------------------------------------------------------------------- #
# the exit status
# --------------------------------------------------------------------------- #
def test_a_collision_makes_the_run_fail(tmp_path, monkeypatch, capsys):
    """A collision is a failure, not a note: it is invisible downstream."""
    a = _file(tmp_path, "a.txt", [(700001, "run_a1"), (700002, "run_a2")])
    b = _file(tmp_path, "b.txt", [(700002, "run_b2"), (700003, "run_b3")])
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(a), str(b)])
    assert cs.main() == 1
    assert "COLLISION" in capsys.readouterr().out


def test_files_sharing_only_identical_tasks_pass(tmp_path, monkeypatch, capsys):
    a = _file(tmp_path, "a.txt", [(700001, "run_1"), (700002, "run_2")])
    b = _file(tmp_path, "b.txt", [(700002, "run_2")])
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(a), str(b)])
    assert cs.main() == 0
    assert "no two files share one" in capsys.readouterr().out


def test_disjoint_files_pass(tmp_path, monkeypatch, capsys):
    a = _file(tmp_path, "a.txt", [(700001, "run_a1")])
    b = _file(tmp_path, "b.txt", [(710001, "run_b1")])
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(a), str(b)])
    assert cs.main() == 0
