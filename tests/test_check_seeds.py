"""
The two seeding failures this study has actually had.

One: a stage that fits a model without a seed, so two of the rungs everything
was compared against were not reproducible. Two: two stages drawing the same
sampler seeds, so their client sampling was identical and their results
correlated with nothing in the output to say so.

Both were found by looking rather than by anything breaking, which is why they
are checked here.
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import check_seeds as cs  # noqa: E402


LINE = ("foa final --results-dir X --parent p_fold{fold} --fold {fold} "
        "--rounds 25 --sampler-seed {seed} --seed 1")


def _file(tmp_path: Path, name: str, seeds) -> Path:
    path = tmp_path / name
    path.write_text("# a header\n" + "\n".join(
        LINE.format(fold=1, seed=s) for s in seeds) + "\n")
    return path


def test_the_sampler_seeds_of_a_file_are_read(tmp_path):
    assert cs.sampler_seeds(_file(tmp_path, "a.txt", [1, 2, 3])) == {1, 2, 3}


def test_comments_and_blank_lines_draw_nothing(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("# --sampler-seed 999\n\n" + LINE.format(fold=1, seed=5) + "\n")
    assert cs.sampler_seeds(path) == {5}


def test_disjoint_files_do_not_collide(tmp_path):
    spans = {"a": {1, 2, 3}, "b": {10, 11}, "c": {20}}
    assert cs.collisions(spans) == []


def test_an_overlap_between_two_stages_is_reported(tmp_path):
    """
    The real case: a 140-cell screen spans two 1000-wide blocks and the next
    stage's block sits inside it.
    """
    screen = set(range(710001, 711396))
    combos = set(range(710001, 710176))
    found = cs.collisions({"reg_screen.txt": screen, "combos.txt": combos})
    assert len(found) == 1
    assert "share 175 sampler seed(s)" in found[0]
    assert "710001-710175" in found[0]


def test_every_pair_is_checked_not_just_neighbours(tmp_path):
    spans = {"a": {1}, "b": {5}, "c": {1}}
    found = cs.collisions(spans)
    assert len(found) == 1 and "a and c" in found[0]


def test_a_collision_makes_the_run_fail(tmp_path, monkeypatch, capsys):
    """A collision is a failure, not a note: it is invisible downstream."""
    a = _file(tmp_path, "a.txt", [700001, 700002])
    b = _file(tmp_path, "b.txt", [700002, 700003])
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(a), str(b)])
    assert cs.main() == 1
    assert "COLLISION" in capsys.readouterr().out


def test_disjoint_files_pass(tmp_path, monkeypatch, capsys):
    a = _file(tmp_path, "a.txt", [700001, 700002])
    b = _file(tmp_path, "b.txt", [710001, 710002])
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(a), str(b)])
    assert cs.main() == 0
    assert "no two files share one" in capsys.readouterr().out
