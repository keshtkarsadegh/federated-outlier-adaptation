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

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
JOBS = REPO / "study" / "artifacts" / "Digits_study01" / "jobs"
REPRODUCE = REPO / "docs" / "REPRODUCE.md"
sys.path.insert(0, str(TOOLS))

import check_seeds as cs  # noqa: E402


LINE = ("foa final --results-dir X --parent {parent} --fold 1 "
        "--rounds 25 --sampler-seed {seed} --seed 1")

PRIVATE = ("foa isolated-train --results-dir X --clients-file c.json "
           "--only-client {client} --fold 1 --init global --seed {seed}")


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


# --------------------------------------------------------------------------- #
# the base a reference stage was emitted with
# --------------------------------------------------------------------------- #
def test_a_reference_file_on_its_own_base_draws_no_complaint(tmp_path):
    path = tmp_path / "d01_c5_references.txt"
    path.write_text("\n".join(
        PRIVATE.format(client=c, seed=750000 + i * 200 + 1)
        for i, c in enumerate(("w1", "w2"))) + "\n")
    assert cs.out_of_base(path) == []


def test_a_base_typed_differently_is_reported(tmp_path):
    """
    The failure the bases exist to catch. A five-client file emitted on the
    ten-client base still emits, still runs, and draws the clients the shipped
    records were NOT drawn with - nothing downstream would say so.
    """
    path = tmp_path / "d01_c5_references.txt"
    path.write_text(PRIVATE.format(client="w1", seed=740001) + "\n")
    problems = cs.out_of_base(path)
    assert len(problems) == 1
    assert "740001 is outside the 750000 base" in problems[0]


def test_a_final_line_is_read_on_its_sampler_seed_not_its_fold(tmp_path):
    """`final` seeds the run with the fold, which is 1 to 5 and never a base."""
    path = tmp_path / "s03_refs_c10.txt"
    path.write_text(LINE.format(parent="d01_c10_m9_control_global_fold1",
                                seed=740001) + "\n")
    assert cs.out_of_base(path) == []


def test_a_file_that_is_not_a_reference_stage_is_left_alone(tmp_path):
    path = tmp_path / "s09_agg_screen2.txt"
    path.write_text(LINE.format(parent="run_a", seed=2011) + "\n")
    assert cs.out_of_base(path) == []


def test_a_base_outside_its_range_fails_the_run(tmp_path, monkeypatch, capsys):
    path = tmp_path / "d01_c20_references.txt"
    path.write_text(PRIVATE.format(client="w1", seed=750001) + "\n")
    monkeypatch.setattr(sys, "argv", ["check_seeds.py", str(path)])
    assert cs.main() == 1
    assert "outside the 720000 base" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(cs.REFERENCE_BASES))
def test_every_shipped_reference_file_is_on_its_own_base(name):
    """The bases are claims about files that ship, so they are read off them."""
    path = JOBS / name
    assert path.is_file(), f"{name} is named as a reference stage and does not ship"
    assert cs.out_of_base(path) == []


def test_the_reproduce_section_prints_the_bases_the_tool_holds():
    """
    Section 5 listed 40000 and 50000 for two stages that were emitted on
    740000 and 750000 - a reader who typed what the table printed would have
    got a file that looks right and draws different clients.
    """
    text = REPRODUCE.read_text()
    section = text[text.index("The reference stages sit in bases of their own"):
                   text.index("`c10d10` was given 13000")]
    # The printed table, not the prose around it - which now explains the two
    # block-sized numbers it used to print, and would match a looser search.
    table = "\n".join(l for l in section.splitlines() if l.startswith("   7"))
    assert set(re.findall(r"\b\d{6}\b", table)) == {
        str(base) for base in cs.REFERENCE_BASES.values()}
    for name in cs.REFERENCE_BASES:
        assert name in section, name
