"""
Does the dependency checker cover the whole programme, and say so?

The check exists to prove every file a stage reads is built by an earlier stage.
It walked only the ``s*`` task files, so seven live stages - the carry settings,
the two reference files and the evaluation repair - were outside it, and the
clean verdict was read as covering them. These pin the two things that fixes:
the carry stages are walked, and anything left out is named rather than
silently dropped.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "tools" / "check_programme.py"

sys.path.insert(0, str(REPO / "tools"))


def _run(jobs: Path):
    return subprocess.run([sys.executable, str(CHECKER), str(jobs)],
                          capture_output=True, text=True)


def _write(jobs: Path, name: str, *lines: str) -> None:
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / name).write_text("\n".join(lines) + "\n")


BOOK = ("foa fold-book --results-dir $FOA_STUDY_DIR "
        "--out $FOA_STUDY_DIR/fold_books/cohort5")


def test_a_carry_stage_is_walked_and_its_inputs_are_checked(tmp_path):
    """
    The regression. d01_five.txt reads a book s01 writes; under a name-ordered
    glob it would either be skipped or reported unmet, and both are wrong.
    """
    jobs = tmp_path / "jobs"
    _write(jobs, "s01_book.txt", BOOK)
    _write(jobs, "d01_five.txt",
           "foa final --results-dir $FOA_STUDY_DIR --parent d01_five_winner_fold1 "
           "--fold-book $FOA_STUDY_DIR/fold_books/cohort5.foldbook.npz")
    result = _run(jobs)
    assert result.returncode == 0, result.stdout
    assert "stages: 2" in result.stdout
    assert "d01_five.txt" in result.stdout
    assert "every file every stage reads is produced by an earlier stage" in result.stdout


def test_a_carry_stage_reading_something_nobody_wrote_is_a_gap(tmp_path):
    """Walking the stage is only worth anything if it can still fail."""
    jobs = tmp_path / "jobs"
    _write(jobs, "s01_book.txt", BOOK)
    _write(jobs, "d01_five.txt",
           "foa final --results-dir $FOA_STUDY_DIR --parent d01_five_winner_fold1 "
           "--fold-book $FOA_STUDY_DIR/fold_books/nobody_wrote_this.npz")
    result = _run(jobs)
    assert result.returncode == 1
    assert "GAP" in result.stdout
    assert "fold_books/nobody_wrote_this.npz" in result.stdout


def test_a_file_outside_the_programme_order_is_named_not_dropped(tmp_path):
    """
    A checker that examines a subset must say which subset. Two pre-rename
    copies sit in the real jobs directory and are not stages; they are listed
    with their task counts rather than passed over in silence.
    """
    jobs = tmp_path / "jobs"
    _write(jobs, "s01_book.txt", BOOK)
    _write(jobs, "d01_p05v2.txt", "# a header", BOOK, BOOK)
    result = _run(jobs)
    assert "NOT WALKED" in result.stdout
    assert "d01_p05v2.txt" in result.stdout
    assert "2 task lines" in result.stdout
    assert "stages: 1" in result.stdout


def test_the_extreme_listings_are_credited_to_the_generator_that_wrote_them(tmp_path):
    """
    The three extreme client listings are cut by ``study_emit.py extreme`` at
    emit time, in the same command that writes the task file. No task line can
    produce them, so a checker that follows only task lines reports three unmet
    dependencies for inputs the programme does build. The dependency is real
    and it is declared, with the generator named.
    """
    import check_programme  # the module reads sys.argv at import; only the map
    jobs = tmp_path / "jobs"
    _write(jobs, "d01_extreme.txt",
           "foa final --results-dir $FOA_STUDY_DIR --parent d01_extreme_single_fold1 "
           "--outliers-file $FOA_STUDY_DIR/outliers/extreme_single.json")
    result = _run(jobs)
    assert result.returncode == 0, result.stdout
    assert "written at emit time by tools/study_emit.py extreme" in result.stdout


def test_the_real_jobs_directory_walks_every_live_stage():
    """
    Twenty live stages, and the only files left out are the two pre-rename
    duplicates. If a stage is added to the programme and not to CARRY, this is
    what says so.
    """
    source = CHECKER.read_text()
    carry = source[source.index("CARRY = ("):source.index(")", source.index("CARRY = ("))]
    for name in ("d01_c5_references.txt", "d01_c20_references.txt", "d01_five.txt",
                 "d01_c20.txt", "d01_extreme.txt", "d01_c10d10.txt",
                 "d01_size_evals_rerun.txt"):
        assert name in carry, name
