"""
What a cohort holds, and why the verdict is measured on the label grid.

The study's leading penalty is motivated by clients that are missing classes.
These cohorts are not, and the claim that they are not has to be measured.
"""
import json, sys
from pathlib import Path
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import describe_cohort as dc  # noqa: E402


def _study(tmp_path, per_writer_labels):
    """A fold book and a label array that agree, as the real ones must."""
    writers = list(per_writer_labels)
    books = tmp_path / "fold_books"; books.mkdir(parents=True, exist_ok=True)
    labels, index = [], []
    for i, w in enumerate(writers):
        for lab in per_writer_labels[w]:
            labels.append(lab); index.append(i)
    np.savez(books / "cohortX.foldbook.npz",
             writer_index=np.array(index, dtype=np.int32),
             assignment=np.zeros((5, len(index)), dtype=np.int8),
             metadata=np.array(json.dumps(
                 {"writers": writers, "writer_table": writers}), dtype=object))
    data = tmp_path / "data"; data.mkdir(exist_ok=True)
    np.save(data / "nist28_labels.npy", np.array(labels, dtype=np.uint8))
    return data


def test_a_writer_holding_every_digit_reports_full_coverage(tmp_path):
    data = _study(tmp_path, {"w1": list(range(10)) * 3})
    rows = dc.describe(tmp_path, data, "cohortX")
    assert rows[0]["classes_present"] == 10
    assert rows[0]["missing"] == []
    assert rows[0]["digit_rows"] == 30


def test_a_missing_class_is_named(tmp_path):
    data = _study(tmp_path, {"w1": [0, 1, 2, 3, 4, 5, 6, 7, 8]})   # no 9
    rows = dc.describe(tmp_path, data, "cohortX")
    assert rows[0]["missing"] == [9]
    assert rows[0]["classes_present"] == 9


def test_letters_are_not_counted_as_digit_rows(tmp_path):
    """The by-class dataset holds 62 classes; this study uses the first ten."""
    data = _study(tmp_path, {"w1": [0, 1, 2, 40, 41, 55]})
    rows = dc.describe(tmp_path, data, "cohortX")
    assert rows[0]["rows"] == 6
    assert rows[0]["digit_rows"] == 3


def test_a_book_that_does_not_match_the_labels_is_fatal(tmp_path):
    data = _study(tmp_path, {"w1": [0, 1, 2]})
    np.save(data / "nist28_labels.npy", np.zeros(99, dtype=np.uint8))
    with pytest.raises(SystemExit, match="different datasets"):
        dc.describe(tmp_path, data, "cohortX")


def test_a_missing_book_is_fatal_rather_than_rederived(tmp_path):
    with pytest.raises(SystemExit, match="cohort is defined by the book"):
        dc.load_book(tmp_path, "nope")


def test_the_verdict_counts_empty_cells_not_short_writers(tmp_path, capsys):
    """
    Twenty writers, two of them missing one class each, is 1% of the grid.

    Counting writers would call that label skew; counting the grid does not,
    and the grid is what a class-motivated penalty responds to.
    """
    rows = [{"cohort": "c", "writer": f"w{i}", "rows": 100, "digit_rows": 100,
             "classes_present": 10, "min_per_class": 5, "max_per_class": 15,
             "missing": []} for i in range(18)]
    rows += [{"cohort": "c", "writer": f"w{i}", "rows": 100, "digit_rows": 90,
              "classes_present": 9, "min_per_class": 0, "max_per_class": 15,
              "missing": [3]} for i in range(18, 20)]
    dc.show(rows, "c")
    out = capsys.readouterr().out
    assert "2 of 200" in out
    assert "NEAR-FULL LABEL COVERAGE" in out
    assert "LABEL SKEW PRESENT" not in out


def test_genuine_skew_is_still_called_skew(tmp_path, capsys):
    rows = [{"cohort": "c", "writer": f"w{i}", "rows": 100, "digit_rows": 40,
             "classes_present": 4, "min_per_class": 0, "max_per_class": 15,
             "missing": [4, 5, 6, 7, 8, 9]} for i in range(10)]
    dc.show(rows, "c")
    assert "LABEL SKEW PRESENT" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Where the labels are looked for
# --------------------------------------------------------------------------- #
def test_an_unset_dataset_variable_names_what_was_tried(tmp_path, monkeypatch, capsys):
    """
    ``Path("")`` is ``Path(".")``, which is truthy and is a directory, so an
    unset FOA_NIST28_DIR used to resolve to the working directory, put the
    fallback out of reach, and fail with a missing-label message naming a bare
    relative path. That reads as a broken dataset rather than an unset
    variable, and it cost a run of this tool to work out which.
    """
    monkeypatch.delenv("FOA_NIST28_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "results" / "studies" / "S"
    (root / "fold_books").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv",
                        ["describe_cohort.py", "--root", str(root), "--cohort", "cohortX"])
    with pytest.raises(SystemExit) as caught:
        dc.main()
    message = str(caught.value)
    assert "no dataset directory" in message
    assert "FOA_NIST28_DIR" in message


def test_the_environment_variable_is_used_when_it_names_a_real_directory(tmp_path, monkeypatch):
    """The documented way in - REPRODUCE.md exports it - has to be the one taken."""
    data = _study(tmp_path, {"w1": list(range(10))})
    monkeypatch.setenv("FOA_NIST28_DIR", str(data))
    monkeypatch.setattr(sys, "argv",
                        ["describe_cohort.py", "--root", str(tmp_path), "--cohort", "cohortX"])
    assert dc.main() == 0
