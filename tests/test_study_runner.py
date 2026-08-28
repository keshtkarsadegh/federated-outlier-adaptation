"""
The runner, executed rather than parsed.

``bash -n`` proves a script parses. It does not prove the checks inside it fire
in an order where the variables they read exist - and ``slurm/env.sh`` runs the
body under ``set -euo pipefail``, so reading one variable one line too early is
not a wrong answer, it is an immediate abort.

That is not hypothetical: a ``case "$TASK"`` guard placed above the line that
assigns ``TASK`` killed 585 array elements at startup, each in about 25 seconds,
and ``bash -n`` had passed. These tests run the body.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = str(Path(__file__).resolve().parents[1] / "tools")

FISHER_LINE = (
    "foa final --results-dir $FOA_STUDY_DIR --classes digits "
    "--set space=fisher anchor=frozen lam=0.1 "
    "fisher_path=$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher"
)
PLAIN_LINE = (
    "foa final --results-dir $FOA_STUDY_DIR --classes digits --aggregation fedavg"
)
FOREIGN_LINE = "foa final --results-dir /somewhere/else --classes digits"


@pytest.fixture()
def runner(tmp_path):
    """
    The runner, with a stub ``env.sh`` that reproduces the one thing that
    mattered: the body runs under ``set -euo pipefail``.
    """
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_study_sbatch

    script = tmp_path / "study_phase.sbatch"
    script.write_text(make_study_sbatch.sbatch_text())

    slurm = tmp_path / "slurm"
    slurm.mkdir()
    (slurm / "env.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\nexport FOA_RESULTS_DIR=%s\n" % tmp_path
    )

    tasks = tmp_path / "tasks.txt"
    tasks.write_text(
        "# a comment, and a blank line follow\n\n"
        + "\n".join([FISHER_LINE, PLAIN_LINE, FOREIGN_LINE])
        + "\n"
    )

    def run(index: int, study: Path, **env):
        environ = {
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "FOA_PROJECT_DIR": str(tmp_path),
            "FOA_SLURM_DIR": str(slurm),
            "FOA_STUDY_DIR": str(study),
            "FOA_DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": str(index),
        }
        environ.update({k: str(v) for k, v in env.items()})
        return subprocess.run(
            ["bash", str(script), str(tasks)],
            env=environ, capture_output=True, text=True,
        )

    return run


@pytest.fixture()
def study(tmp_path):
    """A study that has chosen its g-0 fold."""
    root = tmp_path / "study"
    (root / "logs").mkdir(parents=True)
    (root / "g0_selection.json").write_text(json.dumps({"selected_fold": 2}))
    return root


@pytest.fixture()
def unselected(tmp_path):
    """A study with no selection record yet."""
    root = tmp_path / "unselected"
    (root / "logs").mkdir(parents=True)
    return root


# --------------------------------------------------------------------------- #
# the body runs
# --------------------------------------------------------------------------- #
def test_a_fisher_line_reaches_the_exec_stage(runner, study):
    """
    The regression. This line names $G0_FOLD, so it exercises the check whose
    misplacement aborted the screen.
    """
    result = runner(1, study)
    assert result.returncode == 0, result.stderr
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout
    assert "G0_FOLD=2" in result.stdout
    assert "fisher_path=" in result.stdout
    # and it is the CLI module that would be invoked, not the `foa` shim
    assert "python -u -m federated_outlier_adaptation.cli final" in result.stdout


def test_a_plain_line_reaches_the_exec_stage(runner, study):
    result = runner(2, study)
    assert result.returncode == 0, result.stderr
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout
    assert "--aggregation fedavg" in result.stdout


def test_no_unbound_variable_survives_set_u(runner, study):
    """The failure mode itself: a variable read before it is assigned."""
    for index in (1, 2):
        result = runner(index, study)
        assert "unbound variable" not in result.stderr, result.stderr
        assert result.returncode == 0


# --------------------------------------------------------------------------- #
# the refusals still refuse
# --------------------------------------------------------------------------- #
def test_a_path_outside_the_study_is_refused(runner, study):
    result = runner(3, study)
    assert result.returncode == 78
    assert "REFUSED" in result.stderr and "/somewhere/else" in result.stderr


def test_a_fisher_line_without_a_selection_record_is_refused(runner, unselected):
    """An empty G0_FOLD would expand to a path with a hole in it."""
    result = runner(1, unselected)
    assert result.returncode == 78
    assert "G0_FOLD" in result.stderr


def test_a_plain_line_without_a_selection_record_still_runs(runner, unselected):
    """Only the tasks that need the fold may be blocked by its absence."""
    result = runner(2, unselected)
    assert result.returncode == 0, result.stderr
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout


def test_a_missing_study_root_is_refused(runner, tmp_path):
    result = runner(1, tmp_path / "absent")
    assert result.returncode == 78
    assert "No study at" in result.stderr


def test_an_index_past_the_end_is_an_error(runner, study):
    result = runner(99, study)
    assert result.returncode == 1
    assert "No task at index 99" in result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# the fold, derived and overridable
# --------------------------------------------------------------------------- #
def test_an_explicit_fold_overrides_the_record(runner, study):
    result = runner(1, study, G0_FOLD=4)
    assert result.returncode == 0, result.stderr
    assert "G0_FOLD=4" in result.stdout


def test_the_fold_is_derived_for_tasks_that_do_not_need_it(runner, study):
    """Deriving early is safe; asking whether it is NEEDED is what must wait."""
    result = runner(2, study)
    assert "G0_FOLD=2" in result.stdout


def test_the_header_says_where_the_logs_go(runner, study):
    """
    Slurm does not expand shell variables in ``#SBATCH --output``.

    Without ``--output`` at submit time its per-element log lands in the
    submitting directory, and a crash before the script's own logging leaves no
    trace in the study at all - which is where the crash logs for the failed
    screen were found.
    """
    import make_study_sbatch

    text = make_study_sbatch.sbatch_text()
    assert "does not expand shell variables" in text
    assert '--output="$FOA_STUDY_DIR/logs/%x_%A_%a.log"' in text
    assert "FOA_DRY_RUN=1" in text
