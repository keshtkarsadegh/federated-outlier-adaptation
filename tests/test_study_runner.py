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


# --------------------------------------------------------------------------- #
# the interpreter, which is a runner concern and was not treated as one
# --------------------------------------------------------------------------- #
#
# A whole six-stage chain failed at startup because the job ran under the
# node's /usr/bin/python3 - old enough that `from __future__ import annotations`
# is not a feature it has - and died inside an import with
#
#     SyntaxError: future feature annotations is not defined
#
# which reads as a broken source file and was not one. Nothing before the exec
# looked at the interpreter, and the dry run walked the shell body without ever
# importing the package, so the break passed every check that existed.
#
# These tests pin the two halves of the fix: env.sh resolves and vets an
# interpreter, and the dry run actually loads the task under it.
REAL_ENV_SH = Path(__file__).resolve().parents[1] / "slurm" / "env.sh"
REAL_SBATCH = Path(__file__).resolve().parents[1] / "slurm" / "study_phase.sbatch"


def _workspace(tmp_path, python_target):
    """A workspace whose envs/foa/bin/python is whatever we point it at."""
    env_bin = tmp_path / "ws" / "envs" / "foa" / "bin"
    env_bin.mkdir(parents=True)
    (env_bin / "python").symlink_to(python_target)
    return tmp_path / "ws"


def _source_env(workspace, repo, extra=None):
    script = f'source "{REAL_ENV_SH}"; echo "PY=$FOA_PYTHON"; echo "ENV=$FOA_ENV"'
    environment = dict(os.environ)
    environment.pop("FOA_ENV", None)
    environment.update({"FOA_PROJECT_DIR": str(workspace), "FOA_REPO": str(repo)})
    environment.update(extra or {})
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, env=environment
    )


def test_env_sh_discovers_the_workspace_environment(tmp_path):
    """
    With FOA_ENV unset it must find envs/foa rather than fall through to
    whatever python the node happens to put first on PATH.
    """
    workspace = _workspace(tmp_path, sys.executable)
    repo = Path(__file__).resolve().parents[1]
    result = _source_env(workspace, repo)
    assert result.returncode == 0, result.stderr
    assert f"ENV={workspace}/envs/foa" in result.stdout, result.stdout
    assert f"PY={workspace}/envs/foa/bin/python" in result.stdout, result.stdout


def test_env_sh_refuses_an_interpreter_too_old_to_parse_the_package(tmp_path):
    """
    The failure that started this. It must be refused HERE, by name and version,
    not three hundred lines later inside an import.
    """
    stub = tmp_path / "oldpython"
    stub.write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-V" ]; then echo "Python 3.6.8"; exit 0; fi\n'
        'if [ "$1" = "-c" ]; then exit 1; fi\n'
        "exit 1\n"
    )
    stub.chmod(0o755)
    workspace = _workspace(tmp_path, stub)
    repo = Path(__file__).resolve().parents[1]
    result = _source_env(workspace, repo)
    assert result.returncode == 78, (result.returncode, result.stdout, result.stderr)
    assert "REFUSED" in result.stderr
    assert "3.9+" in result.stderr
    assert "FOA_ENV" in result.stderr, "the message must say how to fix it"


def test_env_sh_reports_the_interpreter_it_chose(tmp_path):
    """
    The original log said nothing about which python it was using, which is why
    a wrong one was invisible. Every run now prints it.
    """
    workspace = _workspace(tmp_path, sys.executable)
    repo = Path(__file__).resolve().parents[1]
    result = _source_env(workspace, repo)
    assert "python=" in result.stdout
    assert "Python 3." in result.stdout


def test_the_dry_run_imports_the_cli_and_parses_the_task(tmp_path):
    """
    A dry run that only walks the shell body passes a chain whose CLI cannot be
    imported at all. It must load the task under the resolved interpreter.
    """
    repo = Path(__file__).resolve().parents[1]
    workspace = _workspace(tmp_path, sys.executable)
    study = tmp_path / "study"
    study.mkdir()
    tasks = tmp_path / "t.txt"
    tasks.write_text(
        "foa final --results-dir $FOA_STUDY_DIR --classes digits "
        "--aggregation fedavg --not-a-real-flag 1\n"
    )
    environment = dict(os.environ)
    environment.pop("FOA_ENV", None)
    environment.update({
        "FOA_PROJECT_DIR": str(workspace), "FOA_REPO": str(repo),
        "FOA_STUDY_DIR": str(study), "FOA_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": "1",
    })
    result = subprocess.run(
        ["bash", str(REAL_SBATCH), str(tasks)],
        capture_output=True, text=True, env=environment,
    )
    assert "STUDY_TASK_DRY_RUN_OK" not in result.stdout, (
        "an unknown flag must not pass the dry run"
    )
    assert result.returncode != 0


def test_the_dry_run_resolves_variables_the_runner_expands(tmp_path):
    """
    A task naming a variable the submitter supplies is not a defect, and must
    not be reported as one - the runner expands it through `eval` at exec time.
    """
    repo = Path(__file__).resolve().parents[1]
    workspace = _workspace(tmp_path, sys.executable)
    study = tmp_path / "study"
    study.mkdir()
    tasks = tmp_path / "t.txt"
    tasks.write_text(
        "foa score-writers --results-dir $FOA_STUDY_DIR --classes digits "
        "--model-path $FOA_STUDY_DIR/m --fold-book $FOA_STUDY_DIR/b.npz "
        "--fold $GINIT_FOLD\n"
    )
    environment = dict(os.environ)
    environment.pop("FOA_ENV", None)
    environment.pop("GINIT_FOLD", None)
    environment.update({
        "FOA_PROJECT_DIR": str(workspace), "FOA_REPO": str(repo),
        "FOA_STUDY_DIR": str(study), "FOA_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": "1",
    })
    result = subprocess.run(
        ["bash", str(REAL_SBATCH), str(tasks)],
        capture_output=True, text=True, env=environment,
    )
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout, (result.stdout, result.stderr)
    assert "GINIT_FOLD" in result.stdout, "an unresolved variable should be named"


# --------------------------------------------------------------------------- #
# Slurm copies the script
# --------------------------------------------------------------------------- #
#
# sbatch does not run the file you hand it. It copies the script into
# /var/spool/slurmd/job<N>/ and runs the copy, so ${BASH_SOURCE[0]} inside it is
# the spool path and its dirname holds no env.sh. A runner that located env.sh
# from its own path therefore worked in every dry run in this repository - which
# invokes it in place - and failed under sbatch, which is the only case that
# matters. Eleven of eleven re-run elements exited 127 with
# "-u: command not found": sourcing env.sh had failed silently and $FOA_PYTHON
# was empty.
#
# These tests run the script the way Slurm does: from a copy, somewhere else.
def _spool_copy(tmp_path):
    """The runner, copied out of the checkout the way sbatch copies it."""
    import shutil

    spool = tmp_path / "spool" / "job12345"
    spool.mkdir(parents=True)
    target = spool / "slurm_script"
    shutil.copy(REAL_SBATCH, target)
    return target


def test_the_runner_finds_env_sh_when_run_from_a_spool_copy(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workspace = _workspace(tmp_path, sys.executable)
    study = tmp_path / "study"
    study.mkdir()
    tasks = tmp_path / "t.txt"
    # --trainer is required by `foa final`; the dry run parses the line, so a
    # task that would not parse tests the wrong thing here.
    tasks.write_text(
        "foa final --results-dir $FOA_STUDY_DIR --classes digits "
        "--trainer BaseTrainer --aggregation fedavg\n"
    )
    environment = dict(os.environ)
    environment.pop("FOA_ENV", None)
    environment.pop("FOA_SLURM_DIR", None)
    environment.update({
        "FOA_PROJECT_DIR": str(workspace), "FOA_REPO": str(repo),
        "FOA_STUDY_DIR": str(study), "FOA_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": "1",
    })
    result = subprocess.run(
        ["bash", str(_spool_copy(tmp_path)), str(tasks)],
        capture_output=True, text=True, env=environment,
    )
    assert "No such file or directory" not in result.stderr, result.stderr
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout, (result.stdout, result.stderr)
    # And the interpreter really got resolved, which is what was empty before.
    assert "python=" in result.stdout


def test_the_runner_refuses_when_env_sh_cannot_be_found_at_all(tmp_path):
    """
    Silently continuing is what turned a missing env.sh into `-u: command not
    found` eleven times. An unsourced env.sh means every later line runs under
    whatever the node happened to provide.
    """
    workspace = _workspace(tmp_path, sys.executable)
    study = tmp_path / "study"
    study.mkdir()
    tasks = tmp_path / "t.txt"
    tasks.write_text("foa final --results-dir $FOA_STUDY_DIR --classes digits "
                     "--trainer BaseTrainer\n")
    environment = dict(os.environ)
    for name in ("FOA_ENV", "FOA_SLURM_DIR", "FOA_REPO"):
        environment.pop(name, None)
    environment.update({
        "FOA_PROJECT_DIR": str(tmp_path / "nowhere"),
        "FOA_STUDY_DIR": str(study), "FOA_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": "1",
    })
    result = subprocess.run(
        ["bash", str(_spool_copy(tmp_path)), str(tasks)],
        capture_output=True, text=True, env=environment,
    )
    assert result.returncode == 78, (result.returncode, result.stdout, result.stderr)
    assert "no env.sh" in result.stderr
    assert "spool" in result.stderr, "the message should name the actual cause"


def test_an_explicit_slurm_dir_still_wins(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    workspace = _workspace(tmp_path, sys.executable)
    study = tmp_path / "study"
    study.mkdir()
    tasks = tmp_path / "t.txt"
    tasks.write_text("foa final --results-dir $FOA_STUDY_DIR --classes digits "
                     "--trainer BaseTrainer\n")
    environment = dict(os.environ)
    environment.pop("FOA_ENV", None)
    environment.update({
        "FOA_PROJECT_DIR": str(workspace), "FOA_SLURM_DIR": str(repo / "slurm"),
        "FOA_STUDY_DIR": str(study), "FOA_DRY_RUN": "1",
        "SLURM_ARRAY_TASK_ID": "1",
    })
    environment.pop("FOA_REPO", None)
    result = subprocess.run(
        ["bash", str(_spool_copy(tmp_path)), str(tasks)],
        capture_output=True, text=True, env=environment,
    )
    assert "STUDY_TASK_DRY_RUN_OK" in result.stdout, result.stderr


def test_the_submitter_carries_what_locates_the_checkout():
    """
    A job that cannot find env.sh cannot run. FOA_SLURM_DIR and FOA_REPO are how
    it is found, so they travel with every submission by value.
    """
    from federated_outlier_adaptation.submission import CARRY

    assert "FOA_SLURM_DIR" in CARRY
    assert "FOA_REPO" in CARRY
