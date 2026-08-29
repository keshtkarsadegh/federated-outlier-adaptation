"""
The submitter: a manifest, a preflight, and no half-submitted chains.

Every task file in this repository was generated, parse-checked and dry-run
before submission. The submission itself never was - it was assembled at a
terminal, once per chain - and four separate failures came out of that, none of
them a bug in the science. These tests are the ones that were missing.

The rule the whole module turns on: **if the preflight finds anything, no job is
submitted at all.** A half-submitted chain has to be found and cancelled, which
is worse than one that never started.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from federated_outlier_adaptation.submission import (
    CARRY, TYPED_GRES, ManifestError, export_set, guard_violations, load_manifest,
    plan, preflight, render, submit, task_count,
)

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "study" / "jobs"


# --------------------------------------------------------------------------- #
@pytest.fixture()
def workspace(tmp_path):
    """A project root with a runner, a study and one task file."""
    project = tmp_path / "project"
    (project / "repo_foa" / "slurm").mkdir(parents=True)
    (project / "repo_foa" / "slurm" / "study_phase.sbatch").write_text("#!/bin/bash\n")
    (project / "jobs" / "v4").mkdir(parents=True)
    study = tmp_path / "study"
    (study / "logs").mkdir(parents=True)
    (project / "jobs" / "v4" / "a.txt").write_text(
        "# a comment\n\n"
        "foa final --results-dir $FOA_STUDY_DIR --classes digits\n"
        "foa final --results-dir $FOA_STUDY_DIR --classes digits --fold 2\n"
    )
    return {"project": project, "study": study}


@pytest.fixture()
def environment(workspace):
    return {
        "FOA_PROJECT_DIR": str(workspace["project"]),
        "FOA_STUDY_DIR": str(workspace["study"]),
        "FOA_ACCOUNT": "acct",
        "FOA_GPU_PARTITION": "gpupart",
        "FOA_CPU_PARTITION": "cpupart",
    }


def _manifest(tmp_path, body: str) -> Path:
    path = tmp_path / "chain.toml"
    path.write_text(body)
    return path


BASIC = '''
study = "S"
jobs_dir = "jobs/v4"
[defaults]
partition = "$FOA_GPU_PARTITION"
gres = "gpu:A100:1"
[[stage]]
name = "one"
tasks = "a.txt"
count = 2
array = "1-2"
'''

#: The form 43 OOM elements were submitted with, kept for the test about it.
UNTYPED = BASIC.replace('gres = "gpu:A100:1"', 'gres = "gpu:1"')


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_a_manifest_parses_into_stages(tmp_path):
    manifest = load_manifest(_manifest(tmp_path, BASIC))
    assert manifest.study == "S" and manifest.jobs_dir == "jobs/v4"
    assert [s.name for s in manifest.stages] == ["one"]
    stage = manifest.stage("one")
    assert (stage.count, stage.array) == (2, "1-2")
    assert stage.gres == "gpu:A100:1", "defaults reach the stage"
    assert stage.runner == "slurm/study_phase.sbatch"


@pytest.mark.parametrize("body,reason", [
    ('jobs_dir = "j"\n[[stage]]\nname="a"\ntasks="a"\ncount=1\narray="1-1"', "study"),
    ('study = "S"\n[[stage]]\nname="a"\ntasks="a"\ncount=1\narray="1-1"', "jobs_dir"),
    ('study = "S"\njobs_dir = "j"', r"no \[\[stage\]\] entries"),
    ('study="S"\njobs_dir="j"\n[[stage]]\nname="a"\ntasks="a"\ncount=1', "array"),
    ('study="S"\njobs_dir="j"\n[[stage]]\nname="a"\ntasks="a"\ncount=1\narray="1-1"\nnonsense=1', "unknown key"),
])
def test_a_malformed_manifest_is_refused_by_name(tmp_path, body, reason):
    with pytest.raises(ManifestError, match=reason):
        load_manifest(_manifest(tmp_path, body))


def test_two_stages_cannot_share_a_name(tmp_path):
    body = BASIC + '\n[[stage]]\nname="one"\ntasks="a.txt"\ncount=1\narray="1-1"\n'
    with pytest.raises(ManifestError, match="two stages named"):
        load_manifest(_manifest(tmp_path, body))


def test_a_dependency_on_a_stage_that_is_not_here_is_refused(tmp_path):
    body = BASIC + '\n[[stage]]\nname="two"\ntasks="a.txt"\ncount=2\narray="1-2"\nafter="nope"\n'
    with pytest.raises(ManifestError, match="not a stage"):
        load_manifest(_manifest(tmp_path, body))


def test_a_missing_manifest_is_a_named_refusal(tmp_path):
    with pytest.raises(ManifestError, match="no manifest"):
        load_manifest(tmp_path / "absent.toml")


def test_task_count_ignores_comments_and_blanks(workspace):
    assert task_count(workspace["project"] / "jobs" / "v4" / "a.txt") == 2


# --------------------------------------------------------------------------- #
# the preflight - one test per way a submission has actually gone wrong
# --------------------------------------------------------------------------- #
def test_a_clean_chain_has_no_problems(tmp_path, environment):
    assert preflight(load_manifest(_manifest(tmp_path, BASIC)), environment) == []


@pytest.mark.parametrize("missing", ["FOA_PROJECT_DIR", "FOA_STUDY_DIR", "FOA_ACCOUNT"])
def test_a_missing_environment_variable_refuses(tmp_path, environment, missing):
    environment.pop(missing)
    problems = preflight(load_manifest(_manifest(tmp_path, BASIC)), environment)
    assert any(missing in p for p in problems), problems


def test_a_count_that_disagrees_with_the_file_refuses(tmp_path, environment):
    body = BASIC.replace("count = 2", "count = 7")
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("holds 2 runnable lines" in p and "says 7" in p for p in problems), problems


def test_a_missing_task_file_refuses(tmp_path, environment):
    body = BASIC.replace('tasks = "a.txt"', 'tasks = "absent.txt"')
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("no task file" in p for p in problems), problems


def test_a_missing_prerequisite_refuses(tmp_path, environment):
    body = BASIC + 'requires = ["$FOA_STUDY_DIR/not_there.json"]\n'
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("prerequisite missing" in p for p in problems), problems


def test_a_prerequisite_an_earlier_stage_produces_is_satisfied(tmp_path, environment):
    """
    The whole point of `produces`: a chain must be able to declare its real
    external inputs without pretending a file it writes itself already exists.
    """
    body = (BASIC + 'produces = ["$FOA_STUDY_DIR/made.json"]\n'
            + '\n[[stage]]\nname="two"\ntasks="a.txt"\ncount=2\narray="1-2"\n'
            + 'requires = ["$FOA_STUDY_DIR/made.json"]\n')
    assert preflight(load_manifest(_manifest(tmp_path, body)), environment) == []


def test_a_gpu_request_on_the_cpu_partition_refuses(tmp_path, environment):
    body = BASIC.replace('name = "one"',
                         'name = "one"\npartition = "$FOA_CPU_PARTITION"')
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("on the CPU partition" in p for p in problems), problems


def test_a_partition_variable_that_does_not_expand_refuses(tmp_path, environment):
    environment.pop("FOA_GPU_PARTITION")
    body = BASIC.replace('name = "one"',
                         'name = "one"\npartition = "$FOA_GPU_PARTITION"')
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("does not expand" in p for p in problems), problems


@pytest.mark.parametrize("bad,reason", [
    ('array = "1-2"', 'array = "one to two"', ),
    ('count = 2\narray = "1-2"', 'count = 2\narray = "1-2"\ntime = "soon"'),
])
def test_a_bad_override_refuses(tmp_path, environment, bad, reason):
    problems = preflight(load_manifest(_manifest(tmp_path, BASIC.replace(bad, reason))),
                         environment)
    assert problems




def test_a_stage_with_its_own_script_is_not_guarded_that_way(tmp_path, environment,
                                                             workspace):
    """A stage writing outside the study root says so by having its own script."""
    (workspace["project"] / "repo_foa" / "slurm" / "prep.sbatch").write_text("#!/bin/bash\n")
    body = '''
study = "S"
jobs_dir = "jobs/v4"
[[stage]]
name = "prep"
tasks = ""
runner = "slurm/prep.sbatch"
count = 1
array = "1-1"
gres = ""
'''
    assert preflight(load_manifest(_manifest(tmp_path, body)), environment) == []


def test_a_script_stage_must_be_a_single_invocation(tmp_path, environment, workspace):
    (workspace["project"] / "repo_foa" / "slurm" / "prep.sbatch").write_text("#!/bin/bash\n")
    body = '''
study = "S"
jobs_dir = "jobs/v4"
[[stage]]
name = "prep"
tasks = ""
runner = "slurm/prep.sbatch"
count = 5
array = "1-5"
gres = ""
'''
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("single invocation" in p or "count must be 1" in p for p in problems)


def test_a_missing_runner_refuses(tmp_path, environment, workspace):
    (workspace["project"] / "repo_foa" / "slurm" / "study_phase.sbatch").unlink()
    problems = preflight(load_manifest(_manifest(tmp_path, BASIC)), environment)
    assert any("no runner" in p for p in problems), problems


# --------------------------------------------------------------------------- #
# what gets submitted
# --------------------------------------------------------------------------- #
def test_the_export_set_names_every_variable_that_is_set(environment):
    exports = export_set(environment)
    assert exports.startswith("ALL,"), "ALL alone is the default being overridden"
    for name, value in environment.items():
        assert f"{name}={value}" in exports
    # And nothing that is unset - an empty export overrides a derivation.
    assert "FOA_ENV=" not in exports


def test_every_sbatch_line_carries_the_export_set(tmp_path, environment):
    manifest = load_manifest(_manifest(tmp_path, BASIC))
    exports = export_set(environment)
    for step in plan(manifest, environment):
        argv = render(step, {})
        assert f"--export={exports}" in argv
        assert f"--account={environment['FOA_ACCOUNT']}" in argv
        assert "--parsable" in argv


def test_stages_wire_afterok_in_order(tmp_path, environment):
    body = (BASIC + '\n[[stage]]\nname="two"\ntasks="a.txt"\ncount=2\narray="1-2"\n'
            + '\n[[stage]]\nname="three"\ntasks="a.txt"\ncount=2\narray="1-2"\nroot=true\n')
    steps = plan(load_manifest(_manifest(tmp_path, body)), environment)
    assert [s["depends_on"] for s in steps] == [None, "one", None]
    argv = render(steps[1], {"one": "12345"})
    assert "--dependency=afterok:12345" in argv


# --------------------------------------------------------------------------- #
# submitting, and not submitting
# --------------------------------------------------------------------------- #
def test_a_failing_preflight_submits_nothing(tmp_path, environment):
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        raise AssertionError("nothing may be submitted")

    body = BASIC.replace("count = 2", "count = 7")
    result = submit(load_manifest(_manifest(tmp_path, body)), environment,
                    dry_run=False, runner=runner, log=lambda *a, **k: None)
    assert result["submitted"] is False and result["problems"]
    assert calls == []


def test_a_dry_run_submits_nothing_and_prints_the_chain(tmp_path, environment):
    printed = []
    result = submit(load_manifest(_manifest(tmp_path, BASIC)), environment,
                    dry_run=True, log=lambda *a, **k: printed.append(" ".join(map(str, a))))
    assert result["submitted"] is False
    assert any(line.startswith("would: sbatch") for line in printed)


def test_a_real_submission_records_the_job_ids(tmp_path, environment):
    class Result:
        returncode = 0
        stdout = "99001\n"
        stderr = ""

    body = BASIC + '\n[[stage]]\nname="two"\ntasks="a.txt"\ncount=2\narray="1-2"\n'
    result = submit(load_manifest(_manifest(tmp_path, body)), environment,
                    dry_run=False, runner=lambda *a, **k: Result(),
                    log=lambda *a, **k: None)
    assert result["submitted"] is True
    assert [s["job_id"] for s in result["stages"]] == ["99001", "99001"]
    record = json.loads(Path(result["record"]).read_text())
    assert record["study"] == "S" and record["export"].startswith("ALL,")
    assert [s["stage"] for s in record["stages"]] == ["one", "two"]


def test_a_failure_midway_reports_what_is_already_running(tmp_path, environment):
    """
    Slurm can still refuse. When it does, the ones already in have to be named,
    because they are running and somebody has to cancel them.
    """
    class Ok:
        returncode = 0
        stdout = "5001\n"
        stderr = ""

    class Bad:
        returncode = 1
        stdout = ""
        stderr = "sbatch: error: invalid partition"

    answers = [Ok(), Bad()]
    body = BASIC + '\n[[stage]]\nname="two"\ntasks="a.txt"\ncount=2\narray="1-2"\n'
    messages = []
    result = submit(load_manifest(_manifest(tmp_path, body)), environment,
                    dry_run=False, runner=lambda *a, **k: answers.pop(0),
                    log=lambda *a, **k: messages.append(" ".join(map(str, a))))
    assert result["submitted"] is False and result["failed_at"] == "two"
    assert result["partial"] == {"one": "5001"}
    assert any("scancel" in m for m in messages)


# --------------------------------------------------------------------------- #
# the shipped manifests
# --------------------------------------------------------------------------- #








# --------------------------------------------------------------------------- #
# chains whose middle task files are generated
# --------------------------------------------------------------------------- #
#
# A grid search writes its own next stage: the screen runs, a selector reads the
# results and emits the full-horizon file, and only then does that file exist.
# The preflight must check the width it was promised without demanding the file
# be there yet.
GENERATED = '''
study = "S"
jobs_dir = "jobs/v4"
[[stage]]
name = "screen"
tasks = "a.txt"
count = 2
array = "1-2"
produces = ["$FOA_STUDY_DIR/jobs/made.txt"]
[[stage]]
name = "full"
tasks = "$FOA_STUDY_DIR/jobs/made.txt"
count = 18
array = "1-18"
'''


def test_a_task_file_an_earlier_stage_produces_need_not_exist_yet(tmp_path, environment):
    assert preflight(load_manifest(_manifest(tmp_path, GENERATED)), environment) == []


def test_a_task_file_nobody_produces_must_exist(tmp_path, environment):
    """The exemption is for generated files, not for missing ones."""
    body = GENERATED.replace('produces = ["$FOA_STUDY_DIR/jobs/made.txt"]\n', "")
    problems = preflight(load_manifest(_manifest(tmp_path, body)), environment)
    assert any("no task file" in p for p in problems), problems


def test_a_variable_rooted_task_path_is_used_as_given(tmp_path, environment):
    """
    Generated files land where the runner's guard allows - inside the study -
    which is not the directory the hand-written ones live in.
    """
    steps = plan(load_manifest(_manifest(tmp_path, GENERATED)), environment)
    full = next(s for s in steps if s["stage"] == "full")
    assert full["tasks"] == f"{environment['FOA_STUDY_DIR']}/jobs/made.txt"
    assert "jobs/v4" not in full["tasks"]


def test_the_generated_stage_still_carries_its_declared_width(tmp_path, environment):
    """
    The width is what the generator's --expect is checked against, so it is the
    one thing that must not be skipped along with the file.
    """
    steps = plan(load_manifest(_manifest(tmp_path, GENERATED)), environment)
    full = next(s for s in steps if s["stage"] == "full")
    assert full["count"] == 18
    assert "--array=1-18" in render(full, {"screen": "1"})


# --------------------------------------------------------------------------- #
# naming the GPU, and its memory
# --------------------------------------------------------------------------- #
#
# 43 elements of a screen died of CUDA OOM. Every one ran on a full 40 GB A100 -
# no MIG slice was involved - while the same cells succeeded on 80 GB cards. On
# this site the gres name `gpu:A100:N` covers BOTH, so naming the type does not
# choose the memory; only the node feature `80gb_vram` does. Both matter, for
# different reasons, and the tool now renders both.
TYPED = BASIC


def test_a_gres_that_names_no_type_is_refused(tmp_path, environment):
    """
    The scheduler may satisfy `gpu:1` with anything it has, and this site's
    shared partition includes a node offering 10 GB MIG slices.
    """
    problems = preflight(load_manifest(_manifest(tmp_path, UNTYPED)), environment)
    assert any("names no GPU type" in p for p in problems), problems


def test_an_untyped_gres_can_be_accepted_deliberately(tmp_path, environment):
    body = UNTYPED.replace('gres = "gpu:1"', 'gres = "gpu:1"\ngres_ambiguous_ok = true')
    assert preflight(load_manifest(_manifest(tmp_path, body)), environment) == []


def test_a_typed_gres_is_rendered_on_every_gpu_stage(tmp_path, environment):
    manifest = load_manifest(_manifest(tmp_path, TYPED))
    assert preflight(manifest, environment) == []
    for step in plan(manifest, environment):
        argv = render(step, {})
        assert "--gres=gpu:A100:1" in argv, argv


def test_a_cpu_stage_asks_for_no_gpu_and_is_not_type_checked(tmp_path, environment,
                                                             workspace):
    (workspace["project"] / "repo_foa" / "slurm" / "prep.sbatch").write_text("#!/bin/bash\n")
    body = '''
study = "S"
jobs_dir = "jobs/v4"
[[stage]]
name = "prep"
tasks = ""
runner = "slurm/prep.sbatch"
count = 1
array = "1-1"
gres = ""
partition = "$FOA_CPU_PARTITION"
'''
    manifest = load_manifest(_manifest(tmp_path, body))
    assert preflight(manifest, environment) == []
    argv = render(plan(manifest, environment)[0], {})
    assert not any(a.startswith("--gres=") for a in argv)


def test_a_node_feature_constraint_is_rendered(tmp_path, environment):
    """
    A gres names the GPU family; it does not say how much memory it has. Where a
    site's cards of one family differ, the feature flag is the only thing that
    chooses - which is exactly what 43 OOM elements cost to learn.
    """
    body = TYPED.replace('array = "1-2"', 'array = "1-2"\nconstraint = "80gb_vram"')
    manifest = load_manifest(_manifest(tmp_path, body))
    assert preflight(manifest, environment) == []
    argv = render(plan(manifest, environment)[0], {})
    assert "--constraint=80gb_vram" in argv


def test_a_constraint_may_come_from_the_environment(tmp_path, environment):
    environment["FOA_GPU_FEATURE"] = "80gb_vram"
    body = TYPED.replace('array = "1-2"', 'array = "1-2"\nconstraint = "$FOA_GPU_FEATURE"')
    argv = render(plan(load_manifest(_manifest(tmp_path, body)), environment)[0], {})
    assert "--constraint=80gb_vram" in argv










def test_no_shipped_task_file_packs_a_consumer_before_its_producer():
    """
    The repository-wide sweep, over the files as they will be submitted.

    The emitter refuses to WRITE a bad stage, but a task file that was written
    before that refusal existed, or by hand, would not be caught by it. An array
    runs by index whatever wrote the file, so the check belongs to the file.

    It covers the study_emit generator steps. A `foa` training line reads and
    writes far more than a table, and ordering those is what the chain's afterok
    edges are for; this is about steps packed into ONE array element range.
    """
    import sys

    tools = str(REPO / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import study_emit

    checked = 0
    for path in sorted(MANIFESTS.glob("*.txt")):
        steps = [line for line in path.read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
        if not any(study_emit.step_what(line) for line in steps):
            continue
        checked += 1
        assert study_emit.ordering_violations(steps) == [], path.name
    assert checked, "the sweep found no generator steps to check at all"
















