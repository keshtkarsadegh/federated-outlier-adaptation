"""
Submitting a chain, from a manifest that is checked in and tested.

Why this exists
---------------
Every task file in this repository is generated, parse-checked and dry-run
before it is submitted.  The *submission* never was: it was assembled at a
terminal, by hand, once per chain.  Four separate failures came out of that and
none of them was a bug in the science -

* a guard read ``$TASK`` before the line that assigns it, and 585 elements died
  at startup;
* a chain was submitted with no ``--export``, the jobs got none of the
  environment, and 296 elements refused - about ten GPU-hours to discover a
  missing variable;
* a stage was submitted before the step that writes what it reads;
* a GPU request went to a partition that has none.

Each was fixed where it happened.  The pattern was not: a hand-assembled
submission is untested code that runs once, at the most expensive moment.

So a chain is now a **manifest** - a file next to the task files, declaring what
each stage submits, what it needs to exist first, and what it produces - and
this module refuses the whole thing before the first ``sbatch`` if any of that
does not hold.  Zero jobs on any failure: a chain that is half submitted is
worse than one that is not, because the half that ran has to be found and
cancelled.

The manifest
------------
TOML, read with the standard library::

    study    = "Digits_study01"
    jobs_dir = "jobs/v4"

    [defaults]
    partition = "$FOA_GPU_PARTITION"
    gres      = "gpu:1"

    [[stage]]
    name     = "p00_prepare"
    tasks    = "d01_p00_prepare.txt"
    count    = 1
    array    = "1-1"
    time     = "06:00:00"
    partition = "$FOA_CPU_PARTITION"
    gres     = ""                       # no GPU: packing the cache is CPU work
    requires = ["$FOA_DATA_DIR/nist/by_write.zip"]
    produces = ["$FOA_NIST28_DIR/nist28_index.json"]

``requires`` is checked at submit time **unless an earlier stage produces it** -
which is what lets a chain declare its real external inputs without pretending
that a file the chain itself writes must already be there.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Variables a job needs in its own environment.  ``env.sh`` refuses without
#: ``FOA_PROJECT_DIR`` and derives the rest - but a derived default is only right
#: when the submitting shell's value was the default too, so everything that is
#: set is carried explicitly rather than re-guessed on the compute node.
CARRY = (
    "FOA_PROJECT_DIR", "FOA_STUDY_DIR", "FOA_REPO", "FOA_SLURM_DIR",
    "FOA_RESULTS_DIR",
    "FOA_DATA_DIR", "FOA_NIST28_DIR",
    "FOA_CACHE_DIR", "FOA_ENV", "FOA_MODULES",
    "FOA_HTTP_PROXY", "FOA_NIST_CLASSES", "FOA_ACCOUNT", "FOA_GPU_PARTITION",
    "FOA_CPU_PARTITION",
)

#: What no chain can run without.
REQUIRED_ENV = ("FOA_PROJECT_DIR", "FOA_STUDY_DIR", "FOA_ACCOUNT")

#: Slurm walltime: ``H:MM:SS``, ``D-HH:MM:SS``, ``MM:SS`` or plain minutes.
WALLTIME = re.compile(r"^(\d+-)?(\d+:)?\d+(:\d+)?$")

#: A GPU request that names a type: ``gpu:A100:1``.  A bare ``gpu:1`` is what a
#: scheduler is free to satisfy with whatever it has - on this site that includes
#: MIG slices of ten gigabytes, which is not what any of this fits in.
TYPED_GRES = re.compile(r"^[a-z]+:[A-Za-z0-9._]+:\d+$")

#: ``1-40``, ``1-40%4``, ``3``, ``1,5,9``.
ARRAY = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*(%\d+)?$")

#: The flags the task-file runner insists point inside the study root.  Kept in
#: step with ``slurm/study_phase.sbatch``'s own list, and checked here so the
#: refusal happens once at submit time rather than once per array element.
PATH_FLAGS = {
    "--results-dir", "--out", "--model-path", "--fold-book", "--clients-file",
    "--writers-file", "--exclude-file", "--old-book", "--old-clients-file",
    "--outliers-file", "--init-checkpoint", "--root", "--data-dir",
    "--cache-dir", "--scores", "--csv", "--counts", "--subset-of",
    "--disjoint-from", "--source", "--target", "--record",
}

#: The runner every task file goes through.  A stage naming a different script
#: is not a task file and is not guarded this way.
TASK_RUNNER = "slurm/study_phase.sbatch"


def guard_violations(path: Path, limit: int = 3) -> List[str]:
    """
    Path-valued flags in a task file that point outside ``$FOA_STUDY_DIR``.

    The runner refuses these, correctly, once per array element - which is how a
    296-element pair of screens spent ten GPU-hours reporting a mistake that a
    single grep would have found. This is that grep, run before the submission.
    """
    found: List[str] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        for previous, word in zip(words, words[1:]):
            if previous in PATH_FLAGS and not word.startswith("-"):
                if not word.startswith("$FOA_STUDY_DIR"):
                    found.append(f"line {number}: {previous} {word}")
                    if len(found) >= limit:
                        return found
    return found


class ManifestError(ValueError):
    """A manifest that cannot be used, with a reason a reader can act on."""


@dataclass
class Stage:
    name: str
    tasks: str
    count: int
    array: str
    time: Optional[str] = None
    partition: Optional[str] = None
    gres: Optional[str] = None
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    after: Optional[str] = None
    #: Slurm node feature(s) this stage needs, e.g. ``80gb_vram``.  A gres name
    #: says how many GPUs and of what family; it does not say how much memory
    #: they have.  Where a site's cards of one family differ in VRAM - and this
    #: one's A100s are 40 GB and 80 GB under a single ``gpu:A100`` name - the
    #: feature flag is the only thing that chooses.
    constraint: Optional[str] = None
    #: Accept a type-ambiguous ``gres`` deliberately.
    gres_ambiguous_ok: bool = False
    #: ``True`` for a stage that starts a branch rather than continuing the chain.
    root: bool = False
    #: The batch script, relative to the repository.  Defaults to the task-file
    #: runner.  A stage that writes OUTSIDE the study root - the packed cache is
    #: shared between studies and belongs to none of them - cannot go through
    #: that runner at all, because its guard refuses exactly such a path. Naming
    #: its own script is the honest way to say so.
    runner: str = "slurm/study_phase.sbatch"


@dataclass
class Manifest:
    study: str
    jobs_dir: str
    stages: List[Stage]
    path: Optional[Path] = None

    def stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise ManifestError(f"no stage named {name!r}")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
_STAGE_KEYS = {"name", "tasks", "count", "array", "time", "partition", "gres",
               "requires", "produces", "after", "root", "runner", "constraint",
               "gres_ambiguous_ok"}


def load_manifest(path) -> Manifest:
    """Read and structurally validate a manifest; raise :class:`ManifestError`."""
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise ManifestError(f"no manifest at {path}")
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{path}: {error}")

    for key in ("study", "jobs_dir"):
        if not raw.get(key):
            raise ManifestError(f"{path}: no {key!r}")
    entries = raw.get("stage") or []
    if not entries:
        raise ManifestError(f"{path}: no [[stage]] entries")

    defaults = raw.get("defaults") or {}
    stages: List[Stage] = []
    seen = set()
    for index, entry in enumerate(entries):
        unknown = set(entry) - _STAGE_KEYS
        if unknown:
            raise ManifestError(
                f"{path}: stage {index} has unknown key(s) {sorted(unknown)}"
            )
        for key in ("name", "tasks", "count", "array"):
            if key not in entry:
                raise ManifestError(f"{path}: stage {index} has no {key!r}")
        name = str(entry["name"])
        if name in seen:
            raise ManifestError(f"{path}: two stages named {name!r}")
        seen.add(name)
        merged = {**defaults, **entry}
        stages.append(Stage(
            name=name,
            tasks=str(entry["tasks"]),
            count=int(entry["count"]),
            array=str(entry["array"]),
            time=merged.get("time"),
            partition=merged.get("partition"),
            gres=merged.get("gres"),
            requires=list(entry.get("requires") or []),
            produces=list(entry.get("produces") or []),
            after=entry.get("after"),
            root=bool(entry.get("root", False)),
            runner=merged.get("runner", "slurm/study_phase.sbatch"),
            constraint=merged.get("constraint"),
            gres_ambiguous_ok=bool(merged.get("gres_ambiguous_ok", False)),
        ))

    for stage in stages:
        if stage.after and stage.after not in seen:
            raise ManifestError(
                f"{path}: stage {stage.name!r} waits on {stage.after!r}, "
                "which is not a stage in this manifest"
            )
        if stage.after and stage.root:
            raise ManifestError(
                f"{path}: stage {stage.name!r} is both a root and after "
                f"{stage.after!r}"
            )
    return Manifest(study=raw["study"], jobs_dir=raw["jobs_dir"],
                    stages=stages, path=path)


def expand(value: str, environment: Optional[Dict[str, str]] = None) -> str:
    """``$VAR`` and ``${VAR}`` from the environment; unset names stay literal."""
    environment = os.environ if environment is None else environment

    def replace(match):
        name = match.group(1) or match.group(2)
        return environment.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)", replace, value)


def task_count(path: Path) -> int:
    """Runnable lines of a task file: the runner skips blanks and ``#``."""
    return sum(
        1 for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
def task_path(stage, jobs_dir: Path, environment: Dict[str, str]) -> Path:
    """
    Where a stage's task file is.

    A bare name is relative to ``jobs_dir``.  A name that starts with ``$`` or
    ``/`` is used as given, because a chain whose middle stages are *generated*
    writes them where the runner's guard allows - inside the study - and that is
    not the directory the hand-written ones live in.
    """
    raw = expand(stage.tasks, environment)
    return Path(raw) if raw.startswith("/") else jobs_dir / raw


def preflight(manifest: Manifest,
              environment: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Everything that must hold before the first ``sbatch``.

    Returns the problems, in the order they were found.  An empty list means the
    chain may be submitted; anything else means **no** job is.
    """
    environment = dict(os.environ if environment is None else environment)
    problems: List[str] = []

    missing = [name for name in REQUIRED_ENV if not environment.get(name)]
    if missing:
        problems.append(
            f"these environment variables are not set: {', '.join(missing)}"
        )

    project = environment.get("FOA_PROJECT_DIR")
    study_dir = environment.get("FOA_STUDY_DIR")
    if project:
        for script in sorted({stage.runner for stage in manifest.stages}):
            runner = Path(project) / "repo_foa" / script
            if not runner.is_file():
                problems.append(f"no runner at {runner}")
    if study_dir and not Path(study_dir).is_dir():
        problems.append(f"no study at {study_dir}")

    needs_gpu_partition = any(
        (stage.gres or "") and not stage.partition for stage in manifest.stages
    )
    if needs_gpu_partition and not environment.get("FOA_GPU_PARTITION"):
        problems.append(
            "a stage asks for a GPU and no partition of its own, and "
            "FOA_GPU_PARTITION is not set"
        )

    jobs_dir = Path(expand(manifest.jobs_dir, environment))
    if not jobs_dir.is_absolute() and project:
        jobs_dir = Path(project) / jobs_dir

    produced: set = set()
    for stage in manifest.stages:
        if stage.tasks:
            tasks = task_path(stage, jobs_dir, environment)
            # A task file an earlier stage produces cannot exist yet, and its
            # length is the thing that stage is given --expect for. Checking it
            # here would refuse every generated chain.
            generated = (stage.tasks in produced
                         or expand(stage.tasks, environment) in produced
                         or str(tasks) in produced)
            if generated:
                pass
            elif not tasks.is_file():
                problems.append(f"{stage.name}: no task file at {tasks}")
            else:
                found = task_count(tasks)
                if found != stage.count:
                    problems.append(
                        f"{stage.name}: {tasks.name} holds {found} runnable "
                        f"lines and the manifest says {stage.count} - the array "
                        "width is written from the manifest, so one of them is "
                        "wrong"
                    )
                if stage.runner == TASK_RUNNER:
                    for violation in guard_violations(tasks):
                        problems.append(
                            f"{stage.name}: {tasks.name} names a path outside "
                            f"$FOA_STUDY_DIR, which the runner refuses - "
                            f"{violation}"
                        )
        elif stage.count != 1 or stage.array != "1-1":
            problems.append(
                f"{stage.name}: a stage with no task file is one invocation of "
                f"its own script, so count must be 1 and array '1-1'; got "
                f"{stage.count} and {stage.array!r}"
            )

        if not ARRAY.match(stage.array):
            problems.append(f"{stage.name}: {stage.array!r} is not an array spec")
        if stage.time and not WALLTIME.match(stage.time):
            problems.append(f"{stage.name}: {stage.time!r} is not a walltime")

        # The conflict that sent a GPU request to a partition without one.
        partition = expand(stage.partition or "", environment)
        gres = (stage.gres or "").strip()
        cpu_partition = environment.get("FOA_CPU_PARTITION", "")
        if gres and cpu_partition and partition == cpu_partition:
            problems.append(
                f"{stage.name}: asks for {gres!r} on the CPU partition "
                f"{partition!r}"
            )
        if gres and not stage.gres_ambiguous_ok and not TYPED_GRES.match(gres):
            problems.append(
                f"{stage.name}: gres {gres!r} names no GPU type. The scheduler "
                "may satisfy it with anything it has, including a MIG slice; "
                "name the type (gpu:A100:1), or set gres_ambiguous_ok = true."
            )
        if gres and stage.partition and partition.startswith("$"):
            problems.append(
                f"{stage.name}: partition {stage.partition!r} does not expand; "
                "the variable is unset"
            )

        for requirement in stage.requires:
            resolved = expand(requirement, environment)
            if requirement in produced or resolved in produced:
                continue
            if resolved.startswith("$"):
                problems.append(
                    f"{stage.name}: prerequisite {requirement!r} does not "
                    "expand; the variable is unset"
                )
            elif not Path(resolved).exists():
                problems.append(
                    f"{stage.name}: prerequisite missing - {resolved}"
                )
        produced.update(stage.produces)
        produced.update(expand(p, environment) for p in stage.produces)

    return problems


# --------------------------------------------------------------------------- #
# building the submission
# --------------------------------------------------------------------------- #
def export_set(environment: Optional[Dict[str, str]] = None) -> str:
    """
    ``--export`` by value.

    ``ALL`` on its own is not enough on a site whose default is ``NONE``: ALL is
    exactly the default being overridden.  Only variables that are actually set
    are named, because exporting one empty would override a derivation with
    nothing.
    """
    environment = os.environ if environment is None else environment
    parts = ["ALL"]
    for name in CARRY:
        value = environment.get(name)
        if value:
            parts.append(f"{name}={value}")
    return ",".join(parts)


def plan(manifest: Manifest,
         environment: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    The ``sbatch`` invocations, in order, with their dependencies resolved.

    A stage waits on the one before it unless it says otherwise: ``root = true``
    starts a branch, ``after = "<stage>"`` names a different predecessor.  That
    covers a linear chain and the two-branch shape a screen pair needs, without
    a dependency language nobody wants to learn.
    """
    environment = dict(os.environ if environment is None else environment)
    project = environment.get("FOA_PROJECT_DIR", "")
    study_dir = environment.get("FOA_STUDY_DIR", "")
    jobs_dir = Path(expand(manifest.jobs_dir, environment))
    if not jobs_dir.is_absolute() and project:
        jobs_dir = Path(project) / jobs_dir
    exports = export_set(environment)

    steps: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    for stage in manifest.stages:
        if stage.root:
            depends = None
        elif stage.after:
            depends = stage.after
        else:
            depends = previous

        argv = ["sbatch", "--parsable", f"--job-name={stage.name}",
                f"--account={environment.get('FOA_ACCOUNT', '')}"]
        partition = expand(stage.partition or environment.get("FOA_GPU_PARTITION", ""),
                           environment)
        if partition:
            argv.append(f"--partition={partition}")
        gres = (stage.gres or "").strip()
        if gres:
            argv.append(f"--gres={gres}")
        constraint = expand(stage.constraint or "", environment)
        if constraint:
            argv.append(f"--constraint={constraint}")
        if stage.time:
            argv.append(f"--time={stage.time}")
        argv += [f"--array={stage.array}",
                 f"--output={study_dir}/logs/%x_%A_%a.log",
                 f"--export={exports}"]
        steps.append({
            "stage": stage.name,
            "tasks": str(task_path(stage, jobs_dir, environment)) if stage.tasks else "",
            "count": stage.count,
            "array": stage.array,
            "depends_on": depends,
            "argv": argv,
            "runner": str(Path(project) / "repo_foa" / stage.runner),
        })
        previous = stage.name
    return steps


def render(step: Dict[str, Any], job_ids: Dict[str, str]) -> List[str]:
    """One step's full argv, with its dependency filled in."""
    argv = list(step["argv"])
    depends = step["depends_on"]
    if depends:
        argv.append(f"--dependency=afterok:{job_ids.get(depends, '<pending>')}")
    argv.append(step["runner"])
    if step["tasks"]:
        argv.append(step["tasks"])
    return argv


# --------------------------------------------------------------------------- #
# submitting
# --------------------------------------------------------------------------- #
def submit(manifest: Manifest, environment: Optional[Dict[str, str]] = None,
           dry_run: bool = True, runner=subprocess.run,
           log=print) -> Dict[str, Any]:
    """
    Preflight, then submit the whole chain - or nothing.

    Returns a record of what was done, which is also written into the study's
    ``logs/`` so that "which jobs were this chain?" has an answer that does not
    depend on anyone's scrollback.
    """
    environment = dict(os.environ if environment is None else environment)
    problems = preflight(manifest, environment)
    if problems:
        log(f"REFUSED: {manifest.study} - {len(problems)} problem(s), "
            "nothing submitted", file=sys.stderr)
        for problem in problems:
            log(f"  - {problem}", file=sys.stderr)
        return {"submitted": False, "problems": problems}

    steps = plan(manifest, environment)
    job_ids: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []

    for step in steps:
        argv = render(step, job_ids)
        if dry_run:
            log("would: " + " ".join(shlex.quote(a) for a in argv))
            job_ids[step["stage"]] = f"<{step['stage']}>"
        else:
            result = runner(argv, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                log(f"FAILED at {step['stage']}: {result.stderr.strip()}",
                    file=sys.stderr)
                log(f"  {len(job_ids)} job(s) were already submitted: "
                    f"{', '.join(job_ids.values())}", file=sys.stderr)
                log("  scancel them before retrying.", file=sys.stderr)
                return {"submitted": False, "partial": dict(job_ids),
                        "failed_at": step["stage"]}
            job_ids[step["stage"]] = result.stdout.strip().split(";")[0]
        rows.append({"stage": step["stage"], "array": step["array"],
                     "count": step["count"], "depends_on": step["depends_on"],
                     "job_id": job_ids[step["stage"]],
                     "tasks": step["tasks"]})

    log("")
    log(f"{'stage':22s} {'array':>10s} {'tasks':>6s}  {'job id':>12s}  after")
    for row in rows:
        log(f"{row['stage']:22s} {row['array']:>10s} {row['count']:>6d}  "
            f"{row['job_id']:>12s}  {row['depends_on'] or '-'}")

    record = {
        "study": manifest.study,
        "manifest": str(manifest.path) if manifest.path else None,
        "submitted": not dry_run,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "export": export_set(environment),
        "stages": rows,
    }
    study_dir = environment.get("FOA_STUDY_DIR")
    if study_dir and not dry_run:
        logs = Path(study_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / f"submission_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(record, indent=2) + "\n")
        log(f"\nrecord -> {out}")
        record["record"] = str(out)
    return record
