"""
Stage B: the digit screens run natively on Shakespeare.

Two things could quietly make this stage meaningless.

If it did not use the *same* tables and the *same* selection rule as the digit
study, "what this task would have chosen on its own" would not be comparable
with what the digit study chose, and the whole native-versus-transferred
comparison would be between two different protocols rather than two datasets.

And if the Fisher-weighted cells were dropped - the easy thing to do, since
their path was the one piece of the digit layout that does not carry over - the
screen would be missing EWC, which is part of the story the chapter tells.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.training import agg_cells, reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import (
    DIGITS_STUDY01, SHAKESPEARE_STUDY01 as CFG,
)

TOOLS = str(Path(__file__).resolve().parents[2] / "tools")


@pytest.fixture()
def native():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_shakespeare_native

    return make_shakespeare_native


# --------------------------------------------------------------------------- #
# the same protocol, natively
# --------------------------------------------------------------------------- #
def test_it_screens_the_digit_studys_own_tables(native):
    """Same cells, or the comparison is between two protocols."""
    folds = (1,)
    assert len(native.agg_screen(folds)) == len(agg_cells.screen_cells()) == 171
    assert len(native.reg_screen(folds)) == len(reg_cells.screen_cells()) == 125


def test_one_fold_by_default_five_on_request(native):
    """
    This study trains one g-0 on one client split. The digit shape is one flag
    away because the choice is the owner's and the cost differs fivefold.
    """
    assert SL.folds_of(CFG) == (1,)
    assert SL.folds_of(DIGITS_STUDY01) == (1, 2, 3, 4, 5)
    assert native.counts((1,))["agg_screen"] == 171
    assert native.counts((1, 2, 3, 4, 5))["agg_screen"] == 855


def test_the_fisher_cells_are_kept_and_addressed_correctly(native):
    """
    Sixteen cells are Fisher-weighted; dropping them would drop EWC. They must
    name this study's own Fisher and must NOT name $G0_FOLD, which a study with
    one g-0 has no selection record to supply - the runner refuses such a task.
    """
    needs = [c for c in reg_cells.screen_cells() if c["needs_fisher"]]
    assert len(needs) == 16

    lines = native.reg_screen((1,))
    fisher_lines = [ln for ln in lines if "fisher_path=" in ln]
    assert len(fisher_lines) == 16
    for line in fisher_lines:
        assert "fisher_path=$FOA_STUDY_DIR/g0/shakespeare/global_results/fisher" in line
        assert "G0_FOLD" not in line

    # And the digit study's own addressing is untouched.
    digit = SL.reg_line(DIGITS_STUDY01, needs[0], 1, SL.SCREEN_ROUNDS, 1)
    assert "g0_fold$G0_FOLD" in digit


def test_the_screen_lines_carry_this_studys_protocol(native):
    for line in native.agg_screen((1,)) + native.reg_screen((1,)):
        assert line.startswith("foa final ")
        assert " --provider shakespeare" in line
        assert " --model char_lstm " in line, (
            "a line that declares the wrong topology still runs - the provider "
            "chooses it - and is then recorded, in its own provenance, as "
            "something it is not"
        )
        assert " --resolution" not in line, "an image flag on a text run"
        assert f" --rounds {SL.SCREEN_ROUNDS} " in line
        assert f" --clients-per-round {CFG.clients_per_round} " in line
        assert " --fold 1 " in line
        assert " --old-fold all" in line
        assert " --init global --global-name g0" in line


def test_the_digit_study_still_declares_its_own_topology(native):
    """The shared line builder is study-driven now; digits must not have moved."""
    from federated_outlier_adaptation.training import agg_cells

    cell = agg_cells.screen_cells()[0]
    digit = SL.agg_line(DIGITS_STUDY01, cell, 1, SL.SCREEN_ROUNDS, 1)
    assert " --model fedavg_cnn " in digit
    assert " --resolution 28 --classes digits " in digit


def test_seeds_are_distinct_and_clear_of_every_prior_range(native):
    lines = native.agg_screen((1,)) + native.reg_screen((1,))
    seeds = [int(ln.split(" --sampler-seed ")[1].split()[0]) for ln in lines]
    assert len(set(seeds)) == len(seeds)
    # s01 runs used 800001-800012, stage A 801000-801009.
    assert min(seeds) >= 810000
    assert all(s >= 802000 for s in seeds)


# --------------------------------------------------------------------------- #
# the chain
# --------------------------------------------------------------------------- #
def test_every_generator_step_is_given_a_matching_expect(native):
    """
    The array widths are fixed before the generators run. A generator that
    emitted a different number exits non-zero, the afterok fails, and the chain
    stops - instead of an array running elements that point at nothing.
    """
    n = native.counts((1,))
    steps = native.generator_lines((1,))
    joined = "\n".join(sum(steps.values(), []))
    assert f"--expect {n['agg_full']}" in joined
    assert f"--expect {n['reg_full']}" in joined
    assert f"--expect {n['reg_hybrid']}" in joined
    assert f"--expect {n['combos']}" in joined
    assert joined.count("--rank-by test") == 3, "the owner's documented rule"


def test_generated_files_land_inside_the_study(native):
    """The runner refuses a path-valued flag that points anywhere else."""
    for step in sum(native.generator_lines((1,)).values(), []):
        words = step.split()
        for prev, word in zip(words, words[1:]):
            if prev in ("--out", "--root"):
                assert word.startswith("$FOA_STUDY_DIR"), (prev, word)


def test_the_chain_orders_selection_after_the_runs_it_reads(native):
    """
    A top-three is computed from full-horizon results, so it cannot be submitted
    beside them. Each generator depends on the array before it.
    """
    script = native.chain_script((1,))
    # The submission lines, in order, by the stage name each one submits.
    names = [
        line.split("submit ", 1)[1].split()[0]
        for line in script.splitlines()
        if line.startswith("J") and "submit " in line
    ]
    assert names == [
        "agg_screen", "gen_agg_full", "agg_full", "gen_agg_top3",
        "reg_screen", "gen_reg", "reg_hybrid", "reg_full", "gen_reg_top3",
        "gen_combos", "combos", "gen_winner",
    ], names
    # Every step but the two screen heads waits on the one before it.
    deps = [line for line in script.splitlines()
            if line.startswith("J") and '"$J' in line and line.count('"') >= 4]
    assert len(deps) >= 10
    assert "--dependency=afterok:$4" in script, "one dependency helper"


def test_the_cost_band_comes_from_measured_rounds(native):
    """
    Not a guess: this project's own Shakespeare runs at m=10, batch 64 - 14.3 s
    a round on selected small clients, 36.7 s on the larger pool.
    """
    assert native.ROUND_SECONDS == (14.3, 36.7)
    rows = native.cost((1,))
    total_low = sum(r[3] for r in rows)
    total_high = sum(r[4] for r in rows)
    assert 45 < total_low < 60, total_low
    assert 120 < total_high < 140, total_high
    # And five folds really is five times the price.
    assert sum(r[3] for r in native.cost((1, 2, 3, 4, 5))) == pytest.approx(
        total_low * 5, rel=1e-6
    )


def test_every_line_parses(native):
    from federated_outlier_adaptation.cli import build_parser

    parser = build_parser()
    for line in native.agg_screen((1,)) + native.reg_screen((1,)):
        parser.parse_args(shlex.split(line.replace("$FOA_STUDY_DIR", "/study"))[1:])


# --------------------------------------------------------------------------- #
# the submission environment
# --------------------------------------------------------------------------- #
#
# The chain was once submitted with no --export at all. Slurm did not propagate
# the submitting shell's environment, every element of both screens started with
# no FOA_PROJECT_DIR, env.sh refused, and 296 array elements failed at startup
# across about three attempts each - roughly ten GPU-hours spent discovering
# that a variable was missing.
#
# The refusal was right. The submission was wrong, and nothing tested it,
# because the chain script was only ever read.
import os
import subprocess


@pytest.fixture()
def chain_run(native, tmp_path):
    """Run the generated chain script in print mode with a chosen environment."""
    script = tmp_path / "s01b_chain.sh"
    script.write_text(native.chain_script((1,)))

    # The script refuses unless the runner and the study exist.
    workspace = tmp_path / "ws"
    (workspace / "repo_foa" / "slurm").mkdir(parents=True)
    (workspace / "repo_foa" / "slurm" / "study_phase.sbatch").write_text("#!/bin/bash\n")
    study = tmp_path / "study"
    study.mkdir()

    def run(go=False, **overrides):
        environment = {
            k: v for k, v in os.environ.items() if not k.startswith("FOA_")
        }
        environment.update({
            "FOA_PROJECT_DIR": str(workspace),
            "FOA_STUDY_DIR": str(study),
            "FOA_ACCOUNT": "acct",
            "FOA_GPU_PARTITION": "gpupart",
        })
        for key, value in overrides.items():
            if value is None:
                environment.pop(key, None)
            else:
                environment[key] = value
        args = ["bash", str(script)] + (["--go"] if go else [])
        return subprocess.run(args, capture_output=True, text=True, env=environment)

    run.workspace = workspace
    run.study = study
    return run


def _required_by_env_sh() -> list:
    """The variables ``env.sh`` refuses to run without."""
    source = (Path(__file__).resolve().parents[2] / "slurm" / "env.sh").read_text()
    assert 'if [ -z "${FOA_PROJECT_DIR:-}" ]; then' in source
    return ["FOA_PROJECT_DIR"]


def test_every_sbatch_carries_the_variables_the_job_needs(chain_run):
    """
    Not `--export=ALL` on its own: on a site whose default is NONE, ALL is
    exactly the default being overridden. Every variable that is set must be
    named and passed by value.
    """
    result = chain_run()
    lines = [ln for ln in result.stderr.splitlines() if ln.startswith("would: sbatch")]
    assert len(lines) == 12, result.stderr

    for line in lines:
        assert "--export=" in line, line
        exported = line.split("--export=")[1].split()[0]
        assert exported.startswith("ALL,"), exported
        for name in _required_by_env_sh():
            assert f"{name}={chain_run.workspace}" in exported, (name, exported)
        assert f"FOA_STUDY_DIR={chain_run.study}" in exported


def test_optional_variables_are_carried_only_when_set(chain_run):
    """
    A variable exported as empty is not the same as one env.sh derives itself,
    and passing `FOA_ENV=` would override the auto-discovery with nothing.
    """
    without = chain_run().stderr.splitlines()[0]
    assert "FOA_SHAKESPEARE_DIR" not in without

    with_it = chain_run(FOA_SHAKESPEARE_DIR="/data/shakespeare").stderr.splitlines()[0]
    assert "FOA_SHAKESPEARE_DIR=/data/shakespeare" in with_it


@pytest.mark.parametrize(
    "missing",
    ["FOA_PROJECT_DIR", "FOA_STUDY_DIR", "FOA_ACCOUNT", "FOA_GPU_PARTITION"],
)
def test_the_preflight_refuses_before_submitting_anything(chain_run, missing):
    """
    Refusing at submit time costs seconds. Refusing inside 349 GPU elements,
    three attempts each, cost about ten GPU-hours.
    """
    result = chain_run(go=True, **{missing: None})
    assert result.returncode == 78, result.stderr
    assert "REFUSED" in result.stderr and missing in result.stderr
    assert "sbatch" not in result.stdout


def test_the_preflight_refuses_a_missing_runner_or_study(chain_run, tmp_path):
    absent = tmp_path / "nowhere"
    assert chain_run(go=True, FOA_PROJECT_DIR=str(absent)).returncode == 78
    assert chain_run(go=True, FOA_STUDY_DIR=str(absent)).returncode == 78


def test_the_generator_steps_submit_nothing_of_their_own(native):
    """
    They run as array elements of this chain, so they inherit its environment.
    A nested sbatch or srun inside one would need its own --export and would be
    the same bug one level down.
    """
    for step in sum(native.generator_lines((1,)).values(), []):
        assert "sbatch" not in step and "srun" not in step, step
        assert step.startswith("$FOA_PYTHON "), step
