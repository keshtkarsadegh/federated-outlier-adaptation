"""
What a reviewer who assembles the tree has to get back, byte for byte.

`docs/REPRODUCE.md` section 4 promises that every task file regenerates from
its generator, and section 6 that every view regenerates from the records.  Two
of those promises had drifted by the time somebody followed the documents
literally on a fresh clone: `s24_combo_full.txt` differed in its header because
the generator's prose had been edited after the file was emitted, and the two
signal extracts differed in every correlation column because `foa signals`
pools over whatever runs the root holds and the root had since grown by two
stages.  Both are pinned here.

**The task-file half runs anywhere**: the generators read the metadata core,
which is in the clone.  **The signals half needs the records** and is skipped
without them - set `FOA_STUDY_DIR` to an assembled root (section 10) to run it.

NOTHING HERE WRITES INTO THE STUDY ROOT.  Several emitters rewrite their own
selection record and append to `tables/BOUNDARY_HITS.txt`, and `foa signals`
would replace the six shipped Pareto plots.  So the regenerations run against a
shadow root: every run folder symlinked, every directory an emitter writes into
copied.  That is also the answer to the reviewer's version of the problem, and
`docs/REPRODUCE.md` section 6 now says so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "study" / "artifacts" / "Digits_study01"
JOBS = ARTIFACTS / "jobs"
PAPER = ARTIFACTS / "tables" / "paper"
MANIFEST = ARTIFACTS / "signals_population.txt"

sys.path.insert(0, str(REPO / "src"))

from federated_outlier_adaptation.analysis.forgetting_signals import (  # noqa: E402
    Population,
    analyse,
    collect_trajectories,
    load_population,
)
from federated_outlier_adaptation.runners.forgetting_signals import (  # noqa: E402
    SIGNAL_KEYS,
)

#: The six runs of the population that the records asset does not carry,
#: because `report_tables.in_study` keeps them out of every table and the same
#: predicate decides what is packed.  They ship in the metadata core instead -
#: a pass cannot be repeated without the runs it was taken over.
CORE_ONLY = (
    "signalcheck_BaseTrainer_grid_search",
    "d01_extreme_single_fold1_AnchoredTrainer_grid_search",
    "d01_extreme_single_fold2_AnchoredTrainer_grid_search",
    "d01_extreme_single_fold3_AnchoredTrainer_grid_search",
    "d01_extreme_single_fold4_AnchoredTrainer_grid_search",
    "d01_extreme_single_fold5_AnchoredTrainer_grid_search",
)


# ------------------------------------------------------------ the population
def test_the_manifest_is_a_set_of_run_names():
    """One run per line, no duplicates, each naming a folder and an arm."""
    population = load_population(MANIFEST)
    assert len(population) == 2471
    for name in population.names:
        assert not name.startswith("/") and not name.endswith("/"), name
        # folder / fold-seed / scenario / arm: the identity
        # `signal_correlations.csv` carries, not a bare folder.
        assert len(name.split("/")) == 4, name
    assert len(population.folders) == 1591


def test_the_manifest_says_which_pass_it_is_and_why_it_is_frozen():
    """
    A frozen population is only honest if it says what it excludes.

    The reason is not an implementation detail: the pass predates two stages,
    and a reader who runs `foa signals` without the manifest gets a different
    and equally valid number.  The file has to say so where it is read.
    """
    header = "\n".join(
        line for line in MANIFEST.read_text().splitlines() if line.startswith("#")
    )
    assert "2026-09-01" in header
    assert "stage 20" in header and "E1-E4" in header
    assert "7,005" in header  # what the pass measures without the manifest
    for folder in CORE_ONLY[:1] + ("d01_extreme_single_fold{1..5}",):
        assert folder.split("/")[0] in header


def test_every_run_the_asset_omits_is_published_in_the_metadata_core():
    """
    The six the records asset leaves out ship here, or the pass cannot be run.

    Only `summary_*.json` and `accuracies_*.json`, as for every other run: the
    core publishes records, never weights.
    """
    for folder in CORE_ONLY:
        records = sorted((ARTIFACTS / folder).rglob("*.json"))
        assert records, folder
        assert {path.name for path in records} == {"summary_0.json", "accuracies_0.json"}
        assert not list((ARTIFACTS / folder).rglob("*.pt")), folder


def test_the_manifest_names_only_runs_the_published_records_can_supply():
    """
    Every named run is either in the asset or in the core, and none is neither.

    The asset's half cannot be checked from a clone; the core's half can, and
    it is the half that would go missing silently.
    """
    population = load_population(MANIFEST)
    core = {name for name in population.names if name.split("/", 1)[0] in CORE_ONLY}
    assert len(core) == 6
    for name in core:
        assert (ARTIFACTS / name).parent.is_dir() or (
            ARTIFACTS / name.rsplit("/", 1)[0]
        ).is_dir(), name


def _write_run(root: Path, folder: str, arm: str, rounds: int = 5) -> str:
    """A minimal stored run of the shape `collect_trajectories` reads."""
    where = root / folder / "fold1_seed_1" / "concurrent_delta"
    where.mkdir(parents=True)
    block = {
        "config": {"trainer": "BaseTrainer", "scenario": "concurrent_delta"},
        "heldout_client_accuracies": [0.50 + 0.01 * i for i in range(rounds)],
        "source_val_accuracies": [0.99 - 0.001 * i for i in range(rounds)],
        "accuracies": [[i, 0.99 - 0.001 * i] for i in range(rounds)],
        **{key: [0.10 * i for i in range(rounds)] for key in SIGNAL_KEYS},
    }
    (where / "summary_0.json").write_text(json.dumps({arm: block}))
    return f"{folder}/fold1_seed_1/concurrent_delta/{arm}"


def test_without_a_population_the_pass_pools_over_whatever_is_there(tmp_path):
    """The behaviour that made the shipped views unreproducible."""
    first = _write_run(tmp_path, "d01_reg_cell_fold1_BaseTrainer_grid_search", "arm")
    second = _write_run(tmp_path, "d01_reg_later_fold1_BaseTrainer_grid_search", "arm")
    assert {run.name for run in collect_trajectories(tmp_path)} == {first, second}


def test_a_population_restricts_the_pass_to_the_runs_it_names(tmp_path):
    """And the later stage - same prefix, same shape - stays out."""
    first = _write_run(tmp_path, "d01_reg_cell_fold1_BaseTrainer_grid_search", "arm")
    _write_run(tmp_path, "d01_reg_later_fold1_BaseTrainer_grid_search", "arm")
    population = Population(frozenset({first}))
    assert [run.name for run in collect_trajectories(tmp_path, population)] == [first]


def test_a_manifest_line_may_carry_a_comment(tmp_path):
    manifest = tmp_path / "population.txt"
    manifest.write_text("# why\na/b/c/d\n\n   e/f/g/h   # and this one\n")
    assert load_population(manifest).names == frozenset({"a/b/c/d", "e/f/g/h"})


def test_an_empty_manifest_is_refused(tmp_path):
    manifest = tmp_path / "population.txt"
    manifest.write_text("# all comment, no runs\n")
    with pytest.raises(ValueError, match="not a population"):
        load_population(manifest)


def test_a_population_the_tree_cannot_supply_is_refused_by_name(tmp_path):
    """
    A pass over part of a population is a different pass, so it stops.

    This is the check that catches a half-assembled reviewer tree, which would
    otherwise produce a table that looks right and is over fewer runs.
    """
    root = tmp_path / "root"
    root.mkdir()
    present = _write_run(root, "d01_reg_cell_fold1_BaseTrainer_grid_search", "arm")
    manifest = tmp_path / "population.txt"
    manifest.write_text(present + "\nd01_reg_absent_fold1_X/fold1_seed_1/s/arm\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        analyse(root, out_dir=tmp_path / "out", plots=False, population=manifest)
    assert "d01_reg_absent_fold1_X" in str(excinfo.value)


# --------------------------------------------------------- the regenerations
#: `(argv after the interpreter, file written, file it must equal)`.  Every
#: command is the one `docs/REPRODUCE.md` section 4 gives, with only the output
#: directory moved.
def _task_file_cases(out: Path, root: Path):
    tools = REPO / "tools"
    emit = [str(tools / "study_emit.py")]
    size = [str(tools / "make_size_references.py"), "--study-dir", str(root)]
    return [
        ([str(tools / "make_digits_p11.py"), "--jobs-dir", str(out)],
         out / "d01_p11.txt", JOBS / "s09_agg_screen2.txt"),
        ([str(tools / "make_digits_p11.py"), "--jobs-dir", str(out)],
         out / "d01_p11_README.md", JOBS / "d01_p11_README.md"),
        ([str(tools / "make_digits_p13.py"), "--jobs-dir", str(out)],
         out / "d01_p13.txt", JOBS / "s16_reg_screen3.txt"),
        ([str(tools / "make_digits_p13.py"), "--jobs-dir", str(out)],
         out / "d01_p13_README.md", JOBS / "s16_reg_screen3_README.md"),
        ([str(tools / "make_digits_p21.py"), "--jobs-dir", str(out)],
         out / "s21_blend_screen.txt", JOBS / "s21_blend_screen.txt"),
        ([str(tools / "make_digits_p21.py"), "--jobs-dir", str(out)],
         out / "s21_blend_screen_README.md", JOBS / "s21_blend_screen_README.md"),
        ([str(tools / "make_digits_p23.py"), "--jobs-dir", str(out)],
         out / "s23_combo_screen.txt", JOBS / "s23_combo_screen.txt"),
        ([str(tools / "make_digits_p23.py"), "--jobs-dir", str(out)],
         out / "s23_combo_screen_README.md", JOBS / "s23_combo_screen_README.md"),
        ([str(tools / "make_digits_p25.py"), "--jobs-dir", str(out)],
         out / "s25_combo_screen_selected.txt", JOBS / "s25_combo_screen_selected.txt"),
        ([str(tools / "make_digits_p25.py"), "--jobs-dir", str(out)],
         out / "s25_combo_screen_selected_README.md",
         JOBS / "s25_combo_screen_selected_README.md"),
        (emit + ["agg-full", "--root", str(root), "--out", str(out / "agg_full.txt"),
                 "--expect", "85"],
         out / "agg_full.txt", JOBS / "s10_agg_full2.txt"),
        (emit + ["reg-full", "--root", str(root), "--out", str(out / "reg_full.txt"),
                 "--expect", "70"],
         out / "reg_full.txt", JOBS / "s17_reg_full4.txt"),
        (emit + ["combos", "--root", str(root), "--out", str(out / "combos.txt"),
                 "--expect", "90"],
         out / "combos.txt", JOBS / "s20_combos4.txt"),
        (emit + ["reg-hybrid", "--root", str(root), "--out", str(out / "hybrid.txt"),
                 "--expect", "15"],
         out / "hybrid.txt", JOBS / "s18_hybrid.txt"),
        (emit + ["reg-hybrid", "--root", str(root), "--out", str(out / "hybrid_seq.txt"),
                 "--expect", "15", "--hybrid-family", "sequential"],
         out / "hybrid_seq.txt", JOBS / "s19_hybrid_seq.txt"),
        (emit + ["five", "--root", str(root), "--out", str(out / "five.txt"),
                 "--expect", "20"],
         out / "five.txt", JOBS / "d01_five.txt"),
        (emit + ["c10d10", "--root", str(root), "--out", str(out / "c10d10.txt"),
                 "--expect", "20"],
         out / "c10d10.txt", JOBS / "d01_c10d10.txt"),
        (emit + ["c20", "--root", str(root), "--out", str(out / "c20.txt"),
                 "--expect", "30"],
         out / "c20.txt", JOBS / "d01_c20.txt"),
        (emit + ["extreme", "--root", str(root), "--out", str(out / "extreme.txt"),
                 "--expect", "10"],
         out / "extreme.txt", JOBS / "d01_extreme.txt"),
        (emit + ["combo-tune-full", "--root", str(root),
                 "--out", str(out / "s24.txt"), "--expect", "10"],
         out / "s24.txt", JOBS / "s24_combo_full.txt"),
        (emit + ["combo-tune-full-selected", "--root", str(root),
                 "--out", str(out / "s26.txt"), "--expect", "10"],
         out / "s26.txt", JOBS / "s26_combo_full_selected.txt"),
        (size + ["--cohort", "cohort_worst10.json", "--book", "cohort10",
                 "--per-round", "9", "8", "--tag", "c10", "--seed-base", "740000",
                 "--out", str(out / "refs_c10.txt")],
         out / "refs_c10.txt", JOBS / "s03_refs_c10.txt"),
        (size + ["--cohort", "cohort_worst5.json", "--book", "cohort5",
                 "--per-round", "4", "--tag", "c5", "--seed-base", "750000",
                 "--out", str(out / "refs_c5.txt")],
         out / "refs_c5.txt", JOBS / "d01_c5_references.txt"),
        (size + ["--cohort", "cohort_worst20.json", "--book", "cohort20",
                 "--per-round", "18", "16", "--tag", "c20", "--seed-base", "720000",
                 "--out", str(out / "refs_c20.txt")],
         out / "refs_c20.txt", JOBS / "d01_c20_references.txt"),
    ]


def _shadow_root(study: Path, into: Path) -> Path:
    """
    A study root an emitter may write into without touching the reviewer's.

    Run folders are symlinked - there are five thousand of them and a gigabyte
    of records - and the small directories the emitters rewrite are copied, so
    a selection record re-emitted here cannot reach the tree the reader just
    checksummed.
    """
    root = into / "shadow"
    root.mkdir()
    writable = {"tables", "jobs"}
    for entry in sorted(study.iterdir()):
        if entry.name in writable and entry.is_dir():
            shutil.copytree(entry, root / entry.name, symlinks=True)
        else:
            (root / entry.name).symlink_to(entry)
    return root


def _study_root():
    root = os.environ.get("FOA_STUDY_DIR")
    if not root or not (Path(root) / "tables").is_dir():
        pytest.skip("no study root; set FOA_STUDY_DIR to an assembled tree")
    return Path(root)


def _run(argv, cwd):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(REPO / "tools"), env.get("PYTHONPATH", "")]
    )
    done = subprocess.run(
        [sys.executable] + argv, cwd=cwd, env=env,
        capture_output=True, text=True,
    )
    assert done.returncode == 0, f"{argv}\n{done.stdout}\n{done.stderr}"


def test_all_twenty_four_task_files_regenerate_byte_for_byte():
    """
    Section 4's promise, for every generator it names, in one place.

    Twenty-three of the twenty-four held when this was first checked by hand.
    `s24_combo_full.txt` did not: its generator's header prose had been edited
    after the file was emitted, so the shipped artefact was stale against the
    only thing that can produce it.  The ten task lines never differed, which
    is why nobody noticed - and why the check is a diff of the file rather than
    of its task lines.
    """
    study = _study_root()
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        root = _shadow_root(study, scratch)
        out = scratch / "jobs"
        out.mkdir()
        seen = set()
        for argv, written, shipped in _task_file_cases(out, root):
            key = tuple(argv)
            if key not in seen:          # one run feeds both of a pair's files
                _run(argv, cwd=REPO)
                seen.add(key)
            assert written.read_bytes() == shipped.read_bytes(), shipped.name
        assert len(_task_file_cases(out, root)) == 24


def test_the_signals_pass_reproduces_the_two_shipped_extracts():
    """
    Section 6's promise for the one view that had stopped keeping it.

    The pass has to be taken over the population the manifest names; over the
    tree as it stands it pools four and a half thousand runs more and moves
    eighteen of the manuscript's macros.  This is the test that would have
    caught that, and the reason `--population` exists.
    """
    study = _study_root()
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        signals = scratch / "signals"
        _run(
            ["-m", "federated_outlier_adaptation.cli", "signals",
             "--root", str(study), "--out", str(signals),
             "--population", str(MANIFEST)],
            cwd=REPO,
        )
        out = scratch / "views"
        _run(
            [str(REPO / "tools" / "export_signals_summary.py"),
             "--root", str(study), "--signals", str(signals), "--out", str(out)],
            cwd=REPO,
        )
        for name in ("signals_summary_extract.csv", "signals_extras_extract.csv"):
            assert (out / name).read_bytes() == (PAPER / name).read_bytes(), name
