"""
The methods paragraph, held to the code and the records that produced it.

A methods section is the one part of a paper nobody can check against a table.
Every other sentence the manuscript sets points at a view, a macro and a
provenance comment; "the clients train with Adam at $10^{-3}$ for five local
epochs" points at nothing, and a default that moves in `src/` moves the paper
without moving a number in it. The failure is silent in both directions: the
code can drift away from the printed recipe, and the printed recipe can drift
away from a doc that still states the old one.

So every statement of that paragraph is asserted here against its own source of
truth, and never against another statement of the same fact:

  * the optimiser, its two constants and what is NOT in the training loop, from
    `trainers/base_trainer.py` and from a sweep of `src/`;
  * the batch size, the local epochs, the horizons, the participation and the
    seeds, from the shipped task lines under `study/artifacts/*/jobs/`, which
    are the lines that actually ran;
  * the model and its parameter count, from `models/fedavg_cnn.py`, built;
  * the cut that drew the bad pool, from `outliers/scoring.py` and the frozen
    `outliers/pools.json`;
  * what the programme cost, from `tables/paper/cost_stages.csv`, against the
    two totals `docs/FAIRNESS_AND_COST.md` prints and the two the extensions'
    READMEs do;
  * and every quantity of the paragraph that the manuscript sets as a macro,
    from `make_numbers.build()` itself, so that the macro and the source agree
    rather than merely both existing.

WHY THE MACROS ARE CHECKED HERE AND NOT ONLY IN THE TEMPLATE. `numbers.tex` is
regenerated from the registry, so the registry is where a value is decided; a
template that still carries last month's number is a stale starting point and
not a wrong paper. Reading `build()` is reading the thing that will be printed.
"""

from __future__ import annotations

import ast
import csv
import glob
import inspect
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
FIGURES = TOOLS / "paper_figures"
SRC = REPO / "src" / "federated_outlier_adaptation"
STUDY = REPO / "study" / "artifacts" / "Digits_study01"
JOBS = STUDY / "jobs"
PAPER = STUDY / "tables" / "paper"
DOCS = REPO / "docs"

for entry in (str(TOOLS), str(FIGURES), str(REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


# --------------------------------------------------------------------- helpers
def _view(name: str) -> list:
    with open(PAPER / name, newline="") as handle:
        return list(csv.DictReader(handle))


def _recipe() -> dict:
    return {row["setting"]: row["value"] for row in _view("recipe.csv")}


@pytest.fixture(scope="module")
def macros() -> dict:
    """The registry `make_numbers.py` prints from, values only."""
    import make_numbers

    return {name: value for name, (value, _src) in make_numbers.build().items()}


@pytest.fixture(scope="module")
def task_lines() -> list:
    """Every federated task line the study shipped, as its flags."""
    lines = []
    for path in sorted(JOBS.glob("*.txt")):
        for raw in path.read_text().splitlines():
            if raw.startswith("foa final"):
                lines.append((path.name, raw.split()))
    assert lines, "no shipped federated task line under study/artifacts/*/jobs"
    return lines


def _flag(tokens: list, name: str):
    """The value of ``--name`` on one task line, or ``None``."""
    return tokens[tokens.index(name) + 1] if name in tokens else None


def _src_text() -> str:
    """Every line of `src/`, concatenated, comments and docstrings included."""
    return "\n".join(Path(p).read_text() for p in sorted(glob.glob(f"{SRC}/**/*.py",
                                                                  recursive=True)))


# ------------------------------------------------------------ the optimiser
def test_the_client_optimiser_is_adam_and_is_the_only_one():
    """
    Adam, named in the trainer that every client trains under.

    Read out of the syntax rather than by instantiating a trainer: constructing
    one builds a model and wants a provider, and the question here is which
    optimiser the source names, which is a property of the file.
    """
    from federated_outlier_adaptation.trainers import base_trainer

    source = inspect.getsource(base_trainer)
    built = set(re.findall(r"optim\.([A-Za-z]+)\(", source))
    assert built == {"Adam"}, built


def test_the_learning_rate_and_weight_decay_are_the_published_defaults():
    """The two constants, from the signature no shipped task line overrides."""
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    defaults = inspect.signature(BaseTrainer.__init__).parameters
    assert defaults["learning_rate"].default == pytest.approx(1e-3)
    assert defaults["weight_decay"].default == pytest.approx(1e-4)


def test_no_task_line_overrides_the_optimiser():
    """
    A default is only the recipe if nothing that ran passed something else.

    The CLI takes neither constant as a flag, so the check is that no shipped
    line spells one at all; a flag added later would fail here before it could
    quietly make the paragraph wrong.
    """
    for path in sorted(JOBS.glob("*.txt")):
        text = path.read_text()
        for flag in ("--learning-rate", "--lr ", "--weight-decay"):
            assert flag not in text, (path.name, flag)


def test_the_recipe_view_agrees_with_the_trainer(macros):
    """
    The payloads recorded what the defaults say, and the macros print it.

    Three readings of one recipe: the signature in `src/`, the view read back
    off seven thousand stored payloads, and the macro the manuscript sets.
    """
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    recipe = _recipe()
    defaults = inspect.signature(BaseTrainer.__init__).parameters
    assert float(recipe["learning_rate"]) == pytest.approx(
        defaults["learning_rate"].default)
    assert float(recipe["weight_decay"]) == pytest.approx(
        defaults["weight_decay"].default)
    assert macros["nLearningRate"] == "$10^{-3}$"
    assert macros["nWeightDecay"] == "$10^{-4}$"


def test_the_training_loop_has_no_schedule_no_amp_and_no_clipping():
    """Three things the paragraph says are absent, asserted as absent."""
    text = _src_text()
    for absent in ("lr_scheduler", "StepLR", "CosineAnnealing", "OneCycle",
                   "autocast", "GradScaler", "clip_grad_norm", "clip_grad_value"):
        assert absent not in text, absent


def test_there_is_no_augmentation_anywhere_in_the_transform():
    """
    The one transform every phase uses is resize, tensor, normalise.

    Asserted on the built pipeline rather than on the source, because an
    augmentation reaches the loader by being in this list and by nothing else.
    """
    from federated_outlier_adaptation.data.datasets import build_transform

    stages = [type(step).__name__ for step in build_transform(28).transforms]
    assert stages == ["Resize", "ToTensor", "Normalize"], stages


def test_the_client_keeps_its_best_validation_epoch_and_does_not_stop_early():
    """
    Both halves of one sentence: best-of-epochs is kept, the stopper is off.

    `train` selects on validation accuracy and restores the selected model at
    the end, and the early-stopping limit it passes the stopper is ``None``
    unless the trainer was constructed with the flag - which the default leaves
    off and which no federated task line sets.
    """
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    assert inspect.signature(BaseTrainer.__init__).parameters[
        "early_stopping"].default is False
    source = inspect.getsource(BaseTrainer.train)
    assert "stopper.best()" in source and "self.model = best" in source
    assert "patience=limit if self.early_stopping else None" in source
    assert int(_recipe()["client_early_stopping"]) == 0

    for path in sorted(JOBS.glob("*.txt")):
        for raw in path.read_text().splitlines():
            if raw.startswith("foa final"):
                assert "--early-stopping" not in raw, path.name


def test_the_optimiser_is_rebuilt_for_every_client_of_every_round():
    """
    No optimiser state crosses a client boundary or a round boundary.

    `set_model` constructs a fresh `optim.Adam`, and the round loop calls it
    once per participating client; a runner that hoisted the call out of that
    loop would carry one client's Adam moments into the next.
    """
    from federated_outlier_adaptation.runners import concurrent_runner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    assert "optim.Adam(" in inspect.getsource(BaseTrainer.set_model)

    def calls(node, attribute):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == attribute
                   for n in ast.walk(node))

    tree = ast.parse(inspect.getsource(concurrent_runner))
    loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
             and calls(n, "set_model") and calls(n, "train")
             and calls(n, "deepcopy")]
    assert loops, "set_model is no longer called inside the per-client loop"


# ------------------------------------------------------------------ the model
def test_the_model_is_the_published_topology_at_the_published_size(macros):
    """
    1,663,370 parameters, counted on the built model rather than quoted.

    The count is the whole specification: it is only reached with 5x5 "same"
    convolutions of 32 and 64 channels, two 2x2 poolings and a 512-unit fully
    connected layer over the 7x7x64 that leaves, and it is the number the
    paper, the docstring and `cost_arms.csv` all print.
    """
    from federated_outlier_adaptation.models.fedavg_cnn import (CONV_WIDTHS, FC_DIM,
                                                                INPUT_SIZE, FedAvgCNN)

    model = FedAvgCNN(num_classes=10, input_size=28, in_channels=1)
    assert (INPUT_SIZE, CONV_WIDTHS, FC_DIM) == (28, (32, 64), 512)
    assert sum(p.numel() for p in model.parameters()) == 1_663_370
    assert model.flatten_size == 7 * 7 * 64

    kinds = [type(step).__name__ for step in model.features]
    assert kinds == ["Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "MaxPool2d"]
    for step in model.features:
        if type(step).__name__ == "Conv2d":
            assert step.kernel_size == (5, 5) and step.padding == (2, 2)
        if type(step).__name__ == "MaxPool2d":
            assert step.kernel_size == 2 and step.stride == 2

    # No normalisation and no dropout: both interact badly with averaging, and
    # a layer added here would change every weight the aggregation rules move.
    assert model.dropout is None
    assert not any("Norm" in type(m).__name__ for m in model.modules())

    assert macros["nParamCount"] == "1\\,663\\,370"
    assert {row["param_count"] for row in _view("cost_arms.csv")} == {"1663370"}


# -------------------------------------------------------------- what ran, how
def test_every_task_line_runs_five_local_epochs_at_batch_sixty_four(macros, task_lines):
    assert {_flag(t, "--epochs") for _, t in task_lines} == {"5"}
    assert {_flag(t, "--batch-size") for _, t in task_lines} == {"64"}
    assert macros.get("nLocalEpochs", "5") == "5"
    assert _recipe()["local_epochs"] == "5" and _recipe()["batch_size"] == "64"


def test_the_two_horizons_are_a_hundred_rounds_and_twenty_five(macros, task_lines):
    """Nothing in between, and the finals horizon is the one the paper sets."""
    horizons = {_flag(t, "--rounds") for _, t in task_lines}
    assert horizons == {"100", "25"}, horizons
    assert macros["nRounds"] == "100"
    stages = _view("cost_stages.csv")
    assert {int(float(r["rounds"])) for r in stages} == {100, 25}


def test_eight_of_the_ten_clients_are_drawn_each_round(macros):
    """
    The search setting's participation, and the fraction the paper calls $d$.

    Only the ten-client cohort at eight per round is $d=0.2$; the carry setting
    draws nine of the same ten and the other cohorts are other sizes, so the
    lines are counted by their cohort rather than globbed together.
    """
    counts = defaultdict(set)
    for path in sorted(JOBS.glob("*.txt")):
        for raw in path.read_text().splitlines():
            if not raw.startswith("foa final"):
                continue
            tokens = raw.split()
            cohort = Path(_flag(tokens, "--outliers-file") or "").name
            counts[cohort].add(_flag(tokens, "--clients-per-round"))
    assert counts["cohort_worst10.json"] == {"8", "9"}, counts["cohort_worst10.json"]

    cohort_rows = [r for r in _view("../cohort_composition.csv")
                   if r["cohort"] == "cohort10"]
    assert len(cohort_rows) == 10
    combinations = next(r for r in _view("cost_stages.csv")
                        if r["stage"] == "combinations")
    assert int(float(combinations["clients_per_round"])) == 8
    assert macros["nDropoutFraction"] == "0.2"


def test_the_run_seed_is_the_fold_and_the_sampler_seed_is_not(task_lines):
    """
    One draw per fold, and a client sampler that does not share its stream.

    ``--seed k --fold k`` for the five folds is what makes a fold a complete
    description of a run: `seeding.fold_seed` mixes the two, so two folds of one
    configuration differ in their initialisation and their sampling and not only
    in their split. The sampler is seeded separately on every line, out of a
    block of its own per stage, so that two stages never draw the same
    participants in the same order.
    """
    for name, tokens in task_lines:
        fold, seed = _flag(tokens, "--fold"), _flag(tokens, "--seed")
        assert fold == seed and fold in ("1", "2", "3", "4", "5"), (name, fold, seed)

    blocks = defaultdict(set)
    seeds = defaultdict(set)
    for name, tokens in task_lines:
        sampler = _flag(tokens, "--sampler-seed")
        if sampler is None:
            continue
        blocks[name].add(int(sampler) // 1000)
        seeds[name].add(int(sampler))

    # The one stage that ships as two files, and therefore the one pair of
    # files allowed to draw on one block.  Anything else sharing a seed would
    # be two stages sampling identically without saying so.
    ONE_STAGE_TWO_FILES = {("s03_refs_c10.txt", "s03b_refs_fl.txt"),
                           ("s18_hybrid.txt", "s19_hybrid_seq.txt")}
    files = sorted(seeds)
    for i, left in enumerate(files):
        for right in files[i + 1:]:
            if (left, right) in ONE_STAGE_TWO_FILES:
                continue
            assert not (blocks[left] & blocks[right]), (left, right)


def test_cudnn_determinism_is_not_switched_on(task_lines):
    """
    Off by default and off in every run, which is why the paper says so.

    Determinism changes convolution algorithm selection and therefore
    throughput; the study bought the throughput and reports five folds instead.
    """
    from federated_outlier_adaptation.utils import seeding

    assert inspect.signature(seeding.set_run_seed).parameters[
        "deterministic"].default is False
    for path in sorted(JOBS.glob("*.txt")):
        assert "--deterministic" not in path.read_text(), path.name


# --------------------------------------------------------------- the detector
def test_the_detector_saw_every_writer_and_was_scored_on_held_out_rows():
    """
    Trained on all of them, ranked on the rows it did not train on.

    Scoring a writer on its own training rows would rank memorisation; the
    partition used is the one the fold the detector was selected from held out,
    which is what `HELDOUT_PARTS` names.
    """
    from federated_outlier_adaptation.outliers import scoring

    assert scoring.HELDOUT_PARTS == ("val", "test")
    lines = [raw for raw in (JOBS / "s01b_detector.txt").read_text().splitlines()
             if raw.startswith("foa global-train")]
    assert lines and all("--population all" in raw for raw in lines), lines

    pools = json.loads((STUDY / "outliers" / "pools.json").read_text())
    assert pools["ranked_writers"] == 3580
    assert pools["excluded_by_eligibility"] == 0


def test_the_cut_is_the_ceiling_of_thirty_percent_with_ties_by_writer_id(macros):
    """
    1,074 of 3,580, and the rule that produced it rather than the number.

    The ceiling matters: a floor at exactly 30% of 3,580 lands on the same
    integer here, so only the source says which of the two the study ran, and
    a corpus of a different size would separate them.
    """
    from federated_outlier_adaptation.outliers import scoring

    source = inspect.getsource(scoring)
    assert "math.ceil(float(bad_fraction) * len(ranked))" in source
    assert "key=lambda writer: (float(accuracies[writer]), writer)" in source

    pools = json.loads((STUDY / "outliers" / "pools.json").read_text())
    assert pools["bad_fraction"] == pytest.approx(0.30)
    assert pools["cut_rank"] == math.ceil(0.30 * pools["ranked_writers"]) == 1074
    assert pools["bad_count"] == 1074
    assert pools["bad_count"] + pools["good_count"] == pools["ranked_writers"]
    assert "ties by writer id" in pools["rule"]
    assert macros["nBadPoolSize"] == "1\\,074"


def test_g_zero_was_trained_on_two_hundred_writers_with_early_stopping():
    """
    The shipped model's own recipe, which is NOT the clients' recipe.

    g-0 is trained once per fold on a draw from the good pool and it does stop
    early - patience ten, never before twenty epochs. That is the one place in
    the study where early stopping runs, and stating it beside the federated
    recipe is what keeps the two from being read as one.
    """
    text = (JOBS / "s02_selection.txt").read_text()
    draw = [raw for raw in text.splitlines() if raw.startswith("foa draw-old-data")]
    assert len(draw) == 1 and "--size 200" in draw[0], draw
    assert "--min-samples 100" in draw[0]

    trains = [raw for raw in text.splitlines() if raw.startswith("foa global-train")]
    assert len(trains) == 5, len(trains)
    for raw in trains:
        tokens = raw.split()
        assert _flag(tokens, "--early-stopping-patience") == "10"
        assert _flag(tokens, "--min-epochs") == "20"
        assert _flag(tokens, "--batch-size") == "64"
        assert _flag(tokens, "--fold") == _flag(tokens, "--seed")


# ---------------------------------------------------------------- what it cost
def test_one_a100_per_task():
    """The hardware sentence, from the section that booked the hardware."""
    assert "One A100 per task" in (DOCS / "REPRODUCE.md").read_text()


def test_the_programme_is_two_thousand_seven_hundred_and_sixty_five_tasks(macros):
    """
    The two totals, summed from the stage view and printed by two documents.

    `FAIRNESS_AND_COST.md` states both at the foot of its own table, and the
    manuscript sets both as macros; all three are this sum, at two precisions.
    """
    stages = _view("cost_stages.csv")
    tasks = sum(int(row["tasks"]) for row in stages)
    hours = sum(float(row["hours"]) for row in stages)
    assert tasks == 2765
    assert round(hours, 2) == 56.81

    text = (DOCS / "FAIRNESS_AND_COST.md").read_text()
    assert "**2,765**" in text and "**56.81**" in text
    assert macros["nProgrammeTasks"] == "2\\,765"
    assert macros["nProgrammeGpuHours"] == "56.8"


def test_a_hundred_round_task_is_minutes_and_a_screen_task_is_about_one():
    """
    The wall-clock sentence, bracketed rather than quoted.

    `cost_stages.csv` times the ROUND LOOP only; the job READMEs price the rest
    of a task - interpreter, torch, the caches and the two books - at about
    forty seconds, and the paragraph's figures are the two added. The median
    hundred-round stage is asserted rather than every stage: the regularisation
    finals run two families per task and the extremes run one client, so the
    range is wider than the sentence and the sentence says "about".
    """
    START_UP = 40.0
    stages = _view("cost_stages.csv")
    full = sorted(float(r["seconds_per_task"]) for r in stages
                  if int(float(r["rounds"])) == 100)
    typical = full[len(full) // 2] + START_UP
    assert 1.5 * 60 <= typical <= 3 * 60, typical

    screen = next(r for r in stages if r["stage"] == "aggregation screen")
    assert 0.5 * 60 <= float(screen["seconds_per_task"]) + START_UP <= 1.5 * 60


def test_each_joint_tuning_extension_cost_about_nineteen_gpu_hours():
    """
    Both extensions, from the README of the stage that reports each screen.

    They are priced apart from the programme on purpose - nothing the paper
    reports reads them - so the figure the paper quotes beside the programme's
    total has to come from these two files and not from `cost_stages.csv`.
    """
    measured = {"s24_combo_full_README.md": 18.4,
                "s26_combo_full_selected_README.md": 18.6}
    for name, hours in measured.items():
        text = (JOBS / name).read_text()
        assert f"**{hours} GPU-h**" in text, name
        # The finals each README prices are "well under a GPU-hour", so the
        # extension's own total is the screen plus something under one.
        assert abs(19 - hours) <= 1.0


# ------------------------------------------- the control that arms a stop rule
def test_the_second_fedavg_control_armed_a_stop_rule_that_never_fired():
    """
    What `FedAvg (control, stop rule armed)` is, and why no run stopped early.

    The arm is plain FedAvg with the oracle stop rule switched on. The rule ends
    a run the round the global accuracy falls below the participating clients'
    own accuracy or below 0.90, and across both schedules and both horizons it
    fired on no fold: `cost_arms.csv` records a full hundred rounds for both
    control arms. So the row is a second FedAvg control at a different
    client-sampling seed, and the protocol's "no run stops early" holds.
    """
    from federated_outlier_adaptation.training import agg_cells

    controls = {cell["id"]: cell["flags"] for cell in agg_cells.control_cells()}
    assert set(controls) == {"control_fedavg", "control_fedavg_earlystop"}
    assert controls["control_fedavg"] == {}
    assert controls["control_fedavg_earlystop"] == {
        "stop_when_global_below_clients": True}

    lines = [raw.split() for path in sorted(JOBS.glob("*.txt"))
             for raw in path.read_text().splitlines()
             if raw.startswith("foa final") and "control_fedavg_earlystop" in raw]
    assert len(lines) == 10, len(lines)
    for tokens in lines:
        assert "--stop-when-global-below-clients" in tokens
        assert _flag(tokens, "--aggregation") == "fedavg"

    arms = {row["arm"]: row for row in _view("cost_arms.csv")}
    for arm in ("aggfull_control/fedavg", "aggfull_control/fedavg_earlystop"):
        assert int(float(arms[arm]["rounds"])) == 100, arm

    import paper_names

    assert "stop rule armed" in paper_names.label("control_fedavg_earlystop")


def test_the_name_the_tables_print_for_it_is_defined_where_it_is_printed():
    """A name a reader cannot decode is a footnote the table has to carry."""
    import make_paper_tables

    note = make_paper_tables.CONTROL_NOTE
    assert "never fired" in note and "second control" in note
    assert "full hundred rounds" in note
