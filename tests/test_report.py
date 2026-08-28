"""
The report layer: from stored runs to tables and figures.

The suite writes synthetic result trees in exactly the shape the pipeline
produces - a ``summary_<i>.json`` next to the per-job ``accuracies_<i>.json``, a
sweep's ``config_points_<i>.json``, the frozen reference artefacts, and a full
provenance ``config`` block on every entry - with trajectories whose metrics can
be computed by hand.  That way the assertions are about the arithmetic of the
report rather than about a number that happened to come out of a run.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

from federated_outlier_adaptation.analysis import report
from federated_outlier_adaptation.analysis import statistics as stats
from federated_outlier_adaptation.cli import main as cli_main

#: Long enough that the whole stability window falls inside the plateau.
ROUNDS = 16

#: Round by which a synthetic run has completed its adaptation, so the
#: rounds-to-target metric has a value strictly inside the budget.
PLATEAU = 4

#: Reference accuracies written into the frozen artefacts of the fixture.
SOURCE_TEST_ACC = 0.9963
COMBINED_SOURCE_ACC = 0.9957
COMBINED_CLIENTS_ACC = 0.9420

#: The shipped model on the source validation split.  It has no artefact - the
#: split is built with the run's own loader seed - so the report reads it off
#: round 0 of the runs that start from the shipped model.
SOURCE_VAL_ACC = 0.985

#: The shipped model on the participants' test data, i.e. round 0 of every
#: global-initialised run of the fixture's client set.
POOL_TEST_ACC = 0.50

#: The participants of the fixture, in the order the payload writes them.
PARTICIPANTS = ("c0", "c1", "c2", "c3")

#: Communication and timing of one synthetic round.
ROUND_SECONDS = 10.0
ROUND_BYTES = 1024 * 1024


def ramp(start: float, delta: float) -> list[float]:
    """A curve that moves by ``delta`` and is flat from :data:`PLATEAU` on."""
    return [start + delta * min(1.0, r / PLATEAU) for r in range(ROUNDS)]


def provenance_block(
    trainer: str,
    aggregation: str,
    scenario: str,
    metadata: str,
    seed: int,
    provider: str = "nist",
    hyperparameters: dict | None = None,
    population: dict | None = None,
    init: str = "global",
    participants: tuple[str, ...] = PARTICIPANTS,
) -> dict:
    """A ``config`` block with the keys ``utils/provenance`` really writes."""
    pool = {
        "participation": 1.0,
        "policy": "uniform",
        "clients_per_round": 10,
        "pool_frac": 0.05,
        "track_clients": True,
        **(population or {}),
    }
    return {
        "trainer": trainer,
        "trainer_kwargs": dict(hyperparameters or {}),
        "hyperparameters": {"learning_rate": 0.001, "weight_decay": 0.0001,
                            **(hyperparameters or {})},
        "scenario": scenario,
        "metadata": metadata,
        "agg_method_name": aggregation,
        "parent_name": None,
        "batch_size": 64,
        "epochs": 100,
        "max_round": ROUNDS,
        "seed": seed,
        "single_outlier": None,
        "outliers_file": "outliers/outlier_pool_frac0.05.json",
        "outlier_writers": list(participants),
        "global_model_path": "global_model",
        "global_model_sha256": "0" * 64,
        "fisher_dir": "global_results/fisher",
        "results_dir": "results",
        "data_dir": "data",
        "torch_version": "2.3.1",
        "cuda_version": "12.1",
        "device": "cuda",
        "gpu_name": "NVIDIA A100",
        "python_version": "3.12.0",
        "git_commit": "0123456789abcdef",
        "hostname": "node001",
        "started_at": "2025-01-01T00:00:00+00:00",
        "finished_at": "2025-01-01T01:00:00+00:00",
        "wall_seconds": ROUND_SECONDS * ROUNDS,
        "global_name": "global",
        "provider": provider,
        "init": init,
        "weighting": "proportional",
        "server_eta": 1.0,
        "client_order": "fixed",
        "stop_when_global_below_clients": False,
        **pool,
    }


def payload(
    trainer: str,
    aggregation: str,
    scenario: str,
    metadata: str,
    seed: int,
    forgetting: float,
    gain: float,
    provider: str = "nist",
    hyperparameters: dict | None = None,
    clients: dict | None = None,
    signals: bool = True,
    init: str = "global",
    start: tuple[float, float] | None = None,
    participants: tuple[str, ...] = PARTICIPANTS,
    population: dict | None = None,
) -> dict:
    """
    One stored run whose metrics follow from its arguments.

    ``forgetting`` and ``gain`` are fractions of accuracy: the run ends
    ``forgetting`` below and ``gain`` above where it started, both reached at
    :data:`PLATEAU`.  A run that starts from the shipped model starts at
    :data:`SOURCE_TEST_ACC` and :data:`POOL_TEST_ACC`, which is what makes the
    fixture's round 0 the shipped model itself; ``start`` overrides the pair for
    a run that begins somewhere else.
    """
    source_start, pool_start = start or (SOURCE_TEST_ACC, POOL_TEST_ACC)
    source = ramp(source_start, -forgetting)
    pool_test = ramp(pool_start, gain)
    pool_val = ramp(pool_start + 0.02, gain)
    heldout = ramp(pool_start + 0.01, gain)
    insample = ramp(pool_start + 0.10, gain)

    block: dict = {
        "scenario": scenario,
        "metadata": metadata,
        "agg_method_name": aggregation,
        "accuracies": [[insample[r], source[r]] for r in range(ROUNDS)],
        "global_clients_all_metrics_acc": COMBINED_CLIENTS_ACC,
        "global_clients_metric_acc": COMBINED_SOURCE_ACC,
        "round_seconds": [ROUND_SECONDS] * ROUNDS,
        "client_seconds": [[1.0] * 4] * ROUNDS,
        "comm_bytes_per_round": [ROUND_BYTES] * ROUNDS,
        "param_count": 1_234_567,
        "device": "cuda (NVIDIA A100)",
        "seed": seed,
        "heldout_client_accuracies": heldout,
        "pool_val_accuracies": pool_val,
        "pool_test_accuracies": pool_test,
        "source_val_accuracies": ramp(
            SOURCE_VAL_ACC if start is None else source_start - 0.005, -forgetting
        ),
        "config": provenance_block(
            trainer,
            aggregation,
            scenario,
            metadata,
            seed,
            provider,
            hyperparameters,
            population,
            init=init,
            participants=participants,
        ),
    }
    if clients:
        block["client_test_accuracies"] = [
            {
                name: begin + (end - begin) * min(1.0, r / PLATEAU)
                for name, (begin, end) in clients.items()
            }
            for r in range(ROUNDS)
        ]
        block["policy"] = "uniform"
        block["participation"] = 1.0
        block["pool_size"] = len(clients)
    if signals:
        block.update(
            {
                "dist_l2_to_global": [0.5 * r for r in range(ROUNDS)],
                "dist_fisher_to_global": [0.1 * r for r in range(ROUNDS)],
                "dist_fisher_norm_to_global": [0.01 * r for r in range(ROUNDS)],
                "retention_known": [1.0 - 0.02 * r for r in range(ROUNDS)],
                "agreement_with_global": [1.0 - 0.03 * r for r in range(ROUNDS)],
                "kl_global_to_current": [0.02 * r for r in range(ROUNDS)],
                "proxy_acc": [None] * ROUNDS,
                "proxy_kl": [None] * ROUNDS,
            }
        )
    return block


def write_final(root: Path, parent: str, trainer: str, seed: int, entries: dict, index: int = 0):
    """
    One final job: a summary plus the per-job accuracy files next to it.

    The driver keeps the round accuracies out of the summary and writes them
    into ``<experiment>/accuracies_<i>.json``, so the fixture does the same.
    """
    scenario, metadata = "concurrent", "weights"
    directory = root / f"{parent}_{trainer}_grid_search" / f"seed_{seed}" / f"{scenario}_{metadata}"
    directory.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, block in entries.items():
        job = directory / name
        job.mkdir(exist_ok=True)
        with open(job / f"accuracies_{index}.json", "w") as handle:
            json.dump(block, handle)
        summary[name] = {key: value for key, value in block.items() if key != "accuracies"}
        summary[name]["json_path"] = str(job / f"accuracies_{index}.json")
    with open(directory / f"summary_{index}.json", "w") as handle:
        json.dump(summary, handle)
    return directory


def write_references(
    root: Path, provider: str = "nist", participants: tuple[str, ...] = PARTICIPANTS
):
    """
    The frozen artefacts every reference point is read from.

    ``outliers/selected_outliers.json`` is part of them: it says which clients
    the combined model was scored on, which is what decides whether its accuracy
    is the head-room of a given arm.
    """
    base = root if provider == "nist" else root / provider
    (base / "global_results").mkdir(parents=True, exist_ok=True)
    (base / "global_clients_results").mkdir(parents=True, exist_ok=True)
    (base / "outliers").mkdir(parents=True, exist_ok=True)
    with open(base / "outliers" / "selected_outliers.json", "w") as handle:
        json.dump(list(participants), handle)
    for path, value in (
        (base / "global_results" / "global_metrics.json", SOURCE_TEST_ACC),
        (
            base / "global_clients_results" / "global_clients_global_metrics.json",
            COMBINED_SOURCE_ACC,
        ),
        (
            base / "global_clients_results" / "global_clients_all_outliers_metrics.json",
            COMBINED_CLIENTS_ACC,
        ),
    ):
        with open(path, "w") as handle:
            json.dump({"test_accuracy": value}, handle)


def experiment(aggregation: str, metadata: str = "weights", scenario: str = "concurrent") -> str:
    return f"base_agg_{aggregation}_{metadata}_{scenario}"


FEDAVG = "con_weighted_cw"
ANCHORED = "con_weighted_cgw"


@pytest.fixture()
def tree(tmp_path) -> Path:
    """
    One setting with three arms over three seeds, on a real trade-off curve.

    ``BaseTrainer`` under FedAvg adapts most and forgets most; the anchored
    aggregation rule is the conservative end; distillation sits between them.
    None of the three dominates another, so all three are on the Pareto front.
    Seed ``s`` shifts every curve by ``0.001 * s`` so the seeds actually differ.
    """
    root = tmp_path / "results"
    write_references(root)
    clients = {"c0": (0.30, 0.55), "c1": (0.50, 0.70), "c2": (0.60, 0.62), "c3": (0.70, 0.95)}
    for seed in (1, 2, 3):
        shift = 0.001 * seed
        write_final(
            root,
            "pool_m10_uniform_fedavg",
            "BaseTrainer",
            seed,
            {
                experiment(FEDAVG): payload(
                    "BaseTrainer", FEDAVG, "concurrent", "weights", seed,
                    forgetting=0.020 + shift, gain=0.30 + shift, clients=clients,
                ),
                experiment(ANCHORED): payload(
                    "BaseTrainer", ANCHORED, "concurrent", "weights", seed,
                    forgetting=0.004 + shift, gain=0.24 + shift, clients=clients,
                ),
            },
        )
        write_final(
            root,
            "pool_m10_uniform_fedavg",
            "DistillationTrainer",
            seed,
            {
                experiment(FEDAVG): payload(
                    "DistillationTrainer", FEDAVG, "concurrent", "weights", seed,
                    forgetting=0.010 + shift, gain=0.28 + shift,
                    hyperparameters={"T": 8.0, "alpha": 0.95}, clients=clients,
                ),
            },
        )
    return root


@pytest.fixture()
def tree_with_reference(tree) -> Path:
    """
    The standard fixture plus an arm that never saw the shipped model.

    It starts from random weights, so its round 0 is a model at chance level.
    Read against that round 0 it looks like the best arm of the tree by a wide
    margin - it "gains" 80 points and "forgets" nothing - while against the
    shipped model it is what it is: five points *below* the shipped model on
    the participants and ten points below it on the source data.
    """
    for seed in (1, 2, 3):
        write_final(
            tree,
            "pool_scratch_m10_uniform_fedavg",
            "BaseTrainer",
            seed,
            {
                experiment(FEDAVG): payload(
                    "BaseTrainer", FEDAVG, "concurrent", "weights", seed,
                    forgetting=-0.80, gain=0.80, init="scratch", start=(0.10, 0.05),
                ),
            },
        )
    return tree


@pytest.fixture()
def built(tree, tmp_path):
    """The report of the standard fixture, built once."""
    return report.build_report(tree, tmp_path / "report")


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def test_every_run_is_found_with_its_provenance(tree):
    references = report.References(tree)
    runs = report.collect_runs(tree, references)
    assert len(runs) == 3 * 3
    assert {run.trainer for run in runs} == {"BaseTrainer", "DistillationTrainer"}
    assert {run.aggregation for run in runs} == {FEDAVG, ANCHORED}
    assert {run.seed for run in runs} == {1, 2, 3}
    assert all(run.provider == "nist" for run in runs)
    assert all(run.parent.endswith("_grid_search") for run in runs)
    # The round accuracies live in the per-job file and are merged back in.
    assert all(len(run.series["source_test"]) == ROUNDS for run in runs)


def test_a_job_file_without_its_summary_is_still_read(tree):
    directory = tree / "pool_m10_uniform_fedavg_BaseTrainer_grid_search" / "seed_1" / "concurrent_weights"
    (directory / "summary_0.json").unlink()
    references = report.References(tree)
    runs = report.collect_runs(tree, references)
    assert len(runs) == 3 * 3
    orphans = [run for run in runs if run.path.endswith("accuracies_0.json")]
    assert len(orphans) == 2


def test_a_summary_above_its_job_folder_is_not_counted_twice(tmp_path):
    """
    ``provider_adaptation`` writes the summary one level above the job file.

    Both layouts have to be recognised: an unrecognised job file would be read
    a second time as an orphan and its run would enter every table twice.
    """
    write_references(tmp_path)
    parent = tmp_path / "adaptation_BaseTrainer_grid_search"
    job = parent / "concurrent_weights" / experiment(FEDAVG)
    job.mkdir(parents=True)
    block = payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1, forgetting=0.01, gain=0.1)
    with open(job / "accuracies_0.json", "w") as handle:
        json.dump(block, handle)
    with open(parent / "summary_0.json", "w") as handle:
        json.dump({experiment(FEDAVG): {**block, "json_path": str(job / "accuracies_0.json")}}, handle)

    runs = report.collect_runs(tmp_path, report.References(tmp_path))
    assert len(runs) == 1
    assert runs[0].parent == "adaptation_BaseTrainer_grid_search"
    assert len(runs[0].series["source_test"]) == ROUNDS


def test_a_sweep_is_read_from_its_config_block(tmp_path):
    write_references(tmp_path)
    directory = tmp_path / "anchored_kd_frozen_grid_search" / "concurrent_weights"
    directory.mkdir(parents=True)
    block = payload("AnchoredTrainer", FEDAVG, "concurrent", "weights", 1,
                    forgetting=0.01, gain=0.2, hyperparameters={"lam": 10, "space": "kd"})
    accuracies = block.pop("accuracies")
    config = block.pop("config")
    # A sweep stores only ``population`` here; the timings live in the
    # provenance block under ``instrumentation``.
    config["instrumentation"] = {
        "round_seconds": block.pop("round_seconds"),
        "client_seconds": block.pop("client_seconds"),
        "comm_bytes_per_round": block.pop("comm_bytes_per_round"),
        "param_count": block.pop("param_count"),
    }
    with open(directory / "accuracies_points_100.json", "w") as handle:
        json.dump({"cfg": accuracies}, handle)
    with open(directory / "config_points_100.json", "w") as handle:
        json.dump({"config": {"cfg": {**config, "population": block}}}, handle)

    runs = report.collect_runs(tmp_path, report.References(tmp_path))
    assert len(runs) == 1
    run = runs[0]
    assert run.kind == "sweep"
    assert run.parent == "anchored_kd_frozen_grid_search"
    assert run.trainer == "AnchoredTrainer"
    assert run.metrics["mean_round_seconds"] == pytest.approx(ROUND_SECONDS)
    assert run.metrics["mean_comm_bytes"] == pytest.approx(1.0)


def test_the_dataset_filter_keeps_one_provider(tmp_path):
    write_references(tmp_path)
    write_references(tmp_path, "shakespeare")
    write_final(tmp_path, "p", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                             forgetting=0.01, gain=0.1)})
    write_final(tmp_path / "shakespeare", "p", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                             forgetting=0.01, gain=0.1, provider="shakespeare")})
    references = report.References(tmp_path)
    assert len(report.collect_runs(tmp_path, references)) == 2
    only = report.collect_runs(tmp_path, references, datasets=["shakespeare"])
    assert [run.provider for run in only] == ["shakespeare"]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_forgetting_and_gain_are_points_against_the_shipped_model(tree):
    runs = {
        (run.trainer, run.aggregation, run.seed): run
        for run in report.collect_runs(tree, report.References(tree))
    }
    run = runs[("BaseTrainer", FEDAVG, 1)]
    assert run.metrics["forgetting"] == pytest.approx((0.02 + 0.001) * 100)
    assert run.metrics["gain"] == pytest.approx((0.30 + 0.001) * 100)
    assert run.metrics["a_new_test"] == pytest.approx(POOL_TEST_ACC + 0.301)
    assert run.metrics["a_src"] == pytest.approx(SOURCE_TEST_ACC - 0.021)
    # The references are the shipped model, not the run's own starting point -
    # here they coincide, because the run starts from the shipped model.
    assert run.metrics["a_src_ref"] == pytest.approx(SOURCE_TEST_ACC)
    assert run.metrics["a_new_test_ref"] == pytest.approx(POOL_TEST_ACC)
    assert run.metrics["round0_source_test"] == pytest.approx(run.metrics["a_src_ref"])
    assert run.metrics["round0_pool_test"] == pytest.approx(run.metrics["a_new_test_ref"])
    # The curve only ever falls, so the worst round is the last one.
    assert run.metrics["worst_forgetting"] == pytest.approx(run.metrics["forgetting"])


def test_the_normalised_gain_uses_the_combined_model_head_room(tree):
    run = next(
        run
        for run in report.collect_runs(tree, report.References(tree))
        if run.trainer == "BaseTrainer" and run.aggregation == FEDAVG and run.seed == 1
    )
    head_room = COMBINED_CLIENTS_ACC - POOL_TEST_ACC
    assert run.metrics["normalised_gain"] == pytest.approx(0.301 / head_room)


def test_no_head_room_when_the_combined_model_covers_other_clients(tmp_path):
    """
    The combined model is scored on the *selected* participants of a provider.

    An arm that adapts to a different set has no head-room to divide by, so the
    report leaves the normalised gain unset rather than dividing by a number
    measured on other data.
    """
    write_references(tmp_path)
    others = ("d0", "d1", "d2")
    write_final(tmp_path, "pool", "BaseTrainer", 1, {
        experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                    forgetting=0.02, gain=0.3, participants=others),
    })
    write_final(tmp_path, "same", "BaseTrainer", 1, {
        experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                    forgetting=0.02, gain=0.3),
    })
    runs = {
        run.parent: run for run in report.collect_runs(tmp_path, report.References(tmp_path))
    }
    elsewhere = runs["pool_BaseTrainer_grid_search"]
    matching = runs["same_BaseTrainer_grid_search"]
    assert elsewhere.metrics["gain"] == pytest.approx(30.0)
    assert elsewhere.metrics["normalised_gain"] is None
    assert matching.metrics["normalised_gain"] is not None


def test_rounds_to_target_finds_the_plateau(tree):
    run = next(iter(report.collect_runs(tree, report.References(tree))))
    assert run.metrics["rounds_to_target"] == PLATEAU
    # The curve is flat from the plateau on, so it is stable at the end.
    assert run.metrics["last_window_std"] == pytest.approx(0.0, abs=1e-9)


def test_cost_is_read_from_the_instrumentation(tree):
    run = next(iter(report.collect_runs(tree, report.References(tree))))
    assert run.metrics["mean_round_seconds"] == pytest.approx(ROUND_SECONDS)
    assert run.metrics["total_wall_seconds"] == pytest.approx(ROUND_SECONDS * ROUNDS)
    assert run.metrics["mean_comm_bytes"] == pytest.approx(1.0)
    assert run.metrics["comm_bytes_to_target"] == pytest.approx(float(PLATEAU))


def test_fairness_summarises_the_per_client_accuracies(tree):
    run = next(iter(report.collect_runs(tree, report.References(tree))))
    # Final accuracies 0.55, 0.70, 0.62, 0.95; the worst decile of four clients
    # is the single worst one, and three of the four improved.
    assert run.metrics["fairness_mean"] == pytest.approx((0.55 + 0.70 + 0.62 + 0.95) / 4)
    assert run.metrics["fairness_worst_decile"] == pytest.approx(0.55)
    assert run.metrics["fairness_improved"] == pytest.approx(1.0)
    assert run.metrics["fairness_clients"] == 4


def test_a_run_without_per_client_series_has_no_fairness(tmp_path):
    write_references(tmp_path)
    write_final(tmp_path, "p", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                             forgetting=0.01, gain=0.1)})
    run = next(iter(report.collect_runs(tmp_path, report.References(tmp_path))))
    assert "fairness_mean" not in run.metrics


def test_signal_drift_is_signed_towards_forgetting(tree):
    run = next(iter(report.collect_runs(tree, report.References(tree))))
    assert run.metrics["drift_dist_l2_to_global"] == pytest.approx(0.5 * (ROUNDS - 1))
    # An accuracy-like signal drifts by its *drop*, so the sign is the same.
    assert run.metrics["drift_agreement_with_global"] == pytest.approx(0.03 * (ROUNDS - 1))
    assert run.metrics["drift_proxy_acc"] is None


# --------------------------------------------------------------------------- #
# arms and intervals
# --------------------------------------------------------------------------- #
def test_arms_group_every_seed_of_one_configuration(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    assert len(arms) == 3
    assert all(len(arm.runs) == 3 for arm in arms)
    assert {arm.label for arm in arms} == {
        f"BaseTrainer / {FEDAVG}",
        f"BaseTrainer / {ANCHORED}",
        f"DistillationTrainer / {FEDAVG} / T=8.0,alpha=0.95",
    }


def test_the_interval_is_the_t_interval_of_the_seeds(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    arm = next(arm for arm in arms if arm.is_baseline)
    summary = arm.summary("gain")
    values = [(0.30 + 0.001 * seed) * 100 for seed in (1, 2, 3)]
    expected = stats.summarise(values)
    assert summary.n == 3
    assert summary.mean == pytest.approx(expected.mean)
    assert summary.ci == pytest.approx(expected.ci)


def test_the_baseline_of_a_setting_is_basetrainer_under_fedavg(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    baseline = next(arm for arm in arms if arm.is_baseline)
    assert report.baseline_for(baseline, arms) is None
    for arm in arms:
        if arm is baseline:
            continue
        assert report.baseline_for(arm, arms) is baseline


def test_a_rule_is_compared_against_fedavg_under_the_same_objective(tmp_path):
    """
    The baseline follows the axis: an aggregation rule is read against FedAvg
    with the *same* trainer, and a FedAvg arm against plain FedAvg.
    """
    write_references(tmp_path)
    hyper = {"T": 8.0, "alpha": 0.95}
    for seed in (1, 2):
        write_final(tmp_path, "pool", "BaseTrainer", seed,
                    {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights",
                                                 seed, forgetting=0.02, gain=0.30)})
        write_final(tmp_path, "pool", "DistillationTrainer", seed, {
            experiment(FEDAVG): payload("DistillationTrainer", FEDAVG, "concurrent", "weights",
                                        seed, forgetting=0.01, gain=0.28, hyperparameters=hyper),
            experiment(ANCHORED): payload("DistillationTrainer", ANCHORED, "concurrent", "weights",
                                          seed, forgetting=0.005, gain=0.24, hyperparameters=hyper),
        })
    arms = {
        arm.label: arm
        for arm in report.group_arms(report.collect_runs(tmp_path, report.References(tmp_path)))
    }
    distil_fedavg = arms[f"DistillationTrainer / {FEDAVG} / T=8.0,alpha=0.95"]
    distil_anchored = arms[f"DistillationTrainer / {ANCHORED} / T=8.0,alpha=0.95"]
    base = arms[f"BaseTrainer / {FEDAVG}"]

    every = list(arms.values())
    assert report.baseline_for(distil_anchored, every) is distil_fedavg
    assert report.baseline_for(distil_fedavg, every) is base
    assert report.baseline_for(base, every) is None


def test_an_arm_without_a_baseline_is_skipped(tmp_path):
    write_references(tmp_path)
    for seed in (1, 2, 3):
        write_final(tmp_path, "solo", "DistillationTrainer", seed,
                    {experiment(FEDAVG): payload("DistillationTrainer", FEDAVG, "concurrent",
                                                 "weights", seed, forgetting=0.01, gain=0.1)})
    arms = report.group_arms(report.collect_runs(tmp_path, report.References(tmp_path)))
    assert report.comparison_table(arms) == []


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
def test_the_comparison_is_seed_matched_against_the_baseline(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    rows = report.comparison_table(arms)
    row = next(
        r for r in rows if r["arm"].startswith("DistillationTrainer") and r["metric"] == "forgetting"
    )
    assert row["n_pairs"] == 3
    assert row["seeds"] == [1, 2, 3]
    # Every seed forgets exactly 1.0 point less than the baseline, so the
    # difference carries no spread and the paired test is certain.
    assert row["difference"] == pytest.approx(-1.0)
    assert row["p_t"] == pytest.approx(0.0)
    assert row["marker"] == "***"
    assert row["baseline"] == f"BaseTrainer / {FEDAVG}"


def test_the_holm_correction_is_applied_within_each_metric(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    rows = report.comparison_table(arms)
    for metric in report.COMPARED_METRICS:
        family = [row for row in rows if row["metric"] == metric]
        assert family
        for row in family:
            assert row["p_t_holm"] >= row["p_t"] - 1e-12
            assert row["p_wilcoxon_holm"] >= row["p_wilcoxon"] - 1e-12


def test_a_single_shared_seed_is_not_compared(tmp_path):
    write_references(tmp_path)
    write_final(tmp_path, "pool", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                             forgetting=0.02, gain=0.3)})
    write_final(tmp_path, "pool", "DistillationTrainer", 1,
                {experiment(FEDAVG): payload("DistillationTrainer", FEDAVG, "concurrent",
                                             "weights", 1, forgetting=0.01, gain=0.2)})
    arms = report.group_arms(report.collect_runs(tmp_path, report.References(tmp_path)))
    assert report.comparison_table(arms) == []


# --------------------------------------------------------------------------- #
# Pareto and budgets
# --------------------------------------------------------------------------- #
def test_the_pareto_front_drops_the_dominated_points():
    points = [
        {"arm": "a", "forgetting": 1.0, "gain": 30.0},
        {"arm": "b", "forgetting": 0.5, "gain": 26.0},
        {"arm": "c", "forgetting": 2.0, "gain": 25.0},  # dominated by a
        {"arm": "d", "forgetting": None, "gain": 40.0},  # incomplete
    ]
    front = [point["arm"] for point in report.pareto_front(points)]
    assert front == ["b", "a"]


def test_the_front_of_the_fixture_is_the_three_arms(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    front = report.pareto_front(report.arm_points(arms))
    assert len(front) == 3
    forgetting = [point["forgetting"] for point in front]
    assert forgetting == sorted(forgetting)


def test_the_budget_selects_on_validation_and_reports_test(tree):
    arms = report.group_arms(report.collect_runs(tree, report.References(tree)))
    points = report.arm_points(arms)
    rows = report.budget_table(points, budgets=(0.25, 0.5, 1.0, 5.0))
    base = {row["budget_pt"]: row for row in rows if row["trainer"] == "BaseTrainer"}

    # The two BaseTrainer arms forget 0.6 and 2.2 points on validation, so the
    # 0.25 budget admits nothing, 1.0 admits the anchored rule alone, and the
    # loosest budget lets the more adaptive FedAvg arm win.
    assert base[0.25]["selected"] is None and base[0.25]["num_feasible"] == 0
    assert base[1.0]["num_feasible"] == 1
    assert base[1.0]["selected"] == f"BaseTrainer / {ANCHORED}"
    assert base[1.0]["a_new_test"] == pytest.approx(0.742)
    assert base[5.0]["num_feasible"] == 2
    assert base[5.0]["selected"] == f"BaseTrainer / {FEDAVG}"
    assert base[5.0]["num_candidates"] == 2

    distil = {row["budget_pt"]: row for row in rows if row["trainer"] == "DistillationTrainer"}
    # Distillation forgets 1.2 points on validation: outside every tight budget.
    assert distil[1.0]["selected"] is None
    assert distil[5.0]["num_feasible"] == 1


# --------------------------------------------------------------------------- #
# method groups: a family is ranked member by member
# --------------------------------------------------------------------------- #
ANCHORED_MEMBERS = (
    ("fisher_scaled", "current", 10, 4.0),
    ("kd", "frozen", 0.1, 8.0),
    ("param_l2", "frozen", 1.0, None),
)


def anchored_tree(root: Path) -> Path:
    """
    One anchored family of three members, each with two strengths.

    The members differ in what their penalty is - the space and the anchor - and
    within a member ``lam`` only says how hard it pulls.  ``fisher_scaled/current``
    adapts furthest and forgets least, ``param_l2/frozen`` least and most, so a
    table that collapses them into one ``AnchoredTrainer`` row reports the family
    by ``fisher_scaled/current`` alone and hides the other two.
    """
    write_references(root)
    for index, (space, anchor, lam, temperature) in enumerate(ANCHORED_MEMBERS):
        for seed in (1, 2, 3):
            entries = {}
            for step, strength in enumerate((lam, lam * 10)):
                hyper = {"space": space, "anchor": anchor, "lam": strength, "mix": 0.5}
                if temperature is not None:
                    hyper["T"] = temperature
                entries[experiment(FEDAVG) + f"_{step}"] = payload(
                    "AnchoredTrainer", FEDAVG, "concurrent", "weights", seed,
                    forgetting=0.001 + 0.001 * index + 0.0005 * step,
                    gain=0.30 - 0.05 * index - 0.02 * step,
                    hyperparameters=hyper,
                )
            write_final(root, f"reg_{space}_{anchor}", "AnchoredTrainer", seed, entries)
    for seed in (1, 2, 3):
        write_final(root, "pool_m10_uniform_fedavg", "BaseTrainer", seed, {
            experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights",
                                        seed, forgetting=0.002, gain=0.10),
        })
    return root


def test_a_family_trainer_is_one_group_per_member():
    assert report.method_group("BaseTrainer", {}) == "BaseTrainer"
    assert report.method_group("DistillationTrainer", {"T": 8.0}) == "DistillationTrainer"
    assert (
        report.method_group("AnchoredTrainer", {"space": "kd", "anchor": "frozen", "lam": 1})
        == "anchored:kd/frozen"
    )
    # Nothing to split on: the trainer stands for itself rather than for a
    # group whose name would be half missing.
    assert report.method_group("AnchoredTrainer", {"lam": 1}) == "AnchoredTrainer"


def test_the_budget_table_selects_each_family_member(tmp_path):
    root = anchored_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    groups = {row["group"] for row in manifest["budget_selection"]}
    assert groups == {
        "BaseTrainer",
        "anchored:fisher_scaled/current",
        "anchored:kd/frozen",
        "anchored:param_l2/frozen",
    }
    for row in manifest["budget_selection"]:
        assert row["num_candidates"] == (1 if row["group"] == "BaseTrainer" else 2)
        assert row["trainer"] == (
            "BaseTrainer" if row["group"] == "BaseTrainer" else "AnchoredTrainer"
        )


def test_the_regularisation_table_is_one_row_per_space_and_anchor(tmp_path):
    root = anchored_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    rows = manifest["regularisation_family"]
    assert [row["group"] for row in rows] == [
        # Ordered by the adaptation reached inside the budget.
        "anchored:fisher_scaled/current",
        "anchored:kd/frozen",
        "anchored:param_l2/frozen",
    ]
    assert [(row["space"], row["anchor"]) for row in rows] == [
        ("fisher_scaled", "current"),
        ("kd", "frozen"),
        ("param_l2", "frozen"),
    ]
    first = rows[0]
    assert first["budget_pt"] == report.REGULARISATION_BUDGET
    assert first["num_candidates"] == 2
    # The weaker penalty adapts furthest, so that is the strength reported.
    assert first["lam"] == 10
    assert first["T"] == 4.0
    assert first["a_new_test"] == pytest.approx(POOL_TEST_ACC + 0.30)
    assert first["gain"] == pytest.approx(30.0)
    assert first["n"] == 3
    assert first["a_new_test_ci"] is not None
    assert first["on_setting_front"] is True
    # A penalty without a temperature reports none rather than a stray number.
    assert rows[-1]["T"] is None
    assert rows[-1]["lam"] == 1.0
    # Only the split family is in this table; the plain trainers are not.
    assert all(row["group"].startswith("anchored:") for row in rows)


def test_the_regularisation_table_is_written_as_a_float(tmp_path):
    root = anchored_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    table = next(t for t in manifest["tables"] if t["name"] == "regularisation_family")
    text = Path(table["tex"]).read_text()
    assert "anchored:kd/frozen" in text.replace("\\_", "_")
    with open(table["csv"]) as handle:
        rows = list(csv.DictReader(handle))
    assert [row["group"] for row in rows] == [row["group"] for row in manifest["regularisation_family"]]
    assert set(rows[0]) == set(report.REGULARISATION_COLUMNS)


def test_a_member_with_nothing_inside_the_budget_keeps_its_row(tmp_path):
    root = tmp_path / "results"
    write_references(root)
    for seed in (1, 2, 3):
        write_final(root, "reg_kd_frozen", "AnchoredTrainer", seed, {
            experiment(FEDAVG): payload(
                "AnchoredTrainer", FEDAVG, "concurrent", "weights", seed,
                forgetting=0.05, gain=0.30,
                hyperparameters={"space": "kd", "anchor": "frozen", "lam": 1, "T": 8.0},
            ),
        })
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    rows = manifest["regularisation_family"]
    assert len(rows) == 1
    assert rows[0]["group"] == "anchored:kd/frozen"
    assert rows[0]["selected"] is None and rows[0]["num_feasible"] == 0
    assert rows[0]["on_setting_front"] is False
    assert rows[0]["a_new_test"] is None


def test_the_group_reaches_every_table_that_ranks_methods(tmp_path):
    root = anchored_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    by_name = {table["name"]: table for table in manifest["tables"]}
    for name in ("runs", "arms", "pareto", "paired_vs_fedavg", "budget_selection_all"):
        with open(by_name[name]["csv"]) as handle:
            rows = list(csv.DictReader(handle))
        assert rows, name
        assert "group" in rows[0], name
        assert any(row["group"].startswith("anchored:") for row in rows), name
    # The front of the setting is reported per group, not per class.
    assert all(point["group"] for point in manifest["pareto_front"])
    assert any(
        point["group"].startswith("anchored:") for point in manifest["pareto_front"]
    )


def test_the_trade_off_figure_draws_every_group(tmp_path):
    """Eight palette slots, more groups: none of them may be dropped."""
    root = anchored_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report")
    figure = next(
        item for item in manifest["figures"] if item["path"].endswith("pareto_families.pdf")
    )
    assert "method group" in figure["caption"]
    assert Path(figure["path"]).read_bytes().startswith(b"%PDF")

    points = [
        {"forgetting": float(index), "gain": float(index), "group": f"g{index}"}
        for index in range(len(report.PALETTE) * 2 + 1)
    ]
    path = report.figure_pareto(points, tmp_path / "many.pdf", group="group", title="many")
    assert path is not None and path.is_file()


# --------------------------------------------------------------------------- #
# the reference of F and G
# --------------------------------------------------------------------------- #
def test_a_run_that_did_not_start_from_the_shipped_model_is_read_against_it(
    tree_with_reference,
):
    """
    ``F`` and ``G`` are the provider's loss and gain, not the run's own journey.

    The scratch run ends at 0.90 on the source data and at 0.85 on the
    participants.  Against its own round 0 that reads as 80 points gained and
    80 points of negative forgetting; against the shipped model it is a 9.63
    point drop on the source data and a 35 point rise on the participants.
    """
    runs = report.collect_runs(tree_with_reference, report.References(tree_with_reference))
    scratch = [run for run in runs if run.init == "scratch"]
    assert len(scratch) == 3
    for run in scratch:
        assert run.metrics["a_src"] == pytest.approx(0.90)
        assert run.metrics["a_new_test"] == pytest.approx(0.85)
        # Its own round 0 is kept, and is not what F and G are read against.
        assert run.metrics["round0_source_test"] == pytest.approx(0.10)
        assert run.metrics["round0_pool_test"] == pytest.approx(0.05)
        assert run.metrics["a_src_ref"] == pytest.approx(SOURCE_TEST_ACC)
        assert run.metrics["a_new_test_ref"] == pytest.approx(POOL_TEST_ACC)
        assert run.metrics["forgetting"] == pytest.approx((SOURCE_TEST_ACC - 0.90) * 100)
        assert run.metrics["gain"] == pytest.approx((0.85 - POOL_TEST_ACC) * 100)
        assert run.metrics["forgetting_val"] == pytest.approx((SOURCE_VAL_ACC - 0.895) * 100)


def test_the_shipped_reference_leaves_the_global_runs_untouched(tree_with_reference):
    """A run that starts from the shipped model reads exactly its own round 0."""
    runs = report.collect_runs(tree_with_reference, report.References(tree_with_reference))
    for run in runs:
        if run.init != "global":
            continue
        assert run.metrics["a_src_ref"] == pytest.approx(run.metrics["round0_source_test"])
        assert run.metrics["a_new_test_ref"] == pytest.approx(run.metrics["round0_pool_test"])
        assert run.metrics["forgetting"] == pytest.approx(
            (run.metrics["round0_source_test"] - run.metrics["a_src"]) * 100
        )
        assert run.metrics["gain"] == pytest.approx(
            (run.metrics["a_new_test"] - run.metrics["round0_pool_test"]) * 100
        )


def test_a_run_without_the_shipped_model_is_marked_as_a_reference(tree_with_reference):
    runs = report.collect_runs(tree_with_reference, report.References(tree_with_reference))
    for run in runs:
        assert run.is_reference == (run.init == "scratch")
        assert (run.kind == report.REFERENCE_KIND) == (run.init == "scratch")


def test_a_locally_fine_tuned_run_is_a_reference_too(tmp_path):
    """The other model the study reports without ever ranking it."""
    write_references(tmp_path)
    write_final(tmp_path, "local_finetune", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent",
                                             "weights", 1, forgetting=0.02, gain=0.3)})
    run = next(iter(report.collect_runs(tmp_path, report.References(tmp_path))))
    assert run.kind == report.REFERENCE_KIND


def test_a_reference_arm_takes_part_in_no_ranking(tree_with_reference, tmp_path):
    """
    The scratch arm is the most adaptive arm of the tree and wins nothing.

    Its ``A_new`` is the highest of the fixture and, read against its own round
    0, its validation forgetting is far inside every budget - so a table that
    let it compete would hand it the whole budget column.
    """
    manifest = report.build_report(
        tree_with_reference, tmp_path / "report", figures=False
    )
    arms = report.group_arms(
        report.collect_runs(tree_with_reference, report.References(tree_with_reference))
    )
    scratch = next(arm for arm in arms if arm.is_reference)
    assert scratch.summary("a_new_test").mean == pytest.approx(0.85)

    selected = {row["selected"] for row in manifest["budget_selection"]}
    assert selected
    for row in manifest["budget_selection"]:
        assert row["a_new_test"] is None or row["a_new_test"] < 0.85
    assert all("init scratch" not in (row["setting"] or "") for row in manifest["budget_selection"])
    assert all(point["arm"] for point in manifest["pareto_front"])
    assert all(
        "init scratch" not in point["setting"] for point in manifest["pareto_front"]
    )
    assert all(
        "init scratch" not in row["setting"] for row in manifest["comparisons"]
    )
    # It is a reference point, so that is where it is reported - with the
    # numbers read against the shipped model.
    row = next(
        row for row in manifest["references"] if "scratch" in str(row["reference"])
    )
    assert row["value"] == pytest.approx(0.85)
    assert row["gain"] == pytest.approx((0.85 - POOL_TEST_ACC) * 100)
    assert row["forgetting"] == pytest.approx((SOURCE_TEST_ACC - 0.90) * 100)
    assert any("reference points" in note for note in manifest["notes"])


def test_the_shipped_pool_accuracy_ignores_the_reference_runs(tree_with_reference):
    """A scratch run must not move the reference every other run is read against."""
    shipped = report.ShippedModel(
        report.collect_runs(tree_with_reference, report.References(tree_with_reference)),
        report.References(tree_with_reference),
    )
    run = next(
        run
        for run in report.collect_runs(tree_with_reference, report.References(tree_with_reference))
        if run.init == "scratch"
    )
    theta = shipped.for_run(run)
    assert theta.pool_test == pytest.approx(POOL_TEST_ACC)
    assert theta.n == 9
    assert theta.spread == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# client sets and settings
# --------------------------------------------------------------------------- #
def test_two_client_sets_are_two_settings(tmp_path):
    """
    Same parent, same trainer, same rule, same seed - other participants.

    Several studies of the tree run one grid over the selected writers and the
    same grid over the five-per-cent pool.  Nothing but the participant list
    tells them apart, so without it their seeds would be pooled into one arm and
    their numbers compared as if they described the same data.
    """
    write_references(tmp_path)
    for index, participants in enumerate((PARTICIPANTS, ("d0", "d1"))):
        write_final(
            tmp_path, "grid", "BaseTrainer", 1,
            {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights",
                                         1, forgetting=0.02, gain=0.3,
                                         participants=participants)},
            index=index,
        )
    arms = report.group_arms(report.collect_runs(tmp_path, report.References(tmp_path)))
    assert len(arms) == 2
    assert len({arm.setting for arm in arms}) == 2
    labels = sorted(report.setting_label(arm.setting) for arm in arms)
    assert labels[0].endswith("w2") and labels[1].endswith("w4")
    # Neither is the other's baseline: they are not the same experiment.
    assert all(report.baseline_for(arm, arms) is None for arm in arms)


def test_the_setting_label_names_the_initialisation(tree_with_reference):
    arms = report.group_arms(
        report.collect_runs(tree_with_reference, report.References(tree_with_reference))
    )
    labels = {report.setting_label(arm.setting) for arm in arms}
    assert any(label.endswith("init scratch") for label in labels)
    assert any(label.endswith("w4") for label in labels)


# --------------------------------------------------------------------------- #
# one setting per headline table
# --------------------------------------------------------------------------- #
def two_setting_tree(tmp_path: Path) -> Path:
    """One trainer per setting, so a cross-setting table mixes the two."""
    write_references(tmp_path)
    for seed in (1, 2, 3):
        write_final(tmp_path, "pool_m10_uniform_fedavg", "BaseTrainer", seed,
                    {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent",
                                                 "weights", seed, forgetting=0.002,
                                                 gain=0.10)})
        write_final(tmp_path, "pool_m5_worst_first_fedavg", "DistillationTrainer", seed,
                    {experiment(FEDAVG): payload("DistillationTrainer", FEDAVG,
                                                 "concurrent", "weights", seed,
                                                 forgetting=0.002, gain=0.30,
                                                 population={"clients_per_round": 5,
                                                             "policy": "worst_first"})})
    return tmp_path


def test_the_headline_budget_table_is_one_setting(tmp_path):
    root = two_setting_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    assert manifest["setting"] == report.MAIN_SETTING
    trainers = {row["trainer"] for row in manifest["budget_selection"]}
    assert trainers == {"BaseTrainer"}
    for row in manifest["budget_selection"]:
        assert row["setting"] is None or report.MAIN_SETTING in row["setting"]

    # The cross-setting selection is still written, and it holds both.
    path = next(
        table["csv"] for table in manifest["tables"]
        if table["name"] == "budget_selection_all"
    )
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["trainer"] for row in rows} == {"BaseTrainer", "DistillationTrainer"}


def test_the_pareto_float_is_one_setting_and_the_csv_is_all_of_them(tmp_path):
    root = two_setting_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False)
    assert {point["trainer"] for point in manifest["pareto_front"]} == {"BaseTrainer"}
    path = next(
        table["csv"] for table in manifest["tables"] if table["name"] == "pareto"
    )
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["trainer"] for row in rows} == {"BaseTrainer", "DistillationTrainer"}
    assert {row["on_setting_front"] for row in rows} == {"True", "False"}


def test_an_explicit_setting_selects_the_other_one(tmp_path):
    root = two_setting_tree(tmp_path / "results")
    manifest = report.build_report(
        root, tmp_path / "report", figures=False, setting="m5 worst_first"
    )
    assert {row["trainer"] for row in manifest["budget_selection"]} == {
        "DistillationTrainer"
    }


def test_an_empty_setting_keeps_every_setting_in_the_headline_table(tmp_path):
    root = two_setting_tree(tmp_path / "results")
    manifest = report.build_report(root, tmp_path / "report", figures=False, setting="")
    assert {row["trainer"] for row in manifest["budget_selection"]} == {
        "BaseTrainer",
        "DistillationTrainer",
    }


def test_a_setting_nothing_matches_is_reported_as_a_note(tmp_path):
    root = two_setting_tree(tmp_path / "results")
    manifest = report.build_report(
        root, tmp_path / "report", figures=False, setting="pool0.5 m40 nowhere"
    )
    assert manifest["budget_selection"] == []
    assert any("headline setting" in note for note in manifest["notes"])



# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def test_latex_escaping_protects_the_data_and_leaves_raw_cells_alone():
    assert report.latex_escape("pool_m10 & 50%") == r"pool\_m10 \& 50\%"
    assert report.latex_escape(None) == "--"
    assert report.latex_escape(report.Raw("$\\pm$")) == "$\\pm$"


def latex_body_rows(text: str) -> list[str]:
    """The data rows of a generated table, between midrule and bottomrule."""
    body = text.split(r"\midrule")[1].split(r"\bottomrule")[0]
    return [line for line in (l.strip() for l in body.splitlines()) if line]


def separators(row: str) -> int:
    """Column separators of a LaTeX row, i.e. the ampersands that are not escaped."""
    return len(re.findall(r"(?<!\\)&", row))


def assert_balanced(text: str) -> int:
    """Every row of a table has exactly one cell per declared column."""
    spec = re.search(r"\\begin\{tabular\}\{([^}]*)\}", text).group(1)
    columns = len(spec)
    assert columns >= 1
    header = text.split(r"\toprule")[1].split(r"\midrule")[0].strip()
    assert separators(header) == columns - 1, header
    for row in latex_body_rows(text):
        assert row.endswith(r"\\")
        assert separators(row) == columns - 1, row
    return columns


def test_a_generated_table_is_a_balanced_booktabs_float():
    text = report.latex_table(
        [{"a": "x_1", "b": 1}, {"a": "y", "b": 2}],
        columns=("a", "b"),
        headers=("Name", "$n$"),
        caption="Caption",
        label="tab:demo",
        notes="A note.",
    )
    assert r"\begin{table}[t]" in text and r"\end{table}" in text
    assert r"\label{tab:demo}" in text
    assert r"x\_1" in text
    assert assert_balanced(text) == 2


def test_an_empty_table_still_compiles():
    text = report.latex_table([], ("a", "b", "c"), None, "Empty", "tab:empty")
    assert assert_balanced(text) == 3
    assert latex_body_rows(text) == [r"-- & -- & -- \\"]


def test_every_generated_tex_table_is_balanced(built):
    for table in built["tables"]:
        if "tex" not in table:
            continue
        assert_balanced(Path(table["tex"]).read_text())


# --------------------------------------------------------------------------- #
# the whole report
# --------------------------------------------------------------------------- #
EXPECTED_TABLES = {
    "references",
    "runs",
    "arms",
    "paired_vs_fedavg",
    "pareto",
    "budget_selection",
    "budget_selection_all",
    "regularisation_family",
    "cost",
    "fairness",
    "signals",
}

#: Tables that are data rather than floats, so they are written as CSV only.
CSV_ONLY_TABLES = {"runs", "budget_selection_all"}


def test_every_table_is_written(built):
    names = {table["name"] for table in built["tables"]}
    assert names == EXPECTED_TABLES
    for table in built["tables"]:
        assert Path(table["csv"]).is_file()
        if table["name"] not in CSV_ONLY_TABLES:
            assert Path(table["tex"]).is_file()
    # The per-run dump and the cross-setting selection are data, not floats.
    for name in CSV_ONLY_TABLES:
        assert "tex" not in next(t for t in built["tables"] if t["name"] == name)


def test_the_run_dump_holds_one_row_per_run(built):
    path = next(t["csv"] for t in built["tables"] if t["name"] == "runs")
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9
    assert {row["trainer"] for row in rows} == {"BaseTrainer", "DistillationTrainer"}
    assert all(row["forgetting"] for row in rows)


def test_the_figures_are_vector_pdfs(built):
    assert built["figures"]
    for figure in built["figures"]:
        path = Path(figure["path"])
        assert path.suffix == ".pdf"
        assert path.is_file() and path.stat().st_size > 0
        assert path.read_bytes().startswith(b"%PDF")


def test_the_index_summarises_the_numbers(built):
    text = (Path(built["out_dir"]) / "index.md").read_text()
    assert "# Experiment report" in text
    assert "Pareto front" in text
    assert "Adaptation under a forgetting budget" in text
    assert "Paired comparisons against the FedAvg baseline" in text
    assert f"{SOURCE_TEST_ACC:.4f}" in text
    for table in built["tables"]:
        assert table["name"] in text
    for figure in built["figures"]:
        assert Path(figure["path"]).name in text


def test_the_manifest_is_stored_next_to_the_index(built):
    with open(Path(built["out_dir"]) / "report.json") as handle:
        stored = json.load(handle)
    assert stored["num_runs"] == built["num_runs"]
    assert stored["datasets"] == ["nist"]


def test_partial_results_are_reported_as_notes(tmp_path):
    write_references(tmp_path)
    write_final(tmp_path, "pool", "BaseTrainer", 1,
                {experiment(FEDAVG): payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                                             forgetting=0.02, gain=0.3)})
    manifest = report.build_report(tmp_path, tmp_path / "report")
    assert manifest["num_runs"] == 1
    assert any("single seed" in note for note in manifest["notes"])
    assert any("No paired comparison" in note for note in manifest["notes"])
    assert any("fairness" in note for note in manifest["notes"])


def test_an_empty_root_produces_an_empty_report(tmp_path):
    manifest = report.build_report(tmp_path, tmp_path / "report", figures=False)
    assert manifest["num_runs"] == 0
    assert manifest["figures"] == []
    assert any("No run JSON" in note for note in manifest["notes"])
    assert (Path(manifest["out_dir"]) / "index.md").is_file()
    for table in manifest["tables"]:
        assert Path(table["csv"]).is_file()


def test_a_tree_written_before_the_signals_existed_is_read(tmp_path):
    write_references(tmp_path)
    block = payload("BaseTrainer", FEDAVG, "concurrent", "weights", 1,
                    forgetting=0.02, gain=0.3, signals=False)
    write_final(tmp_path, "pool", "BaseTrainer", 1, {experiment(FEDAVG): block})
    manifest = report.build_report(tmp_path, tmp_path / "report", figures=False)
    assert manifest["num_runs"] == 1
    assert manifest["signals"] == []
    assert any("signals" in note for note in manifest["notes"])


def test_a_stored_signal_analysis_is_reused(tree, tmp_path):
    (tree / "signals").mkdir(exist_ok=True)
    with open(tree / "signals" / "signals_summary.json", "w") as handle:
        json.dump(
            {
                "correlations": [
                    {"run": "__pooled__", "method": "__all__", "signal": "proxy_acc",
                     "target": "source_val", "rounds": 99, "pearson": 0.5, "spearman": 0.6},
                ]
            },
            handle,
        )
    manifest = report.build_report(tree, tmp_path / "report2", figures=False)
    assert [row["signal"] for row in manifest["signals"]] == ["proxy_acc"]
    assert manifest["signals"][0]["source"] == "foa signals"


# --------------------------------------------------------------------------- #
# the CLI
# --------------------------------------------------------------------------- #
def test_the_cli_builds_the_report(tree, tmp_path, capsys):
    out = tmp_path / "cli_report"
    assert cli_main(["report", "--root", str(tree), "--out", str(out), "--no-figures"]) == 0
    payload_out = json.loads(capsys.readouterr().out)
    assert payload_out["num_runs"] == 9
    assert payload_out["figures"] == 0
    assert payload_out["out_dir"] == str(out)
    assert (out / "tables" / "arms.tex").is_file()
    assert (out / "index.md").is_file()


def test_the_cli_defaults_the_report_directory_into_the_root(tree, capsys):
    assert cli_main(["report", "--root", str(tree), "--no-figures"]) == 0
    assert (tree / "report" / "index.md").is_file()
    capsys.readouterr()


def test_the_cli_defaults_to_the_headline_setting(tree, tmp_path, capsys):
    from federated_outlier_adaptation import cli

    # The default is mirrored in the parser so ``--help`` needs no import.
    assert cli.MAIN_SETTING == report.MAIN_SETTING

    out = tmp_path / "setting_report"
    assert cli_main(["report", "--root", str(tree), "--out", str(out), "--no-figures"]) == 0
    payload_out = json.loads(capsys.readouterr().out)
    assert payload_out["setting"] == report.MAIN_SETTING
    with open(out / "tables" / "budget_selection.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(
        not row["setting"] or report.MAIN_SETTING in row["setting"] for row in rows
    )
    assert (out / "tables" / "budget_selection_all.csv").is_file()


def test_the_cli_takes_a_setting_filter(tree, tmp_path, capsys):
    out = tmp_path / "other_setting"
    assert (
        cli_main(
            ["report", "--root", str(tree), "--out", str(out), "--no-figures",
             "--setting", "m5 worst_first"]
        )
        == 0
    )
    capsys.readouterr()
    with open(out / "tables" / "budget_selection.csv") as handle:
        assert list(csv.DictReader(handle)) == []
    with open(out / "tables" / "budget_selection_all.csv") as handle:
        assert list(csv.DictReader(handle))


def test_the_cli_accepts_custom_budgets(tree, tmp_path, capsys):
    out = tmp_path / "budget_report"
    assert (
        cli_main(
            ["report", "--root", str(tree), "--out", str(out), "--no-figures",
             "--budgets", "0.5", "3.0"]
        )
        == 0
    )
    capsys.readouterr()
    with open(out / "tables" / "budget_selection.csv") as handle:
        budgets = {row["budget_pt"] for row in csv.DictReader(handle)}
    assert budgets == {"0.5", "3.0"}
