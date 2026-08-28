"""
Constraint-respecting forgetting signals: the tracker and the analysis.

Two halves.  The first drives both runners against the synthetic provider and
checks that the signal series appear, are aligned with ``accuracies``, and stay
inert for a provider without a proxy set.  The second runs the analysis over
*synthetic trajectories* written to disk in the two stored shapes, so the
correlations, the simulated stopping rules and the constrained selection are
tested against trajectories whose answer is known by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.analysis import forgetting_signals as analysis
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.forgetting_signals import (
    SIGNAL_KEYS,
    ForgettingSignalTracker,
    build_tracker,
    compare_to_reference,
    empty_info,
    float_state,
    load_fisher,
)
from federated_outlier_adaptation.runners.forgetting_signals import (
    reference_predictions as _reference_predictions,
)
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

BATCH_SIZE = 4
EPOCHS = 1
MAX_ROUND = 2

#: Signals that need no data at all and are therefore always present.
PARAMETER_SIGNALS = (
    "dist_l2_to_global",
    "dist_fisher_to_global",
    "dist_fisher_norm_to_global",
)


def simulate(provider, scenario="concurrent", agg_name="con_weighted_cgw", **kwargs):
    trainer = BaseTrainer(provider=provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider, seed=11, **kwargs)
    agg_method = getattr(select_class(scenario, "weights"), agg_name)
    accuracies, *_ = runner.simulate(
        exp_name="test_signals",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    return runner, accuracies


# --------------------------------------------------------------------------- #
# the tracker inside the runners
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scenario,agg_name",
    [("concurrent", "con_weighted_cgw"), ("sequential", "seq_fedavg_update")],
)
def test_every_signal_is_logged_once_per_round(synthetic_provider, scenario, agg_name):
    runner, accuracies = simulate(synthetic_provider, scenario=scenario, agg_name=agg_name)
    info = runner.population_info()
    for key in SIGNAL_KEYS:
        assert key in info
        assert len(info[key]) == len(accuracies), key


def test_parameter_distances_start_at_zero_and_grow(synthetic_provider):
    runner, _ = simulate(synthetic_provider)
    info = runner.population_info()
    for key in PARAMETER_SIGNALS:
        series = info[key]
        assert series[0] == pytest.approx(0.0, abs=1e-9), key
        assert all(value >= 0.0 for value in series), key
    assert info["dist_l2_to_global"][-1] > 0.0


def test_round_zero_matches_the_reference_model(synthetic_provider):
    """Before any client trains, the current model *is* the reference model."""
    runner, _ = simulate(synthetic_provider)
    info = runner.population_info()
    assert info["agreement_with_global"][0] == pytest.approx(1.0)
    assert info["kl_global_to_current"][0] == pytest.approx(0.0, abs=1e-6)
    # Retention is defined on the samples the reference got right; an untrained
    # model on random data may get none, in which case there is nothing to
    # retain and the signal is reported as unavailable rather than as 1.0.
    retention = info["retention_known"][0]
    assert retention is None or retention == pytest.approx(1.0)


def test_client_signals_stay_in_range(synthetic_provider):
    runner, _ = simulate(synthetic_provider)
    info = runner.population_info()
    for key in ("retention_known", "agreement_with_global"):
        assert all(0.0 <= v <= 1.0 for v in info[key] if v is not None), key
    assert all(value >= -1e-6 for value in info["kl_global_to_current"])


def test_retention_counts_only_what_the_reference_got_right():
    """
    The three client-side signals, on predictions chosen by hand.

    Four samples with labels ``[0, 1, 0, 1]``.  The reference is right on three
    of them (0, 1, 3); the current model is right on two of those (0, 3) and
    agrees with the reference on three of the four.
    """
    from torch import nn

    class TableModel(nn.Module):
        """Returns the row of a fixed logit table addressed by the input."""

        def __init__(self, predictions):
            super().__init__()
            table = torch.full((len(predictions), 2), -4.0)
            for row, klass in enumerate(predictions):
                table[row, klass] = 4.0
            self.register_buffer("table", table)

        def forward(self, x):
            return self.table[x.long().reshape(-1)]

    labels = torch.tensor([0, 1, 0, 1])
    rows = torch.arange(4).float().unsqueeze(1)
    loader = DataLoader(TensorDataset(rows, labels), batch_size=3, shuffle=False)

    reference_model = TableModel([0, 1, 1, 1])
    reference = _reference_predictions(reference_model, loader, "cpu")
    assert reference.correct.tolist() == [True, True, False, True]

    scores = compare_to_reference(TableModel([0, 0, 1, 1]), loader, reference, "cpu")
    assert scores["accuracy"] == pytest.approx(0.5)
    assert scores["retention_known"] == pytest.approx(2 / 3)
    assert scores["agreement"] == pytest.approx(0.75)
    assert scores["kl"] > 0.0
    # Against itself the same model is a perfect match.
    same = compare_to_reference(reference_model, loader, reference, "cpu")
    assert same["retention_known"] == pytest.approx(1.0)
    assert same["agreement"] == pytest.approx(1.0)
    assert same["kl"] == pytest.approx(0.0, abs=1e-6)


def test_provider_without_a_proxy_set_reports_none(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider)
    info = runner.population_info()
    assert info["proxy_acc"] == [None] * len(accuracies)
    assert info["proxy_kl"] == [None] * len(accuracies)
    assert info["signals_info"]["proxy_samples"] == 0
    assert info["signals_info"]["proxy_set"] == {}


def test_signals_describe_their_own_provenance(synthetic_provider):
    runner, _ = simulate(synthetic_provider)
    block = runner.population_info()["signals_info"]
    assert block["fisher_available"] is True
    assert block["fisher_dir"].endswith("fisher")
    assert block["heldout_samples"] > 0


def test_empty_info_has_the_same_keys():
    payload = empty_info()
    assert set(payload) == {*SIGNAL_KEYS, "signals_info"}


# --------------------------------------------------------------------------- #
# a provider that does define a proxy set
# --------------------------------------------------------------------------- #
class ProxyProvider:
    """Thin wrapper adding a fixed proxy set to the synthetic provider."""

    def __init__(self, inner, samples: int = 8):
        self._inner = inner
        generator = torch.Generator().manual_seed(5)
        images = torch.rand(samples, 1, 128, 128, generator=generator)
        labels = torch.arange(samples) % inner.num_classes
        self._dataset = TensorDataset(images, labels)

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def proxy_loader(self, batch_size: int = 64):
        return DataLoader(self._dataset, batch_size=batch_size, shuffle=False)

    def proxy_info(self) -> dict:
        return {"name": "unit_proxy", "size": len(self._dataset), "hash": "deadbeef"}


def test_proxy_signals_are_logged_when_a_proxy_exists(synthetic_provider):
    provider = ProxyProvider(synthetic_provider)
    runner, accuracies = simulate(provider)
    info = runner.population_info()
    assert len(info["proxy_acc"]) == len(accuracies)
    assert all(0.0 <= value <= 1.0 for value in info["proxy_acc"])
    assert info["proxy_kl"][0] == pytest.approx(0.0, abs=1e-6)
    assert info["signals_info"]["proxy_samples"] == 8
    assert info["signals_info"]["proxy_set"]["hash"] == "deadbeef"


def test_tracker_is_deterministic_across_two_passes(synthetic_provider):
    """The cached reference must not drift: the same model gives the same row."""
    provider = ProxyProvider(synthetic_provider)
    model = provider.make_model()
    tracker = build_tracker(provider, model, device="cpu", batch_size=BATCH_SIZE)
    first = tracker.record(model)
    second = tracker.record(model)
    for key, value in first.items():
        if value is None:
            assert second[key] is None
        else:
            assert second[key] == pytest.approx(value)


def test_fisher_distance_is_disabled_without_the_artefact(synthetic_provider, tmp_path):
    tracker = ForgettingSignalTracker(
        provider=synthetic_provider,
        reference_state=float_state(synthetic_provider.make_model()),
        reference_model=synthetic_provider.make_model(),
        device="cpu",
        fisher_dir=tmp_path / "missing",
        batch_size=BATCH_SIZE,
    )
    assert load_fisher(tmp_path / "missing") is None
    values = tracker.parameter_distances(synthetic_provider.make_model())
    assert values["dist_fisher_to_global"] is None
    assert values["dist_fisher_norm_to_global"] is None
    assert values["dist_l2_to_global"] >= 0.0


# --------------------------------------------------------------------------- #
# the analysis over synthetic trajectories
# --------------------------------------------------------------------------- #
ROUNDS = 12


def make_block(scale: float = 1.0, adaptation_gain: float = 0.03) -> dict:
    """
    A trajectory in which every signal is an exact monotone function of the
    round, so the correlations must come out at +/-1 and the stopping rules must
    fire at a round that can be computed by hand.

    ``scale`` stretches the whole run: a run with a tenth of the drift forgets a
    tenth as much *and* moves its signals a tenth as far, which is the regime
    where a signal-based budget has to prefer it.
    """
    source = [0.9 - 0.02 * scale * r for r in range(ROUNDS)]
    heldout = [0.5 + adaptation_gain * r for r in range(ROUNDS)]
    return {
        "accuracies": [[heldout[r], source[r]] for r in range(ROUNDS)],
        "heldout_client_accuracies": heldout,
        "source_val_accuracies": source,
        "dist_l2_to_global": [0.5 * scale * r for r in range(ROUNDS)],
        "dist_fisher_to_global": [0.1 * scale * r for r in range(ROUNDS)],
        "dist_fisher_norm_to_global": [0.01 * scale * r for r in range(ROUNDS)],
        "retention_known": [1.0 - 0.03 * scale * r for r in range(ROUNDS)],
        "agreement_with_global": [1.0 - 0.04 * scale * r for r in range(ROUNDS)],
        "kl_global_to_current": [0.02 * scale * r for r in range(ROUNDS)],
        "proxy_acc": [0.95 - 0.01 * scale * r for r in range(ROUNDS)],
        "proxy_kl": [0.03 * scale * r for r in range(ROUNDS)],
        "config": {"trainer": "BaseTrainer", "scenario": "concurrent", "metadata": "weights"},
    }


@pytest.fixture
def results_root(tmp_path) -> Path:
    """
    Two stored runs, one in each shape the pipeline writes.

    ``base_agg_a`` is a fast run stored as a final summary; ``cfg_b`` is a run
    that drifts a fortieth as fast, stored the way a sweep stores it.
    """
    final_dir = tmp_path / "unit_final_BaseTrainer_grid_search" / "concurrent_weights"
    final_dir.mkdir(parents=True)
    fast = make_block(scale=1.0)
    # The driver writes the round accuracies into the per-job file only, so the
    # fixture reproduces that split faithfully.
    job_dir = final_dir / "base_agg_a"
    job_dir.mkdir()
    with open(job_dir / "accuracies_100.json", "w") as handle:
        json.dump({"accuracies": fast.pop("accuracies")}, handle)
    with open(final_dir / "summary_100.json", "w") as handle:
        json.dump({"base_agg_a": fast}, handle)

    grid_dir = tmp_path / "unit_grid_search" / "concurrent_weights"
    grid_dir.mkdir(parents=True)
    slow = make_block(scale=0.025, adaptation_gain=0.01)
    accuracies = slow.pop("accuracies")
    config = slow.pop("config")
    with open(grid_dir / "accuracies_points_100.json", "w") as handle:
        json.dump({"cfg_b": accuracies}, handle)
    with open(grid_dir / "config_points_100.json", "w") as handle:
        json.dump({"config": {"cfg_b": {**config, "population": slow}}}, handle)
    return tmp_path


def test_correlation_helpers_agree_with_the_textbook_cases():
    assert analysis.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert analysis.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    # Monotone but not linear: Spearman is exact where Pearson is not.
    assert analysis.spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert analysis.pearson([1, 2, 3, 4], [1, 4, 9, 16]) < 1.0
    assert analysis.pearson([1, 1, 1], [1, 2, 3]) is None
    assert analysis.spearman([1.0], [2.0]) is None
    assert analysis.rank([10, 20, 20, 30]) == [1.0, 2.5, 2.5, 4.0]


def test_collect_reads_both_stored_shapes(results_root):
    runs = analysis.collect_trajectories(results_root)
    assert len(runs) == 2
    assert {run.method for run in runs} == {"BaseTrainer_concurrent_weights"}
    assert all(run.usable for run in runs)
    assert all(run.rounds == ROUNDS for run in runs)
    # The round accuracies of a final run live in the per-job file, not in the
    # summary; without them the source *test* target would be empty.
    assert all(len(run.source_test) == ROUNDS for run in runs)
    assert all(run.forgetting(on="test")[-1] is not None for run in runs)


def test_a_summary_without_its_job_file_still_reads(results_root):
    """Only the source test target is lost; everything else survives."""
    job = results_root / "unit_final_BaseTrainer_grid_search" / "concurrent_weights"
    (job / "base_agg_a" / "accuracies_100.json").unlink()
    runs = {run.name: run for run in analysis.collect_trajectories(results_root)}
    orphan = next(run for name, run in runs.items() if "base_agg_a" in name)
    assert orphan.source_test == []
    assert orphan.rounds == ROUNDS
    assert orphan.usable


def test_signals_correlate_perfectly_with_forgetting(results_root):
    rows = analysis.correlation_table(analysis.collect_trajectories(results_root))
    per_run = [row for row in rows if row["run"] != "__pooled__"]
    assert per_run
    for row in per_run:
        expected = analysis.SIGNAL_DIRECTION[row["signal"]]
        # Each signal is an affine function of the round and so is forgetting,
        # so both coefficients are exactly +/-1 within one run.
        assert row["spearman"] == pytest.approx(expected), row["signal"]
        assert row["pearson"] == pytest.approx(expected), row["signal"]

    pooled = {(r["signal"], r["target"]): r for r in rows if r["run"] == "__pooled__"}
    assert set(pooled) == {
        (signal, target)
        for signal in SIGNAL_KEYS
        for target in analysis.FORGETTING_TARGETS
    }
    for (signal, target), row in pooled.items():
        # Pooling two runs with different drifts breaks the exact tie, but the
        # sign of the relationship must survive, on either definition of
        # forgetting.
        assert row["spearman"] * analysis.SIGNAL_DIRECTION[signal] > 0.5, (signal, target)
        assert row["rounds"] == 2 * ROUNDS


def test_drift_is_non_negative_and_starts_at_zero(results_root):
    run = analysis.collect_trajectories(results_root)[0]
    for signal in SIGNAL_KEYS:
        drift = run.drift(signal)
        assert drift[0] == pytest.approx(0.0)
        assert all(value >= -1e-12 for value in drift), signal


def test_stopping_rule_fires_at_the_first_round_above_the_budget():
    # agreement falls by 0.04 per round, so a budget of 0.1 is first exceeded
    # at round 3 (drift 0.12).
    drift = [0.0, 0.04, 0.08, 0.12, 0.16]
    assert analysis.stop_round(drift, 0.1) == 3
    assert analysis.stop_round(drift, 0.5) == 4  # never exceeded: the fixed budget
    assert analysis.stop_round([0.0], 0.1) == 0


def test_stopping_table_compares_against_both_references(results_root):
    runs = analysis.collect_trajectories(results_root)
    rows = analysis.stopping_table(runs, deltas=(0.05, 0.2))
    assert rows
    for row in rows:
        assert row["fixed_round"] == row["rounds"] - 1
        # Source validation only ever falls here, so the oracle keeps round 0.
        assert row["oracle_round"] == 0
        assert 0 <= row["stop_round"] <= row["fixed_round"]
        assert row["adaptation_gap"] == pytest.approx(
            row["oracle_adaptation"] - row["adaptation"]
        )
    tight = [r for r in rows if r["signal"] == "agreement_with_global" and r["delta"] == 0.05]
    loose = [r for r in rows if r["signal"] == "agreement_with_global" and r["delta"] == 0.2]
    assert min(r["stop_round"] for r in tight) <= min(r["stop_round"] for r in loose)


def test_selection_prefers_the_configuration_within_the_budget(results_root):
    runs = analysis.collect_trajectories(results_root)
    rows = analysis.selection_table(runs, deltas=(0.02,), eps=0.005)
    row = next(r for r in rows if r["signal"] == "proxy_acc")
    # Only the slowly drifting sweep run stays inside both budgets, so the
    # signal picks it and matches the oracle exactly.
    assert row["num_allowed"] == 1
    assert row["selected"] is not None
    assert row["oracle"] == row["selected"]
    assert row["adaptation_gap"] == pytest.approx(0.0)


def test_analyse_writes_every_table(results_root):
    summary = analysis.analyse(results_root, deltas=(0.05,), plots=True)
    out_dir = Path(summary["out_dir"])
    assert out_dir == results_root / "signals"
    for name in (
        "signal_correlations.csv",
        "signal_stopping.csv",
        "signal_selection.csv",
        "signals_summary.json",
    ):
        assert (out_dir / name).is_file(), name
    assert summary["num_usable_runs"] == 2
    assert summary["plots"]
    assert all(Path(path).is_file() for path in summary["plots"])


def test_analyse_survives_a_tree_without_signals(tmp_path):
    """A results root produced before the signals existed must not raise."""
    legacy = tmp_path / "legacy_BaseTrainer_grid_search" / "concurrent_weights"
    legacy.mkdir(parents=True)
    with open(legacy / "summary_100.json", "w") as handle:
        json.dump({"base_agg_a": {"accuracies": [[0.5, 0.9], [0.6, 0.88]]}}, handle)

    summary = analysis.analyse(tmp_path, plots=False)
    assert summary["num_runs"] == 0
    assert summary["correlations"] == []
    assert summary["stopping"] == []
    assert (Path(summary["out_dir"]) / "signal_correlations.csv").is_file()
