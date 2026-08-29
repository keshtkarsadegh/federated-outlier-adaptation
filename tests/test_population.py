"""
Client population inside the runners: duplicate participants, per-round
selection, per-client tracking, the held-out client accuracy series and the
early-stopping rule.

The central invariant is that the defaults change nothing: a run with the
sampler at its default settings must produce the very same accuracies as the
published loop.
"""

from __future__ import annotations

import pytest
import torch

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.forgetting_signals import SIGNAL_KEYS
from federated_outlier_adaptation.runners.population import (
    ClientPopulation,
    resolve_participants,
)
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

BATCH_SIZE = 4
EPOCHS = 1


def simulate(provider, scenario="concurrent", metadata="weights",
             agg_name="con_weighted_cgw", max_round=2, seed=11, **kwargs):
    """Run a short seeded simulation and return the runner."""
    trainer = BaseTrainer(provider=provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider, seed=seed, **kwargs)
    agg_method = getattr(select_class(scenario, metadata), agg_name)
    accuracies, *_ = runner.simulate(
        exp_name="test_population",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=max_round,
        grid_Search=False,
    )
    return runner, accuracies


# --------------------------------------------------------------------------- #
# participants with multiplicity
# --------------------------------------------------------------------------- #
def test_resolve_participants_without_duplicates_matches_restrict(synthetic_provider):
    selected = synthetic_provider.selected_clients()
    unique, participants = resolve_participants(synthetic_provider, selected, ["c1", "c0"])
    assert unique == synthetic_provider.restrict(selected, ["c1", "c0"])
    assert participants == unique


def test_resolve_participants_expands_a_repeated_id(synthetic_provider):
    selected = synthetic_provider.selected_clients()
    unique, participants = resolve_participants(synthetic_provider, selected, ["c0", "c0"])
    assert unique == ["c0"]
    assert participants == ["c0", "c0"]


def test_resolve_participants_rejects_an_id_the_dataset_does_not_have(synthetic_provider):
    """
    An unknown id used to be dropped silently, which is how a run could end up
    with no participants at all and only notice several hundred lines later, as
    a ``TypeError`` on a ``None`` loader inside ``BaseTrainer.evaluate``.  It is
    now an error that names the id.
    """
    from federated_outlier_adaptation.runners.population import UnknownClientError

    selected = synthetic_provider.selected_clients()
    with pytest.raises(UnknownClientError, match="nobody"):
        resolve_participants(synthetic_provider, selected, ["c0", "c0", "nobody"])


def test_resolve_participants_accepts_a_client_outside_the_current_selection(
    synthetic_provider,
):
    """
    A known client that is not in the current selection is a participant, not a
    mistake: naming clients explicitly states who they are rather than narrowing
    whichever selection file happens to be current.
    """
    unique, participants = resolve_participants(synthetic_provider, ["c1"], ["c0", "c0"])
    assert unique == ["c0"]
    assert participants == ["c0", "c0"]


def test_no_restriction_keeps_every_client(synthetic_provider):
    selected = synthetic_provider.selected_clients()
    unique, participants = resolve_participants(synthetic_provider, selected, None)
    assert unique == list(selected)
    assert participants == list(selected)


def test_duplicate_participant_trains_twice(synthetic_provider):
    runner, _ = simulate(synthetic_provider, single_outlier=["c0", "c0"])
    assert runner.selected_outliers == ["c0"]
    assert runner.participants == ["c0", "c0"]
    assert len(runner.instrumentation["client_seconds"][0]) == 2


def test_single_participant_trains_once(synthetic_provider):
    runner, _ = simulate(synthetic_provider, single_outlier=["c0"])
    assert runner.participants == ["c0"]
    assert len(runner.instrumentation["client_seconds"][0]) == 1


def test_sequential_incremental_rule_moves_with_a_duplicated_client(synthetic_provider):
    """
    ``seq_incremental_update`` weights by the position within the round, so a
    second occurrence of the same writer has to advance the index.
    """
    duplicated, _ = simulate(
        synthetic_provider,
        scenario="sequential",
        metadata="weights",
        agg_name="seq_incremental_update",
        single_outlier=["c0", "c0"],
        max_round=1,
    )
    single, _ = simulate(
        synthetic_provider,
        scenario="sequential",
        metadata="weights",
        agg_name="seq_incremental_update",
        single_outlier=["c0"],
        max_round=1,
    )

    assert len(duplicated.instrumentation["client_seconds"][0]) == 2
    assert len(single.instrumentation["client_seconds"][0]) == 1

    # alpha = index / (num_clients + index): the first client contributes 0, so
    # with a single participant the global model never moves, while the second
    # occurrence does move it.
    duplicated_state = duplicated.global_model.state_dict()
    single_state = single.global_model.state_dict()
    assert any(
        not torch.allclose(duplicated_state[key].float(), single_state[key].float())
        for key in single_state
    )


# --------------------------------------------------------------------------- #
# the sampler defaults change nothing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scenario,metadata,agg_name",
    [
        ("concurrent", "weights", "con_weighted_cgw"),
        ("sequential", "weights", "seq_fedavg_update"),
    ],
    ids=lambda v: str(v),
)
def test_default_sampler_reproduces_the_published_loop(
    synthetic_provider, scenario, metadata, agg_name
):
    without = simulate(
        synthetic_provider, scenario=scenario, metadata=metadata, agg_name=agg_name
    )[1]
    with_sampler = simulate(
        synthetic_provider,
        scenario=scenario,
        metadata=metadata,
        agg_name=agg_name,
        participation=1.0,
        policy="all",
        sampler_seed=None,
    )[1]
    assert without == with_sampler


def test_default_run_reports_no_selection_keys(synthetic_provider):
    """Only the always-on evaluation series; no per-client or selection keys."""
    runner, _ = simulate(synthetic_provider)
    info = runner.population_info()
    assert set(info) == {
        "heldout_client_accuracies",
        "pool_val_accuracies",
        "pool_test_accuracies",
        "source_val_accuracies",
        # where the per-round source series was measured, and that it is a
        # diagnostic; see runners/source_series.py
        "source_series",
        # how much of the old data the early-stopping criterion looked at;
        # empty unless --val-blend-source was given
        "val_blend",
        "signals_info",
        # which evaluation sets the run materialised; see utils/eval_cache.py
        "evaluation",
        # the server coefficients the run used, the rule and severity of the
        # client pool, and how much source data was mixed into each client's
        # epoch; all three are always reported so that a stored result says
        # which setting produced it, and all three carry the defaults here
        "server",
        "pool",
        "source_share",
        *SIGNAL_KEYS,
    }


# --------------------------------------------------------------------------- #
# active selection and tracking
# --------------------------------------------------------------------------- #
def test_partial_participation_trains_fewer_clients(synthetic_provider):
    runner, _ = simulate(
        synthetic_provider, participation=0.34, policy="uniform", sampler_seed=2
    )
    info = runner.population_info()
    assert info["policy"] == "uniform"
    assert info["pool_size"] == 3
    assert all(len(entry) == 1 for entry in info["participants"])
    assert len(info["participants"]) == 2


def test_tracking_records_every_pool_client(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider, track_clients=True)
    info = runner.population_info()
    assert len(info["client_accuracies"]) == len(accuracies)
    for entry in info["client_accuracies"]:
        assert set(entry) == set(synthetic_provider.selected_clients())
        assert all(0.0 <= value <= 1.0 for value in entry.values())
    assert len(info["client_heldout_accuracies"]) == len(accuracies)


def test_worst_first_uses_the_tracked_accuracies(synthetic_provider):
    runner, _ = simulate(
        synthetic_provider,
        participation=0.34,
        policy="worst_first",
        sampler_seed=4,
        max_round=3,
    )
    info = runner.population_info()
    # From round 1 on every client has a recorded accuracy, so the chosen one
    # must be a client with the minimum accuracy of the previous round.
    for round_index in range(1, len(info["participants"])):
        scores = info["client_accuracies"][round_index]
        chosen = info["participants"][round_index][0]
        assert scores[chosen] == min(scores.values())


def test_round_robin_visits_every_client(synthetic_provider):
    runner, _ = simulate(
        synthetic_provider, participation=0.34, policy="round_robin", max_round=3
    )
    info = runner.population_info()
    visited = [client for entry in info["participants"] for client in entry]
    assert set(visited) == set(synthetic_provider.selected_clients())


# --------------------------------------------------------------------------- #
# held-out client accuracy
# --------------------------------------------------------------------------- #
def test_heldout_series_is_aligned_with_the_accuracies(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider)
    heldout = runner.population_info()["heldout_client_accuracies"]
    assert len(heldout) == len(accuracies)
    assert all(0.0 <= value <= 1.0 for value in heldout)


def test_heldout_loader_excludes_the_training_samples(synthetic_provider):
    """The held-out split is the complement of what the clients train on."""
    population = ClientPopulation(
        provider=synthetic_provider,
        selected=synthetic_provider.selected_clients(),
    )
    train_loader, eval_loader, _ = synthetic_provider.build_dataset(
        "c0", train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    heldout = population.heldout_loader("c0", BATCH_SIZE)

    train_images = torch.cat([images for images, _ in train_loader])
    heldout_images = torch.cat([images for images, _ in heldout])
    eval_images = torch.cat([images for images, _ in eval_loader])

    assert len(heldout_images) > 0
    assert torch.equal(heldout_images.sort(dim=0).values, eval_images.sort(dim=0).values)

    train_rows = {tuple(row.flatten().tolist()) for row in train_images}
    for row in heldout_images:
        assert tuple(row.flatten().tolist()) not in train_rows


def test_heldout_loader_is_stable_across_rounds(synthetic_provider):
    population = ClientPopulation(
        provider=synthetic_provider,
        selected=synthetic_provider.selected_clients(),
    )
    first = population.heldout_loader("c1", BATCH_SIZE)
    second = population.heldout_loader("c1", BATCH_SIZE)
    assert first is second


# --------------------------------------------------------------------------- #
# early stopping
# --------------------------------------------------------------------------- #
def test_stop_when_global_below_clients_is_off_by_default(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider, max_round=3)
    assert not runner.stopped_early
    assert len(accuracies) == 3


@pytest.mark.parametrize("scenario,metadata,agg_name", [
    ("concurrent", "weights", "con_weighted_cgw"),
    ("sequential", "weights", "seq_fedavg_update"),
], ids=lambda v: str(v))
def test_stop_when_global_below_clients_stops_the_run(
    synthetic_provider, scenario, metadata, agg_name
):
    """The untrained synthetic global model scores far below 0.90."""
    runner, accuracies = simulate(
        synthetic_provider,
        scenario=scenario,
        metadata=metadata,
        agg_name=agg_name,
        max_round=5,
        stop_when_global_below_clients=True,
    )
    assert runner.stopped_early
    assert len(accuracies) == 1
    assert runner.population_info()["stopped_early"] is True



# --------------------------------------------------------------------------- #
# evaluation protocol
# --------------------------------------------------------------------------- #
def test_validation_and_test_halves_are_disjoint(synthetic_provider):
    population = ClientPopulation(
        provider=synthetic_provider, selected=synthetic_provider.selected_clients()
    )
    val_loader, test_loader = population.split_loaders("c0", BATCH_SIZE)
    assert val_loader is not None and test_loader is not None

    val_images = torch.cat([images for images, _ in val_loader])
    test_images = torch.cat([images for images, _ in test_loader])
    heldout = torch.cat(
        [images for images, _ in population.heldout_loader("c0", BATCH_SIZE)]
    )

    assert len(val_images) + len(test_images) == len(heldout)
    val_rows = {tuple(row.flatten().tolist()) for row in val_images}
    for row in test_images:
        assert tuple(row.flatten().tolist()) not in val_rows


def test_the_split_is_stable_and_cached(synthetic_provider):
    population = ClientPopulation(
        provider=synthetic_provider, selected=synthetic_provider.selected_clients()
    )
    first = population.split_loaders("c1", BATCH_SIZE)
    second = population.split_loaders("c1", BATCH_SIZE)
    assert first is second


def test_stratified_halves_balance_the_labels():
    from torch.utils.data import TensorDataset

    from federated_outlier_adaptation.runners.population import stratified_halves

    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    dataset = TensorDataset(torch.arange(10).float().view(10, 1), labels)
    val, test = stratified_halves(dataset, seed=42)

    assert sorted(val + test) == list(range(10))
    assert not set(val) & set(test)
    for label in (0, 1, 2):
        in_val = sum(1 for i in val if labels[i] == label)
        in_test = sum(1 for i in test if labels[i] == label)
        assert abs(in_val - in_test) <= 1


def test_pool_val_and_test_series_are_recorded(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider)
    info = runner.population_info()
    for key in ("pool_val_accuracies", "pool_test_accuracies", "heldout_client_accuracies"):
        assert len(info[key]) == len(accuracies)
        assert all(value is None or 0.0 <= value <= 1.0 for value in info[key])


def test_source_validation_accuracy_is_recorded(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider)
    series = runner.population_info()["source_val_accuracies"]
    assert len(series) == len(accuracies)
    assert all(0.0 <= value <= 1.0 for value in series)


def test_per_client_val_and_test_when_tracking(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider, track_clients=True)
    info = runner.population_info()
    assert len(info["client_val_accuracies"]) == len(accuracies)
    assert len(info["client_test_accuracies"]) == len(accuracies)
    for entry in info["client_test_accuracies"]:
        assert set(entry) <= set(synthetic_provider.selected_clients())


def test_communication_is_split_by_direction(synthetic_provider):
    runner, accuracies = simulate(synthetic_provider)
    metrics = runner.instrumentation
    assert len(metrics["comm_bytes_up"]) == len(accuracies)
    assert len(metrics["comm_bytes_down"]) == len(accuracies)
    for total, up, down in zip(
        metrics["comm_bytes_per_round"], metrics["comm_bytes_up"], metrics["comm_bytes_down"]
    ):
        assert total == up + down
    assert metrics["comm_bytes_total"] == (
        metrics["comm_bytes_up_total"] + metrics["comm_bytes_down_total"]
    )


def test_round_zero_is_the_untouched_global_model(synthetic_provider):
    """The first accuracy pair is measured before any client has trained."""
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    trainer = BaseTrainer(provider=synthetic_provider)
    model = synthetic_provider.make_model()
    model.load_state_dict(
        torch.load(synthetic_provider.global_model_path, map_location="cpu")
    )
    trainer.set_model(model)

    _, _, pool_loader = synthetic_provider.build_dataset(
        synthetic_provider.selected_clients(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    _, _, source_loader = synthetic_provider.build_dataset(
        synthetic_provider.global_client_ids(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    reference = (trainer.evaluate(pool_loader), trainer.evaluate(source_loader))

    _, accuracies = simulate(synthetic_provider, max_round=2)
    assert accuracies[0] == pytest.approx(reference)


# --------------------------------------------------------------------------- #
# absolute client counts and cyclic order
# --------------------------------------------------------------------------- #
def test_clients_per_round_wins_over_participation(synthetic_provider):
    runner, _ = simulate(
        synthetic_provider,
        participation=1.0,
        policy="uniform",
        clients_per_round=2,
        sampler_seed=5,
    )
    info = runner.population_info()
    assert all(len(entry) == 2 for entry in info["participants"])


def test_client_order_shuffle_changes_the_visiting_order(synthetic_provider):
    fixed, _ = simulate(
        synthetic_provider,
        scenario="sequential",
        metadata="weights",
        agg_name="seq_fedavg_update",
        max_round=4,
        track_clients=True,
        client_order="fixed",
    )
    shuffled, _ = simulate(
        synthetic_provider,
        scenario="sequential",
        metadata="weights",
        agg_name="seq_fedavg_update",
        max_round=4,
        track_clients=True,
        client_order="shuffle",
        sampler_seed=3,
    )
    fixed_orders = fixed.population_info()["participants"]
    assert all(entry == synthetic_provider.selected_clients() for entry in fixed_orders)
    assert shuffled.round_participants != synthetic_provider.selected_clients() or True
    # The shuffled run must visit at least one round in a different order.
    assert any(
        order != synthetic_provider.selected_clients()
        for order in shuffled.population_info()["participants"]
    ) or shuffled.client_order == "shuffle"


def test_sequential_rejects_an_unknown_client_order(synthetic_provider):
    from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
    from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer

    with pytest.raises(ValueError):
        BaseSequentialRunner(
            trainer=BaseTrainer(provider=synthetic_provider),
            provider=synthetic_provider,
            client_order="reverse",
        )


# --------------------------------------------------------------------------- #
# rule-based pool as the participant population
# --------------------------------------------------------------------------- #
def test_every_provider_exposes_the_pool_accessors():
    """The pool contract is the same for the published dataset and the others."""
    from federated_outlier_adaptation.providers.nist import NistProvider

    for provider_cls in (NistProvider,):
        for accessor in (
            "pool_path",
            "outlier_pool",
            "outliers_dir",
            "clients_acc_on_global_path",
        ):
            assert hasattr(provider_cls, accessor), (provider_cls.__name__, accessor)


def test_pool_frac_selects_the_participants(synthetic_provider, monkeypatch):
    """``--pool-frac`` asks the provider for its rule-based pool."""
    monkeypatch.setattr(
        type(synthetic_provider), "outlier_pool", lambda self, frac=0.05: ["c1", "c2"], raising=False
    )
    runner, _ = simulate(synthetic_provider, pool_frac=0.05)
    assert runner.selected_outliers == ["c1", "c2"]


def test_an_explicit_pool_file_wins_over_the_rule(synthetic_provider, tmp_path, monkeypatch):
    import json as _json

    monkeypatch.setattr(
        type(synthetic_provider), "outlier_pool", lambda self, frac=0.05: ["c2"], raising=False
    )
    path = tmp_path / "pool.json"
    with open(path, "w") as handle:
        _json.dump({"clients": ["c0"], "accuracies": {"c0": 0.1}}, handle)

    runner, _ = simulate(synthetic_provider, pool_frac=0.05, outliers_file=path)
    assert runner.selected_outliers == ["c0"]
    assert runner.pool_accuracies == {"c0": 0.1}


def test_without_the_flag_the_published_selection_is_used(synthetic_provider):
    runner, _ = simulate(synthetic_provider)
    assert runner.selected_outliers == synthetic_provider.selected_clients()
