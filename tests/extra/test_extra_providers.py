"""
The additional providers must drive the unchanged runners and trainers.

Every dataset-specific behaviour is exercised through the
:class:`~federated_outlier_adaptation.providers.base.DatasetProvider`
interface, and both runners are run for two rounds with the plain and the
distillation trainer on the toy datasets.
"""

from __future__ import annotations

import json

import pytest
import torch

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.providers import (
    available_providers,
    get_provider,
    provider_config_block,
)
from federated_outlier_adaptation.providers.base import DatasetProvider
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.trainers.distillation_trainer import DistillationTrainer

MAX_ROUND = 2
EPOCHS = 1
BATCH_SIZE = 4

AGGREGATIONS = [
    ("concurrent", "weights", "con_weighted_cgw"),
    ("sequential", "weights", "seq_fedavg_update"),
]


@pytest.fixture(params=["shakespeare", "cifar10"])
def provider(request, shakespeare_provider, cifar10_provider):
    return shakespeare_provider if request.param == "shakespeare" else cifar10_provider


# ------------------------------------------------------------------ registry
def test_registry_knows_every_provider():
    assert available_providers() == ["cifar10", "nist", "shakespeare"]


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError):
        get_provider("mnist")


def test_registry_builds_the_added_providers(shakespeare_dir, cifar_dir, tmp_path):
    shakespeare = get_provider(
        "shakespeare", data_dir=shakespeare_dir, results_dir=tmp_path / "r1"
    )
    cifar = get_provider("cifar10", data_dir=cifar_dir, results_dir=tmp_path / "r2")
    assert shakespeare.name == "shakespeare"
    assert cifar.name == "cifar10"


# ------------------------------------------------------------------ contract
def test_provider_implements_the_interface(provider):
    assert isinstance(provider, DatasetProvider)
    assert provider.num_classes > 1
    assert provider.all_client_ids()
    assert set(provider.local_client_ids()) | set(provider.global_client_ids()) == set(
        provider.all_client_ids()
    )
    assert set(provider.local_client_ids()) & set(provider.global_client_ids()) == set()


def test_client_split_is_deterministic(provider):
    first = provider.make_client_split()
    second = provider.make_client_split()
    assert first == second


def test_sample_counts_are_positive(provider):
    for client in provider.all_client_ids():
        assert provider.sample_count(client) > 0


def test_build_dataset_shapes(provider):
    client = provider.local_client_ids()[0]
    train, val, test = provider.build_dataset(
        client, train_rate=0.6, eval_rate=0.2, batch_size=BATCH_SIZE
    )
    total = sum(len(loader.dataset) for loader in (train, val, test) if loader is not None)
    assert total == provider.sample_count(client)

    x, y = next(iter(train))
    assert y.dtype == torch.int64
    assert x.shape[0] == y.shape[0]
    logits = provider.make_model()(x)
    assert logits.shape == (x.shape[0], provider.num_classes)


def test_test_only_split(provider):
    client = provider.local_client_ids()[0]
    train, val, test = provider.build_dataset(client, train_rate=0.0, eval_rate=0.0)
    assert train is None and val is None
    assert len(test.dataset) == provider.sample_count(client)


def test_training_loaders_match_the_evaluation_split_sizes(provider):
    clients = provider.global_client_ids()
    train, val, _ = provider.build_training_dataset(
        clients, train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    plain_train, plain_val, _ = provider.build_dataset(
        clients, train_rate=0.6, eval_rate=0.4, batch_size=BATCH_SIZE
    )
    assert len(train.dataset) == len(plain_train.dataset)
    assert len(val.dataset) == len(plain_val.dataset)


def test_provenance_block_carries_the_dataset_hashes(provider):
    block = provider_config_block(provider)
    assert block["provider"] == provider.name
    assert block["dataset_files"]
    for entry in block["dataset_files"].values():
        assert entry["sha256"]


# ------------------------------------------------------------------- runners
def run_simulation(trainer_cls, provider, scenario, metadata, agg_name):
    trainer = trainer_cls(provider=provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(trainer=trainer, provider=provider)
    agg_method = getattr(select_class(scenario, metadata), agg_name)
    accuracies, _, _, results_dir = runner.simulate(
        exp_name=f"test_{agg_name}",
        global_name="global",
        aggregate_method=agg_method,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=MAX_ROUND,
        grid_Search=False,
    )
    return runner, accuracies, results_dir


@pytest.mark.parametrize("trainer_cls", [BaseTrainer, DistillationTrainer], ids=lambda c: c.__name__)
@pytest.mark.parametrize("scenario,metadata,agg_name", AGGREGATIONS, ids=lambda v: str(v))
def test_runner_completes_two_rounds(provider, trainer_cls, scenario, metadata, agg_name):
    runner, accuracies, results_dir = run_simulation(
        trainer_cls, provider, scenario, metadata, agg_name
    )

    assert len(accuracies) == MAX_ROUND
    for clients_acc, global_acc in accuracies:
        assert 0.0 <= clients_acc <= 1.0
        assert 0.0 <= global_acc <= 1.0
    assert results_dir == provider.results_dir
    assert runner.instrumentation["param_count"] > 0
    assert len(runner.instrumentation["comm_bytes_per_round"]) == MAX_ROUND


def test_selected_clients_come_from_the_accuracy_ranking(provider, tmp_path):
    accuracies = [
        {client: 0.9 - 0.01 * index}
        for index, client in enumerate(provider.local_client_ids())
    ]
    target = provider.results_dir / "outliers" / "clients_acc_on_global.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(accuracies, handle)

    provider._outliers_file = None
    (provider.results_dir / "outliers" / "selected_outliers.json").unlink(missing_ok=True)

    worst = min(accuracies, key=lambda entry: list(entry.values())[0])
    assert provider.selected_clients(k=1) == list(worst)


def write_accuracies(provider, values=None):
    """Store a synthetic per-client accuracy record and return it."""
    clients = provider.local_client_ids()
    if values is None:
        values = [0.9 - 0.01 * index for index in range(len(clients))]
    payload = [{client: value} for client, value in zip(clients, values)]
    target = provider.results_dir / "outliers" / "clients_acc_on_global.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump(payload, handle)
    return target, dict(zip(clients, values))


def test_pool_file_names_follow_the_convention():
    from federated_outlier_adaptation.outliers.client_accuracy import pool_file_name

    assert pool_file_name(frac=0.05) == "outlier_pool_frac0.05.json"
    assert pool_file_name(frac=0.2) == "outlier_pool_frac0.2.json"
    assert pool_file_name(threshold=0.5) == "outlier_pool_thr0.5.json"


def test_write_pool_takes_the_worst_fraction(provider):
    from federated_outlier_adaptation.outliers.client_accuracy import load_pool, write_pool

    target, table = write_accuracies(provider)
    out_dir = provider.results_dir / "outliers"
    path, payload = write_pool(target, out_dir, frac=0.5, eligible=provider.eligible_clients())

    assert path.name == "outlier_pool_frac0.5.json"
    assert payload["rule"] == "fraction"
    assert payload["eligible_clients"] == len(provider.eligible_clients())
    assert payload["size"] == max(1, int(0.5 * payload["eligible_clients"]))
    # Worst first, and every member is at or below every non-member.
    accuracies = [payload["accuracies"][c] for c in payload["clients"]]
    assert accuracies == sorted(accuracies)
    outside = [v for c, v in table.items() if c not in payload["clients"]]
    assert not outside or max(accuracies) <= min(outside)
    assert load_pool(path) == payload["clients"]
    assert set(payload["accuracies"]) == set(payload["clients"])


def test_write_pool_threshold_rule(provider):
    from federated_outlier_adaptation.outliers.client_accuracy import write_pool

    target, table = write_accuracies(provider)
    cut = sorted(table.values())[0] + 1e-9
    _, payload = write_pool(
        target,
        provider.results_dir / "outliers",
        frac=None,
        threshold=cut,
        eligible=provider.eligible_clients(),
    )
    assert payload["rule"] == "threshold"
    assert payload["frac"] is None
    assert all(value < cut for value in payload["accuracies"].values())


def test_pool_is_never_empty(provider):
    from federated_outlier_adaptation.outliers.client_accuracy import build_pool

    target, _ = write_accuracies(provider)
    payload = build_pool(target, frac=0.0001, eligible=provider.eligible_clients())
    assert payload["size"] == 1


def test_provider_reads_its_pool_file(provider):
    from federated_outlier_adaptation.outliers.client_accuracy import write_pool

    target, _ = write_accuracies(provider)
    path, payload = write_pool(
        target, provider.results_dir / "outliers", frac=0.5, eligible=provider.eligible_clients()
    )
    assert provider.pool_path(0.5) == path
    assert provider.outlier_pool(0.5) == payload["clients"]
    # Without a stored file the pool is derived from the accuracy record.
    path.unlink()
    assert provider.outlier_pool(0.5) == payload["clients"]


def test_fisher_pass_works_for_every_topology(provider, tmp_path):
    """The Fisher pass runs the model in evaluation mode; recurrent layers included."""
    from federated_outlier_adaptation.training.fisher import (
        compute_and_save_fisher_and_params,
        load_fisher_and_params,
    )

    loader, _, _ = provider.build_dataset(provider.global_client_ids(), batch_size=BATCH_SIZE)
    model = provider.make_model()
    compute_and_save_fisher_and_params(
        model=model,
        dataloader=loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        fisher_path=tmp_path / "fisher",
        max_batches=2,
    )
    fisher, params = load_fisher_and_params(tmp_path / "fisher", "cpu")
    assert set(fisher) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(fisher) == set(params)
    assert any(float(value.abs().sum()) > 0 for value in fisher.values())


def test_adaptation_entry_point_writes_provenance(provider):
    from federated_outlier_adaptation.training.provider_adaptation import run_adaptation

    payload = run_adaptation(
        provider,
        trainer_name="BaseTrainer",
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=1,
        parent_name="unit_adaptation",
    )
    assert len(payload["accuracies"]) == 1
    config = payload["config"]
    assert config["provider"] == provider.name
    assert config["trainer"] == "BaseTrainer"
    assert config["dataset_files"]
    assert json.loads(open(payload["json_path"]).read())["accuracies"] == payload["accuracies"]
