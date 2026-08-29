"""
The evaluation cache: same tensors, same numbers, one pass per set.

The point of the cache is that it is invisible in the results.  These tests
therefore compare it against the loader path everywhere it replaces one: the
float tensors it feeds the model, the accuracies it reports, the per-client
accuracies it derives from a single pass, the forgetting signals, and the two
runners end to end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset

from federated_outlier_adaptation.aggregation.selector import select_class
from federated_outlier_adaptation.runners.concurrent_runner import BaseConcurrentRunner
from federated_outlier_adaptation.runners.sequential_runner import BaseSequentialRunner
from federated_outlier_adaptation.trainers.base_trainer import BaseTrainer
from federated_outlier_adaptation.utils.eval_cache import (
    GRAYSCALE_U8,
    IDENTITY,
    POOL_CLIENTS_SET,
    POOL_INSAMPLE_SET,
    POOL_TEST_SET,
    POOL_VAL_SET,
    SOURCE_TEST_SET,
    SOURCE_VAL_SET,
    GpuEvalCache,
    materialise,
    resolve_eval_path,
)

BATCH_SIZE = 4
EPOCHS = 1


def _digit_dataset(tmp_path, count: int = 6):
    """A packed-cache dataset, the back-end every NIST run actually reads."""
    from federated_outlier_adaptation import config
    from federated_outlier_adaptation.data.datasets import MemmapDigitDataset, build_transform
    from federated_outlier_adaptation.data.nist_cache import NistDigitCache

    import json

    rng = np.random.default_rng(0)
    images = (rng.integers(0, 2, size=(count, 128, 128)) * 255).astype(np.uint8)
    packed = np.packbits(images.reshape(count, -1) > 127, axis=1)
    np.save(tmp_path / config.CACHE_ARRAY_NAME, packed)
    labels = [int(index % 3) for index in range(count)]
    with open(tmp_path / config.CACHE_INDEX_NAME, "w") as handle:
        json.dump(
            {
                "paths": [f"by_write/x/w{index}/img{index}.png" for index in range(count)],
                "labels": labels,
                "writers": [f"w{index}" for index in range(count)],
                "packed": True,
                "shape": [128, 128],
                "missing": [],
            },
            handle,
        )
    cache = NistDigitCache(tmp_path)
    rows = list(range(count))
    return MemmapDigitDataset(cache, rows, labels, build_transform())


# --------------------------------------------------------------------------- #
# the tensors
# --------------------------------------------------------------------------- #
def test_the_cached_tensors_are_the_loader_tensors(tmp_path):
    """
    The whole construction rests on this: the batched decode of the compact
    payload is bit-identical to the dataset's per-sample transform.
    """
    dataset = _digit_dataset(tmp_path)
    payload, labels, decoder = materialise(dataset)
    assert decoder == GRAYSCALE_U8
    assert payload.dtype == torch.uint8

    decoded = decoder(payload)
    expected = torch.stack([dataset[index][0] for index in range(len(dataset))])
    assert decoded.shape == expected.shape
    assert torch.equal(decoded, expected)
    assert labels.tolist() == [dataset[index][1] for index in range(len(dataset))]


def test_a_subset_keeps_the_order_of_its_indices(tmp_path):
    dataset = _digit_dataset(tmp_path)
    subset = Subset(dataset, [4, 1, 2])
    payload, labels, decoder = materialise(subset)
    expected = torch.stack([subset[index][0] for index in range(len(subset))])
    assert torch.equal(decoder(payload), expected)
    assert labels.tolist() == [subset[index][1] for index in range(len(subset))]


def test_a_concatenation_is_the_concatenation_of_its_parts(tmp_path):
    dataset = _digit_dataset(tmp_path)
    combined = ConcatDataset([Subset(dataset, [0, 1]), Subset(dataset, [3])])
    payload, labels, decoder = materialise(combined)
    expected = torch.stack([combined[index][0] for index in range(len(combined))])
    assert torch.equal(decoder(payload), expected)
    assert labels.tolist() == [0, 1, 0]


def test_a_dataset_without_a_compact_form_falls_back_to_expansion():
    images = torch.rand(5, 1, 8, 8)
    labels = torch.arange(5) % 2
    payload, cached_labels, decoder = materialise(TensorDataset(images, labels))
    assert decoder == IDENTITY
    assert torch.equal(decoder(payload), images)
    assert cached_labels.tolist() == labels.tolist()




def test_a_non_standard_transform_declines_to_be_cached(tmp_path):
    dataset = _digit_dataset(tmp_path)
    dataset.transform = lambda image: torch.zeros(1, 128, 128)
    assert dataset.eval_payload() is None


# --------------------------------------------------------------------------- #
# the numbers
# --------------------------------------------------------------------------- #
def _cache_with(dataset, name="set", segments=None, batch_size=BATCH_SIZE):
    """
    A cache scoring the set in the loader's batches.

    The batch size is the loader's on purpose: these tests assert *exact*
    equality, and a convolution evaluated in differently sized batches can
    differ in the last bits, which is a property of the accelerator kernels and
    not of the cache.
    """
    cache = GpuEvalCache(device="cpu", batch_size=batch_size)
    if segments is None:
        assert cache.add(name, dataset)
    else:
        assert cache.add_segments(name, segments)
    return cache


def test_cached_accuracy_equals_the_loader_accuracy(synthetic_provider, tmp_path):
    trainer = BaseTrainer(provider=synthetic_provider)
    _, _, loader = synthetic_provider.build_dataset(
        synthetic_provider.selected_clients(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    cache = _cache_with(loader.dataset)
    cache.bind(trainer.get_model())
    assert cache.accuracy("set") == pytest.approx(trainer.evaluate(loader), abs=0.0)


def test_per_client_accuracies_come_from_one_pass(synthetic_provider):
    """The segmented set reproduces what per-client loaders would report."""
    trainer = BaseTrainer(provider=synthetic_provider)
    clients = synthetic_provider.selected_clients()
    loaders = {
        client: synthetic_provider.build_dataset(
            client, train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
        )[2]
        for client in clients
    }
    cache = _cache_with(
        None,
        segments=[(client, loaders[client].dataset) for client in clients],
    )
    cache.bind(trainer.get_model())
    cached = cache.segment_accuracies("set")
    for client in clients:
        assert cached[client] == pytest.approx(trainer.evaluate(loaders[client]), abs=0.0)
    assert cache.segment_sizes("set") == {
        client: len(loaders[client].dataset) for client in clients
    }


def test_one_forward_pass_per_set_per_round(synthetic_provider):
    """Everything a round derives from a set reads the same memoised pass."""
    calls = {"n": 0}

    class CountingModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            calls["n"] += 1
            return self.inner(x)

    _, _, loader = synthetic_provider.build_dataset(
        synthetic_provider.selected_clients(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    cache = GpuEvalCache(device="cpu", batch_size=1024)
    assert cache.add("set", loader.dataset)
    cache.bind(CountingModel(synthetic_provider.make_model()))

    cache.accuracy("set")
    cache.accuracy("set")
    cache.logits("set")
    assert calls["n"] == 1

    cache.bind(CountingModel(synthetic_provider.make_model()))
    cache.accuracy("set")
    assert calls["n"] == 2


def test_an_unmaterialisable_set_is_simply_absent():
    class Awkward(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, item):
            raise OSError("no data here")

    cache = GpuEvalCache(device="cpu")
    assert cache.add("set", Awkward()) is False
    assert not cache.has("set")


# --------------------------------------------------------------------------- #
# the runners
# --------------------------------------------------------------------------- #
def _simulate(provider, eval_path, scenario="concurrent", metadata="weights",
              agg_name="con_weighted_cgw", max_round=2, seed=11, **kwargs):
    torch.manual_seed(0)
    trainer = BaseTrainer(provider=provider)
    runner_cls = BaseSequentialRunner if scenario == "sequential" else BaseConcurrentRunner
    runner = runner_cls(
        trainer=trainer, provider=provider, seed=seed, eval_path=eval_path, **kwargs
    )
    accuracies, *_ = runner.simulate(
        exp_name="test_eval_cache",
        global_name="global",
        aggregate_method=getattr(select_class(scenario, metadata), agg_name),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        max_round=max_round,
        grid_Search=False,
    )
    return runner, accuracies


SERIES = (
    "heldout_client_accuracies",
    "pool_val_accuracies",
    "pool_test_accuracies",
    "source_val_accuracies",
    "retention_known",
    "agreement_with_global",
    "kl_global_to_current",
)


@pytest.mark.parametrize(
    "scenario,metadata,agg_name",
    [
        ("concurrent", "weights", "con_weighted_cgw"),
        ("sequential", "weights", "seq_fedavg_update"),
    ],
    ids=lambda v: str(v),
)
def test_the_two_paths_agree_end_to_end(synthetic_provider, scenario, metadata, agg_name):
    """
    The equality check in miniature: the same seeded run on both paths.

    Training is seeded and runs on the CPU, so the models of the two runs are
    the same and every evaluation series must match exactly.
    """
    loader_runner, loader_acc = _simulate(
        synthetic_provider, "loader", scenario, metadata, agg_name, track_clients=True
    )
    cache_runner, cache_acc = _simulate(
        synthetic_provider, "cache", scenario, metadata, agg_name, track_clients=True
    )

    assert len(loader_acc) == len(cache_acc)
    for expected, actual in zip(loader_acc, cache_acc):
        assert actual == pytest.approx(expected, abs=1e-6)
    loader_info = loader_runner.population_info()
    cache_info = cache_runner.population_info()
    for key in SERIES:
        assert cache_info[key] == pytest.approx(loader_info[key], abs=1e-6), key
    for key in ("client_accuracies", "client_val_accuracies", "client_test_accuracies"):
        cached_rounds = cache_info[key]
        loader_rounds = loader_info[key]
        assert len(cached_rounds) == len(loader_rounds), key
        for cached, expected in zip(cached_rounds, loader_rounds):
            assert set(cached) == set(expected), key
            for client, value in expected.items():
                assert cached[client] == pytest.approx(value, abs=1e-6), (key, client)


def test_the_default_path_is_the_cache(synthetic_provider):
    runner, _ = _simulate(synthetic_provider, None)
    assert runner.eval_path == resolve_eval_path(None) == "cache"
    assert runner.eval_cache is not None


def test_the_run_records_which_sets_it_cached(synthetic_provider):
    runner, _ = _simulate(synthetic_provider, "cache", track_clients=True)
    evaluation = runner.population_info()["evaluation"]
    assert evaluation["eval_path"] == "cache"
    for name in (
        SOURCE_TEST_SET,
        SOURCE_VAL_SET,
        POOL_INSAMPLE_SET,
        POOL_CLIENTS_SET,
        POOL_VAL_SET,
        POOL_TEST_SET,
    ):
        assert name in evaluation["sets"], name


def test_the_loader_path_caches_nothing(synthetic_provider):
    runner, _ = _simulate(synthetic_provider, "loader")
    assert runner.eval_cache is None
    assert runner.population_info()["evaluation"] == {"eval_path": "loader"}


# --------------------------------------------------------------------------- #
# --insample-every
# --------------------------------------------------------------------------- #
def test_insample_every_defaults_to_measuring_every_round(synthetic_provider):
    runner, accuracies = _simulate(synthetic_provider, "cache", max_round=4, track_clients=True)
    info = runner.population_info()
    assert "insample_every" not in info
    assert runner.population.insample_rounds == list(range(len(accuracies)))
    assert all(entry[0] is not None for entry in accuracies)


def test_insample_every_thins_the_in_sample_metrics(synthetic_provider):
    runner, accuracies = _simulate(
        synthetic_provider, "cache", max_round=4, track_clients=True, insample_every=2
    )
    info = runner.population_info()
    assert info["insample_every"] == 2
    # every second round plus the last one
    assert info["insample_rounds"] == [0, 2, 3]
    assert len(info["client_accuracies"]) == len(accuracies)
    # the skipped round repeats the previous measurement, so the series stays
    # aligned and typed
    assert accuracies[1][0] == accuracies[0][0]


def test_insample_every_leaves_the_other_series_alone(synthetic_provider):
    thinned, _ = _simulate(
        synthetic_provider, "cache", max_round=4, track_clients=True, insample_every=2
    )
    full, _ = _simulate(synthetic_provider, "cache", max_round=4, track_clients=True)
    thinned_info = thinned.population_info()
    full_info = full.population_info()
    for key in ("pool_val_accuracies", "pool_test_accuracies", "source_val_accuracies"):
        assert thinned_info[key] == pytest.approx(full_info[key], abs=1e-6), key


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def test_the_payload_stays_in_host_memory_without_an_accelerator(synthetic_provider):
    _, _, loader = synthetic_provider.build_dataset(
        synthetic_provider.selected_clients(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    cache = _cache_with(loader.dataset)
    assert cache.summary()["on_device"] == []
    assert cache.summary()["device"] == "cpu"


def test_the_provider_exposes_its_evaluation_arrays(synthetic_provider):
    payload, labels, decoder = synthetic_provider.evaluation_arrays(
        synthetic_provider.selected_clients(), split="insample", batch_size=BATCH_SIZE
    )
    _, _, loader = synthetic_provider.build_dataset(
        synthetic_provider.selected_clients(), train_rate=0.0, eval_rate=0.0, batch_size=BATCH_SIZE
    )
    assert payload.shape[0] == len(loader.dataset)
    assert labels.shape[0] == len(loader.dataset)
    assert decoder(payload).shape == torch.stack(
        [loader.dataset[i][0] for i in range(len(loader.dataset))]
    ).shape


def test_an_unknown_evaluation_split_is_rejected(synthetic_provider):
    with pytest.raises(ValueError):
        synthetic_provider.evaluation_arrays(["c0"], split="everything")
