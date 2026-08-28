"""
The server-side proxy sets of the three providers.

A proxy set is public data that belongs to neither the source population nor
the clients, so the server may read it without violating the constraint the
study works under.  Each provider defines one differently and each definition
has to hold up: the MNIST archives must be decoded in memory and rendered into
the NIST input format with the right polarity, the CIFAR-10 reserve must belong
to no client, and the Shakespeare roles must be gone from the eligible
population.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from federated_outlier_adaptation.data import cifar10 as cifar10_data
from federated_outlier_adaptation.data import mnist as mnist_data
from federated_outlier_adaptation.providers.cifar10 import Cifar10Provider
from federated_outlier_adaptation.providers.nist import NistProvider
from federated_outlier_adaptation.providers.shakespeare import ShakespeareProvider

BATCH_SIZE = 4


# --------------------------------------------------------------------- MNIST
def _idx_bytes(array: np.ndarray) -> bytes:
    """Encode a uint8 array in the IDX container format."""
    header = struct.pack(">BBBB", 0, 0, 0x08, array.ndim)
    header += struct.pack(f">{array.ndim}I", *array.shape)
    return header + array.astype(np.uint8).tobytes()


@pytest.fixture(scope="session")
def mnist_raw_dir(tmp_path_factory) -> Path:
    """Four synthetic ``idx*.gz`` archives with six train and four test images."""
    directory = tmp_path_factory.mktemp("mnist_raw")
    rng = np.random.default_rng(3)

    train_images = rng.integers(0, 256, size=(6, 28, 28), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(4, 28, 28), dtype=np.uint8)
    # One image with no ink at all, so the polarity can be asserted exactly.
    test_images[0] = 0
    arrays = {
        "train_images": train_images,
        "train_labels": np.arange(6, dtype=np.uint8) % 10,
        "test_images": test_images,
        "test_labels": np.arange(4, dtype=np.uint8) % 10,
    }
    for key, name in mnist_data.IDX_FILES.items():
        with gzip.open(directory / name, "wb") as handle:
            handle.write(_idx_bytes(arrays[key]))
    return directory


@pytest.fixture(scope="session")
def mnist_dir(tmp_path_factory, mnist_raw_dir) -> Path:
    """A prepared MNIST proxy set (npz plus index) in a temp directory."""
    directory = tmp_path_factory.mktemp("mnist_data")
    mnist_data.prepare(raw_dir=mnist_raw_dir, out_dir=directory, log=lambda *a: None)
    return directory


def test_idx_decoding_round_trips():
    array = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    decoded = mnist_data.decode_idx(_idx_bytes(array))
    np.testing.assert_array_equal(decoded, array)


def test_prepare_writes_one_npz_and_one_index(mnist_dir, mnist_raw_dir):
    files = sorted(path.name for path in mnist_dir.iterdir())
    assert files == ["mnist.npz", "mnist_index.json"]
    # Nothing was extracted next to the archives either.
    assert sorted(p.name for p in mnist_raw_dir.iterdir()) == sorted(
        mnist_data.IDX_FILES.values()
    )

    with open(mnist_dir / "mnist_index.json") as handle:
        index = json.load(handle)
    assert index["num_test"] == 4
    assert index["invert_polarity"] is True
    assert set(index["sources"]) == set(mnist_data.IDX_FILES)
    assert all(entry["sha256"] for entry in index["sources"].values())


def test_prepare_reports_a_missing_archive(tmp_path):
    with pytest.raises(FileNotFoundError):
        mnist_data.prepare(raw_dir=tmp_path, out_dir=tmp_path, log=lambda *a: None)


def test_proxy_images_are_rendered_in_the_nist_format(mnist_dir):
    data = mnist_data.MnistData(npz_path=mnist_dir / "mnist.npz")
    dataset = data.proxy_dataset()
    assert len(dataset) == 4

    image, label = dataset[0]
    assert image.shape == (1, 128, 128)
    assert isinstance(label, int)
    # The inkless image becomes a pure white 128x128 frame, which is what a NIST
    # background is: 255 -> ToTensor 1.0 -> Normalize(0.5, 0.5) -> +1.
    assert torch.allclose(image, torch.ones_like(image))


def test_proxy_subsample_is_deterministic(mnist_dir):
    data = mnist_data.MnistData(npz_path=mnist_dir / "mnist.npz")
    first = data.rows(size=2, seed=7)
    second = data.rows(size=2, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.size == 2
    # A size covering the whole set keeps the storage order.
    np.testing.assert_array_equal(data.rows(size=99), np.arange(4))


def test_nist_provider_exposes_the_mnist_proxy(mnist_dir, tmp_path):
    provider = NistProvider(
        results_dir=tmp_path / "results",
        data_dir=tmp_path / "data",
        mnist_dir=mnist_dir,
    )
    loader = provider.proxy_loader(batch_size=BATCH_SIZE)
    assert loader is not None
    images, labels = next(iter(loader))
    assert images.shape[1:] == (1, 128, 128)
    assert labels.shape[0] == images.shape[0]

    info = provider.proxy_info()
    assert info["name"] == "mnist_test"
    assert info["size"] == 4
    assert info["hash"] and info["source_sha256"]
    # Recomputing must give the same digest, so a result file pins the set.
    assert NistProvider(
        results_dir=tmp_path / "results", data_dir=tmp_path / "data", mnist_dir=mnist_dir
    ).proxy_info()["hash"] == info["hash"]


def test_proxy_size_is_capped_by_the_environment(mnist_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FOA_NIST_PROXY_SIZE", "2")
    provider = NistProvider(
        results_dir=tmp_path / "results", data_dir=tmp_path / "data", mnist_dir=mnist_dir
    )
    assert provider.proxy_size == 2
    assert sum(len(batch[1]) for batch in provider.proxy_loader(batch_size=BATCH_SIZE)) == 2
    capped = provider.proxy_info()

    monkeypatch.delenv("FOA_NIST_PROXY_SIZE")
    full = NistProvider(
        results_dir=tmp_path / "results", data_dir=tmp_path / "data", mnist_dir=mnist_dir
    ).proxy_info()
    assert full["size"] == 4
    # The size is part of the digest, so a capped run cannot be mistaken for a
    # full one on the strength of its provenance block.
    assert capped["hash"] != full["hash"]


def test_proxy_size_still_reads_the_legacy_variable(mnist_dir, tmp_path, monkeypatch):
    """The name of the earlier repository keeps working."""
    monkeypatch.delenv("FOA_NIST_PROXY_SIZE", raising=False)
    monkeypatch.setenv("FAL_NIST_PROXY_SIZE", "2")
    provider = NistProvider(
        results_dir=tmp_path / "results", data_dir=tmp_path / "data", mnist_dir=mnist_dir
    )
    assert provider.proxy_size == 2


def test_a_malformed_proxy_size_is_ignored(mnist_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("FOA_NIST_PROXY_SIZE", "not-a-number")
    provider = NistProvider(
        results_dir=tmp_path / "results", data_dir=tmp_path / "data", mnist_dir=mnist_dir
    )
    assert provider.proxy_size is None
    assert provider.proxy_info()["size"] == 4


def test_nist_provider_without_the_proxy_set_is_unchanged(tmp_path):
    provider = NistProvider(
        results_dir=tmp_path / "results",
        data_dir=tmp_path / "data",
        mnist_dir=tmp_path / "absent",
    )
    assert provider.proxy_loader() is None
    assert provider.proxy_info() == {}


# ------------------------------------------------------------------ CIFAR-10
def test_reserve_proxy_is_label_balanced_and_disjoint():
    rows = np.arange(100)
    labels = np.tile(np.arange(10), 10)
    proxy, rest = cifar10_data.reserve_proxy(rows, labels, size=20, seed=42)
    assert proxy.size == 20
    assert rest.size == 80
    assert not set(proxy.tolist()) & set(rest.tolist())
    np.testing.assert_array_equal(np.sort(np.concatenate([proxy, rest])), rows)
    values, counts = np.unique(labels[proxy], return_counts=True)
    assert values.size == 10 and set(counts.tolist()) == {2}


def test_reserve_proxy_defaults_to_nothing():
    rows = np.arange(10)
    proxy, rest = cifar10_data.reserve_proxy(rows, np.zeros(10, dtype=int), size=0)
    assert proxy.size == 0
    np.testing.assert_array_equal(rest, rows)


def test_reserved_images_belong_to_no_client():
    labels = np.tile(np.arange(10), 600)
    is_test = np.zeros(labels.size, dtype=bool)
    is_test[5000:] = True

    _, pools, proxy, stats = cifar10_data.build_clients(
        labels, is_test, num_clients=25, alpha=0.3, seed=42, proxy_size=100
    )
    assert proxy.size == 100 == stats["proxy_size"]
    joined = np.sort(np.concatenate(pools))
    assert not set(joined.tolist()) & set(proxy.tolist())
    np.testing.assert_array_equal(
        np.sort(np.concatenate([joined, proxy])), np.arange(labels.size)
    )


def test_prepare_with_a_proxy_reserve(cifar_archive, tmp_path):
    summary = cifar10_data.prepare(
        archive_path=cifar_archive,
        out_dir=tmp_path / "prepared",
        num_clients=4,
        alpha=0.5,
        seed=42,
        proxy_size=10,
        log=lambda *a: None,
    )
    assert summary["proxy_size"] == 10

    data = cifar10_data.Cifar10Data(
        npz_path=tmp_path / "prepared" / "cifar10.npz",
        clients_path=tmp_path / "prepared" / "cifar10_clients.json",
    )
    assert data.proxy_rows().size == 10
    assigned = {row for client in data.all_clients() for row in data.clients[client].tolist()}
    assert not assigned & set(data.proxy_rows().tolist())

    provider = Cifar10Provider(
        results_dir=tmp_path / "results",
        data_dir=tmp_path / "prepared",
        num_global_clients=2,
        min_selection_samples=1,
        model_kwargs={"widths": (8, 16), "fc_dim": 16},
    )
    loader = provider.proxy_loader(batch_size=BATCH_SIZE)
    assert loader is not None
    assert sum(len(batch[1]) for batch in loader) == 10
    info = provider.proxy_info()
    assert info["name"] == "cifar10_reserved_test"
    assert info["size"] == 10 and info["hash"]


def test_partition_without_a_reserve_has_no_proxy(cifar10_provider):
    assert cifar10_provider.proxy_loader() is None
    assert cifar10_provider.proxy_info() == {}


def test_a_second_partition_name_keeps_the_first_reachable(cifar_archive, tmp_path):
    out = tmp_path / "prepared"
    cifar10_data.prepare(
        archive_path=cifar_archive, out_dir=out, num_clients=4, seed=42, log=lambda *a: None
    )
    cifar10_data.prepare(
        archive_path=cifar_archive,
        out_dir=out,
        num_clients=4,
        seed=42,
        proxy_size=10,
        clients_name="cifar10_clients_proxy1000.json",
        log=lambda *a: None,
    )
    assert (out / "cifar10_clients.json").is_file()
    assert (out / "cifar10_clients_proxy1000.json").is_file()


# --------------------------------------------------------------- Shakespeare
def _proxy_shakespeare(shakespeare_dir, results_dir) -> ShakespeareProvider:
    """Toy provider whose carve-out is allowed to take one of the two roles."""
    return ShakespeareProvider(
        results_dir=results_dir,
        data_dir=shakespeare_dir,
        global_fraction=0.5,
        min_selection_samples=1,
        proxy_roles=1,
        proxy_max_fraction=1.0,
        proxy_sequences=4,
        model_kwargs={"embed": 4, "hidden": 16, "layers": 1},
    )


def test_proxy_roles_are_deterministic(shakespeare_dir, tmp_path):
    first = _proxy_shakespeare(shakespeare_dir, tmp_path / "a").proxy_clients()
    second = _proxy_shakespeare(shakespeare_dir, tmp_path / "b").proxy_clients()
    assert first == second
    assert len(first) == 1


def test_proxy_roles_leave_the_eligible_population(shakespeare_dir, tmp_path):
    provider = _proxy_shakespeare(shakespeare_dir, tmp_path / "r")
    reserved = set(provider.proxy_clients())
    assert reserved
    assert reserved <= set(provider.local_client_ids())
    assert not reserved & set(provider.eligible_clients())


def test_a_stored_pool_is_filtered_on_read(shakespeare_dir, tmp_path):
    from federated_outlier_adaptation.outliers.client_accuracy import write_pool

    provider = _proxy_shakespeare(shakespeare_dir, tmp_path / "r")
    clients = provider.local_client_ids()
    target = provider.results_dir / "outliers" / "clients_acc_on_global.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as handle:
        json.dump([{c: 0.5 - 0.01 * i} for i, c in enumerate(clients)], handle)

    # A pool written over *all* held-out roles, i.e. the rule as it was before
    # the proxy set existed.
    write_pool(target, provider.results_dir / "outliers", frac=1.0, eligible=clients)
    pool = provider.outlier_pool(1.0)
    assert set(pool) == set(clients) - set(provider.proxy_clients())


def test_proxy_loader_yields_the_reserved_sequences(shakespeare_dir, tmp_path):
    provider = _proxy_shakespeare(shakespeare_dir, tmp_path / "r")
    loader = provider.proxy_loader(batch_size=BATCH_SIZE)
    assert loader is not None
    total = sum(len(batch[1]) for batch in loader)
    assert total == provider.proxy_offsets().size <= 4

    inputs, targets = next(iter(loader))
    assert inputs.dtype == torch.int64
    logits = provider.make_model()(inputs)
    assert logits.shape == (inputs.shape[0], provider.num_classes)

    info = provider.proxy_info()
    assert info["name"] == "shakespeare_reserved_roles"
    assert info["roles"] == provider.proxy_clients()
    assert info["hash"]


def test_default_carve_out_leaves_a_small_corpus_alone(shakespeare_provider):
    """The 25% cap means a toy corpus keeps every one of its roles."""
    assert shakespeare_provider.proxy_clients() == []
    assert shakespeare_provider.proxy_loader() is None
    assert shakespeare_provider.proxy_info() == {}
