"""Archive decoding, Dirichlet partition and loaders of the CIFAR-10 dataset."""

from __future__ import annotations

import json

import numpy as np
import torch

from federated_outlier_adaptation.data import cifar10 as cf


def test_default_partition_is_two_hundred_clients():
    """The held-out half must stay large enough for a bottom-fraction pool."""
    from federated_outlier_adaptation.providers.cifar10 import (
        DEFAULT_GLOBAL_CLIENTS,
        Cifar10Provider,
    )

    assert cf.DEFAULT_NUM_CLIENTS == 200
    assert cf.DEFAULT_ALPHA == 0.3
    assert cf.DEFAULT_SEED == 42
    assert DEFAULT_GLOBAL_CLIENTS == 100
    # 200 clients minus 100 source clients leaves 100 held out.
    assert cf.DEFAULT_NUM_CLIENTS - DEFAULT_GLOBAL_CLIENTS == 100
    import inspect

    signature = inspect.signature(Cifar10Provider.__init__)
    assert signature.parameters["num_global_clients"].default == DEFAULT_GLOBAL_CLIENTS


def test_smaller_layout_is_still_reachable(cifar_archive, cifar_dir, tmp_path):
    """The earlier 100 / 80 / 20 partition remains available through parameters."""
    from federated_outlier_adaptation.providers.cifar10 import Cifar10Provider

    summary = cf.prepare(
        archive_path=cifar_archive,
        out_dir=tmp_path / "small",
        num_clients=100,
        alpha=0.3,
        seed=42,
        log=lambda *a: None,
    )
    assert summary["num_clients"] == 100

    provider = Cifar10Provider(
        results_dir=tmp_path / "results", data_dir=cifar_dir, num_global_clients=4
    )
    local, glob = provider.make_client_split()
    assert len(glob) == 4
    assert len(local) == len(provider.all_client_ids()) - 4


def test_archive_is_read_in_memory(cifar_archive, tmp_path):
    images, labels, is_test = cf.read_archive(cifar_archive)
    assert images.shape == (120, 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels.shape == (120,)
    assert is_test.sum() == 20
    # Nothing was unpacked next to the archive.
    assert sorted(p.name for p in cifar_archive.parent.iterdir()) == ["cifar-10-python.tar.gz"]


def test_dirichlet_partition_is_a_partition():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, size=5000)
    parts, proportions = cf.dirichlet_partition(labels, num_clients=20, alpha=0.3, seed=42)

    assert len(parts) == 20
    joined = np.sort(np.concatenate(parts))
    np.testing.assert_array_equal(joined, np.arange(labels.size))
    assert sum(part.size for part in parts) == labels.size
    assert proportions.shape == (20, 10)
    for row, part in zip(proportions, parts):
        if part.size:
            assert abs(row.sum() - 1.0) < 1e-9


def test_dirichlet_partition_is_seed_deterministic():
    labels = np.tile(np.arange(10), 500)
    first, _ = cf.dirichlet_partition(labels, num_clients=15, alpha=0.3, seed=42)
    second, _ = cf.dirichlet_partition(labels, num_clients=15, alpha=0.3, seed=42)
    other, _ = cf.dirichlet_partition(labels, num_clients=15, alpha=0.3, seed=7)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert any(a.size != b.size or not np.array_equal(a, b) for a, b in zip(first, other))


def test_dirichlet_partition_is_skewed_at_small_alpha():
    labels = np.tile(np.arange(10), 1000)
    _, skewed = cf.dirichlet_partition(labels, num_clients=50, alpha=0.1, seed=1)
    _, uniform = cf.dirichlet_partition(labels, num_clients=50, alpha=100.0, seed=1)
    # The lower the concentration, the more mass a client puts on few labels.
    assert skewed.max(axis=1).mean() > uniform.max(axis=1).mean()


def test_test_pool_follows_the_client_label_proportions():
    labels = np.tile(np.arange(10), 600)
    is_test = np.zeros(labels.size, dtype=bool)
    is_test[5000:] = True

    client_ids, pools, proxy_rows, stats = cf.build_clients(
        labels, is_test, num_clients=25, alpha=0.3, seed=42
    )
    assert proxy_rows.size == 0
    assert len(client_ids) == len(pools) == 25
    assert stats["num_samples"] == labels.size
    joined = np.sort(np.concatenate(pools))
    np.testing.assert_array_equal(joined, np.arange(labels.size))
    assert stats["train_pool_samples"] + stats["test_pool_samples"] == labels.size


def test_prepare_round_trips(cifar_archive, tmp_path):
    summary = cf.prepare(
        archive_path=cifar_archive,
        out_dir=tmp_path / "prepared",
        num_clients=4,
        alpha=0.5,
        seed=42,
        log=lambda *a: None,
    )
    assert summary["num_images"] == 120
    assert summary["num_clients"] == 4

    with open(tmp_path / "prepared" / "cifar10_clients.json") as handle:
        index = json.load(handle)
    assert index["source_sha256"]
    assert set(index["clients"]) == {"c000", "c001", "c002", "c003"}

    data = cf.Cifar10Data(
        tmp_path / "prepared" / "cifar10.npz", tmp_path / "prepared" / "cifar10_clients.json"
    )
    raw_images, raw_labels, _ = cf.read_archive(cifar_archive)
    np.testing.assert_array_equal(data.images, raw_images)
    np.testing.assert_array_equal(data.labels, raw_labels)
    assert sum(data.sample_count(c) for c in data.all_clients()) == 120


def test_prepare_is_deterministic(cifar_archive, tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    for target in (first, second):
        cf.prepare(archive_path=cifar_archive, out_dir=target, num_clients=4, log=lambda *a: None)
    assert (first / "cifar10.npz").read_bytes() == (second / "cifar10.npz").read_bytes()
    assert (first / "cifar10_clients.json").read_text() == (
        second / "cifar10_clients.json"
    ).read_text()


def test_build_dataset_respects_the_nist_rates(cifar_dir):
    data = cf.Cifar10Data(cifar_dir / "cifar10.npz", cifar_dir / "cifar10_clients.json")
    client = data.all_clients()[0]
    total = data.sample_count(client)

    train, val, test = data.build_dataset(client, train_rate=0.6, eval_rate=0.2, batch_size=8)
    assert len(train.dataset) + len(val.dataset) + len(test.dataset) == total

    train, val, test = data.build_dataset(client, train_rate=0.0, eval_rate=0.0, batch_size=8)
    assert train is None and val is None
    assert len(test.dataset) == total


def test_loader_yields_normalised_images(cifar_dir):
    data = cf.Cifar10Data(cifar_dir / "cifar10.npz", cifar_dir / "cifar10_clients.json")
    loader, _, _ = data.build_dataset(data.all_clients()[0], batch_size=8)
    x, y = next(iter(loader))
    assert x.shape[1:] == (3, 32, 32)
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
    assert int(y.max()) < 10
    assert float(x.abs().max()) < 10.0


def test_augmentation_only_affects_the_training_loader(cifar_dir):
    data = cf.Cifar10Data(cifar_dir / "cifar10.npz", cifar_dir / "cifar10_clients.json")
    client = data.all_clients()[0]

    plain, _, _ = data.build_dataset(client, batch_size=4, augment=False)
    augmented, _, evaluation = data.build_dataset(client, batch_size=4, augment=True)

    torch.manual_seed(0)
    reference = plain.dataset[0][0]
    torch.manual_seed(0)
    assert torch.allclose(plain.dataset[0][0], reference)
    assert augmented.dataset.augment is True
    assert evaluation.dataset.augment is False
