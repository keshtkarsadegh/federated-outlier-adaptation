"""
The server-side proxy set of the NIST provider.

A proxy set is public data that belongs to neither the source population
nor the clients, so the server may read it without violating the constraint
the study works under.  Here it is MNIST: the archives must be decoded in
memory and rendered into the NIST input format with the right polarity, and
the subsample must be deterministic, or a signal measured on it means
nothing from one run to the next.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from federated_outlier_adaptation.data import mnist as mnist_data
from federated_outlier_adaptation.providers.nist import NistProvider

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
