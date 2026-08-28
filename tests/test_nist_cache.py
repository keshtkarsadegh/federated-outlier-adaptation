"""
Dataset cache tests.

The packed cache must be a drop-in replacement for the PNG reader: for the same
image it has to produce a bit-identical tensor after the shared transform.  The
cache is always built by streaming from a zip archive; nothing is extracted.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from federated_outlier_adaptation import config
from federated_outlier_adaptation.data.datasets import (
    MemmapDigitDataset,
    NISTDigitDataset,
    build_transform,
)
from federated_outlier_adaptation.data.nist_cache import NistDigitCache, build_cache

IMAGE_SIZE = 128
N_IMAGES = 6


def _make_png_tree(root: Path, binary: bool) -> dict[str, int]:
    """Write a handful of PNG files mimicking the by_write layout."""
    rng = np.random.default_rng(0)
    labels: dict[str, int] = {}
    for i in range(N_IMAGES):
        rel = f"by_write/hsf_0/f000{i}_00/d000{i}_00/img_{i}.png"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            array = (rng.integers(0, 2, size=(IMAGE_SIZE, IMAGE_SIZE)) * 255).astype(np.uint8)
        else:
            array = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.uint8)
        Image.fromarray(array, mode="L").save(path)
        labels[rel] = i % 10
    return labels


def _zip_tree(root: Path, labels: dict[str, int], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for rel in labels:
            archive.write(root / rel, arcname=rel)


@pytest.mark.parametrize("binary", [True, False], ids=["bilevel", "grayscale"])
def test_cache_matches_png_reader(tmp_path, binary):
    root = tmp_path / "data"
    labels = _make_png_tree(root, binary=binary)

    labels_json = tmp_path / "digits_labels.json"
    labels_json.write_text(json.dumps(labels))

    zip_path = tmp_path / "by_write.zip"
    _zip_tree(root, labels, zip_path)

    out_dir = tmp_path / "cache"
    summary = build_cache(zip_path, labels_json, out_dir, size=IMAGE_SIZE, log=lambda *_: None)

    assert summary["rows"] == N_IMAGES
    assert summary["missing"] == 0
    assert summary["packed"] is binary

    cache = NistDigitCache.open_if_available(out_dir)
    assert cache is not None
    assert len(cache) == N_IMAGES
    assert cache.has_all(list(labels))

    transform = build_transform()
    rel_paths = list(labels)
    png_dataset = NISTDigitDataset(
        [root / rel for rel in rel_paths], [labels[rel] for rel in rel_paths], transform
    )
    cache_dataset = MemmapDigitDataset(
        cache, cache.rows_for(rel_paths), [labels[rel] for rel in rel_paths], transform
    )

    assert len(png_dataset) == len(cache_dataset)
    for i in range(len(png_dataset)):
        png_image, png_label = png_dataset[i]
        cache_image, cache_label = cache_dataset[i]
        assert png_label == cache_label
        if binary:
            assert torch.equal(png_image, cache_image)
        else:
            # Grayscale images are stored verbatim, so they must match exactly too.
            assert torch.equal(png_image, cache_image)


def test_concurrent_readers_all_get_the_packed_cache(tmp_path):
    """
    The default provider is a process-wide singleton and the drivers fan out
    over threads, so several of them reach the lazy cache at the same moment.
    Every one of them has to receive the cache: a thread that sees ``None``
    builds a PNG-backed dataset, which cannot work where the archive was never
    unpacked.
    """
    import threading

    from federated_outlier_adaptation.data.datasets import NistDataset

    root = tmp_path / "data"
    labels = _make_png_tree(root, binary=True)
    labels_json = tmp_path / "digits_labels.json"
    labels_json.write_text(json.dumps(labels))
    zip_path = tmp_path / "by_write.zip"
    _zip_tree(root, labels, zip_path)
    out_dir = tmp_path / "cache"
    build_cache(zip_path, labels_json, out_dir, size=IMAGE_SIZE, log=lambda *_: None)

    dataset = NistDataset(data_dir=root, labels_json=labels_json, cache_dir=out_dir)

    # Slow the open down so the window a second thread could slip through is
    # wide open; without the guard this test fails deterministically.
    from federated_outlier_adaptation.data import nist_cache

    original = nist_cache.NistDigitCache.open_if_available

    def slow_open(cache_dir):
        import time

        time.sleep(0.2)
        return original(cache_dir)

    nist_cache.NistDigitCache.open_if_available = staticmethod(slow_open)
    try:
        seen = []
        barrier = threading.Barrier(4)

        def reader():
            barrier.wait()
            seen.append(dataset.cache)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        nist_cache.NistDigitCache.open_if_available = original

    assert len(seen) == 4
    assert all(entry is not None for entry in seen)
    assert len({id(entry) for entry in seen}) == 1


def test_concurrent_builders_use_the_cache_backend(tmp_path):
    """The same race, seen through ``build_dataset``: no PNG fallback."""
    import threading

    from federated_outlier_adaptation.data.datasets import NistDataset

    root = tmp_path / "data"
    labels = _make_png_tree(root, binary=True)
    labels_json = tmp_path / "digits_labels.json"
    labels_json.write_text(json.dumps(labels))
    zip_path = tmp_path / "by_write.zip"
    _zip_tree(root, labels, zip_path)
    out_dir = tmp_path / "cache"
    build_cache(zip_path, labels_json, out_dir, size=IMAGE_SIZE, log=lambda *_: None)

    dataset = NistDataset(data_dir=root, labels_json=labels_json, cache_dir=out_dir)
    writers = sorted({rel.split("/")[2] for rel in labels})

    backends = []
    barrier = threading.Barrier(len(writers))

    def build(writer):
        barrier.wait()
        _, _, test_loader = dataset.build_dataset(
            [writer], train_rate=0.0, eval_rate=0.0, batch_size=2
        )
        backends.append(type(test_loader.dataset).__name__)

    threads = [threading.Thread(target=build, args=(w,)) for w in writers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert backends
    assert set(backends) == {"MemmapDigitDataset"}


def test_cache_absent_returns_none(tmp_path):
    assert NistDigitCache.open_if_available(tmp_path / "missing") is None


def test_cache_file_names_come_from_config():
    assert config.CACHE_ARRAY_NAME.endswith(".npy")
    assert config.CACHE_INDEX_NAME.endswith(".json")
