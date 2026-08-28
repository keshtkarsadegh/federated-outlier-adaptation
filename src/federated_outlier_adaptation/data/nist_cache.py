"""
Compact, inode-friendly cache of the NIST SD19 digit images.

The raw ``by_write`` archive contains roughly 400 000 PNG files.  Unpacking it
exhausts the inode quota of most shared file systems, so the cache is built by
streaming **directly from ``by_write.zip``** - the archive is never extracted,
not even partially and not to a temporary directory.

Two files are produced:

``nist_digits_u8.npy``
    ``uint8`` array of shape ``[N, 128, 128]``, or packed bits of shape
    ``[N, 2048]`` when every image is strictly black/white (the usual case for
    SD19, which stores bilevel scans).
``nist_digits_index.json``
    ``{"paths", "labels", "writers", "packed", "shape", "missing"}``.

Row ``i`` corresponds to ``paths[i]``; the row order follows the iteration order
of ``digits_labels.json`` so the cache and the manifest stay aligned.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from federated_outlier_adaptation import config

DEFAULT_IMAGE_SIZE = 128
PROBE_DEFAULT = 2000


def _load_image(zip_file: zipfile.ZipFile, name: str, size: int) -> np.ndarray:
    """Decode one archive member into a ``uint8`` grayscale array."""
    with zip_file.open(name) as handle:
        image = Image.open(io.BytesIO(handle.read())).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def build_cache(
    zip_path: Path,
    labels_json: Path,
    out_dir: Path,
    size: int = DEFAULT_IMAGE_SIZE,
    probe: int = PROBE_DEFAULT,
    progress_every: int = 20000,
    log=print,
) -> dict:
    """
    Build the packed cache from ``by_write.zip`` without extracting it.

    Args:
        zip_path: Path of ``by_write.zip``.
        labels_json: ``digits_labels.json`` manifest.
        out_dir: Destination directory for the two cache files.
        size: Target edge length (images are already 128x128 in SD19).
        probe: Number of leading images inspected to decide whether the data is
            bilevel and can therefore be bit-packed.
        progress_every: Progress log interval in images.
        log: Callable used for progress output.

    Returns:
        A summary dict with ``rows``, ``packed``, ``seconds``, ``missing`` and
        the two output paths.
    """
    zip_path = Path(zip_path)
    labels_json = Path(labels_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_json) as handle:
        label_dict = json.load(handle)

    paths = list(label_dict.keys())
    labels = [int(label_dict[p]) for p in paths]
    writers = [p.split("/")[2] for p in paths]
    n = len(paths)
    log(f"{n} images, target {size}x{size}")

    zip_file = zipfile.ZipFile(zip_path)
    names = set(zip_file.namelist())
    missing = [p for p in paths if p not in names]
    if missing:
        log(f"WARNING: {len(missing)} manifest paths are absent from the archive, e.g. {missing[:3]}")

    binary = True
    for path in paths[:probe]:
        if path not in names:
            continue
        array = _load_image(zip_file, path, size)
        if not np.isin(array, (0, 255)).all():
            binary = False
            break
    log(f"binary={binary}")

    array_path = out_dir / config.CACHE_ARRAY_NAME
    index_path = out_dir / config.CACHE_INDEX_NAME
    shape = (n, size * size // 8) if binary else (n, size, size)
    store = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.uint8, shape=shape)

    t0 = time.time()
    for i, path in enumerate(paths):
        if path not in names:
            continue
        array = _load_image(zip_file, path, size)
        if binary:
            if not np.isin(array, (0, 255)).all():
                # Defensive fallback: the probe was not representative.
                array = np.where(array >= 128, 255, 0).astype(np.uint8)
            store[i] = np.packbits(array.reshape(-1) > 127)
        else:
            store[i] = array
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            log(f"{i + 1}/{n}  {elapsed:.0f}s  eta {eta:.0f}s")
    store.flush()
    del store
    zip_file.close()

    meta = {
        "paths": paths,
        "labels": labels,
        "writers": writers,
        "packed": bool(binary),
        "shape": [size, size],
        "missing": missing,
    }
    with open(index_path, "w") as handle:
        json.dump(meta, handle)

    seconds = time.time() - t0
    log(f"done in {seconds:.0f}s -> {out_dir}")
    return {
        "rows": n,
        "packed": bool(binary),
        "seconds": seconds,
        "missing": len(missing),
        "array_path": str(array_path),
        "index_path": str(index_path),
    }


class NistDigitCache:
    """
    Read-only view over a built cache.

    ``image(row)`` returns a PIL ``L`` image so the very same transform pipeline
    can be applied as for the PNG reader.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        index_path = self.cache_dir / config.CACHE_INDEX_NAME
        array_path = self.cache_dir / config.CACHE_ARRAY_NAME
        with open(index_path) as handle:
            meta = json.load(handle)
        self.paths: List[str] = meta["paths"]
        self.labels: List[int] = meta["labels"]
        self.writers: List[str] = meta["writers"]
        self.packed: bool = bool(meta["packed"])
        self.height, self.width = meta["shape"]
        self.missing = set(meta.get("missing") or [])
        self._array = np.load(array_path, mmap_mode="r")
        self._row_of = {path: i for i, path in enumerate(self.paths)}

    # ------------------------------------------------------------------ lookup
    @classmethod
    def open_if_available(cls, cache_dir: Path) -> Optional["NistDigitCache"]:
        """Return a cache instance, or ``None`` when the files are absent."""
        cache_dir = Path(cache_dir)
        if not (cache_dir / config.CACHE_INDEX_NAME).is_file():
            return None
        if not (cache_dir / config.CACHE_ARRAY_NAME).is_file():
            return None
        try:
            return cls(cache_dir)
        except (OSError, ValueError, KeyError):  # pragma: no cover - broken cache
            return None

    def __len__(self) -> int:
        return len(self.paths)

    def has_all(self, rel_paths: Iterable[str]) -> bool:
        """True when every requested path is present (and not marked missing)."""
        for path in rel_paths:
            if path not in self._row_of or path in self.missing:
                return False
        return True

    def rows_for(self, rel_paths: Sequence[str]) -> List[int]:
        """Map relative image paths to cache rows."""
        return [self._row_of[path] for path in rel_paths]

    def array(self, row: int) -> np.ndarray:
        """Return the raw ``uint8`` image of a row."""
        data = self._array[row]
        if self.packed:
            bits = np.unpackbits(np.asarray(data, dtype=np.uint8))
            return (bits[: self.height * self.width] * 255).astype(np.uint8).reshape(
                self.height, self.width
            )
        return np.asarray(data, dtype=np.uint8)

    def image(self, row: int) -> Image.Image:
        """Return the row as a PIL grayscale image."""
        return Image.fromarray(self.array(row), mode="L")


def default_cache() -> Optional[NistDigitCache]:
    """Open the cache at the configured location, if it exists."""
    return NistDigitCache.open_if_available(config.CACHE_DIR)
