"""
MNIST as a server-side *proxy* set for the NIST provider.

The federated protocol of this study forbids any decision that reads the source
population's data.  A server that wants to watch how much of the source task a
model still solves therefore needs data of its own - public data that belongs to
neither the source writers nor the clients.  For handwritten digits the obvious
candidate is MNIST:

> Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-based learning applied to
> document recognition*, Proceedings of the IEEE 86(11), 1998.

MNIST is derived from the same NIST special databases as SD19 but from a
different writer population and with a different rendering, so it is a genuine
proxy rather than a second sample of the source set.

**The archives are never extracted.**  The four ``idx`` files are downloaded as
``.gz`` and read *in memory* with :mod:`gzip`; the decoded arrays are stored as a
single ``mnist.npz`` next to one JSON index, exactly the way the NIST cache and
the two additional datasets are stored.

Rendering to the NIST input format
----------------------------------
NIST SD19 ``by_write`` scans are bilevel with **black ink (0) on a white
background (255)** - the border of every cached image is pure white.  MNIST is
the other way round (white strokes on a black background), so the proxy images
are **inverted** (``255 - x``) before the standard NIST transform is applied:

    invert -> ``Resize((128, 128))`` -> ``ToTensor`` -> ``Normalize(0.5, 0.5)``

which is :func:`~federated_outlier_adaptation.data.datasets.build_transform`
verbatim, so a proxy batch is indistinguishable in shape, scale and polarity
from a NIST batch.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from federated_outlier_adaptation import config

#: Number of classes of the proxy task; identical to the NIST digit task.
NUM_CLASSES = 10

#: Edge length of a raw MNIST image.
RAW_SIZE = 28

#: The four files of the distribution, keyed by the array they carry.
IDX_FILES: Dict[str, str] = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

#: Mirror the archives are fetched from.  ``yann.lecun.com`` retired the
#: original path; this is the mirror the torchvision loader uses.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

#: NIST SD19 stores black ink on a white background, MNIST the other way round,
#: so the proxy images are inverted before the NIST transform is applied.
INVERT_POLARITY = True

#: Data-type codes of the IDX container format.
_IDX_DTYPES = {
    0x08: np.dtype(np.uint8),
    0x09: np.dtype(np.int8),
    0x0B: np.dtype(">i2"),
    0x0C: np.dtype(">i4"),
    0x0D: np.dtype(">f4"),
    0x0E: np.dtype(">f8"),
}


# ---------------------------------------------------------------- extraction
def decode_idx(payload: bytes) -> np.ndarray:
    """
    Decode one IDX buffer into a numpy array.

    Args:
        payload: The *decompressed* content of an ``idx`` file.

    Returns:
        numpy.ndarray: The array the file describes, in C order.
    """
    if len(payload) < 4:
        raise ValueError("IDX payload is too short to carry a header")
    zero_a, zero_b, code, dimensions = struct.unpack(">BBBB", payload[:4])
    if zero_a or zero_b:
        raise ValueError("Not an IDX buffer: the two leading bytes must be zero")
    dtype = _IDX_DTYPES.get(code)
    if dtype is None:
        raise ValueError(f"Unknown IDX data type code 0x{code:02x}")
    offset = 4 + 4 * dimensions
    shape = struct.unpack(f">{dimensions}I", payload[4:offset])
    array = np.frombuffer(payload, dtype=dtype, offset=offset)
    return np.ascontiguousarray(array.reshape(shape))


def read_gz(path: Path) -> np.ndarray:
    """Read one ``idx*.gz`` archive **in memory**; nothing is written to disk."""
    path = Path(path)
    with open(path, "rb") as handle:
        payload = gzip.decompress(handle.read())
    return decode_idx(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------- preparation
def prepare(
    raw_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    log=print,
) -> dict:
    """
    Build ``mnist.npz`` and ``mnist_index.json`` from the four ``.gz`` archives.

    Args:
        raw_dir: Directory holding the four downloaded archives.  Defaults to
            :data:`~federated_outlier_adaptation.config.MNIST_DIR`.
        out_dir: Destination of the two produced files; defaults to ``raw_dir``.
        log: Progress callable.

    Returns:
        dict: Summary with the two paths, the array shapes and the checksums.
    """
    raw_dir = Path(raw_dir) if raw_dir else config.MNIST_DIR
    out_dir = Path(out_dir) if out_dir else raw_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    sources: Dict[str, dict] = {}
    for key, name in IDX_FILES.items():
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; download the four MNIST archives from {MIRROR}"
            )
        log(f"reading {path}")
        arrays[key] = read_gz(path)
        sources[key] = {"file": name, "sha256": _sha256(path), "bytes": path.stat().st_size}

    if arrays["test_images"].shape[1:] != (RAW_SIZE, RAW_SIZE):
        raise ValueError(f"Unexpected MNIST image shape {arrays['test_images'].shape}")

    npz_path = out_dir / config.MNIST_NPZ_NAME
    index_path = out_dir / config.MNIST_INDEX_NAME
    np.savez(
        npz_path,
        train_images=arrays["train_images"].astype(np.uint8),
        train_labels=arrays["train_labels"].astype(np.uint8),
        test_images=arrays["test_images"].astype(np.uint8),
        test_labels=arrays["test_labels"].astype(np.uint8),
    )

    index = {
        "dataset": "mnist",
        "role": "proxy set of the nist provider",
        "mirror": MIRROR,
        "num_classes": NUM_CLASSES,
        "raw_image_shape": [RAW_SIZE, RAW_SIZE],
        "array_file": npz_path.name,
        "num_train": int(arrays["train_images"].shape[0]),
        "num_test": int(arrays["test_images"].shape[0]),
        "invert_polarity": INVERT_POLARITY,
        "render": "invert -> Resize((128,128)) -> ToTensor -> Normalize(0.5,0.5)",
        "sources": sources,
    }
    with open(index_path, "w") as handle:
        json.dump(index, handle, indent=2)

    summary = {
        "npz": str(npz_path),
        "index": str(index_path),
        "npz_bytes": npz_path.stat().st_size,
        "num_train": index["num_train"],
        "num_test": index["num_test"],
    }
    log(f"{index['num_test']} test images -> {npz_path}")
    return summary


# ------------------------------------------------------------------- access
class MnistProxyDataset(Dataset):
    """
    MNIST images rendered into the NIST input format.

    ``__getitem__`` returns ``(x, y)`` with ``x`` a ``[1, 128, 128]`` float
    tensor produced by the *unmodified* NIST transform pipeline and ``y`` the
    digit label, so the proxy set can be fed to a NIST model as it is.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None,
        invert: bool = INVERT_POLARITY,
    ):
        from federated_outlier_adaptation.data.datasets import build_transform

        self.images = np.asarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels)
        self.invert = bool(invert)
        self.transform = transform if transform is not None else build_transform()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, item):
        array = self.images[item]
        if self.invert:
            array = 255 - array
        image = Image.fromarray(np.ascontiguousarray(array), mode="L")
        return self.transform(image), int(self.labels[item])

    def _pil(self, item) -> Image.Image:
        array = self.images[item]
        if self.invert:
            array = 255 - array
        return Image.fromarray(np.ascontiguousarray(array), mode="L")

    def eval_payload(self):
        """
        Compact form of the proxy set for the evaluation cache.

        The 28x28 sources are enlarged to the NIST input size once, while they
        are still ``uint8`` and by the transform's own ``Resize``, so the cached
        pass and the loader pass see the same pixels.
        """
        from federated_outlier_adaptation.data.datasets import (
            grayscale_payload,
            resize_stage,
        )

        resize = resize_stage(self.transform)
        if resize is None:
            return None
        images = (self._pil(item) for item in range(len(self)))
        return grayscale_payload(images, [int(label) for label in self.labels], resize)


class MnistData:
    """Read-only view over a prepared MNIST proxy set."""

    def __init__(self, npz_path: Optional[Path] = None, index_path: Optional[Path] = None):
        self.npz_path = Path(npz_path) if npz_path else config.MNIST_NPZ
        self.index_path = Path(index_path) if index_path else config.MNIST_INDEX_JSON
        self.index: dict = {}
        if self.index_path.is_file():
            with open(self.index_path) as handle:
                self.index = json.load(handle)

        with np.load(self.npz_path) as archive:
            # Only the test split is materialised: it is the proxy set, and the
            # training split is kept in the file purely for completeness.
            self.test_images: np.ndarray = np.asarray(archive["test_images"], dtype=np.uint8)
            self.test_labels: np.ndarray = np.asarray(archive["test_labels"], dtype=np.uint8)

    @classmethod
    def open_if_available(
        cls, npz_path: Optional[Path] = None, index_path: Optional[Path] = None
    ) -> Optional["MnistData"]:
        """Return a view over the prepared files, or ``None`` when absent."""
        path = Path(npz_path) if npz_path else config.MNIST_NPZ
        if not path.is_file():
            return None
        try:
            return cls(npz_path=path, index_path=index_path)
        except (OSError, ValueError, KeyError):  # pragma: no cover - broken file
            return None

    def __len__(self) -> int:
        return int(self.test_images.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.index.get("num_classes", NUM_CLASSES))

    def rows(self, size: Optional[int] = None, seed: int = 42) -> np.ndarray:
        """
        Row indices of the proxy set, deterministically subsampled if asked.

        ``size`` of ``None`` or a value covering the whole test set returns the
        rows in storage order, so the default proxy set is the full, unshuffled
        MNIST test set.
        """
        total = len(self)
        if size is None or int(size) >= total:
            return np.arange(total, dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total, size=int(size), replace=False))

    def proxy_dataset(
        self, size: Optional[int] = None, seed: int = 42, resolution: Optional[int] = None
    ) -> MnistProxyDataset:
        """
        The proxy set as a torch dataset in the NIST input format.

        Args:
            resolution: Edge length the images are rendered at.  ``None`` keeps
                the published 128x128 rendering; the 28x28 setting passes its
                own so the proxy set matches the model's input.
        """
        from federated_outlier_adaptation.data.datasets import build_transform

        rows = self.rows(size=size, seed=seed)
        transform = build_transform(resolution) if resolution else None
        return MnistProxyDataset(
            self.test_images[rows], self.test_labels[rows], transform=transform
        )


# ---------------------------------------------------------------- entry point
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the MNIST proxy set from the four idx archives (never extracted)."
    )
    parser.add_argument("--raw-dir", default=None, help="Directory holding the four .gz archives.")
    parser.add_argument("--out-dir", default=None, help="Destination of the npz and the index.")
    args = parser.parse_args(argv)

    summary = prepare(raw_dir=args.raw_dir, out_dir=args.out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
