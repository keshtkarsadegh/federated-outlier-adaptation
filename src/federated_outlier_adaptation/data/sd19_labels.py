"""
By-class labels for the ``by_write`` hierarchy of NIST SD19.

SD19 ships the same 814 255 scans twice: ``by_class`` carries the character
label in the directory name and ``by_write`` carries the writer id.  The two
hierarchies are joined by the MD5 checksum of the image file, which NIST
publishes as two plain text logs next to the archives:

    https://s3.amazonaws.com/nist-srd/SD19/by_class_md5.log
    https://s3.amazonaws.com/nist-srd/SD19/by_write_md5.log

Each line is ``<md5> <path>``.  In ``by_class`` the first path component after
``by_class/`` is the character's ASCII code in hexadecimal - ``30``-``39`` for
the digits, ``41``-``5a`` for the upper-case and ``61``-``7a`` for the
lower-case letters - so the label is read straight off the path, and the writer
comes from the matching ``by_write`` line.

This is the same join the digit-only pipeline performs in
:mod:`federated_outlier_adaptation.data.labels`, generalised from ten classes
to all sixty-two and reduced to two streamed passes over the logs, so no
archive is ever extracted and no per-image file is ever written.

Class order
-----------
:data:`CLASS_ORDER` is the by-class order of EMNIST and of every FEMNIST
implementation: ``0-9`` are the digits, ``10-35`` the upper-case and ``36-61``
the lower-case letters.  The digit ablation keeps labels ``0-9`` unchanged, so
a digit run and a 62-class run agree on what a "3" is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from federated_outlier_adaptation.logging_utils import NistLogger

#: URLs of the two checksum logs.
BY_CLASS_MD5_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class_md5.log"
BY_WRITE_MD5_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_write_md5.log"

#: File names the logs are stored under.
BY_CLASS_MD5_NAME = "by_class_md5.log"
BY_WRITE_MD5_NAME = "by_write_md5.log"

#: Characters in by-class order: digits, upper case, lower case.
CLASS_ORDER = (
    [chr(code) for code in range(0x30, 0x3A)]
    + [chr(code) for code in range(0x41, 0x5B)]
    + [chr(code) for code in range(0x61, 0x7B)]
)

#: Character -> label index.
CLASS_INDEX = {character: index for index, character in enumerate(CLASS_ORDER)}

#: Selectable character sets.  ``digits`` is the ablation of plan v4 section 1.
CLASS_SETS = ("all", "digits")

#: Number of classes of each set.
CLASS_COUNTS = {"all": 62, "digits": 10}


def class_label(hex_code: str) -> Optional[int]:
    """
    Label index of a ``by_class`` directory name, or ``None`` when it is not one.

    Args:
        hex_code: The directory name, e.g. ``"41"`` for ``A``.
    """
    try:
        character = chr(int(hex_code, 16))
    except ValueError:
        return None
    return CLASS_INDEX.get(character)


def class_map(classes: str = "all") -> Dict[int, str]:
    """The label -> character map of a character set."""
    if classes not in CLASS_SETS:
        raise ValueError(f"Unknown class set {classes!r}; expected one of {CLASS_SETS}")
    limit = CLASS_COUNTS[classes]
    return {index: CLASS_ORDER[index] for index in range(limit)}


def download_md5_logs(target_dir) -> Tuple[Path, Path]:
    """
    Fetch the two checksum logs unless they are already present.

    Args:
        target_dir: Directory the logs are stored in (next to the archives).

    Returns:
        ``(by_class_log, by_write_log)``.
    """
    from federated_outlier_adaptation.data.download import download_file

    target_dir = Path(target_dir)
    by_class = download_file(BY_CLASS_MD5_URL, target_dir, BY_CLASS_MD5_NAME)
    by_write = download_file(BY_WRITE_MD5_URL, target_dir, BY_WRITE_MD5_NAME)
    return by_class, by_write


def _iter_log(path: Path) -> Iterable[Tuple[str, str]]:
    """Stream ``(md5, path)`` pairs of a checksum log."""
    with open(path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def hash_to_label(by_class_log, classes: str = "all") -> Dict[str, int]:
    """
    Map every image checksum of ``by_class`` to its label index.

    Args:
        by_class_log: The ``by_class_md5.log`` file.
        classes: ``"all"`` (62 classes) or ``"digits"`` (labels 0-9 only, the
            remaining hashes are dropped).

    Returns:
        ``{md5: label}``.
    """
    if classes not in CLASS_SETS:
        raise ValueError(f"Unknown class set {classes!r}; expected one of {CLASS_SETS}")
    limit = CLASS_COUNTS[classes]

    table: Dict[str, int] = {}
    for digest, path in _iter_log(Path(by_class_log)):
        if "/by_class/" in path:
            tail = path.split("/by_class/", 1)[1]
        elif path.startswith("by_class/"):
            tail = path[len("by_class/") :]
        else:
            continue
        label = class_label(tail.split("/", 1)[0])
        if label is None or label >= limit:
            continue
        table[digest] = label
    return table


def build_label_map(
    by_class_log,
    by_write_log,
    classes: str = "all",
    log=NistLogger.info,
) -> Dict[str, int]:
    """
    Map every ``by_write`` image path to its character label.

    Args:
        by_class_log: ``by_class_md5.log``.
        by_write_log: ``by_write_md5.log``.
        classes: ``"all"`` or ``"digits"``.
        log: Callable used for progress output.

    Returns:
        ``{relative path inside the archive: label}``, e.g.
        ``{"by_write/hsf_0/f0000_14/d0000_14/d0000_14_00000.png": 3}``.  The
        paths are exactly the archive member names, so the cache builder can
        look them up in the zip without any further transformation.
    """
    by_hash = hash_to_label(by_class_log, classes=classes)
    log(f"{len(by_hash)} labelled checksums from {Path(by_class_log).name}")

    labels: Dict[str, int] = {}
    for digest, path in _iter_log(Path(by_write_log)):
        label = by_hash.get(digest)
        if label is None:
            continue
        parts = path.split("/")
        if "by_write" not in parts:
            continue
        relative = "/".join(parts[parts.index("by_write") :])
        labels[relative] = label
    log(f"{len(labels)} by_write images labelled ({classes})")
    return labels


def writer_of(relative_path: str) -> Optional[str]:
    """
    Writer id of a ``by_write`` path, e.g. ``f0000_14``.

    The layout is ``by_write/hsf_<n>/<writer>/<partition>/<image>.png``, so the
    writer is the third component - the same rule the digit pipeline uses.
    """
    parts = relative_path.split("/")
    return parts[2] if len(parts) > 2 else None


def write_label_map(labels: Dict[str, int], path) -> Path:
    """Store a label map as JSON, in the shape of ``digits_labels.json``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(labels, handle)
    return path
