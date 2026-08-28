"""
Dataset download helpers.

The NIST SD19 ``by_write`` archive holds roughly 400 000 files.  It is
deliberately **never extracted**: unpacking it exhausts the inode quota of
shared and cluster file systems.  The archive is downloaded once and then read
in place by
:func:`federated_outlier_adaptation.data.nist_cache.build_cache`, which
streams the images out of the zip and stores them in a single ``.npy`` file.

The functions here therefore only fetch files; nothing in the package unpacks
an archive.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

from federated_outlier_adaptation import config
from federated_outlier_adaptation.logging_utils import NistLogger

NIST_BY_WRITE_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
NIST_BY_CLASS_URL = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"


def download_file(url: str, target_dir: Path, filename: Optional[str] = None) -> Path:
    """
    Download ``url`` into ``target_dir`` unless the file is already there.

    Args:
        url: Source URL.
        target_dir: Destination directory (created if needed).
        filename: Override for the local file name.

    Returns:
        Path of the downloaded file.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / (filename or os.path.basename(url))

    if target_path.exists():
        NistLogger.info(f"File already exists, skipping download: {target_path}")
        return target_path

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB

    downloaded = 0
    with open(target_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(
                    f"\rDownloading {target_path.name}: {percent:.1f}% "
                    f"({downloaded / 1024 / 1024:.2f} MB)",
                    end="",
                )
    print(f"\nDownload complete: {target_path}")
    return target_path


def download_nist_archive(target_dir: Optional[Path] = None, url: str = NIST_BY_WRITE_URL) -> Path:
    """
    Fetch the NIST SD19 archive used by the experiments.

    The archive stays zipped; see the module docstring.
    """
    target_dir = Path(target_dir) if target_dir else config.DATA_DIR / "nist"
    return download_file(url, target_dir)


def download_hash_by_class(url: str, target_dir: Path) -> Path:
    """Fetch one of the SD19 checksum manifests."""
    return download_file(url, target_dir)


if __name__ == "__main__":
    download_nist_archive()
