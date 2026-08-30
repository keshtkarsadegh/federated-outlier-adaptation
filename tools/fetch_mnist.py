#!/usr/bin/env python3
"""
Fetch the MNIST archives the proxy set is built from, and pin them by hash.

    python tools/fetch_mnist.py --dest "$FOA_DATA_DIR/mnist"
    python tools/fetch_mnist.py --dest data/mnist --check   # verify only

WHY THIS EXISTS. Two of the eight forgetting signals - proxy accuracy and proxy
KL - are computed on public data the server is allowed to own, and in this study
that is the MNIST test set. When the archives are absent the provider reports no
proxy set and both signals record ``None`` for every round, silently: the runs
succeed, the summaries are complete in every other respect, and the two columns
are simply empty. That happened, on a stage that cost forty GPU-hours, which is
why fetching them is a step in the programme rather than something someone
remembers to do.

MNIST is mirrored as widely as SD19 and with the same hazard: a different
rendering still trains and still reproduces nothing. So the hashes are the gate,
exactly as in ``fetch_sd19.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

#: The four idx archives, and the SHA-256 this study was built against.
FILES = {
    "train-images-idx3-ubyte.gz":
        "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz":
        "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz":
        "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz":
        "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--check", action="store_true",
                    help="Verify what is on disk; download nothing.")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    bad = 0
    for name, want in FILES.items():
        path = args.dest / name
        if not path.is_file():
            if args.check:
                print(f"  {name}: MISSING")
                bad += 1
                continue
            print(f"  {name}: downloading")
            subprocess.run(["curl", "-fsSL", "-o", str(path), MIRROR + name], check=True)

        got = digest(path)
        if want and got != want:
            # A corrupted download and a different release are worth telling
            # apart, so the file is set aside rather than deleted.
            path.rename(path.with_suffix(path.suffix + ".rejected"))
            print(f"  {name}: SHA-256 MISMATCH\n      want {want}\n      got  {got}")
            bad += 1
        else:
            print(f"  {name}: {'verified' if want else got}")

    if bad:
        print(f"\n{bad} file(s) wrong or missing.")
        return 1
    print("\nAll four archives present and verified.")
    print("Next: foa prepare-data --dataset mnist --raw-dir <dest>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
