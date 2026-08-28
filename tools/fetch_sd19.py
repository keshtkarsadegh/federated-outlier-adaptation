"""
Fetch NIST Special Database 19 and verify it against the study's checksums.

    python tools/fetch_sd19.py --dest "$FOA_DATA_DIR/nist"
    python tools/fetch_sd19.py --dest data/nist --check      # verify, download nothing

Downloads the three files the packed cache is built from, resumes a partial
download, and **refuses anything whose SHA-256 does not match**.  Idempotent: a
file already present and correct is left alone and reported as such, so the
script can be re-run after an interruption without re-fetching 542 MB.

Why the checksums are the point
-------------------------------
SD19 has been mirrored widely and the mirrors are not all the same release.  A
byte-different archive will convert, train, and reproduce nothing - silently,
because every number downstream still looks plausible.  So the hashes are the
gate: a mismatch stops the script and the bad file is left in place under a
``.rejected`` name rather than deleted, because a corrupted download and a
different release are worth telling apart.

Why no third-party mirror is built in
-------------------------------------
Only NIST's own host is in the default list.  Baking in someone else's copy
would make this script the thing that decides where your data comes from, and a
mirror that changes hands is a supply-chain problem the reader cannot see.  Pass
your own with ``--mirror`` (repeatable) or ``FOA_SD19_MIRRORS``; the SHA-256
check is what makes any mirror safe to use, so an unverified one adds risk and
no convenience.

    python tools/fetch_sd19.py --dest data/nist --mirror https://example.org/sd19
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

#: NIST's own distribution host for the Special Reference Data collection.
OFFICIAL = "https://s3.amazonaws.com/nist-srd/SD19"

#: Human-readable landing page, for a reader who wants the licence terms.
LANDING = "https://www.nist.gov/srd/nist-special-database-19"

#: The three files, with the SHA-256 the published results were produced from.
#: ``by_class.zip`` itself (~4 GB) is deliberately absent: only its checksum log
#: is needed, because the character labels are recovered by joining the two logs
#: on MD5 rather than by unpacking the second archive.
FILES: Dict[str, str] = {
    "by_write.zip": "39958e28827eb0d7d54f7e4c31c6cc36689b38aa218a4fc1e810c5413e7a35b8",
    "by_write_md5.log": "11e20fff1a3b934270b0b4dfe3e7928e933c2973ae0ec17938d0456be95a8c03",
    "by_class_md5.log": "b2a76dfb555a1fc3764672bb8514455727d28fbdd18df4ce5abbd85a879f430c",
}

#: Read size for hashing: large enough that a 542 MB file is not a syscall
#: benchmark, small enough to stay out of the way on a login node.
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def mirrors(extra: Optional[List[str]] = None) -> List[str]:
    """Where to try, in order: NIST first, then anything the caller supplied."""
    found = [OFFICIAL]
    for source in (extra or []) + (os.environ.get("FOA_SD19_MIRRORS", "").split()):
        source = source.rstrip("/")
        if source and source not in found:
            found.append(source)
    return found


def download(url: str, target: Path) -> bool:
    """
    Fetch one file, resuming a partial one.  True when curl reported success.

    ``-C -`` resumes; a server that cannot resume restarts the transfer, which
    is correct but slow, and is why the checksum is verified afterwards either
    way rather than trusted from a byte count.
    """
    if shutil.which("curl") is None:
        print("FATAL: curl is not installed; it is what makes the download "
              "resumable.", file=sys.stderr)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-fL", "-C", "-", "--retry", "3", "--retry-delay", "5",
         "-o", str(target), url],
        check=False,
    )
    return result.returncode == 0


def acquire(name: str, expected: str, dest: Path, sources: List[str],
            check_only: bool) -> str:
    """
    One file: verify what is there, fetch it when it is not, verify again.

    Returns one of ``ok``, ``fetched``, ``missing``, ``mismatch``.
    """
    target = dest / name
    if target.is_file():
        digest = sha256(target)
        if digest == expected:
            print(f"  {name}: present and verified")
            return "ok"
        if check_only:
            print(f"  {name}: SHA-256 MISMATCH", file=sys.stderr)
            return "mismatch"
        # Not deleted: a truncated download and a different SD19 release are
        # worth telling apart, and only the file itself can say which it is.
        rejected = target.with_suffix(target.suffix + ".rejected")
        print(f"  {name}: SHA-256 mismatch; moving to {rejected.name} and refetching",
              file=sys.stderr)
        print(f"    expected {expected}\n    got      {digest}", file=sys.stderr)
        target.replace(rejected)
    elif check_only:
        print(f"  {name}: absent", file=sys.stderr)
        return "missing"

    for source in sources:
        url = f"{source}/{name}"
        print(f"  {name}: fetching from {source}")
        if not download(url, target):
            continue
        digest = sha256(target)
        if digest == expected:
            print(f"  {name}: verified")
            return "fetched"
        rejected = target.with_suffix(target.suffix + ".rejected")
        print(f"  {name}: SHA-256 mismatch from {source}; kept as {rejected.name}",
              file=sys.stderr)
        print(f"    expected {expected}\n    got      {digest}", file=sys.stderr)
        target.replace(rejected)
    return "mismatch"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", default=None,
        help="Where the files go. Default $FOA_DATA_DIR/nist, else data/nist.",
    )
    parser.add_argument(
        "--mirror", action="append", default=[],
        help="Additional base URL to try after NIST. Repeatable.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify what is already there and download nothing.",
    )
    args = parser.parse_args()

    if args.dest:
        dest = Path(args.dest)
    else:
        dest = Path(os.environ.get("FOA_DATA_DIR", "data")) / "nist"
    sources = mirrors(args.mirror)

    print(f"NIST SD19 -> {dest}")
    print(f"licence and terms: {LANDING}")
    print(f"sources: {', '.join(sources)}")
    if args.check:
        print("(--check: verifying only)")

    results = {name: acquire(name, digest, dest, sources, args.check)
               for name, digest in FILES.items()}

    bad = [n for n, r in results.items() if r in ("mismatch", "missing")]
    if bad:
        print(f"\nFAILED for {bad}.", file=sys.stderr)
        if args.check:
            print("Run without --check to fetch them.", file=sys.stderr)
        else:
            print("A mismatch means a different SD19 release or a corrupt "
                  "transfer; nothing downstream will reproduce. Try another "
                  "mirror with --mirror, or fetch by hand from:", file=sys.stderr)
            print(f"  {LANDING}", file=sys.stderr)
        return 1

    print("\nAll three files present and verified.")
    print("Next: build the packed 28x28 cache (no GPU, ~10 min).")
    print()
    print("  foa prepare-data --dataset nist \\")
    print(f"      --zip {dest / 'by_write.zip'} \\")
    print('      --out "$FOA_NIST28_DIR" \\')
    print("      --resolution 28 --classes all")
    print()
    print("  --resolution 28 is NOT the default (128 is). A 128 px cache will")
    print("  train happily and reproduce nothing. See docs/DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
