#!/usr/bin/env python3
"""
The release asset: every stored run record, packed and scrubbed.

    python tools/build_records_asset.py --root "$FOA_STUDY_DIR" \
        --out "$FOA_STUDY_DIR/release/Digits_study01_records.tar.gz"

Half a gigabyte of machine output does not belong in git, and the metadata core
under ``study/artifacts/`` deliberately carries none of it. So the numbers every
table is computed from travel as an asset instead: the ``accuracies_*.json`` and
``summary_0.json`` of every run folder, plus the reference-rung evaluations, laid
out relative to the study root so the archive unpacks straight over one. Neither
half is a study root on its own; ``docs/REPRODUCE.md`` §10 says how they are
brought together.

The asset had been built by hand at a terminal, which is the one thing this
repository's generators exist to make impossible: nobody could rebuild it, and
nobody could say what was in it without unpacking it. This is that command.

WHAT GOES IN IS A CELL LIST, NOT A GLOB. A run folder that a stage no longer
defines is still on the disk that produced the study - a run that happened
cannot be un-run - and packing it would publish an arm no shipped table names.
Membership is decided by ``report_tables.in_study``, the same predicate every
reader uses, so the asset and the tables cannot come to disagree about what the
study is.

NOTHING IS PACKED THAT HAS NOT BEEN SCRUBBED. Every record is copied into a
staging tree and passed through ``sanitize_artifacts`` there, so the absolute
paths of the machine that ran the study never reach the archive and the
originals under the study root are never written to. The scrub is idempotent, so
a rebuild of an already-clean tree changes nothing and still proves it.

THE ARCHIVE IS DETERMINISTIC. Entries are sorted, and their owner, mode and
timestamp are fixed, so two builds of the same records produce the same bytes
and the published checksum is a property of the records rather than of the
afternoon they were packed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import sanitize_artifacts  # noqa: E402
from report_tables import in_study  # noqa: E402  - one definition of membership

#: The two file names a run folder contributes. The per-round series and the
#: final evaluation are what every table reads; the checkpoints beside them
#: answer no question the records do not, and are not published.
RUN_FILES = ("summary_0.json", "accuracies_*.json")

#: Top-level folders that are not runs but are read as references: the isolated
#: and centralized rungs every federated number is measured against. Their
#: evaluations are flat JSON files rather than a run tree.
REFERENCE_PREFIXES = ("isolated_", "centralized_")

#: A fixed timestamp for every entry. Any constant does; this one is the epoch,
#: which is obviously not a date anybody should read as provenance.
FIXED_MTIME = 0


def run_records(root: Path, tag: str) -> List[Path]:
    """Every packed file under the study's own run folders, sorted."""
    found: List[Path] = []
    for folder in sorted(root.glob(f"{tag}*")):
        if not folder.is_dir() or not in_study(folder.name):
            continue
        for pattern in RUN_FILES:
            found += sorted(folder.rglob(pattern))
    return found


def reference_records(root: Path) -> List[Path]:
    """The reference rungs' evaluation files, sorted."""
    found: List[Path] = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and folder.name.startswith(REFERENCE_PREFIXES):
            found += sorted(folder.rglob("*.json"))
    return found


def stage(root: Path, records: List[Path], staging: Path) -> int:
    """Copy the records into ``staging`` and scrub them there. Returns changes."""
    changed = 0
    for path in records:
        target = staging / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        changed += sanitize_artifacts.process(target, write=True)
    return changed


def pack(staging: Path, out: Path) -> None:
    """A deterministic gzip of the staged tree, entries sorted."""
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(p for p in staging.rglob("*"))
    with open(out, "wb") as handle:
        # mtime=0 rather than now: the gzip header carries a timestamp, and a
        # timestamp is the one byte that would change on a rebuild of identical
        # records.
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=FIXED_MTIME) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in entries:
                    info = archive.gettarinfo(str(path),
                                              arcname=str(path.relative_to(staging)))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = FIXED_MTIME
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_dir():
                        archive.addfile(info)
                    else:
                        with open(path, "rb") as body:
                            archive.addfile(info, body)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="The study root the records are read from.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where the .tar.gz is written; .sha256 goes beside it.")
    ap.add_argument("--study-tag", default="d01",
                    help="Prefix the study's run folders carry.")
    ap.add_argument("--staging", type=Path, default=None,
                    help="Where the scrubbed copies are staged "
                         "(default: beside --out, removed afterwards).")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"FATAL: {args.root} is not a study root.")

    runs = run_records(args.root, args.study_tag)
    references = reference_records(args.root)
    records = runs + references
    if not records:
        raise SystemExit(
            f"FATAL: no records under {args.root}. The asset is a read of the "
            "stored runs and there is nothing here to pack."
        )
    top = lambda paths: {p.relative_to(args.root).parts[0] for p in paths}
    folders = top(records)
    print(f"{len(runs)} run records under {len(top(runs))} run folders, "
          f"{len(references)} reference records under {len(top(references))} "
          "reference folders")

    staging = args.staging or args.out.parent / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        changed = stage(args.root, records, staging)
        print(f"{changed} machine paths rewritten in the staged copies "
              f"(the originals are not written to)")
        left = sanitize_artifacts.leftovers(staging)
        if left:
            raise SystemExit(
                f"FATAL: {len(left)} absolute path(s) survived the scrub, "
                f"first {left[0]}. Refusing to publish them."
            )
        pack(staging, args.out)
    finally:
        if args.staging is None:
            shutil.rmtree(staging, ignore_errors=True)

    digest = sha256(args.out)
    (args.out.parent / (args.out.name + ".sha256")).write_text(
        f"{digest}  {args.out.name}\n")
    print(f"  -> {args.out}  {args.out.stat().st_size:,} bytes")
    print(f"  -> {args.out}.sha256  {digest}")
    print(f"{len(records)} files across {len(folders)} top-level entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
