#!/usr/bin/env python3
"""
The manifest that says the shipped artefacts are the ones that were shipped.

    python tools/artifact_checksums.py --dir study/artifacts
    python tools/artifact_checksums.py --dir study/jobs
    python tools/artifact_checksums.py --dir study/artifacts --check

`README.md` and `docs/VERIFY.md` have both told a reader to run
``sha256sum -c SHA256SUMS`` in these two directories since the first release.
Under `study/jobs` the manifest listed fifty-one files of which forty-six had
moved into the metadata core and one no longer matched, and under
`study/artifacts` there was no manifest at all - so the first command a reviewer
was asked to run either failed loudly or failed silently. This writes both, from
the tree rather than from a memory of it, which is the only way a checksum file
stays true across a stage being re-run.

WHAT IT WRITES. One line per file, ``<sha256>  ./<path>`` - the shape
``sha256sum`` writes and reads, so `sha256sum -c SHA256SUMS` is the check and
nothing here has to be trusted to do the verifying. Paths are relative to the
directory, so the manifest travels with it.

SORTED BY BYTES, NOT BY LOCALE. ``sha256sum -c`` does not care about order, but
a manifest that reorders itself between two machines shows up as a diff in every
review that follows. Sorting on the raw path, rather than through whatever
collation the shell happened to have, is what makes two runs produce one file.

THE MANIFEST IS NOT IN THE MANIFEST. It cannot be: its own hash is not known
until it is written. ``--check`` therefore also reports files that exist and are
NOT listed, which is the failure ``sha256sum -c`` cannot see - a shipped artefact
that no manifest covers verifies perfectly by saying nothing about it.

RUN IT AFTER `sanitize_artifacts.py`, never before: normalising a machine path
rewrites the file, and a hash taken first would condemn every artefact the
sanitiser touched.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

MANIFEST = "SHA256SUMS"

#: How sha256sum writes a line, and therefore how it reads one.
SEPARATOR = "  "


def digest(path: Path) -> str:
    """One file's SHA-256, read in blocks so a fold book does not have to fit."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def catalogue(root: Path) -> Dict[str, str]:
    """``{relative path: sha256}`` for every file under ``root`` but the manifest."""
    found: Dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name == MANIFEST:
            continue
        found[path.relative_to(root).as_posix()] = digest(path)
    return dict(sorted(found.items()))


def render(entries: Dict[str, str]) -> str:
    return "".join(f"{sha}{SEPARATOR}./{name}\n" for name, sha in entries.items())


def parse(text: str) -> Dict[str, str]:
    """A manifest back into ``{relative path: sha256}``."""
    entries: Dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        sha, _, name = line.partition(SEPARATOR)
        name = name.strip()
        if name.startswith("./"):
            name = name[2:]
        entries[name] = sha.strip()
    return entries


def check(root: Path, entries: Dict[str, str]) -> int:
    """Compare the tree against the manifest on disk; return a process status."""
    path = root / MANIFEST
    if not path.is_file():
        print(f"FATAL: no {path}.", file=sys.stderr)
        return 1
    listed = parse(path.read_text())

    changed = sorted(name for name in entries.keys() & listed.keys()
                     if entries[name] != listed[name])
    missing = sorted(listed.keys() - entries.keys())
    unlisted = sorted(entries.keys() - listed.keys())

    for label, names in (("changed since the manifest", changed),
                         ("listed and not on disk", missing),
                         ("on disk and in no manifest", unlisted)):
        if names:
            print(f"{len(names)} {label}:", file=sys.stderr)
            for name in names[:20]:
                print(f"  ./{name}", file=sys.stderr)
    if changed or missing or unlisted:
        return 1
    print(f"{len(entries)} file(s) under {root} match {MANIFEST}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="study/artifacts",
                        help="Directory to write or check the manifest of.")
    parser.add_argument("--check", action="store_true",
                        help="Compare the tree against the manifest; write nothing.")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"FATAL: no directory {root}.", file=sys.stderr)
        return 1

    entries = catalogue(root)
    if not entries:
        print(f"FATAL: {root} holds no file to checksum.", file=sys.stderr)
        return 1
    if args.check:
        return check(root, entries)

    (root / MANIFEST).write_text(render(entries))
    print(f"wrote {root / MANIFEST}: {len(entries)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
