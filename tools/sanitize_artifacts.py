"""
Normalise absolute machine paths in the shipped artefacts to placeholders.

    python tools/sanitize_artifacts.py --dir study/artifacts
    python tools/sanitize_artifacts.py --dir study/artifacts --check

Every artefact carries provenance: which model produced it, which fold book it
was scored against, which client list it was cut from.  Those fields recorded
the absolute path on the machine that ran the study, which is right for a lab
notebook and wrong for a public release - it pins the reader's attention on
somebody's home directory and, worse, makes the record look machine-specific
when the *content* is not.

So the paths are rewritten to the same placeholders the task files use, and
nothing else is touched.  Only string **values** that look like paths are
considered; no number, no key, no writer id and no accuracy is altered, and a
file whose paths are already placeholders is left byte-identical.  That last
property is what makes this safe to re-run: it is idempotent by construction,
and ``--check`` proves it without writing.

The placeholders are the ones the runner exports, so a reader who has set up the
study can expand them mechanically:

    $FOA_STUDY_DIR/fold_books/cohort20.foldbook.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, List, Tuple

#: Longest first: ``.../studies/<name>`` has to win over its own parent, or the
#: study root would be rewritten as ``$FOA_PROJECT_DIR/results_v4/studies/...``
#: and the placeholder would carry the machine's directory names after all.
RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^/[^\s\"]*?/studies/Digits_study01(?=/|$)"), "$FOA_STUDY_DIR"),
    (re.compile(r"^/[^\s\"]*?/data/nist28(?=/|$)"), "$FOA_NIST28_DIR"),
    (re.compile(r"^/[^\s\"]*?/(?:data|results_v4|results|envs|jobs)(?=/|$)"),
     lambda m: "$FOA_PROJECT_DIR" + m.group(0)[m.group(0).rfind("/"):]),
]

#: A value is a candidate only if it is an absolute POSIX path.  A sentence that
#: happens to contain a slash is prose and is left alone.
ABSOLUTE = re.compile(r"^/[^\s]")


def normalise(value: str) -> str:
    """One string value, with its machine prefix replaced by a placeholder."""
    if not ABSOLUTE.match(value):
        return value
    for pattern, replacement in RULES:
        new, count = pattern.subn(replacement, value, count=1)
        if count:
            return new
    return value


def walk(node: Any) -> Tuple[Any, int]:
    """Rewrite every string value in a JSON tree; return the tree and a count."""
    if isinstance(node, str):
        new = normalise(node)
        return new, int(new != node)
    if isinstance(node, list):
        changed = 0
        out = []
        for item in node:
            value, n = walk(item)
            out.append(value)
            changed += n
        return out, changed
    if isinstance(node, dict):
        changed = 0
        out = {}
        for key, item in node.items():
            value, n = walk(item)
            out[key] = value
            changed += n
        return out, changed
    return node, 0


def process(path: Path, write: bool) -> int:
    """Rewrite one JSON file in place; return how many values changed."""
    try:
        original = path.read_text()
        data = json.loads(original)
    except (OSError, ValueError):
        return 0
    rewritten, changed = walk(data)
    if not changed:
        return 0
    if write:
        # Two spaces, trailing newline: the shape every other artefact of this
        # study is written in, so a sanitised file and a freshly emitted one are
        # not distinguishable by their formatting.
        path.write_text(json.dumps(rewritten, indent=2) + "\n")
    return changed


def leftovers(root: Path) -> List[str]:
    """Absolute paths still present anywhere under ``root``, as file:line."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in {".npz", ".pt"} or path.name == "SHA256SUMS":
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if re.search(r"/user/|/home/[a-z]|/mnt/lustre", line):
                found.append(f"{path}:{number}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="study/artifacts")
    parser.add_argument(
        "--check", action="store_true",
        help="Report what would change and exit non-zero; write nothing.",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"FATAL: no directory {root}.", file=sys.stderr)
        return 1

    total, touched = 0, []
    for path in sorted(root.rglob("*.json")):
        changed = process(path, write=not args.check)
        if changed:
            total += changed
            touched.append(f"{path} ({changed})")

    verb = "would rewrite" if args.check else "rewrote"
    print(f"{verb} {total} path value(s) in {len(touched)} file(s)")
    for line in touched:
        print(f"  {line}")

    remaining = leftovers(root)
    if remaining:
        print(f"\n{len(remaining)} absolute path(s) remain:", file=sys.stderr)
        for line in remaining[:20]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("no absolute machine path remains under", root)
    return 1 if (args.check and total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
