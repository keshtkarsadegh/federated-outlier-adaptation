"""
Submit a study chain from its manifest - or refuse, having submitted nothing.

    python tools/submit_chain.py jobs/v4/digits.chain.toml
    python tools/submit_chain.py jobs/v4/digits.chain.toml --go

Also available as ``foa submit <manifest>``.

Without ``--go`` this prints exactly what it would submit and touches nothing.
With it, the chain goes in as one wired sequence, and a stage→job-id table is
written into the study's ``logs/`` so that "which jobs were this chain?" has an
answer that does not depend on anyone's scrollback.

The preflight runs first and always. If any part of it fails, **no job is
submitted at all** - a half-submitted chain is worse than none, because the half
that ran has to be found and cancelled.

See :mod:`federated_outlier_adaptation.submission` for what a manifest declares
and why this exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from federated_outlier_adaptation.submission import (  # noqa: E402
    ManifestError, load_manifest, submit,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="The chain manifest (TOML).")
    parser.add_argument(
        "--go", action="store_true",
        help="Actually submit. Without it nothing is submitted and the "
             "invocations are printed.",
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 78

    result = submit(manifest, dry_run=not args.go)
    if not result.get("submitted") and result.get("problems"):
        return 78
    if not result.get("submitted") and result.get("failed_at"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
