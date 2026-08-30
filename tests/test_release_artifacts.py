"""
The three release guarantees: clean provenance, a fetchable source, live tables.

Each of these exists because the failure it prevents is silent. A shipped
artefact carrying an absolute home directory still parses and still verifies; a
mirror serving a different SD19 release still converts and still trains; a
hand-maintained table still renders long after the runs it describes have moved.
Nothing downstream complains in any of the three cases.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = str(REPO / "tools")
ARTIFACTS = REPO / "study" / "artifacts"


@pytest.fixture()
def tools():
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    return TOOLS


# --------------------------------------------------------------------------- #
# 1. provenance paths
# --------------------------------------------------------------------------- #
#: What must never appear in a released artefact: somebody's home directory.
ABSOLUTE_HOME = re.compile(r"/user/|/home/[a-z]|/mnt/lustre|/Users/")


def test_no_absolute_machine_path_survives_in_the_release():
    """
    The whole point of the normalisation, asserted over the shipped tree.

    Checked on the raw text rather than on parsed JSON, so a path hidden in a
    key, a list, or a free-text field is caught as readily as one in the field
    the sanitiser was written for.
    """
    offenders = []
    for path in sorted(ARTIFACTS.rglob("*")):
        if not path.is_file() or path.suffix in {".npz", ".pt"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if ABSOLUTE_HOME.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{number}")
    assert offenders == [], f"absolute machine paths in the release: {offenders[:10]}"


def test_the_sanitiser_is_idempotent(tools, tmp_path):
    """
    Re-running it must be a no-op, or a release step could not be repeated
    safely after a partial run.
    """
    import sanitize_artifacts

    target = tmp_path / "a.json"
    target.write_text(json.dumps(
        {"model": "/user/x/.project/p/results_v4/studies/Digits_study01/g0_model",
         "accuracy": 0.5, "writers": ["f0000_14"]}, indent=2))
    first = sanitize_artifacts.process(target, write=True)
    assert first == 1
    once = target.read_text()
    assert sanitize_artifacts.process(target, write=True) == 0
    assert target.read_text() == once


def test_the_sanitiser_touches_only_paths(tools):
    """Numbers, writer ids and prose are not paths and must survive untouched."""
    import sanitize_artifacts

    payload = {
        "accuracy": 0.9168490153172867,
        "samples": 457,
        "writers": ["f3642_03", "f2248_68"],
        "rule": "the model's accuracy on every row the writer holds",
        "note": "see results/ for the layout",          # relative: not a path value
        "model": "/nfs/anything/studies/Digits_study01/g0_model",
    }
    out, changed = sanitize_artifacts.walk(payload)
    assert changed == 1
    assert out["model"] == "$FOA_STUDY_DIR/g0_model"
    for key in ("accuracy", "samples", "writers", "rule", "note"):
        assert out[key] == payload[key]


# --------------------------------------------------------------------------- #
# 2. the SD19 fetcher
# --------------------------------------------------------------------------- #
def test_the_documented_checksums_are_the_ones_the_fetcher_enforces(tools):
    """
    One set of hashes, three places that quote them. If the fetcher and the
    release manifest ever disagree, a reviewer verifies against one thing and
    downloads another.
    """
    import fetch_sd19

    manifest = (REPO / "study" / "UPSTREAM.sha256").read_text()
    for name, digest in fetch_sd19.FILES.items():
        assert f"{digest}  {name}" in manifest, name
    assert "by_class.zip" not in fetch_sd19.FILES, (
        "by_class.zip is ~4 GB and is not needed; only its checksum log is"
    )
    data_doc = (REPO / "docs" / "DATA.md").read_text()
    for digest in fetch_sd19.FILES.values():
        assert digest in data_doc


def test_only_the_official_host_is_built_in(tools):
    """
    A third-party mirror baked in would decide where a reader's data comes from,
    and a mirror that changes hands is invisible to them.
    """
    import fetch_sd19

    assert fetch_sd19.mirrors() == [fetch_sd19.OFFICIAL]
    assert "nist" in fetch_sd19.OFFICIAL
    extra = fetch_sd19.mirrors(["https://example.org/sd19/"])
    assert extra == [fetch_sd19.OFFICIAL, "https://example.org/sd19"]


def test_a_wrong_hash_is_refused_and_the_file_is_kept(tools, tmp_path):
    """
    Refused, and preserved: a truncated download and a different SD19 release
    are worth telling apart, and only the bytes can say which it was.
    """
    import fetch_sd19

    target = tmp_path / "by_write_md5.log"
    target.write_text("not the real archive")
    result = fetch_sd19.acquire(
        "by_write_md5.log", fetch_sd19.FILES["by_write_md5.log"],
        tmp_path, sources=[], check_only=False,
    )
    assert result == "mismatch"
    assert not target.exists()
    assert (tmp_path / "by_write_md5.log.rejected").is_file()


def test_a_correct_file_is_left_alone(tools, tmp_path):
    """Idempotent: re-running after an interruption must not refetch 542 MB."""
    import fetch_sd19
    import hashlib

    body = b"a small stand-in for a large archive"
    target = tmp_path / "by_write.zip"
    target.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    before = target.stat().st_mtime_ns
    assert fetch_sd19.acquire("by_write.zip", digest, tmp_path, [], False) == "ok"
    assert target.stat().st_mtime_ns == before


# --------------------------------------------------------------------------- #
# 3. the table generator
# --------------------------------------------------------------------------- #
def test_the_scaling_table_pins_one_fold_explicitly(tools):
    """
    P18 and P19 have five folds on disk and P20 has one. A generator that read
    "whatever exists" would average five for some rungs and one for others, and
    the difference would read as a size effect.
    """
    import make_tables

    assert make_tables.FOLD_ONLY == 1
    source = (REPO / "tools" / "make_tables.py").read_text()
    body = source[source.index("def scaling_table"):source.index("def _cell")]
    assert "CV_FOLDS" not in body, "the scaling table must not average folds"
    assert "FOLD_ONLY" in body


def test_the_ten_client_anchor_is_a_fold_one_run_not_the_cv_mean(tools):
    """Every cell in the scaling table is one draw, anchors included."""
    import make_tables

    anchors = [row[2] for row in make_tables.SCALING_ROWS if row[0].startswith("10")]
    assert "combo_trimmed_0p4_feature_l2_lam0p1" in anchors
    # The master ladder reads the same run over CV_FOLDS; the scaling table
    # must not, and _cell is the only path it uses.
    source = (REPO / "tools" / "make_tables.py").read_text()
    cell = source[source.index("def _cell"):source.index("def _balanced_note")]
    assert "FOLD_ONLY" in cell and "CV_FOLDS" not in cell


def test_both_shipped_tables_are_what_the_generator_emits():
    """
    The regression this file exists for. A table that no longer matches its runs
    must fail loudly rather than be quietly overwritten - which is why the
    generator has a --check mode and why it is exercised here.

    Skipped without the run folders: a clone has the tables but not the 1,900
    run directories they were computed from.
    """
    import os

    root = os.environ.get("FOA_STUDY_DIR")
    if not root or not (Path(root) / "tables").is_dir():
        pytest.skip("no study root; set FOA_STUDY_DIR to run the full check")
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import make_tables

    for name, builder in (("master", make_tables.master_table),
                          ("scaling", make_tables.scaling_table)):
        shipped = (ARTIFACTS / "tables" / f"{name}_table.md").read_text()
        assert builder(Path(root)) == shipped, f"{name}_table.md is out of date"
