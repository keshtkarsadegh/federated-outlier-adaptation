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
import os
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


def test_every_hash_printed_in_the_data_doc_is_in_the_manifest():
    """
    The test above pins the three upstream archives, which the fetcher knows
    about.  It says nothing about the four DERIVED hashes DATA.md prints for
    the packed cache, and those are the ones a reader checks their own build
    against -- so a stale one there sends a reader hunting a corruption that
    is really a typo in the prose.  study/UPSTREAM.sha256 is the manifest of
    record; every ``<sha256>  <name>`` line in the doc has to appear in it.
    """
    manifest = (REPO / "study" / "UPSTREAM.sha256").read_text()
    doc = (REPO / "docs" / "DATA.md").read_text()
    printed = re.findall(r"^([0-9a-f]{64})  (\S+)$", doc, re.M)
    assert printed, "DATA.md prints no checksum lines at all"
    for digest, name in printed:
        assert f"{digest}  {name}" in manifest, (
            f"docs/DATA.md prints {digest[:12]}... for {name}, which is not "
            f"what study/UPSTREAM.sha256 records for it"
        )


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

    **The two tables live in the study root, not in the metadata core.** They
    are `make_tables.py`'s own markdown summaries of the run folders, not views
    the manuscript reads - `report_tables.py` writes those - so the core does
    not carry them and neither does the records asset. The comparison is
    therefore against the root's own copy, and is skipped by name where there is
    none rather than failing on a file that was never meant to be there.
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
        written = Path(root) / "tables" / f"{name}_table.md"
        if not written.is_file():
            pytest.skip(f"{written} is not on this root; make_tables.py writes it")
        assert builder(Path(root)) == written.read_text(), (
            f"{name}_table.md is out of date"
        )


# --------------------------------------------------------------------------- #
# 3. the command docs/DATA.md tells a reader to verify with
# --------------------------------------------------------------------------- #
#: The seven entries live in two directories, and no `cd` reaches both.
UPSTREAM = REPO / "study" / "UPSTREAM.sha256"


def _upstream_names():
    return re.findall(r"^[0-9a-f]{64}  (\S+)$", UPSTREAM.read_text(), re.M)


def _bash_blocks(doc: Path):
    return re.findall(r"```bash\n(.*?)```", doc.read_text(), re.S)


def test_no_document_checks_the_whole_upstream_manifest_from_one_directory():
    """
    The defect this replaced: `cd "$FOA_NIST28_DIR" && sha256sum -c …/UPSTREAM.sha256`
    exits 1 on a correct fetch and a correct cache.

    `UPSTREAM.sha256` lists all seven entries as bare basenames; three of them
    are only ever in `$FOA_DATA_DIR/nist` and four only ever in
    `$FOA_NIST28_DIR`, so from either directory the other four - or three - come
    back FAILED open or read. A reader who is doing everything right reads three
    failures and goes looking for a corruption that is not there, which is worse
    than no check at all. The correct form selects the entries first and feeds
    them in - `grep … | sha256sum -c -` - so what is forbidden is naming the
    manifest as `sha256sum -c`'s own argument, in prose as much as in a fence.
    """
    whole_file = re.compile(r"sha256sum -c\s+[^\s|`]*UPSTREAM\.sha256")
    for doc in sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]:
        found = whole_file.findall(doc.read_text())
        assert not found, (
            f"{doc.name} tells a reader to run `{found[0]}`, which exits 1 on a "
            f"correct fetch: no directory holds all seven basenames"
        )


def test_the_documented_verify_command_passes_on_a_correct_fetch(tmp_path):
    """
    The replacement, executed rather than read.

    Run against a tree laid out the way sections 1 and 2 prescribe - sources in
    `$FOA_DATA_DIR/nist`, cache in `$FOA_NIST28_DIR` - the block must print one
    OK per entry in the manifest and exit 0, and must still exit 1 when a byte
    moves. A verify command is worth exactly what it costs to run it, so this
    runs it; the hashes are of stand-in payloads, because what is under test is
    the command's shape and coverage, not SD19.
    """
    import hashlib
    import subprocess

    names = _upstream_names()
    assert len(names) == 7, f"UPSTREAM.sha256 has {len(names)} entries, expected 7"

    sources = tmp_path / "data" / "nist"
    cache = tmp_path / "nist28"
    sources.mkdir(parents=True)
    cache.mkdir(parents=True)
    lines = []
    for name in names:
        home = cache if name.startswith("nist28_") else sources
        payload = name.encode()
        (home / name).write_bytes(payload)
        lines.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
    manifest = tmp_path / "UPSTREAM.sha256"
    manifest.write_text("\n".join(lines) + "\n")

    blocks = [b for b in _bash_blocks(REPO / "docs" / "DATA.md")
              if "sha256sum -c -" in b and "FOA_NIST28_DIR" in b]
    assert len(blocks) == 1, (
        f"expected exactly one whole-manifest verify block in DATA.md, found {len(blocks)}"
    )
    script = blocks[0].replace("/path/to/repo/study/UPSTREAM.sha256", str(manifest))
    env = {"PATH": os.environ["PATH"],
           "FOA_DATA_DIR": str(tmp_path / "data"),
           "FOA_NIST28_DIR": str(cache)}

    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    verified = re.findall(r"^(\S+): OK$", done.stdout, re.M)
    assert sorted(verified) == sorted(names), (
        f"the documented command verifies {sorted(verified)}, not all seven entries"
    )

    (cache / "nist28_images.npy").write_bytes(b"a different release of SD19")
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    assert done.returncode == 1, "a changed cache file has to fail the check"
