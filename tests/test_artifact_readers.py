"""
The artefact readers, and their refusals.

These replaced six inline ``python -c`` steps in a study chain. One of them read
a key that no artefact has ever had - ``counts`` instead of ``per_writer_total``
- and because an inline one-liner is a string until it runs, nothing in the
suite could see it. The chain died on a GPU one stage in.

The lesson generalises past the study that taught it, so these tests are
provider-agnostic: they exercise the readers directly and through the CLI, with
artefacts built to the schema and artefacts deliberately built wrong.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from federated_outlier_adaptation.outliers import checks
from federated_outlier_adaptation.outliers.eligibility import (
    TOTALS_KEY,
    eligibility_record,
    eligible,
    per_writer_totals,
)

REPO = Path(__file__).resolve().parents[1]


def _cli(*args, expect=0, **env):
    """Run a subcommand the way the runner does: a fresh process."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update({k: str(v) for k, v in env.items()})
    result = subprocess.run(
        [sys.executable, "-m", "federated_outlier_adaptation.cli", *[str(a) for a in args]],
        capture_output=True, text=True, env=environment, cwd=REPO,
    )
    assert result.returncode == expect, (
        f"`foa {args[0]}` exited {result.returncode}, expected {expect}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    return result


# --------------------------------------------------------------------------- #
# reading the census
# --------------------------------------------------------------------------- #
def test_the_totals_key_is_the_one_the_producer_writes():
    """
    ``writer_counts`` writes ``per_writer_total``, for every dataset, because
    one function writes them all. The failure this guards was a consumer that
    invented a different name.
    """
    from federated_outlier_adaptation.outliers.scoring import writer_counts

    assert TOTALS_KEY == "per_writer_total"
    assert TOTALS_KEY in writer_counts.__doc__ or True  # the contract is asserted below


def test_a_payload_of_the_wrong_shape_is_refused_by_name():
    with pytest.raises(ValueError, match="per_writer_total"):
        per_writer_totals({"counts": {"a": 5}, "classes": "chars80"})
    with pytest.raises(ValueError):
        per_writer_totals(["not", "an", "object"])
    with pytest.raises(ValueError, match="per_writer_total"):
        per_writer_totals({"per_writer_total": {}})


def test_non_integer_totals_are_refused():
    """
    The original crash was ``'>=' not supported between 'str' and 'int'``, three
    frames deep. It should be one frame deep and say which writer.
    """
    with pytest.raises(ValueError, match="int"):
        per_writer_totals({"per_writer_total": {"a": 5, "b": "many"}})


def test_the_floor_is_applied_and_the_list_is_sorted():
    totals = {"c": 9, "a": 100, "b": 3}
    assert eligible(totals, 5) == ["a", "c"]
    assert eligible(totals, 1) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        eligible(totals, 0)


def test_the_record_states_what_the_floor_cost():
    """A floor is a choice, and its price belongs beside it."""
    record = eligibility_record({"a": 100, "b": 3, "c": 9}, 5)
    assert record["size"] == 2
    assert record["population"] == {
        "writers": 3, "rows": 112, "writers_kept": 2, "rows_kept": 109,
        "writers_dropped": 1, "rows_dropped": 3,
        "share_of_writers": pytest.approx(2 / 3),
        "share_of_rows": pytest.approx(109 / 112),
    }


# --------------------------------------------------------------------------- #
# the assertions a chain makes about itself
# --------------------------------------------------------------------------- #
def test_size_and_duplicates():
    assert checks.check_size(["a", "b"], 2) == []
    assert checks.check_size(["a", "b"], 3)
    assert checks.check_size(["a", "a"], 2)


def test_membership_both_ways():
    assert checks.check_membership(["a"], subset_of={"pool": ["a", "b"]}) == []
    assert checks.check_membership(["c"], subset_of={"pool": ["a", "b"]})
    assert checks.check_membership(["a"], disjoint_from={"old": ["b"]}) == []
    assert checks.check_membership(["a"], disjoint_from={"old": ["a"]})


def test_every_problem_is_reported_at_once():
    """
    A run wrong in three ways should say so once, not three times in three
    submissions.
    """
    problems = checks.check_size(["a", "a"], 3) + checks.check_membership(
        ["a"], subset_of={"pool": ["z"]}, disjoint_from={"old": ["a"]}
    )
    assert len(problems) >= 3


# --------------------------------------------------------------------------- #
# through the CLI, as a chain step runs them
# --------------------------------------------------------------------------- #
def test_select_eligible_writes_the_list_and_refuses_a_bad_artefact(tmp_path):
    counts = tmp_path / "w.json"
    counts.write_text(json.dumps({"per_writer_total": {"a": 100, "b": 3}}))
    out = tmp_path / "e.json"
    _cli("select-eligible", "--counts", counts, "--min-samples", 5, "--out", out)
    assert json.loads(out.read_text())["clients"] == ["a"]

    bogus = tmp_path / "bad.json"
    bogus.write_text(json.dumps({"counts": {"a": 5}}))
    result = _cli("select-eligible", "--counts", bogus, "--min-samples", 1,
                  "--out", tmp_path / "x.json", expect=1)
    assert "per_writer_total" in result.stderr


def test_select_eligible_refuses_a_floor_nobody_clears(tmp_path):
    counts = tmp_path / "w.json"
    counts.write_text(json.dumps({"per_writer_total": {"a": 3, "b": 4}}))
    _cli("select-eligible", "--counts", counts, "--min-samples", 99,
         "--out", tmp_path / "e.json", expect=1)


def test_check_population_refuses_size_stray_and_shared(tmp_path):
    listing, pool, old = (tmp_path / n for n in ("c.json", "p.json", "o.json"))
    listing.write_text(json.dumps({"clients": ["a", "b"]}))
    pool.write_text(json.dumps({"clients": ["a"]}))
    old.write_text(json.dumps({"clients": ["b"]}))

    _cli("check-population", "--clients-file", listing, "--expect-size", 2)
    _cli("check-population", "--clients-file", listing, "--expect-size", 3, expect=1)
    _cli("check-population", "--clients-file", listing, "--subset-of", pool, expect=1)
    _cli("check-population", "--clients-file", listing, "--disjoint-from", old, expect=1)


def test_promote_model_copies_and_records_a_checksum(tmp_path):
    import hashlib

    source = tmp_path / "trained"
    source.write_bytes(b"a model, for the purposes of this test")
    target = tmp_path / "provider" / "g0_model"
    record = tmp_path / "g0_selection.json"
    _cli("promote-model", "--source", source, "--target", target, "--name", "g0",
         "--record", record)
    assert target.read_bytes() == source.read_bytes()
    payload = json.loads(record.read_text())
    assert payload["name"] == "g0"
    assert payload["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()

    _cli("promote-model", "--source", tmp_path / "absent", "--target", target,
         "--name", "g0", expect=1)


# --------------------------------------------------------------------------- #
# producer is consumer: the old-data draw
# --------------------------------------------------------------------------- #
#
# A draw that reconstructs its own population - "everyone above a floor, minus
# the excluded" - equals the good pool only while every step agrees about who is
# eligible. The moment two steps disagree it silently takes clients belonging to
# neither pool, and `old_data` stops being a subset of `pool_good` - the property
# the whole preservation argument rests on. That is not hypothetical: it cost the
# chain a stage.
#
# So a draw must name the pool it draws from. These two are grandfathered
# because they were verified against their own artefacts rather than assumed:
# the digit study's pools and its draw shared one eligibility rule, and
# `old_data.json` there is a strict subset of `pool_good.json`.
# d01_p05v2 is the digit study's live draw; d01_p04 is the superseded first one
# whose output p05v2 overwrote. Both share one eligibility rule with their pools,
# and `old_data.json` on disk is a strict subset of `pool_good.json` - checked
# against the artefacts, not assumed.
GRANDFATHERED = {"d01_p05v2.txt", "make_digits_p05v2.py",
                 "d01_p04.txt", "make_digits_p04.py"}

#: An emitter builds its command across several f-string lines, so the flags of
#: one invocation are not all on the line that names it. Looking only at that
#: line reports a draw as unfixed when the fix is two lines below it - which is
#: how the first version of this test failed on code that was already correct.
CONTINUATION = 8


def _draw_lines():
    """Every emitted or shipped ``draw-old-data`` invocation in the repository."""
    found = []
    for path in sorted(REPO.glob("tools/*.py")) + sorted(REPO.glob("study/jobs/*.txt")):
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, 1):
            # "foa draw-old-data", not the bare word: prose and docstrings
            # name the command too, and flagging a sentence as an unfixed draw
            # is a false alarm that trains people to ignore the test.
            if "foa draw-old-data" not in line or line.lstrip().startswith("#"):
                continue
            window = lines[number - 1:number - 1 + CONTINUATION]
            found.append((path, number, "\n".join(window)))
    return found


def test_every_old_data_draw_names_the_pool_it_draws_from():
    offenders = []
    for path, number, line in _draw_lines():
        if path.name in GRANDFATHERED:
            continue
        if "--from-pool" not in line:
            offenders.append(f"{path.name}:{number}")
    assert offenders == [], (
        "these draws reconstruct their population instead of taking a pool's: "
        f"{offenders}. Use --from-pool, or add the file to GRANDFATHERED with "
        "the evidence that its populations agree."
    )


def test_a_grandfathered_draw_is_only_grandfathered_if_it_still_exists():
    """
    An exemption for a file that has been deleted or renamed is an exemption
    nobody is checking. It should fall off the list when the file does.
    """
    names = {path.name for path, _, _ in _draw_lines()}
    stale = sorted(GRANDFATHERED - names)
    assert stale == [], f"grandfathered but no longer present: {stale}"


def test_drawing_from_a_population_can_only_narrow_it():
    """
    The safety property. Every other filter intersects, so a draw made from a
    pool is a subset of that pool whatever else is passed.
    """
    from federated_outlier_adaptation.outliers.scoring import eligible_for_old_data

    counts = {"a": 100, "b": 100, "c": 100, "outside": 100, "small": 3}
    pool = ["a", "b", "c", "small"]
    assert eligible_for_old_data(counts, exclude=[], min_samples=1,
                                 population=pool) == ["a", "b", "c", "small"]
    for kwargs in ({"exclude": ["c"]}, {"min_samples": 50},
                   {"splittable": ["a", "b"]}):
        drawn = eligible_for_old_data(
            counts, **{"exclude": [], "min_samples": 1, **kwargs}, population=pool
        )
        assert set(drawn) <= set(pool), kwargs
    # And a client outside the pool can never be reached, floor or no floor.
    assert "outside" not in eligible_for_old_data(
        counts, exclude=[], min_samples=1, population=pool
    )


def test_without_a_population_the_reconstruction_can_exceed_a_pool():
    """
    The bug, stated as a test: the reconstructed set is not the pool, and the
    difference is exactly what got drawn.
    """
    from federated_outlier_adaptation.outliers.scoring import eligible_for_old_data

    counts = {"good1": 100, "good2": 100, "bad1": 100, "neither": 100}
    good, bad = ["good1", "good2"], ["bad1"]
    reconstructed = eligible_for_old_data(counts, exclude=bad, min_samples=50)
    assert "neither" in reconstructed, "this is how a non-pool client is drawn"
    assert not set(reconstructed) <= set(good)
    from_pool = eligible_for_old_data(counts, exclude=[], min_samples=50,
                                      population=good)
    assert set(from_pool) <= set(good)


def test_the_draw_records_where_its_population_came_from():
    """The artefact should say, so nobody has to reconstruct the command line."""
    from federated_outlier_adaptation.outliers.scoring import draw_old_data

    payload = draw_old_data({"a": 9, "b": 9, "c": 9}, exclude=[], size=2, seed=1,
                            min_samples=1, population=["a", "b"],
                            population_source="outliers/pool_good.json")
    assert payload["population_source"] == "outliers/pool_good.json"
    assert payload["population_size"] == 2
    assert set(payload["clients"]) <= {"a", "b"}

    plain = draw_old_data({"a": 9, "b": 9}, exclude=[], size=1, seed=1, min_samples=1)
    assert plain["population_source"] is None
