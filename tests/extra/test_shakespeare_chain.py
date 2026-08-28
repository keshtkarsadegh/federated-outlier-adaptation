"""
The chain's CPU spine, executed end to end against real artefacts.

The gap this closes
-------------------
A stage of the Shakespeare chain died on the cluster reading a key that no
artefact has ever had. Every check that existed passed it: the CLI imported,
the task line parsed, the runner's guard was happy. None of them looked at what
the *previous stage had written*.

That is one level up from syntax, and it needs the real thing. So this builds a
study root the way the chain does - one command at a time, each reading what the
one before it wrote - and any consumer that assumes a shape its producer does
not write fails here, in the suite, in seconds.

Every step runs as a **subprocess**, the same way the runner invokes it. Running
them in-process would share one import of the configuration module, whose paths
are read from the environment once; a chain that only works because the first
command warmed a cache is not the chain.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def chain(shakespeare_dir, tmp_path):
    """A runner for the CLI, pointed at a fresh study root over the toy corpus."""
    study = tmp_path / "study"
    (study / "outliers").mkdir(parents=True)
    (study / "fold_books").mkdir(parents=True)

    def run(*args, expect=0):
        environment = dict(os.environ)
        environment.update({
            "FOA_SHAKESPEARE_DIR": str(shakespeare_dir),
            "FOA_RESULTS_DIR": str(study),
            "PYTHONPATH": str(REPO / "src") + os.pathsep + environment.get("PYTHONPATH", ""),
        })
        environment.pop("FOA_FOLD_BOOK", None)
        environment.pop("FOA_FOLD", None)
        result = subprocess.run(
            [sys.executable, "-m", "federated_outlier_adaptation.cli",
             *[str(a) for a in args]],
            capture_output=True, text=True, env=environment, cwd=REPO,
        )
        assert result.returncode == expect, (
            f"`foa {args[0]}` exited {result.returncode}, expected {expect}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        return result

    run.study = study
    return run


def _clients(path):
    payload = json.loads(Path(path).read_text())
    return payload["clients"] if isinstance(payload, dict) else payload


# --------------------------------------------------------------------------- #
def test_the_population_spine_runs_end_to_end(chain):
    """
    writer-counts -> select-eligible -> fold-book -> check-population, then the
    scoring half: score-writers -> split-pools -> draw-old-data -> the cohort.

    Each step reads the previous step's file off disk. A key that does not exist
    stops the test at the step that assumed it.
    """
    import torch

    study = chain.study
    outliers = study / "outliers"
    books = study / "fold_books"
    common = ["--provider", "shakespeare", "--results-dir", str(study)]

    # 1. the census -------------------------------------------------------
    counts_path = outliers / "writer_counts.json"
    chain("writer-counts", *common, "--out", str(counts_path))
    counts = json.loads(counts_path.read_text())
    # The schema every consumer below depends on, asserted once, here.
    assert "per_writer_total" in counts, sorted(counts)
    assert "counts" not in counts, "the key the broken consumer invented"
    assert all(isinstance(v, int) for v in counts["per_writer_total"].values())

    totals = counts["per_writer_total"]
    # A floor that actually excludes somebody - a floor nobody feels would not
    # exercise the step - but low enough that the toy corpus still supports an
    # old-data draw and a cohort afterwards.
    ordered = sorted(totals.values())
    floor = max(2, ordered[max(0, len(ordered) // 10 - 1)])

    # 2. eligibility ------------------------------------------------------
    eligible_path = outliers / "eligible.json"
    chain("select-eligible", *common, "--counts", str(counts_path),
          "--min-samples", floor, "--out", str(eligible_path))
    eligible = _clients(eligible_path)
    assert eligible == sorted(w for w, n in totals.items() if n >= floor)
    assert eligible, "the floor left nobody"
    record = json.loads(eligible_path.read_text())
    assert record["population"]["writers"] == len(totals)
    assert record["min_samples"] == floor

    # 3. the book every later split descends from -------------------------
    chain("fold-book", *common, "--out", str(books / "all_users"),
          "--clients-file", str(eligible_path), "--folds", 3, "--seed", 42,
          "--train-rate", 0.6, "--eval-rate", 0.2)
    all_book = books / "all_users.foldbook.npz"
    assert all_book.is_file()

    # 4. and the assertion the chain makes about it ------------------------
    chain("check-population", *common, "--clients-file", str(eligible_path),
          "--fold-book", str(all_book), "--folds", 1,
          "--counts", str(counts_path), "--label", "all_users_book")

    # 5. a model to score with (the chain trains one; here one is enough).
    #    Built at the provider's OWN default shape, because that is what the
    #    subprocess will construct before loading these weights - a checkpoint
    #    from a differently-sized model is a different bug than the one this
    #    test is about.
    from federated_outlier_adaptation.models.char_lstm import CharLSTM

    model_path = study / "ginit_model"
    torch.manual_seed(0)
    torch.save(CharLSTM(vocab=counts["num_classes"]).state_dict(), model_path)

    scores_path = outliers / "clients_acc_on_global.json"
    chain("score-writers", *common, "--model-path", str(model_path),
          "--fold-book", str(all_book), "--fold", 1, "--batch-size", 32,
          "--out", str(outliers / "writer_scores.json"),
          "--accuracies-name", "clients_acc_on_global.json")
    assert scores_path.is_file(), "split-pools reads this next"

    # 6. the coarse cut ---------------------------------------------------
    chain("split-pools", *common, "--bad-fraction", 0.5,
          "--scores", str(scores_path), "--out", str(outliers / "pools.json"))
    bad = _clients(outliers / "pool_bad.json")
    good = _clients(outliers / "pool_good.json")
    assert bad and good and not (set(bad) & set(good))

    # 7. the old data, drawn from GOOD only --------------------------------
    # Sized from what is actually drawable: GOOD writers that clear the floor.
    # The real study draws 100 from ~740; the toy corpus has a handful.
    drawable = [w for w in good if totals[w] >= floor]
    assert drawable, "no eligible GOOD writer to draw old data from"
    old_size = max(1, len(drawable) - 1)
    old_path = outliers / "old_data.json"
    chain("draw-old-data", *common, "--size", old_size, "--seed", 7,
          "--min-samples", floor, "--exclude-file", str(outliers / "pool_bad.json"),
          "--out", str(old_path))
    old = _clients(old_path)
    assert set(old) <= set(good), "an old writer came from the BAD pool"
    assert len(old) == old_size

    chain("fold-book", *common, "--out", str(books / "old_data"),
          "--clients-file", str(old_path), "--folds", 3, "--seed", 42,
          "--train-rate", 0.6, "--eval-rate", 0.2)
    chain("check-population", *common, "--clients-file", str(old_path),
          "--expect-size", len(old), "--subset-of", str(outliers / "pool_good.json"),
          "--fold-book", str(books / "old_data.foldbook.npz"), "--folds", 1, 2, 3,
          "--counts", str(counts_path), "--label", "old_data")

    # 8. the shipped model picks the cohort --------------------------------
    chain("score-pool", *common, "--model-path", str(model_path),
          "--clients-file", str(outliers / "pool_bad.json"), "--batch-size", 32,
          "--out", str(outliers / "bad_scores_on_g0.json"),
          "--accuracies-name", "bad_acc_on_g0.json")
    flat = outliers / "bad_acc_on_g0.json"
    assert flat.is_file(), "select-outliers reads this next"

    k = min(2, len(bad))
    # --out, because the provider's own outliers directory is one level deeper
    # than this study root - the mismatch this test exists to catch.
    cohort_path = outliers / f"cohort_worst{k}.json"
    chain("select-outliers", *common, "--mode", "worst", "--k", k,
          "--scores", str(flat), "--out", str(cohort_path), "--require-trainable",
          "--tag", "test_cohort", "--force", "--no-accuracy-table")
    assert cohort_path.is_file(), sorted(p.name for p in outliers.iterdir())
    cohort = _clients(cohort_path)
    assert set(cohort) <= set(bad)

    chain("fold-book", *common, "--out", str(books / f"cohort{k}"),
          "--clients-file", str(cohort_path), "--folds", 3, "--seed", 42,
          "--train-rate", 0.6, "--eval-rate", 0.2)
    chain("check-population", *common, "--clients-file", str(cohort_path),
          "--expect-size", k, "--subset-of", str(outliers / "pool_bad.json"),
          "--disjoint-from", str(old_path),
          "--fold-book", str(books / f"cohort{k}.foldbook.npz"), "--folds", 1,
          "--counts", str(counts_path), "--label", "cohort", "--show")

    # 9. the do-nothing row, and the promotion the runs resolve ------------
    chain("evaluate-book", *common, "--model-path", str(model_path),
          "--fold-book", str(books / f"cohort{k}.foldbook.npz"), "--fold", 1,
          "--part", "test", "--clients-file", str(cohort_path),
          "--batch-size", 32, "--tag", "cohort_fold1",
          "--out", str(study / "g0_perfold_evaluations.json"))
    banked = json.loads((study / "g0_perfold_evaluations.json").read_text())
    assert "cohort_fold1" in banked
    assert banked["cohort_fold1"]["accuracy"] is not None

    chain("promote-model", *common, "--source", str(model_path),
          "--target", str(study / "shakespeare" / "g0_model"), "--name", "g0",
          "--record", str(study / "g0_selection.json"))
    promoted = json.loads((study / "g0_selection.json").read_text())
    assert promoted["name"] == "g0" and len(promoted["sha256"]) == 64
    assert (study / "shakespeare" / "g0_model").is_file()


# --------------------------------------------------------------------------- #
# the checks themselves must fail when they should
# --------------------------------------------------------------------------- #
def test_check_population_refuses_a_wrong_size(chain, tmp_path):
    listing = tmp_path / "c.json"
    listing.write_text(json.dumps({"clients": ["a", "b"]}))
    chain("check-population", "--provider", "shakespeare",
          "--clients-file", str(listing), "--expect-size", 3, expect=1)


def test_check_population_refuses_a_stray_or_shared_client(chain, tmp_path):
    listing = tmp_path / "c.json"
    pool = tmp_path / "p.json"
    old = tmp_path / "o.json"
    listing.write_text(json.dumps({"clients": ["a", "b"]}))
    pool.write_text(json.dumps({"clients": ["a"]}))
    old.write_text(json.dumps({"clients": ["b"]}))
    chain("check-population", "--provider", "shakespeare",
          "--clients-file", str(listing), "--subset-of", str(pool), expect=1)
    chain("check-population", "--provider", "shakespeare",
          "--clients-file", str(listing), "--disjoint-from", str(old), expect=1)


def test_select_eligible_refuses_an_artefact_of_the_wrong_shape(chain, tmp_path):
    """
    The exact failure. `{"counts": ...}` is not what `foa writer-counts` writes,
    and reading it must stop here rather than three steps later.
    """
    bogus = tmp_path / "w.json"
    bogus.write_text(json.dumps({"counts": {"a": 5}, "classes": "chars80"}))
    result = chain("select-eligible", "--provider", "shakespeare",
                   "--counts", str(bogus), "--min-samples", 1,
                   "--out", str(tmp_path / "e.json"), expect=1)
    assert "per_writer_total" in result.stderr


def test_select_eligible_refuses_a_floor_nobody_clears(chain, tmp_path):
    counts = tmp_path / "w.json"
    counts.write_text(json.dumps({"per_writer_total": {"a": 3, "b": 4}}))
    chain("select-eligible", "--provider", "shakespeare", "--counts", str(counts),
          "--min-samples", 99, "--out", str(tmp_path / "e.json"), expect=1)
