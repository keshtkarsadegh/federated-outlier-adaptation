#!/usr/bin/env python3
"""
Every table this study reports, from the stored runs.

The numbers in the manuscript come from here and from nowhere else. That is the
point: a figure quoted from a script someone ran once at a terminal cannot be
checked, and cannot be regenerated when a stage is re-run. Nothing in this file
trains anything - it reads the result folders and the selection records that the
programme has already written.

    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what references
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what agg-winners
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what reg-winners
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what sizes
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what extremes
    python tools/report_tables.py --root "$FOA_STUDY_DIR" --what all --csv out/

THE RULE THE TABLES ARE ORDERED BY. A configuration is worth what it added on
the new clients less the source knowledge it spent to get it:

    score = (adaptation - A0) - (P0 - preservation)

A0 is the shipped model on THAT SETTING'S OWN cohort test rows, P0 the shipped
model on the source population's. P0 is one number because there is one source
population; A0 is not, because five writers, ten, twenty and the extreme pair
are four different cohorts and g-0 is not equally good on them. Reading a
twenty-client run against the ten-client cohort's do-nothing accuracy reports a
gain the run never made. Both are read from the study's own evaluations, never
assumed, and a root missing the book a setting needs is refused rather than
defaulted to another cohort's.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from federated_outlier_adaptation.training.extreme_cells import (  # noqa: E402
    CASES as EXTREME_CASES,
)

SIGNALS = ["dist_l2_to_global", "dist_fisher_to_global", "dist_fisher_norm_to_global",
           "agreement_with_global", "kl_global_to_current", "retention_known",
           "proxy_acc", "proxy_kl"]


# --------------------------------------------------------------- baselines
#: The shipped model's own accuracy on each cohort this study measures against,
#: one evaluation file per cohort. All four were produced the same way - forward
#: passes of g-0 over that cohort's test rows, per fold, no training - and they
#: disagree because the cohorts do. A setting scored against another cohort's
#: number reports a gain that is really the difference between two populations.
COHORT_BOOKS = {
    "c10": "g0_perfold_evaluations.json",
    "c5": "g0_c5_evaluations.json",
    "c20": "g0_c20_evaluations.json",
    "extreme": "g0_extreme_evaluations.json",
}

#: Which cohort each run-folder stem federates. The search stages and the
#: study's own carry setting all ran on the ten-writer cohort, so they share a
#: book; the settings the winners were carried into do not.
SETTING_COHORT = {
    "agg_": "c10", "aggfull_": "c10", "reg_": "c10", "regfull_": "c10",
    "combo_": "c10", "ctune_": "c10", "ctunesel_": "c10", "c10d10_": "c10",
    "five_": "c5",
    "c20d10_": "c20", "c20d20_": "c20",
    "extreme_": "extreme",
}


def cohort_of(stem: str) -> str:
    """
    Which cohort a run-folder stem belongs to.

    Matched on the LONGEST registered stem, so ``c20d20_`` is never read as
    ``c20d10_`` and ``c10d10_`` is never read as a size reference. A stem this
    table does not know is refused rather than defaulted to the ten-writer
    cohort - defaulting is precisely the mistake the table exists to stop.
    """
    bare = stem.split("_", 1)[1] if stem.startswith("d01_") else stem
    bare = bare.rstrip("_") + "_"
    for key in sorted(SETTING_COHORT, key=len, reverse=True):
        if bare.startswith(key):
            return SETTING_COHORT[key]
    raise SystemExit(
        f"{stem!r} names no cohort this study evaluated g-0 on. Add it to "
        "SETTING_COHORT with the book its rows came from: a setting read "
        "against another cohort's shipped-model accuracy has no score."
    )


def baselines(root: Path, cohort: str = "c10") -> tuple:
    """
    The shipped model's own two accuracies: A0 on ``cohort``, P0 on the source.

    ``cohort`` names one of :data:`COHORT_BOOKS`. Ten writers is the default
    because that is where the search ran; every setting the winners were
    carried into must name its own, and :func:`baselines_for` does it from the
    run-folder stem so no caller has to remember.
    """
    def mean_accuracy(name):
        path = root / name
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
        records = payload if isinstance(payload, list) else list(payload.values())
        values = [r["accuracy"] for r in records
                  if isinstance(r, dict) and isinstance(r.get("accuracy"), (int, float))]
        return sum(values) / len(values) if values else None

    book = COHORT_BOOKS.get(cohort)
    if book is None:
        raise SystemExit(f"{cohort!r} is not a cohort this study evaluated g-0 on")
    a0 = mean_accuracy(book)
    p0 = mean_accuracy("g0_evaluations.json")
    if a0 is None or p0 is None:
        raise SystemExit(
            f"{root} has no shipped-model evaluations for cohort {cohort!r} "
            f"({book} and g0_evaluations.json); every table is measured "
            "against them and they are not guessed."
        )
    return a0, p0


def baselines_for(root: Path, stem: str) -> tuple:
    """The two accuracies the runs under ``stem`` must be read against."""
    return baselines(root, cohort_of(stem))


#: The settings the selected arms were carried into, in the order they are
#: read: the study's own federation first, then the two directions it was
#: moved in. Each is a table of its own rather than four rows of one, because
#: the arms mean the same thing in each and the settings do not - putting them
#: in one table would invite a reader to rank across federations, which is the
#: one comparison none of these runs supports.
#: Named by the run-folder stem the stage writes under, so ``--study-tag``
#: reaches these tables the way it reaches every other one.
CARRY_SETTINGS = (
    ("c10d10", "TEN CLIENTS, ONE DROPPED - nine of ten, the study's own setting"),
    ("five", "FIVE CLIENTS, ONE DROPPED - four of five"),
    ("c20d10", "TWENTY CLIENTS, TWO DROPPED - eighteen of twenty"),
    ("c20d20", "TWENTY CLIENTS, FOUR DROPPED - sixteen of twenty"),
)

#: The extreme arrangements. One prefix, two arms; they hold precisely the same
#: rows and differ only in whether the aggregation ever sees them separately.
EXTREME_STEM = "extreme_"


def in_study(folder: str) -> bool:
    """
    Whether a run folder belongs to an arrangement this study still defines.

    A GLOB IS NOT A CELL LIST. The extreme stage once ran a third arrangement -
    the cohort's worst writer alone - and its run folders are still on the disk
    that produced this study, because a run that happened cannot be un-run. A
    reader that globbed the stage prefix would put it back into every table,
    every stage total and every all-arms mean, and would do it silently: the
    folders look exactly like the ones that belong. So the case is checked
    against :data:`~federated_outlier_adaptation.training.extreme_cells.CASES`,
    which is where the stage is defined, and a folder naming a case that is not
    there is not part of this study's record.

    Every other stage's folders pass through: only the extreme prefix carries a
    case name this can be checked against.
    """
    if EXTREME_STEM not in folder:
        return True
    case = folder.split(EXTREME_STEM, 1)[1].split("_fold")[0]
    return case in EXTREME_CASES


# ------------------------------------------------------------------ runs
def _check_preservation_series(body) -> None:
    """
    Refuse a series that behaves like the cohort rather than the source.

    Preservation starts near the shipped model's own accuracy and decays. A
    series that starts far below it and climbs is the adaptation pool under
    another name, and reading it as preservation understates forgetting by
    several points without failing.
    """
    series = [v for v in (body.get("source_val_accuracies") or []) if v is not None]
    if len(series) >= 2 and series[-1] > series[0] + 0.05:
        raise SystemExit(
            "the preservation series rises by more than five points across the "
            "run, which is what the ADAPTATION pool does. Check the field."
        )



def cell_and_family(path: Path, prefix: str) -> tuple:
    """
    Which arm a stored summary belongs to, and under which schedule.

    A run folder holds BOTH schedules - the concurrent (parallel) family and the
    sequential (cyclic) one - so the schedule is part of the key. Reading only
    one of them, or averaging across them, silently answers a different question
    than the table asks.

    A REGULARISATION RUN COMPUTES BOTH SCHEDULES, and the folder is named for
    the one it was selected to serve. Reading the other half is reading a result
    the selection never chose - the same penalty judged under a schedule it was
    not picked for. The folder's own tag decides, and the cell comes back
    ``None`` for the half that was not selected, which the caller drops.
    """
    folder = next(p for p in Path(path).parts if p.startswith(prefix))
    cell = folder[len(prefix):].split("_fold")[0]
    family = "parallel" if Path(path).parent.name == "concurrent_delta" else "cyclic"
    for tag, schedule in (("concurrent_", "parallel"), ("sequential_", "cyclic")):
        if cell.startswith(tag):
            return (cell[len(tag):] if family == schedule else None), family
    return cell, family


def read_runs(root: Path, prefix: str) -> dict:
    """
    Final-round adaptation and preservation of every run under a prefix.

    The key is the arm and its schedule, decided by :func:`cell_and_family`.
    """
    out = defaultdict(lambda: {"a": [], "p": [], "signals": []})
    for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
        cell, family = cell_and_family(Path(path), prefix)
        if cell is None or not in_study(str(Path(path))):
            continue
        body = next(iter(json.loads(Path(path).read_text()).values()))

        # REPORT ON TEST. Selection reads the validation halves - that is what a
        # selection is allowed to see - and reporting reads the test ones. Both
        # were mixed here until it was checked: this file quoted a validation
        # preservation against an adaptation series that was neither, and
        # differed from the selectors by about a third of a point on every row.
        final = body.get("final_evaluation") or {}
        clients = final.get("clients") or {}
        adaptation = ([clients["accuracy"]] if isinstance(clients.get("accuracy"), (int, float))
                      else body.get("heldout_client_accuracies") or [])

        # PRESERVATION IS THE SOURCE POPULATION, not the pool of clients being
        # adapted to. pool_test_accuracies tracks the cohort and RISES through a
        # run; source_val_accuracies tracks the writers the shipped model was
        # trained on and falls. Reading the first as preservation reports a run
        # that forgot seven points when it forgot one, and it does so silently,
        # because both are plausible-looking series of the right length.
        old_eval = final.get("old") or {}
        preservation = ([old_eval["mean"]] if isinstance(old_eval.get("mean"), (int, float))
                        else body.get("source_val_accuracies") or [])
        if adaptation:
            out[(cell, family)]["a"].append(adaptation[-1])
        if preservation:
            out[(cell, family)]["p"].append(preservation[-1])
        _check_preservation_series(body)
        out[(cell, family)]["signals"].append(
            sum(1 for k in SIGNALS if any(v is not None for v in (body.get(k) or [])))
        )
    return out


def summarise(runs: dict, a0: float, p0: float, folds: int = 5) -> list:
    rows = []
    for (cell, family), v in runs.items():
        if len(v["a"]) < folds:
            continue
        a, p = st.mean(v["a"]), st.mean(v["p"])
        rows.append({
            "cell": cell, "family": family,
            "adaptation": a, "adaptation_sd": st.pstdev(v["a"]),
            "preservation": p, "preservation_sd": st.pstdev(v["p"]),
            "gained": a - a0, "spent": p0 - p,
            "score": (a - a0) - (p0 - p),
            "folds": len(v["a"]), "signals": min(v["signals"]),
        })
    return sorted(rows, key=lambda r: -r["score"])


def show(rows: list, title: str, a0: float, p0: float) -> None:
    print(f"\n{title}")
    print(f"reference: the shipped model  adapt {a0:.4f}  preserve {p0:.4f}")
    print("basis: TEST (final_evaluation). Selection ran on validation.")
    # sized to the content: a truncated id names a different arm, and the
    # combination ids are the two cells they were built from joined together
    width = max([len(r["cell"]) for r in rows] + [len("cell")]) + 2
    print(f"{'cell':<{width}}{'sched':<10}{'adapt':>8}{'+-':>7}{'preserve':>10}"
          f"{'gained':>9}{'spent':>8}{'score':>8}{'sig':>5}")
    print("-" * (width + 68))
    for r in rows:
        print(f"{r['cell']:<{width}}{r['family']:<10}{r['adaptation']:>8.4f}"
              f"{r['adaptation_sd']:>7.4f}{r['preservation']:>10.4f}"
              f"{r['gained']*100:>8.2f}p{r['spent']*100:>7.2f}p"
              f"{r['score']*100:>7.2f}p{r['signals']:>4}/8")


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# ------------------------------------------------------------- references
def references(root: Path, a0: float, p0: float, tag: str) -> list:
    """The four rungs every federated number is read against."""
    rows = []

    def pooled(pattern, key):
        found = []
        for path in glob.glob(str(root / pattern)):
            d = json.loads(Path(path).read_text())
            block = d.get("pooled", {})
            value = block.get(key, d.get(key))
            if isinstance(value, dict):
                value = value.get("mean")
            if value is not None:
                found.append(value)
        return st.mean(found) if found else None

    for label, folder, init in (("isolated (from g-0)", f"isolated_{tag}", "global"),
                                ("isolated (scratch)", f"isolated_{tag}", "scratch"),
                                ("centralized (from g-0)", f"centralized_{tag}", "global"),
                                ("centralized (scratch)", f"centralized_{tag}", "scratch")):
        stem = "isolated" if folder.startswith("isolated") else "centralized"
        a = pooled(f"{folder}/{stem}_{init}_*.json", "own")
        p = pooled(f"{folder}/{stem}_{init}_*.json", "old")
        if a is None:
            continue
        rows.append({"cell": label, "family": "-", "adaptation": a, "adaptation_sd": 0.0,
                     "preservation": p if p is not None else float("nan"),
                     "preservation_sd": 0.0, "gained": a - a0,
                     "spent": p0 - p if p is not None else float("nan"),
                     "score": (a - a0) - (p0 - p) if p is not None else float("nan"),
                     "folds": 5, "signals": 0})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--what", default="all",
                    choices=("all", "references", "agg-screen", "agg-winners",
                             "reg-screen", "reg-winners", "combos",
                             "sizes", "extremes"))
    ap.add_argument("--tag", default="c10", help="Reference cohort tag.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--csv", type=Path, default=None, help="Also write CSVs here.")
    args = ap.parse_args()

    t = args.study_tag
    wanted = {"all": ("references", "agg-screen", "agg-winners",
                      "reg-screen", "reg-winners",
                      "combos", "sizes", "extremes")}.get(args.what, (args.what,))

    for what in wanted:
        # ONE TABLE PER SETTING. The arms are the same four everywhere, so a
        # single table would read as a ranking across federations - and no run
        # here supports that comparison: a five-client cohort and a
        # twenty-client one are different populations of writers, not the same
        # measurement at two sizes.
        if what == "sizes":
            for name, title in CARRY_SETTINGS:
                # EACH SETTING AGAINST ITS OWN COHORT. The shipped model is not
                # equally good on five writers, ten and twenty, so one A0 across
                # the four rows would report the difference between cohorts as a
                # gain the arm made.
                a0, p0 = baselines_for(args.root, name)
                rows = summarise(read_runs(args.root, f"{t}_{name}_"), a0, p0)
                if not rows:
                    print(f"\n{title}\n  nothing on disk yet")
                    continue
                show(rows, title, a0, p0)
                if args.csv:
                    write_csv(rows, args.csv / f"sizes_{name}.csv")
            continue
        if what == "references":
            a0, p0 = baselines(args.root, args.tag)
            rows = references(args.root, a0, p0, args.tag)
            title = f"THE REFERENCE RUNGS ({args.tag})"
        else:
            prefix, title = {
                "agg-screen":  (f"{t}_agg_",     "AGGREGATION SCREEN (short horizon: a ranking, not a result)"),
                "agg-winners": (f"{t}_aggfull_", "AGGREGATION WINNERS (full horizon)"),
                "reg-screen":  (f"{t}_reg_",     "REGULARISATION SCREEN (short horizon)"),
                "reg-winners": (f"{t}_regfull_", "REGULARISATION WINNERS (full horizon)"),
                "combos":      (f"{t}_combo_",   "COMBINATIONS: server rule x client penalty (full horizon)"),
                "extremes":    (f"{t}_{EXTREME_STEM}",
                                                 "THE EXTREME CASES (full participation): "
                                                 "two writers, and the same two merged into one client"),
            }[what]
            a0, p0 = baselines_for(args.root, prefix)
            rows = summarise(read_runs(args.root, prefix), a0, p0)
        if not rows:
            print(f"\n{title}\n  nothing on disk yet")
            continue
        show(rows, title, a0, p0)
        if args.csv:
            write_csv(rows, args.csv / f"{what}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
