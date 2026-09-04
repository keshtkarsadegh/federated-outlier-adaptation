#!/usr/bin/env python3
"""
Who the gain reaches, and what it cost.

The score tables report a cohort mean, and a mean can rise while the writer it
was supposed to help does not move. This file asks the two questions the mean
hides:

    FAIRNESS  the distribution of final accuracy ACROSS the cohort's clients,
              not pooled over their rows - and whether the worst-served client
              is better off than it was under the shipped model.
    COST      what the programme actually spent: seconds in the round loop and
              bytes on the wire, read from the runner's own instrumentation.

    python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what fairness
    python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what cost
    python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what all --csv out/

NOTHING HERE RETRAINS ANYTHING, and nothing is recomputed from weights. Every
quantity is already in the stored payloads:

    final_evaluation.clients.per_client   one accuracy per client, TEST rows,
                                          written at the end of the run
    round_seconds                         wall seconds of each round
    client_seconds                        wall seconds of each participant
    comm_bytes_per_round                  payload x participants x 2 directions
    param_count                           scalars in the exchanged state dict

THE BASIS IS TEST, the same basis ``report_tables.py`` reports on, and it comes
from the same structure: ``final_evaluation.clients.accuracy`` is the pooled
number that file prints and ``final_evaluation.clients.per_client`` is that
same evaluation broken out by writer. The per-round ``client_test_accuracies``
series is NOT used - it is measured inside the loop against a different
denominator, and mixing the two would report a fairness table that does not add
up to the score table beside it.

THE REFERENCE IS THE SHIPPED MODEL, PER CLIENT. ``report_tables.py`` measures
against one pooled ``A0``; a fairness table cannot, because the question is
whether a *particular* writer improved. The per-writer numbers are read from the
study's own g-0 evaluation books, per fold, and the book is chosen by which one
covers the run's clients rather than by the stage's name. A client no book
covers - the merged writer of the ``dual`` extreme is the only one in this
study - reports its accuracy and leaves the reference columns empty rather than
borrowing a number from a writer it is not.

THE SECONDS ARE THE ROUND LOOP, not the job. ``round_seconds`` is measured by
the runner from the top of a round to the end of aggregation, so it covers
local training, the in-loop evaluations and the signals, and it does not cover
process start-up, dataset caching, model loading or the final evaluation. It is
therefore a floor on the allocation a stage needed and not a substitute for it:
the GPU-hours in ``docs/REPRODUCE.md`` are what was booked, these are what the
loop used.

AND THEY ARE WALL CLOCK ON A SHARED PARTITION. Two stages that ran on different
nodes, or on the same node beside different neighbours, are not a controlled
timing comparison, and a seconds-per-round difference of tens of percent between
stages says as much about the queue as about the method. The comparison the
table does support is WITHIN a stage, where the arms ran the same shape of job
under the same allocation - which is the comparison the cost question actually
asks: what did the penalty add to the plain baseline beside it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import report_tables

#: The stages a fairness table is asked about. Screens are excluded on purpose:
#: a twenty-five round ranking is a ranking, and asking which writer it served
#: reads a distribution that the stage never claimed to have settled.
FAIRNESS_VIEWS = ("agg-winners", "reg-winners", "combos", "sizes", "extremes")

#: Every stage that spent GPU time, in the order it ran. Each is one or more
#: run-folder prefixes; the size references are three prefixes because the
#: rungs at five, ten and twenty clients are three separate emissions of one
#: idea. Named to line up with the cost table of ``docs/REPRODUCE.md``.
COST_STAGES = (
    ("aggregation screen", ("{t}_agg_",)),
    ("aggregation finals", ("{t}_aggfull_",)),
    ("regularisation screen", ("{t}_reg_",)),
    ("regularisation finals", ("{t}_regfull_",)),
    ("combinations", ("{t}_combo_",)),
    ("ten clients, one dropped", ("{t}_c10d10_",)),
    ("five clients, one dropped", ("{t}_five_",)),
    ("twenty, two dropped", ("{t}_c20d10_",)),
    ("twenty, four dropped", ("{t}_c20d20_",)),
    ("extremes", ("{t}_extreme_",)),
    ("size references", ("{t}_c5_m", "{t}_c10_m", "{t}_c20_m")),
)

#: The arms a cost-per-arm table is worth printing for: the four configurations
#: that were carried, at the setting they were crowned on, against the two
#: unmodified controls that were run at the same participation.
ARM_COST_PREFIXES = ("{t}_c10d10_", "{t}_aggfull_control_")

#: The shipped model's own per-writer accuracies, per fold. All three books are
#: the same model on the same rows; they differ in which fold book cut the test
#: split, so a run is matched to one by coverage rather than by name.
G0_BOOKS = ("g0_perfold_evaluations.json", "g0_c5_evaluations.json",
            "g0_c20_evaluations.json")


# ------------------------------------------------------------ pure helpers
def percentile(values, q: float) -> float:
    """
    Linear-interpolated percentile of a sample, ``q`` in [0, 1].

    Written out rather than taken from ``statistics.quantiles`` because a
    cohort of ten has no exact quartile and the two libraries disagree about
    which neighbour to lean on. Fixing the convention here means a p25 quoted
    in the manuscript can be checked with a pocket calculator.
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of an empty sample")
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def distribution(accuracies) -> dict:
    """
    The spread of one cohort's client accuracies at one fold.

    ``gap`` is the mean less the minimum: how far the cohort's average sits
    above the client it served worst. It is the whole point of the table - a
    configuration that raises the mean by raising the gap has helped the
    clients that needed it least.
    """
    values = [float(v) for v in accuracies]
    if not values:
        raise ValueError("a fold with no clients")
    mean = st.fmean(values)
    return {
        "clients": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "mean": mean,
        "max": max(values),
        "gap": mean - min(values),
    }


def fold_mean(rows, key: str) -> float:
    """Mean of one statistic across the folds of an arm."""
    return st.fmean([row[key] for row in rows])


def pick_reference(clients, books) -> dict:
    """
    The shipped model's accuracy for each of these clients, from one book.

    ``books`` maps a book's name to its per-writer accuracies at this fold. The
    smallest book that covers every client wins: the ten- and five-client
    cohorts share a fold book and therefore agree writer for writer, while the
    twenty-client book cut a different test split and disagrees, so a run must
    be read against the book its own rows came from. Nothing is returned when
    no book covers the cohort - the caller leaves the column empty.
    """
    wanted = set(clients)
    covering = [(len(writers), name, writers) for name, writers in sorted(books.items())
                if wanted <= set(writers)]
    if not covering:
        return {}
    _, _, writers = min(covering, key=lambda item: (item[0], item[1]))
    return {client: float(writers[client]) for client in wanted}


def share_improved(finals: dict, reference: dict) -> float:
    """Fraction of clients whose own accuracy beats its own shipped-model one."""
    paired = [(finals[c], reference[c]) for c in finals if c in reference]
    if not paired:
        return float("nan")
    return sum(1 for now, before in paired if now > before) / len(paired)


# --------------------------------------------------------------- reading
def g0_books(root: Path) -> dict:
    """
    Per-fold, per-writer accuracies of the shipped model.

    Returns ``{fold: {book_name: {writer: accuracy}}}``. A book that is not on
    disk is skipped; a root with none of them cannot report a reference column
    and says so rather than defaulting one.
    """
    out: dict = defaultdict(dict)
    for name in G0_BOOKS:
        path = root / name
        if not path.is_file():
            continue
        for entry in json.loads(path.read_text()).values():
            writers = entry.get("per_writer")
            fold = entry.get("fold")
            if writers and fold is not None:
                out[int(fold)][name] = writers
    return dict(out)


def payloads(root: Path, prefixes) -> list:
    """
    Every stored run under these prefixes, with its folder and its cell.

    One run FOLDER can hold two payloads - a stage that computes both schedules
    writes ``concurrent_delta`` and ``sequential_weights`` from a single job -
    so the folder is carried alongside the cell: seconds are a property of the
    job and accuracies are a property of the schedule.
    """
    out = []
    for prefix in prefixes:
        for path in sorted(glob.glob(f"{root}/{prefix}*/**/summary_0.json", recursive=True)):
            try:
                document = json.loads(Path(path).read_text())
            except (OSError, ValueError):
                continue
            body = next(iter(document.values()), None)
            if not isinstance(body, dict):
                continue
            cell, family = report_tables.cell_and_family(Path(path), prefix)
            folder = next(p for p in Path(path).parts if p.startswith(prefix))
            # A RUN FOLDER IS NOT A MEMBERSHIP CLAIM. An arrangement the stage
            # no longer defines still has its folders on the disk that ran the
            # study, and counting them would put its seconds into the cost
            # table and its writers into the fairness one.
            if not report_tables.in_study(folder):
                continue
            out.append({"path": Path(path), "folder": folder, "cell": cell,
                        "family": family, "body": body})
    return out


# -------------------------------------------------------------- fairness
def fairness_rows(root: Path, prefix: str, books: dict, folds: int = 5) -> list:
    """One row per arm: the fold-mean of each order statistic of the cohort."""
    per_arm = defaultdict(list)
    for run in payloads(root, (prefix,)):
        if run["cell"] is None:
            continue
        final = (run["body"].get("final_evaluation") or {})
        clients = (final.get("clients") or {}).get("per_client") or {}
        fold = final.get("fold")
        if not clients or fold is None:
            continue
        reference = pick_reference(clients, books.get(int(fold), {}))
        row = distribution(clients.values())
        row["improved"] = share_improved(clients, reference)
        row["reference_mean"] = st.fmean(reference.values()) if reference else None
        row["reference_min"] = min(reference.values()) if reference else None
        row["per_client"] = {c: (a, reference.get(c)) for c, a in clients.items()}
        per_arm[(run["cell"], run["family"])].append(row)

    rows = []
    for (cell, family), measured in per_arm.items():
        if len(measured) < folds:
            continue
        covered = [row for row in measured if row["reference_mean"] is not None]
        rows.append({
            "cell": cell, "family": family, "folds": len(measured),
            "clients": measured[0]["clients"],
            "min": fold_mean(measured, "min"),
            "p25": fold_mean(measured, "p25"),
            "median": fold_mean(measured, "median"),
            "mean": fold_mean(measured, "mean"),
            "max": fold_mean(measured, "max"),
            "gap": fold_mean(measured, "gap"),
            "g0_min": fold_mean(covered, "reference_min") if covered else None,
            "g0_mean": fold_mean(covered, "reference_mean") if covered else None,
            "lift_worst": (fold_mean(measured, "min") - fold_mean(covered, "reference_min")
                           if covered else None),
            "lift_mean": (fold_mean(measured, "mean") - fold_mean(covered, "reference_mean")
                          if covered else None),
            "improved": fold_mean(covered, "improved") if covered else None,
        })
    return sorted(rows, key=lambda r: -r["min"])


def client_rows(root: Path, prefix: str, books: dict, folds: int = 5) -> list:
    """
    The same measurement one row per CLIENT, for the CSV only.

    An arm's row says the worst-served client rose; this says which writer it
    was, which is the only form in which the claim can be argued with.
    """
    per_arm = defaultdict(lambda: defaultdict(list))
    for run in payloads(root, (prefix,)):
        if run["cell"] is None:
            continue
        final = (run["body"].get("final_evaluation") or {})
        clients = (final.get("clients") or {}).get("per_client") or {}
        fold = final.get("fold")
        if not clients or fold is None:
            continue
        reference = pick_reference(clients, books.get(int(fold), {}))
        for client, accuracy in clients.items():
            per_arm[(run["cell"], run["family"])][client].append(
                (float(accuracy), reference.get(client))
            )

    rows = []
    for (cell, family), clients in sorted(per_arm.items()):
        counts = {len(v) for v in clients.values()}
        if max(counts) < folds:
            continue
        for client, pairs in sorted(clients.items(), key=lambda kv: st.fmean(p[0] for p in kv[1])):
            shipped = [p[1] for p in pairs if p[1] is not None]
            accuracy = st.fmean(p[0] for p in pairs)
            rows.append({
                "cell": cell, "family": family, "client": client, "folds": len(pairs),
                "accuracy": accuracy,
                "shipped": st.fmean(shipped) if shipped else None,
                "delta": accuracy - st.fmean(shipped) if shipped else None,
            })
    return rows


def show_fairness(rows: list, title: str, a0: float) -> None:
    print(f"\n{title}")
    print(f"reference: the shipped model, per client  pooled adapt {a0:.4f}")
    print("basis: TEST (final_evaluation.clients.per_client). Selection ran on validation.")
    width = max([len(r["cell"]) for r in rows] + [len("cell")]) + 2
    print(f"{'cell':<{width}}{'sched':<10}{'n':>3}{'min':>8}{'p25':>8}{'med':>8}"
          f"{'mean':>8}{'max':>8}{'gap':>8}{'d-worst':>9}{'d-mean':>8}{'lifted':>8}")
    print("-" * (width + 81))
    for r in rows:
        def points(value):
            return f"{value * 100:+.2f}p" if value is not None else "-"
        lifted = f"{r['improved'] * 100:.0f}%" if r["improved"] is not None else "-"
        print(f"{r['cell']:<{width}}{r['family']:<10}{r['clients']:>3}"
              f"{r['min']:>8.4f}{r['p25']:>8.4f}{r['median']:>8.4f}"
              f"{r['mean']:>8.4f}{r['max']:>8.4f}{r['gap']:>8.4f}"
              f"{points(r['lift_worst']):>9}{points(r['lift_mean']):>8}{lifted:>8}")


# ------------------------------------------------------------------ cost
def cost_row(runs: list) -> dict:
    """
    What one group of runs spent, from the runner's own instrumentation.

    Seconds are summed per FOLDER and averaged over folders, because a folder
    is a job and a job is what an allocation is billed for. Bytes are the
    round's two directions together, which is what ``comm_bytes_per_round``
    already holds.
    """
    by_folder: dict = defaultdict(float)
    by_folder_bytes: dict = defaultdict(int)
    rounds, seconds, per_round_bytes, participants, params = [], [], [], [], set()
    for run in runs:
        body = run["body"]
        durations = [float(v) for v in (body.get("round_seconds") or [])]
        volumes = [int(v) for v in (body.get("comm_bytes_per_round") or [])]
        if not durations:
            continue
        by_folder[run["folder"]] += sum(durations)
        by_folder_bytes[run["folder"]] += sum(volumes)
        rounds.append(len(durations))
        seconds.extend(durations)
        per_round_bytes.extend(volumes)
        participants.extend(len(p) for p in (body.get("participants") or []) if p)
        if body.get("param_count"):
            params.add(int(body["param_count"]))
    if not seconds:
        return {}
    return {
        "tasks": len(by_folder), "payloads": len(rounds),
        "rounds": st.median(rounds),
        "clients_per_round": st.fmean(participants) if participants else None,
        "seconds_per_round": st.fmean(seconds),
        "seconds_per_task": st.fmean(by_folder.values()),
        "hours": sum(by_folder.values()) / 3600.0,
        "mb_per_round": st.fmean(per_round_bytes) / 1e6 if per_round_bytes else None,
        "gb_per_task": st.fmean(by_folder_bytes.values()) / 1e9 if by_folder_bytes else None,
        "gb_total": sum(by_folder_bytes.values()) / 1e9,
        "param_count": min(params) if params else None,
    }


def cost_by_stage(root: Path, tag: str) -> list:
    rows = []
    for name, prefixes in COST_STAGES:
        runs = payloads(root, tuple(p.format(t=tag) for p in prefixes))
        row = cost_row(runs)
        if row:
            rows.append({"stage": name, **row})
    return rows


def cost_by_arm(root: Path, tag: str) -> list:
    """
    What each arm cost, per JOB rather than per schedule.

    An arm that computes both schedules is one job that wrote two payloads, so
    grouping cost by ``(cell, schedule)`` the way the fairness table does would
    charge the same seconds to two rows. The accuracy of a run belongs to its
    schedule; the seconds belong to the job, and only the job was billed.
    """
    rows = []
    for prefix in ARM_COST_PREFIXES:
        prefix = prefix.format(t=tag)
        grouped = defaultdict(list)
        for run in payloads(root, (prefix,)):
            folder = run["folder"]
            grouped[folder[len(prefix):].split("_fold")[0]].append(run)
        for cell, runs in grouped.items():
            row = cost_row(runs)
            if row:
                rows.append({"arm": f"{prefix.rstrip('_').split('_', 1)[1]}/{cell}", **row})
    return sorted(rows, key=lambda r: -r["seconds_per_task"])


def show_cost(rows: list, label: str, title: str) -> None:
    print(f"\n{title}")
    print("basis: the runner's own round timer. Local training, the in-loop")
    print("evaluations and the signals; not job start-up or the final evaluation.")
    print("wall clock on a shared partition: compare arms within a stage, not stages.")
    width = max([len(r[label]) for r in rows] + [len(label)]) + 2
    print(f"{label:<{width}}{'tasks':>6}{'files':>6}{'rounds':>7}{'k/rd':>6}"
          f"{'s/round':>9}{'s/task':>9}{'hours':>8}{'MB/rd':>8}{'GB/task':>9}{'GB':>8}")
    print("-" * (width + 76))
    for r in rows:
        print(f"{r[label]:<{width}}{r['tasks']:>6}{r['payloads']:>6}{r['rounds']:>7.0f}"
              f"{(r['clients_per_round'] or 0):>6.1f}{r['seconds_per_round']:>9.3f}"
              f"{r['seconds_per_task']:>9.1f}{r['hours']:>8.2f}"
              f"{(r['mb_per_round'] or 0):>8.1f}{(r['gb_per_task'] or 0):>9.2f}"
              f"{r['gb_total']:>8.1f}")
    print("-" * (width + 76))
    print(f"{'total':<{width}}{sum(r['tasks'] for r in rows):>6}"
          f"{sum(r['payloads'] for r in rows):>6}{'':>7}{'':>6}{'':>9}{'':>9}"
          f"{sum(r['hours'] for r in rows):>8.2f}{'':>8}{'':>9}"
          f"{sum(r['gb_total'] for r in rows):>8.1f}")


def show_programme(rows: list) -> None:
    """
    The shape of the cost table ``docs/REPRODUCE.md`` quotes, measured.

    The booked GPU-hours in that table are an allocation; these are the hours
    the round loop spent inside it. They are printed in the same stage order so
    the two can be read side by side, and the difference between them is
    everything a job does that is not a round.
    """
    print("\nTHE PROGRAMME'S MEASURED COMPUTE (round loop only)")
    width = max(len(r["stage"]) for r in rows) + 2
    print(f"{'stage':<{width}}{'tasks':>6}{'rounds':>8}{'GPU-hours':>11}")
    print("-" * (width + 25))
    for r in rows:
        print(f"{r['stage']:<{width}}{r['tasks']:>6}{r['rounds']:>8.0f}{r['hours']:>11.2f}")
    print("-" * (width + 25))
    print(f"{'whole programme':<{width}}{sum(r['tasks'] for r in rows):>6}{'':>8}"
          f"{sum(r['hours'] for r in rows):>11.2f}")


# ------------------------------------------------------------------- csv
def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--what", default="all", choices=("all", "fairness", "cost"))
    ap.add_argument("--view", default="all", choices=("all",) + FAIRNESS_VIEWS,
                    help="Restrict the fairness half to one stage.")
    ap.add_argument("--study-tag", default="d01")
    ap.add_argument("--folds", default=5, type=int,
                    help="Arms measured on fewer folds than this are dropped.")
    ap.add_argument("--csv", type=Path, default=None, help="Also write CSVs here.")
    args = ap.parse_args()

    a0, _ = report_tables.baselines(args.root)
    tag = args.study_tag

    if args.what in ("all", "fairness"):
        books = g0_books(args.root)
        if not books:
            raise SystemExit(
                f"{args.root} has no per-writer shipped-model book, so no client's "
                "accuracy can be read against its own reference. Refusing rather "
                "than reporting a distribution with nothing to compare it to."
            )
        views = FAIRNESS_VIEWS if args.view == "all" else (args.view,)
        for view in views:
            if view == "sizes":
                settings = report_tables.CARRY_SETTINGS
            else:
                settings = {
                    "agg-winners": (("aggfull", "AGGREGATION WINNERS (full horizon)"),),
                    "reg-winners": (("regfull", "REGULARISATION WINNERS (full horizon)"),),
                    "combos": (("combo", "COMBINATIONS: server rule x client penalty"),),
                    "extremes": ((report_tables.EXTREME_STEM.rstrip("_"),
                                  "THE EXTREME CASES (full participation)"),),
                }[view]
            for name, title in settings:
                prefix = f"{tag}_{name}_"
                rows = fairness_rows(args.root, prefix, books, args.folds)
                if not rows:
                    print(f"\nPER-CLIENT SPREAD - {title}\n  nothing on disk yet")
                    continue
                show_fairness(rows, f"PER-CLIENT SPREAD - {title}", a0)
                if args.csv:
                    for row in rows:
                        row.pop("per_client", None)
                    write_csv(rows, args.csv / f"fairness_{name}.csv")
                    detail = client_rows(args.root, prefix, books, args.folds)
                    if detail:
                        write_csv(detail, args.csv / f"fairness_{name}_clients.csv")

    if args.what in ("all", "cost"):
        stages = cost_by_stage(args.root, tag)
        if not stages:
            print("\nCOST\n  nothing on disk yet")
            return 0
        params = {r["param_count"] for r in stages if r["param_count"]}
        show_cost(stages, "stage", "WHAT THE PROGRAMME SPENT, BY STAGE")
        print(f"exchanged model: {min(params):,} scalars, "
              f"{min(params) * 4 / 1e6:.2f} MB per direction per client per round")
        show_programme(stages)
        arms = cost_by_arm(args.root, tag)
        if arms:
            show_cost(arms, "arm", "WHAT THE SELECTED ARMS COST, AT THE STUDY'S OWN SETTING")
        if args.csv:
            write_csv(stages, args.csv / "cost_stages.csv")
            if arms:
                write_csv(arms, args.csv / "cost_arms.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
