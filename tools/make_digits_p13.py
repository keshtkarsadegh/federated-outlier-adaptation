"""
Emit P13 of the digits study: the regularisation screen.

    python tools/make_digits_p13.py --jobs-dir $FOA_PROJECT_DIR/jobs/v4

Writes two text files: a login-node job.

P11 asked what the *server* rule is worth. P13 asks the other half - what the
**client-side penalty** is worth - under the same protocol with plain FedAvg on
the server, changing only the term added to the client's loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from federated_outlier_adaptation.training import reg_cells
from federated_outlier_adaptation.training import study_lines as SL
from federated_outlier_adaptation.training.study_config import DIGITS_STUDY01 as CFG

#: Sampler seeds of the screen. Disjoint from every earlier range in this study:
#: P09 30001-30505, P11 702001-703685, P11-ext 703691-703705, P12 704001-704175.
SEED_OFFSET = 6000

#: The winning g-0 fold, whose Fisher the fisher-weighted cells read. The runner
#: derives it from the study's own g0_selection.json; recorded here for the
#: header and the README.
G0_FOLD = 2


def lines() -> list[str]:
    """One line per (cell, fold), at the screening horizon."""
    return [
        SL.reg_line(
            CFG, cell, fold, SL.SCREEN_ROUNDS,
            CFG.seed_base + SEED_OFFSET + index * 10 + fold,
        )
        for index, cell in enumerate(reg_cells.screen_cells())
        for fold in SL.FOLDS
    ]


#: How many cells the screen ran before the boundary extensions were added.
SCREENED_CELLS = 117

#: How large the table was when THIS study's extension file was emitted.
#:
#: A second extension was appended later - the strength floors found on Speech
#: Commands - and without this bound ``ext_lines`` would silently grow by that
#: study's cells, which were never part of the digit extension and whose forty
#: lines have already been submitted and reported. Both ends of the window are
#: pinned, so this file regenerates byte-identical however far the table grows.
EXTENDED_CELLS = 125


def screened_lines() -> list[str]:
    """The lines the original screen ran - unchanged, and asserted so."""
    return lines()[: SCREENED_CELLS * len(SL.FOLDS)]


def ext_lines() -> list[str]:
    """
    The boundary extensions: the cells appended to the table after the screen.

    ``lines()`` enumerates the whole table and the extension cells were
    *appended*, so the first ``SCREENED_CELLS * 5`` lines are byte for byte the
    ones already submitted and the tail is exactly the new work. Slicing is what
    keeps the two files consistent - regenerating an extension can never
    disagree with the screen it extends.
    """
    return lines()[SCREENED_CELLS * len(SL.FOLDS): EXTENDED_CELLS * len(SL.FOLDS)]


def header(tasks: list[str]) -> list[str]:
    screened = reg_cells.screen_cells()[:SCREENED_CELLS]
    grouped: dict = {}
    for cell in screened:
        grouped.setdefault(cell["method"], []).append(cell)
    fisher_cells = sum(1 for c in screened if c["needs_fisher"])
    return [
        f"# d01_p13.txt - {CFG.name}, P13: WHAT THE REGULARISER IS WORTH.",
        "#",
        "# P11 asked what the SERVER rule is worth. This asks the other half:",
        "# what the CLIENT-SIDE PENALTY is worth. Every line runs P11's protocol -",
        f"# ten cohort writers, {CFG.clients_per_round} of {CFG.cohort_size} per round, E=5, batch 64, the",
        "# cohort fold book, g-0 as the only initialisation, plain FedAvg on the",
        "# server - and changes only the term added to the client's loss.",
        "#",
        "# THE ANCHOR IS THE FROZEN g-0 IN EVERY CELL. That is the premise: the",
        "# penalty measures distance from the model the study is trying not to",
        "# forget.",
        "#",
        "# BOTH SCHEDULES PER TASK. --aggregation fedavg runs the parallel family",
        "# and the cyclic one in one job, as this stage has always run. A penalty",
        "# that helps a server average and one that helps a sequential walk are",
        "# different findings, so the selector ranks them separately.",
        "#",
        f"# {len(screened)} cells x {len(SL.FOLDS)} folds = {len(tasks)} tasks:",
    ] + [
        f"#   {method:<14} {len(group):>3} cells   {len(group) * len(SL.FOLDS):>4} tasks"
        for method, group in grouped.items()
    ] + [
        "#",
        "# THE HYBRID IS NOT IN THE SCREEN. A blend of two penalties is only worth",
        "# pricing once each one's own strength is known; screening it here would",
        "# mean guessing three coefficients before any of them was measured. Its",
        "# three cells are emitted afterwards from the kd and fisher winners.",
        "#",
        "# SCREENING HORIZON: 25 ROUNDS, NOT 100. Nothing here is a reported",
        "# number - a 25-round result is a ranking signal. The winners are re-run",
        "# at 100 rounds by P14, which is gated separately and NOT authorised.",
        "#",
        f"# THE FISHER. {fisher_cells} cells ({fisher_cells * len(SL.FOLDS)} tasks) need the Fisher diagonal of the",
        f"# shipped model. g-0 v2 won on FOLD {G0_FOLD}, and its Fisher is on disk at",
        f"#   $FOA_STUDY_DIR/g0_fold{G0_FOLD}/global_results/fisher",
        "# (verified: fisher.pt and global_params.pt, 6.7 MB each, for all five",
        "# folds). The lines name it as g0_fold$G0_FOLD/... and study_phase.sbatch",
        "# DERIVES G0_FOLD from the study's own g0_selection.json - so NO EXPORT",
        "# IS NEEDED. A number already written down should not also have to be",
        "# remembered. If the record is ever missing, a task that needs the fold",
        "# is REFUSED (exit 78) rather than run with an empty path component.",
        "# Export G0_FOLD only to override the record deliberately.",
        "#",
        "# EVERY LINE reports the standard three categories: the pooled clients",
        "# test on the cohort's fold-k rows, the per-client column, and the old",
        "# data's FIVE fold test partitions scored separately with their mean and",
        "# spread.",
        "#",
        "# SEEDS.",
        f"#   {CFG.seed_base} + {SEED_OFFSET} + cell_index * 10 + fold",
        "# disjoint from P09, P11, the P11 extension and P12.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\",
        "#          $J/study_phase.sbatch $J/d01_p13.txt",
        "#",
    ]


def readme(tasks: list[str]) -> str:
    screened = reg_cells.screen_cells()[:SCREENED_CELLS]
    grouped: dict = {}
    for cell in screened:
        grouped.setdefault(cell["method"], []).append(cell)
    fisher_cells = sum(1 for c in screened if c["needs_fisher"])
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    # both families, and the teacher forward pass the output-space penalties add
    train_s = 2 * per_round * 1.4 * SL.SCREEN_ROUNDS
    task_s = train_s + 6 + 2 + 40
    rows = "\n".join(
        f"| {method} | {len(group)} | {len(group) * len(SL.FOLDS)} |"
        for method, group in grouped.items()
    )
    return f"""# {CFG.name} - P13: the regularisation screen

P11 asked what the **server** rule is worth. P13 asks the other half - what the
**client-side penalty** is worth - under the same protocol with plain FedAvg on
the server, changing only the term added to the client's loss.

The anchor is the **frozen g-0** in every cell. That is the premise: the penalty
measures distance from the model the study is trying not to forget.

| method | cells | tasks |
|---|---|---|
{rows}
| **total** | **{len(screened)}** | **{len(tasks)}** |

Both schedules run per task (`--aggregation fedavg`), as this stage always has.
A penalty that helps a server average and one that helps a sequential walk are
different findings, so the selector ranks them separately.

## The Fisher - no export needed

{fisher_cells} cells ({fisher_cells * len(SL.FOLDS)} tasks) need the Fisher
diagonal of the shipped model. **g-0 v2 won on fold {G0_FOLD}**, and I checked
the artefacts are on disk before trusting the path: `fisher.pt` and
`global_params.pt`, 6.7 MB each, present for **all five** folds at
`g0_fold<k>/global_results/fisher`.

The lines name it as `$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, and
**`study_phase.sbatch` now derives `G0_FOLD` from the study's own
`g0_selection.json`**. So no export is required.

That is the safer of the two options you offered: a number that is already
written down should not also have to be remembered, and a forgotten export would
expand to an empty path component and fail {fisher_cells * len(SL.FOLDS)} array
elements at startup. If the selection record is ever missing, a task that needs
the fold is **refused with exit 78** rather than run with a broken path. Export
`G0_FOLD` only to override the record deliberately.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. Winners are re-run at 100 rounds by **P14, which is gated
separately and is not authorised**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)}%40 \\
       $J/study_phase.sbatch $J/d01_p13.txt
```

## After the screen

Two CPU commands, minutes each, both stamped so the files they write cannot be
mistaken for authorised work.

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/{CFG.name}
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-full --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p14.txt --expect {SL.counts(CFG)["reg_full"]}
```

**Per family, not per method.** Each penalty's best cell is chosen within each
schedule, giving {len(SL.REG_METHODS)} methods x {len(SL.FAMILIES)} families x
{len(SL.FOLDS)} folds = **{SL.counts(CFG)["reg_full"]} lines**. The full-horizon
parents carry the family they were selected for - the same cell often wins in
both, and without the tag two tasks would name one folder and race into it.

Then the blend, once both halves are known:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-hybrid --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p14_hybrid.txt --expect {len(reg_cells.HYBRID_MIXES) * len(SL.FOLDS)}
```

The hybrid is deliberately **not** in the screen: a blend is only worth pricing
once each penalty's own strength is known. Screening it would have meant
guessing the KD half's `lam` and `T` and the Fisher half's `lam` before any was
measured - a three-dimensional grid for a question that only becomes meaningful
after the two one-dimensional ones are settled. It costs
{len(reg_cells.HYBRID_MIXES)} cells x {len(SL.FOLDS)} folds =
**{len(reg_cells.HYBRID_MIXES) * len(SL.FOLDS)} tasks** instead of hundreds.
`mix = 1` would reproduce the KD winner exactly, which is what makes three blend
points readable as a line between the two methods.

Every emitter takes `--expect` and exits non-zero on any other count, and every
file it writes is stamped `NOT YET AUTHORISED`.

Grid-edge winners go to `tables/BOUNDARY_HITS.txt` and to stdout and never halt
anything.

## Cost

At `0.31 + 0.0345 x {CFG.clients_per_round} x 5 = {per_round:.2f}` s per round per family, with
both families per task and roughly 40% added for the teacher forward pass the
output-space penalties need on every batch:

| | seconds |
|---|---|
| training, {SL.SCREEN_ROUNDS} rounds, two families, +40% | {train_s:.0f} |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~{task_s:.0f} (~{task_s / 60:.1f} min)** |

**{len(tasks)} tasks is roughly {len(tasks) * task_s / 3600:.0f} GPU-h**, about
{len(tasks) * task_s / 3600 / 40 * 60:.0f} minutes of wall clock at `%40`.

The Fisher-weighted cells are cheaper than that estimate - a Fisher penalty is a
parameter-space term with no teacher pass - and the kd/ntd/logit/feature cells
are the ones carrying the 40%. Taking the whole file at the higher rate is the
honest way round.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-dir", required=True)
    args = parser.parse_args()
    jobs = Path(args.jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    screened, ext = screened_lines(), ext_lines()
    (jobs / "d01_p13.txt").write_text(
        "\n".join(header(screened) + screened) + "\n"
    )
    (jobs / "d01_p13_ext.txt").write_text(
        "\n".join(ext_header(ext) + ext) + "\n"
    )
    (jobs / "d01_p13_README.md").write_text(readme(screened))
    (jobs / "d01_p13_ext_README.md").write_text(ext_readme(ext))
    print(f"wrote {jobs}/d01_p13.txt: {len(screened)} tasks")
    print(f"wrote {jobs}/d01_p13_ext.txt: {len(ext)} tasks")
    print(f"wrote {jobs}/d01_p13_README.md")
    print(f"wrote {jobs}/d01_p13_ext_README.md")
    return 0




# --------------------------------------------------------------------------- #
# the boundary extensions
# --------------------------------------------------------------------------- #
def _ext_groups() -> dict:
    """The extension cells, grouped by the row each one probes."""
    ext = reg_cells.boundary_extension_cells()
    return {
        "kd": [c for c in ext if c["method"] == "kd"],
        "ntd": [c for c in ext if c["method"] == "ntd"],
    }


def ext_header(tasks: list[str]) -> list[str]:
    groups = _ext_groups()
    return [
        f"# d01_p13_ext.txt - {CFG.name}, P13 BOUNDARY EXTENSIONS.",
        "#",
        "# Three of the screen's winners sat on the edge of the row that was",
        "# searched. A winner at the end of its row means the optimum may lie",
        "# outside it, so the rows are widened rather than the results being",
        "# reported as though the ranges had been choices.",
        "#",
        "# 1. KD TEMPERATURE, extended DOWN.",
        "#    kd_T1_a0p99 won in BOTH families, and T=1 is the low end of",
        f"#    {list(reg_cells.KD_TEMPERATURES)}.",
        "#",
        "#    The row is a T x alpha GRID, so extending it properly would be two",
        "#    new temperatures x seven alphas = 14 cells, 70 tasks - a full sweep",
        "#    to probe one edge. Instead the extension is narrowed to the top of",
        f"#    the alpha row: {list(reg_cells.KD_EXT_ALPHAS)}. The justification is in the screen's",
        "#    own result - alpha's optimum was INTERIOR to its row and the two",
        "#    families agreed on 0.99, so alpha is not the axis in question. 0.95",
        "#    comes along only to catch that optimum drifting as T falls; if it",
        "#    does, the alpha row needs its own extension and this file will have",
        "#    said so.",
        f"#    {len(reg_cells.KD_EXT_TEMPERATURES)} temperatures x {len(reg_cells.KD_EXT_ALPHAS)} alphas ="
        f" {len(groups['kd'])} cells, {len(groups['kd']) * len(SL.FOLDS)} tasks.",
        "#",
        "# 2. NTD TAU, extended DOWN (concurrent family).",
        f"#    tau=0.5 is the low end of {list(reg_cells.NTD_TAUS)}; extended at the",
        f"#    betas that won or came close there: {list(reg_cells.NTD_EXT_TAU_BETAS)}.",
        "#",
        "# 3. NTD BETA, extended DOWN (sequential family).",
        f"#    beta=0.001 is the low end of {list(reg_cells.NTD_BETAS)};",
        f"#    extended at the two lowest taus, {list(reg_cells.NTD_EXT_BETA_TAUS)}, which is",
        "#    where that winner lives.",
        "#",
        "#    The two NTD extensions share one (beta, tau) grid, so they are",
        "#    deduplicated before emission. No pair repeats:",
    ] + [
        f"#      {c['id']}" for c in groups["ntd"]
    ] + [
        f"#    {len(groups['ntd'])} cells, {len(groups['ntd']) * len(SL.FOLDS)} tasks - not 4 + 4.",
        "#",
        "# APPENDED TO THE TABLE, NOT INSERTED INTO THE ROWS. A screening seed is",
        "# a function of a cell's index, so inserting would have re-seeded every",
        f"# cell after it and made the {SCREENED_CELLS * len(SL.FOLDS)} lines already submitted",
        "# unreproducible from the table describing them. Appended, d01_p13.txt is",
        "# byte for byte what was run and these are the whole of the new work.",
        "# The selector groups by the cell's method, so each extension is a",
        "# sibling of the row it extends and boundary detection now evaluates the",
        "# widened ranges.",
        "#",
        "# The method count is unchanged - a wider row is not a new method - so",
        f"# P14 stays {SL.counts(CFG)['reg_full']} lines.",
        "#",
        f"# {len(tasks)} tasks. Same protocol as the screen: 25 rounds, both families",
        f"# per task, {CFG.clients_per_round} of {CFG.cohort_size} per round, digits, g-0 init, anchor=frozen,",
        "# --old-fold all, fresh sampler seeds continuing the screen's own scheme.",
        "#",
        "#   J=$FOA_PROJECT_DIR/jobs/v4",
        f"#   sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\",
        "#          --output=\"$FOA_STUDY_DIR/logs/%x_%A_%a.log\" \\",
        "#          $J/study_phase.sbatch $J/d01_p13_ext.txt",
        "#",
        "# AFTER THESE RUN, re-run the reg selection - see d01_p13_ext_README.md.",
        "# Any (method, family) whose winner moved gets a 5-line patch file,",
        "# gated exactly as P14 was.",
        "#",
    ]


def ext_readme(tasks: list[str]) -> str:
    groups = _ext_groups()
    per_round = 0.31 + 0.0345 * CFG.clients_per_round * 5
    task_s = 2 * per_round * 1.4 * SL.SCREEN_ROUNDS + 48
    return f"""# {CFG.name} - P13 boundary extensions

Three of the screen's winners sat on the **edge** of the row that was searched.
A winner at the end of its row means the optimum may lie outside it, so the rows
are widened - **{len(tasks)} tasks** in total, same protocol as the screen.

| # | row | edge | extension | cells | tasks |
|---|---|---|---|---|---|
| 1 | KD temperature | `T=1`, low end of `{list(reg_cells.KD_TEMPERATURES)}` | `T in {list(reg_cells.KD_EXT_TEMPERATURES)}` x `alpha in {list(reg_cells.KD_EXT_ALPHAS)}` | {len(groups['kd'])} | {len(groups['kd']) * len(SL.FOLDS)} |
| 2+3 | NTD tau and beta | `tau=0.5` and `beta=0.001`, both low ends | see below | {len(groups['ntd'])} | {len(groups['ntd']) * len(SL.FOLDS)} |

## Why the KD extension is narrowed

`kd_T1_a0p99` won in **both** families, and `T=1` is the low end of its row.

The KD row is a **T x alpha grid**, so extending it properly would be two new
temperatures x seven alphas = **14 cells, 70 tasks** - a full sweep to probe one
edge. That is out of proportion to the question.

So the extension is narrowed to the top of the alpha row,
`{list(reg_cells.KD_EXT_ALPHAS)}`, and the justification is in the screen's own
result rather than in convenience: **alpha's optimum was interior to its row,
and the two families agreed on 0.99.** Alpha is not the axis in question. `0.95`
comes along only to catch that optimum drifting as `T` falls - and if it does,
the alpha row needs its own extension and these four cells will have said so.

That is {len(groups['kd'])} cells, {len(groups['kd']) * len(SL.FOLDS)} tasks
instead of 70.

## The two NTD extensions share a grid

- **concurrent**: `tau=0.5` is the low end of `{list(reg_cells.NTD_TAUS)}`,
  extended to `tau=0.25` at the betas that won or came close there,
  `{list(reg_cells.NTD_EXT_TAU_BETAS)}`.
- **sequential**: `beta=0.001` is the low end of `{list(reg_cells.NTD_BETAS)}`,
  extended to `beta=0.0001` at the two lowest taus,
  `{list(reg_cells.NTD_EXT_BETA_TAUS)}`, which is where that winner lives.

Both extend the same `(beta, tau)` grid, so they are **deduplicated** before
emission. No pair repeats, and the total is
**{len(groups['ntd'])} cells, not 4 + 4**:

{chr(10).join(f"- `{c['id']}`" for c in groups['ntd'])}

## Appended, not inserted

A screening seed is a function of the cell's index in the table, so inserting
these into their rows would have re-seeded every cell after them - and the
{SCREENED_CELLS * len(SL.FOLDS)} lines already submitted would no longer be
reproducible from the table that describes them.

They are **appended**. `d01_p13.txt` regenerates byte for byte identical to what
was run, and these {len(tasks)} lines are the whole of the new work. The
selector groups by the cell's `method`, so each extension is a sibling of the
row it extends and boundary detection now evaluates the widened ranges.

The method count is unchanged - a wider row is not a new method - so **P14 stays
{SL.counts(CFG)['reg_full']} lines**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-{len(tasks)} \\
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \\
       $J/study_phase.sbatch $J/d01_p13_ext.txt
```

## After they run: re-select, and patch only what moved

The re-selection is the ordinary `reg-full` command over the widened table:

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/{CFG.name}
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-full --study {CFG.name} --root $S \\
    --out $P/jobs/v4/d01_p14.txt --expect {SL.counts(CFG)["reg_full"]}
```

It prints each `(method, family)` winner. For any that **moved**, emit that
pair's patch - five lines, not seventy:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \\
    reg-patch --study {CFG.name} --root $S \\
    --method kd --family concurrent \\
    --out $P/jobs/v4/d01_p14_kd_concurrent_patch.txt --expect {len(SL.FOLDS)}
```

Only three pairs can move - `kd` in either family, `ntd` in either - because
only those two rows were widened. The patch carries **the same seeds its P14
sibling used**, so a patched line and the line it replaces are the same run with
a different coefficient. It states `CHANGED` or `UNCHANGED` into
`tables/p13ext_<method>_<family>_reselection.json` and to stdout, refuses if
that pair was never measured, and is stamped `NOT AUTHORISED` exactly as P14
was.

`reg-top3` needs no special handling: it ranks whatever full-horizon folders
exist, so a patched result simply competes with the one it supersedes and the
better one wins.

## Cost

At `{per_round:.2f}` s per round per family, both families, plus the teacher
forward pass the KD and NTD penalties need: about **{task_s:.0f} s per task**.
{len(tasks)} tasks is roughly **{len(tasks) * task_s / 3600:.2f} GPU-h** -
minutes of wall clock. Each patch, if needed, is 5 tasks at the full horizon:
about {5 * (2 * per_round * 1.4 * SL.FULL_ROUNDS + 48) / 3600:.2f} GPU-h.
"""


if __name__ == "__main__":
    raise SystemExit(main())
