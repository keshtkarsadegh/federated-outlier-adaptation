# The manuscript's figures, from the published records

Seven scripts, one style module, and the name table they all print through.
Each script reads CSV *views* and renders; not one of them computes an accuracy,
opens a run folder or knows where the study lives. That separation is the point:
the views are built by a tool that reads the records, so a figure in the paper is
reachable from the published data by two commands rather than by a script
somebody ran once at a terminal.

    records  --(tools/export_*.py)-->  CSV views  --(fig_*.py)-->  PDF

## Running them

The ten views these scripts need already travel in this repository, under
`study/artifacts/Digits_study01/tables/` — `paper_figures/` for the per-round
views and `paper/references.csv` for the four reference rungs. So with nothing
downloaded and no study root assembled:

```bash
cd tools/paper_figures
FOA_PAPER_OUT=/tmp/figures python fig_problem.py
FOA_PAPER_OUT=/tmp/figures python fig_baselines.py      # and the other five
```

`FOA_PAPER_OUT` is where the PDF and its draft caption are written; without it
they land beside the script, which is what the manuscript build wants and not
what a source checkout does. `FOA_PAPER_DATA` overrides where the views are read
from — one directory, or several separated the way `PATH` is.

## Rebuilding the views from the records

Assemble a study root as `docs/REPRODUCE.md` §10 describes — the released
`Digits_study01_records.tar.gz` unpacked over a copy of
`study/artifacts/Digits_study01/` — and then:

```bash
export FOA_STUDY_DIR=/path/to/Digits_study01
python tools/export_traces.py         --root "$FOA_STUDY_DIR" --out views/
python tools/export_baseline_views.py --root "$FOA_STUDY_DIR" --out views/
python tools/export_combo_folds.py    --root "$FOA_STUDY_DIR" --out views/
python tools/report_tables.py         --root "$FOA_STUDY_DIR" --what all --csv views/
FOA_PAPER_DATA=$PWD/views python tools/paper_figures/fig_problem.py
```

The first three reproduce the ten views in `tables/paper_figures/` byte for byte;
the fourth supplies `references.csv`, which `fig_problem` and `fig_baselines`
read for the centralized ceiling and for `g-0`'s own source accuracy.

## Which figure reads what

| figure | views | the tool that writes them |
|---|---|---|
| `fig_problem` | `traces_control.csv`, `references.csv` | `export_traces.py`, `report_tables.py --what references` |
| `fig_aggs` | `traces_aggfull.csv` | `export_traces.py` |
| `fig_regs` | `traces_regfull.csv` | `export_traces.py` |
| `fig_blends` | `traces_blends.csv` | `export_traces.py` |
| `fig_combo` | `traces_combo.csv` | `export_traces.py` |
| `fig_extremes` | `traces_extreme.csv`, `extreme_stop_rounds.csv` | `export_traces.py` |
| `fig_baselines` | `g0_training.csv`, `isolated_clients.csv`, `references.csv` | `export_baseline_views.py`, `report_tables.py --what references` |

`combos_folds.csv` is exported beside them by `export_combo_folds.py` and is read
by the manuscript's combination table rather than by a figure; it travels with
the other nine because it is the same kind of thing and is built the same way.

## What a reader should know before quoting one

**The six per-round figures are on the VALIDATION halves** — `pool_val_accuracies`
against `source_val_accuracies`, the halves a selection is allowed to see. The
tables report the test halves. The two must not be quoted against each other, and
every caption says so.

**Round 0 is the shipped model.** Every trace view carries `g-0`'s own two
accuracies at round 0, so all the arms of a figure start from one shared point
and the curves show only what adaptation then spends and buys. `figstyle` refuses
a view whose arms do not agree at round 0.

**`fig_baselines` is on the test basis**, because the isolation records store
test evaluations only. It is the one figure here that is, and it is drawn on a
different canvas for that reason.

**Nothing is rasterised and nothing is dated.** The PDFs are vector throughout
and are written with no creation timestamp, so two runs over the same views
produce the same bytes — a regenerated figure that differs means the data
differed.

## The files

| file | what it is |
|---|---|
| `figstyle.py` | the shared canvas, palette, limits and CSV readers |
| `paper_names.py` | run-record identifier to published method name, the one translation |
| `fig_*.py` | one figure each: render only, no arithmetic |
