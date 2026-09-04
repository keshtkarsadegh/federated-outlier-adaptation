# The manuscript's figures, numbers and tables, from the published records

Seven figure scripts, one style module, the name table they all print through,
and the two generators that write every number and every table the manuscript
sets. Each of them reads CSV *views* and renders; not one of them computes an
accuracy, opens a run folder or knows where the study lives. That separation is
the point: the views are built by tools that read the records, so a figure in the
paper — or a number in a sentence — is reachable from the published data by two
commands rather than by a script somebody ran once at a terminal.

    records  --(the export and report tools)-->  CSV views  --+->  fig_*.py             ->  PDF
                                                              +->  make_numbers.py      ->  numbers.tex
                                                              +->  make_paper_tables.py ->  tables/*.tex

## Running them

Every view these scripts need already travels in this repository, under
`study/artifacts/Digits_study01/tables/`: `paper_figures/` for the ten per-round
and per-fold views, `paper/` for the twenty-one views of the paper bundle,
`stopping/` for the stopping tables and `tables/` itself for what the stage tools
left there. So with nothing downloaded and no study root assembled:

```bash
cd tools/paper_figures
FOA_PAPER_OUT=/tmp/figures python fig_problem.py
FOA_PAPER_OUT=/tmp/figures python fig_baselines.py      # and the other five
```

`FOA_PAPER_OUT` is where the PDF and its draft caption are written; without it
they land beside the script, which is what the manuscript build wants and not
what a source checkout does. `FOA_PAPER_DATA` overrides where the views are read
from — one directory, or several separated the way `PATH` is.

## The numbers and the tables

`make_numbers.py` writes `numbers.tex`, the file every quantitative sentence of
the manuscript reads its value from; `make_paper_tables.py` writes the twelve
`tables/*.tex` the body and the appendices set. Both read the same shipped views
and neither derives anything the CSVs do not already say:

```bash
cp /path/to/manuscript/numbers.tex /tmp/paper/          # see the caveat below
FOA_PAPER_OUT=/tmp/paper python tools/paper_figures/make_numbers.py
FOA_PAPER_OUT=/tmp/paper python tools/paper_figures/make_paper_tables.py
```

`numbers.tex` lands in `$FOA_PAPER_OUT` and the tables in `$FOA_PAPER_OUT/tables/`.
`--check` on either writes nothing and exits non-zero if what is on disk is not
what the views say, which is the form the manuscript build uses. Without
`FOA_PAPER_OUT` both refuse to run from a source checkout rather than write LaTeX
into the tracked tree; in the manuscript checkout, where they sit beside `data/`
and `numbers.tex`, beside the script is the right answer and they use it.

**`numbers.tex` IS REWRITTEN IN PLACE, NOT WRITTEN FROM NOTHING.** The file
carries `% BEGIN GENERATED` and `% END GENERATED`, and only the block between
them is rewritten: the macro *names* are read from the block, and each is filled
from the registry the script builds out of the views. So a copy of the
manuscript's own `numbers.tex` has to be in `$FOA_PAPER_OUT` first — there is
nothing to fill in otherwise — and a name with no entry in the registry is
written back as `\TBD{unmapped}` and reported, with a non-zero exit.

**FIFTY-NINE DEFINITIONS SIT ABOVE THE MARKER AND ARE NOT GENERATED.** They are
hand-maintained on purpose and the generator never touches them:

| what | how many | why it is not generated |
|---|---|---|
| `\pub…` | 46 | facts of the earlier published single-seed runs, not of this study |
| `\nBadFraction`, `\nGoodPoolSize`, `\nDigitRows`, `\nDigitWriters`, `\nSamplesPerClassPerWriter`, `\nMacsPerImage`, `\nOldSize`, `\nOldMinSamples`, `\nLocalEpochs`, `\nBatchSize` | 10 | protocol constants — configuration the study was *given*, not a measurement it produced, so no view carries them |
| `\nProxySamples`, `\nProxyAccStart`, `\nProxyAccEnd` | 3 | measurements of the public proxy set whose source has not been extracted into a view yet |

Everything else — 182 macros — is generated, and each one carries a trailing
comment naming the CSV file, the row and the column it was read from, so any
number in the paper is traceable to a file without leaving the manuscript.

Two further table files, `tables/sensitivity_agg.tex` and
`tables/sensitivity_reg.tex`, are written by a third generator that lives in the
manuscript checkout; the views they read, `weight_sensitivity_agg.csv` and
`weight_sensitivity_reg.csv`, ship here like the rest.

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

The numbers and the tables read wider than the figures do — thirty views between
them — so rebuilding *their* inputs takes seven more commands into the same
directory:

```bash
python tools/fairness_cost.py      --root "$FOA_STUDY_DIR" --what all  --csv views/
python tools/stopping_table.py     --root "$FOA_STUDY_DIR"             --csv views/
python tools/compare_arms.py       --root "$FOA_STUDY_DIR" --what all  --csv views/
python tools/weight_sensitivity.py --root "$FOA_STUDY_DIR" --grid both --csv views/
python tools/describe_cohort.py    --root "$FOA_STUDY_DIR" --all       --csv views/
python tools/export_signals_summary.py  --root "$FOA_STUDY_DIR" --out views/
python tools/export_decouple_example.py --root "$FOA_STUDY_DIR" --out views/
FOA_PAPER_DATA=$PWD/views FOA_PAPER_OUT=/tmp/paper \
    python tools/paper_figures/make_numbers.py
```

That is `fairness_*.csv` and `cost_*.csv` from the first, `stopping_all.csv` and
`stopping_extreme.csv` from the second, `blends.csv` and `composition.csv` from
the third, the two `weight_sensitivity_*.csv` from the fourth and
`cohort_composition.csv` from the fifth. `cohort_table.csv` is a stage artefact
rather than a report, written when the cohort was drawn.

The last two commands are the two extracts. `export_signals_summary.py` joins the
correlations `foa signals` wrote into `$FOA_STUDY_DIR/signals/` to the stop
`stopping_table.py` reads off the same arms, which is what the forgetting-signals
section asks of a signal at once; `export_decouple_example.py` rebuilds the
five-writer rank comparison the cohort-selection section quotes, and because it
reads only `outliers/` it also runs against `study/artifacts/Digits_study01/` in
a clone with no records unpacked:

```bash
python tools/export_decouple_example.py \
    --root study/artifacts/Digits_study01 --out views/
```

All three views still ship, so that `make_numbers.py` runs from a bare clone.

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
| `make_numbers.py` | the generated block of `numbers.tex`: 182 macros, each with its cell |
| `make_paper_tables.py` | twelve `tables/*.tex`, every printed cell checked against its CSV |
