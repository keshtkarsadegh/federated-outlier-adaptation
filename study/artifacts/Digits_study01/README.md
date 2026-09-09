# Digits_study01 - the artefacts that travel with the source

This directory is the study's **metadata core**: the records that say what was
selected, what was run, against which rows, and what came out - everything a
reader needs to check a claim in the manuscript against something other than the
manuscript. It is 7.3 MB and 152 files besides this one, which is small enough
to live in git and be diffed like source.

It is not the study. The study is 3,855 run folders and about 4 million files.
What is here is the part of it that is *evidence* rather than *bulk*, plus the
exact commands that reproduce the bulk. Everything excluded is named below with
the reason it is excluded and the way to get it back; nothing has been left out
silently.

Every absolute path in these files has been rewritten to the placeholder the
runner exports, by `tools/sanitize_artifacts.py`. `$FOA_STUDY_DIR` is this study
root, `$FOA_NIST28_DIR` the converted dataset, `$FOA_PROJECT_DIR` their common
parent. The originals on the machine that ran the study still carry their real
paths and were not touched; only these copies were rewritten.

## What is here, and what reads it

| artefact | what it is | the tool that reads it |
|---|---|---|
| `selection.lock.json` | the frozen cohort: which writers, from which scores | `freeze_selection.py verify` |
| `g0_evaluations.json` | the shipped model on the source population - `P0` | `compare_arms.py`, `report_tables.py`, `study_emit.py` |
| `g0_perfold_evaluations.json` | the shipped model on the cohort's rows - `A0` | `compare_arms.py`, `report_tables.py`, `study_emit.py` |
| `g0_c5_evaluations.json`, `g0_c10_evaluations.json`, `g0_c20_evaluations.json` | the same two baselines at each federation size | `report_tables.py --what sizes` |
| `g0_selection.json` | which fold's g-0 was shipped, and why | `study_emit.py`, the trainers' Fisher export |
| `g0_fold*/global_results/global_metrics.json` | g-0's own training history, per fold | `export_baseline_views.py` |
| `fold_books/*.foldbook.npz` | the row-level splits every run was scored against | `federated_outlier_adaptation.data.fold_book` |
| `outliers/*.json`, `outliers/*.csv` | writer scores, the pools, the cohorts, the extreme arrangements | `freeze_selection.py`, `describe_cohort.py`, `check_programme.py` |
| `tables/p11_*.json`, `tables/p13_reg_method_winners.json` | each method's winner, per schedule | `study_emit.py`, `study_record.py` |
| `tables/p12_agg_top3.json`, `tables/p13_reg_top3.json` | the two shortlists the cross was emitted from | `compare_arms.py`, `study_emit.py combos` |
| `tables/p14_hybrid_construction*.json` | which two cells each family's blend was built from | `study_emit.py reg-hybrid`, `compare_arms.py --what blends` |
| `tables/p15_*.json` | the combination grid and its winner | `report_tables.py --what combos`, `study_emit.py` |
| `tables/p18_*.json`, `tables/p20_*.json`, `tables/p21_c10_d10.json` | the carried arms at each federation size | `report_tables.py --what sizes` |
| `tables/p21_blend_winners.json` | which blend cell each schedule's own screen crowned, and how many it ranked | `study_emit.py blend-full`, `study_record.py` |
| `tables/p23_combo_tune_winners.json` | **the extension**: the jointly-tuned cell per schedule, its dials and the rule it runs | `study_emit.py combo-tune-full`, `export_extension_views.py` |
| `tables/cohort_table.*`, `tables/cohort_composition.csv` | who is in the cohort and what they hold | `describe_cohort.py` |
| `tables/clients_acc_on_g0.json` | the shipped model per cohort client | `study_emit.py`, `analyse_signals_fairness_cost.py` |
| `tables/paper/*.csv` | the manuscript bundle, one CSV per view | `report_tables.py --what all --csv` |
| `tables/paper/extension_combo_tune.csv`, `extension_combo_screen.csv` | **the extension**, in the same directory and under a name that says so: the tuned pair against the four arms it is offered against, and the screen that chose it. Read by no figure and no generator | `export_extension_views.py` |
| `tables/paper_figures/traces_*.csv`, `extreme_stop_rounds.csv` | the per-round views the figures are drawn on | `export_traces.py`, then `tools/paper_figures/fig_*.py` |
| `tables/paper_figures/g0_training.csv`, `isolated_clients.csv` | the two baseline views | `export_baseline_views.py`, `fig_baselines.py` |
| `tables/paper_figures/combos_folds.csv` | each combination against BOTH of its halves | `export_combo_folds.py` |
| `tables/stopping/stopping_*.csv` | what the fixed horizon cost every arm | `stopping_table.py --csv` |
| `tables/stopping/plateau_*.csv` | what a plateau on the cohort's own accuracy would have delivered instead | `plateau_rule.py --out` |
| `tables/combos.csv`, `composition.csv`, `blends.csv` | the paired differences, folds expanded | `compare_arms.py --csv` |
| `tables/weight_sensitivity_*.csv` | do the winners survive a different `w`? | `weight_sensitivity.py` |
| `tables/BOUNDARY_HITS.txt` | winners sitting at the edge of their grid row | written by `study_emit.py`, read by hand |
| `tables/gallery/cohort_vs_typical.*` | the cohort against a typical draw | `make_digits_p05v2.py` |
| `jobs/s2[1-4]_*.txt` | the blend's screen and finals, and the extension's two stages - submitted after the views above were first cut | `check_seeds.py`, `check_programme.py`, `submit_chain.py` |
| `jobs/*.txt` | every task line that was submitted, verbatim | `check_seeds.py`, `check_programme.py`, `submit_chain.py` |
| `jobs/*_README.md` | what each stage's file was emitted for | read by hand |
| `figures/extreme_stopping.png` | when the extreme cases should have stopped | `extreme_stopping.py --fig` |
| `signals/signals_pareto_*.png` | the eight forgetting signals, cost against catch | `foa signals` |

`tables/superseded/` and `jobs/superseded/` are deliberately absent: they are the
runs a later stage replaced, they are kept in the study tree as a lab record, and
shipping them would put two answers to the same question in front of a reader
with nothing to say which is current.

## What is not here, and how to get it

**Checkpoints - 51 MB.** `g0_model`, the five `g0_fold*/global_model`, and
`global_results/fisher_g0`. The five `g0_fold*/global_results/global_metrics.json`
beside them are records rather than weights - the training history the shipped
model's own figure is drawn from - and they are here. The weights are not: they cannot be
diffed, they do not answer any question this directory exists to answer, and they
would be most of the repository's size. `g0_model` is the one artefact that
cannot be regenerated from what is here - it is the shipped model the whole study
is measured against - and it is the owner's to publish separately if a reader
needs to rerun rather than re-read.

**The two bulk signal files - 195 MB.** `signals/signals_summary.json` (136 MB)
and `signals/signal_stopping.csv` (59 MB), together with the 9 MB
`signals/signal_correlations.csv`. All three are derived: one pass over the run
records recomputes them exactly.

    foa signals --root "$FOA_STUDY_DIR"

writes all of them, plus `signal_selection.csv` and the six Pareto PNGs that
*are* shipped here, into `$FOA_STUDY_DIR/signals/`. Add `--no-plots` to skip the
figures. Nothing about them is a judgement call, which is why they are the first
thing to drop.

**The per-run records - a release asset.** Every `accuracies_*.json` and
`summary_0.json` under the 3,855 `d01_*` run folders, plus the 380 JSONs of the
`centralized_c*` and `isolated_c*` reference rungs: 12,210 files, 1.0 GB raw.
These are the numbers every table in the study is computed from, so they are
published - but as a release asset rather than in the tree, because git is the
wrong place for half a gigabyte of machine output that no one will diff.

    Digits_study01_records.tar.gz
    sha256 f0ed575ddd7bdadeadb35b2553f5115378058978901fb7228a92b5542dacf911
    130,058,517 bytes

Paths inside it are relative to the study root, so it unpacks over
`$FOA_STUDY_DIR` - but the asset is only half of a study root and this directory
is the other half. The two have to be brought together before any tool will run,
which is three commands: see **Assembling the reviewer tree** in
`docs/REPRODUCE.md` section 10. The asset was sanitised by the same
`tools/sanitize_artifacts.py` run over a staged copy before packing, so it
carries no machine paths either.

**Everything else in the study tree.** Per-round checkpoints, the client-level
intermediate JSONs, the per-run PNGs, `logs/`. Regenerated only by rerunning the
stage, which is what `docs/REPRODUCE.md` is for.

## Checking this directory

    python tools/sanitize_artifacts.py --dir study/artifacts/Digits_study01 --check
    pytest tests/test_release_artifacts.py

The first exits zero only if no absolute machine path survives anywhere under
this directory and no JSON value would still be rewritten. The second asserts the
same thing over the raw text - including keys, lists and free-text fields, which
the sanitiser's JSON walk would not reach - because a shipped artefact carrying
somebody's home directory still parses, still verifies, and is noticed by nothing
downstream.
