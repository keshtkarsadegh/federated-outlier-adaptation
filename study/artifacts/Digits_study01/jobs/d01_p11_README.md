# Digits_study01 - P11: the aggregation screen

P09 fixed the server rule at plain FedAvg and varied nothing, so it cannot say
how much of the adaptation - or of the forgetting - was the rule's doing. P11
varies the rule and nothing else.

10 outlier clients; 200 old writers; 8 of 10 per round (20% dropout); classes=digits. g-0 (winner fold 2) is the only initialisation.

| | |
|---|---|
| cells | **96** |
| folds | 5 |
| **tasks** | **480** |
| horizon | 25 rounds |

## A ranking horizon, not a reporting one

Nothing in this file is a reported number. A 25-round result is a ranking
signal; reading it as a result is reading a race at the quarter mark. The
winners are re-run at 100 rounds by **P12, which is gated
separately and is not yet authorised**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-480%40 \
       $J/study_phase.sbatch $J/d01_p11.txt
```

No exports needed: `study_phase.sbatch` defaults `FOA_STUDY_DIR` to
`studies/Digits_study01` and `FOA_NIST_CLASSES` to `digits`, and refuses any line
whose paths point elsewhere.

This file is independent of P08, P09 and P10 - it reads only the cohort, its
book, the old book and g-0, and writes only its own folders - so it can run
alongside them.

## After the screen: the selection

A CPU command, minutes, no GPU:

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/Digits_study01
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    agg-full --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p12.txt --expect 85
```

It writes `tables/p11_agg_method_winners.json` - each method's best cell with
its adaptation, spread and preservation - and emits `d01_p12.txt`.

**No method filtering.** Every one of the 17 methods gets its
best coefficients re-run, so the full stage compares eighteen methods rather
than one winner against nothing. That is 17 x
5 = **85 lines**, and
`--expect` makes the emitter exit non-zero if it produces any other number.

**`d01_p12.txt` IS NOT AUTHORISED TO RUN.** The emitter marks it in its own
header. It exists so the file is ready for review, not so it can be submitted.

### Grid-edge winners

A winner sitting at the lowest or highest value of its row means the optimum may
lie outside the range searched. Those are written to
`tables/BOUNDARY_HITS.txt` and printed - and they **never halt** anything: the
run is still the best of what was tried, and the fact is worth reporting rather
than acting on automatically.

## Method grouping

The full stage runs one line per **method** at its best coefficients, so
"method" is a definition, not an intuition. The 96 cells group into
17: a coefficient row like `eta_*` is one method at several
settings, while `weight_uniform` and `weight_capped` are two - a weighting rule
is not a coefficient of another weighting rule - and the two controls are
likewise distinct, because a control with the oracle stop rule armed is not a
setting of the control without it.

## Seeds

Each line's sampler seed is `700000 + 2000 + cell_index * 10 + fold`:
deterministic, distinct per line, and well clear of P09's 30001-30505, so no two
lines anywhere in this study draw the same participation pattern.

## Cost

At `0.31 + 0.0345 x 8 x 5 = 1.69` s per round per family:

| | seconds |
|---|---|
| training, 25 rounds, one family | 42 |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~90 (~1.5 min)** |

The two control cells run both families and cost about twice the training part.

**480 tasks is roughly 12 GPU-h**, about
18 minutes of wall clock at
`--array=...%40`.

Startup is a large share of a short task here: the cohort is ten small writers,
so a screening run trains on 579 images per round-epoch while the
evaluation reads ~5,297 old test rows per round. Batching the five folds
of a cell into one element would cut the total substantially - but one task per
cell-fold keeps a failure to one element and matches every other stage.
