# Digits_study01 - P13: the regularisation screen

P11 asked what the **server** rule is worth. P13 asks the other half - what the
**client-side penalty** is worth - under the same protocol with plain FedAvg on
the server, changing only the term added to the client's loss.

The anchor is the **frozen g-0** in every cell. That is the premise: the penalty
measures distance from the model the study is trying not to forget.

| method | cells | tasks |
|---|---|---|
| param_l2 | 7 | 35 |
| fisher | 8 | 40 |
| fisher_scaled | 8 | 40 |
| logit_l2 | 5 | 25 |
| feature_l2 | 5 | 25 |
| kd | 49 | 245 |
| ntd | 35 | 175 |
| **total** | **117** | **585** |

Both schedules run per task (`--aggregation fedavg`), as this stage always has.
A penalty that helps a server average and one that helps a sequential walk are
different findings, so the selector ranks them separately.

## The Fisher - no export needed

16 cells (80 tasks) need the Fisher
diagonal of the shipped model. **g-0 v2 won on fold 2**, and I checked
the artefacts are on disk before trusting the path: `fisher.pt` and
`global_params.pt`, 6.7 MB each, present for **all five** folds at
`g0_fold<k>/global_results/fisher`.

The lines name it as `$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, and
**`study_phase.sbatch` now derives `G0_FOLD` from the study's own
`g0_selection.json`**. So no export is required.

That is the safer of the two options you offered: a number that is already
written down should not also have to be remembered, and a forgotten export would
expand to an empty path component and fail 80 array
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
sbatch --account=$FOA_ACCOUNT --array=1-585%40 \
       $J/study_phase.sbatch $J/d01_p13.txt
```

## After the screen

Two CPU commands, minutes each, both stamped so the files they write cannot be
mistaken for authorised work.

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/Digits_study01
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    reg-full --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p14.txt --expect 70
```

**Per family, not per method.** Each penalty's best cell is chosen within each
schedule, giving 7 methods x 2 families x
5 folds = **70 lines**. The full-horizon
parents carry the family they were selected for - the same cell often wins in
both, and without the tag two tasks would name one folder and race into it.

Then the blend, once both halves are known:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    reg-hybrid --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p14_hybrid.txt --expect 15
```

The hybrid is deliberately **not** in the screen: a blend is only worth pricing
once each penalty's own strength is known. Screening it would have meant
guessing the KD half's `lam` and `T` and the Fisher half's `lam` before any was
measured - a three-dimensional grid for a question that only becomes meaningful
after the two one-dimensional ones are settled. It costs
3 cells x 5 folds =
**15 tasks** instead of hundreds.
`mix = 1` would reproduce the KD winner exactly, which is what makes three blend
points readable as a line between the two methods.

Every emitter takes `--expect` and exits non-zero on any other count, and every
file it writes is stamped `NOT YET AUTHORISED`.

Grid-edge winners go to `tables/BOUNDARY_HITS.txt` and to stdout and never halt
anything.

## Cost

At `0.31 + 0.0345 x 9 x 5 = 1.86` s per round per family, with
both families per task and roughly 40% added for the teacher forward pass the
output-space penalties need on every batch:

| | seconds |
|---|---|
| training, 25 rounds, two families, +40% | 130 |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~178 (~3.0 min)** |

**585 tasks is roughly 29 GPU-h**, about
43 minutes of wall clock at `%40`.

The Fisher-weighted cells are cheaper than that estimate - a Fisher penalty is a
parameter-space term with no teacher pass - and the kd/ntd/logit/feature cells
are the ones carrying the 40%. Taking the whole file at the higher rate is the
honest way round.
