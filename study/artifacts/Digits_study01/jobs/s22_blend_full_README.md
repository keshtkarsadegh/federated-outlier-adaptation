# Digits_study01 - P22: the blend's finals, at the coefficients that were screened

P21 gave the kd+fisher blend the grid it never had - 13 EWC lambdas x 6
temperatures x 3 mixes, 234 cells x 5 folds, 25 rounds, both schedules per task,
1,170 elements, all COMPLETED. This stage is what that screen is for: each
schedule's best cell, re-run at the full 100-round horizon on the same five
folds, so the blend can be read beside the seven penalties of `s17_reg_full4.txt`
on the basis they were all measured on.

**These are new arms, not a re-run of `s18_hybrid.txt` or `s19_hybrid_seq.txt`.**
Those swept `mix` alone at a `lam` and a `T` inherited from the KD winner - the
Fisher half riding two decades below the row EWC was screened over. The cells
here carry coefficients the screen actually measured. The old blends keep their
folders, their seeds and their results; nothing here supersedes them.

## The rule, unchanged

`study_emit.py blend-full` applies **reg-full's** selection to the blend's own
catalogue, which reg-full cannot see: `reg_cells.blend_cells()` is deliberately
not part of `screen_cells()`, so the loop over the seven methods walks past it.
Same score, same weight, same basis, same tie-break, same per-family selection:

    score = (adaptation - A_0) - (P_0 - preservation),   w = 1

on the **validation** columns of the 25-round screen, fold-mean over all five
folds, argmax within each schedule. The shipped model's own accuracies are read
from the study rather than assumed: `A_0 = 0.8225` on the cohort, `P_0 = 0.9986`
on the source population. One winner per schedule, because the blend reaches the
finals as one method and a schedule is a different claim from the other.

## What was selected

| schedule | cell | lambda_ewc | T | mix | `lam` | adaptation | preservation | score |
|---|---|---|---|---|---|---|---|---|
| concurrent | `blend_lam0p1_T0p5_mix0p25` | 0.1 | 0.5 | 0.25 | 0.133333 | 0.8810 | 0.9958 | +0.0557 |
| sequential | `blend_lam0p1_T0p25_mix0p5` | 0.1 | 0.25 | 0.5 | 0.2 | 0.8801 | 0.9962 | +0.0552 |

234 cells were scored in each schedule; none fell back to a test column. The
next four in each, on the same rule:

| schedule | runner-up | adaptation | preservation | score |
|---|---|---|---|---|
| concurrent | `blend_lam0p1_T0p5_mix0p5` | 0.8801 | 0.9950 | +0.0540 |
| concurrent | `blend_lam0p1_T0p25_mix0p5` | 0.8772 | 0.9964 | +0.0524 |
| concurrent | `blend_lam0p1_T0p25_mix0p25` | 0.8765 | 0.9957 | +0.0511 |
| concurrent | `blend_lam0p1_T2_mix0p5` | 0.8754 | 0.9967 | +0.0511 |
| sequential | `blend_lam0p1_T2_mix0p25` | 0.8792 | 0.9960 | +0.0541 |
| sequential | `blend_lam0p1_T1_mix0p25` | 0.8783 | 0.9957 | +0.0529 |
| sequential | `blend_lam0p1_T0p5_mix0p5` | 0.8782 | 0.9956 | +0.0527 |
| sequential | `blend_lam0p1_T1_mix0p5` | 0.8773 | 0.9960 | +0.0523 |

The selection record is `tables/p21_blend_winners.json` in the study root, and a
copy ships beside P13's and P14's under `study/artifacts/Digits_study01/tables/`.

## Both winners sit at a grid edge, and it is the same edge

Reported by the emitter, appended to `tables/BOUNDARY_HITS.txt`, and not a
reason to stop - but it is the finding to read first:

    blend_lam0p1_T0p5_mix0p25:  lam = 0.133333 is the LOW end of its row
    blend_lam0p1_T0p5_mix0p25:  mix = 0.25     is the LOW end of {0.25, 0.5, 0.75}
    blend_lam0p1_T0p25_mix0p5:  T   = 0.25     is the LOW end of {0.25 ... 8}

Both schedules chose **`lambda_ewc = 0.1`, the bottom of the EWC row**, and the
whole top of each ranking is that one strength. The blend wants a *weaker*
Fisher term than the row EWC was screened over provides - which is, read the
other way round, the reason the inherited-coefficient blends of s18 and s19 did
so well with a Fisher half two decades below that row. The optimum may lie below
0.1, and this stage does not settle it: it prices the best of what was searched.

## The lines

10 tasks: 2 schedules x 5 folds, at 100 rounds, 8 of 10 clients per round, E=5,
batch 64, the cohort fold book, g-0 as the only initialisation, plain FedAvg on
the server, the frozen g-0 as the anchor.

**The schedule is in the parent**, exactly as in `s17_reg_full4.txt`: each task
runs *both* families, and the tag records which one the folder is read for. Two
tasks would otherwise be free to choose the same cell and name one folder.

**The rate is the search rate, not the study rate.** 8 of 10, because that is
what the screen these winners come from ran at and what the seven penalties of
s17 were re-run at. A final emitted at the study's own 9 of 10 would price the
blend on an easier loop than the table it is going to be read in.

## Seeds

    700000 + 26000 + arm_index * 10 + fold   =  726001-726005, 726011-726015

Its own block. The screen opened at 22000 and runs to 724335; the combination
stage opens at 730351; two arms span twenty, so the finals sit in the gap
between them rather than at the next round number. `tools/check_seeds.py` is
run over every shipped task file at once, which is what actually establishes
the disjointness - a hand-kept range is what let the combination stage draw
seeds the regularisation screen had already used.

## The Fisher - no export needed

Both cells read the Fisher diagonal of the shipped model. The lines name it as
`$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, and
`slurm/study_phase.sbatch` resolves `$G0_FOLD` from the study's own
`g0_selection.json`. A task that needs the fold when the record is missing is
refused with exit 78 rather than run with an empty path component.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-10%20 \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s22_blend_full.txt
```

## Cost

The screen measured 87 s per element at 25 rounds with both schedules and the
KD half's teacher pass. About 48 s of that is interpreter, torch, the two books
and the final evaluation, so the training portion is roughly 39 s and scales
with the horizon: 4 x 39 + 48 is about **3.5 minutes per task**, and 10 tasks
that run at once is well under a GPU-hour. This is the cheapest stage in the
programme; the 1,170-element screen it selects from cost 28 GPU-h.
