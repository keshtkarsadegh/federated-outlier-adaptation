# Digits_study01 - P21: the kd+fisher blend's own screen

The blend is the one penalty in this study that was **never screened**. P13 left
it out of the regularisation screen on purpose - a mixture is only worth pricing
once each half's own strength is known - and `study_emit.py reg-hybrid` then
built it from the kd and fisher winners and swept `mix` alone. Its `lam` and its
`T` were **inherited from the KD half**, not searched. It then held the best mean
score in both schedules and the nine highest preservation figures in the table,
which is exactly why leaving its coefficients unsearched is no longer tolerable:
the strongest method in the study is the one whose grid nobody ran.

This stage runs that grid, at the same ranking horizon and under the same
protocol as every other penalty: 8 of 10 per round, E=5, batch 64, the cohort
fold book, g-0 as the only initialisation, plain FedAvg on the server, the frozen
g-0 as the anchor, both schedules per task.

## The grid

    lambda_ewc in {0.1, 1, 2, 3, 5, 8, 10, 20, 50, 100, 200, 500, 1000}
    T          in {0.25, 0.5, 1, 2, 4, 8}
    mix        in {0.25, 0.5, 0.75}
    = 13 x 6 x 3 = 234 cells x 5 folds = 1170 tasks

**The same ranges the parents were screened over, not a reduced set.** The
strength row is `fisher`'s own thirteen lambdas from `REG_GRID_RANGES.md` §2 -
the row that covers the earlier work's `[0.2, 1, 4, 8, 10, 20, 100, 200]` and
resolves 1-10 - and the temperature row is `kd`'s own six from §6, the row that
stops at 8 because preservation was measured saturating there. Neither is
narrowed for this stage. A blend screened over less than its parents were would
answer a smaller question than the one being asked.

## The scale question, and how it is resolved

`AnchoredTrainer` computes

    lam * (mix * KD + (1 - mix) * Fisher)

so `lam` is **not** a free third knob: the two halves' coefficients are locked in
the ratio `mix : (1 - mix)`, and a single `lam` cannot put both parents' rows on
both halves at once. It can put one of them on one half exactly, and the Fisher
half is the one that needs it.

The blends that ran inherit `lam` from the **KD** half - 0.111111 in the parallel
schedule, 0.010101 in the cyclic one - so their Fisher term runs at
`lam * (1 - mix)`: 0.083 down to 0.028 where the concurrent selection chose
`lambda = 8`, and 0.0076 down to 0.0025 where the sequential one chose 0.1. That
is two decades below the row EWC was screened over, and it is the same fact
`REG_GRID_RANGES.md` already records as *"`mix = 0` does not reproduce the Fisher
winner"*. The blend has never been run with a Fisher term of the strength its
Fisher parent was selected at.

So the strength axis is written as the **EWC lambda the Fisher half receives**
and converted on the way to the line:

    lam = lambda_ewc / (1 - mix)

which is the same move the `kd` row makes in writing itself in `alpha` and
emitting `(1 - alpha) / alpha`: a cell names the quantity whose range was argued
for, not the number the trainer happens to take. The KD half then rides at
`lambda_ewc * mix / (1 - mix)`, which at `mix = 0.25` runs 0.033 to 333 and at
`mix = 0.75` runs 0.3 to 3000 - so the bottom of the row places both halves
inside their parents' live regions and the top is EWC-dominant by construction.

| `lambda_ewc` | `lam` at mix 0.25 | at mix 0.5 | at mix 0.75 |
|---|---|---|---|
| 0.1 | 0.133333 | 0.2 | 0.4 |
| 1 | 1.33333 | 2 | 4 |
| 2 | 2.66667 | 4 | 8 |
| 3 | 4 | 6 | 12 |
| 5 | 6.66667 | 10 | 20 |
| 8 | 10.6667 | 16 | 32 |
| 10 | 13.3333 | 20 | 40 |
| 20 | 26.6667 | 40 | 80 |
| 50 | 66.6667 | 100 | 200 |
| 100 | 133.333 | 200 | 400 |
| 200 | 266.667 | 400 | 800 |
| 500 | 666.667 | 1000 | 2000 |
| 1000 | 1333.33 | 2000 | 4000 |

**The two inherited points are below this row.** 0.111111 and 0.010101 are KD-half
lambdas; expressed as the Fisher-half coefficient they are 0.083 and below, under
the EWC row's floor of 0.1. That is the finding this stage exists to act on
rather than a gap to be papered over: the row starts where EWC's row starts,
because that is the range EWC was screened over. The three cells that ran keep
their full-horizon results and still compete in `reg-top3`; nothing here
supersedes them.

## Alpha is not a knob in the blend

`AnchoredTrainer` has no `alpha` parameter. `alpha` is the literature's
parameterisation of the `kd` space, converted to the trainer's `lam` by
`(1 - alpha) / alpha` because the two objectives have the same minimiser. Inside
the blend the KD term's coefficient is `lam * mix`, so `alpha` has no independent
existence and gridding it would be gridding `lam` twice. It is omitted, and the
grid stays three-dimensional.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by a later stage, which is
**gated separately and is not authorised**.

## The Fisher - no export needed

Every cell reads the Fisher diagonal of the shipped model. The study's own
`g0_selection.json` names **fold 1**, and this stage reads that number
rather than restating it. The lines name the directory as
`$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, which
`slurm/study_phase.sbatch` derives from the study's own `g0_selection.json`. So
no export is required; a task that needs the fold when the record is missing is
refused with exit 78 rather than run with an empty path component.

## Seeds

    700000 + 22000 + cell_index * 10 + fold

= 722001-724335, disjoint from every range this study has drawn.
234 cells span 2330, so this stage runs through three of the
thousand-wide blocks the scheme assumes - which is why the window was opened on
width rather than on the next free thousand.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-1170%20 \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s21_blend_screen.txt
```

## Cost

At `0.31 + 0.0345 x 8 x 5 = 1.69` s per round per family, with both
families per task and roughly 40% added for the teacher forward pass the KD half
needs on every batch:

| | seconds |
|---|---|
| training, 25 rounds, two families, +40% | 118 |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~166 (~2.8 min)** |

**1170 tasks is roughly 54 GPU-h** by that model, about
162 minutes of wall clock at `%20`. The model is the one P13's
README used and it ran high: that screen's 700 tasks cost 18.2 GPU-h measured,
94 s each against the 166 s predicted, which puts this stage nearer
**30 GPU-h**. Both are quoted because a stage this size should
be authorised against the pessimistic one.
