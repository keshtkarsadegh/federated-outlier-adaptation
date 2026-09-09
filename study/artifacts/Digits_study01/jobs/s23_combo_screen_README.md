# Digits_study01 - P23: joint tuning of the best pair (**an extension**)

**This is not a stage of the study.** Every stage the paper reports has run, and
`COMBINATIONS.md` already records what the combination cross found. This is an
extension added afterwards, it is named as one in the task file's header, in its
chain manifest and in its selection record, and `REPRODUCE.md` keeps it out of
the core stage table deliberately. Nothing in the core programme reads anything
it produces: the shortlists, the cross and the crowning are unchanged and are not
re-emitted.

## The question the cross could not ask

`s20_combos4.txt` crossed the top three server rules with the top three penalties
per schedule. A pair there is **a rule at the coefficients it won on alone beside
a penalty at the coefficients it won on alone** - the two shortlists were
selected independently, and nothing in the cross ever moved the two together. So
the study's finding, that the two halves do not measurably compose, is measured
at exactly one point of a joint grid: the point where each half is best *in the
other's absence*. That is the honest reading of what was run, and it is also the
reading with the obvious gap in it.

This extension screens the joint grid, for the pair that leads each schedule.

## The pair

Read by **TEST** score off the two shipped views - `tables/paper/agg-winners.csv`
and `tables/paper/reg-winners.csv` - which is the owner's documented departure
from ranking on validation and is the order those views carry.

| schedule | best rule alone | best penalty alone |
|---|---|---|
| parallel (`concurrent`) | `eta_0p95` - `con_delta_eta`, score 0.0775 | `hybrid_seq_mix0p75` - the KD+EWC blend, score 0.1050 |
| cyclic (`sequential`) | `seq_delta_capped` - score 0.0702 | `hybrid_seq_mix0p75` - score 0.1069 |

The penalty is the same object in both schedules and the rule is not, which is
why the grid is not the same size on both sides: `eta_0p95` is one setting of a
coefficient row and `seq_delta_capped` is a rule with no coefficient at all.

**Every non-grid flag is copied from the shipped `s20` lines that pair these two
halves** - `eta_0p95_hybrid_seq_mix0p5` and `seq_delta_capped_hybrid_mix0p5` - so
a line here differs from one there in the horizon and in the four (or five)
coefficients, and in nothing else: same cohort, same books, same rate, same
`--outer-workers 1 --inner-workers 1`, same `--extended-aggregations`, and the
`fisher_path` convention `s19_hybrid_seq.txt` established.

## The grid

    c_ewc  in {0.05, 0.1, 0.3}      EWC coefficient
    c_kd   in {0.05, 0.11, 0.2}      KD coefficient
    T      in {0.25, 2}         KD temperature
    m      in {0.25, 0.5, 0.75}      blend weight
    eta_s  in {0.9, 0.95, 1}      PARALLEL ONLY

    3 x 3 x 2 x 3 = 54 penalty settings
      x 3 server steps = 162 parallel cells
      x 1 (no rule knob)        = 54 cyclic cells
      = 216 cells x 5 folds = 1080 tasks at 25 rounds

### Where the values come from

**`c_ewc`.** EWC's own selections in this study sat at `lambda = 8` on the
parallel schedule and `lambda = 0.1` on the cyclic one - two decades apart, which
already says the coefficient is not the same object in the two loops. What
settles the range is the blend's own screen: `tables/p21_blend_winners.json`
records `lambda_ewc = 0.1` chosen in **both** schedules, and 0.1 is the *bottom*
of the thirteen-value row `REG_GRID_RANGES.md` section 2 was screened over. A
winner on the floor of a row is a statement that the live region is at or below
it, so this row brackets the floor - 0.05 under it, 0.1 at it, 0.3 above - rather
than reaching back up to 8. Nothing here re-searches the standalone EWC row; that
selection stands.

**`c_kd`.** The `kd` row is written in the literature's `alpha` and emits
`lam = (1 - alpha) / alpha`, so its two winners - `kd_T0p25_a0p9` on the parallel
schedule and `kd_T2_a0p99` on the cyclic one - are `lam = 0.111` and
`lam = 0.0101`. This row brackets the first (0.05 below, 0.11 at it, 0.2 above)
and steps down toward the second. As with `c_ewc`, the bracket is placed on a
measured winner rather than on a decade chosen by eye.

**`T`.** The two temperatures the `kd` row's own winners sat at: 0.25 and 2. Not
the full six-point row of section 6. This grid pays for four other axes and the
two values here are measured points rather than a bracket around a guess -
and the blend's own screen chose 0.5 and 0.25, both inside the interval they
span.

**`m`.** The three the emitted blends swept, so this axis can be read directly
against `s18`, `s19` and the `hybrid_*` rows of `reg-winners.csv`.

**`eta_s`.** `eta_0p95` is a setting of a coefficient row, so the rule half of
the parallel pair *has* a knob and this grid moves it: 0.95 is the winner, 0.9 is
one step below, and 1.0 is the plain full step - so the row also says what the
rule is worth at all when the penalty beside it moves. `seq_delta_capped` has no
coefficient, which is why the cyclic half is a third the size.

## How the dials reach the trainer

`AnchoredTrainer` computes

    lam * ( mix * KD + (1 - mix) * Fisher )

and the penalty this grid names is

    m * c_kd * KD + (1 - m) * c_ewc * Fisher

Matching term by term gives `lam * mix = m * c_kd` and
`lam * (1 - mix) = (1 - m) * c_ewc`, hence

    lam = m * c_kd + (1 - m) * c_ewc
    mix = m * c_kd / lam

which is :func:`reg_cells.combo_tune_lam_mix`, sitting beside
`blend_lam_of_ewc` and doing the other half of that function's job: where the
blend's own screen wrote the strength in **one** parent's units and let the other
half ride, this names **both** coefficients and solves for the pair the trainer
takes.

**The trainer's `mix` is not the owner's `m`.** It is the KD half's share of the
total penalty weight, and it moves with `c_ewc` and `c_kd` as well as with `m` -
the two coincide only where the two coefficients are equal, which on this grid is
`c_ewc = c_kd = 0.05` alone. A grid that wrote `m` straight into `mix` would be
sweeping a different object than the one it named, and nothing in the emitted
line would say so: the line would carry a legal coefficient and run.

| `c_ewc` | `c_kd` | `m` | `lam` | trainer `mix` |
|---|---|---|---|---|
| 0.05 | 0.05 | 0.25 | 0.05 | 0.25 |
| 0.05 | 0.05 | 0.5 | 0.05 | 0.5 |
| 0.05 | 0.05 | 0.75 | 0.05 | 0.75 |
| 0.05 | 0.11 | 0.25 | 0.065 | 0.423077 |
| 0.05 | 0.11 | 0.5 | 0.08 | 0.6875 |
| 0.05 | 0.11 | 0.75 | 0.095 | 0.868421 |
| 0.05 | 0.2 | 0.25 | 0.0875 | 0.571429 |
| 0.05 | 0.2 | 0.5 | 0.125 | 0.8 |
| 0.05 | 0.2 | 0.75 | 0.1625 | 0.923077 |
| 0.1 | 0.05 | 0.25 | 0.0875 | 0.142857 |
| 0.1 | 0.05 | 0.5 | 0.075 | 0.333333 |
| 0.1 | 0.05 | 0.75 | 0.0625 | 0.6 |
| 0.1 | 0.11 | 0.25 | 0.1025 | 0.268293 |
| 0.1 | 0.11 | 0.5 | 0.105 | 0.52381 |
| 0.1 | 0.11 | 0.75 | 0.1075 | 0.767442 |
| 0.1 | 0.2 | 0.25 | 0.125 | 0.4 |
| 0.1 | 0.2 | 0.5 | 0.15 | 0.666667 |
| 0.1 | 0.2 | 0.75 | 0.175 | 0.857143 |
| 0.3 | 0.05 | 0.25 | 0.2375 | 0.0526316 |
| 0.3 | 0.05 | 0.5 | 0.175 | 0.142857 |
| 0.3 | 0.05 | 0.75 | 0.1125 | 0.333333 |
| 0.3 | 0.11 | 0.25 | 0.2525 | 0.108911 |
| 0.3 | 0.11 | 0.5 | 0.205 | 0.268293 |
| 0.3 | 0.11 | 0.75 | 0.1575 | 0.52381 |
| 0.3 | 0.2 | 0.25 | 0.275 | 0.181818 |
| 0.3 | 0.2 | 0.5 | 0.25 | 0.4 |
| 0.3 | 0.2 | 0.75 | 0.225 | 0.666667 |

(`T` and `eta_s` pass through untouched; the table is the same for both
temperatures and, on the parallel side, for all three server steps.)

## Deduplication

**0 cell(s) removed.** Two dial settings collide when they hand the
trainer the same `(lam, mix, T)` under the same rule - two folders, two seeds and
one experiment, with nothing in the run records to say so. The check is on the
coefficients rather than on the dials, because the dials are what differ. It
finds nothing on this grid: a collision needs `m * c_kd` and `(1 - m) * c_ewc` to
repeat *together*, and the nine `(m, c_kd)` products here are all distinct. The
check is emitted anyway, so that a widened row cannot quietly pay twice.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by `s24_combo_full.txt`,
which `study_emit.py combo-tune-full` emits under the same rule every other
selection in this study uses.

## The Fisher - no export needed

Every cell reads the Fisher diagonal of the shipped model. The study's own
`g0_selection.json` names **fold 1**, and this stage reads that number
rather than restating it. The lines name the directory as
`$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, which
`slurm/study_phase.sbatch` derives from the study's own `g0_selection.json`. So
no export is required; a task that needs the fold when the record is missing is
refused with exit 78 rather than run with an empty path component.

## Seeds

    700000 + 60000 + cell_index * 10 + fold

= 760001-762155, disjoint from every range this study has drawn. 216
cells span 2150, more than the thousand-wide block the scheme assumes, so
the window is opened on width rather than on the next free thousand - P21's
lesson. The highest seed any shipped file draws is 750105, so the extension's
blocks also sit visibly clear of the programme's rather than interleaved with
them.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-1080%20 \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s23_combo_screen.txt
```

## Cost

At `0.31 + 0.0345 x 8 x 5 = 1.69` s per round per family, with **one**
family per task - a combination line names one rule - and roughly 40% added for
the teacher forward pass the KD half needs on every batch:

| | seconds |
|---|---|
| training, 25 rounds, one family, +40% | 59 |
| per-round evaluations | ~6 |
| final three-category evaluation | ~2 |
| interpreter, torch, cache, two books | ~40 |
| **wall per task** | **~107 (~1.8 min)** |

**1080 tasks is roughly 32 GPU-h** by that model, about
96 minutes of wall clock at `%20`. The model ran low on P13,
whose two-family 25-round tasks cost 94 s measured against 78 s predicted; the
same ratio on a one-family task puts this nearer **14 GPU-h**.
Both are quoted because a stage this size should be authorised against the
pessimistic one.
