# Digits_study01 - P24: the jointly-tuned pair at the full horizon (**an extension**)

**This is not a stage of the study.** It is the finals of the extension
`s23_combo_screen.txt` screened, and nothing the paper reports reads what either
produces: the shortlists, the combination cross, the crowning, the carry
settings and the paper views are unchanged and are not re-emitted. Its record,
`tables/p23_combo_tune_winners.json`, says the same thing in its own text.

E1 screened the joint grid of the pair that leads each schedule - 216 cells x 5
folds, 25 rounds, 1,080 elements, all COMPLETED, 18.4 GPU-h. This stage is what
that screen is for: each schedule's best cell, re-run at the full 100-round
horizon on the same five folds, so the tuned arm can be read beside the eighteen
combinations of `s20_combos4.txt` on the basis they were all measured on.

**These are new arms, not a re-run of `s20_combos4.txt`.** Every pair there is a
server rule at the coefficients it won on *alone* beside a penalty at the
coefficients it won on *alone*; the cells here carry coefficients a joint screen
actually measured. The eighteen combinations keep their folders, their seeds and
their results; nothing here supersedes them.

## The rule, unchanged

`study_emit.py combo-tune-full` applies **reg-full's** selection to the
extension's own catalogue and to nothing else. Same score, same weight, same
basis, same tie-break, same per-schedule selection:

    score = (adaptation - A_0) - (P_0 - preservation),   w = 1

on the **validation** columns of the 25-round screen, fold-mean over all five
folds, argmax within each schedule. The shipped model's own accuracies are read
from the study rather than assumed: `A_0 = 0.8225` on the cohort, `P_0 = 0.9986`
on the source population. A different rule here would make the arm it crowns
incomparable with every other winner in the study, which is the one thing an
extension must not do - it is added to be read *against* the programme.

A schedule is ranked only against its own cells. A cell names one server rule,
which lives in one family, so the parallel half is 162 cells with a server step
to move and the cyclic half is 54 with no rule knob at all.

## What was selected

| schedule | cell | `c_ewc` | `c_kd` | `T` | `m` | `eta_s` | `lam` | trainer `mix` | adaptation | preservation | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| concurrent | `ctune_ewc0p05_kd0p05_T0p25_mix0p75_eta0p9` | 0.05 | 0.05 | 0.25 | 0.75 | 0.9 | 0.05 | 0.75 | 0.8904 | 0.9949 | +0.0643 |
| sequential | `ctune_ewc0p05_kd0p05_T0p25_mix0p75` | 0.05 | 0.05 | 0.25 | 0.75 | - | 0.05 | 0.75 | 0.8893 | 0.9961 | +0.0643 |

162 cells were scored in the parallel schedule and 54 in the cyclic one; none
fell back to a test column and every cell carried all five folds. **The two
schedules chose the same four penalty dials**, which is not something the grid
forced: the two halves were ranked separately over different-sized fields.

The next four in each, on the same rule:

| schedule | runner-up | adaptation | preservation | score |
|---|---|---|---|---|
| concurrent | `ctune_ewc0p05_kd0p11_T0p25_mix0p75_eta1` | 0.8884 | 0.9956 | +0.0629 |
| concurrent | `ctune_ewc0p05_kd0p05_T2_mix0p75_eta0p95` | 0.8876 | 0.9956 | +0.0622 |
| concurrent | `ctune_ewc0p1_kd0p05_T2_mix0p5_eta0p95` | 0.8875 | 0.9957 | +0.0621 |
| concurrent | `ctune_ewc0p05_kd0p05_T2_mix0p75_eta1` | 0.8876 | 0.9954 | +0.0620 |
| sequential | `ctune_ewc0p05_kd0p05_T0p25_mix0p25` | 0.8885 | 0.9960 | +0.0635 |
| sequential | `ctune_ewc0p05_kd0p2_T0p25_mix0p5` | 0.8885 | 0.9955 | +0.0629 |
| sequential | `ctune_ewc0p05_kd0p11_T0p25_mix0p5` | 0.8877 | 0.9963 | +0.0629 |
| sequential | `ctune_ewc0p05_kd0p2_T0p25_mix0p25` | 0.8874 | 0.9958 | +0.0622 |

The whole top of each ranking is `c_ewc = 0.05`. The worst cell in either
schedule still scores +0.034, so the grid separates the arm from nothing: every
point of it is a working configuration and the spread across 216 cells is about
three points of score.

The selection record is `tables/p23_combo_tune_winners.json` in the study root,
and a copy ships beside P13's and P21's under
`study/artifacts/Digits_study01/tables/`.

## Every winner sits at a grid edge, in every dial but one

Reported by the emitter, appended to `tables/BOUNDARY_HITS.txt`, and not a
reason to stop - but it is the finding to read first. Nine hits across the two
winners:

    ctune_..._eta0p9:  c_ewc = 0.05  LOW  end of {0.05, 0.1, 0.3}
    ctune_..._eta0p9:  c_kd  = 0.05  LOW  end of {0.05, 0.11, 0.2}
    ctune_..._eta0p9:  T     = 0.25  LOW  end of {0.25, 2}
    ctune_..._eta0p9:  m     = 0.75  HIGH end of {0.25, 0.5, 0.75}
    ctune_..._eta0p9:  eta_s = 0.9   LOW  end of {0.9, 0.95, 1}
    ctune_...:         the same four, without the server step

Four of the five parallel dials and all four cyclic ones are at an end of their
row. Read plainly: the joint optimum wants a **weaker** penalty than either half
was selected at alone, and - on the parallel side - a **smaller** server step
than the rule's own row bottoms out at within this grid. That is the same
direction P22 found for the blend, where both schedules chose the floor of the
EWC row, and it is consistent with what the screen shows across all 216 cells:
score falls monotonically as either coefficient rises.

**This stage does not settle where the optimum is.** It prices the best of what
was searched, on a grid the owner specified, and reports honestly that the best
of it is at the edge.

## The lines

10 tasks: 2 schedules x 5 folds, at 100 rounds, 8 of 10 clients per round, E=5,
batch 64, the cohort fold book, g-0 as the only initialisation, the frozen g-0
as the anchor, and one server rule per line.

**One line is one schedule.** Each cell names a single aggregation rule - so a
task produces one result, which is `s20_combos4.txt`'s shape and not the
screens'. A `--aggregation fedavg` line would run the cyclic family under a rule
chosen for the parallel one.

**The schedule is in the parent as well**, as in `s17_reg_full4.txt` and
`s22_blend_full.txt`. The cell ids already carry it - only the parallel rule has
a coefficient to name - but a folder that says which schedule's finals it
belongs to cannot be misread by a selector that does not know that.

**The rate is the search rate, not the study rate.** 8 of 10, because that is
what the screen these winners come from ran at and what `s20_combos4.txt` was
emitted at. A final emitted at the study's own 9 of 10 would price the tuned arm
on an easier loop than the table it is going to be read in.

## Seeds

    700000 + 64000 + arm_index * 10 + fold   =  764001-764005, 764011-764015

Its own block, above the screen's 760001-762155 and above 750105, the highest
seed any stage of the programme draws - so the extension's two blocks sit
visibly outside the programme's rather than interleaved with it.
`tools/check_seeds.py` is run over every shipped task file at once, which is what
actually establishes the disjointness: a hand-kept range is what let the
combination stage draw seeds the regularisation screen had already used.

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
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s24_combo_full.txt
```

## Cost

The screen measured 61 s per element at 25 rounds with one schedule and the KD
half's teacher pass. About 40 s of that is interpreter, torch, the two books and
the final evaluation, so the training portion is roughly 21 s and scales with the
horizon: `4 x 21 + 40` is about **two minutes per task**, and ten tasks that run
at once is well under a GPU-hour. This is the cheapest stage in the extension;
the 1,080-element screen it selects from cost **18.4 GPU-h**.
