# Digits_study01 - P26: the SELECTED pair, jointly tuned, at the full horizon (**an extension**)

**This is not a stage of the study.** It is the finals of the extension
`s25_combo_screen_selected.txt` screened, and nothing the paper reports reads
what either produces: the shortlists, the combination cross, the crowning, the
carry settings and the paper views are unchanged and are not re-emitted. Its
record, `tables/p25_combo_tune_selected_winners.json`, says the same thing in
its own text.

**It does not supersede `s24_combo_full.txt` either.** That stage finalised the
joint grid of the pair each schedule leads with by TEST score; this one
finalises the grid of the pair the study SELECTED - `tables/p12_agg_top3.json`
was cut on VALIDATION and says so in its own `rank_by`, and on the parallel
schedule its test order leads with a different rule. Two selections over two
catalogues, two records, two seed blocks and two sets of folders, read beside
each other.

E3 screened that grid - 216 cells x 5 folds, 25 rounds, 1,080 elements, all
COMPLETED, **18.6 GPU-h** at 62 s per element. This stage is what that screen is
for: each schedule's best cell, re-run at the full 100-round horizon on the same
five folds, so the tuned arm can be read beside the eighteen combinations of
`s20_combos4.txt` and beside P24's two arms on the basis they were all measured
on.

## The rule, unchanged

`study_emit.py combo-tune-full-selected` applies **reg-full's** selection to
this grid's own catalogue and to nothing else - the same function P24 runs, over
a different catalogue and a different stem. Same score, same weight, same basis,
same tie-break, same per-schedule selection:

    score = (adaptation - A_0) - (P_0 - preservation),   w = 1

on the **validation** columns of the 25-round screen, fold-mean over all five
folds, argmax within each schedule. The shipped model's own accuracies are read
from the study rather than assumed: `A_0 = 0.8225` on the cohort, `P_0 = 0.9986`
on the source population. A different rule here would make the arm it crowns
incomparable with every other winner in the study - including P24's, which is
the arm this one most needs to be read against.

A schedule is ranked only against its own cells. A cell names one server rule,
which lives in one family, so the parallel half is 162 cells with an anchor
half-life to move and the cyclic half is 54 with no rule knob at all.

## What was selected

| schedule | cell | `c_ewc` | `c_kd` | `T` | `m` | `h` | `lam` | trainer `mix` | adaptation | preservation | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| concurrent | `ctunesel_ewc0p1_kd0p05_T2_mix0p75_h4` | 0.1 | 0.05 | 2 | 0.75 | 4R | 0.0625 | 0.6 | 0.8921 | 0.9960 | +0.0670 |
| sequential | `ctunesel_ewc0p05_kd0p11_T0p25_mix0p75` | 0.05 | 0.11 | 0.25 | 0.75 | - | 0.095 | 0.868421 | 0.8903 | 0.9959 | +0.0652 |

162 cells were scored in the parallel schedule and 54 in the cyclic one; none
fell back to a test column and every cell carried all five folds. **The two
schedules did not choose the same dials**, which is the opposite of what P23's
grid did - there both halves landed on one setting of all four penalty dials.
The only dial they agree on here is `m = 0.75`.

The next four in each, on the same rule:

| schedule | runner-up | adaptation | preservation | score |
|---|---|---|---|---|
| concurrent | `ctunesel_ewc0p05_kd0p2_T0p25_mix0p75_h4` | 0.8865 | 0.9958 | +0.0613 |
| concurrent | `ctunesel_ewc0p05_kd0p05_T2_mix0p5_h2` | 0.8859 | 0.9964 | +0.0612 |
| concurrent | `ctunesel_ewc0p1_kd0p05_T2_mix0p75_h2` | 0.8856 | 0.9961 | +0.0607 |
| concurrent | `ctunesel_ewc0p05_kd0p05_T2_mix0p25_h4` | 0.8857 | 0.9958 | +0.0604 |
| sequential | `ctunesel_ewc0p05_kd0p05_T2_mix0p75` | 0.8876 | 0.9964 | +0.0629 |
| sequential | `ctunesel_ewc0p1_kd0p2_T0p25_mix0p5` | 0.8875 | 0.9956 | +0.0620 |
| sequential | `ctunesel_ewc0p05_kd0p11_T2_mix0p75` | 0.8866 | 0.9964 | +0.0620 |
| sequential | `ctunesel_ewc0p1_kd0p05_T2_mix0p5` | 0.8857 | 0.9962 | +0.0608 |

The worst cell in either schedule still scores +0.036, so this grid separates
the arm from nothing: every point of it is a working configuration, and the
spread across 216 cells is about three points of score on the parallel side and
two on the cyclic one. That is the same shape P23's grid had.

The selection record is `tables/p25_combo_tune_selected_winners.json` in the
study root, and a copy ships beside P13's, P21's and P23's under
`study/artifacts/Digits_study01/tables/`.

## The parallel winner is the half-life the programme never screened

Seven boundary hits across the two winners, reported by the emitter and appended
to `tables/BOUNDARY_HITS.txt`. Not a reason to stop, and the first thing to read:

    ctunesel_..._h4:  c_kd = 0.05  LOW  end of {0.05, 0.11, 0.2}
    ctunesel_..._h4:  T    = 2     HIGH end of {0.25, 2}
    ctunesel_..._h4:  m    = 0.75  HIGH end of {0.25, 0.5, 0.75}
    ctunesel_..._h4:  h    = 4R    HIGH end of {1R, 2R, 4R}
    ctunesel_...:     c_ewc = 0.05 LOW  end of {0.05, 0.1, 0.3}
    ctunesel_...:     T     = 0.25 LOW  end of {0.25, 2}
    ctunesel_...:     m     = 0.75 HIGH end of {0.25, 0.5, 0.75}

`c_ewc = 0.1` on the parallel side is the one dial of the nine that is not at an
end of its row.

**The parallel schedule chose `h = 4R`, and 4R is outside the row the programme
screened.** `agg_cells.ANCHOR_HALFLIVES_R` runs {2R, 1R, 0.5R, 0.25R, 0.125R}
and stops at 2R because at that half-life the displacement from g-0 does not
halve inside the run at all; `s25_combo_screen_selected_README.md` says in
advance that a bracket around a winner on that edge has to step outside the
searched range to have a neighbour above, and that 4R is a *weaker*
intervention than every value the screen tried. The joint screen then chose that
neighbour. Read plainly: **once the penalty beside it moves, the selected server
rule is wanted weaker than anything the aggregation screen offered** - the grid
is asking for the anchor to be turned further down, not up. That is the same
direction P24 found on the other grid, where the parallel winner took the
smallest server step in its row, and it is consistent with the blend's own
screen choosing the floor of the EWC row in both schedules.

**This stage does not settle where the optimum is.** It prices the best of what
was searched, on a grid the owner specified, and reports honestly that the best
of it is at the edge - on the parallel side, at an edge that is already outside
the programme.

## The lines

10 tasks: 2 schedules x 5 folds, at 100 rounds, 8 of 10 clients per round, E=5,
batch 64, the cohort fold book, g-0 as the only initialisation, the frozen g-0
as the anchor, and one server rule per line.

**One line is one schedule.** Each cell names a single aggregation rule - so a
task produces one result, which is `s20_combos4.txt`'s shape and not the
screens'. A `--aggregation fedavg` line would run the cyclic family under a rule
chosen for the parallel one.

**The half-life is resolved at this horizon, not carried from the screen.** A
cell stores `h` as a multiple of the run length and
`study_lines.resolve_agg_flags` emits `lambda_s = 1 - 2 ** (-1/(h * rounds))`,
so the crowned parallel cell is the same intervention here as in the screen that
chose it: `--server-anchor 0.006907504562964073` at 25 rounds becomes
`0.0017313674026074866` at 100. A cell that stored the coefficient would be four
times as fast an anchor in the finals as in the screen.

**The schedule is in the parent as well**, as in `s17_reg_full4.txt`,
`s22_blend_full.txt` and `s24_combo_full.txt`. The cell ids already carry it -
only the parallel rule has a coefficient to name - but a folder that says which
schedule's finals it belongs to cannot be misread by a selector that does not
know that. The stem is `d01_ctuneselfull_`, which is not a prefix match for
`d01_ctunefull_`, so P24's selector cannot collect these runs and this one
cannot collect P24's.

**The rate is the search rate, not the study rate.** 8 of 10, because that is
what the screen these winners come from ran at and what `s20_combos4.txt` was
emitted at. A final emitted at the study's own 9 of 10 would price the tuned arm
on an easier loop than the table it is going to be read in.

## Seeds

    700000 + 74000 + arm_index * 10 + fold   =  774001-774005, 774011-774015

Its own block, above this extension's screen at 770001-772155 and above both of
P23's blocks, which run to 764015 - so the two extensions' four blocks sit
visibly outside the programme's rather than interleaved with it or with each
other. `tools/check_seeds.py` is run over every shipped task file at once, which
is what actually establishes the disjointness: a hand-kept range is what let the
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
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s26_combo_full_selected.txt
```

## Cost

The screen measured 62 s per element at 25 rounds with one schedule and the KD
half's teacher pass. About 40 s of that is interpreter, torch, the two books and
the final evaluation, so the training portion is roughly 22 s and scales with the
horizon: `4 x 22 + 40` is about **two minutes per task**, and ten tasks that run
at once is well under a GPU-hour. This is the cheapest stage in either
extension; the 1,080-element screen it selects from cost **18.6 GPU-h**.
