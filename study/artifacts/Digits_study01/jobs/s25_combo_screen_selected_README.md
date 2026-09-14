# Digits_study01 - P25: joint tuning of the SELECTED pair (**an extension**)

**This is not a stage of the study, and it is not a correction to P23.** Every
stage the paper reports has run, `COMBINATIONS.md` already records what the
combination cross found, and P23's screen and finals stand exactly as they are
and are not re-emitted. This is a second extension, it is named as one in the
task file's header, in its chain manifest and in its selection record, and
`REPRODUCE.md` keeps both out of the core stage table deliberately.

## Why a second joint screen

P23 screened the joint grid of the pair that leads each schedule **by TEST
score** - the owner's documented departure from ranking on validation, and the
order the two shipped winners views carry. That is a defensible pair to tune.
It is not the pair the study selected.

`tables/p12_agg_top3.json` says in its own `rank_by` field that the aggregation
shortlist was cut on **validation**, and on the parallel schedule the two
orderings disagree about which rule comes first:

| schedule | validation head (selected) | test head (P23 tuned this) |
|---|---|---|
| parallel (`concurrent`) | `anchor_h2` | `eta_0p95` |
| cyclic (`sequential`) | `seq_fedavg` | `seq_delta_capped` |

So P23's finding - that moving the two halves together finds nothing the point
of independent bests did not already have - was measured on a rule the
programme did not choose. This screens the joint grid of the pair it did.

## The pair

| schedule | selected rule | penalty |
|---|---|---|
| parallel (`concurrent`) | `anchor_h2` - the server anchor at half-life h = 2R | `hybrid_seq_mix0p75` - the KD+EWC blend |
| cyclic (`sequential`) | `seq_fedavg` - cyclic FedAvg, parameter-free | `hybrid_seq_mix0p75` - the same blend |

The rule is read from the head of `p12_agg_top3.json`'s own ranking, on the basis
that record names, rather than typed here: a shortlist re-cut on the other basis
would move this whole grid without anything failing. **The penalty half is
P23's, unchanged**, and is read out of `make_digits_p23.THE_PAIR` rather than
re-derived - this grid moves the rule half and nothing else, so the two joint
grids differ in exactly one place and can be read against each other, which is
the only reason to run the second one.

(The validation shortlist's own *penalty* leaders are different arms again -
`logit_l2_lam0p001` on the parallel schedule and `ntd_b0p01_t2` on the cyclic
one, neither of them a KD+EWC blend. Tuning one of those jointly would need a
different mapping onto the trainer's coefficients and would be a different grid,
not this one widened. It is not run and it is not claimed.)

**Every non-grid flag is copied from the shipped `s20` lines that pair these two
halves** - `anchor_h2_hybrid_seq_mix0p5` and
`seq_fedavg_hybrid_mix0p5` - so a line here differs from one there in
the horizon and in the four (or five) coefficients, and in nothing else: same
cohort, same books, same rate, same `--outer-workers 1 --inner-workers 1`, same
`--extended-aggregations`, and the `fisher_path` convention
`s19_hybrid_seq.txt` established.

## The grid

    c_ewc  in {0.05, 0.1, 0.3}      EWC coefficient
    c_kd   in {0.05, 0.11, 0.2}      KD coefficient
    T      in {0.25, 2}         KD temperature
    m      in {0.25, 0.5, 0.75}      blend weight
    h      in {1R, 2R, 4R}       PARALLEL ONLY

    3 x 3 x 2 x 3 = 54 penalty settings
      x 3 anchor half-lives = 162 parallel cells
      x 1 (no rule knob)      = 54 cyclic cells
      = 216 cells x 5 folds = 1080 tasks at 25 rounds

The four penalty rows are **P23's four penalty rows**, at the same values, for
the reason above; `REG_GRID_RANGES.md` section 9 records where each of them came
from and that reading is not repeated here. What is new is the rule axis.

### The anchor row, and where it leaves the programme

`anchor_h2` is one setting of a coefficient row, so the parallel pair's rule
half *has* a knob and this grid moves it. The row is the selected value with one
neighbour each side - and the neighbour above is **outside anything the study
screened**, which is a fact about the selection and is stated rather than
smoothed over.

`agg_cells.ANCHOR_HALFLIVES_R` is
{2R, 1R, 0.5R, 0.25R, 0.125R} and stops at 2R deliberately:
the coefficient is `lambda_s = 1 - 2 ** (-1/h)` in rounds, so at h = 2R the
displacement from g-0 never halves inside the run at all, and nothing slower
than that is an intervention. **The rule the study selected therefore sat on the
top edge of its own row.** A bracket around it has no neighbour above inside the
screened range, so 4R steps outside it, and 4R is a *weaker* intervention than
every value the screen tried rather than a stronger one. The row is:

| h | `--server-anchor` at 25 rounds | at 100 rounds | inside the screened row? |
|---|---|---|---|
| 1R | 0.0273451 | 0.0069075 | yes |
| 2R | 0.0137673 | 0.00345974 | yes |
| 4R | 0.0069075 | 0.00173137 | **no** |

A cell stores `h` and not the coefficient, and
`study_lines.resolve_agg_flags` turns it into `--server-anchor` at the horizon
the line runs - which is why one setting means the same intervention on this
screen and in the finals its winner is re-run in. A cell that stored the
coefficient would be two different rules at the two horizons, which is the
failure that row of `resolve_agg_flags` exists to document.

`seq_fedavg` has no coefficient at all, which is why the cyclic half of this
grid is a third the size - the same asymmetry P23 had, arriving from a different
rule.

## How the penalty dials reach the trainer

`AnchoredTrainer` computes

    lam * ( mix * KD + (1 - mix) * Fisher )

and the penalty this grid names is

    m * c_kd * KD + (1 - m) * c_ewc * Fisher

Matching term by term gives `lam * mix = m * c_kd` and
`lam * (1 - mix) = (1 - m) * c_ewc`, hence

    lam = m * c_kd + (1 - m) * c_ewc
    mix = m * c_kd / lam

which is :func:`reg_cells.combo_tune_lam_mix`, the same function P23's grid
resolves through and not a second copy of it.

**The trainer's `mix` is not the owner's `m`.** It is the KD half's share of the
total penalty weight, and it moves with `c_ewc` and `c_kd` as well as with `m` -
the two coincide only where the two coefficients are equal, which on this grid
is `c_ewc = c_kd = 0.05` alone. A grid that wrote `m` straight into `mix` would
be sweeping a different object than the one it named, and nothing in the emitted
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

(`T` and `h` pass through untouched; the table is the same for both temperatures
and, on the parallel side, for all three half-lives.)

## Deduplication

**0 cell(s) removed.** Two dial settings collide when they hand the
trainer the same `(lam, mix, T)` under the same rule - two folders, two seeds and
one experiment, with nothing in the run records to say so. The check is on the
coefficients rather than on the dials, because the dials are what differ. It
finds nothing here, for P23's reason: a collision needs `m * c_kd` and
`(1 - m) * c_ewc` to repeat *together*, and the nine `(m, c_kd)` products are
all distinct. The check is emitted anyway, so that a widened row cannot quietly
pay twice.

## A ranking horizon

25 rounds, not 100. Nothing here is a reported number; a 25-round result is a
ranking signal. The winners are re-run at 100 rounds by
`s26_combo_full_selected.txt`, which `study_emit.py combo-tune-full-selected`
emits under the same rule every other selection in this study uses.

## The Fisher - no export needed

Every cell reads the Fisher diagonal of the shipped model. The study's own
`g0_selection.json` names **fold 1**, and this stage reads that number
rather than restating it. The lines name the directory as
`$FOA_STUDY_DIR/g0_fold$G0_FOLD/global_results/fisher`, which
`slurm/study_phase.sbatch` derives from the study's own `g0_selection.json`. So
no export is required; a task that needs the fold when the record is missing is
refused with exit 78 rather than run with an empty path component.

## Seeds

    700000 + 70000 + cell_index * 10 + fold

= 770001-772155, disjoint from every range this study has drawn and from
both of P23's: 216 cells span 2150, more than the thousand-wide block
the scheme assumes, so the window is opened on width rather than on the next
free thousand - P21's lesson, and P23's. The highest seed any shipped file draws
is 764015, so this block sits clear of the programme's and of the first
extension's rather than interleaved with either.

## Submit

```bash
sbatch --account=$FOA_ACCOUNT --partition=<gpu partition> --gres=gpu:1 \
       --array=1-1080%20 \
       --export=ALL,FOA_PROJECT_DIR=$FOA_PROJECT_DIR,FOA_STUDY_DIR=$FOA_STUDY_DIR \
       slurm/study_phase.sbatch study/artifacts/Digits_study01/jobs/s25_combo_screen_selected.txt
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
96 minutes of wall clock at `%20`. P23's screen of the same
shape booked that and measured 18.4 GPU-h, which is nearer the **14 GPU-h** the
measured per-task cost of a one-family 25-round task gives. Both are quoted
because a stage this size should be authorised against the pessimistic one.
