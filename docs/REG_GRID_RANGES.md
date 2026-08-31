# Regularisation grid ranges: what each knob does, and where its values sit

**Working notes, not paper material.** This is the record of a design discussion
about where each penalty's range belongs. The search that produced it is not a
result and is not reported; only the ranges it settles are, and the findings at
the end.

Companion to `GRID_RANGES.md`, which does the same for the nine aggregation
knobs. Same method: state the mechanism, put the old range next to the new one,
check what the literature uses, then decide.

## The selection rule these ranges are searched under

One formula, both families, unchanged from the aggregation side:

    score = (adaptation - A0) - w * (P0 - preservation)

with `A0 = 0.8225` (g-0's own accuracy on the cohort - what doing nothing gets)
and `P0 = 0.9986` (g-0's own accuracy on the source data - what is at risk),
both read from disk rather than written down. **`w = 1`**, as the aggregation
grid was run and selected under. A single weight across both families is what
makes the two tables readable against each other.

---

## Where the old numbers come from

Every "old range" below was read out of the previous study's own code and
results, not from memory:

| method | old grid, as declared | as executed |
|---|---|---|
| `param_l2` | `lambda_prox in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]` | ran |
| `fisher` | `ewc_lambda in [0.2, 1, 4, 8, 10, 20, 100, 200]` | ran |
| `fisher_scaled` | same trainer, one fixed value (8.0) | ran |
| `logit_l2` | `lambda_consis in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]` | ran |
| `feature_l2` | `beta in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]` | **never ran** |
| `kd` | `T in [1,2,4,8,10,50] x alpha in [0.1,0.5,0.95]` | **only alpha=0.95** |
| `ntd` | `beta in [0.3,1,3] x tau in [1,3]` | **never ran** |

Three of the seven were never actually measured, and a fourth was measured on
one axis only. The `BEST_*` constants for those four - `BEST_FEATURE_BETA`,
`BEST_NTD_BETA`, `BEST_NTD_TAU`, `BEST_KD_ALPHA` - were therefore not selected
by anything. That matters beyond bookkeeping: `ntd` is the method currently
leading this study.

---

## 1. `param_l2` - FedProx's proximal term

    loss = CE + mu * 0.5 * ||theta - theta_g||^2

A constant-force spring to the frozen g-0. No decay, no adaptivity; the pull
grows in proportion to how far the client has already travelled.

**Convention.** The published implementation is not FedProx's quantity. It sums
the *per-tensor mean* squared error, so each tensor is divided by its own
element count. Matching gradients tensor by tensor gives `mu = 2 * lambda / n`,
and `n` varies enormously across this model:

| tensor | n | equivalent mu at lambda = 0.9 |
|---|---|---|
| `conv_0.b` | 32 | 5.6e-2 |
| `fc1.b` | 512 | 3.5e-3 |
| `conv_0.w` | 800 | 2.3e-3 |
| `fc2.w` | 31,744 | 5.7e-5 |
| `conv_1.w` | 51,200 | 3.5e-5 |
| `fc1.w` | 1,605,632 | 1.1e-6 |

So the old `BEST_PROX_LAMBDA = 0.9` applied an effective `mu` spanning **4.7
orders of magnitude** - clamping the biases hard and leaving the 1.6M-parameter
FC weight matrix essentially free. It was not one penalty strength; it was a
size-weighted penalty nobody chose. Our cells run `--fedprox-convention`, so our
`mu` is FedProx's `mu`.

**Range: unchanged.**

    mu in {0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0}          7 cells

FedProx (Li et al., MLSys 2020) tunes `mu` over `{0.001, 0.01, 0.1, 1}` and
reports different winners across that whole span. This row contains that grid,
one decade below, one above, and the exact zero control - `lam <= 0` returns
zero penalty, so `mu = 0` is an exact no-op rather than a near-miss of one.

Not edge-limited, and adding cells buys nothing.

## 2. `fisher` - EWC

    loss = CE + lambda * 0.5 * sum_i F_i * (theta_i - theta_g,i)^2

The same spring, with per-weight stiffness from g-0's Fisher diagonal: weights
the shipped model relied on are held, weights it barely used are free. It spends
restraint selectively rather than uniformly.

**Three parameters, one swept.** `F` and `theta_g` are fixed, and should be
stated as design choices rather than left implicit:

* **`F`** - the Fisher diagonal of g-0 on g-0's own data, **normalised to mean
  1** (`normalise_fisher=True`). The normalisation is what makes `lambda` a
  comparable coefficient instead of a number absorbing the Fisher's arbitrary
  scale, and it is why this row does not need the eight decades an
  un-normalised penalty would.
* **`theta_g`** - the frozen g-0. The trainer supports a moving anchor; frozen
  and moving are not two tunings but two questions, and only frozen measures
  distance from the shipped model.

**Old range:** `[0.2, 1, 4, 8, 10, 20, 100, 200]`, swept.

**New range:**

    lambda in {0.1, 1, 2, 3, 5, 8, 10, 20, 50, 100, 200, 500, 1000}   13 cells

Covers the old span and resolves it: 1-10 filled in where the old grid stepped
1 -> 4 -> 8, and 8.0 present explicitly because it is that study's reported
setting. EWC's own published values sit in the hundreds to thousands, so the row
runs to 1000. If a winner lands on 1000 that is a genuine edge and the row gets
extended; the point of placing it here is that the edge would then mean
something.

## 3. `fisher_scaled` - EWC under the dynamic cap

    raw     = sum_i F_i (theta_i - anchor_i)^2
    coeff   = min( CE / raw , lambda )
    penalty = (coeff / 2) * raw

Substituting `coeff = CE/raw` makes the penalty exactly `CE/2`, so the rule is:
**the penalty may never exceed half the cross-entropy.**

Near the anchor `raw` is tiny, `CE/raw` is huge, the `min` picks `lambda`, and
the method is plain `fisher`. As the client drifts `raw` grows, `CE/raw` falls,
and once it drops below `lambda` the penalty saturates at `CE/2` and stays
there. It is a safety valve against the regulariser swamping the task loss.

**Consequence: above the saturation point every `lambda` is the same run.** The
row will contain duplicates at the top, and where they start is something the
data tells us rather than something we choose. The old row - eight decades -
was almost certainly four or five duplicate cells at its top end.

**Old range:** never swept; fixed at `8.0`. This is the method that value
actually belongs to, and `8` there is a ceiling on a ratio, not a spring
constant - not comparable to a plain EWC lambda.

**New range: the same 13 values as `fisher`.** Deliberately matched. This method
exists as an ablation of the cap; if the two rows swept different lambdas, a
difference between them could not be attributed to the cap rather than to the
coefficient. A test asserts the two rows are identical.

## 4. `logit_l2`

    penalty = lam * MSE( student_logits , anchor_logits )

Constrains the model's outputs, not its weights: it may move anywhere in
parameter space provided it keeps saying the same thing.

**One knob, not two.** `LOGIT_L2_T = 2.0` is recorded with every cell and the
space never reads it - `_output_penalty` is a plain MSE. Sweeping it would be
five identical runs per lambda. Recorded for provenance; it should be stated in
the paper rather than left in a table looking like a swept parameter.

**Old range:** `[1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]`, swept, with these results
(concurrent-delta, adaptation / preservation):

| lambda | adapt | preserve |
|---|---|---|
| 0.001 | 0.9497 | 0.9911 |
| 0.01 | 0.9284 | 0.9942 |
| 0.1 | 0.9091 | 0.9950 |
| 0.5 | 0.8511 | 0.9958 |
| 0.9 | 0.8530 | 0.9957 |
| 1.0 | 0.8395 | 0.9958 |

Monotone, and decisive about where the live region is: by `lambda = 0.5`
adaptation has collapsed to 0.85, barely above the 0.8225 of doing nothing,
while preservation has flattened. **Everything at and above 1.0 is a frozen
model.** Their own winner sat on their bottom edge.

**Our previous range was `{0.1, 0.316, 1, 3.16, 10}` - it overlapped theirs in
two cells and swept a decade *above* 1.0 where they had swept three decades
below it.** Four of five cells were in the dead region and the live region was
not sampled at all. Logits on 62 classes run to order 10, so squared differences
are order 1-100 against a CE of about 0.6; the scale argument and their data
agree.

**New range:**

    lam in {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}          8 cells

Covers their full range, adds a step below their bottom edge, keeps `0.1` (their
reported setting) for comparability, stops at 1.0 where adaptation is gone.

## 5. `feature_l2`

    penalty = lam * MSE( h_student(x) , h_anchor(x) )

`h` is the 512-d post-`fc1` ReLU representation. One layer earlier than
`logit_l2`: the internal representation must stay put, the head above it is free.

**Two corrections to the record belong here.**

The old `aligned_feature` grid is **not** this method. There were two trainers:

    CFAlignedFeatureTrainer  -> matches pre-softmax logits  -> our logit_l2
    FeatureAlignmentTrainer  -> matches penultimate features -> our feature_l2

So `aligned_feature_grid_search` measured logit matching. The old work had *two*
grids measuring the same space under different method names.

And the representation grid was **declared and never run** - `feature_alignment.py`
defines `beta in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]` and there is no
`feature_alignment_grid_search` directory in those results. `BEST_FEATURE_BETA`
was never selected by a sweep.

**Scale:** ReLU activations are non-negative and order 1, so squared differences
are far smaller than the logit case and the useful region sits *higher*, not
lower - the opposite direction from method 4.

**New range:**

    lam in {0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100}                 9 cells

Contains the declared-but-unrun old grid so the comparison is possible. The
previous row was five decade steps and was described as deliberately "short and
wide because the evidence is thin" - that was the right instinct when we thought
prior numbers existed. Knowing there are none, thin evidence argues for
measuring the knob properly, not for sampling it sparsely.

## 6. `kd` - Hinton distillation

    ours:       CE + lam * T^2 * KL( p_anchor^T || p_student^T )
    literature: alpha * CE + (1 - alpha) * T^2 * KL
    same minimiser when   lam = (1 - alpha) / alpha

**`T` and `alpha` multiply.** The effective weight is `lam * T^2`:

| alpha | lam | at T=1 | at T=8 | at T=50 |
|---|---|---|---|---|
| 0.10 | 9.00 | 9.0 | 576 | 22500 |
| 0.50 | 1.00 | 1.0 | 64 | 2500 |
| 0.95 | 0.053 | 0.05 | 3.4 | 132 |
| 0.99 | 0.010 | 0.01 | 0.65 | 25 |

`alpha = 0.1, T = 1` and `alpha = 0.99, T = 8` are within a factor of 1.5. This
is the same coupling as FedAdam's `lr` and `tau` on the aggregation side: a
square grid over two knobs whose product is most of what matters. `T` does have
a second, independent effect - it softens the teacher's distribution - so the
axes are not fully redundant, but much of the grid moves along lines of constant
penalty.

**Old range:** declared `T in [1,2,4,8,10,50] x alpha in [0.1,0.5,0.95]` = 18
cells; **executed 6** - every family swept T at `alpha = 0.95` only, with one
stray `alpha = 0.5` cell. `alpha` was never swept, so `BEST_KD_ALPHA = 0.95` is
the only value that ran, not a selected one.

Their executed T row (alpha = 0.95, concurrent-delta):

| T | adapt | preserve |
|---|---|---|
| 1 | 0.9342 | 0.9885 |
| 2 | 0.9168 | 0.9902 |
| 4 | 0.9246 | 0.9939 |
| 8 | 0.9052 | 0.9946 |
| 10 | 0.9188 | 0.9945 |
| 50 | 0.9052 | 0.9949 |

Preservation saturates by `T = 8`: `T = 50` buys 0.0003 over it and costs
adaptation. The top of the T row is dead.

**New range:**

    T     in {0.25, 0.5, 1, 2, 4, 8}                                  6
    alpha in {0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99}                    7   = 42

Drops 16, 32, 50 - measured saturated in both studies. Reaches 0.25 because the
first screen's winner sat on the low T edge; what was a boundary extension is
now part of the row. The alpha row is kept full because it has never once been
swept.

## 7. `ntd` - not-true distillation

    penalty = beta * tau^2 * KL( p_anchor^tau || p_student^tau )
              restricted to the non-true classes

`kd` with the true class masked out of the KL. The true class is what the client
is supposed to be learning, so forcing agreement there fights adaptation
directly; masking it means the penalty preserves only the teacher's relative
ranking of everything the sample is *not*. It is the one penalty in the set
whose design separates the two objectives rather than trading them off, and it
is currently the best method in this study: **0.9335 / 0.9898**, 11.1 points of
adaptation for 0.88 of preservation.

`beta` and `tau` couple exactly as `lam` and `T` do in `kd`.

**Old range:** declared `beta in [0.3, 1, 3] x tau in [1, 3]`; **never run.**
The method now leading the study had no prior measurement anywhere.

**New range:**

    beta in {0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30}                     8
    tau  in {0.5, 1, 2, 3, 4, 8}                                      6   = 48

`beta` gains a step at the strong end so that end is not an edge. `tau` gains
8.0 so this row overlaps the kd row - these are the same knob in the same units,
and a temperature worth measuring in one is worth measuring in the other. Given
that kd saturates by T = 8, bringing kd down to ntd's scale is the right
direction rather than pushing ntd up to kd's old 50.

---

## Totals

| | cells | tasks at 5 folds |
|---|---|---|
| superseded screen | 125 | 625 |
| this screen | 140 | 700 |

| method | cells | tasks |
|---|---|---|
| `param_l2` | 7 | 35 |
| `fisher` | 13 | 65 |
| `fisher_scaled` | 13 | 65 |
| `logit_l2` | 8 | 40 |
| `feature_l2` | 9 | 45 |
| `kd` | 42 | 210 |
| `ntd` | 48 | 240 |
| **total** | **140** | **700** |

Larger by 75 tasks, and spent differently: `kd` loses 11 cells from a saturated
top end, while the three rows that had never been measured at all - `fisher`
resolved, `feature_l2`, `ntd` - gain them.

## No boundary extensions

The superseded screen appended four kd cells and four ntd cells after finding
winners on those rows' edges. Those probes are inside the rows now. Nothing is
carried forward: the rows are placed against what the earlier work measured
rather than around a guess, which is what produced the edges in the first place.

If a winner lands on an edge again the remedy is unchanged from the aggregation
side - widen that row, re-emit, re-select - but it starts from a row that was
placed on evidence.

---

## Findings that belong in the paper

### Three of the seven penalties were never measured

`feature_l2` and `ntd` had grids defined and never executed; `kd` declared a
two-axis grid and swept one axis. Their `BEST_*` constants were reported as
selected hyperparameters. The method now leading this study, `ntd`, is among
them.

This is not a criticism to bury in a limitations paragraph - it is the reason
this study exists in its current form, and it is checkable from that work's own
repository: the grid specs are in `grid_search/*.py` and the result directories
they name are absent.

### Named methods that are one method, again

The aggregation side found five of eighteen entries were duplicates or special
cases. The same collapse holds here:

* `CFAlignedFeatureTrainer` and `CFLogitConsistencyTrainer` both match logits -
  two grids, one space.
* `fisher_scaled` above its saturation point is one behaviour however `lambda`
  is set, so the top of that row is a single cell repeated.
* every space returns zero penalty at `lam <= 0`, so each method's zero cell is
  the same run - which is why only `param_l2_mu0` is emitted.

### A search range is a claim, and ours was wrong in a specific direction

Our `logit_l2` row swept a decade *above* the value where the earlier work had
already measured adaptation collapsing, and never sampled the three decades
below it where their winner lived. That is not a coarse grid; it is a grid in
the wrong place, and no amount of running it would have found the optimum.

The fix that generalises: **before setting a range, read what the previous
measurements say about where the knob stops mattering.** Four of the seven rows
here moved after doing that, and two of them moved by more than a decade.

---

# Provenance: where every range came from

Nothing below is a remembered number. Each row names the file it was read out
of, so a reader can check it. Three sources are distinguished, because they
carry different weight:

* **paper** - a grid published by the work that introduced the method.
* **old repo** - the previous study's own code, at the path given. A grid that
  is *declared* there is not necessarily a grid that *ran*; the "executed"
  column says which.
* **measured here** - a placement decided by numbers in this study or in the
  old study's result files, cited to the directory they sit in.

| knob | source | where |
|---|---|---|
| `param_l2` mu | paper | FedProx, Li et al., *Federated Optimization in Heterogeneous Networks*, MLSys 2020 - tunes `mu` over {0.001, 0.01, 0.1, 1} |
| `param_l2` old value | old repo | `constants.py: BEST_PROX_LAMBDA = 0.9`; grid at `grid_search/prox.py:22`, `lambda_prox in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]` |
| `fisher` lambda | paper | EWC, Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017 - reported values in the hundreds to thousands |
| `fisher` old grid | old repo | `grid_search/ewc.py:24`, `ewc_lambda in [0.2, 1, 4, 8, 10, 20, 100, 200]` |
| `fisher_scaled` old value | old repo | `constants.py: BEST_EWC_LAMBDA = 8.0`; the cap is `EWCTrainer.ewc_loss` |
| `logit_l2` lam | measured here | old grid `grid_search/logit_consistency.py:23`, `[1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]`; **placement** from its own results in `results/logit_consistency_grid_search/*/accuracies_points_100.csv`, which show adaptation collapsed to 0.851 by `lam = 0.5` |
| `feature_l2` lam | old repo (declared only) | `grid_search/feature_alignment.py:28`, `beta in [1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]`. **No `results/feature_alignment_grid_search` directory exists** - never executed |
| `kd` T, alpha | paper + measured here | Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network*, 2015, for the `alpha`/`T` form; old grid `grid_search/distillation.py:24`, `T in [1,2,4,8,10,50] x alpha in [0.1,0.5,0.95]`. **Top of the T row cut** on that grid's own results in `results/distillation_grid_search/*/accuracies_points_100.csv`: preservation saturates by `T = 8`, and `T = 50` buys 0.0003 over it while costing adaptation |
| `ntd` beta, tau | paper + old repo (declared only) | FedNTD, Lee et al., *Preservation of the Global Knowledge by Not-True Distillation in Federated Learning*, NeurIPS 2022; old grid `grid_search/ntd.py:26`, `beta in [0.3,1,3] x tau in [1,3]`. **No `results/ntd_grid_search` directory exists** - never executed |

**What the executed column showed.** Of the seven penalties, `param_l2`,
`fisher` and `logit_l2` were genuinely swept; `kd` declared a two-axis grid and
swept one axis (every executed cell is `alpha = 0.95` bar a single stray
`alpha = 0.5` in `concurrent_weights`); `feature_l2` and `ntd` were never run at
all. So `BEST_FEATURE_BETA`, `BEST_NTD_BETA`, `BEST_NTD_TAU` and `BEST_KD_ALPHA`
were reported as selected hyperparameters without a selection behind them.

For completeness, the aggregation-side citations, which `GRID_RANGES.md` uses:
FedAvg (McMahan et al., AISTATS 2017); adaptive server optimisers FedAdam and
FedYogi with their `lr x tau` grid (Reddi et al., *Adaptive Federated
Optimization*, ICLR 2021); server momentum FedAvgM (Hsu, Qi & Brown, 2019);
coordinate-wise trimmed mean and median (Yin et al., ICML 2018).

---

# The selection rule

## The formula

    score(cell) = (adaptation - A_0)  -  w * (P_0 - preservation)
                   \_____________/         \________________/
                    what it GAINED           what it SPENT

One formula, both grids, both schedules, `w = 1`.

`A_0` and `P_0` are **the shipped model's own two accuracies**, measured rather
than assumed, and read off disk at selection time by `shipped_baselines()` in
`tools/study_emit.py`:

    A_0 = 0.8225   g-0 on the cohort's test rows          <- g0_perfold_evaluations.json
    P_0 = 0.9986   g-0 on the source population's rows    <- g0_evaluations.json

So the two terms are **differences from doing nothing**, not raw accuracies:

* `adaptation - A_0` is what adapting bought on the new writers. A cell that
  changes nothing scores 0 here.
* `P_0 - preservation` is what it cost on the writers the model already served.
  A cell that forgets nothing scores 0 here.

A configuration is worth choosing when the first exceeds the second.

## Why not rank on adaptation alone

Because it does not measure the thing the study is about. Ranking on adaptation
picks, for every method, the setting that constrains least - the weakest anchor,
the largest server step, `mu = 0` - so every "winner" is the cell nearest to
plain FedAvg and the resulting table appears to show that no method preserves
anything. That conclusion would be an artefact of the ranking, not a property of
the methods.

## The weight `w`, honestly

`w` is a choice and it was made deliberately at 1, matching the aggregation
grid. It is **not** true that the choice never matters - a claim to that effect
sat in `trade_score`'s docstring and has been corrected. On the aggregation
screen, **5 of 17 methods change winner between `w = 1` and `w = 3`**
(`eta`, `fedadam`, `fedavgm`, `fedyogi`, `seq_mix`), all toward gentler
settings; `anchor`, `trimmed`, `weight_q` and the whole sequential family do
not. Four of the five that move are near break-even and flip back at `w = 2`;
only `fedadam` moves decisively, trading 4.76 points of adaptation for 1.73 of
preservation.

The honest way to report this is a sensitivity row, not a defended constant.

## Two horizons, two jobs

    screen  25 rounds   RANKS cells within a method.   Nothing here is reported.
    finals 100 rounds   REPORTS the winners, 5 folds.

The screen exists because the full grid at the reporting horizon is unaffordable.
A 25-round result is a ranking signal, and this study now has direct evidence of
the difference between the two jobs - see the finals below.

---

# What is recorded beyond the two headline numbers

## The two headline quantities, defined

Read by `select_reg_screen.read_family_runs`, per (cell, fold, family):

* **adaptation** - the last value of `pool_val_accuracies`: the cohort's
  held-out **validation** half. Validation, not test, is the protocol letter;
  the test column is carried alongside and both orderings are recorded, so a
  selection can be re-read either way.
* **preservation** - `final_evaluation.old.mean`: the source population's five
  fold test partitions, scored separately, with their mean and spread.

**Preservation is `source_val_accuracies`, and it FALLS.** `pool_test_accuracies`
tracks the cohort and RISES through a run; reading it as preservation reports a
run that forgot seven points when it forgot one, and does so silently. That
substitution happened once in this project and is now refused at read time by
`_check_preservation_series` in `tools/report_tables.py`, which rejects any
preservation series climbing more than five points across a run.

## The eight forgetting signals

The study's own constraint is that **a running experiment may not read the
source population**: once the model is shipped, those writers are gone. Every
number reported on the source split is therefore an *evaluation* quantity - fine
for the paper's tables, forbidden as an input to a stopping or selection rule.

So every run also records, every round, the eight quantities a server *is*
allowed to compute (`runners/forgetting_signals.py`):

| signal | what it needs | direction |
|---|---|---|
| `dist_l2_to_global` | nothing but the weights | grows with forgetting |
| `dist_fisher_to_global` | the shipped Fisher the server already owns | grows |
| `dist_fisher_norm_to_global` | the same, unit-mass normalised | grows |
| `retention_known` | the clients' own held-out half | shrinks |
| `agreement_with_global` | the same | shrinks |
| `kl_global_to_current` | the same | grows |
| `proxy_acc` | public data the server owns (MNIST) | shrinks |
| `proxy_kl` | the same | grows |

`retention_known` is the pointed one: accuracy of the current model restricted
to the samples the *shipped* model got right - "how much of what it already knew
is left" - computed only from data the clients are allowed to be asked about.

The cost is one extra forward pass per round over the pool's validation halves
and one over the proxy set, the same order as the evaluation passes already
made. g-0's predictions on both reference sets are computed once per run and
cached.

**Verified populated in this programme: 8 of 8 signals, on both horizons** -
25-point series across the screen, 100-point across the finals. This matters
because in an earlier run four of the eight were silently empty: the MNIST proxy
set had never been built and the winning fold's Fisher was never promoted to the
name the artefact resolver looks for. Both are fixed, and the fold selector now
promotes the Fisher as part of choosing the fold.

`analysis/forgetting_signals.py` then asks the three questions that decide
whether a signal can stand in for the forbidden source split: does it correlate
with true forgetting; would stopping on it have worked, against oracle stopping
and against no stopping at all; and would selecting on it have worked, with the
gap to what an oracle would have picked.

---

# Results

## The screen - 140 cells, 700 tasks, 25 rounds

All 700 completed, no failures, including the strongest corner in the grid
(`ntd_b30_t8`, effective weight `30 x 8^2 = 1920`, never previously run - it did
not diverge, it froze: adaptation 0.8081, *below* the 0.8225 of doing nothing).

Winners, `w = 1`:

| method | concurrent | | | sequential | | |
|---|---|---|---|---|---|---|
| | cell | adapt/preserve | score | cell | adapt/preserve | score |
| `ntd` | `b0.01_t0.5` | 0.9127 / 0.9934 | **8.50** | `b0.01_t1` | 0.9147 / 0.9942 | **8.78** |
| `logit_l2` | `lam0.003` | 0.9080 / 0.9941 | 8.11 | `lam0.003` | 0.9118 / 0.9939 | 8.46 |
| `kd` | `T8_a0.99` | 0.9062 / 0.9952 | 8.03 | `T0.5_a0.99` | 0.9137 / 0.9911 | 8.37 |
| `param_l2` | `mu0` *(control)* | 0.9100 / 0.9898 | 7.88 | `mu0` *(control)* | 0.9109 / 0.9900 | 7.99 |
| `feature_l2` | `lam0.1` | 0.9053 / 0.9940 | 7.82 | `lam0.01` | 0.9045 / 0.9920 | 7.54 |
| `fisher_scaled` | `lam500` | 0.8820 / 0.9962 | 5.71 | `lam0.1` | 0.8829 / 0.9959 | 5.78 |
| `fisher` | `lam0.1` | 0.8811 / 0.9962 | 5.63 | `lam0.1` | 0.8838 / 0.9954 | 5.81 |

Two things to read off it. **Six of the seven winners sit at the weak end of
their row**, and `param_l2`'s winner is `mu = 0` - literally no penalty. And
**only three penalties beat that no-penalty control at all**, two of them by
under a quarter point.

The boundary rule reports nine edge hits, eight at the weak end:
`param_l2_mu0` (both families), `fisher_lam0.1` (both), `fisher_scaled_lam0.1`,
the `alpha = 0.99` corner of `kd` (both), `ntd t = 0.5`; the one exception is
`kd_T8`, at the top of the trimmed T row.

## The finals - 14 winners, 70 tasks, 100 rounds, 5 folds

| | concurrent | | | sequential | | |
|---|---|---|---|---|---|---|
| method | cell | adapt/preserve | score | cell | adapt/preserve | score |
| `ntd` | `b0.01_t0.5` | 0.9223 / 0.9845 | 8.57 | `b0.01_t1` | 0.9251 / 0.9878 | **9.18** |
| `logit_l2` | `lam0.003` | 0.9213 / 0.9903 | **9.05** | `lam0.003` | 0.9204 / 0.9903 | 8.96 |
| `kd` | `T8_a0.99` | 0.9166 / 0.9894 | 8.49 | `T0.5_a0.99` | 0.9212 / 0.9748 | 7.50 |
| `fisher_scaled` | `lam500` | 0.9119 / 0.9927 | 8.34 | `lam0.1` | 0.9063 / 0.9924 | 7.76 |
| `fisher` | `lam0.1` | 0.9054 / 0.9915 | 7.57 | `lam0.1` | 0.9138 / 0.9914 | 8.40 |
| `feature_l2` | `lam0.1` | 0.9101 / 0.9865 | 7.56 | `lam0.01` | 0.9184 / 0.9819 | 7.92 |
| `param_l2` | `mu0` *(control)* | 0.9090 / 0.9706 | 5.85 | `mu0` *(control)* | 0.9128 / 0.9686 | 6.03 |

## The finding: the screen ranks, but it cannot price

The no-penalty control is **4th of 7 at 25 rounds and last of 7 at 100**, in
both families, by 2.7 and 3.2 points. Its preservation column says why:

    forgetting with no penalty at all     25 rounds   ~0.9   points
                                         100 rounds    2.80 / 3.00 points

At the screening horizon there is about one point of forgetting available to
protect, so any penalty strong enough to bite reads as pure cost, and selection
walks to the weak end of every row. At the reporting horizon there are three,
and **every penalty pays for itself**.

This explains the earlier study's regularisation winners - 100x to 80,000x
weaker than the values it published - without having to call them a mistake.
They were selected correctly, under a rule applied at a horizon that cannot see
the quantity being traded. The eight weak-end boundary hits are the same effect:
they are not ranges placed too narrowly, they are rows being asked to reach a
setting that does not exist, because the honest answer at 25 rounds is "no
penalty".

## What is stable, and what is not

**Stable.** `ntd` and `logit_l2` are the top two at both horizons and in both
families, and the ranking `ntd > logit_l2 > kd` holds across all four screen
tables. `logit_l2_lam0.003` wins its row in both families at both horizons - and
it is a cell that **did not exist in the previous range**, sitting three decades
below where the earlier grid looked.

**Not stable.** `kd` is 3rd concurrent and 6th sequential at full horizon; its
sequential winner `T0.5_a0.99` gives up 2.4 points of preservation. The
low-temperature corner does not survive the cyclic schedule.

**The structural result.** The three penalties that beat the control at the
screen are exactly the three **output-space** ones - they constrain what the
model says. The two **weight-space** ones, `param_l2` and `fisher`, are the two
that lose to it, and `feature_l2`, which constrains the representation between
those two levels, lands between them. That ordering holds on both schedules.

## Cost

| stage | tasks | rounds | GPU-hours |
|---|---|---|---|
| screen | 700 | 25 | ~25 |
| finals | 70 | 100 | ~8 |
