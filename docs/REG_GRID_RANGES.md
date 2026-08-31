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
