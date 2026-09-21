# When to stop, without the data that would tell you

*Read 2026-09-09, against `Digits_study01`: 83 hundred-round arms, and the
2,465 payloads the shipped signals pass audited.*

Every stage in this study runs to a fixed hundred-round horizon and reports the
last round. This document asks what that costs, and whether anything the server
is **allowed** to compute could have told it to stop earlier. Two answers come
out of it: the eight forgetting signals recover +0.46p of the oracle's 1.48p
(section 4), and a plateau on the cohort's own accuracy - which is not a proxy
for forgetting and does not have to be - recovers +1.19p (section 5).

---

## 1. The constraint, and why it makes stopping hard

A shipped model is adapted at the clients that received it. The writers it was
originally trained on are not there: their rows sat on a server that has moved
on, and a deployed system has no way to score itself against them. So although
every run in this study *stores* `source_val_accuracies` round by round, no rule
this study proposes may read it. It is an evaluation quantity.

That is the whole difficulty. Forgetting is defined on data the system cannot
see, so a stopping rule has to be built on a proxy. Each run therefore records
the eight signals of `runners/forgetting_signals.py` every round - three
parameter-space distances, three client-side quantities, two proxy-set
quantities - and every one of them can be computed with no source data at all.

    dist_l2_to_global   dist_fisher_to_global   dist_fisher_norm_to_global
    retention_known     agreement_with_global   kl_global_to_current
    proxy_acc           proxy_kl

`tools/check_signals.py --all` counts them: all fifteen prefixes report 8/8 at
their own horizon.

---

## 2. Three columns, and only one of them is deployable

`tools/stopping_table.py` reports each hundred-round arm three times.

| column | what it is | may a deployed system run it? |
|---|---|---|
| **final** | the last round - what every reported table quotes | yes, trivially |
| **oracle** | the round with the best score, chosen by looking at the whole run | **no** - it reads the source population |
| **stopped** | the first round a permitted signal's drift exceeds a budget | yes |

The oracle prices the horizon. It is a bound on what stopping is worth and never
a method: it is labelled oracle everywhere it appears for that reason.

The stopped column comes in two forms, and the difference matters. The **best
rule per arm** is the `(signal, budget)` that arm would have wanted - it bounds
what the eight signals contain, but choosing it needs the same source split the
oracle needs, so it is not deployable either. The **one rule** fixes a single
`(signal, budget)` for every arm of every stage in advance and reports what that
one rule delivers. That last number is the honest one.

Everything is decided on the **validation** halves - `pool_val_accuracies`
against `source_val_accuracies` - because a stopping rule is a selection, and a
selection may only see what a selection is allowed to see. The score is the
study's own rule,

    score = (adaptation - A0) - (P0 - preservation)

with `A0` and `P0` read off disk through `stopping_table.stage_baselines`, never
written down. `P0` is one number everywhere, because there is one source
population; `A0` is the shipped model's accuracy on the writers the STAGE
federates - 0.8225 on the ten-writer cohort the search ran on, 0.8032 at five
clients, 0.8441 at twenty and 0.7590 on the extreme pair - because a stage
scored against another cohort's do-nothing is not scored at all.
`docs/REPRODUCE.md` §6 has the mapping and the book each comes from.

Within one arm both shift every round by the same constant, so which round or
which rule wins does not depend on them at all. That is why the per-cohort fix
moved every absolute score in this document and left every round, every fire
count and every gain exactly where it was: the levels below are read against
four do-nothings now, and no difference between two of them ever depended on
which.

Rounds are one-based over the stored series, whose first entry is the shipped
model before any client has trained. That entry is the reference every drift is
measured from, which is why a permitted rule can never stop before round 2.

---

## 3. What the horizon costs, stage by stage

The gap between the oracle stop and the fixed horizon, over the 83 arms of the
eight hundred-round stages:

| stage | arms | min | median | max | median r\* |
|---|---|---|---|---|---|
| aggregation finals | 19 | 1.00p | 2.17p | 4.62p | 38 |
| regularisation finals | 28 | 0.00p | 0.58p | 1.92p | 94 |
| combinations | 18 | 0.12p | 0.54p | 1.01p | 87 |
| ten clients, one dropped | 5 | 0.32p | 0.85p | 2.38p | 63 |
| five clients, one dropped | 5 | 0.76p | 1.46p | 6.10p | 44 |
| twenty clients, two dropped | 3 | 0.15p | 0.23p | 0.51p | 94 |
| twenty clients, four dropped | 3 | 0.04p | 0.08p | 0.83p | 93 |
| **the extremes** | **2** | **5.96p** | **11.57p** | **17.19p** | **22** |

The regularisation row is 28 arms rather than the 26 it was: the blend's own
finals are two more full-horizon regularisation arms and they belong in every
reg-full view. Neither of them fires under any rule below, and both sit at the
median of that stage - so the population this document is read over grew by two
and the answer it gives did not change. That is one of the two reasons a number
here has moved since it was first written; the other is the per-stage `A0` of
section 2, which moved the levels and nothing else.

**The horizon is close to harmless everywhere the federation has something to
average, and catastrophic where it does not.** Seven stages have a median cost
under two and a quarter points and a worst arm of six; the two extremes cost
5.96 and 17.19 points. The median oracle round says the same thing from the
other side: the regularisation finals, the combinations and both twenty-client
settings peak in the nineties - they are still improving when the horizon
arrives - while the extremes peak at rounds 8 and 36.

Within the well-behaved stages the cost is not uniform. The three arms that pay
most are the ones with the weakest hold on the source model: the two adaptive
server rules (`fedadam` 4.62p, `fedyogi` 4.16p) and the five-client unmodified
control (6.10p), which is the setting with the fewest updates to average.

---

## 4. What one permitted rule delivers

Chosen on the mean score it stops at, over all 83 arms:

> **stop when `proxy_acc` has drifted more than 0.05 from its round-1 value**

`proxy_acc` is the shipped model's accuracy on a fixed public proxy set (MNIST
here), which the server keeps; the budget is five points of it. The runner-up is
`proxy_kl > 0.5` at 8.24p mean against this rule's 8.31p, and the fixed horizon
is 7.85p.

| stage | arms | final | one rule | vs final | worst arm | best arm | oracle |
|---|---|---|---|---|---|---|---|
| aggregation finals | 19 | 5.82p | 6.68p | **+0.86p** | -2.50p | +2.96p | 8.31p |
| regularisation finals | 28 | 8.26p | 8.17p | -0.09p | -5.28p | +1.08p | 8.98p |
| combinations | 18 | 8.86p | 8.86p | 0.00p | 0.00p | 0.00p | 9.44p |
| ten clients, one dropped | 5 | 7.81p | 8.30p | +0.49p | 0.00p | +1.41p | 9.02p |
| five clients, one dropped | 5 | 10.41p | 12.09p | **+1.69p** | -0.05p | +4.94p | 13.26p |
| twenty clients, two dropped | 3 | 6.18p | 6.18p | 0.00p | 0.00p | 0.00p | 6.47p |
| twenty clients, four dropped | 3 | 6.22p | 6.22p | 0.00p | 0.00p | 0.00p | 6.53p |
| the extremes | 2 | 10.92p | 17.65p | **+6.73p** | +1.00p | +12.46p | 22.49p |
| **all** | **83** | **7.85p** | **8.31p** | **+0.46p** | -5.28p | +12.46p | 9.33p |

**One rule, fixed before any of these runs started, recovers six and three
quarters of the extremes' eleven-and-a-half-point median loss and does not
damage the main settings.** On three of the eight stages it never fires on a
single arm and returns the horizon exactly. It gains most where the horizon
costs most, which is the behaviour a stopping rule has to have to be worth
adding.

It is not free. The mean is what the rule is chosen on; the worst arm is what it
costs somebody, and one regularisation arm (`fisher_lam8`, parallel) loses 5.28
points to it, with `fedadam` losing 2.50. Both are arms whose signal moved early
for reasons that were not forgetting. A rule fixed in advance is a rule that is
sometimes wrong, and that column is printed beside the mean so the trade is
visible rather than averaged away.

**39 of the 83 arms are best served by a rule that never fires.** For those arms
the horizon already *is* the best round reachable by any permitted stop, and no
rule can beat it - which is the other half of the finding in section 3, seen
through the signals rather than through the oracle.

---

## 5. The rule that does not need a proxy at all

Sections 2 and 4 take the constraint at its word: forgetting lives on data the
server has lost, so a rule has to be built on something that stands in for it.
But there is a series in every one of these runs that is neither forbidden nor a
proxy for anything. **The cohort's own validation accuracy is held by the
clients.** The server reads it every round to report a number, it is the
adaptation half of the study's own score, and nothing about it is unavailable
the way `source_val_accuracies` is unavailable. It says nothing about
forgetting - and it does not have to, because the round the cohort stops
improving turns out to be close to the round after which the horizon is only
spending the source model.

`tools/plateau_rule.py` prices it. Let `c(t)` be the fold-mean cohort validation
accuracy and `b(t)` its best value so far. The rule **fires** at the first round
where `c` has failed to exceed `b + eps` for `k` consecutive rounds. What the
server deploys is not the round it fired on: it has been keeping the checkpoint
of the best cohort round all along, so it deploys `t_kept = argmax c(t)` over
`t <= t_fire`, earliest round on ties. An arm the rule never fires on runs the
full horizon and is scored at its last round, exactly as the tables report it.
The setting the paper reports is **`k = 20`, `eps = 0`**.

Over the same 83 arms, on the same basis, against the same horizon:

| over all 83 arms | mean score | vs the horizon |
|---|---|---|
| the fixed horizon | 7.85p | - |
| one permitted signal, fixed in advance (section 4) | 8.31p | +0.46p |
| **the plateau rule, best checkpoint kept** | **9.04p** | **+1.19p** |
| the oracle - a bound, never a rule | 9.33p | +1.48p |

**The plateau recovers four fifths of what stopping is worth on this study, and
the eight signals recover under a third of it.** The rule fires on 50 of the 83
arms; on the other 33 it returns the horizon exactly.

| stage | arms | final | plateau | vs final | fires | oracle |
|---|---|---|---|---|---|---|
| aggregation finals | 19 | 5.82p | 8.04p | **+2.22p** | 19 | 8.31p |
| regularisation finals | 28 | 8.26p | 8.66p | +0.40p | 10 | 8.98p |
| combinations | 18 | 8.86p | 9.17p | +0.31p | 11 | 9.44p |
| ten clients, one dropped | 5 | 7.81p | 8.70p | +0.89p | 4 | 9.02p |
| five clients, one dropped | 5 | 10.41p | 12.87p | **+2.47p** | 4 | 13.26p |
| twenty clients, two dropped | 3 | 6.18p | 6.18p | 0.00p | 0 | 6.47p |
| twenty clients, four dropped | 3 | 6.22p | 6.22p | 0.00p | 0 | 6.53p |
| the extremes | 2 | 10.92p | 22.49p | **+11.57p** | 2 | 22.49p |
| **all** | **83** | **7.85p** | **9.04p** | **+1.19p** | **50** | **9.33p** |

The extreme row is the one to read twice. **The plateau reaches the oracle on
both extreme arrangements**: `dual` fires at round 28 and keeps round 8 for
22.95p, `double` fires at round 56 and keeps round 36 for 22.03p, and those are
the oracle rounds and the oracle scores. On the two runs where the horizon costs
most, a rule with no access to the source population finds exactly the round a
rule with full access would have chosen - because on those runs the cohort's
accuracy peaks where the score does.

**Exactly one arm of the eighty-three is hurt, and it loses half a point.** The
combination `weight_q0 x hybrid_seq_mix0p5` on the parallel schedule keeps a
round worth 0.50p less than its hundredth. Against the one-signal rule of
section 4, which costs its worst arm 5.28 points, that is the more important
number than the mean: a rule fixed in advance is a rule that is sometimes wrong,
and this one is wrong by very little.

**Keeping the checkpoint is where most of the gain is.** Deploying the round the
rule fired on instead scores 8.34p, +0.49p - about what the signals deliver. The
plateau is not principally a better stopping detector; it is the observation that
a server which keeps its best cohort round does not have to detect the peak, only
notice afterwards that it has passed.

### Is `k = 20`, `eps = 0` fitted to the extremes?

It is the best of sixteen `(k, eps)` cells on the mean over all 83 arms, and the
two extreme arrangements are the arms with by far the most to gain, so the
objection writes itself. The answer is in `plateau_grid.csv`, which carries the
mean over the 81 non-extreme arms beside the mean over all of them:
**choosing the setting on the non-extreme arms alone, with both extremes held
out entirely, picks `k = 20`, `eps = 0` as well.** The grid is also flat around
it - `eps` of 0.001 and 0.0025 give 9.02p and 9.03p - so the setting is a region
rather than a point. `tests/test_plateau.py` pins the hold-out result and the
headline row, so a regenerated view that moved either is a failing test.

The grid is deliberately small and coarse: four patiences, four margins, one
tie-break. A finer sweep on these same runs would be a setting fitted to them,
which is the objection this section exists to answer rather than to earn.

**What it is not.** It is not the oracle, which maximises the *score* - a
function of the source population - where this maximises the cohort accuracy,
which is not. It is not a per-arm choice: one `(k, eps)` is fixed for every arm
of every stage. And it is not free of the horizon: an arm that fires at round 56
still ran 56 rounds, which is why `plateau_arms.csv` reports the fire round
beside the kept round rather than folding the two together.

### And chosen where it is not then measured?

Holding the two extremes out is the weakest form of that question: it removes
two arms. `tools/plateau_holdout.py` asks the strong one - choose `(k, eps)` on
a designated subset and evaluate it, unchanged, on a disjoint one - in three
directions, over **28 protocols**: by fold, by stage, and by ten seeded random
halves of the arms. A fold split is the one that needs care, because the
published rule runs on the fold MEAN of an arm's five folds: there is one trace
per arm and cutting it in half would not be a fold split at all. So the arm is
**re-averaged over the selected folds alone**, which is the trace a study run on
those folds alone would have had, and the five-fold filter is applied to the
full five so that a protocol changes the traces and never which arms are in the
table.

**All 28 protocols choose `k = 20`.** Eleven of them pick a positive margin
beside it, and `held_gain_at_primary` prices what that cost: nothing, to a
hundredth of a point. **And no protocol loses on the set it did not see** - the
worst held-out gain of the 28 is zero, on the two federations where the rule
never fires at all. Leave one fold out, choose on the other four and evaluate
on the fold the setting never saw: **+1.83p, sd 0.49p** over the five folds. Cut
the arms in half at random, ten times: **+1.33p, sd 0.23p** on the half held
out. Both sit above the in-sample +1.19p rather than below it, which is not a
stronger result but a different one - a single fold is a noisier trace than the
five-fold mean, and a noisier trace is one the fixed horizon costs more on.

The two extreme arrangements are read off the full five-fold traces whatever
the setting was chosen on, because that is what the rounds above are quoted
from: **`dual` keeps round 8 and `double` round 36, at the same scores, under
all 28 protocols.** All 28 are on the page: `tables/kholdout.tex` prints the six
families, each headed with the number of protocols it stands for, and the two
whose splits are interchangeable are summarised with a mean and a standard
deviation rather than dropped. The appendix table used to print six rows that
expanded to nineteen, so the twenty-eight its own caption counted could not be
reached from it. `plateau_holdout_protocols.csv` carries the rows,
`plateau_holdout_extremes.csv` the 56 extreme checks and
`plateau_holdout_grids.csv` the thirteen fold sets against all sixteen cells;
`tests/test_plateau_holdout.py` pins the count, the patience, the sign of every
held-out gain and both extreme rounds.

---

## 6. Only five of the eight signals can be steered at all

The budget grid is shared across signals - nine budgets from 0.001 to 0.5 - which
is deliberate: a budget tuned per signal is a budget fitted to the runs it is
read on. But three signals are not on that scale, and over the 83 arms their
budgets do nothing. Distinct stop rounds produced by the nine budgets, averaged
over arms:

| signal | distinct stops | median stop round | range |
|---|---|---|---|
| `dist_l2_to_global` | 1.00 | 2 | 2 - 2 |
| `dist_fisher_to_global` | 1.00 | 100 | 100 - 100 |
| `dist_fisher_norm_to_global` | 1.06 | 100 | 77 - 100 |
| `retention_known` | 2.98 | 16 | 2 - 100 |
| `agreement_with_global` | 3.04 | 2 | 2 - 100 |
| `kl_global_to_current` | 3.39 | 2 | 2 - 100 |
| `proxy_acc` | 3.72 | 4 | 2 - 100 |
| `proxy_kl` | 5.11 | 2 | 2 - 100 |

The L2 distance to the shipped model reaches order one within a single round, so
every budget in the grid fires at round 2 and the rule degenerates to "never
adapt". The two Fisher distances live at 1e-10, so no budget ever fires and the
rule degenerates to "never stop". **This is a fact about the units, not about
the signals**: the three parameter-space distances would each need a grid of
their own, and a grid chosen per signal on these same runs would be fitted to
them. They are reported as unusable under a shared budget rather than rescaled
until they look usable.

The two proxy-set signals are the only pair that both track forgetting and sit
on a scale a shared budget can address. Of the 44 arms whose per-arm best rule
fires at all, 41 are served by one of those two and 3 by
`kl_global_to_current`; the other five signals serve none.

---

## 7. The extremes: the case that made this necessary

The figure is `$FOA_STUDY_DIR/figures/extreme_stopping.png`, written by
`tools/extreme_stopping.py --fig`: one panel per case, adaptation and
preservation, with the oracle stop marked.

The two extreme arrangements are the runs where the federation has next to
nothing left to average - `dual` is one client holding two writers' rows
merged, and `double` is the same rows with a client boundary between them,
which makes it the only one of the pair with a second update. Both reach their
best trade in the first tens of rounds and spend the rest of the horizon taking
the source model apart.

| case | r\* | oracle | final | cost | one rule | at round | vs final |
|---|---|---|---|---|---|---|---|
| `dual` | 8 | 22.95p | +5.76p | 17.19p | 18.22p | 26 | **+12.46p** |
| `double` | 36 | 22.03p | +16.08p | 5.96p | 17.07p | 59 | +1.00p |

The rows reproduce `tools/extreme_stopping.py` exactly, because
`stopping_table.py` imports that tool's fold mean, oracle round and score
functions rather than restating them. Running `stopping_table.py --stage
extreme` and `extreme_stopping.py` against the same root is a check that the two
have not come apart.

`dual` reaches 22.95p at round 8 and ends the horizon at 5.76p: it gives back
seventeen of the twenty-three points it had. Stopped by the one permitted rule it
ends at 18.22p, twelve and a half points above where the horizon leaves it. The
arrangement is not what costs it those points. The horizon is. Read against the
ten-writer cohort's do-nothing rather than its own, `dual`'s horizon score used to
print negative and the sentence here used to say it ended below the shipped
model; it does not, and the seventeen points the horizon takes are the finding
either way, because a difference between two rounds of one arm never depended on
which `A0` was subtracted from both.

---

## 8. What `retention_known` reads while the model comes apart

`retention_known` is the fraction of the source model's correct predictions the
current model still gets right, measured on data the clients hold. On the two
extremes it never leaves 1.0 by more than **0.0080** while the source population
falls by **15.79** and **6.14** points.

It is not only the extremes. Over all 83 hundred-round arms, `retention_known`
at budgets 0.1, 0.2 and 0.5 **never fires on a single arm** - the mean score of
those three rules is 7.85p, which is the fixed horizon to the last hundredth of
a point. Only budgets an order of magnitude tighter move it at all.

Per-run rank correlation against the source validation drop puts it last of the
eight (median Spearman -0.461, against -0.819 for `agreement_with_global`, which
measures a related thing on the same data). A reader who trusted it would have
seen nothing wrong at round 100 of a run that had lost a sixth of the source
population.

---

## 9. Do the signals track forgetting at all?

`foa signals` correlates each signal with both definitions of true forgetting -
the source validation drop and the source test drop - per run and pooled over
every round of every run. Over 2,466 runs and 104,753 rounds:

**That pass predates the blend's two stages**, so this section's correlations
are read over the programme as it stood before them, while sections 3 to 5 are
read over all 83 arms. The signal files are derived - one pass over the run
records rewrites them - and re-running it moves the populations below and none
of the rankings this section draws; it has not been re-run because nothing here
turns on the two arms it would add. The command is in section 10.

| signal | pooled Spearman (val) | per-run median Spearman (val) |
|---|---|---|
| `kl_global_to_current` | +0.694 | **+0.920** |
| `proxy_kl` | +0.702 | +0.894 |
| `dist_l2_to_global` | +0.624 | +0.827 |
| `agreement_with_global` | -0.587 | -0.819 |
| `dist_fisher_to_global` | +0.571 | +0.807 |
| `dist_fisher_norm_to_global` | +0.571 | +0.807 |
| `proxy_acc` | -0.500 | -0.515 |
| `retention_known` | -0.384 | -0.461 |

The two targets agree to within 0.005 on every row, so the choice of forgetting
definition changes nothing here.

**The pooled column is much weaker than the per-run one, and that gap is the
whole difficulty.** Within a run most signals order the rounds almost perfectly.
Across runs they do not share a scale: the drift that means five points of
forgetting in one arm means half a point in another, and a rule has to fix one
threshold for all of them.

So the strongest *correlate* is not the best *rule*. Each signal's best budget,
scored as the mean over the 83 arms of the round it stops at:

    proxy_acc                   0.05     8.31p
    proxy_kl                    0.5      8.24p
    agreement_with_global       0.2      8.11p
    dist_fisher_norm_to_global  0.001    7.86p   (fires on 5 arms, for a hundredth)
    dist_fisher_to_global       0.001    7.85p   (never fires: the horizon)
    retention_known             0.1      7.85p   (never fires: the horizon)
    kl_global_to_current        0.5      7.29p
    dist_l2_to_global           0.001    1.58p   (always fires at round 2)

`kl_global_to_current` has the best per-run correlation of the eight and the
worst score of any rule that fires: at every budget in the grid it stops *too
early*, below the 7.85p the horizon gives for free. `proxy_acc` is seventh of
eight on correlation and first as a rule. Ranking signals by how well they track
forgetting inside a run would have chosen the wrong one.

### Two cautions about the module's other two tables

**Its oracle is a different oracle.** `analysis/forgetting_signals.py` defines
the oracle stop as the round with the highest *source validation accuracy*,
which is round 0 for 97.8% of the runs in this study: the round that preserves
best is the one where nothing has happened yet. That is a bound on preservation
alone and it is why `stopping_table.py` defines its oracle on the *score*
instead. The two are not comparable and are not meant to be.

**Its selection table pools cohorts.** `signal_selection.csv` runs one selection
over every run in the tree, so the configuration it picks is an extreme
one-client run at adaptation 1.0 and 0.60 forgetting: perfect on a cohort of
one client. That table is only interpretable inside one stage. The study's real
selection is `foa select`, per stage, per cohort.

---

## 10. Regenerating all of it

```bash
export FOA_STUDY_DIR=$FOA_PROJECT_DIR/results/studies/Digits_study01

# are all eight signals present, on every stage, at every horizon?
python tools/check_signals.py --root $FOA_STUDY_DIR --all

# (a) correlations, (b) simulated stopping, (c) selection under a budget
#     -> $FOA_STUDY_DIR/signals/{signal_correlations,signal_stopping,
#        signal_selection}.csv, signals_summary.json, signals_pareto_*.png
#     --population names the runs the shipped views were computed over; without
#     it the pass pools over whatever the root holds now. docs/REPRODUCE.md S6.
foa signals --root $FOA_STUDY_DIR --out <dir> \
       --population study/artifacts/Digits_study01/signals_population.txt

# fixed horizon vs oracle stop vs permitted signal, per arm, all eight stages
#     -> $FOA_STUDY_DIR/tables/stopping/stopping_*.csv
python tools/stopping_table.py --root $FOA_STUDY_DIR \
       --csv $FOA_STUDY_DIR/tables/stopping

# the plateau on the cohort's own accuracy, section 5
#     -> $FOA_STUDY_DIR/tables/stopping/plateau_*.csv
python tools/plateau_rule.py --root $FOA_STUDY_DIR \
       --out $FOA_STUDY_DIR/tables/stopping

# is that setting an artefact of the arms it was chosen on? section 5
#     -> $FOA_STUDY_DIR/tables/stopping/plateau_holdout_*.csv
python tools/plateau_holdout.py --root $FOA_STUDY_DIR \
       --out $FOA_STUDY_DIR/tables/stopping

# the extremes on their own, and the figure in section 7
python tools/stopping_table.py --root $FOA_STUDY_DIR --stage extreme
python tools/extreme_stopping.py --root $FOA_STUDY_DIR \
       --fig $FOA_STUDY_DIR/figures/extreme_stopping.png
```

`stopping_table.py` reads the eight hundred-round prefixes and nothing else. The
twenty-five-round screens are excluded deliberately: a screen ranks and does not
report, and a stopping round read off one would be a claim about a horizon
nothing was measured at.
