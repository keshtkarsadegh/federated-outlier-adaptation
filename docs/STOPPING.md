# When to stop, without the data that would tell you

*Read 2026-09-01, against `Digits_study01`: 2,465 audited payloads, 81 hundred-round arms.*

Every stage in this study runs to a fixed hundred-round horizon and reports the
last round. This document asks what that costs, and whether anything the server
is **allowed** to compute could have told it to stop earlier.

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

with `A0` and `P0` read from `g0_perfold_evaluations.json` and
`g0_evaluations.json` through `report_tables.baselines`, never written down.
They shift every round of every arm by the same constant, so which round or
which rule wins does not depend on them at all.

Rounds are one-based over the stored series, whose first entry is the shipped
model before any client has trained. That entry is the reference every drift is
measured from, which is why a permitted rule can never stop before round 2.

---

## 3. What the horizon costs, stage by stage

The gap between the oracle stop and the fixed horizon, over the 81 arms of the
eight hundred-round stages:

| stage | arms | min | median | max | median r\* |
|---|---|---|---|---|---|
| aggregation finals | 19 | 1.00p | 2.17p | 4.62p | 38 |
| regularisation finals | 26 | 0.00p | 0.58p | 1.92p | 94 |
| combinations | 18 | 0.12p | 0.54p | 1.01p | 87 |
| ten clients, one dropped | 5 | 0.32p | 0.85p | 2.38p | 63 |
| five clients, one dropped | 5 | 0.76p | 1.46p | 6.10p | 44 |
| twenty clients, two dropped | 3 | 0.15p | 0.23p | 0.51p | 94 |
| twenty clients, four dropped | 3 | 0.04p | 0.08p | 0.83p | 93 |
| **the extremes** | **2** | **5.96p** | **11.57p** | **17.19p** | **22** |

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

Chosen on the mean score it stops at, over all 81 arms:

> **stop when `proxy_acc` has drifted more than 0.05 from its round-1 value**

`proxy_acc` is the shipped model's accuracy on a fixed public proxy set (MNIST
here), which the server keeps; the budget is five points of it. The runner-up is
`proxy_kl > 0.5` at 8.12p mean against this rule's 8.20p, and the fixed horizon
is 7.73p.

| stage | arms | final | one rule | vs final | worst arm | best arm | oracle |
|---|---|---|---|---|---|---|---|
| aggregation finals | 19 | 5.82p | 6.68p | **+0.86p** | -2.50p | +2.96p | 8.31p |
| regularisation finals | 26 | 8.27p | 8.17p | -0.10p | -5.28p | +1.08p | 9.00p |
| combinations | 18 | 8.86p | 8.86p | 0.00p | 0.00p | 0.00p | 9.44p |
| ten clients, one dropped | 5 | 7.81p | 8.30p | +0.49p | 0.00p | +1.41p | 9.02p |
| five clients, one dropped | 5 | 8.47p | 10.16p | **+1.69p** | -0.05p | +4.94p | 11.33p |
| twenty clients, two dropped | 3 | 8.33p | 8.33p | 0.00p | 0.00p | 0.00p | 8.63p |
| twenty clients, four dropped | 3 | 8.37p | 8.37p | 0.00p | 0.00p | 0.00p | 8.69p |
| the extremes | 2 | 4.57p | 11.29p | **+6.73p** | +1.00p | +12.46p | 16.14p |
| **all** | **81** | **7.73p** | **8.20p** | **+0.47p** | -5.28p | +12.46p | 9.23p |

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

**37 of the 81 arms are best served by a rule that never fires.** For those arms
the horizon already *is* the best round reachable by any permitted stop, and no
rule can beat it - which is the other half of the finding in section 3, seen
through the signals rather than through the oracle.

---

## 5. Only five of the eight signals can be steered at all

The budget grid is shared across signals - nine budgets from 0.001 to 0.5 - which
is deliberate: a budget tuned per signal is a budget fitted to the runs it is
read on. But three signals are not on that scale, and over the 81 arms their
budgets do nothing. Distinct stop rounds produced by the nine budgets, averaged
over arms:

| signal | distinct stops | median stop round | range |
|---|---|---|---|
| `dist_l2_to_global` | 1.00 | 2 | 2 - 2 |
| `dist_fisher_to_global` | 1.00 | 100 | 100 - 100 |
| `dist_fisher_norm_to_global` | 1.06 | 100 | 77 - 100 |
| `retention_known` | 2.96 | 16 | 2 - 100 |
| `agreement_with_global` | 3.04 | 2 | 2 - 100 |
| `kl_global_to_current` | 3.38 | 2 | 2 - 100 |
| `proxy_acc` | 3.73 | 4 | 2 - 100 |
| `proxy_kl` | 5.10 | 2 | 2 - 100 |

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

## 6. The extremes: the case that made this necessary

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
| `dual` | 8 | 16.60p | -0.59p | 17.19p | 11.87p | 26 | **+12.46p** |
| `double` | 36 | 15.68p | +9.72p | 5.96p | 10.72p | 59 | +1.00p |

The rows reproduce `tools/extreme_stopping.py` exactly, because
`stopping_table.py` imports that tool's fold mean, oracle round and score
functions rather than restating them. Running `stopping_table.py --stage
extreme` and `extreme_stopping.py` against the same root is a check that the two
have not come apart.

`dual` ends the horizon *below* the shipped model on the study's own rule;
stopped by the one permitted rule it ends more than eleven points above it. The
arrangement is not what made that number negative. The horizon is.

---

## 7. What `retention_known` reads while the model comes apart

`retention_known` is the fraction of the source model's correct predictions the
current model still gets right, measured on data the clients hold. On the two
extremes it never leaves 1.0 by more than **0.0080** while the source population
falls by **15.79** and **6.14** points.

It is not only the extremes. Over all 81 hundred-round arms, `retention_known`
at budgets 0.1, 0.2 and 0.5 **never fires on a single arm** - the mean score of
those three rules is 7.73p, which is the fixed horizon to the last hundredth of
a point. Only budgets an order of magnitude tighter move it at all.

Per-run rank correlation against the source validation drop puts it last of the
eight (median Spearman -0.461, against -0.819 for `agreement_with_global`, which
measures a related thing on the same data). A reader who trusted it would have
seen nothing wrong at round 100 of a run that had lost a sixth of the source
population.

---

## 8. Do the signals track forgetting at all?

`foa signals` correlates each signal with both definitions of true forgetting -
the source validation drop and the source test drop - per run and pooled over
every round of every run. Over 2,466 runs and 104,753 rounds:

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
scored as the mean over the 81 arms of the round it stops at:

    proxy_acc                   0.05     8.20p
    proxy_kl                    0.5      8.12p
    agreement_with_global       0.2      7.99p
    dist_fisher_norm_to_global  0.001    7.73p
    dist_fisher_to_global       0.001    7.73p   (never fires: the horizon)
    retention_known             0.1      7.73p   (never fires: the horizon)
    kl_global_to_current        0.5      7.19p
    dist_l2_to_global           0.001    1.48p   (always fires at round 2)

`kl_global_to_current` has the best per-run correlation of the eight and the
worst score of any rule that fires: at every budget in the grid it stops *too
early*, below the 7.73p the horizon gives for free. `proxy_acc` is seventh of
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

## 9. Regenerating all of it

```bash
export FOA_STUDY_DIR=$FOA_PROJECT_DIR/results/studies/Digits_study01

# are all eight signals present, on every stage, at every horizon?
python tools/check_signals.py --root $FOA_STUDY_DIR --all

# (a) correlations, (b) simulated stopping, (c) selection under a budget
#     -> $FOA_STUDY_DIR/signals/{signal_correlations,signal_stopping,
#        signal_selection}.csv, signals_summary.json, signals_pareto_*.png
python -m federated_outlier_adaptation.analysis.forgetting_signals \
       --root $FOA_STUDY_DIR
#     the same thing through the CLI:
foa signals --root $FOA_STUDY_DIR

# fixed horizon vs oracle stop vs permitted signal, per arm, all eight stages
#     -> $FOA_STUDY_DIR/tables/stopping/stopping_*.csv
python tools/stopping_table.py --root $FOA_STUDY_DIR \
       --csv $FOA_STUDY_DIR/tables/stopping

# the extremes on their own, and the figure in section 6
python tools/stopping_table.py --root $FOA_STUDY_DIR --stage extreme
python tools/extreme_stopping.py --root $FOA_STUDY_DIR \
       --fig $FOA_STUDY_DIR/figures/extreme_stopping.png
```

`stopping_table.py` reads the eight hundred-round prefixes and nothing else. The
twenty-five-round screens are excluded deliberately: a screen ranks and does not
report, and a stopping round read off one would be a claim about a horizon
nothing was measured at.
