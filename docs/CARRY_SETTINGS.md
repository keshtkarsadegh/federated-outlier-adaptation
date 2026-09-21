# Do the selected arms hold when the federation changes?

*2026-09-01*

Every method comparison in this study was made on one federation: ten clients
with nine drawn per round. A result measured at one setting is a result about
that setting until something else is measured, so the four arms the combination
cross crowned were carried into three other federations and into two extreme
arrangements where there is barely a federation at all.

Nothing here is searched. The arms are read out of the study's own selection
records at the moment each stage is emitted - `tables/p15_stage_winner.json` for
the crowning and `tables/p15_combination_grid.json` for the cross it crowned - so
a re-crowning moves these stages with it rather than leaving them reporting a
configuration nothing currently chooses.

```bash
python tools/report_tables.py --root "$FOA_STUDY_DIR" --what sizes
python tools/report_tables.py --root "$FOA_STUDY_DIR" --what extremes
python tools/extreme_stopping.py --root "$FOA_STUDY_DIR" \
    --fig "$FOA_STUDY_DIR/figures/extreme_stopping.png"
```

---

## 1. What was carried

| arm | schedule | server rule | client penalty | what it is |
|---|---|---|---|---|
| `winner` | concurrent | `anchor_h2` | `ntd_b0.01_t0.5` | the cross's strongest concurrent pair |
| `balanced` | concurrent | `eta_0.95` | `hybrid_seq_mix0.5` | the concurrent pair that gave up least preservation |
| `sequential` | sequential | `seq_fedavg` | `ntd_b0.01_t2` | the strongest sequential pair |
| `control` | both | `fedavg` | none | plain averaging, no penalty |

**The control is the arm that makes the rest readable.** Without it, every arm
moving together between two settings could be the methods or could be the
participation, and no table would separate them. It runs at five and ten clients,
where the question is *what does the federation do*; the twenty-client point runs
the three winners only, because the ten-client pair - nine of ten against eight
of ten, with the control in both - already answers that question and paying for
it again at a larger size would buy nothing.

## 2. What was varied, and what was held

| setting | clients | dropout | drawn | cohort | book | tasks | job file | record |
|---|---|---|---|---|---|---|---|---|
| ten, one dropped | 10 | 0.1 | 9 | the study cohort | `cohort10` | 20 | `d01_c10d10.txt` | `p21_c10_d10.json` |
| five, one dropped | 5 | 0.1 | 4 | worst five | `cohort10` | 20 | `d01_five.txt` | `p18_five_client.json` |
| twenty, two dropped | 20 | 0.1 | 18 | worst twenty | `cohort20` | 15 | `d01_c20.txt` | `p20_c20_d10.json` |
| twenty, four dropped | 20 | 0.2 | 16 | worst twenty | `cohort20` | 15 | `d01_c20.txt` | `p20_c20_d20.json` |

The twenty-client rows share one task file: both dropout levels go in together
because they share a cohort, a book and a setup chain, and because the pair is
the measurement - one dropout level at a new size says nothing the ten-client
runs have not already said. Thirty tasks, fifteen per level.

Held fixed in all four: the shipped model, the anchor, the teacher, the local
budget, the hundred-round horizon and the three evaluation categories. The
participation is never written into a stage - it is
`floor((1 - dropout) * clients)`, the same expression that gives the nine of ten
every other stage of this study runs.

**The ten-client setting had never been run as a stage.** It is the participation
every screen, every final and the crowning itself drew, but until now it existed
only inside the selection records, so the other three settings were being read
against a middle that had to be borrowed. It runs on seed block 13000 - the gap
between the five-client stage at 12000 and the dropout stage at 14000 - because
the three ten-client stages carry the same four configurations and a shared block
would have them drawing identical client-sampling sequences with nothing in any
output to say the three were correlated.

**Two of these are different populations, not one population at three sizes.**
Five clients is the *worst five* of the same ranking the ten-client cohort was cut
from, and twenty is that cohort extended by the next ten writers down it. A row
in one table therefore cannot be subtracted from a row in another to get "the cost
of scale": the cohorts get easier as they grow, which is visible in the adaptation
column, is visible again in the four `A0` values of section 3, and is a property
of the ranking rather than of the method.

Each setting is now scored against its own do-nothing - that is what those four
`A0` values are - and that closed a real defect rather than a presentational one.
A setting outside the search cohort used to be credited with the gain it would
have made on the ten writers instead of the gain it made on the writers it
actually federates, which flatters a harder cohort and punishes an easier one.
What the fix does not buy is a quantity comparable across the tables. Two
settings scored against two different do-nothings are two measurements that share
a formula, so subtracting one score from another still measures the distance
between two populations of writers rather than the cost of scale. What the tables
*do* support is reading the arms against each other within a setting, and reading
the shape of that ordering across settings; every number is now honest about its
own federation, which is what that reading needs and not a substitute for it.

## 3. The settings

Score is the study's rule, `(adaptation - A0) - (P0 - preservation)`, on the test
halves. Both are the shipped model's own accuracies, read out of the study's
evaluation files rather than written down - but they are not read the same number
of times. `P0 = 0.9986` is one number in every table below, because there is one
source population and one model that served it. `A0` is one number per setting,
because the four federations below are four different populations of writers, and
what doing nothing gets on one of them is not what doing nothing gets on another:

| setting | cohort | evaluation book | `A0` |
|---|---|---|---|
| ten clients, one dropped | the study cohort, worst ten | `g0_perfold_evaluations.json` | 0.8225 |
| five clients, one dropped | worst five | `g0_c5_evaluations.json` | 0.8032 |
| twenty clients, either rate | worst twenty | `g0_c20_evaluations.json` | 0.8441 |
| the two extreme arrangements | the worst two | `g0_extreme_evaluations.json` | 0.7590 |

The ten-client value is the one the whole search ran under, and it has not moved.
The other three were the ten-client value until this was fixed, which is to say
three of the four settings were scored against a do-nothing measured on writers
they never federate. `report_tables.COHORT_BOOKS` holds the mapping and
`cohort_of()` applies it to the run-folder stem, so a stem naming no cohort this
study evaluated g-0 on is refused rather than defaulted to the ten.

**Ten clients, one dropped - nine of ten, the study's own setting**

| arm | sched | adapt | +- | preserve | gained | spent | score |
|---|---|---|---|---|---|---|---|
| sequential | cyclic | 0.9328 | 0.0154 | 0.9898 | 11.03p | 0.88p | **10.15p** |
| balanced | parallel | 0.9290 | 0.0120 | 0.9921 | 10.65p | 0.64p | 10.01p |
| winner | parallel | 0.9300 | 0.0210 | 0.9877 | 10.75p | 1.08p | 9.67p |
| control | parallel | 0.9196 | 0.0132 | 0.9704 | 9.71p | 2.82p | 6.89p |
| control | cyclic | 0.9160 | 0.0186 | 0.9703 | 9.35p | 2.82p | 6.52p |

**Five clients, one dropped - four of five**

| arm | sched | adapt | +- | preserve | gained | spent | score |
|---|---|---|---|---|---|---|---|
| sequential | cyclic | 0.9366 | 0.0191 | 0.9866 | 13.34p | 1.20p | **12.14p** |
| balanced | parallel | 0.9327 | 0.0222 | 0.9889 | 12.95p | 0.97p | 11.98p |
| winner | parallel | 0.9324 | 0.0208 | 0.9812 | 12.92p | 1.74p | 11.18p |
| control | cyclic | 0.9309 | 0.0287 | 0.9371 | 12.77p | 6.14p | 6.63p |
| control | parallel | 0.9253 | 0.0382 | 0.9368 | 12.21p | 6.18p | 6.04p |

**Twenty clients, two dropped - eighteen of twenty**

| arm | sched | adapt | +- | preserve | gained | spent | score |
|---|---|---|---|---|---|---|---|
| sequential | cyclic | 0.9095 | 0.0046 | 0.9939 | 6.55p | 0.46p | **6.08p** |
| winner | parallel | 0.9071 | 0.0050 | 0.9931 | 6.31p | 0.54p | 5.77p |
| balanced | parallel | 0.8944 | 0.0104 | 0.9958 | 5.03p | 0.28p | 4.76p |

**Twenty clients, four dropped - sixteen of twenty**

| arm | sched | adapt | +- | preserve | gained | spent | score |
|---|---|---|---|---|---|---|---|
| sequential | cyclic | 0.9100 | 0.0042 | 0.9944 | 6.60p | 0.41p | **6.18p** |
| winner | parallel | 0.9001 | 0.0048 | 0.9935 | 5.61p | 0.51p | 5.10p |
| balanced | parallel | 0.8884 | 0.0099 | 0.9961 | 4.43p | 0.25p | 4.18p |

## 4. What the four settings say

### The sequential arm tops every one of them

`seq_fedavg` with `ntd_b0.01_t2` is first at nine of ten, at four of five, at
eighteen of twenty and at sixteen of twenty - 10.15p, 12.14p, 6.08p and 6.18p,
each against its own cohort's do-nothing.
It was crowned on one setting and it holds on four, which is the strongest claim
in this document: the ordering the cross produced is not an artefact of the
federation it was produced on.

It wins differently at different sizes, and that is worth stating. At five and
ten clients it leads on adaptation while spending about as little as the
preserving arm; at twenty it leads on adaptation alone, where `balanced` protects
more (0.25-0.28p spent against 0.41-0.46p) and still finishes third because it
gains a point and a half less.

### The control collapses at five clients, and only at five

The unmodified baseline spends **6.14p (cyclic) and 6.18p (parallel)** of the
source model at five clients, against **2.82p on both schedules** at ten - more
than twice the forgetting for an adaptation that is barely higher, 0.9309 against
0.9160 on cyclic and 0.9253 against 0.9196 on parallel. The spend column is the
one to read across the two settings, because both spends are measured against the
one source population; the gained column is not, because the two settings
subtract different `A0` values. That is the clearest single result in the size
sweep, because it is measured against the same control at another participation
rather than against a different method.

The reading: with four updates to average per round instead of nine, plain
averaging has much less of the source model's behaviour left in the average to
hold it in place, so the same local budget moves the global model further from
where it started. The three selected arms do not collapse with it - their spend
rises only from 0.64-1.08p to 0.97-1.74p - which is what the penalties and the
anchored server rules are for, and is the argument that the methods matter *most*
where the federation is smallest.

### Twenty at two dropped and twenty at four dropped are the same result

Doubling the dropout at twenty clients moves the sequential arm by +0.10p
(6.08p to 6.18p), the winner by -0.67p and the balanced arm by -0.57p. Against
fold spreads of 0.004-0.010 in adaptation these are small, and the ordering is
identical: sequential, winner, balanced, at both rates.

Eighteen of twenty and sixteen of twenty are, for these arms, the same setting.
That is a robustness statement and not a null result to be apologised for: it
says the twenty-client point is a *size* effect rather than a participation one,
and it is the reason the twenty-client stage was worth running at two rates and
is not worth running at a third.

## 5. The extreme cases

Two arrangements of the cohort's worst two writers, at full participation -
dropping a client from a two-client federation is not a participation study, it
is a coin flip on whether the round happens.

| case | what it is | adapt | preserve | gained | spent | score |
|---|---|---|---|---|---|---|
| `double` | two clients: the worst two | 0.9685 | 0.9363 | 20.95p | 6.23p | **14.72p** |
| `dual` | one client holding both writers' rows merged | 0.9612 | 0.8558 | 20.22p | 14.28p | 5.94p |

Both subtract `A0 = 0.7590`, the shipped model's accuracy on these two writers'
own test rows and on nothing else. The worst two of the cohort are a great deal
harder than the ten, so reading the pair against the ten-writer 0.8225 charged it
for a difficulty that belongs to the cohort rather than to the arrangement.

`double` and `dual` hold **precisely the same rows**. They differ only in whether
the aggregation ever sees them as two updates or as one. That gap - 14.72p
against 5.94p, with the data held exactly constant - is what client boundaries
are worth, and it is 8.78 points whichever `A0` is subtracted, because the same
one is subtracted from both. It is a question no ten-client stage can ask,
because there the boundaries and the data always move together.

### The mechanism: there is no averaging left

At one client the server has one update to combine, and combining one update is
not aggregation. Whatever the server rule is called, the loop is plain
fine-tuning on the outlier writer, and every mechanism this study selected for -
anchoring toward the shipped model, trimming the tails of an ordered coordinate,
averaging a penalty's effect across clients - needs a second update to have any
effect at all. `dual` is therefore not "federated learning at n = 1"; it is the
ablation that shows how much of the preservation was coming from the averaging
rather than from the penalty.

`double` is the one of the pair with a second update, and it ends 8.78 points of
score ahead of `dual` for it. Both nonetheless end **above** the shipped model's
own trade, `dual` at 5.94p, and that is the sentence the per-cohort baseline
changed: read against the ten writers' do-nothing `dual` came out at -0.41p and
appeared to have been worse than not adapting at all, which was an artefact of
scoring it against writers it does not federate. What `dual` loses is not the
adaptation - that is within a point of `double`'s - it is the preservation:
14.28 points of source accuracy against `double`'s 6.23.

### Adaptation saturates in tens of rounds; the horizon spends the rest

The round series say the loss is not inherent to the arrangement - most of it is
the fixed hundred-round horizon. Reading the study's own selection rule off the
validation columns and taking the round that maximises it:

| case | oracle stop | score there | score at round 100 | the horizon cost | adaptation it bought |
|---|---|---|---|---|---|
| `dual` | round 8 | 22.95p | 5.76p | 17.19p | -1.96p |
| `double` | round 36 | 22.03p | 16.08p | 5.96p | -1.30p |

Both cases reach essentially their best adaptation within the first tens of
rounds and then spend between six and seventeen points of score buying
**negative** adaptation. The two peaks are within a point of each other
(22.03p and 22.95p): the arrangements barely differ in what they can reach, they
differ in how fast the horizon takes it away afterwards, and the ordering of the
two by how long they last - 8 and 36 - is the ordering by how much averaging
they have.

**This is an oracle and not a method.** The stopping round is chosen by looking
at the whole run, and nothing in this study proposes a criterion that would have
found it online. It is reported to attribute the loss, not to fix it.

### Retention is blind to it

Every run carries eight forgetting signals, and `retention_known` is the one that
claims to say how much of the original behaviour survives. On both of these
cases it never leaves 1.0:

| case | max deviation of `retention_known` from 1.0 | source population drop |
|---|---|---|
| `dual` | 0.0080 | 15.79p |
| `double` | 0.0080 | 6.14p |

A signal reading 1.0 through a sixteen point collapse is a finding, not a
diagnostic. Whatever `retention_known` measures on these runs, it is not what the
preservation column measures, and a reader who had watched it instead of the
source-population curve would have seen nothing wrong at round 100. It is
reported beside the drop for exactly that reason, and it is why the extreme-case
figure plots the two accuracy curves rather than any signal.

### The decision: kept, with the discussion

Both cases are **kept in the paper and discussed**, not dropped as a
degenerate corner and not reported as a failure of the method. The reasoning:

* they are the only measurement in the study that separates client boundaries
  from client data, and that separation is a positive result;
* they show where the selected mechanisms stop working and why, which is a
  limitation the paper should state itself rather than leave to a reader;
* the loss they show is mostly the horizon, and saying so is a claim about the
  protocol that the round series support directly.

What they must not be reported as is a method comparison. `dual` is not the
method losing to something; it is the method with the thing it needs removed.

### The figure and the command

The figure lives with the study rather than in the repo, because it is an output
of the runs and not a document: `$FOA_STUDY_DIR/figures/extreme_stopping.png`.
Three panels, one per case, each plotting the cohort's validation accuracy and
the source population's against the round number, with the oracle stop marked.
Regenerate it with:

```bash
python tools/extreme_stopping.py --root "$FOA_STUDY_DIR" \
    --fig "$FOA_STUDY_DIR/figures/extreme_stopping.png"
```

Without `--fig` the same command prints the per-case trace tables, the oracle
stop, the stopped-against-final comparison and the retention check that every
number in this section comes from.
