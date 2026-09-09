# Who the gain reaches, and what it cost

*Read 2026-09-09, against `Digits_study01`: 4,825 stored payloads, 2,765 jobs.*

Every score in this study is a cohort mean. A mean can rise while the writer the
cohort was assembled around does not move, and a method chosen on a mean has no
obligation to have helped anyone in particular. This document reads the two
things the mean hides, from the same payloads and on the same test basis:

```bash
python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what fairness
python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what cost
python tools/fairness_cost.py --root "$FOA_STUDY_DIR" --what all --csv "$FOA_STUDY_DIR/tables/paper/"
```

Nothing is retrained and nothing is recomputed from weights. Both halves read
keys the runner already wrote: `final_evaluation.clients.per_client` for the
distribution, and `round_seconds`, `comm_bytes_per_round` and `param_count` for
the cost. All 4,825 payloads carry all five.

---

## 1. What the fairness table measures

`report_tables.py` prints `final_evaluation.clients.accuracy`, one number pooled
over the cohort's test rows. `final_evaluation.clients.per_client` is that same
evaluation broken out by writer, so the two cannot disagree - the fairness table
is the score table's adaptation column, opened up.

| column | what it is |
|---|---|
| `min` `p25` `med` `mean` `max` | the fold-mean of each order statistic of the cohort's client accuracies |
| `gap` | `mean - min`: how far the cohort's average sits above the client it served worst |
| `d-worst` | `min` less the shipped model's own worst client, in points |
| `d-mean` | `mean` less the shipped model's own cohort mean, in points |
| `lifted` | share of clients whose own accuracy beats **its own** shipped-model accuracy |

**The reference is per client, not pooled.** Asking whether a writer improved
needs that writer's own number under `g-0`, so the reference is read per fold
from the study's own evaluation books - `g0_perfold_evaluations.json`,
`g0_c5_evaluations.json`, `g0_c20_evaluations.json` - and the book is chosen by
which one *covers* the run's clients rather than by the stage's name. The ten-
and five-client cohorts share a fold book and agree writer for writer; the
twenty-client book cut a different test split and does not, so a twenty-client
run must be read against its own. Reconstructed this way the ten-client
references reproduce `tables/clients_acc_on_g0.json` to the last digit, which is
the check that the matching rule is the right one.

One client in the study has no reference: the merged writer of the `dual`
extreme, `f3642_03+f0048_00`, is not a writer any book scored. Its row reports
its accuracy and leaves the reference columns empty rather than borrowing a
number from either half.

## 2. The winner lifts the average further than it lifts the worst client

At the combination cross, the crowned arm and the arm that serves the cohort
most evenly are **not the same arm**, and the gap between them is invisible in
the mean:

| combination | sched | min | mean | gap | d-worst | lifted | rank on `min` | rank on score |
|---|---|---|---|---|---|---|---|---|
| `eta_0.95` x `logit_l2_lam0.001` | parallel | **0.8628** | 0.9377 | 0.0749 | +14.24p | 94% | 1 of 18 | 6 of 18 |
| `anchor_h2` x `ntd_b0.01_t0.5` (crowned) | parallel | 0.8174 | 0.9364 | 0.1190 | +9.71p | 92% | 15 of 18 | 8 of 18 |

Their cohort means differ by **0.13 points**. Their worst-served clients differ
by **4.5 points**. The crowned arm is seventh of eighteen on adaptation and
fifteenth of eighteen on the client it served worst, and its gap of 0.1190 is
the widest in the stage: it bought its mean disproportionately from writers that
were already doing well.

The same shape appears one stage earlier. In the regularisation finals the score
winner under both schedules is `hybrid_seq_mix0.75` - 0.8353 `min` cyclic,
0.8350 parallel - while `logit_l2_lam0.001` reaches **0.8551** with the
narrowest gap in the stage (0.0814) from sixth place on score.

**The answer to the section's question, plainly: yes, the winners lift the worst
client, and no, they are not the arms that lift it most.** Every arm of every
headline stage leaves the worst-served client above the shipped model except
two, named in §3. What the selection rule does not do is *prefer* the arms
that spread the gain, because nothing in `score = (adaptation - A0) - (P0 -
preservation)` can see a distribution.

## 3. Where it fails: `balanced` at twenty clients

Carried into the larger federations, one arm stops reaching the bottom of its
cohort at all:

| setting | arm | min | d-worst | mean | d-mean | lifted |
|---|---|---|---|---|---|---|
| twenty, two dropped | `winner` | 0.7202 | +6.43p | 0.9025 | +6.37p | 70% |
| twenty, two dropped | `sequential` | 0.6790 | +2.31p | 0.9027 | +6.39p | 71% |
| twenty, two dropped | `balanced` | 0.6107 | **-4.52p** | 0.8846 | +4.58p | 63% |
| twenty, four dropped | `sequential` | 0.7040 | +4.81p | 0.9040 | +6.52p | 71% |
| twenty, four dropped | `winner` | 0.6940 | +3.81p | 0.8925 | +5.37p | 70% |
| twenty, four dropped | `balanced` | 0.5821 | **-7.38p** | 0.8775 | +3.88p | 62% |

`balanced` is the arm selected for *giving up least preservation*, and at twenty
clients it is the only arm anywhere in this study whose worst-served client ends
**below the model it started from** - by four and a half points at one dropped
rate and seven and a half at the other - while its cohort mean rises by four.
That is the failure mode this table exists to catch, and no score column in the
study reports it.

The ordering also changes with the question. At eighteen of twenty `sequential`
tops the score table and `winner` tops the worst client, by four points.

At the study's own setting the picture is the opposite and cleanly so: all three
selected arms lift the worst client above **both** unmodified controls.

| ten clients, one dropped | min | d-worst | gap | lifted |
|---|---|---|---|---|
| `balanced` | 0.8420 | +12.17p | 0.0871 | 92% |
| `sequential` | 0.8385 | +11.82p | 0.0954 | 92% |
| `winner` | 0.8351 | +11.48p | 0.0959 | 88% |
| `control` parallel | 0.8227 | +10.24p | 0.0979 | 84% |
| `control` cyclic | 0.8118 | +9.14p | 0.1056 | 82% |

At five clients the ordering breaks once more, in the other direction: `winner`
finishes at 0.8402, **below both controls** (0.8586 cyclic, 0.8541 parallel),
with a cohort mean above them.

Across the four settings the score rule crowns `sequential` every time. The
worst-served writer crowns `balanced` at ten, `sequential` at five and at
sixteen of twenty, and `winner` at eighteen of twenty. The two readings coincide
at two settings of the four, and where they part they part by points rather than
by decimals.

The extremes are the one place the two readings agree, because there is almost
no cohort left to be unequal: `double` lifts both of its writers, the
worse-served of them by 22.10p, and `dual` holds one merged client, whose
spread is zero by construction and whose reference no book carries.

## 4. What it cost

| stage | tasks | rounds | measured GPU-h |
|---|---|---|---|
| aggregation screen | 480 | 25 | 2.19 |
| aggregation finals | 85 | 100 | 1.97 |
| regularisation screen | 1,870 | 25 | 37.51 |
| regularisation finals | 110 | 100 | 8.79 |
| combinations | 90 | 100 | 1.76 |
| ten clients, one dropped | 20 | 100 | 0.62 |
| five clients, one dropped | 20 | 100 | 0.30 |
| twenty, two dropped | 15 | 100 | 0.65 |
| twenty, four dropped | 15 | 100 | 0.57 |
| extremes | 10 | 100 | 0.06 |
| size references | 50 | 100 | 2.40 |
| **whole programme** | **2,765** | | **56.81** |

**These are round-loop hours, not booked hours.** `round_seconds` is measured by
the runner from the top of a round to the end of aggregation, so it covers local
training, the in-loop evaluations and the eight signals, and it does not cover
process start-up, dataset caching, model loading or the final evaluation. Read
against the ~112 GPU-h of allocation in [`REPRODUCE.md`](REPRODUCE.md#9-cost),
about **half of what was booked was spent inside the round loop**.

Two rows differ from that table because this one is read off disk rather than
off the submission plan, and each is a stem that holds more than one stage. The
regularisation screen row is 1,870 tasks - `s16_reg_screen3`'s 700 and the
blend's own 1,170 screen, which write under one prefix because they screen one
grid's rows - and the regularisation finals row is 110, the 70 of
`s17_reg_full4` plus the two hybrid emissions and the blend's ten finals. The
size references are 50 tasks that §9 omits entirely. **The extension of §3 is
not in this table at all**: it is not part of the programme, its two stages are
priced in their own READMEs, and a row for them here would put 1,090 tasks
nobody reports into the total.

The screens dominate, and the blend's screen doubled the gap. The two
twenty-five round stems are **39.70 h of the 56.81** - every hundred-round stage
in the programme put together is 17.11 h. That is the intended shape rather than
a problem: screening is where a cheap horizon buys a ranking over 2,350 tasks,
and it is precisely what the finals exist not to repeat.

### The methods are free; the federation is not

Within the study's own setting, where the arms ran the same shape of job:

| arm | s/round | vs control | MB/round |
|---|---|---|---|
| `control` (plain FedAvg) | 0.853 | - | 119.8 |
| `sequential` | 0.897 | +5.2% | 119.8 |
| `winner` | 0.899 | +5.4% | 119.8 |
| `balanced` | 0.947 | +11.0% | 119.8 |

The control runs both schedules from one job, so its job spent 170.7 s and moved
23.95 GB against the 89.7-94.7 s and 11.98 GB of the single-schedule arms. Per
round, which is the comparable unit, it is the cheapest of the four.

Every mechanism this study selected for - the anchored and trimmed server rules,
the logit and feature penalties, the blended schedules - costs between **5% and
11% of wall-clock round time and nothing at all in communication**. The bytes
are identical because `comm_bytes_per_round` is `parameters x 4 bytes x
participants x 2 directions` and no method here changes any of the four: the
model is 1,663,370 scalars, 6.65 MB per direction per client per round, and a
hundred-round run at nine of ten clients moves 11.98 GB.

What *does* move the cost is the federation. Five clients is 53.2 MB per round
and eighteen is 239.5 MB - the communication bill is set by how many clients are
drawn, which is a property of the deployment and not of anything selected here.
The whole programme moved 11.7 TB of simulated traffic.

**Wall clock on a shared partition is not a controlled timing comparison.** Two
stages that ran on different nodes, or beside different neighbours, can differ by
tens of percent for reasons that have nothing to do with the method - which is
why the table above compares arms *within* one stage and the stage table above
it is read as a budget rather than as a benchmark.

---

## 5. What this changes about the selection rule

Nothing, and that is a deliberate answer rather than an omission. The rule was
frozen before the runs and re-ranking on a fairness column afterwards would be
selecting on the reported halves. What the two tables establish is narrower and
checkable:

* the selected arms do reach the worst-served client at the setting they were
  selected on, and by more than the unmodified control does;
* the arm that reaches it *most* is not the arm the rule crowned, at either the
  regularisation finals or the combination cross, and the difference is 4.5
  points at the worst client for 0.13 points of cohort mean;
* one selected arm, `balanced`, fails the worst client outright at twenty
  clients, and only this table says so;
* the cost of every mechanism is single-digit percent of round time and zero
  bytes, so none of the above is a trade against compute.

A rule that scored the worst client instead of the mean is a different study.
This one reports the distribution beside the mean and says which arm each
column prefers.
