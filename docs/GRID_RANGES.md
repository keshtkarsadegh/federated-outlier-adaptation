# Grid ranges: what each knob does, and where its values should sit

**Working notes.** The search that produced these ranges is not a result and is
not reported. The two findings at the end of this file are, and belong in the
paper. This is the record of a design discussion
about where each hyperparameter's range belongs. The search that produced it is
not a result and is not reported; only the ranges it settles are.

## Why this exists

Every regularisation winner the study selected was 100x to 80,000x weaker than
the value published in the earlier work, and every one sat at the weakest end of
its swept range. The ranges were not the cause - they cover and in places exceed
the literature, and the published values are inside them. The cause was that
selection ran on a 25-round screen where total forgetting is about one point,
so any real penalty looked like pure cost. Those settings were then applied to
100-round runs where eight points are at stake.

The ranges still need work, but for a different reason: several of them spend
cells where the knob provably does nothing, and several sample a continuous
parameter at two arbitrary points.

---

## The aggregation knobs

### 1. `anchor` (lambda_s) - the server's pull toward g-0

    theta_{t+1} = theta_t + eta * Delta_bar - lambda_s * (theta_t - theta_g)

The last term removes a fraction `lambda_s` of the current displacement from
g-0 every round, so displacement decays geometrically and **lambda_s is a
half-life measured in rounds**:

| lambda_s | half-life |
|---|---|
| 0.01 | 69 rounds |
| 0.05 | 14 rounds |
| 0.10 | 6.6 rounds |
| 0.30 | 1.9 rounds |
| 1.00 | instant |

At `lambda_s = 1.0` the update becomes `theta_g + eta * Delta_bar`: the model is
reset to g-0 every round and carries one round of learning. It is not a strict
regulariser, it is a setting where progress cannot accumulate - which is why its
adaptation (0.8295) sits barely above doing nothing (0.8225).

Below 0.01 the half-life exceeds the run, so the spring never acts. That is why
four of our seven cells were indistinguishable.

**Decision:** express this knob as a half-life, not a coefficient. Sweep
`h/R` in `{1/8, 1/4, 1/2, 1, 2}` and convert per horizon with
`lambda_s = 1 - 2^(-1/h)`. The same setting then means the same thing at 25 and
100 rounds, and transfers to the other federation sizes without re-tuning.

Old range: the knob did not exist in the earlier work.

### 2. `eta` - the server step size

Scales how much of the averaged client update is applied. `eta = 1` is plain
FedAvg. Smaller damps every round, so it preserves more; the relationship is
smooth with no cliff.

Our range only went **downward** from 1.0. The literature treats
`eta = 1` as a default that "is not always optimal, even for FedAvg", and
`eta = 2` and `3` are tested in published work. Over-relaxation was never tried.

**Decision:** `eta in {0.1, 0.3, 0.5, 0.6, 0.8, 0.95, 1.0, 2.0}` - the damping
region at the density we want, the FedAvg default, and one over-relaxation point.

Old range: `server_eta` existed but was fixed at 1.0 and never swept.

### 3. `fedavgm` (beta) - server momentum

Momentum averages the update direction over roughly `1/(1 - beta)` rounds, so
beta is also a timescale:

| beta | memory |
|---|---|
| 0.5 | 2 rounds |
| 0.9 | 10 rounds |
| 0.97 | 33 rounds |
| 0.997 | 333 rounds - longer than the whole run |

Momentum is a device for continuing in the direction you were already going,
which is exactly what preservation cannot afford. It is the most destructive
knob in the grid: beta = 0.9 costs 4.9 points of preservation where every other
knob costs under one.

beta = 0.9 is the canonical FedAvgM value, chosen to accelerate convergence when
training from scratch on non-IID data - a different problem from ours. No
published work tunes server momentum for knowledge preservation; the forgetting
literature works on the client side.

**Decision:** `beta in {0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}`. Dense where the knob
is still affordable, keeping 0.9 so the paper can report what the standard
setting costs here. Drop 0.97 and 0.997: their memory exceeds the run.

### 4 and 5. `fedadam` and `fedyogi`

`tau` is the adaptivity floor: large `tau` makes the server behave like plain
averaging, small `tau` makes it aggressively adaptive. It is the preservation
knob of this pair - 1e-5 gives 0.9763, 0.1 gives 0.9977. `lr` is the step size;
below 1e-3 the model stops moving.

The two share a grid and between them consumed 132 of 177 cells - 75 per cent of
the aggregation screen - for a resolution the method's own paper does not use.

Reddi et al. (ICLR 2021), who introduced both, sweep:

    server lr  in {0.001, 0.01, 0.1, 1.0}
    tau        in {1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0}
    beta_1 = 0.9, beta_2 = 0.99 fixed

Our `lr` went to 10, ten times beyond anything published, and our `tau` stopped
at 1e-5 where the paper goes to 1e-8. Both mattered: the first FedYogi winner
was `lr = 10`, which cost 2.2 points of preservation for no adaptation gain, and
the boundary rule then spent three rounds walking `tau` down toward a value the
original paper already recommends.

**Decision:** adopt the published grid, extended at the end our own data wants:

    lr  in {0.001, 0.01, 0.1, 1.0}
    tau in {1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1}      = 28 cells each

### 6. `trimmed` (beta)

Removes the beta-proportion highest and lowest values per coordinate, then
averages. A Byzantine-robustness device: beta is meant to exceed the assumed
fraction of adversaries, and beta < 0.5 by construction.

We have no adversaries. The extreme updates it discards are the writers the
shipped model serves worst - the clients the study is about. The data confirms
it is not acting as a restraint: more trimming *raises* adaptation and *lowers*
preservation, the opposite of a drift control.

It nevertheless finished second in the finals with the smallest fold spread of
any method.

**Decision:** `beta in {0.1, 0.2, 0.3, 0.4}` - the full meaningful range, four
cells - and report it in the paper as a robustness aggregator evaluated outside
its design purpose.

### 7. Client weighting

`max_cap = 1.0 / K`, hardcoded, in both `weight_capped` and `seq_delta_capped`.
With K = 10 that caps every client at the *average* client's weight, which makes
`capped` very nearly `uniform` - and the measurements agree, 0.9090/0.9892
against 0.9120/0.9897, within noise. Two of three cells measure the same thing.

The literature treats this as one continuous family, `w_i ∝ n_i^q`, with q = 1
proportional and q = 0 uniform. We sampled two nearly-coincident points of that
continuum and called them separate methods. Proportional weighting is known to
over-emphasise data-rich clients, which matters here: our writers hold between
about 8 and 60 held-out rows.

**Decision:** sweep the cap as one knob -
`cap in {1/K, 2/K, 4/K, infinity}`, where 1/K is the current setting, infinity
is proportional, and the two middle values are unmeasured.

### 8. `seq_mix` (alpha) - the cyclic mixing weight

Each client blends its update into the running model with weight alpha before
passing it on, so information from a given client decays with a half-life in
client visits - another timescale.

Serial federated learning is *known* to forget: published work attributes its
lower plateau to "knowledge loss from previous sites", and Cyclical Weight
Consolidation exists specifically to fix it. Our data agrees - this is the
second most destructive knob in the grid, 4.8 points across its range.

Above alpha = 0.3 both axes get worse together, so the top of our range is not a
trade at all.

**Decision:** `alpha in {0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5}` - four cells
below the current winner instead of one, 0.5 kept as the unrestrained reference,
0.7 and 0.9 dropped as established waste.

**For the paper, not the grid:** the published remedy for cyclic forgetting is
not in our design space. Worth naming as a limitation.

### 9. `median`

Coordinate-wise median. **No hyperparameter - nothing to sweep.**

It assumes clients hold equal amounts of data, which our writers do not, and
with 8 participants the median discards the information of 6 of them in every
coordinate. It scores -0.62 at full horizon: it spends more preservation than it
gains adaptation.

**Decision:** one cell, unchanged. Report it as a robustness aggregator
evaluated outside its assumptions.

---

## Totals

| | cells | tasks at 5 folds |
|---|---|---|
| current | 169 | 845 |
| proposed | ~98 | ~490 |

Almost the whole reduction is FedAdam and FedYogi, 132 cells to 56, by using the
grid their own paper uses. Everything else stays the same size or grows slightly,
because those are the places we were under-sampling a real knob.

---

## The change that still has to be made in code

Three of these knobs - `anchor`, `fedavgm` and `seq_mix` - are **timescales
expressed as coefficients**. Their meaning depends on the number of rounds, so
the same number is a different intervention at 25 rounds and at 100:

    lambda_s = 0.05  is a quarter-of-the-run half-life at 25 rounds
                     and a half-of-the-run half-life at 100

That is why the screen chose settings that could not hold the full horizon: for
these knobs the mismatch is arithmetic, not bad luck.

The fix is to parameterise them in rounds - half-life or memory as a fraction of
the run - and let the emitter convert to the coefficient at whatever horizon is
running. The screen and the final then measure the same thing, and a winner
transfers to the 5-, 20- and extreme-client points without re-tuning.

**Not yet implemented.** To be applied together with the rest of the range
changes, in one pass, once the regularisation side has been through the same
discussion.

---

## Method 1 in full: the anchor, as it goes in the paper

**Status: agreed, to be implemented. Not yet in code.**

### The rule

    theta_{t+1}  =  theta_t  +  eta * Delta_bar_t  -  lambda_s * (theta_t - theta_g)

| symbol | meaning |
|---|---|
| theta_t | the global model at round t |
| theta_g | g-0, the shipped model, fixed |
| Delta_bar_t | the averaged client update of round t |
| eta | server step: how much of the update is applied |
| lambda_s | anchor strength: the fraction of the distance to g-0 removed each round |

### What lambda_s does

Isolating the anchor by setting `Delta_bar = 0`:

    theta_{t+1} - theta_g  =  (1 - lambda_s) * (theta_t - theta_g)

so after n rounds

    distance(n)  =  (1 - lambda_s)^n * distance(0)

The distance from the shipped model shrinks geometrically. `lambda_s` is
therefore not a step size - `eta` is the step size - but the fraction of
*accumulated* displacement handed back each round.

### Why it must be written as a half-life

Solving `(1 - lambda_s)^h = 1/2`:

    lambda_s  =  1 - 2^(-1/h)

with `h` the number of rounds needed to pull the model halfway back to g-0.

    lambda_s = 0.10    ->  h = 6.6 rounds
    lambda_s = 0.027   ->  h = 25 rounds
    lambda_s = 0.0069  ->  h = 100 rounds

A fixed coefficient does not describe a fixed intervention. `lambda_s = 0.05` is
a quarter-of-the-run half-life over 25 rounds and a half-of-the-run half-life
over 100: the same number, applied four times as often, is a materially
different constraint. That is why a value chosen on the 25-round screen could
not hold the 100-round run - for this knob the mismatch is arithmetic, not luck.

At `lambda_s = 1` the update collapses to `theta_g + eta * Delta_bar_t`: the
model returns to g-0 every round and carries a single round of learning. It is
not a strict regulariser but a setting in which nothing accumulates, which is
why its adaptation (0.8295) sits barely above never adapting at all (0.8225).

### The range

Swept as a half-life relative to the run length R, converted per horizon:

| setting | lambda_s at R=25 | lambda_s at R=100 |
|---|---|---|
| h = 2R | 0.014 | 0.0035 |
| h = R | 0.027 | 0.0069 |
| h = R/2 | 0.054 | 0.014 |
| h = R/4 | 0.106 | 0.027 |
| h = R/8 | 0.199 | 0.054 |

Five cells. Nothing slower than `2R`, where the anchor never acts within the
run - four of the seven original cells were in that dead zone. Nothing faster
than `R/8`, where the model stops accumulating.

### To implement

* the cell stores `h_over_R`, not `server_anchor`
* the line emitter computes `lambda_s = 1 - 2^(-1/(h_over_R * rounds))`
* the emitted line still carries a concrete `--server-anchor`, so a task file
  remains readable and re-runnable on its own
* the selection record reports both, so a winner can be quoted either way

---

## Methods 2 to 9 in full

### 2. `eta` - the server step

    theta_{t+1}  =  theta_t  +  eta * Delta_bar_t

`eta` scales how much of the averaged client update is applied. `eta = 1` is
plain FedAvg; smaller damps every round; larger over-relaxes.

This is a step size in the ordinary sense, and it is the one knob in the family
that behaves the way a learning rate does. It has no fixed point, so unlike the
anchor its effect does not compound into a clean decay: total travel is roughly
`eta * R * |Delta|`, but `|Delta|` shrinks as clients converge, so there is no
exact conversion to a horizon-free quantity. It is still horizon-sensitive in
the obvious direction - twice the rounds, roughly twice the travel - which is
one more reason a screen at a quarter of the reported horizon misleads.

Measured, `eta = 0.75` beats `eta = 1.0` on **both** axes. That is not a trade;
it is the full-size step overshooting. It is also mild evidence against the
untested direction, so over-relaxation is dropped.

**Range:** `eta in {0.1, 0.3, 0.5, 0.6, 0.8, 0.95, 1.0}`, swept directly.

**Old range:** the parameter existed, fixed at 1.0, never swept.

### 3. `fedavgm` - server momentum

    m_t      =  beta * m_{t-1}  +  Delta_bar_t
    theta_{t+1}  =  theta_t  +  eta * m_t

Note the buffer is a **sum**, not an average. If the clients keep producing a
similar update, it settles at `m -> Delta/(1 - beta)`, so the applied step is

    eta * Delta / (1 - beta)

Momentum therefore **multiplies the effective step size** by `1/(1 - beta)`:

| beta | step multiplier | memory |
|---|---|---|
| 0.1 | 1.11x | 1.1 rounds |
| 0.3 | 1.43x | 1.4 |
| 0.5 | 2x | 2 |
| 0.7 | 3.3x | 3.3 |
| 0.9 | 10x | 10 |
| 0.97 | 33x | 33 |
| 0.997 | 333x | 333 - longer than the run |

`beta = 0.9` with `eta = 1` is approximately `eta = 10` without momentum. That
is the whole explanation for its 4.9-point preservation collapse, and it says
`beta` and `eta` are not independent knobs: they multiply.

This is worth contrasting with FedAdam, whose first moment is an **average**
(`(1 - beta_1)` factor) and therefore does not amplify. Two knobs that look
alike behave differently, and the difference is one factor in one line.

Momentum is a device for continuing in the direction already travelled, which is
what preservation cannot afford. `beta = 0.9` is canonical for FedAvgM because
that work trains from scratch on non-IID data - a different problem.

**Range:** `beta in {0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}`, swept directly. Dropped:
0.97 and 0.997, whose memories exceed the run and whose amplification is 33x and
333x.

**Old range:** none - no aggregation hyperparameter was swept in the earlier work.

### 4 and 5. `fedadam` and `fedyogi`

    m_t  =  beta_1 * m_{t-1}  +  (1 - beta_1) * Delta_bar_t
    v_t  =  beta_2 * v_{t-1}  +  (1 - beta_2) * Delta_bar_t^2          (Adam)
    v_t  =  v_{t-1} - (1 - beta_2) * sign(v_{t-1} - Delta_bar_t^2) * Delta_bar_t^2   (Yogi)
    theta_{t+1}  =  theta_t  +  eta * m_t / (sqrt(v_t) + tau)

with `beta_1 = 0.9`, `beta_2 = 0.99` fixed, as in the paper that introduced them.

**`tau` is the preservation knob**, and what it does is interpolate between two
different optimisers:

    sqrt(v_t) >> tau     update is  eta * m_t / sqrt(v_t)  - every coordinate
                         moves +/- eta regardless of its gradient. Fully adaptive.

    tau >> sqrt(v_t)     denominator is just tau, so the update is
                         eta * m_t / tau  - a plainly scaled average. This is
                         FedAvg with step eta/tau.

Measured: `tau = 1e-5` preserves 0.9763, `tau = 1e-1` preserves 0.9977.

**The two knobs are coupled.** In the tau-dominated regime behaviour is governed
by the ratio `eta/tau`, not by either alone, which is why 66 cells bought so
little: much of that grid varied both while leaving the ratio unchanged.

**Yogi's difference.** Adam's `v` is a geometric average, so one large round can
multiply it up and a quiet stretch can let it decay. Yogi changes `v`
additively, bounded, in whichever direction is warranted. Since `v` is the
denominator, a collapsing `v` means a suddenly enormous step - the precise event
that destroys preservation - and Yogi exists to prevent it.

**Range, both methods:**

    lr   in {0.001, 0.01, 0.1, 1.0}                        4
    tau  in {1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1}        7      = 28 cells each

This is the grid of Reddi et al., extended at the low end of `tau` toward 1e-8,
which that paper already sweeps and which our own boundary rule spent three
rounds crawling toward.

**Dropped:** `lr = 10`, ten times beyond anything published. It was the *first*
FedYogi winner at 0.9194/0.9587; the cell we now choose gives 0.9185/0.9807 -
the same adaptation to within 0.001, with 2.2 points more preservation. The
out-of-literature top end did not merely waste grid, it won under the old rule
and cost two points for nothing.

**Old range:** neither method existed in the earlier work.

### 6. `trimmed` - trimmed mean

Per coordinate: sort the K client values, discard the `t` largest and `t`
smallest, average the rest.

    theta_{t+1}[j]  =  mean( sorted(Delta_1[j] ... Delta_K[j])[t : K-t] ),   t = floor(beta * K)

**`beta` is quantised by K.** With eight participants it can only bite in steps
of an eighth:

    beta = 0.1  ->  floor(0.8) = 0 trimmed  ->  this is the plain mean
    beta = 0.2  ->  1 from each end          ->  6 of 8 averaged
    beta = 0.3  ->  2                        ->  4 of 8
    beta = 0.4  ->  3                        ->  2 of 8

**Our `beta = 0.1` cell trimmed nothing** - it is the control under another
name, and the numbers agree to within a sampler seed (0.9025/0.9896 against
0.9043/0.9910). One of the two trimmed cells was a duplicate of FedAvg.

The method discards the most extreme updates. In a Byzantine setting those are
attackers; here they are the writers the shipped model serves worst - the
clients the study is about. The data confirms it does not act as a restraint:
trimming more *raises* adaptation and *lowers* preservation, the opposite of a
drift control.

**Range: sweep the integer, emit the fraction.**

    t in {1, 2, 3}          beta = (t + 0.5) / K

The half-step keeps floating-point rounding from dropping `floor(beta*K)` to
`t - 1`. A fixed `beta` does not transfer across federation sizes - 0.2 discards
one client of eight but four of twenty - whereas a fixed `t` means the same
thing everywhere.

**Old range:** none.

### 7. Client weighting

    theta_{t+1}  =  theta_t  +  eta * sum_k p_k * Delta_k

The three named schemes differ only in `p_k`:

    proportional   p_k = n_k / sum(n)          FedAvg's default
    uniform        p_k = 1 / K
    capped         p_k = min(n_k/sum(n), 1/K), renormalised

**The cap is hardcoded at `1/K`,** which is the average weight, so every client
above average is clipped to exactly average and only below-average clients keep
their share. That is very nearly uniform, and the measurements agree:
0.9090/0.9892 against 0.9120/0.9897, within noise. Two of three cells measure
one thing.

The literature writes this as a single continuous family, `p_k ∝ n_k^q`, with
`q = 1` proportional and `q = 0` uniform. We sampled two nearly coincident
points of that continuum and called them separate methods.

It matters here more than usual: our clients are individual writers holding
roughly 8 to 60 held-out rows, so `q` decides whether the writer the model
serves worst counts as much as the one it serves best.

**Range:** `q in {0, 0.25, 0.5, 0.75, 1.0}`, five cells. No horizon dependence.

**Old range:** the same three named schemes existed; the parameter beneath them
was never swept, then or now.

### 8. `seq_mix` - the cyclic mixing weight

    theta  <-  (1 - alpha) * theta  +  alpha * theta_k

confirmed against `sequential_methods.py`: it blends **weights**, not deltas.

Each visit moves the model a fraction `alpha` toward that client and away from
everything before it, so an earlier client's contribution is multiplied by
`(1 - alpha)` at every subsequent visit. After a full pass over K clients, what
survives of the model that began the round is

    retention  r  =  (1 - alpha)^K

At K = 8: `alpha = 0.05` retains 0.66, `alpha = 0.2` retains 0.17, `alpha = 0.5`
retains 0.004 - the round's starting point erased within one cycle.

This is the serial-forgetting mechanism the literature names: sequential FL is
known to reach a lower plateau through "knowledge loss from previous sites", and
Cyclical Weight Consolidation exists to fix it. Our data agrees - second most
destructive knob in the grid, 4.8 points across its range, and above 0.3 both
axes worsen together.

**`alpha`'s natural unit is client visits, not rounds**, and there are K visits
per round. So the same `alpha` erases far more per round at K = 20 than at
K = 10, and a winner found at ten clients would silently change meaning when
carried to five or twenty.

**Range: sweep the retention, emit alpha.**

    r in {0.7, 0.5, 0.3, 0.1, 0.03}          alpha = 1 - r^(1/K)

At K = 8 that is alpha ~ {0.043, 0.083, 0.142, 0.250, 0.346}; at K = 20,
~{0.018, 0.034, 0.059, 0.109, 0.161} - the same behaviour, correctly rescaled.

**For the paper, not the grid:** the published remedy for cyclic forgetting is
not in our design space. Worth naming as a limitation.

**Old range:** none.

### 9. `median`

Coordinate-wise median. **No hyperparameter.**

It assumes clients hold equal amounts of data, which our writers do not, and
with eight participants the median of eight values discards the information of
six of them in every coordinate. It scores -0.62 at full horizon: it spends more
preservation than it gains adaptation.

**Range:** one cell, unchanged. Report it as a robustness aggregator evaluated
outside its assumptions.

**Old range:** none.

---

## Two findings that belong in the paper

### The design space collapses further than we claimed

Section 4 argues that writing every server rule as one update turns a list of
named methods into a grid with named knobs, so that rules which coincide become
visibly the same point. Carrying that through, three entries of our own grid are
not separate methods:

* `seq_fedavg_update` is `seq_mix` at `alpha = n_k / N`
* `seq_equal_update` is `seq_mix` at `alpha = 1/(i+1)`
* `seq_fixed_ratio_update` is `seq_mix` at `alpha = 0.3`

(stated in `seq_mix_alpha`'s own docstring), and two more collapse empirically:

* `weight_capped` at `cap = 1/K` is within noise of `weight_uniform`
* `trimmed` at `beta = 0.1` with eight participants trims nothing, so it *is*
  the plain mean

Five of eighteen aggregation entries are duplicates or special cases. That is
the paper's own thesis applied to the paper's own table, and it is a stronger
claim than the one currently made.

### A coefficient is not an intervention

Three of these knobs are **timescales written as coefficients**, and their
meaning depends on quantities that change between stages:

| knob | really measures | depends on |
|---|---|---|
| `lambda_s` | half-life of displacement | rounds R |
| `alpha` | retention per cycle | clients K |
| `t` (trimmed) | clients discarded | clients K |

So `lambda_s = 0.05` is a quarter-of-the-run half-life at 25 rounds and a
half-of-the-run half-life at 100; `alpha = 0.2` retains 17 per cent per round at
K = 8 and 1 per cent at K = 20; `beta = 0.2` discards one client of eight and
four of twenty.

This is not a tuning detail. It is why a screen at a quarter of the reported
horizon selected settings that could not hold the full run, and why a winner
found at ten clients cannot simply be carried to five or twenty. The remedy is
to sweep the quantity with a stable meaning - half-life, retention, count - and
let the emitter compute the coefficient for the stage that is running. The task
file still carries a concrete number, so it remains readable and re-runnable on
its own.

A study that sweeps coefficients across several horizons and federation sizes is
not sweeping one thing. That is worth saying plainly, because the mistake is
easy, silent, and ours.
