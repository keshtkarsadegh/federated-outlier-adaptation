# Grid ranges: what each knob does, and where its values should sit

**Working notes, not paper material.** This is the record of a design discussion
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
