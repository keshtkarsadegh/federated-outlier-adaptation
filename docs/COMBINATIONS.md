# Do the two halves compose?

Every stage before this one changed **one** thing. The aggregation grid varied
the **server rule** and left the client loss alone; the regularisation grid
varied the **client penalty** and left the server on plain FedAvg. Neither can
say whether the two compose - whether a server rule that preserves and a penalty
that preserves preserve *twice*, or whether they are two names for the same
restraint and stacking them buys nothing.

That is the only question this stage asks, so it is a cross and not another
sweep: the top three server rules by the top three penalties, per schedule, at
the reporting horizon.

    3 rules x 3 penalties x 2 schedules x 5 folds = 90 tasks, 100 rounds

The shortlists are read from `tables/p12_agg_top3.json` and
`tables/p13_reg_top3.json`, which are themselves selected from the full-horizon
finals - not from the screens, and not from anything written down in source.

## Reproducing every number here

```bash
python tools/study_emit.py agg-top3 --root "$FOA_STUDY_DIR" --out "$FOA_STUDY_DIR/tables/p12_agg_top3.json"
python tools/study_emit.py reg-top3 --root "$FOA_STUDY_DIR" --out "$FOA_STUDY_DIR/tables/p13_reg_top3.json"
python tools/study_emit.py combos   --root "$FOA_STUDY_DIR" --out jobs/s13_combos2.txt --expect 90 --clients-per-round 9

python tools/check_seeds.py "$FOA_STUDY_DIR"/jobs/s*.txt      # every stage at once

python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what combos
python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what composition
```

## The shortlists it was built from

| | concurrent | sequential |
|---|---|---|
| server rules | `anchor_h2`, `eta_0.95`, `weight_q0` | `seq_delta_capped`, `seq_fedavg`, `seq_mix_r0.3` |
| penalties | `logit_l2_lam0.003`, `ntd_b0.01_t0.5`, `kd_T8_a0.99` | `ntd_b0.01_t1`, `logit_l2_lam0.003`, `fisher_lam0.1` |

The two schedules shortlist different penalties: `kd` fell to last on the cyclic
schedule at full horizon and `fisher` rose to third, so the cross is not the same
nine pairs twice.

## The result, read the wrong way first

Comparing each pair's mean against its two halves' means, **eleven of eighteen
combinations beat both halves**, by +0.04 to +0.73 points. Read that way the
stage is a clean positive: the two mechanisms compose.

**That reading is wrong, and it is wrong in a way a table of means cannot show.**

## The result, paired by fold

The five folds are the same five partitions for every arm, so a fold that is
hard for one arm is hard for all of them. Differencing within a fold removes
that shared difficulty. Doing so:

| | gains | fold sd | positive on every fold |
|---|---|---|---|
| concurrent | +0.04 to +0.45 | 0.47 to 1.39 | **1 of 9** |
| sequential | +0.09 to +0.73 | 0.22 to 2.07 | **0 of 9** |

Every gain is smaller than the spread it came from. What the means were hiding:

    anchor_h2 x ntd            +0.30 mean    +2.5  -0.9  -0.9  +0.3  +0.5
    seq_delta_capped x logit   +0.73 mean    +0.3  +2.5  -0.1  +1.0  -0.1
    weight_q0 x kd             +0.23 mean    -1.5  +0.1  +2.3  +0.1  +0.1

Each is one good fold and four flat or negative ones.

The single survivor is `eta_0.95 x kd_T8_a0.99`: +0.36, sd 0.47, positive on all
five folds - and even that is carried by one fold of the five.

One row is consistently **negative**: `weight_q0 x logit_l2_lam0.003`, on every
fold. Adding that server rule to that penalty genuinely hurts.

**Conclusion: the two halves do not measurably compose at five folds.** The best
combination is still the best single number in the study -
`seq_delta_capped x logit_l2_lam0.003`, 0.9270 adaptation / 0.9910 preservation,
score 9.69 - but its margin over the penalty alone is inside fold noise and must
not be reported as a gain.

## What does survive: the penalty is the mechanism

The same paired test applied to the two halves themselves gives an effect three
times larger, and it holds:

| | difference | positive on every fold |
|---|---|---|
| concurrent | +1.06 to +2.19 | 3 of 9 (all 9 positive on the mean) |
| sequential | +1.51 to +2.68 | **8 of 9** |

Every client penalty beats every server rule, by one to two and a half points.
On the cyclic schedule this is close to unconditional. On the parallel schedule
the effect is the same size but noisier - fold 1 is where the near-zeros sit,
and it is the fold that keeps rows out of the consistent column throughout.

So the honest statement of what this study found is:

> **The client-side penalty is where the preservation comes from. The server
> rule contributes little once a penalty is present, and the two do not
> measurably add.**

That is a weaker claim than "they compose", and a more useful one: it says which
half of the update to spend effort on.

## What would settle the margins

The gains are ~0.4 points against a fold sd of ~1.0 at n=5. Ten folds would
roughly halve the standard error and would settle whether the ~0.4 effects are
real - about 250 extra tasks (the 18 combinations and their halves), ~30
GPU-hours. Worth it only if the composition claim has to appear in the paper;
the effect that carries the paper is the penalty-versus-server-rule gap, which
is already separated from noise.

## A note on how this was nearly reported wrong twice

**The stale directories.** The first reading of the finals mixed 60 leftover run
folders from a superseded programme into a glob, putting cells that no longer
exist in any grid at the top of the table. The combination stage had 90 such
folders of its own, built from `anchor0.00333333`, `trimmed_0.2` and old
regularisation cells - none of which exist under the reparameterised grids. A
stage's result folders must be deleted before it is re-run, and a table that
reads a prefix must be checked against the catalogue that defines it.

**The seed collision.** The first combination emission drew seeds 710001-710175,
which the 140-cell regularisation screen already occupies. Blocks are 1000 wide
and that screen spans 1400. Both stages would have run, both would have written
results, and the two would have drawn the same client-sampling sequence with
nothing in the output to say so. `tools/check_seeds.py` now cross-checks every
task file's span when they are passed together.

Neither failure would have raised an error.

---

## Addendum, 2026-09-09: an extension that tunes the pair jointly

**Nothing above is superseded, and nothing above is re-emitted.** The shortlists,
the eighteen combinations, the fold-paired reading and the conclusion this
document draws all stand exactly as they are. This paragraph records that an
**extension** now exists beside them, and says what it screens.

The cross above paired two shortlists at **one setting each**. Every pair in it
is a server rule at the coefficients it won on *alone* beside a penalty at the
coefficients it won on *alone*: the two shortlists were selected independently -
one under plain FedAvg, the other with no penalty - and no stage of the
programme ever moved the two together. So the conclusion recorded above, that
the two halves do not measurably compose at five folds, is measured at one point
of a joint grid: the point at which each half is best in the other's absence.
That is what was run, and it is what the sentence should be read as saying.

`s23_combo_screen.txt` screens the joint grid for the pair that leads each
schedule by TEST score of `tables/paper/agg-winners.csv` and
`tables/paper/reg-winners.csv` - `eta_0p95` with the KD+EWC blend on the
parallel schedule, `seq_delta_capped` with the same blend on the cyclic one -
over the EWC coefficient, the KD coefficient, the KD temperature, the blend
weight, and, on the parallel side, the server step: 216 cells, 1,080 tasks, 25
rounds, at the same search rate and under the same protocol as the pair's own
shipped `s20` lines, whose every non-grid flag it copies. Its winners are
re-run at the reporting horizon by a stage of its own, under the study's own
selection rule - gain less spend at `w = 1`, on the validation columns.

**It is an extension and it does not alter the core selection.** It is outside
the stage table in `REPRODUCE.md` by design, it has its own section in
`REG_GRID_RANGES.md`, its cells are in no catalogue that `reg-top3`, `combos` or
`stage-winner` reads, and its record - `tables/p23_combo_tune_winners.json` -
says so in its own text. Whatever it finds is read *beside* this document, not
into it.
