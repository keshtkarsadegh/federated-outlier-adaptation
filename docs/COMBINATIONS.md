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
python tools/study_emit.py combos   --root "$FOA_STUDY_DIR" --out /tmp/x.txt --expect 90

python tools/check_seeds.py "$FOA_STUDY_DIR"/jobs/s*.txt      # every stage at once

python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what combos
python tools/compare_arms.py --root "$FOA_STUDY_DIR" --what composition
```

## The shortlists it was built from

| | concurrent | sequential |
|---|---|---|
| server rules | `anchor_h2`, `eta_0p95`, `weight_q0` | `seq_delta_capped`, `seq_fedavg`, `seq_mix_r0p3` |
| penalties | `logit_l2_lam0p001`, `ntd_b0p01_t0p5`, `hybrid_seq_mix0p5` | `ntd_b0p01_t2`, `logit_l2_lam0p003`, `hybrid_mix0p5` |

The two schedules shortlist different penalties - each row's own winner at the
full horizon, at that schedule's own coefficients - so the cross is not the same
nine pairs twice. Both lists carry a kd+fisher blend, which is the arm the
finals table is topped by and the reason the blend was later given a screen of
its own.

## The result, read the wrong way first

Comparing each pair's mean against its two halves' means, **seven of eighteen
combinations beat both halves**, by +0.09 to +1.21 points. Read that way seven
rows of the stage look like a clean positive: the two mechanisms compose.

**That reading is wrong, and it is wrong in a way a table of means cannot show.**

## The result, paired by fold

The five folds are the same five partitions for every arm, so a fold that is
hard for one arm is hard for all of them. Differencing within a fold removes
that shared difficulty. Doing so:

Differencing each pair against its better half:

| | gains | fold sd | positive on every fold |
|---|---|---|---|
| concurrent | -0.57 to +1.21 | 0.40 to 1.46 | **2 of 9** |
| sequential | -0.20 to +0.80 | 0.24 to 1.43 | **0 of 9** |

Almost every gain is smaller than the spread it came from. What the means were
hiding:

    seq_mix_r0p3 x hybrid_mix0p5       +0.80 mean   +2.0  +1.0  -0.6  +0.7  +0.9
    seq_delta_capped x ntd_b0p01_t2    +0.44 mean   +1.3  +0.6  +0.5  -1.0  +0.9
    seq_delta_capped x hybrid_mix0p5   +0.28 mean   +1.8  +1.0  -1.8  -0.5  +1.0

Each is carried by one or two folds and contradicted by another.

The two survivors are both the same penalty under two different rules:
`anchor_h2 x ntd_b0p01_t0p5` at +1.21, sd 0.58, and `eta_0p95 x ntd_b0p01_t0p5`
at +0.77, sd 0.40, positive on all five folds. They are the same two rows that
clear **both** halves on every fold in `tables/paper_figures/combos_folds.csv`.

No row is negative on every fold. Four are negative on the mean, the worst of
them `weight_q0 x hybrid_seq_mix0p5` at -0.57: adding that server rule to that
penalty does not help, and the folds do not agree that it hurts either.

**Conclusion: the two halves do not measurably compose at five folds.** The best
combination is still the best single number in the study -
`seq_delta_capped x ntd_b0p01_t2`, 0.9375 adaptation / 0.9902 preservation,
score 10.661 - but its margin over the penalty alone is +0.44 against a fold sd
of 0.89 and must not be reported as a gain.

## What does survive: the penalty is the mechanism

The same paired test applied to the two halves themselves gives an effect three
times larger, and it holds:

| | difference | positive on every fold |
|---|---|---|
| concurrent | +1.25 to +3.16 | **9 of 9** |
| sequential | +1.81 to +4.14 | 6 of 9 (all 9 positive on the mean) |

Every client penalty beats every server rule, by one and a quarter to four
points, and every one of the eighteen rows is positive on the mean. On the
parallel schedule it is unconditional: nine rows, forty-five folds, no
exception. On the cyclic schedule the three rows that fall out of the consistent
column are all the same penalty - the `hybrid_mix0p5` blend, whose fold spread
is twice the other two penalties' - and not the same rule.

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

### What it found, read beside the table above

The two cells it crowned were re-run at the reporting horizon and are in
`tables/paper/extension_combo_tune.csv`, five arms per schedule on the test
basis: the rule alone, the penalty alone, the untuned pair of those same two
halves, the tuned pair, and the pair the stage above crowned - the arm the
programme shipped, rather than whichever of the eighteen leads the column it
would be read in.

    concurrent  tuned pair  9.34p   vs untuned -0.75 (2 of 5 folds)
                                    vs penalty alone -1.16 (1 of 5)
                                    vs crowned pair -0.88 (1 of 5)
    sequential  tuned pair  9.14p   vs untuned +0.03 (3 of 5 folds)
                                    vs penalty alone -1.55 (0 of 5)
                                    vs crowned pair -1.38 (0 of 5)

**Moving the two halves together did not find a pair that beats the point where
each half is best alone.** On neither schedule does the tuned pair clear the
penalty on its own, and on neither does it clear the pair the stage crowned;
the one difference that is even positive is +0.03 on three folds of five, which
is the same shape of non-result as every row of the paired table above. So the
sentence this document draws - that the two halves do not measurably compose -
is now measured at more than one point of the joint grid rather than at one, and
it says the same thing.

The screen that chose those two cells ranked on the validation columns at 25
rounds, and this table reports test at 100: a screen ranks and cannot price, and
this is what that distinction costs when a screen's winner is finally priced.
The screen's own top five per schedule, with the dials and where each sits in
its row, are in `tables/paper/extension_combo_screen.csv`. Every dial of both
winners but one is at an end of its row, which is the finding to read first and
is why the pair is not offered as a tuned configuration to use.

---

## Addendum, 2026-09-14: the same question, asked of the pair that was selected

The extension above tunes the pair that leads each schedule **by TEST score**.
The shortlist those rules come from was cut on **validation** -
`tables/p12_agg_top3.json` says so in its own `rank_by` - and on the parallel
schedule the two orderings disagree about which rule comes first: the validation
head is `anchor_h2`, the test head is `eta_0p95`. So the paragraphs above
answered the joint-grid question for a rule the programme did not choose.

`s25_combo_screen_selected.txt` asks it of the pair it did: `anchor_h2` with the
same KD+EWC blend on the parallel schedule, `seq_fedavg` with the same blend on
the cyclic one. The rule half is read from that record on the basis the record
names; the penalty half, the four penalty rows and the mapping onto the
trainer's coefficients are the first extension's, unchanged, so the two grids
differ in one place and can be read against each other. On the parallel side the
rule axis is the anchor's half-life over {1R, 2R, 4R}, and 4R is outside the row
`agg_cells` screened - the selected setting sat at the top of that row, so a
bracket around it has no neighbour above inside it. 216 cells, 1,080 tasks, 25
rounds, at the same search rate and under the same protocol as the pair's own
shipped `s20` lines, whose every non-grid flag it copies.

**It is an extension too, and it does not alter the core selection.** It is
outside the stage table in `REPRODUCE.md`, it has its own section in
`REG_GRID_RANGES.md`, its cells are in no catalogue that `reg-top3`, `combos` or
`stage-winner` reads, and its finals' record says so in its own text. Whatever
it finds is read beside this document, not into it.
