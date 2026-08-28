# Digits_study01 - P13 boundary extensions

Three of the screen's winners sat on the **edge** of the row that was searched.
A winner at the end of its row means the optimum may lie outside it, so the rows
are widened - **40 tasks** in total, same protocol as the screen.

| # | row | edge | extension | cells | tasks |
|---|---|---|---|---|---|
| 1 | KD temperature | `T=1`, low end of `[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0]` | `T in [0.5, 0.25]` x `alpha in [0.95, 0.99]` | 4 | 20 |
| 2+3 | NTD tau and beta | `tau=0.5` and `beta=0.001`, both low ends | see below | 4 | 20 |

## Why the KD extension is narrowed

`kd_T1_a0p99` won in **both** families, and `T=1` is the low end of its row.

The KD row is a **T x alpha grid**, so extending it properly would be two new
temperatures x seven alphas = **14 cells, 70 tasks** - a full sweep to probe one
edge. That is out of proportion to the question.

So the extension is narrowed to the top of the alpha row,
`[0.95, 0.99]`, and the justification is in the screen's own
result rather than in convenience: **alpha's optimum was interior to its row,
and the two families agreed on 0.99.** Alpha is not the axis in question. `0.95`
comes along only to catch that optimum drifting as `T` falls - and if it does,
the alpha row needs its own extension and these four cells will have said so.

That is 4 cells, 20 tasks
instead of 70.

## The two NTD extensions share a grid

- **concurrent**: `tau=0.5` is the low end of `[0.5, 1.0, 2.0, 3.0, 4.0]`,
  extended to `tau=0.25` at the betas that won or came close there,
  `[0.01, 0.001]`.
- **sequential**: `beta=0.001` is the low end of `[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]`,
  extended to `beta=0.0001` at the two lowest taus,
  `[0.5, 1.0]`, which is where that winner lives.

Both extend the same `(beta, tau)` grid, so they are **deduplicated** before
emission. No pair repeats, and the total is
**4 cells, not 4 + 4**:

- `ntd_b0p01_t0p25`
- `ntd_b0p001_t0p25`
- `ntd_b0p0001_t0p5`
- `ntd_b0p0001_t1`

## Appended, not inserted

A screening seed is a function of the cell's index in the table, so inserting
these into their rows would have re-seeded every cell after them - and the
585 lines already submitted would no longer be
reproducible from the table that describes them.

They are **appended**. `d01_p13.txt` regenerates byte for byte identical to what
was run, and these 40 lines are the whole of the new work. The
selector groups by the cell's `method`, so each extension is a sibling of the
row it extends and boundary detection now evaluates the widened ranges.

The method count is unchanged - a wider row is not a new method - so **P14 stays
70 lines**.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-40 \
       --output="$FOA_STUDY_DIR/logs/%x_%A_%a.log" \
       $J/study_phase.sbatch $J/d01_p13_ext.txt
```

## After they run: re-select, and patch only what moved

The re-selection is the ordinary `reg-full` command over the widened table:

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/Digits_study01
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    reg-full --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p14.txt --expect 70
```

It prints each `(method, family)` winner. For any that **moved**, emit that
pair's patch - five lines, not seventy:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    reg-patch --study Digits_study01 --root $S \
    --method kd --family concurrent \
    --out $P/jobs/v4/d01_p14_kd_concurrent_patch.txt --expect 5
```

Only three pairs can move - `kd` in either family, `ntd` in either - because
only those two rows were widened. The patch carries **the same seeds its P14
sibling used**, so a patched line and the line it replaces are the same run with
a different coefficient. It states `CHANGED` or `UNCHANGED` into
`tables/p13ext_<method>_<family>_reselection.json` and to stdout, refuses if
that pair was never measured, and is stamped `NOT AUTHORISED` exactly as P14
was.

`reg-top3` needs no special handling: it ranks whatever full-horizon folders
exist, so a patched result simply competes with the one it supersedes and the
better one wins.

## Cost

At `1.86` s per round per family, both families, plus the teacher
forward pass the KD and NTD penalties need: about **178 s per task**.
40 tasks is roughly **1.98 GPU-h** -
minutes of wall clock. Each patch, if needed, is 5 tasks at the full horizon:
about 0.79 GPU-h.
