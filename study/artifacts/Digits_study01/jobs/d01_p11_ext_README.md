# Digits_study01 - P11 boundary extension: the trim row

The screen's trimmed-mean winner was `trimmed_0p2`, sitting on the **high edge**
of the row that was searched, `[0.1, 0.2]`. A winner at the end of its row means
the optimum may lie outside it, so the row is widened: `trim_frac` 0.3 and 0.4,
**10 tasks** (2 cells x 5 folds), same protocol as the screen.

## What trimming actually happens at nine clients

The rule drops `int(f * K)` updates from each end per coordinate, and only bites
when that is at least one. Measured against the implementation at K = 9:

| `trim_frac` | `int(f x 9)` | dropped each end | **updates surviving** |
|---|---|---|---|
| 0.1 | 0 | 0 | **9** |
| 0.2 | 1 | 1 | **7** |
| 0.3 | 2 | 2 | **5** |
| 0.4 | 3 | 3 | **3** |

**At nine clients, `trim_frac=0.1` is not a trimmed mean.** `int(0.1 x 9)` is
zero, so `trimmed_0p1` computes the plain coordinate-wise mean and is
arithmetically identical to `con_delta_eta` at `eta=1`. The row the boundary
rule saw was therefore effectively *[no trimming, drop one]* - `trimmed_0p2` was
the only cell in it that trimmed anything at all.

That makes the extension more clearly warranted than the edge alone suggested,
and it belongs in the write-up rather than being left for a reader to
rediscover from the aggregation source.

`0.5` is not reachable: `ServerState` refuses `trim_fraction` outside `[0, 0.5)`,
and `int(0.5 x 9) = 4` would leave a single update, which is not a mean of
anything. **0.4 is the top of the row that can exist here.**

## Appended, not inserted

A screening seed is a function of the cell's index in the table, so inserting
0.3 and 0.4 into the trim row would have re-seeded every cell after it - and the
screen that has already run would no longer be reproducible from the table that
describes it.

They are **appended**. The 845 lines of
`d01_p11.txt` are byte for byte what was submitted, and these ten are the whole
of the new work. The selector groups by id prefix, so it sees all four trim
cells as siblings regardless of position, and boundary detection now evaluates
the widened range `[0.1, 0.2, 0.3, 0.4]`.

## Submit

```bash
J=$FOA_PROJECT_DIR/jobs/v4
sbatch --account=$FOA_ACCOUNT --array=1-10 \
       $J/study_phase.sbatch $J/d01_p11_ext.txt
```

## After they run: re-select

```bash
P=$FOA_PROJECT_DIR
S=$P/results_v4/studies/Digits_study01
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    agg-full --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p12.txt --expect 90
```

This re-runs the whole per-method selection over the widened table. Two
outcomes:

- **The trimmed winner is unchanged** - the extension confirmed the edge was the
  optimum, `d01_p12.txt` is regenerated identical, and nothing needs re-running.
  `tables/BOUNDARY_HITS.txt` will still record `trimmed_0p4` as an edge if it
  wins, which is then a genuine finding rather than an artefact of a short row.
- **A new trimmed winner emerges** - only the trimmed method's five lines change.
  Emit just those:

```bash
cd $P/repo_foa && PYTHONPATH=$PWD/src $P/envs/fal/bin/python tools/study_emit.py \
    agg-trimmed-patch --study Digits_study01 --root $S \
    --out $P/jobs/v4/d01_p12_trimmed_patch.txt --expect 5
```

**5 tasks**, not 90. The emitter compares the new winner against
the cell P12 was built from and says which case it is; the patch file is stamped
`NOT AUTHORISED` exactly as `d01_p12.txt` was, because a re-run is the owner's
call.

## Cost

At `1.69` s per round per family, 25 rounds plus evaluation and
startup is about **90 s per task**. 10 tasks is roughly
**0.25 GPU-h** - minutes of wall clock. The patch, if
it is needed, is 5 tasks at the full horizon: about
0.30 GPU-h.
