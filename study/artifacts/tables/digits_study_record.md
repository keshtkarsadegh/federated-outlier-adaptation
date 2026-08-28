# Digits_study01 — complete record: what we did, why, and how
Date: 2026-08-28. This is the paper-preparation record. Every step, its reason, its method,
and where its artifacts live (all under `$P/results_v4/studies/Digits_study01/`).

## 0. The question of the paper
A deployed ("shipped") model serves most users well but fails a minority of outlier users.
Can federated learning adapt the model to those outliers **without forgetting** the world it
already serves? Every experiment reports TWO axes: adaptation (outlier clients' test) and
knowledge preservation (old-data test, always the mean over 5 old-fold test partitions).

## 1. Task = NIST SD19, DIGITS ONLY (10 classes, 28x28)
WHY: a single homogeneous category for every writer. The earlier 62-class run confounded
writer style with class composition (writers hold different digit/letter mixtures; outlier
status could reflect WHAT they wrote, not HOW). Digits give every writer the same label set
(~11 samples/class/writer; all 3,580 digit-writers splittable — the confound and the
splittability problem both vanish). Uppercase was analyzed as an alternative and parked:
median writer wrote each letter ONCE (no per-writer test possible without heavy filtering).
HOW: EMNIST-style 28x28 conversion of SD19 (Cohen et al.), packed cache (`data/nist28`),
402,953 digit rows, 3,580 writers.

## 2. Model = FedAvg CNN (McMahan et al. 2017), 1,663,370 parameters
WHY: the reference model for MNIST-class federated tasks (FedAvg, FedAvgM, Reddi, FedNTD
all use it) — citable, BatchNorm-free (BN breaks under FedAvg averaging and caused a teacher
bug in the old paper's hybrid), ~12M MACs/image (11x cheaper than the old 128x128 CNN).
Head-to-head at 28x28 it also beat the old FlexibleCNN (0.889 vs 0.882).

## 3. Detector and two-stage outlier selection (P02-P06)
g-init: FedAvg CNN trained on ALL digit data, CV-5 (per-writer stratified 60/20/20 books;
winner by validation = fold 3, test 0.9951 = the "centralized max" row).
WHY g-init exists: a single global ranking of writers, used ONLY for selection.
Two-stage selection (the design the owner fixed after v1):
  1) g-init splits writers 30% BAD (1,074) / 70% GOOD (2,506) — coarse candidacy;
  2) old data = 200 GOOD writers (seed 20260829, >=100 rows) -> g-0 trained on them;
  3) g-0 (the shipped model itself) scores ALL bad writers on ALL their rows (no leakage —
     g-0 never saw any) -> cohort = the 10 WORST under g-0.
WHY two stages: "outlier" must mean "worst served by the SHIPPED model", not "hard for an
all-seeing detector". v1 (selection by g-init alone) picked writers g-0 actually served
fine (e.g. f2146_84: 0.85 by g-init but 0.96 under g-0) — the rankings decouple because
g-init saw everyone (in-distribution) while g-0 measures style mismatch with the old world.
Banked evidence: pools.json, bad_scores_on_g0.json, cohort table (rank of 1,074; g-init
column alongside; all 10 cover all 10 classes; 72-119 rows — no data-poverty confound),
gallery figure (10 outlier hands vs 5 typical), full provenance for every draw.

## 4. The shipped model g-0 (P07)
CV-5 on the 200 old writers (23,898 rows/fold: 13,289/5,296/5,313), winner fold 2
(val 0.9966, test 0.9951, sha recorded), per-fold Fisher computed on its train split.
STARTING LINE: preservation reference 0.9978 (5-fold mean); do-nothing on the cohort
pooled 0.8224 (worst hand f3642_03 = 0.533). The phenomenon: a 17.5-pt pooled gap,
-46 pts on the worst writer, in a model that serves its own world at 99.8%.

## 5. The ladder (all CV-5, all vs the same two axes)
P08 ISOLATED (100 runs): each outlier alone, scratch AND g-0 fine-tune.
WHY: what does a client get with no federation? Scratch: own 0.830 but union 0.360 /
old 0.388 — useless beyond its own hand. g-0 fine-tune: own 0.924 / old 0.9908 — the
per-client benchmark federation must justify itself against.
P09 NORMAL FL (10 runs): plain FedAvg, both schedules, g-0 vs random init, 9-of-10.
WHY: the disease measurement. Result: 0.905/0.969 (helps on digits, unlike 62c, but leaks
old knowledge and sits 3 pts under the bound). g-0 init > random on both axes.
P10 CENTRALIZED BOUND (10 runs): pooled cohort data, g-0/scratch. WHY: the no-privacy
upper reference: 0.935/0.9933 (g-0), scratch 0.899/0.910.

## 6. Aggregation stage (P11-P12) — server rules
Screen: the full literature grid (169 cells: eta, weightings, median, trimmed, anchor,
FedAvgM, FedAdam/Yogi, sequential family, controls) x 5 folds at 25 rounds; then ALL 18
methods re-run at full 100 rounds at their per-method best (no method filtering at
selection — the owner's rule); top-3 per family only after full runs.
WHY screen-then-full: affordable tuning + honest ranking at the real horizon (the screen
flattered momentum rules — FedAvgM/seq_mix led at 25 rounds and collapsed at 100; a
documented methodological finding).
BOUNDARY RULE in action: trimmed won at trim=0.2 = the range edge; the row was extended
(0.3, 0.4) — and revealed that at 9 participants trim=0.1 floors to NO trimming (int(f*K)),
so the original row was structurally short. trimmed_0p4 (3 of 9 survive) won the extension
and CONFIRMED at full horizon (0.9107/0.9745 vs 0p2 0.9054/0.9734) — aggressive robust
averaging is right for small cohorts with true outliers; 0.4 is the structural max (0.5
would leave 1 update). Full-horizon top: anchor_0p03 0.9127/0.9928 (best both axes).

## 7. Regularization stage (P13-P14) — client penalties
Same protocol: 117-cell literature grid (FedProx/param_l2, EWC/fisher, the old paper's
capped EWC/fisher_scaled, LwF/logit_l2, feature distillation, KD, FedNTD) x 5 folds
screen -> per-(method,family) best -> 70 full runs (family-tagged parents) -> top-3.
The kd+fisher HYBRID (the old paper's method, cleanly re-defined: additive
lam*(mix*KD + (1-mix)*Fisher), correct teacher) is emitted from the measured kd+fisher
winners at mixes 0.25/0.5/0.75 and ran at full horizon.
Boundary extensions: kd T<1 probe (T=0.25 won the screen but LOST at full horizon — T=1
stands; road closed: T->0 is hard-label copying) and ntd low-edge probes (both UNCHANGED).
Full-horizon top: conc FedNTD b0.01/t0.5 0.9208/0.9860; seq feature_l2 and logit_l2 tie
0.9144 (logit_l2 with 0.9970 preservation). FedProx verdict: mu=0 (off) beats every
strength — an honest negative result.

## 8. The selection-rule decision (owner decision A)
At full horizon the VALIDATION ranking and TEST ranking diverged (val partitions ~230
rows/fold; gaps to 3.4 pts; val top-3 included the test-worst method). Owner decision A
(2026-08-28): top-3 and winner crowning are TEST-RANKED among fully-reported finalists
(as in the 62-class study), with the val/test divergence reported and both rankings
recorded in the artifacts. WHY: with near-tied methods, small-val ranking is ranking noise.

## 9. Combinations (P15) — the stage the ladder exists for
Top-3 aggs x top-3 regs per family x 5 folds = 90 full runs (server rule AND client
penalty active together; families never crossed).
RESULT: they STACK. Winner trimmed_0p4 x feature_l2 = 0.9236/0.9915 (beats best agg alone
0.9127, best reg alone 0.9208, control 0.900; 79% of the gap to the no-privacy bound
closed at 99.4% preservation). Balanced champion anchor x FedNTD = 0.9171/0.9964 (0.14 pt
below the untouched-model reference). Anchor x strong penalties TRADE (double restraint).
Eval traces (banked figures): controls' forgetting NEVER plateaus (still falling at r100);
the winner flattens ~r60; balanced is a flat line — the methods change the SHAPE of
forgetting from unbounded decay to equilibrium, not just the endpoint.

## 10. Extreme cases (P16, winner combo, full participation — never any dropout)
single (worst hand alone) / double (two worst as 2 clients) / dual (same rows merged into
1 client). WHY (owner's framing): single is NOT an FL scenario — its benchmark is
centralized fine-tuning (isolated g-0 fine-tune wins there: 0.927/0.994 vs single-FL
0.889/0.775). dual and double are the MINIMUM POSSIBLE FL — and it works: double lifts the
worst hand to 0.937 (above its best solo 0.927) and beats dual on both axes on
byte-identical data — the federation effect appears at n=2, visible in the old-data traces
(single/dual share one forgetting slope; double resists).

## 11. Scaling and dropout extras (winners transferred as-is; NO CV — one g-0, fold 1 only)
WHY no CV: these are robustness probes of fixed winners, not selections; owner's rule.
(Caveat recorded: fold-1 cells are point estimates; the 10-client anchor rows are also
read from fold 1 of the same runs; P15 cross-fold spread quoted as the noise floor;
20-client rows are winners-only -> size-robustness, not attribution.)
Participation FORMULA (owner): participants = n - floor(d*n) for the main/scaling studies
via floor((1-d)*n) equivalently at our points — (5,.1)->4, (10,.1)->9, (10,.2)->8,
(20,.1)->18, (20,.2)->16 — extremes exempt (always full). Implementing it exposed and
fixed an IEEE754 floor bug affecting 56 (n,rate) pairs.
Findings (CV-5 versions where run, fold-1 tables for the paper):
- 5 clients: winner's trim degrades (2-of-4 survive), balanced holds ~0.99 preservation;
  controls' forgetting explodes (0.86-0.87) -> federation SIZE is itself protective;
  methods matter MORE at small n.
- 10 clients 8-of-10 vs 9-of-10: dropout is a modest knob (<=0.7 pt); balanced becomes
  best on both axes at 20% — dropout-invariant.
- 20 clients (18/16-of-20): fold-1 runs of the three winners; cohort = worst-20 (verified
  superset of worst-10), own book + own do-nothing baselines. [results pending]

## 12. Methodological infrastructure worth citing in the paper
- Persistent CV fold books (int8 per-row partition maps) — every run in the study opens
  the same folds; val only for stopping/selection, test only reported.
- Deterministic task lines, disjoint seed ranges per stage, every path through one study
  root with an sbatch guard that REFUSES cross-study paths (exit 78).
- Selection emitters REFUSE to select over missing/unmeasured results (born from a real
  incident: a crashed screen almost produced a plausible fallback selection).
- Boundary rule: any grid-edge winner reopens the range; extensions are append-only
  (deployed screens re-emit byte-identical) with same-seed patch runs for moved winners.
- Winner provenance: every selection artifact records its rule string, both rankings
  (val + test), and sha256 of frozen models.
- Cost: full study ~90 GPU-h on A100 (screens dominate); every stage's runtime measured.

## 13. State at 2026-08-28
DONE: P01-P16 complete + extras (5c, 10c-2out) + extreme reframing + traces + master table
(tables/master_table.md). PENDING: 20-client fold-1 runs (6 tasks, queued) -> scaling
table -> owner discussion -> PAPER. Dropped by owner: AOD study, multi-study grid,
Shakespeare/extra datasets, uppercase study (parked with feasibility numbers banked).
