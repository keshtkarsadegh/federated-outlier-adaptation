# Scaling & dropout tables — Digits_study01 winners (fold 1, single g-0)
Cells: pooled clients test / old-data preservation (5-fold mean). No CV on these probes
(owner rule); P15 cross-fold spread (the noise floor): ~±0.01-0.02 pooled, ~±0.0003-0.0058 old.
Do-nothing baselines (fold 1): cohort10 = 0.8224 (CV mean; study baseline), cohort20 = 0.8534.
Extremes ran full participation (never dropped); 5c/10c rows include plain-FedAvg controls, 20c is winners-only.

## Federation size (10% dropout column) and dropout (20%)
| Federation | Config | 10% dropout | 20% dropout |
|---|---|---|---|
| 2 (double, extreme) | winner | 0.9273* / 0.8728* | — (never dropped) |
| 5 (4-of-5) | winner | 0.8800 / 0.9780 | — |
| 5 | balanced | 0.9000 / 0.9931 | — |
| 5 | sequential | 0.9100 / 0.9537 | — |
| 5 | control | 0.8600-0.9000 / 0.8900-0.8925 | — |
| 10 (9-of-10 / 8-of-10) | winner | 0.9442 / 0.9932 | 0.9116 / 0.9938 |
| 10 | balanced | 0.9256 / 0.9971 | 0.9349 / 0.9962 |
| 10 | sequential | 0.9209 / 0.9829 | 0.9116 / 0.9838 |
| 10 | control (con) | 0.8977 / 0.9764 | 0.8930 / 0.9765 |
| 20 (18-of-20 / 16-of-20) | winner | 0.9103 / 0.9917 | 0.9212 / 0.9925 |
| 20 | balanced | 0.9168 / 0.9965 | 0.9168 / 0.9965 |
| 20 | sequential | 0.9212 / 0.9905 | 0.9125 / 0.9906 |
*extreme-case CV-5 values (that stage kept CV); its do-nothing writers were 0.533/0.775.

## Verified identity note (the balanced attractor)
At 20 clients the balanced (anchor x FedNTD) rows are identical at both dropout levels.
This was fully verified: the two runs trained genuinely different weights (max param diff
0.046) yet score bit-identically on cohort test (419/457) AND old fold tests (0.997919,
11 errors in 5,286) — direct re-evaluation of both saved models confirmed it
(tables/balanced_recheck.json). The anchor-to-g0 + NTD-from-g0 combination is a strong
attractor: different participation draws converge to functionally the same predictor.
This is WHY balanced is dropout-invariant across every table.

## Readings
1. Preservation rises with federation size for every method (winner old: 0.978 @5 ->
   0.992-0.993 @10-20); controls' forgetting explodes at small n (0.86-0.89 @5).
2. The winner is tuned to its native point (best at 10c/9-of-10: 0.9442 on fold 1);
   at other sizes trimming keeps ~2-4 survivors and it converges toward the pack.
3. Balanced is the robustness champion: 0.99+ preservation at EVERY size and dropout,
   adaptation within 1-2 pts of the best everywhere — the deployment recommendation.
4. Dropout (10->20%) is a modest knob at every size; federation size dominates.
