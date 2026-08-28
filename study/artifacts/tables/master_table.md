# Master table — Digits_study01 (10 outliers / 200 old / 10% dropout / CV-5)

All values = mean over 5 folds. Columns: pooled clients test / per-client mean / old-data preservation (mean of 5 old-fold tests). Reference: g-init centralized max 0.9951.

| Rung | pooled | per-client | old |
|---|---|---|---|
| g-0 untouched (do nothing) | 0.8224 | 0.8144 | 0.9978 |
| Isolated, scratch | 0.3601 (union) | 0.8303 (own) | 0.3876 |
| Isolated, g-0 fine-tune | 0.8205 (union) | 0.9240 (own) | 0.9908 |
| Normal FL (g-0 init, concurrent) | 0.9045 | 0.9013 | 0.9686 |
| Normal FL (g-0 init, sequential) | 0.8999 | 0.8969 | 0.9695 |
| Centralized bound (g-0) | 0.9351 | 0.9355 | 0.9933 |
| Best agg conc: anchor 0.03 | 0.9127 | 0.9116 | 0.9928 |
| Best agg seq: seq_delta_capped | 0.9053 | 0.9017 | 0.9710 |
| Best reg conc: FedNTD b0.01 t0.5 | 0.9208 | 0.9200 | 0.9860 |
| Best reg seq: feature_l2 0.01 | 0.9144 | 0.9119 | 0.9827 |
| COMBO WINNER: trimmed0.4 x feature_l2 | 0.9236 | 0.9237 | 0.9915 |
| Combo balanced: anchor x FedNTD | 0.9171 | 0.9166 | 0.9964 |
| Combo seq best: shuffle x feature_l2 | 0.9198 | 0.9183 | 0.9817 |

## Extreme cases (winner combo) — the minimum-possible FL

Framing (owner): SINGLE is not an FL scenario — its proper benchmark is centralized
training on that client (isolated, both inits). DUAL and DOUBLE are the minimum
possible FL solutions, and they work.

### Single client (f3642_03, do-nothing 0.533) vs its centralized alternatives

| Approach | own test | old |
|---|---|---|
| centralized scratch (isolated) | 0.8278 | 0.1464 |
| centralized g-0 fine-tune (isolated) | 0.9269 | 0.9940 |
| single-client "FL" (winner combo) | 0.8892 | 0.7746 |

For one client, plain g-0 fine-tuning is the right tool — FL machinery adds nothing.

### The minimum FL (two writers: f3642_03 0.533, f2248_68 0.775 under g-0)

| Setup | pooled | per-writer | old |
|---|---|---|---|
| best solo (g-0 fine-tune, per writer) | - | 0.927 / 0.908 | 0.994 / 0.996 |
| DUAL: one client, merged rows (FL-for-single-clients) | 0.9048 | 0.905 | 0.8058 |
| DOUBLE: two clients federating | 0.9273 | 0.937 / 0.918 | 0.8728 |

The minimum-possible federation (two clients) WORKS: f3642_03 reaches 0.937 -
above its best solo result - and double beats dual on both columns on identical
data. Preservation at n=2 (0.873) is below the 10-client study (0.9915): the
full federation is part of the old-knowledge protection.
