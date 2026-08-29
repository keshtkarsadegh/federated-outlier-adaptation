> **Superseded.** This describes the PRIOR pipeline (see
> `docs/PRIOR_PIPELINE.md`), not the live study `Digits_study01`.
> For the live study use `docs/DATA.md`, `docs/RUNBOOK.md` and
> `docs/VERIFY.md`. Kept because the code it documents is still present.

# Results layout

What a results tree contains and how to read a stored run. The full narrative is in
[README](../README.md) section 5 (output layout and provenance) and section 3 (frozen
artefacts); this page is the map.

The root is `$FOA_RESULTS_DIR` (default `<repo>/results`, per command `--results-dir`).

---

## Tree

```
results/
├── writer_split.json                     frozen split
├── global_model                          frozen teacher
├── global_results/                       global metrics + fisher/
├── global_clients_model                  combined reference model
├── global_clients_results/               combined reference metrics
├── outliers/                             per-client accuracies, selection, plots
├── <method>_grid_search/                 sweeps
│   └── [seed_<n>/]<scenario>_<metadata>/
│       ├── accuracies_points_100.json    {experiment name: [[clients_acc, global_acc], ...]}
│       ├── accuracies_points_100.csv     top-6 ranking (written by `foa figures`)
│       └── config_points_100.json        provenance of every configuration
├── <parent>_<Trainer>_grid_search/       final runs
│   └── [seed_<n>/]<scenario>_<metadata>/
│       ├── summary_<i>.json
│       ├── overlay_*.png
│       └── base_agg_<agg>_<metadata>_<scenario>/
│           ├── accuracies_<i>.json
│           └── accuracies_over_rounds_*.png
└── logs/                                 debug log of the package logger
```

`<scenario>` is `concurrent` or `sequential`, `<metadata>` is `weights` or `delta`, and
`<agg>` is the aggregation rule. A run without `--seed` writes directly under the sweep
folder; a seeded one adds the `seed_<n>/` level.

## Frozen inputs

These files are read, not written, by every phase. They pin the experimental setting, and
the commands that could produce them reuse them whenever they exist.

| Artefact | Meaning |
| --- | --- |
| `writer_split.json` | the 3% / 97% split of writers into the server set and the client pool |
| `global_model` | state dict of the global model: baseline, teacher and anchor |
| `global_results/global_metrics.json` | curves and test accuracy of the global model |
| `global_results/fisher/{fisher.pt,global_params.pt}` | diagonal Fisher information and the EWC anchor |
| `outliers/clients_acc_on_global.json` | per-client accuracy of the global model |
| `outliers/selected_outliers.json` | the five low-accuracy writers used as participants |
| `global_clients_model`, `global_clients_results/*.json` | the combined reference model behind every plot's reference line |

The published writer split cannot be re-derived (it came from a directory glob whose order
is not reproducible), which is why it is treated as a frozen input rather than a step.

## Per-round series

Every result file stores its series aligned index by index with `accuracies`, and **index
0 is always the untouched global model**, measured before any client has trained, so each
run carries its own reference point.

| Key | Meaning |
| --- | --- |
| `accuracies` | clients on their full data, source writers on their full test set |
| `heldout_client_accuracies` | the pool's held-out 40% (validation + test halves) |
| `pool_val_accuracies` | the validation half — **choose** with this |
| `pool_test_accuracies` | the test half — **report** this |
| `source_val_accuracies` | the source writers' 40% validation split |

The held-out 40% is split once, stratified by label and seeded with 42, so selection and
reporting never share a sample.

**Forgetting signals** (recorded every round, no source data required):
`dist_l2_to_global`, `dist_fisher_to_global`, `dist_fisher_norm_to_global`,
`retention_known`, `agreement_with_global`, `kl_global_to_current`, `proxy_acc`,
`proxy_kl`, plus a `signals_info` block naming the reference, the sample counts, the
Fisher directory and the proxy set. `proxy_acc`/`proxy_kl` are `None` when the provider
defines no proxy set, so the keys exist unconditionally and the values say whether they
mean anything.

**Cost:** `round_seconds`, `client_seconds`, `param_count`, `device`, `seed`, and
communication split by direction — `comm_bytes_down`, `comm_bytes_up` and the totals
`comm_bytes_up_total`, `comm_bytes_down_total`, `comm_bytes_total`. The pre-existing
`comm_bytes_per_round` is their per-round sum and is unchanged.

**Client population** keys (`client_accuracies`, `participants`, `participation`,
`policy`, `sampler_seed`, `pool_size`, `stopped_early`, …) appear only when a run actually
deviates from the published loop, i.e. under a selection policy, partial participation or
`--track-clients`.

## The `config` block

Every `summary_*.json`, `accuracies_*.json` and `config_points_*.json` carries one. It
records the trainer class and the hyperparameters actually used (`T`, `alpha`,
`ewc_lambda`, `lambda_prox`, `lambda_consis`, `beta`, `learning_rate`, `weight_decay`),
the scenario, metadata and aggregation name, the batch size, epochs and rounds, the seed,
the outlier file and the writer ids it resolved to, the global model path with its
SHA-256, the Fisher directory, the proxy set with its content hash, torch and CUDA
versions, device and GPU name, Python version, git commit, hostname, and the start and end
timestamps.

Two runs can therefore never be confused on the strength of their file names alone: the
block says which code, which frozen inputs and which budget produced the numbers.

## Compatibility

Keys added after the earlier version of the study are additive. `accuracies`,
`global_clients_all_metrics_acc` and `global_clients_metric_acc` keep their names and
formats, so the selectors, plotting scripts and stored artefacts of the earlier version
continue to work against a tree produced by this repository.
