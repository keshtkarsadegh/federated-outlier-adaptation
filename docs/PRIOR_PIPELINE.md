> **This document describes the PRIOR pipeline**, not the study this repository
> now reports. It is kept because the code it documents is still present
> (`foa matrix`, the 128 px cache, the Shakespeare and CIFAR-10 providers) and
> because the prior study's outputs are archived in `results_prior_study.tar.gz`.
>
> For the live study — NIST SD19 digits, `Digits_study01` — start at the
> repository README, then `docs/DATA.md`, `docs/RUNBOOK.md`, `docs/VERIFY.md`.
>
> Two statements below are actively wrong for the live study and are corrected
> here: `foa prepare-data` must be given `--resolution 28` (it defaults to 128),
> and the cohort is ten writers cut from the shipped model's ranking, not the
> five frozen writers named below.

# federated-outlier-adaptation — Knowledge-preserving adaptation to outlier clients in federated learning (reproducible pipeline: NIST SD19 by writer, LEAF Shakespeare, CIFAR-10)

This repository implements a **modular framework for adapting a strong global model to
outlier clients** on the [NIST Special Database of Handwritten Characters](https://www.nist.gov/srd/nist-special-database-19),
[LEAF Shakespeare](https://leaf.cmu.edu/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

The project tackles the problem of **adapting strong global models to new or outlier clients without destroying global knowledge**. It systematically compares:

- **Communication scheduling:** concurrent vs sequential
- **Aggregation forms:** full weights vs model deltas, client-only vs client+global
- **Regularization:** FedProx, Elastic Weight Consolidation (EWC), knowledge distillation, selective layer freezing, logit consistency
- **Loss designs:** cross-entropy vs knowledge distillation

A strong global baseline (≈ 99.6% test accuracy) is used, and experiments quantify trade-offs between **stability** (knowledge preservation) and **plasticity** (client adaptation).

---

## Relation to the previous repository

The earlier public repository
[`federated_adaptive_learning_nist`](https://github.com/keshtkarsadegh/federated_adaptive_learning_nist)
accompanies the earlier version of the study. **This repository is a reorganised,
extended and fully reproducible successor** of it: the notebook-driven scripts became an
installable package with a single command-line entry point, the protocol was extended to
two further datasets, multi-seed job matrices and a statistical reporting layer, and
every run records a provenance block.

The published artefacts of the earlier version are kept under `results/` **unchanged**,
so the numbers of the earlier version remain reproducible from this checkout. What moved:

| Earlier version | This repository |
| --- | --- |
| import root `federated_adaptive_learning_nist` | `federated_outlier_adaptation` |
| console script `fal` | `foa` (`fal` stays as an alias for one release) |
| environment variables `FAL_*` | `FOA_*` (the `FAL_*` names are still read as a fallback) |

---

## Table of contents

1. [Installation](#1-installation)
2. [Data preparation](#2-data-preparation)
3. [Frozen artefacts](#3-frozen-artefacts)
4. [Running the pipeline](#4-running-the-pipeline)
5. [Output layout and provenance](#5-output-layout-and-provenance)
6. [Regenerating figures and tables](#6-regenerating-figures-and-tables)
7. [Running on Slurm](#7-running-on-slurm)
8. [Tests](#8-tests)
9. [Repository structure](#9-repository-structure)
10. [Configuration reference](#10-configuration-reference)
11. [Known issues in the published scripts](#11-known-issues-in-the-published-scripts)
12. [Additional datasets](#12-additional-datasets)

Two condensed companions live in `docs/`: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
(the end-to-end command sequence, locally and on Slurm) and
[`docs/RESULTS_LAYOUT.md`](docs/RESULTS_LAYOUT.md) (what a results tree contains).

---

## 1. Installation

Python 3.12 is required. The package installs as `federated_outlier_adaptation` and
provides the `foa` console script.

**pip**

```bash
git clone https://github.com/paymankeshtkaruni/federated-outlier-adaptation.git
cd federated-outlier-adaptation

python -m venv .venv && source .venv/bin/activate

# GPU (CUDA 12.1) build used for the published runs:
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install -e . --no-deps
```

On a CPU-only machine, skip the PyTorch index line; `pip install -r requirements.txt`
installs the CPU wheels of the same versions.

**conda**

```bash
conda env create -f environment.yml -p ./envs/foa
conda activate ./envs/foa
pip install -e . --no-deps
```

Check the installation:

```bash
foa --help
python -m federated_outlier_adaptation.cli --help
```

---

## 2. Data preparation

The experiments use the writer-partitioned subset `by_write` of NIST SD19.

> **The archive is never extracted.** `by_write.zip` contains roughly 400 000 PNG
> files; unpacking it exhausts the inode quota of most shared and cluster file
> systems. Every code path in this repository reads the archive in place with
> `zipfile` and stores the decoded images in a single `.npy` file. Do not run
> `unzip` on it, not even into a scratch or temporary directory.

**Step 1 - download the archive** (568 MB, one file):

```bash
curl -o data/nist/by_write.zip https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
```

or

```bash
python -c "from federated_outlier_adaptation.data.download import download_nist_archive; download_nist_archive()"
```

**Step 2 - build the packed cache** (one command, no extraction):

```bash
foa prepare-data --zip data/nist/by_write.zip
```

This streams every image referenced by the manifest `data/by_write/digits_labels.json`
out of the archive and writes two files into `$FOA_CACHE_DIR` (default `data/cache`):

| File | Content |
| --- | --- |
| `nist_digits_u8.npy` | `uint8` array `[N, 128, 128]`, or packed bits `[N, 2048]` when every image is bilevel (the usual case for SD19) |
| `nist_digits_index.json` | `paths`, `labels`, `writers`, `packed`, `shape`, `missing` |

Row `i` corresponds to `paths[i]`; the row order follows the manifest order.

The dataset layer picks the cache up automatically. If the cache is absent it falls
back to reading individual PNG files from `data/by_write/`, which requires the archive
to have been unpacked elsewhere and is only kept as a compatibility path. Both readers
produce bit-identical tensors (`convert("L")` -> `Resize(128)` -> `ToTensor` ->
`Normalize(0.5, 0.5)`); this is asserted by `tests/test_nist_cache.py`.

The cache is opened lazily and exactly once per dataset instance. Because the default
provider is a process-wide singleton and the drivers fan their jobs out over threads,
that open is guarded: a second thread arriving mid-open waits for the cache instead of
seeing `None` and quietly building a PNG-backed loader that could not read anything.
Whenever the PNG reader *is* chosen while a cache exists, the log says so, so a partial
cache is diagnosed at build time rather than as a `FileNotFoundError` in the first batch.

**Regenerating the manifest** (only needed if you rebuild from the SD19 checksum logs):

```bash
python -m federated_outlier_adaptation.data.labels
```

---

## 3. Frozen artefacts

The following files under `results/` are inputs, not outputs. They define the exact
experimental setting of the published results and are reused by every phase:

| Artefact | Meaning |
| --- | --- |
| `results/writer_split.json` | The 3% / 97% split of writers into the *global* (server) set and the *local* (client) pool. |
| `results/global_model` | State dict of the trained global model, used as the baseline and as the teacher/anchor of every regularised trainer. |
| `results/global_results/global_metrics.json` | Train/validation curves and the test accuracy of the global model. |
| `results/global_results/fisher/{fisher.pt,global_params.pt}` | Diagonal Fisher information and the anchor parameters used by EWC. |
| `results/outliers/clients_acc_on_global.json` | Per-client accuracy of the global model over all local writers. |
| `results/outliers/selected_outliers.json` | The five low-accuracy writers used as federated participants: `f3503_07`, `f2169_69`, `f2307_62`, `f3784_14`, `f2248_68`. |
| `results/global_clients_model`, `results/global_clients_results/*.json` | The combined model (global writers + outliers) that provides the reference lines of every plot. |

`foa global-train` and `foa select-outliers` **reuse these files when they exist** and
only regenerate them when they are missing (or with `--force`). Regeneration emits a
warning: a freshly drawn writer split cannot be expected to reproduce the published one
bit for bit, so downstream numbers would drift away from the paper.

---

## 4. Running the pipeline

All commands accept `--results-dir`, `--data-dir` and `--cache-dir` to redirect I/O.

| Phase | Command | Paper section |
| --- | --- | --- |
| 1. Global model, Fisher, outliers | `foa global-train` | Global baseline, outlier identification |
| 1b. Federated global model | `foa global-train-fl` | Alternative starting point |
| 2. Combined reference model | `foa combined-train` | Upper-bound reference |
| 3. Baseline federated runs | `foa base-fl` | Communication scheduling and aggregation form |
| 4. All aggregation variants | `foa all-aggs` | Aggregation comparison |
| 5. Hyperparameter sweeps | `foa grid --method {ewc,prox,kd,logit,aligned,kd_ewc,feature,freeze,ntd,anchored}` | Regularisation grid searches |
| 6. Final runs per method | `foa final --trainer <Trainer>` | Final comparison |
| 7. Extreme cases | `foa grid --method extreme --case <case>` then `foa extreme --case <case>` | Minimal-client scenarios |
| 8. Job matrices | `foa matrix --plan <plan> --out tasks.txt` | Multi-seed and population studies |
| 9. Reference points | `foa local-finetune` | No-federation lower/upper bound |
| 10. Configuration choice | `foa select --eps 0.005` | Constrained, validation-based selection |
| 11. Signal analysis | `foa signals --root <results>` | Decisions without source data |
| 12. Tables and figures | `foa report --root <results> --out report/` | The whole evaluation protocol |

### Trainers

| Class | Objective | Default hyperparameters |
| --- | --- | --- |
| `BaseTrainer` | cross-entropy | lr 1e-3, weight decay 1e-4, `early_stopping=False`, `patience=5` |
| `EWCTrainer` | CE + Fisher penalty | `ewc_lambda=8.0` |
| `DistillationTrainer` | CE + KD against the global model | `T=8.0`, `alpha=0.95` |
| `DistillationEWCTrainer` | CE + KD + Fisher penalty | `T=8.0`, `alpha=0.95`, `ewc_lambda=8.0` |
| `CFProxTrainer` | CE + proximal term | `lambda_prox=0.9` |
| `CFLogitConsistencyTrainer` | CE + logit L2 to the frozen global model | `lambda_consis=0.1` |
| `CFAlignedFeatureTrainer` | CE + logit alignment to the teacher | `beta=0.1` |
| `FeatureAlignmentTrainer` | CE + MSE on the penultimate representation | `beta=0.1` |
| `FreezeTrainer` | CE with part of the network frozen | `scope="body"`, lr 1e-3 |
| `DistillationFreezeTrainer` | KD with part of the network frozen | `scope="body"`, `T=8.0`, `alpha=0.95` |
| `NTDTrainer` | CE + not-true distillation | `beta=1.0`, `tau=1.0` |
| `AnchoredTrainer` | CE + one anchored penalty (the whole family) | `space="kd"`, `anchor="frozen"`, `lam=1.0`, `T=4.0` |

`CFAlignedFeatureTrainer` and `FeatureAlignmentTrainer` differ in *where* they
align: the former matches pre-softmax logits (the published objective, kept
untouched), the latter the 128-dimensional post-`fc1` representation.

**Model contract.** The alignment and freezing objectives need two accessors on
the dataset model, both implemented by `FlexibleCNN`:

| Accessor | Meaning |
| --- | --- |
| `penultimate(x)` | post-`fc1` ReLU activation, before dropout |
| `forward(x, return_features=False)` | logits, or `(logits, features)` when asked |
| `features`, `fc1` | the sub-modules `scope="conv"` / `scope="body"` freeze |

`forward(x)` returns exactly what it returned before, so every existing run is
unaffected. A model for another dataset only has to provide `penultimate(x)`;
the single-pass `return_features` variant is used when it exists.

**The early-stopped FedAvg arm.** Every regularised trainer in the table stops a
client update as soon as the validation accuracy has not improved for `patience`
epochs, while the published baseline runs all of them and keeps the
best-validation model. A comparison between them therefore mixes two effects:
the objective and the local budget. `BaseTrainer` takes `early_stopping=True`
(and `patience`, default 5) to adopt exactly the same rule, so an early-stopped
FedAvg arm isolates the objective:

```bash
foa final --trainer BaseTrainer --parent lean_m10_uniform_fedavg_es \
    --set early_stopping=true --aggregation fedavg --seed 1
```

The default is `False`, i.e. the published behaviour, and the flag is recorded
in the `config` block under `hyperparameters` and `trainer_kwargs`. The lean
plans emit this arm next to the plain FedAvg one.

**Layer freezing** clears `requires_grad` on the frozen part, builds the
optimizer over the trainable parameters only (so weight decay never touches the
frozen weights) and keeps normalisation layers inside the frozen part in `eval`
mode, so their running statistics do not move either. All three survive the
deep copy the runners perform every round.

### Hyperparameter overrides

Trainers keep all-default constructors (the registry instantiates them by name), so the
defaults above always reproduce the published configuration. Individual values are
overridden per run without editing `constants.py`:

```bash
foa final --trainer DistillationTrainer --set T=4 alpha=0.95
foa final --trainer EWCTrainer --trainer-kwargs '{"ewc_lambda": 20}'
```

`--set NAME=VALUE` parses the value as JSON (so `T=4` is an int and `alpha=0.95` a
float) and falls back to a plain string. The effective values are written into the
`config` block of every result file.

### Round, epoch and batch-size budget

`foa final`, `foa extreme` and `foa grid` take the three budget knobs of a run
directly. All three default to *unset*, which leaves the published budget in
place, so an existing command line is unaffected:

| Option | `final` / `extreme` default | `grid` default |
| --- | --- | --- |
| `--rounds R` | 100 federated rounds | 50 |
| `--epochs E` | 100 local epochs per round | 50 |
| `--batch-size B` | 64 | 512 |

```bash
foa final --trainer BaseTrainer --rounds 2 --epochs 1 --parent quick_check
foa grid --method anchored --space kd --anchor frozen --rounds 5 --epochs 2
```

The effective values land in the `config` block as `max_round`, `epochs` and
`batch_size`, next to every other run parameter, so a shortened run is never
mistaken for a published one. `foa grid` additionally takes `--parent FOLDER`,
which redirects the sweep's output directory (default: the sweep's own folder),
and `foa local-finetune` takes the same `--parent` (default `local_finetune`).

### Evaluation cost: `--eval-path` and `--insample-every`

Every round scores the current global model on the same fixed sets: the source
split, the pool's in-sample data, the pool's held-out validation and test
halves, every client on its own data, and the server-side proxy set. In the main
NIST setting that is roughly 55 000 images per round. The published loop rebuilt
them batch by batch through single-process `DataLoader`s, decoding one image at
a time from the packed cache on the CPU, which dominates the wall clock as soon
as the client updates run on an accelerator.

`--eval-path cache` (the **default**) materialises each set **once** per run in
its compact storage form (`uint8` images, `int64` token windows), keeps it on
the accelerator when there is room, and turns a round into one batched forward
pass per set. Everything derived from a set - the pooled accuracy, the
per-client accuracies, and the `retention_known` / `agreement_with_global` /
`kl_global_to_current` / `proxy_kl` signals - reads that one pass.

`--eval-path loader` keeps the published per-round loader passes. The two paths
produce the same numbers, which is the point: the batched decode is the
dataset's own transform applied to a whole batch (the resize is done on the
`uint8` image by the very same `Resize` object, and `ToTensor` + `Normalize` are
elementwise), and the pooled numbers are still the sample-count weighted mean of
per-client accuracies. A dataset that cannot be materialised - an augmenting
one, a non-standard transform - simply keeps its loader for that set, so the
cache can never change what a run computes.

```bash
# the two paths, same configuration, separate result roots
foa final --trainer DistillationTrainer --parent speedtest_pool_m10_uniform_fedavg \
    --pool-frac 0.05 --clients-per-round 10 --policy uniform --sampler-seed 1 \
    --track-clients --aggregation fedavg --seed 1 --rounds 20 --eval-path loader
foa final ... --eval-path cache

# element-by-element comparison plus the per-round timings
python -m federated_outlier_adaptation.analysis.eval_equality \
    --loader $SPEED/loader --cache $SPEED/cache
```

`slurm/run_speed_check.sbatch` runs exactly that pair and the comparison.

`--insample-every N` (default `1`, unchanged) measures the pool's *in-sample*
metrics - `accuracies[r][0]` and the per-client tracking - every `N`-th round
and in the last round; every other series stays per round. The series keep their
length and their alignment with `accuracies`; the entries of a skipped round
repeat the most recent measurement, and the additive keys `insample_every` and
`insample_rounds` record which rounds were measured. With the default neither
key appears.

Which sets a run materialised is recorded under `evaluation` in the result
payload, and `eval_path` / `insample_every` in the `config` block.

### Extreme cases

The extreme-case runs restrict the participant set to a single low-accuracy writer and
use the knowledge-distillation trainer:

```bash
# grid search over T x alpha for one case
foa grid --method extreme --case single

# final runs with the configuration reported in the paper
foa extreme --case single    # T=50,  alpha=0.98, ["f3503_07"]
foa extreme --case double    # T=8,   alpha=0.95, ["f3503_07", "f2307_62"]
foa extreme --case dual      # T=8,   alpha=0.95, ["f3503_07", "f3503_07"]
```

`foa extreme --case <case>` applies the client set and hyperparameters from
`constants.EXTREME_CASE_CLIENTS` / `constants.EXTREME_CASE_HYPERPARAMETERS`; both can be
overridden with `--set`.

**The `double` and `dual` cases changed.** In the published pipeline the client
restriction was an *intersection* with the selected-outlier list, so a repeated
writer id contributed a single participant and both cases behaved exactly like
`single`. A repeated id now yields one participant per occurrence - each with
its own local training run, its own sample count and its own entry in the
aggregation - while the client evaluation loader is still built over the unique
ids. `dual` is therefore a true replication of `f3503_07`, and `double` now
uses two distinct low-accuracy writers. Both write into new output folders:

| Case | Output tag | Stored (single-participant) results |
| --- | --- | --- |
| `single` | `single_outlier_...` | unchanged, still reproduced |
| `double` | `double_writers_...` | `double_outlier_...` kept, never overwritten |
| `dual` | `dual_replicated_...` | `dual_outlier_...` kept, never overwritten |

The stored `dual_outlier_*` and `double_outlier_*` folders under `results/`
therefore document single-participant runs, not the cases their names suggest.
Restrictions without duplicates keep the previous `provider.restrict`
semantics unchanged.

### Client population and per-round selection

Larger populations and partial participation are configured on `foa final` and
`foa extreme`. The defaults reproduce the published loop exactly (every client
trains in every round):

| Option | Default | Meaning |
| --- | --- | --- |
| `--outliers-file` | provider default | explicit client pool file |
| `--pool-frac F` | - | draw from the provider's rule-based pool (bottom fraction F) |
| `--participation C` | `1.0` | fraction of the pool training per round |
| `--policy` | `all` | `all`, `uniform`, `worst_first`, `round_robin` |
| `--sampler-seed` | `None` | seed of the selection RNG |
| `--track-clients` | off | evaluate every pool client on its own data each round |
| `--clients-per-round M` | - | absolute number of clients per round; wins over `--participation` |
| `--stop-when-global-below-clients` | off | stop once the global model falls below the clients accuracy or below 0.90 |
| `--client-order` | `fixed` | cyclic visiting order; `shuffle` permutes per round |
| `--init` | `global` | start from the global model, or from a fresh one (`scratch`) |

`worst_first` picks the clients with the lowest most recent accuracy on their
own data; unknown accuracies and ties are broken randomly with the sampler
seed. FedAvg-style weights are `n_k / N` over the participants *of the round*,
and the sequential incremental/progressive rules use the position within the
round.

**The pool is defined by a rule, not a list.** The main experimental setting
draws its participants from the bottom fraction of the non-contributor writers
under the frozen global model, so the population is reproducible from the
accuracy file alone:

```bash
foa select-outliers --mode pool --pool-frac 0.05       # outlier_pool_frac0.05.json
foa select-outliers --mode pool --pool-threshold 0.9   # outlier_pool_thr0.9.json
foa global-train --pool-frac 0.05                      # write it as part of phase 1
```

Every dataset writes these files through the same helper
(`outliers/client_accuracy.py: write_pool`) and under the same names, and every
provider exposes `pool_path(frac)` and `outlier_pool(frac)`. The payload lists
the client ids worst-first together with their baseline accuracies, the rule
that produced it and the population it was drawn from.

A run draws from the pool either by rule or by path:

```bash
foa final --trainer BaseTrainer --pool-frac 0.05 --clients-per-round 10 --policy uniform
foa final --trainer BaseTrainer --outliers-file results/outliers/outlier_pool_frac0.05.json
```

`--pool-frac` is the portable form - it resolves through the provider, so the
same task line works for every dataset - and an explicit `--outliers-file`
overrides it. Plain list files (`selected_outliers*.json`) and rule-based pool
files are both accepted.

The published five-writer cohort remains available and is the default, so every
existing command is unaffected; it is kept as the `fixed_cohort` continuity
ablation.

Deterministic pools of the `k` worst clients:

```bash
foa select-outliers --mode lowest --k 20      # results/outliers/selected_outliers_k20_lowest.json
foa select-outliers --mode lowest --k 50

foa final --trainer BaseTrainer \
    --outliers-file results/outliers/selected_outliers_k50_lowest.json \
    --participation 0.2 --policy worst_first --sampler-seed 1 --track-clients \
    --parent e6_k50lowest_c02_worst_first --seed 1
```

`--mode sample` (the default) keeps the published random draw from the bottom
`--bottom-frac`; `k=5` in that mode keeps the frozen `selected_outliers.json`.

### Aggregation rules

Every aggregation rule of this repository is one of two shapes.

**Parallel (concurrent) rules** take one server step towards a combination of
the client updates, optionally pulling back towards the frozen global model:

```
theta_{t+1} = theta_t + eta_s * A({Delta_k}, {p_k}) - lambda_s (theta_t - theta_g)
Delta_k     = theta_k - theta_t
```

with the server step size `eta_s`, an aggregator `A`, the client weights `p_k`
and the anchor pull `lambda_s`.

**Cyclic (sequential) rules** fold one client at a time into the global model:

```
theta <- (1 - alpha_i) theta + alpha_i theta_k
```

with a mixing schedule `alpha_i` that depends on the client's position `i` in
the round.

#### The published rules in that parameterisation

| Rule | eta_s | p_k | lambda_s |
| --- | --- | --- | --- |
| `con_weighted_cw` | 1.0 | proportional `n_k/N` | 0 |
| `con_weighted_cgw` | 0.5 | proportional | 0 |
| `con_scaled_cw` | 1.0 | uniform `1/K` | 0 |
| `con_scaled_cgw` | 0.5 | uniform | 0 |
| `con_capped_cw` | 1.0 | capped at `1/K`, renormalised | 0 |
| `con_capped_cgw` | 0.5 | capped | 0 |
| `con_delta_weighted_cgd` | 1.0 | proportional | 0 |
| `con_delta_scaled_cgd` | 1.0 | uniform | 0 |
| `con_delta_capped_cgd` | 1.0 | capped | 0 |

| Cyclic rule | alpha_i |
| --- | --- |
| `seq_fixed_ratio_update` | 0.3 (constant) |
| `seq_equal_update` | `1 / (i + 1)` - incremental averaging |
| `seq_fedavg_update` | `n_k / N` |
| `seq_incremental_update` | `i / (K + i)` - progressive |
| `seq_delta_fedavg_update` | `n_k / N`, written in delta form |
| `seq_delta_scaled` | `1 / K` |
| `seq_delta_capped` | `min(n_k/N, 1/K)` |
| `seq_delta_progressive_update` | `i / (K + i)`, written in delta form |

#### Algebraically identical pairs

Writing a rule in weight form or in delta form does not change it. The
following pairs compute the same update and are asserted to be bit-comparable
on random tensors in `tests/test_aggregation_rules.py`:

| Weight form | Delta form | Why |
| --- | --- | --- |
| `con_weighted_cw` | `con_delta_weighted_cgd` | `theta + sum p_k (theta_k - theta) = sum p_k theta_k` since `sum p_k = 1` |
| `con_scaled_cw` | `con_delta_scaled_cgd` | same, with `p_k = 1/K` |
| `con_capped_cw` | `con_delta_capped_cgd` | same, with the capped and renormalised `p_k` |
| `seq_fedavg_update` | `seq_delta_fedavg_update` | `(1-a) theta + a theta_k = theta + a (theta_k - theta)` |
| `seq_incremental_update` | `seq_delta_progressive_update` | same identity with `a = i/(K+i)` |

The paper reports these as separate variants; they are separate *names*, not
separate methods, and any difference between their stored numbers comes from
run-to-run nondeterminism alone.

#### Extended rules (opt-in)

The rules below are excluded from every default sweep - `list_methods()` hides
them, so `all-aggs` and the grid searches keep the published method set. Enable
them with `--extended-aggregations`, or name one directly with
`--aggregation <rule>`. They are grouped by the hypothesis they test.

**H1 - server step size.** `theta + eta_s sum_k p_k Delta_k`:
`con_delta_eta025`, `con_delta_eta05`, `con_delta_eta1`. By construction
`eta1` equals FedAvg (`con_weighted_cw`) and `eta05` equals the anchored
published rule (`con_weighted_cgw`).

**H2 - client weighting.** `--weighting proportional|uniform|capped` chooses
`p_k` for every extended rule; the three schemes are the ones the published
rules hard-code.

**H3 - robust aggregation.** `con_delta_median` takes the coordinate-wise
median of the client updates, `con_delta_trimmed_mean` drops the top and bottom
20% per coordinate before averaging. Both then apply `eta_s`.

**H4 - server-side anchoring.**
`theta + eta_s sum_k p_k Delta_k - lambda_s (theta - theta_g)` with the frozen
global model `theta_g`: `con_delta_anchor_lam01` (`lambda_s = 0.1`) and
`con_delta_anchor_lam05` (`0.5`). The runner stores `theta_g` on the server
state before the first round.

**H5 - server optimisers.** `con_delta_fedavgm` (server momentum 0.9, step
1.0), `con_delta_fedadam` and `con_delta_fedyogi` (Reddi et al. 2021; server
lr `1e-2`, `b1 = 0.9`, `b2 = 0.99`, `tau = 1e-3`, pseudo-gradient `-Delta`).
These carry state across rounds, which the stateless rules do not: the runner
owns a `ServerState` and passes it to any rule whose signature accepts
`server_state`.

**H6 - cyclic mixing schedules.** The four `alpha_i` schemes above; unchanged.

**H7 - cyclic order.** `--client-order fixed|shuffle`. `fixed` (default) visits
the round's participants in the sampler's order; `shuffle` draws a fresh seeded
permutation each round, so the position-dependent schedules stop favouring the
same client.

#### Choosing rules per run

`--aggregation` accepts a variant key or a concrete rule name:

| Value | Meaning |
| --- | --- |
| omitted / `all` | every rule of each family (the published behaviour) |
| `fedavg` | the FedAvg-equivalent rule of each family |
| `anchored` | the half-step (`_cgw`) rule where the family has one |
| `capped` | the capped rule of each family |
| a rule name | only the families that contain that rule are run |

Give such runs their own `--parent` so they do not overwrite the full-family
results.

### Regularisation as one family

`AnchoredTrainer` writes every regularised objective as

```
Loss = CE(f_theta(x), y) + lam * D_space(theta ; anchor)
```

so the family can be swept with one grid. `space` selects where the distance is
measured (`param_l2`, `fisher`, `fisher_scaled`, `logit_l2`, `kd`, `ntd`,
`feature_l2`, `kd+fisher`) and `anchor` selects what it is measured against:
`frozen` (the global model, what every published trainer uses) or `current`
(the model the client received this round, i.e. the true FedProx formulation).

```bash
foa grid --method anchored --space kd --anchor frozen
foa final --trainer AnchoredTrainer --set space="fisher" anchor="current" lam=1.0
```

The module docstring of `trainers/anchored_trainer.py` lists which published
trainer each configuration reproduces; `tests/test_anchored_trainer.py` asserts
those equivalences on random batches. The published trainers themselves are
untouched and remain the default way to reproduce the paper.

### Selecting a configuration

The legacy ranking `4 * clients_acc + 10 * global_acc^2` is kept unchanged. The
principled alternative maximises adaptation subject to a bound on forgetting,
decided on validation data only:

```bash
foa select --root results --eps 0.005          # 0.5 percentage points of source accuracy
```

It maximises the mean clients' held-out accuracy over the last five rounds
subject to `source_val_acc(theta_g) - source_val_acc <= eps`, where the
reference is the run's own round-0 value. It writes
`constrained_selection.json` (winner plus every record) and `pareto_front.csv`
(the adaptation/forgetting front).

**The regularisation family's finals are chosen this way.** Each
`(space, anchor)` sweep has its own lambda (and, for the softmax-based spaces,
its own temperature), and the value is only known once the sweep has finished.
`foa select --emit-finals` runs the constrained selection over every stored
`anchored_<space>_<anchor>_grid_search` folder, at every requested budget, and
writes both the decision and the runs it implies:

```bash
foa select --root $FOA_RESULTS_DIR --emit-finals \
    --eps 0.0025 0.005 0.01 --seeds 1 2 3 \
    --out $FOA_PROJECT_DIR/jobs/tasks_reg_finals.txt
```

Each sweep folder gains one `constrained_selection_eps<eps>.json` per budget -
the winner, its `lam`/`T`, the feasible count and the front - and the task file
holds one `foa final --trainer AnchoredTrainer ... --set space=.. anchor=..
lam=<selected> [T=<selected>] --parent reg_<space>_<anchor>_eps<eps>` line per
selection and seed. A sweep that a budget admitted nothing from contributes no
line, so a final never silently runs the trainer defaults.

`foa matrix --plan regularisation_family` therefore has two shapes. Given a
results root that already holds those selection files it emits the sweeps
followed by the selected finals; without them it emits the sweeps followed by
the single `foa select --emit-finals` line above, which writes
`tasks_reg_finals.txt` next to the results root for a second job array:

```bash
foa matrix --plan regularisation_family --results-dir $FOA_RESULTS_DIR --out tasks_reg.txt
# ... run the sweeps, then the last line of tasks_reg.txt, then:
sbatch --array=1-$(wc -l < $FOA_RESULTS_DIR/tasks_reg_finals.txt)%8 \
       slurm/run_matrix.sbatch $FOA_RESULTS_DIR/tasks_reg_finals.txt
```

### Decisions without source data

The setting of this study is that the source population is **gone**: once the
global model has been shipped, the writers it was trained on cannot be read
again. Every source number this repository stores - `accuracies[r][1]` on the
source test set, `source_val_accuracies` on the source validation split - is
therefore an **evaluation** quantity, reported so that the trade-off can be
measured. A *method* that used one of them to decide when to stop or which
configuration to keep would be reading data the constraint forbids. The
published stop rule `--stop-when-global-below-clients` is exactly such a rule
and is kept only as an **oracle stopping** reference.

**What the constraint does allow.** Three sources of evidence stay inside it:
the parameters of the model itself, the clients' own held-out data, and public
data the server owns. Every run therefore logs, in both runners and in every
round, the following additive keys next to its accuracies:

| Key | What it measures | Needs |
| --- | --- | --- |
| `dist_l2_to_global` | `‖θ_t − θ_g‖₂` over the float parameters | nothing |
| `dist_fisher_to_global` | `Σᵢ Fᵢ (θ_t,ᵢ − θ_g,ᵢ)²` with the shipped Fisher | nothing |
| `dist_fisher_norm_to_global` | the same with `F` normalised to unit mass | nothing |
| `retention_known` | accuracy of `θ_t` on the samples `θ_g` got right | clients' held-out |
| `agreement_with_global` | fraction with `argmax θ_t == argmax θ_g` | clients' held-out |
| `kl_global_to_current` | mean `KL(softmax θ_g ‖ softmax θ_t)` | clients' held-out |
| `proxy_acc` | accuracy on the server-side proxy set | proxy set |
| `proxy_kl` | the same mean KL on the proxy set | proxy set |

`θ_g` is the model the run started from, the same reference the anchored server
rule pulls back towards. The three client-side signals are computed on the
**validation half** of the pool's held-out split - data the clients never train
on. All eight series are index-aligned with `accuracies`, so index 0 is again
the untouched global model, and a run that stops early keeps them aligned.
`signals_info` records the reference, the sample counts, the Fisher directory
and the proxy-set description with its content hash; the same description is
written into the `config` block under `proxy_set`.

The cost is bounded: `θ_g`'s predicted class, class probabilities and entropy
are computed **once per run** on both reference sets and cached, so a round adds
one forward pass over the pool's validation halves and one over the proxy set.
The two Fisher distances read the same `fisher.pt` the EWC objective uses; a run
whose Fisher directory is absent reports `None` for them.

**The proxy-set assumption.** A server that wants to watch how much of the
source task a model still solves needs data of its own. The assumption is that
it has *some* public data of the same task - not the source clients' data, and
not the participants' data. Each provider realises that differently:

| Provider | Proxy set | Size |
| --- | --- | --- |
| `nist` | the public **MNIST test set**, inverted to the NIST polarity (black ink on white) and resized to 128×128 with the unmodified NIST transform | 10 000 images |
| `cifar10` | CIFAR-10 evaluation images **reserved by the partition** (`prepare --proxy-size 1000`), so they belong to no client | 1 000 images |
| `shakespeare` | 20 held-out roles carved out with seed 42 and **removed from the eligible population**, so they can never be participants | 2 000 sequences |

MNIST comes from the same NIST special databases as SD19 but from a different
writer population and with a different rendering, which is what makes it a
proxy rather than a second sample of the source set. Its polarity is the
opposite of SD19's (white strokes on black, against SD19's black ink on a white
background whose every border pixel is 255), so the proxy images are inverted
before the transform. Build it once, from the four `idx` archives, which are
read in memory and never extracted:

```bash
mkdir -p data/mnist && cd data/mnist
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
    curl -O https://ossci-datasets.s3.amazonaws.com/mnist/$f
done
cd -

foa prepare-data --dataset mnist --raw-dir data/mnist
```

That writes `data/mnist/mnist.npz` and `data/mnist/mnist_index.json`
(`FOA_MNIST_DIR` relocates both). **A checkout without them behaves exactly as
before:** the proxy loader is absent, `proxy_acc` and `proxy_kl` are `None`, and
nothing else changes. The same holds for a CIFAR-10 partition prepared without
`--proxy-size` and for a corpus too small for the Shakespeare carve-out.

**Running the analysis.** `foa signals` reads every run under a results root -
`summary_*.json` from the final and extreme runs, `config_points_*.json` from
the sweeps - and answers three questions:

```bash
foa signals --root results                       # the default budget grid
foa signals --root results --deltas 0.005 0.01 0.05 --eps 0.005
```

| Output | Content |
| --- | --- |
| `signals/signal_correlations.csv` | Pearson and Spearman between each signal and true forgetting, per run and pooled, against both the source validation drop and the source test drop (`target`) |
| `signals/signal_stopping.csv` | for every (run, signal, budget δ): the round the rule stops at and its (adaptation, forgetting), next to oracle stopping (best round by source validation) and the fixed budget (round R) |
| `signals/signal_selection.csv` | the configuration each (signal, δ) selects by held-out adaptation, and the gap to the configuration an oracle would pick under a true-forgetting budget `eps` |
| `signals/signals_pareto_<method>.png` | the (forgetting, adaptation) front the budget grid traces out, per method |
| `signals/signals_summary.json` | all three tables plus the run inventory |

A budget is applied to a signal's **drift**: the rise of a distance or the drop
of an accuracy relative to the run's own round-0 value, so one grid means the
same thing for every signal, and "proxy accuracy dropped by more than δ" is
just the `proxy_acc` row.

### Reference points

Two references bracket every federated number:

```bash
foa local-finetune --trainer BaseTrainer      # each client alone, no aggregation
foa final --trainer BaseTrainer --init scratch  # federation from a fresh model
```

### Federated pre-training of the global model

The published global model is trained centrally on the pooled data of the
server-side writers. `foa global-train-fl` produces an alternative starting
point from the same data with FedAvg (participation 0.1, one local epoch, 100
rounds, proportional weighting, server step 1.0):

```bash
foa global-train-fl --rounds 100 --participation 0.1 --local-epochs 1
foa final --trainer DistillationTrainer --global-name global_fl
```

It writes `global_fl_model`, `global_results/global_fl_metrics.json` and
`global_results/fisher_fl/`, and never touches the centrally trained artefacts.
`--global-name` is available on `base-fl`, `all-aggs`, `grid`, `final` and
`extreme`; the teacher checkpoint and the Fisher directory follow the name.

### Running on another dataset

Every command takes `--provider {nist,shakespeare,cifar10}`; the default `nist`
reproduces the published pipeline unchanged. The provider supplies the client
partition, the model topology and its own results root, so nothing about the
NIST artefacts is touched:

```bash
foa prepare-data --dataset shakespeare --zip data/shakespeare/pg100.txt
foa global-train --provider shakespeare --k-values 5 20 50
foa select-outliers --provider shakespeare --mode pool --pool-frac 0.05
foa final --provider shakespeare --trainer DistillationTrainer --seed 1
```

`foa matrix --plan <plan> --provider shakespeare` replicates any plan on
another dataset: every task line gains `--provider`, and the pool files are
resolved under that dataset's results root. The plan sizes are identical across
datasets, so the GPU-hour estimate of a plan multiplies by the number of
datasets it is run on.

On the cluster, `slurm/prepare_extra_data.sbatch <dataset> <archive>` builds the
packed cache and `slurm/extra_global_train.sbatch <provider>` produces the
global baseline, the client accuracies, the deterministic selections and the
rule-based pool.

The dataset-specific preparation and provider details are documented in section
12.

### Reproducible runs

By default a run is **unseeded**, which reproduces the behaviour and the output paths of
the published experiments (only the dataset split is seeded, with `seed=42`). Passing
`--seed S` seeds `random`, `numpy` and `torch`, gives the DataLoaders a seeded
generator, writes the seed into the result files and inserts a `seed_<S>` component into
the output directory so seeded and unseeded runs never overwrite each other.

```bash
foa final --trainer DistillationTrainer --seed 1234
# -> results/final_result_DistillationTrainer_grid_search/seed_1234/<scenario>_<metadata>/
```

### Alternative outlier sets

```bash
foa select-outliers --k 8            # writes results/outliers/selected_outliers_k8.json
foa final --trainer EWCTrainer --outliers-file results/outliers/selected_outliers_k8.json
```

`k=5` keeps the published file name `selected_outliers.json` and is never overwritten
unless `--force` is given.

### Job matrices

Large studies are Cartesian products of trainers, seeds, client populations and
hyperparameters. `foa matrix` expands a named plan into one `foa` command per
line and prints the task count together with a GPU-hour estimate:

```bash
foa matrix --list                                  # every plan with its size
foa matrix --plan e1_seeds --out tasks_e1.txt
```

| Plan | Content |
| --- | --- |
| `e1_seeds` | the published finals over seeds 1-5, a second KD variant (`T=4`), and the extreme double case |
| `e2_e5_grids` | the feature-alignment and layer-freezing sweeps |
| `e2_e5_finals` | finals for those two trainers with the values in `constants.py` (placeholder until the sweeps have chosen) |
| `e6_pools` | builds the `k=20` / `k=50` pools; run before `e6_population` |
| `e6_population` | the two alternative selection policies on the pool setting |
| `dual` | the truly duplicated client, seeds 1-3 |
| `pool` | builds the rule-based outlier pool of the main setting |
| `fixed_cohort` | the published five-writer cohort, for continuity |
| `pool_confirm_grids` | KD and EWC sweeps repeated on the pool setting |
| `aggregation_hypotheses` | H1-H7 of the aggregation family on the pool |
| `regularisation_family` | every distance space x anchor, then the constrained-selection finals (or the `foa select --emit-finals` line that writes them) |
| `references` | local fine-tuning and a from-scratch federated run |
| `lean_nist_sweeps` | the three regularisation sweeps of the lean protocol: `kd`, `ntd`, `kd+fisher`, frozen anchor, short ranges, one seed, no per-client tracking |
| `lean_nist_finals_pre` | the finals that need no selection: FedAvg and early-stopped FedAvg over five seeds, plus the selection-policy study at `m=5` |
| `lean_nist_reg_finals` | the finals the constrained selection implies at `eps=0.005` (or the single `foa select --emit-finals` line that writes them) |
| `lean_agg_hypotheses` | the aggregation redesign reduced to ten arms x two trainer arms x three seeds |
| `lean_replica` | the whole lean protocol on one further dataset (`--provider shakespeare` / `cifar10`) |
| `smoke` | cheap end-to-end checks of every code path (2 rounds, 1 local epoch) |
| `resume` | pseudo-plan: re-emit only the unfinished lines of `--from TASKFILE` |
| `regen_base` | baseline federated runs (both stopping rules) and the aggregation comparison |
| `regen_grids` | every sweep, including the three extreme cases |
| `regen_finals` | the main pool setting: 3 round sizes x 7 trainers x aggregation variants |
| `regen_extreme` | the three extreme cases, seeds 1-3 |
| `regen_all` | the four `regen_*` plans in dependency order (grids before the finals that consume them) |

The `regen_*` plans regenerate every published phase with the reproducible
pipeline. Point `FOA_RESULTS_DIR` at a **separate results root** holding copies
of the frozen inputs (`writer_split.json`, `global_model`,
`global_results/fisher/`, `outliers/`), so the published tree is never written
to:

```bash
export FOA_RESULTS_DIR=$FOA_PROJECT_DIR/results_v2
mkdir -p $FOA_RESULTS_DIR
cp -r results/writer_split.json results/global_model results/global_results \
      results/global_clients_model results/global_clients_results \
      results/outliers $FOA_RESULTS_DIR/

foa matrix --plan regen_all --out tasks_regen.txt
```

Every generated task carries `--skip-existing`, which returns immediately when
the job's `summary_<i>.json` (finals, extreme cases) or
`accuracies_points_<i>.json` (sweeps) is already on disk. A timed-out job array
can therefore be resubmitted unchanged.

#### The lean protocol

The `lean_*` plans are the reduced version of the study: the same questions, the
arms that actually answer them. Three levers make them cheap - short
hyperparameter ranges (`--lams`, `--temperatures`), one seed per sweep, and no
`--track-clients` on sweeps - and the evaluation cache removes what used to be
the per-round floor. They run in dependency order:

```bash
foa matrix --plan lean_nist_sweeps      --out tasks_lean_sweeps.txt
foa matrix --plan lean_nist_finals_pre  --out tasks_lean_finals.txt
# after the sweeps: writes tasks_reg_finals.txt via `foa select --emit-finals`
foa matrix --plan lean_nist_reg_finals --results-dir $FOA_RESULTS_DIR
foa matrix --plan lean_agg_hypotheses   --out tasks_lean_hyp.txt
foa matrix --plan lean_replica --provider shakespeare --out tasks_lean_shake.txt
```

#### Wave two: resubmitting only what is missing

`--skip-existing` makes a resubmitted array *safe*, but a 600-element array
whose forty timed-out lines are scattered through it still spends 560 queue
slots on tasks that return in seconds. `--plan resume` applies the same rule
ahead of submission and writes a task file of the lines whose expected outputs
are absent:

```bash
foa matrix --plan resume --from tasks_lean_hyp.txt --out tasks_lean_hyp_wave2.txt
N=$(wc -l < tasks_lean_hyp_wave2.txt)
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
       --array=1-$N%8 slurm/run_matrix.sbatch tasks_lean_hyp_wave2.txt
```

A line whose outputs cannot be predicted (`foa select`, `foa signals`,
`foa select-outliers`, `foa global-train-fl`) is always kept.

#### Cost estimates from a measured unit cost

`foa matrix` prints a GPU-hour estimate. By default it assumes a flat
`--minutes-per-task`. Given `--seconds-per-round S` it instead counts the runs
and rounds each line expands into - a `--aggregation fedavg` final is four runs
of 100 rounds, a `kd` sweep with five lambdas and two temperatures is forty runs
of 50 rounds - and prices them at the measured unit cost. Sweeps run at a
different budget (batch 512, 50 local epochs) and take their own
`--sweep-seconds-per-round`:

```bash
foa matrix --plan lean_agg_hypotheses --seconds-per-round 8
foa matrix --plan lean_nist_sweeps --seconds-per-round 8 --sweep-seconds-per-round 25
```

### The smoke plan

`smoke` is the plan to run *before* a real one, and after any change to the
runners, trainers, aggregation rules or providers. Its twenty-one tasks drive
every code path the study uses at two rounds and one local epoch, which costs
minutes rather than GPU-days:

| Task group | What it exercises |
| --- | --- |
| `foa select-outliers --mode pool` | the rule-based pool of the main setting |
| `foa global-train-fl` | federated pre-training, into `smoke_global_fl` |
| `foa local-finetune` | the no-federation reference |
| `foa final`, 9 trainers | every trainer through the four (scenario, metadata) jobs |
| `--aggregation con_delta_anchor_lam01 / con_delta_median` | the opt-in H4 and H3 rules |
| `--client-order shuffle` | the cyclic visiting order (H7) |
| `foa extreme --case dual` | the duplicated-client case |
| `foa grid --method anchored` | the sweep engine |
| `--provider shakespeare`, `--provider cifar10` | the two additional datasets |
| `foa select` | the constrained selection over what the sweep wrote |
| `foa signals` | the forgetting-signal analysis over everything the plan wrote |

The last two tasks only read, and they are appended after the training tasks, so
adding them never renumbers a line of the task file.

Every training task writes under a `smoke_*` parent and carries
`--skip-existing`; the federated pre-training writes `smoke_global_fl_model`
and never the published `global_model`. Point `FOA_RESULTS_DIR` at a scratch
root holding copies of the frozen inputs all the same:

```bash
export FOA_RESULTS_DIR=$FOA_PROJECT_DIR/results_smoke
foa matrix --plan smoke --out $FOA_PROJECT_DIR/jobs/tasks_smoke.txt
```

---

## 5. Output layout and provenance

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

Every `summary_*.json`, `accuracies_*.json` and `config_points_*.json` carries a
**`config` block** recording:

trainer class and the hyperparameters actually used (`T`, `alpha`, `ewc_lambda`,
`lambda_prox`, `lambda_consis`, `beta`, `learning_rate`, `weight_decay`), scenario,
metadata, aggregation name, batch size, epochs, rounds, seed, outlier file and the
writer ids it resolved to, the global model path with its SHA-256, the Fisher directory,
torch and CUDA versions, device and GPU name, Python version, git commit, hostname and
the start/end timestamps.

Alongside it, the same files gained the additive instrumentation keys `round_seconds`,
`client_seconds`, `comm_bytes_per_round` (state-dict bytes x clients x 2), `param_count`,
`device` and `seed`. The pre-existing keys - in particular `accuracies`,
`global_clients_all_metrics_acc` and `global_clients_metric_acc` - keep their names and
formats, so all downstream selectors and plotting scripts continue to work unchanged.

**Evaluation protocol.** The `accuracies` entries score the clients on their
*full* data, which includes the 60% they train on, and the source writers on
their full test set. Both keep their names and values. Alongside them every
round now records, always:

| Key | Meaning |
| --- | --- |
| `heldout_client_accuracies` | the pool's 40% held-out split (val + test halves) |
| `pool_val_accuracies` | the validation half of that split - use this to choose |
| `pool_test_accuracies` | the test half - use this to report |
| `source_val_accuracies` | the source writers' 40% validation split |

The clients' held-out 40% is divided once, stratified by label and seeded with
42, into a validation and a test half, so model selection and the reported
numbers never share a sample. All series are aligned index by index with
`accuracies`, and index 0 is always the untouched global model, measured before
any client has trained - which makes every run carry its own reference point.

**Constraint-respecting signals.** Alongside them every round records the eight
signals of *Decisions without source data* (section 4) - `dist_l2_to_global`,
`dist_fisher_to_global`, `dist_fisher_norm_to_global`, `retention_known`,
`agreement_with_global`, `kl_global_to_current`, `proxy_acc`, `proxy_kl` - plus
the `signals_info` block describing the reference, the sample counts, the Fisher
directory and the proxy set. They are aligned with `accuracies` like every other
series. `proxy_acc` and `proxy_kl` are `None` for a provider that defines no
proxy set, which is the case for any checkout that has not prepared one, so the
keys appear unconditionally and the values say whether they mean anything. The
`config` block gains `proxy_set` with the proxy description and its content
hash.

Communication is reported per round and cumulatively, split by direction:
`comm_bytes_down` (global model to the participants), `comm_bytes_up` (their
updates), and the totals `comm_bytes_up_total`, `comm_bytes_down_total`,
`comm_bytes_total`. The pre-existing `comm_bytes_per_round` is their per-round
sum and is unchanged.

**Client population keys.** These appear only when the run actually deviates
from the published loop, i.e. when a selection policy, partial participation or
`--track-clients` is in effect:

| Key | Content |
| --- | --- |
| `client_accuracies` | round -> `{client: accuracy}` on the client's own data |
| `client_heldout_accuracies` | round -> `{client: accuracy}` on the client's held-out split |
| `participants` | round -> the client ids that trained |
| `participation`, `policy`, `sampler_seed`, `pool_size` | the selection configuration |
| `heldout_participant_accuracies` | round -> the same, restricted to that round's participants |
| `client_val_accuracies`, `client_test_accuracies` | round -> `{client: accuracy}` per split half |
| `stopped_early` | present when `--stop-when-global-below-clients` is used |

The `config` block additionally records `participants` (with multiplicity),
`participation`, `policy`, `sampler_seed`, `track_clients` and
`stop_when_global_below_clients`.

---

## 6. Regenerating figures and tables

Figures are generated from the stored JSON files; no retraining is needed.

```bash
# top-6 ranking per grid search (writes accuracies_points_*.csv) and the heatmaps
foa figures

# individual figures
python -m federated_outlier_adaptation.plotting.grid_heatmaps
python -m federated_outlier_adaptation.plotting.global_results_plot
python -m federated_outlier_adaptation.plotting.outlier_samples
python -m federated_outlier_adaptation.analysis.collect_all_aggs
python -m federated_outlier_adaptation.analysis.all_aggs_heatmap
```

The ranking score used to pick the top configurations of a sweep is
`4 * clients_accuracy + 10 * global_accuracy^2`, evaluated on the row with the best
global accuracy among the last five rounds
(`federated_outlier_adaptation/analysis/top_selector.py`).

The paper-specific one-off scripts that used to live inside `results/` now live in
`federated_outlier_adaptation/analysis/`; each takes the results directory as a
parameter instead of relying on the current working directory.

### Reporting

`foa report` turns a whole results root into the tables and figures of the
manuscript. It retrains nothing: every number comes from the stored JSON, and
the provenance `config` block of each file says which trainer, aggregation rule,
seed, dataset and client population produced it.

```bash
foa report --root $FOA_RESULTS_DIR --out report/
foa report --root $FOA_RESULTS_DIR --out report/ --datasets nist --no-figures
foa report --root $FOA_RESULTS_DIR --out report/ --budgets 0.25 0.5 1.0 --level 0.99
foa report --root $FOA_RESULTS_DIR --out report/ --setting "pool0.05 m5 worst_first"
foa report --root $FOA_RESULTS_DIR --out report/ --setting ""   # every setting
```

**What it reads.** `summary_*.json` and the per-job `accuracies_*.json` of the
final, baseline and extreme runs - under any parent, including `seed_<n>` and
the provider subfolders - the `config_points_*.json` of the sweeps, the frozen
`global_metrics.json` / `global_clients_*` artefacts of every provider, and any
`local_finetune.json`. A per-job file whose summary is missing is read on its
own, so a job array interrupted between the two writes is not lost.

**What it computes.** Per run, at the fixed budget `R` (the last stored round):

| Metric | Meaning |
| --- | --- |
| `A_src` | source test accuracy after adaptation |
| `F` | `A_src(θ_g) − A_src`, in percentage points |
| `A_new` in-sample / val / test | clients' own data, and the two halves of the 40 % they never train on |
| `G` | `A_new_test − A_new_test(θ_g)` on the arm's own client set, in points |
| normalised gain | `G` over the head-room to the combined model, where that model covers the same client set |
| rounds-to-target | first round within 95 % of the final gain |
| worst-round forgetting | the largest drop anywhere in the trajectory |
| last-10-round std | stability of the adaptation curve at the end |
| fairness | mean, worst decile, CoV and fraction improved over the per-client test accuracies, for runs with `--track-clients` |
| cost | mean round seconds, wall clock, MB per round and MB to the target round |
| signals | the final drift of each constraint-respecting signal, and its correlation with true forgetting |

**The reference of `F` and `G` is always the shipped model θ_g**, never a run's
own starting point. `A_src(θ_g)` is the frozen `global_results/global_metrics.json`
of the provider. `A_new_test(θ_g)` has no artefact - it depends on who
participates - so it is read off round 0 of the runs of *that client set* which
started from θ_g; they all share it, the median is used and any disagreement is
reported in `index.md`. `F (val)` uses θ_g on the run's own source validation
split, which is built with the run's loader seed: its own round 0 for a run that
starts from θ_g, the round 0 of the runs sharing its seed otherwise.

For a run initialised from θ_g this is exactly its round 0, so the numbers are
unchanged. For a run that is **not** - `--init scratch`, a locally fine-tuned
model - round 0 is a different model, and reading against it would report how
far the run travelled from that model instead of what the provider lost and
gained: a scratch run then shows a large "gain" and a large negative
"forgetting" and wins any table it is allowed into. Those runs are reference
points, not methods. They are recorded with `kind = reference`, reported in
`references.csv|tex` with their corrected `F`/`G`, and excluded from
`budget_selection`, `pareto`, `paired_vs_fedavg` and the signal correlations.
Accuracies stay fractions; `F`, `G` and the standard deviations are percentage
points.

**How it aggregates.** Runs that differ only in their seed form one *arm*, and
every metric is reported as mean, standard deviation and a 95 % Student-t
interval next to `n`. Each arm is then compared against the **FedAvg baseline of
its own setting** - same provider, scenario, metadata, pool, round size, policy,
round budget **and client set** - seed by seed, with a paired t-test, a Wilcoxon
signed-rank test and Cohen's `d_z`; p-values are Holm-corrected within each
metric and marked `*`/`**`/`***`. Because the setting, not the parent folder,
defines the partner, the aggregation-hypothesis runs pair with the baselines
that live under `pool_m10_uniform_fedavg_*`.

The client set is part of the setting by name, taken from `outlier_writers` (or
`single_outlier`) in the provenance block, and its size appears in the setting
label as `w<n>`; a non-default initialisation appears as `init <name>`. Several
studies of this tree run the same grid over the five selected writers and over
the 5 % pool with everything else at its default, so without the participant
list their seeds would be pooled into one arm and their numbers compared as if
they described the same data.

**Methods are ranked per group.** Most trainers are one method, so the group is
the trainer. `AnchoredTrainer` is a *family*: the penalty space and the anchor
decide what the objective is - a Fisher penalty towards the frozen θ_g and a
not-true-distillation penalty towards the current global model are different
methods - while `lam` and `T` are the strength of one objective. It is therefore
split into `anchored:<space>/<anchor>` (`anchored:kd/frozen`,
`anchored:fisher_scaled/current`, …) everywhere the report ranks methods:
`budget_selection`, `budget_selection_all`, `pareto` (CSV column, `.tex` float
and `on_setting_front`), `paired_vs_fedavg` and `pareto_families.pdf`. The
`group` column is carried through `runs.csv` and `arms.csv` as well, and the
split is declared in one place, `GROUP_HYPERPARAMETERS`.

`regularisation_family.csv|tex` reports that family for the headline setting,
one row per (space, anchor): the `lam` - and `T` where the penalty has one -
that member reaches furthest with inside the 0.5 pt forgetting budget, its
`A_new` (test) mean ± 95 % CI, `G`, `F`, `n`, and whether it sits on the Pareto
front of the setting. Rows are ordered by that adaptation, so the table reads as
a ranking of the penalties; a member with nothing inside the budget keeps its
row and sorts last.

**One setting per headline table.** `--setting` (default `pool0.05 m10 uniform`,
matched as a substring of the setting label) restricts `budget_selection`, the
Pareto float and the two Pareto figures to a single setting; selecting the best
arm per trainer across settings answers a different question for every trainer -
one wins with ten uniform clients per round, the next with five worst-first
ones. The complete cross-setting selection is always written to
`budget_selection_all.csv`, and `pareto.csv` always holds every arm with
`on_front` (all settings) next to `on_setting_front` (the headline one). Pass
`--setting ""` to range over everything.

**Outputs.**

```
report/
├── index.md                what every table and figure says, with the numbers
├── report.json             the same as a machine-readable manifest
├── tables/
│   ├── references.csv|tex        θ_g, the combined model, local fine-tuning,
│   │                             every reference arm with its F and G
│   ├── runs.csv                  one row per run (CSV only: data, not a float)
│   ├── arms.csv|tex              seed-aggregated mean ± CI per configuration
│   ├── paired_vs_fedavg.csv|tex  paired tests with Holm correction and markers
│   ├── pareto.csv|tex            the (F, G) front (float: headline setting)
│   ├── budget_selection.csv|tex  max A_new_test subject to F ≤ ε on validation
│   ├── budget_selection_all.csv  the same selection across every setting
│   ├── regularisation_family.csv|tex  the anchored family, one row per
│   │                             (penalty space, anchor) at the 0.5 pt budget
│   ├── cost.csv|tex              seconds and megabytes per arm
│   ├── fairness.csv|tex          per-client level, tail and spread
│   └── signals.csv|tex           signals against true forgetting
└── figures/
    ├── pareto_families.pdf       G against F, by method group
    ├── pareto_hypotheses.pdf     the same plane, by aggregation hypothesis
    ├── hypothesis_*.pdf          point plots with confidence intervals
    ├── curves_*.pdf              learning curves with a band over the seeds
    ├── fairness.pdf              distribution of the per-client accuracies
    └── signals_correlation.pdf   signal/forgetting correlation heat map
```

The `.tex` files are complete `booktabs` floats with a `\label{tab:<name>}`, so
the manuscript can `\input` them directly; add `\usepackage{booktabs}` to the
preamble. The figures are vector PDFs with embedded Type-42 fonts, drawn from a
single colour-blind-safe palette in which every series also carries its own
marker or line style, so nothing depends on colour alone.

`foa report` reuses `signals/signals_summary.json` when `foa signals` has
already run over the same tree, so the two commands can never disagree.

**Partial results are the normal case.** A results root that is still filling up
produces the same tables with smaller `n`; arms with a single seed report no
interval instead of a zero one, comparisons without a baseline or with fewer
than two seed-matched pairs are skipped, and every such gap is counted in the
"Notes on coverage" section of `index.md`.

---

## 7. Running on Slurm

`slurm/env.sh` centralises module loading, conda activation, the proxy settings and the
`FOA_*` paths. Every job script sources it.

```bash
export FOA_PROJECT_DIR=/path/to/project        # shared project directory
export FOA_ENV=$FOA_PROJECT_DIR/envs/foa       # conda prefix
export FOA_REPO=$FOA_PROJECT_DIR/repo_foa      # this checkout on the cluster
export FOA_ACCOUNT=auto_tads                   # default
export FOA_CPU_PARTITION=standard96s:shared    # default
export FOA_GPU_PARTITION=grete:shared          # default
export FOA_MNIST_DIR=$FOA_PROJECT_DIR/data/mnist   # default; the NIST proxy set

cd $FOA_REPO

sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT slurm/prepare_data.sbatch /path/to/by_write.zip
sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT slurm/run_tests.sbatch
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT slurm/run_grid.sbatch kd
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT slurm/run_final.sbatch DistillationTrainer 1234

# evaluation speed-up: the same configuration on both --eval-path values,
# into $FOA_SPEED_DIR (default $FOA_PROJECT_DIR/results_speed), plus the
# element-by-element comparison of the two result trees
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT slurm/run_speed_check.sbatch nist 20

# job array over a task file produced by `foa matrix`
foa matrix --plan e1_seeds --out $FOA_PROJECT_DIR/jobs/tasks_e1.txt
N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_e1.txt)
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
       --array=1-$N%8 slurm/run_matrix.sbatch $FOA_PROJECT_DIR/jobs/tasks_e1.txt
```

`slurm/run_matrix.sbatch` runs one GPU task per array index, writes a per-task
log under `$FOA_PROJECT_DIR/logs/matrix/` and exits with the task's status. To
resume after a time-out, resubmit the identical array - completed tasks are
skipped in seconds thanks to `--skip-existing`. To retry a subset, pass an
explicit list, e.g. `--array=7,19,23-30%4`.

Three job-array templates read the same task file:

| Script | GPU per array element | Tasks per element |
| --- | --- | --- |
| `run_matrix.sbatch` | `gpu:A100:1` | 1 |
| `run_matrix_packed.sbatch` | `gpu:A100:1`, 32 CPUs, 96 GB | `K = $FOA_TASKS_PER_JOB` (default 4), concurrently |
| `run_matrix_mig.sbatch` | `gpu:1g.10gb:1` (optional) | 1 |

**Time limits are generous by design.** The templates ask for 24 h (finals,
`run_final.sbatch` and the three array scripts) and 36 h (sweeps,
`run_grid.sbatch`). Both are well above the expected runtime on purpose: the
accounting charges the time a job *uses*, not the time it *asked for*, so a
limit that outlasts the slowest arm costs nothing, while a limit that does not
costs a whole resubmission. A task file of sweeps submitted through
`run_matrix.sbatch` should therefore be given the sweep budget explicitly
(`--time=36:00:00`); lower any limit with `--time=` when the queue, rather than
the runtime, is the bottleneck.

| Script | Limit | Why |
| --- | --- | --- |
| `run_final.sbatch`, `run_matrix*.sbatch` | 24 h | one finals task: four aggregation families x 100 rounds |
| `run_grid.sbatch` | 36 h | one sweep task: the whole hyperparameter product |
| `run_speed_check.sbatch` | 8 h | two 20-round runs plus the comparison |
| `run_tests.sbatch` | 30 min | CPU suite |

**Packed arrays.** Short tasks leave a whole A100 idle for most of the element,
and the queue only grants a few elements at a time. `run_matrix_packed.sbatch`
runs `K` *consecutive* lines of the task file concurrently in one element and
`wait`s for them, so element `i` covers lines `(i-1)*K+1 … i*K`. The array size
is therefore `ceil(N / K)`, each task still gets its own
`logs/matrix/task_<arrayjob>_<line>.log`, and the element exits with the worst
status of its tasks:

```bash
foa matrix --plan smoke --out $FOA_PROJECT_DIR/jobs/tasks_smoke.txt
N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_smoke.txt)
K=${FOA_TASKS_PER_JOB:-4}
A=$(( (N + K - 1) / K ))

sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
       --export=ALL,FOA_TASKS_PER_JOB=$K --array=1-$A%4 \
       slurm/run_matrix_packed.sbatch $FOA_PROJECT_DIR/jobs/tasks_smoke.txt
```

Keep the per-task worker counts low when packing (the `smoke` plan passes
`--outer-workers 2 --inner-workers 2`), so `K x outer x inner` processes stay
inside `--cpus-per-task`. To retry a failed line, submit the element that holds
it: `element = ceil(line / K)`.

**GPU memory versus worker fan-out.** A task fans out twice: `--outer-workers`
threads over the (scenario, metadata, aggregation) jobs, and `--inner-workers`
*processes* inside each of them, so a task holds up to `outer x inner` CUDA
contexts at once. Each context costs its own CUDA runtime plus the activations
of one client update, and the sweeps train at batch 512 on 128x128 inputs, which
is roughly ten times the activation footprint of a final run at batch 64.

| Task | Fan-out | Fits on 40 GB |
| --- | --- | --- |
| `foa final`, `foa extreme` | `2 x 2` at batch 64 | yes, comfortably |
| `foa grid` | `2 x 3` at batch 512 | yes |
| `foa grid` with the engine defaults | `2 x 8` at batch 512 | **no** - out of memory |

`foa matrix` therefore emits every sweep line with
`--outer-workers 2 --inner-workers 3` unless the plan already chose its own
counts (the `smoke` plan does). The grid engine's own defaults are unchanged, so
a hand-written `foa grid` command still behaves exactly as before - on an 80 GB
accelerator it is fine, and on a 40 GB one pass the flags. When packing several
tasks into one element with `run_matrix_packed.sbatch`, multiply again: keep
`K x outer x inner` inside both the CPU count and the accelerator.

**MIG slices (optional).** The models of this study are around a million
parameters, so a single-run task fits into a 10 GB MIG slice.
`run_matrix_mig.sbatch` is `run_matrix.sbatch` with `--gres=gpu:1g.10gb:1`; on
sites that hand out slices it is scheduled much sooner than a job holding a
full accelerator. Check with `sinfo -o "%G"` whether the partition offers
`1g.10gb`; where it does not, use the other two templates. No plan requires it.

Compute nodes on many HPC sites have no direct route to the internet. `slurm/env.sh`
exports `http_proxy`/`https_proxy` (default `http://www-cache.gwdg.de:3128`, override
with `FOA_HTTP_PROXY`) so that `pip`, `conda` and `requests` work inside a job.

---

## 8. Tests

The test suite runs on the CPU against a small synthetic provider (random 128x128
images, three clients, three classes, two rounds) and never touches the real dataset or
the published results.

```bash
python -m pytest -q
```

It covers every trainer through both runners with a representative aggregation method of
each of the four families, the timing and communication instrumentation, seed
reproducibility, the provenance block, equality of the cache and PNG dataset readers, and
that every module of the package imports and every CLI subcommand is wired up.

The features added on top of the published pipeline are covered by
`tests/test_model_features.py` (the penultimate accessors),
`tests/test_new_trainers.py` (feature alignment and layer freezing, including
that frozen weights and their normalisation statistics do not move),
`tests/test_client_sampler.py` and `tests/test_population.py` (selection
policies, duplicate participants, the held-out accuracy series and that the
defaults reproduce the published loop bit for bit),
`tests/test_outlier_selection.py` (the deterministic pools) and
`tests/test_matrix.py` (plan sizes, that every generated task parses as a CLI
invocation, and the contract of the `smoke` plan: cheap budget, `smoke_*`
parent, `--skip-existing`, and never the published global model).
`tests/test_cli_options.py` additionally pins the round/epoch/batch-size
overrides: unset means unset, so the driver and the sweep engine keep their
published budgets.

`tests/test_forgetting_signals.py` covers the constraint-respecting signals from
both ends: that both runners log all eight series once per round, that round 0
reproduces the reference model exactly (agreement 1, KL 0, distances 0), that a
provider without a proxy set reports `None` and changes nothing, and that the
analysis recovers correlations, stopping rounds and the selected configuration
from synthetic trajectories whose answer is known by construction.
`tests/test_statistics.py` pins the closed forms the report layer decides with
(t quantiles against a printed table, an exactly enumerated Wilcoxon null
distribution, the Holm step-down arithmetic), `tests/test_report.py` builds
synthetic result trees in the real stored shape - summary plus per-job file,
sweep config block, frozen artefacts, full provenance - and checks the metrics,
the confidence intervals, the seed-matched comparisons, the Pareto front, the
budget selection and that every generated LaTeX table is a balanced booktabs
float, and `tests/test_regularisation_finals.py` covers the per-sweep selection
and the task lines it emits.
`tests/extra/test_proxy_sets.py` covers the three proxy definitions: the IDX
decoding and the NIST polarity of the rendered MNIST images, that the reserved
CIFAR-10 rows belong to no client, and that the Shakespeare roles are gone from
the eligible population and from any stored pool.

`tests/test_eval_cache.py` is the equality check of the evaluation cache in
miniature. It asserts the property everything else rests on - the batched decode
of the compact payload is **bit-identical** to the dataset's per-sample
transform, for the packed-cache reader and through `Subset` and `ConcatDataset` -
then that a cached accuracy equals the loader accuracy exactly, that the
per-client accuracies come from a single forward pass (counted with an
instrumented model), that a dataset which cannot be materialised is simply
absent from the cache, and that both runners produce the same series on
`--eval-path loader` and `--eval-path cache`. It also pins `--insample-every`:
the default measures every round, a larger value measures every `N`-th round and
the last one, and no other series is affected.
`tests/test_early_stopping.py` covers the early-stopped FedAvg arm: off by
default (every local epoch is run), `1 + patience` epochs on a model that cannot
improve, the best-validation model restored either way, and the flag arriving
through the registry, the `--set` override and the provenance block.

---

## 9. Repository structure

```
federated-outlier-adaptation/
├── code/                                  notebooks documenting each phase
├── data/
│   └── by_write/digits_labels.json        image path -> digit label manifest
├── docs/                                  reproducibility and results-layout notes
├── results/                               frozen artefacts and experiment outputs
├── slurm/                                 env.sh + job templates
├── tests/                                 pytest suite (CPU, synthetic data)
├── src/federated_outlier_adaptation/
│   ├── config.py                          all filesystem locations
│   ├── constants.py                       hyperparameters only
│   ├── logging_utils.py                   project logger
│   ├── model.py                           FlexibleCNN
│   ├── cli.py                             `foa` entry point
│   ├── aggregation/                       concurrent_methods, sequential_methods, selector
│   ├── data/                              datasets, nist_cache, download, labels, hashes
│   ├── providers/                         base interface + NIST implementation
│   ├── trainers/                          base + 6 regularised trainers, registry
│   ├── runners/                           concurrent_runner, sequential_runner
│   ├── training/                          driver + phase scripts + fisher/evaluation
│   │                                      (incl. matrix plans and tasks: resume/cost)
│   ├── grid_search/                       common engine + one module per sweep
│   ├── outliers/                          client accuracies and selection
│   ├── plotting/                          training curves, overlays, heatmaps
│   ├── analysis/                          rankings, tables and paper figures
│   │                                      (incl. eval_equality: loader vs cache)
│   └── utils/                             seeding, provenance, instrumentation,
│                                          eval_cache (materialised evaluation sets)
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

The notebooks in `code/` (`step1` ... `step6`) document the original phase-by-phase
workflow and now call the same package functions the CLI does. The CLI is the supported
entry point; the notebooks are kept as documentation.

---

## 10. Configuration reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOA_REPO_ROOT` | inferred from the package location | repository root |
| `FOA_DATA_DIR` | `<repo>/data` | dataset root (`by_write/digits_labels.json`) |
| `FOA_RESULTS_DIR` | `<repo>/results` | experiment inputs and outputs |
| `FOA_CACHE_DIR` | `<data>/cache` | packed dataset cache |
| `FOA_LOG_DIR` | `<results>/logs` | log files |

The same values are settable per command with `--data-dir`, `--results-dir` and
`--cache-dir`.

Every variable was named `FAL_*` in the earlier repository. Both prefixes are read:
`FOA_<NAME>` wins, and `FAL_<NAME>` is used when it is unset, so an existing site
configuration keeps working without an edit. `slurm/env.sh` follows the same rule, and
also falls back to a conda prefix at `$FOA_PROJECT_DIR/envs/fal` when no
`$FOA_PROJECT_DIR/envs/foa` exists.

---

## 11. Known issues in the published scripts

These are documented for transparency; the behaviour of the published numbers is
unchanged.

- **Client restriction was an intersection.** `single_outlier` filtered the selected
  outlier list, so a repeated writer id contributed a single participant. The `double`
  and `dual` extreme cases therefore used the same participant as `single` and differed
  only in hyperparameters and output folder; the stored `double_outlier_*` and
  `dual_outlier_*` folders are single-participant runs. Repeated ids now expand into
  independent participants and the two cases write into new folders
  (`double_writers_*`, `dual_replicated_*`), so the stored results are preserved.
- **Notebook `code/step6_extreme_cases.ipynb`** calls
  `extreme_cases_grid_search(single_outlier=["f3503_07", "f3503_07"], extreme_case="double")`
  in all three grid-search cells, so the `single` and `dual` sweeps stored under
  `results/single_outlier_distillation_grid_search/` and
  `results/dual_outlier_distillation_grid_search/` were produced with the `double`
  configuration. Use `foa grid --method extreme --case <case>` to run each case
  correctly.
- **Aggregation names.** Several grid-search task lists referenced aggregation methods
  that do not exist (`agg_weighted_cgw`, `sequential_agg_delta_fedavg_update`, ...), so
  those sweeps raised `ValueError` before starting. They now reference the real methods
  (`con_weighted_cgw`, `con_delta_weighted_cgd`, `seq_fedavg_update`,
  `seq_delta_fedavg_update`).
- **`all-aggs` output folder.** The sequential/weights job of the all-aggregations sweep
  wrote into `prove_fl_BaseTrainer_grid_search/` instead of
  `all_aggs_fl_BaseTrainer_grid_search/`. New runs write into the latter.
- **The published writer split cannot be re-derived.** It was produced from a directory
  glob whose order is not reproducible. The file is therefore treated as frozen input.

---

## 12. Additional datasets

The federated protocol is dataset agnostic. Everything a dataset contributes -
clients, loaders, model topology, artefact locations - sits behind the
`DatasetProvider` interface (`providers/base.py`), so the runners, trainers and
aggregation methods run on a new dataset without a single change. Two datasets are
provided next to NIST SD19:

| Provider | Task | Clients | Model | Classes |
| --- | --- | --- | --- | --- |
| `nist` (default) | handwritten digit classification | writers | `FlexibleCNN` | 10 |
| `shakespeare` | next-character prediction | speaking roles | `CharLSTM` | 80 |
| `cifar10` | image classification | 200 Dirichlet(0.3) partitions | `CifarCNN` | 10 |

### 12.1 Sources and licences

**LEAF Shakespeare.** The corpus is the Project Gutenberg *Complete Works of William
Shakespeare* (eBook #100), which is in the **public domain** in the United States.
The task definition and the preprocessing follow LEAF:

> S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konecny, H. B. McMahan, V. Smith,
> A. Talwalkar, *LEAF: A Benchmark for Federated Settings*, arXiv:1812.01097, 2018.

LEAF downloads `http://www.gutenberg.org/files/100/old/1994-01-100.zip`. That path has
been retired: gutenberg.org and every mirror checked (aleph, xmission, pglaf, uwaterloo)
answer 404 for it. The current plain-text edition of the **same eBook #100** is used
instead - `https://www.gutenberg.org/cache/epub/100/pg100.txt`, whose header still reads
`Release date: January 1, 1994 [eBook #100]` with a later "most recently updated" stamp.
It is the same corpus in a re-typeset edition: speaker cues stand alone on their line in
full capitals rather than being indented by two spaces, and punctuation is typographic.
`data/shakespeare.py` therefore implements LEAF's *rules* (play to role to utterance,
roles named `<PLAY>_<ROLE>`, 80-character windows with stride 1, LEAF's 80-symbol
`ALL_LETTERS` alphabet) against that layout and folds the typographic characters back
onto the ASCII alphabet. The result is very close to the published LEAF statistics
(1 180 roles / 4 074 206 sequences here versus 1 129 / 4 226 158 in LEAF).

**CIFAR-10.** Downloaded from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`.

> A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, Technical
> Report, University of Toronto, 2009.

### 12.2 Preparation

Neither archive is ever extracted; both are read in memory (`zipfile` / `tarfile`) and
each dataset becomes **one array file plus one JSON index**, exactly as the NIST cache
does.

```bash
# raw downloads (small, one file each)
curl -o data/shakespeare/pg100.txt https://www.gutenberg.org/cache/epub/100/pg100.txt
curl -o data/cifar10/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# preparation
python -m federated_outlier_adaptation.data.shakespeare \
    --raw data/shakespeare/pg100.txt --out-dir data/shakespeare
python -m federated_outlier_adaptation.data.cifar10 \
    --raw data/cifar10/cifar-10-python.tar.gz --out-dir data/cifar10 \
    --num-clients 200 --alpha 0.3 --seed 42 --proxy-size 1000
```

`--proxy-size N` holds `N` of the 10 000 evaluation images back, stratified by
label and seeded, before the remaining ones are dealt out to the clients, so the
reserve belongs to no client and can serve as the server-side proxy set (section
4). It defaults to `0`, which reproduces the partition that existed before the
proxy sets; `--clients-name` writes the index under a second file name so an
existing partition stays reachable. **Changing the reserve changes the client
pools**, so a partition regenerated with `--proxy-size` needs its global model,
client accuracies and outlier pools regenerated with it.

`data/shakespeare.py` also accepts LEAF's original `1994-01-100.zip` with `--raw`; the
`.txt` inside is read straight out of the archive.

| File | Content |
| --- | --- |
| `data/shakespeare/shakespeare.npz` | `tokens` (concatenated `uint8` character stream), `user_starts`, `user_lengths`, `seq_starts`, `seq_counts`, `seq_length`, `vocab_size` |
| `data/shakespeare/shakespeare_index.json` | role ids, per-role sequence counts, vocabulary, corpus checksum, parsing statistics |
| `data/cifar10/cifar10.npz` | `images` `uint8[60000, 32, 32, 3]`, `labels`, `is_test` |
| `data/cifar10/cifar10_clients.json` | per-client sample indices (200 clients), the reserved `proxy` rows, label statistics, archive checksum |
| `data/mnist/mnist.npz` | `train_images`/`test_images` `uint8[·, 28, 28]` and their labels - the NIST proxy set |
| `data/mnist/mnist_index.json` | image counts, polarity, render description, SHA-256 of each of the four `idx` archives |

Because the 80-character windows overlap with stride 1, storing them explicitly would
cost eighty times the corpus. The npz stores the character stream and the per-role
offsets instead; `ShakespeareData.X` / `.y` reconstruct the `[Nseq, 80]` and `[Nseq]`
arrays exactly, and `ShakespeareData.build_dataset` addresses the windows directly.

### 12.3 Client protocol

Both datasets follow the NIST protocol: a pooled global model is trained on a subset of
clients, the held-out clients are scored by that model, and the `k` lowest-accuracy
held-out clients become the federated participants.

- **Shakespeare** - a seeded (42) shuffle assigns **10 %** of the roles (118 of 1 180) to
  the global pool, which carries 403 234 of the 4 074 206 sequences; the remaining 1 062
  roles are held out. Participants are the lowest-accuracy held-out roles with at least
  100 sequences, minus the **20 roles reserved as the proxy set** (section 4): a second
  seeded shuffle carves them out of the eligible held-out roles, they are excluded from
  `eligible_clients()` and filtered out of any stored pool on read, so the population a
  run draws from and the data the server observes it with are disjoint. The carve-out is
  capped at a quarter of the eligible roles, which is never binding on the real corpus
  and leaves a toy corpus untouched.
- **CIFAR-10** - the 50 000 training images are dealt out to **200 clients** with a
  `Dirichlet(alpha=0.3)` prior per label (seeded 42); **1 000 of the 10 000 evaluation
  images are reserved as the proxy set** (section 4, `--proxy-size 1000`, 100 per class)
  and the remaining 9 000 follow with each client's own label proportions, so a client's
  pool keeps one label profile (83 - 1 048 samples per client, 295 on average). The
  reserve belongs to no client, which is what lets the server read it. A seeded shuffle puts **100**
  clients in the global pool and holds **100** out. The held-out half is deliberately
  large: the main experimental setting draws its participants from a *pool* of the worst
  5 - 20 % of the held-out clients, which needs a held-out population of that order. The
  earlier smaller layout stays reachable - prepare with `--num-clients 100` and construct
  `Cifar10Provider(num_global_clients=80)` for a 100 / 80 / 20 partition.

In both cases each client owns a single pool of samples, and the train / validation /
test split of that pool is `build_dataset`'s job with exactly the NIST rates (stratified
by label, 60 / 20 / 20 by default, 60 / 40 / 0 for client adaptation and 0 / 0 / 100 for
the evaluation loaders). The protocol is therefore identical across the three datasets.
CIFAR-10 augmentation (four-pixel padded random crop plus horizontal flip) is applied by
`build_training_dataset` only, i.e. to pooled server-side training, never to client
adaptation or evaluation.

### 12.4 Artefacts and running

Artefacts land under `RESULTS_DIR/<provider>/` with the NIST file names:

```
results/shakespeare/            (and results/cifar10/)
├── writer_split.json           global / local client split
├── global_model                pooled global model
├── global_results/
│   ├── global_metrics.json     train, validation and test accuracy
│   ├── provider_global_summary.json
│   └── fisher/{fisher.pt,global_params.pt}
├── global_clients_model        combined reference model
├── global_clients_results/     combined reference metrics
└── outliers/
    ├── clients_acc_on_global.json    accuracy of every held-out client
    ├── selected_clients_k{5,20,50}.json
    ├── selected_outliers.json        mirror of the k=5 list
    └── outlier_pool_frac{0.05,0.2}.json   rule-based pool, ids + baseline accuracies
```

```bash
# 1. pooled global model, Fisher, per-client accuracies, selected-client lists
python -m federated_outlier_adaptation.training.global_model \
    --provider shakespeare --epochs 100 --patience 5 --batch-size 64 --k 5 20 50

# 2. combined reference model (global clients + participants)
python -m federated_outlier_adaptation.training.combined_model \
    --provider shakespeare --epochs 15

# 3. one adaptation configuration through the unchanged runner
python -m federated_outlier_adaptation.training.provider_adaptation \
    --provider shakespeare --trainer BaseTrainer --scenario concurrent \
    --metadata weights --agg con_weighted_cgw --rounds 5 --epochs 2
```

Every module without `--provider` keeps doing what it did before, so `foa global-train`,
`foa combined-train` and all published commands stay on NIST and reproduce their exact
output paths.

The baselines produced by those three commands on one A100 (Adam 1e-3, weight decay
1e-4, batch 64, up to 100 epochs with early stopping):

| | Shakespeare | CIFAR-10 |
| --- | --- | --- |
| Clients (global / held out) | 118 / 1 062 | 100 / 100 |
| Pooled training samples | 403 234 sequences | 30 610 images |
| Epochs until early stop | 23 (patience 5) | 50 (patience 10) |
| Global model test accuracy | **0.5686** | **0.9288** |
| Held-out client accuracy, median | 0.5330 | 0.8631 |
| Held-out client accuracy, min - max | 0.2263 - 0.6545 | 0.7181 - 0.9387 |
| Combined reference, global data | 0.5615 | 0.9285 |
| Combined reference, participants | 0.3893 | 0.8994 |

The CIFAR-10 column is the partition **with the 1 000-image proxy reserve**; the
clients therefore hold 9 000 rather than 10 000 evaluation images and the whole
partition shifts with them. The earlier reserve-free partition reached 0.9154
with 31 101 pooled training images; it is still reachable as
`data/cifar10/cifar10_clients_proxy0.json` together with its artefacts under
`results/cifar10_noproxy/`.

The Shakespeare figure sits at the upper end of the 50 - 55 % that LEAF reports for
FedAvg on this task, as a pooled model should. Its five lowest-accuracy held-out roles
are the ones that do not speak English - `FRENCH_SOLDIER`, `KATHARINE` and `ALICE` of
*Henry V*, `WILLIAM` of *The Merry Wives of Windsor* (the Latin lesson) - which is the
outlier structure the adaptation experiments are about.

Programmatically a run is pointed at a dataset with the registry:

```python
from federated_outlier_adaptation.providers import get_provider

provider = get_provider("cifar10")                     # or "shakespeare", "nist"
trainer = BaseTrainer(provider=provider)
runner = BaseConcurrentRunner(trainer=trainer, provider=provider)
```

The `config` provenance block of a run produced this way additionally records the
provider name, its results and data roots, and the SHA-256 of every prepared dataset
file (`providers.provider_config_block`).

### 12.5 Outlier pools

The fixed `selected_clients_k<k>.json` lists name the participants of a run directly.
The pool files describe the *population* those participants are drawn from, which is
what a run with per-round client selection needs:

```bash
python -m federated_outlier_adaptation.outliers.client_accuracy \
    --provider cifar10 --pool-frac 0.05 0.2
```

`write_pool(clients_acc_path, out_dir, frac=..., threshold=..., eligible=...)` applies one
of two rules to the eligible held-out clients (Shakespeare: roles with at least 100
sequences; CIFAR-10: clients with at least 20 samples):

- `frac` - the worst `max(1, int(frac * n))` clients, the truncating rule the published
  NIST bottom fraction already uses;
- `threshold` - every eligible client below an accuracy cut-off, written to
  `outlier_pool_thr<t>.json`.

Unlike `selection.get_low_acc_writers`, which samples randomly inside the bottom
fraction and defines the frozen NIST outlier set, a pool *is* the ranking and is
therefore deterministic. The file records the ids worst first plus each member's
baseline accuracy under the global model:

```json
{
  "rule": "fraction", "frac": 0.05, "size": 5, "eligible_clients": 100,
  "clients": ["c056", "c062", "c004", "c026", "c067"],
  "accuracies": {"c056": 0.7761, "c062": 0.7813},
  "min_accuracy": 0.7761, "max_accuracy": 0.7938, "mean_accuracy": 0.7865,
  "source": ".../outliers/clients_acc_on_global.json"
}
```

`provider.outlier_pool(frac)` returns the ids, reading the stored file when it exists
and deriving them from `clients_acc_on_global.json` otherwise;
`client_accuracy.load_pool(path)` also accepts a plain JSON list of ids. Pools are
written by the global-training phase with `--pool-frac`, or afterwards from the stored
accuracies with the command above. The pools generated for the two datasets:

| Pool | Shakespeare (934 eligible) | CIFAR-10 (100 eligible) |
| --- | --- | --- |
| `frac=0.05` | 45 roles, accuracy 0.2263 - 0.4772 | 5 clients, 0.7181 - 0.7625 (mean 0.7452) |
| `frac=0.20` | 186 roles, accuracy 0.2263 - 0.5135 | 20 clients, 0.7181 - 0.8160 (mean 0.7862) |

The Shakespeare figures are the stored pools (47 and 190 roles, drawn from 954
eligible before the proxy set existed) after the 20 reserved roles are filtered
out on read; regenerating them from `clients_acc_on_global.json` writes the same
populations directly.

---

### 12.6 Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOA_SHAKESPEARE_DIR` | `<data>/shakespeare` | corpus, `shakespeare.npz`, `shakespeare_index.json` |
| `FOA_CIFAR10_DIR` | `<data>/cifar10` | archive, `cifar10.npz`, `cifar10_clients.json` |
| `FOA_MNIST_DIR` | `<data>/mnist` | the four `idx` archives, `mnist.npz`, `mnist_index.json` (the NIST proxy set) |
| `FOA_NIST_PROXY_SIZE` | unset (all 10 000) | cap the MNIST proxy set with a seeded subsample, for runs where a full pass per round is too expensive; the size enters the proxy hash |

`RESULTS_DIR/shakespeare` and `RESULTS_DIR/cifar10` follow `FOA_RESULTS_DIR`.

---

## Citation

This repository accompanies the manuscript:

S. Keshtkar, M. Bidollahkhani, J. M. Kunkel, *Knowledge-Preserving Adaptation to Outlier
Clients in Federated Learning: An Empirical Study of Aggregation, Scheduling and
Regularization*, submitted, 2026.

```bibtex
@unpublished{keshtkar2026outlier,
  author = {Keshtkar, Sadegh and Bidollahkhani, Michael and Kunkel, Julian M.},
  title  = {Knowledge-Preserving Adaptation to Outlier Clients in Federated Learning:
            An Empirical Study of Aggregation, Scheduling and Regularization},
  note   = {Submitted},
  year   = {2026}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

---

## License

This project is released under the MIT License.
