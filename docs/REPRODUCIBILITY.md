> **Superseded.** This describes the PRIOR pipeline, not the live study
> `Digits_study01`.
> For the live study use `docs/REPRODUCE.md` - which maps every claim to the
> command that regenerates it - together with `docs/DATA.md`, `docs/RUNBOOK.md`
> and `docs/VERIFY.md`. Kept because the code it documents is still present.

# Reproducibility

The end-to-end command sequence, from an empty checkout to the tables and figures of the
manuscript. Every step is explained in depth in the [README](../README.md); this page is
the ordered list of commands and nothing else.

Two conventions hold throughout:

- **No archive is ever extracted.** `by_write.zip` is read in place; the dataset
  becomes one array file plus one JSON index. Running `unzip` on `by_write.zip` (≈ 400 000 PNG files) exhausts the inode quota
  of most shared file systems.
- **Frozen artefacts under `results/` are inputs.** `writer_split.json`, `global_model`,
  the Fisher directory and the selected-outlier list define the experimental setting and
  are reused whenever they exist (README section 3). Do not regenerate them unless you
  intend to leave the published setting.

---

## 1. Environment

Python 3.12.

```bash
git clone https://github.com/paymankeshtkaruni/federated-outlier-adaptation.git
cd federated-outlier-adaptation

# GPU (CUDA 12.1), the build used for the published runs
pip install torch==2.3.1 torchvision==0.18.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

Or with conda: `conda env create -f environment.yml -p ./envs/foa`, then
`conda activate ./envs/foa` and `pip install -e . --no-deps`.

The package installs as `federated_outlier_adaptation` and provides the `foa` console
script (`fal`, the name of the earlier repository, is kept as an alias for one release).
Everything is equally reachable as `python -m federated_outlier_adaptation.cli`, which is
what the Slurm templates use so no `PATH` assumption is needed.

## 2. Data

```bash
# NIST SD19, writer-partitioned subset (568 MB, one file)
curl -o data/nist/by_write.zip https://s3.amazonaws.com/nist-srd/SD19/by_write.zip
foa prepare-data --zip data/nist/by_write.zip

```

The NIST cache lands in `$FOA_CACHE_DIR` (default `data/cache`) as
`nist_digits_u8.npy` + `nist_digits_index.json`. See README section 2 for the file
contents.

## 3. Pipeline

```bash
foa global-train                     # global model, Fisher, per-client accuracies (reuses frozen files)
foa combined-train                   # combined global + outliers reference model
foa base-fl                          # baseline federated runs
foa all-aggs                         # baseline trainer against every aggregation rule
foa grid --method kd                 # one hyperparameter sweep (see --method for the full list)
foa final --trainer DistillationTrainer --seed 1234
foa extreme --case single            # minimal-client scenarios
foa local-finetune                   # no-federation reference point
```

Multi-seed and population studies are expressed as **plans** rather than typed out:

```bash
foa matrix --list                                   # the available plans
foa matrix --plan e1_seeds --out jobs/tasks_e1.txt  # one `foa ...` line per task
```

Every generated line carries `--skip-existing`, so a task file can be resubmitted after a
time-out and only the missing runs execute.

## 4. Selection, signals and report

```bash
foa select --eps 0.005                        # constrained, validation-based configuration choice
foa signals --root results/                   # constraint-respecting forgetting signals
foa figures                                   # grid-search rankings and heatmaps
foa report --root results/ --out report/      # every table and figure of the protocol
```

Model selection uses `pool_val_accuracies` and reporting uses `pool_test_accuracies`, so
the two never share a sample (README section 5).

## 5. Slurm

`slurm/env.sh` centralises module loading, conda activation, the outbound proxy and the
`FOA_*` paths; every job template sources it.

```bash
export FOA_PROJECT_DIR=/path/to/project           # shared project directory
export FOA_REPO=$FOA_PROJECT_DIR/repo_foa         # this checkout on the cluster
export FOA_ENV=$FOA_PROJECT_DIR/envs/foa          # conda prefix
export FOA_ACCOUNT=<account>
export FOA_CPU_PARTITION=<cpu partition>
export FOA_GPU_PARTITION=<gpu partition>

cd $FOA_REPO
sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT \
       slurm/prepare_data.sbatch /path/to/by_write.zip
sbatch --partition=$FOA_CPU_PARTITION --account=$FOA_ACCOUNT slurm/run_tests.sbatch
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT slurm/run_grid.sbatch kd
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
       slurm/run_final.sbatch DistillationTrainer 1234

# job array over a task file produced by `foa matrix`
foa matrix --plan e1_seeds --out $FOA_PROJECT_DIR/jobs/tasks_e1.txt
N=$(wc -l < $FOA_PROJECT_DIR/jobs/tasks_e1.txt)
sbatch --partition=$FOA_GPU_PARTITION --account=$FOA_ACCOUNT \
       --array=1-$N%8 slurm/run_matrix.sbatch $FOA_PROJECT_DIR/jobs/tasks_e1.txt
```

Three array templates read the same task file: `run_matrix.sbatch` (one task per full
GPU), `run_matrix_packed.sbatch` (`K = $FOA_TASKS_PER_JOB` tasks sharing one GPU) and
`run_matrix_mig.sbatch` (one task per MIG slice). README section 7 covers the worker
fan-out and the GPU-memory budget each of them implies.

On sites whose compute nodes have no direct route to the internet, `slurm/env.sh` exports
`http_proxy`/`https_proxy` (override with `FOA_HTTP_PROXY`) so `pip`, `conda` and
`requests` work inside a job.

**Migrating an existing site configuration.** The variables were named `FAL_*` in the
earlier repository. Both prefixes are read - `FOA_<NAME>` wins, `FAL_<NAME>` is the
fallback - and `slurm/env.sh` also uses a conda prefix at `$FOA_PROJECT_DIR/envs/fal` when
no `envs/foa` exists, so nothing has to be renamed before the first run.

**Without installing into a shared environment.** The job scripts call
`python -m federated_outlier_adaptation.cli`, and `slurm/env.sh` puts `$FOA_REPO/src` on
`PYTHONPATH` before it checks whether the package imports. A checkout can therefore be
tested against an environment it is not installed into:

```bash
PYTHONPATH=$FOA_REPO/src python -m pytest -q tests
PYTHONPATH=$FOA_REPO/src python -m federated_outlier_adaptation.cli --help
```

## 6. Tests

```bash
python -m pytest -q
```

The suite runs on the CPU against a small synthetic provider and never touches the real
dataset or the published results. README section 8 lists what each module covers.

## 7. Provenance

Every `summary_*.json`, `accuracies_*.json` and `config_points_*.json` carries a `config`
block recording the trainer and its hyperparameters, the scenario, aggregation rule and
budget, the seed, the outlier file and the writer ids it resolved to, the global model
path with its SHA-256, torch/CUDA/Python versions, device, git commit, hostname and the
start/end timestamps. A stored result is therefore self-describing: it says which code and
which frozen inputs produced it. See [`RESULTS_LAYOUT.md`](RESULTS_LAYOUT.md).
