"""
Experiment constants.

Only hyperparameters live here; every filesystem location is resolved through
:mod:`federated_outlier_adaptation.config`.

The ``BEST_*`` values are the configurations selected by the grid searches and
used for the final runs reported in the paper.  They can be overridden per run
without editing this file, via ``--set NAME=VALUE`` on the CLI or the
``trainer_kwargs`` argument of the training driver.
"""

# --- global-model training ---------------------------------------------------
NUMBER_EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIENTS_NUMBER_EPOCHS = 10
BATCH_SIZE = 16

# --- selected hyperparameters used for the final runs ------------------------
BEST_EWC_LAMBDA = 8.0
BEST_KD_T = 8.0
BEST_KD_ALPHA = 0.95
BEST_PROX_LAMBDA = 0.9
BEST_LOGIT_LAMBDA = 0.1
BEST_FEATURE_BETA = 0.1

# --- representation-level feature alignment (FeatureAlignmentTrainer) --------
# Initial value; the feature-alignment grid search selects the reported one.
BEST_FEATURE_ALIGN_BETA = 0.1

# --- not-true distillation (NTDTrainer) --------------------------------------
# Initial values; the not-true distillation grid search selects the reported ones.
BEST_NTD_BETA = 1.0
BEST_NTD_TAU = 1.0

# --- unified anchored regularisation (AnchoredTrainer) -----------------------
# Defaults of the family trainer; the anchored grid search selects the
# reported combination of space, anchor and lambda.
DEFAULT_ANCHOR_SPACE = "kd"
DEFAULT_ANCHOR = "frozen"
BEST_ANCHOR_LAM = 1.0
BEST_ANCHOR_T = 4.0
BEST_ANCHOR_MIX = 0.5

# --- layer freezing (FreezeTrainer) ------------------------------------------
# "conv": the convolutional feature extractor only.
# "body": the feature extractor plus the first fully connected layer.
FREEZE_SCOPE = "body"

# --- extreme-case configurations reported in the paper -----------------------
# Reproduce with e.g.  foa extreme --case single --set T=50 alpha=0.98
EXTREME_CASE_HYPERPARAMETERS = {
    "single": {"T": 50.0, "alpha": 0.98},
    "double": {"T": 8.0, "alpha": 0.95},
    "dual": {"T": 8.0, "alpha": 0.95},
}

# --- extreme-case client selections ------------------------------------------
# "single": one outlier.
# "double": two distinct low-accuracy writers.
# "dual":   the same outlier replicated, i.e. two independent participants that
#           hold identical data.
#
# The published runs restricted the client set by intersection, so a repeated
# id contributed a single participant and "double"/"dual" behaved like
# "single".  Both cases therefore write into their own output folders
# (EXTREME_CASE_TAGS) and never overwrite the stored results.
EXTREME_CASE_CLIENTS = {
    "single": ["f3503_07"],
    "double": ["f3503_07", "f2307_62"],
    "dual": ["f3503_07", "f3503_07"],
}

# --- extreme-case output folder tags -----------------------------------------
EXTREME_CASE_TAGS = {
    "single": "single_outlier",
    "double": "double_writers",
    "dual": "dual_replicated",
}
