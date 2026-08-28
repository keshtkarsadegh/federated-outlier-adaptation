"""
Run-level seeding helpers.

Seeding is strictly opt-in: with ``seed=None`` nothing is touched and a run
behaves exactly as it did before this module existed (only the dataset split
seed of ``build_dataset`` applies, as in the published experiments).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_run_seed(seed: Optional[int], deterministic: bool = False) -> Optional[int]:
    """
    Seed ``random``, ``numpy`` and ``torch`` (CPU and CUDA) for one run.

    Args:
        seed: Seed value.  ``None`` is a no-op and keeps the historical
            behaviour of the code base - unless a fold is configured, in which
            case the fold itself becomes the seed.
        deterministic: When True, also switch cuDNN into deterministic mode.
            Off by default because it changes convolution algorithm selection
            and therefore throughput.

    Returns:
        The seed that was applied, or ``None``.
    """
    # The fold is part of the draw: two folds of one configuration must differ
    # in their initialisation and their sampling, not only in their split.
    seed = fold_seed(seed, configured_fold())
    if seed is None:
        return None

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def make_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """
    Build a ``torch.Generator`` for DataLoader shuffling.

    Returns ``None`` when ``seed`` is ``None`` so that DataLoaders keep their
    default (unseeded) global RNG behaviour.
    """
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


#: Multipliers used to fold a fold number into a seed - the constants of a
#: well-tested 64-bit linear congruential generator, so that fold 2 of seed 1 and
#: fold 1 of seed 2 are unrelated rather than one apart.
_SEED_MULTIPLIER = 6364136223846793005
_FOLD_MULTIPLIER = 1442695040888963407


def fold_seed(seed: Optional[int], fold: Optional[int]) -> Optional[int]:
    """
    Mix a fold number into a seed.

    ``fold=None`` returns the seed untouched, which is what keeps every existing
    split, initialisation and sampler byte-for-byte what it was.  A seed of
    ``None`` with a fold given becomes a seed of its own, so a fold is a
    complete description of a draw even where no seed was passed.
    """
    if fold is None:
        return seed
    base = 0 if seed is None else int(seed)
    return (base * _SEED_MULTIPLIER + int(fold) * _FOLD_MULTIPLIER) % (2 ** 31)


def configured_fold() -> Optional[int]:
    """The fold of this run, from the environment; ``None`` when unset."""
    from federated_outlier_adaptation import config

    return config.fold()


def seed_suffix(seed: Optional[int]) -> str:
    """
    The output-directory component of a run: ``[fold<k>_]seed_<n>``.

    A fold is part of the path, not only of the provenance, because five folds
    of one configuration are five different runs over five different splits and
    must not overwrite each other - and ``--skip-existing`` must be able to tell
    them apart.  Without a fold the component is exactly the ``seed_<n>`` every
    existing result already lives under.
    """
    fold = configured_fold()
    tag = "" if seed is None else f"seed_{int(seed)}"
    if fold is None:
        return tag
    return f"fold{int(fold)}_{tag}" if tag else f"fold{int(fold)}"
