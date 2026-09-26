"""Leakage-safe temporal splitting utilities.

All split points are positional and preserve chronology.  In particular, the
test partition is never yielded by :func:`walk_forward_splits`.
"""

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np


@dataclass(frozen=True)
class TemporalSplit:
    train: slice
    validation: slice
    test: slice


def temporal_split_indices(
    n_samples: int,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
) -> TemporalSplit:
    """Return contiguous train/validation/test slices in chronological order."""
    if n_samples < 3:
        raise ValueError("At least three samples are required")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    test_start = int(n_samples * (1.0 - test_fraction))
    validation_start = int(n_samples * (1.0 - test_fraction - validation_fraction))
    if validation_start < 1 or test_start <= validation_start or test_start >= n_samples:
        raise ValueError("The requested split creates an empty partition")
    return TemporalSplit(
        train=slice(0, validation_start),
        validation=slice(validation_start, test_start),
        test=slice(test_start, n_samples),
    )


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 3,
    min_train_fraction: float = 0.5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield expanding-window train and non-overlapping validation indices.

    ``n_samples`` must describe only the development (train + validation)
    partition.  Callers therefore cannot accidentally expose final-test rows.
    """
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1")
    if not 0 < min_train_fraction < 1:
        raise ValueError("min_train_fraction must be in (0, 1)")

    initial_train = max(1, int(n_samples * min_train_fraction))
    fold_size = (n_samples - initial_train) // n_splits
    if fold_size < 1:
        raise ValueError("Not enough development samples for the requested folds")

    for fold in range(n_splits):
        train_end = initial_train + fold * fold_size
        val_end = n_samples if fold == n_splits - 1 else train_end + fold_size
        yield np.arange(train_end), np.arange(train_end, val_end)
