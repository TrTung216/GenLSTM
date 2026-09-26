import numpy as np
import pytest

from src.validation import temporal_split_indices, walk_forward_splits


def test_temporal_split_is_ordered_and_disjoint():
    split = temporal_split_indices(100, validation_fraction=0.2, test_fraction=0.2)
    train = np.arange(100)[split.train]
    validation = np.arange(100)[split.validation]
    test = np.arange(100)[split.test]

    assert len(train) == 60
    assert train[-1] < validation[0]
    assert validation[-1] < test[0]
    assert len(set(train) & set(validation) & set(test)) == 0


def test_walk_forward_expands_without_future_leakage():
    folds = list(walk_forward_splits(80, n_splits=3, min_train_fraction=0.5))
    assert len(folds) == 3
    for train, validation in folds:
        assert train[-1] < validation[0]
        assert set(train).isdisjoint(validation)
    assert len(folds[1][0]) > len(folds[0][0])
    assert folds[-1][1][-1] == 79


@pytest.mark.parametrize("fractions", [(0.8, 0.2), (0.0, 0.2), (0.2, 1.0)])
def test_temporal_split_rejects_invalid_fractions(fractions):
    with pytest.raises(ValueError):
        temporal_split_indices(100, fractions[0], fractions[1])
