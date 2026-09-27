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


def _synthetic_market_frame(n=180):
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = np.linspace(100.0, 160.0, n)
    return pd.DataFrame({
        "Open": base,
        "High": base + 1.0,
        "Low": base - 1.0,
        "Close": base + np.sin(np.arange(n) / 5.0),
        "Volume": np.linspace(1_000_000, 2_000_000, n),
        "VIX": np.linspace(12.0, 25.0, n),
        "TNX": np.linspace(1.0, 4.0, n),
    }, index=idx)


def test_walk_forward_preprocessing_refits_scalers_per_fold(monkeypatch):
    """Every fold must fit fresh scalers using training rows only."""
    import src.data_prep as data_prep

    x_fit_lengths = []
    y_fit_lengths = []

    original_x_fit = data_prep.RobustScaler.fit
    original_y_fit = data_prep.StandardScaler.fit

    def record_x_fit(self, values, *args, **kwargs):
        x_fit_lengths.append(len(values))
        return original_x_fit(self, values, *args, **kwargs)

    def record_y_fit(self, values, *args, **kwargs):
        y_fit_lengths.append(len(values))
        return original_y_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(data_prep.RobustScaler, "fit", record_x_fit)
    monkeypatch.setattr(data_prep.StandardScaler, "fit", record_y_fit)

    folds = data_prep.prepare_walk_forward_folds(
        _synthetic_market_frame(), window_size=20, n_splits=3,
    )

    assert len(folds) == 3
    assert len(x_fit_lengths) == len(y_fit_lengths) == 3
    assert x_fit_lengths == y_fit_lengths
    assert x_fit_lengths[0] < x_fit_lengths[1] < x_fit_lengths[2]

    for X_train, y_train, X_val, y_val in folds:
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_train) > 0
        assert len(X_val) > 0


def test_walk_forward_scaler_never_sees_validation_outlier(monkeypatch):
    """A future validation outlier must not be present in scaler.fit input."""
    import src.data_prep as data_prep

    frame = _synthetic_market_frame()
    seen_maxima = []
    original_fit = data_prep.RobustScaler.fit

    def record_fit(self, values, *args, **kwargs):
        seen_maxima.append(float(np.max(values[:, 3])))
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(data_prep.RobustScaler, "fit", record_fit)

    # Put an extreme Close value at the end. It belongs to future development
    # validation/test chronology and must not contaminate earlier scaler fits.
    frame.iloc[-1, frame.columns.get_loc("Close")] = 1_000_000.0
    data_prep.prepare_walk_forward_folds(frame, window_size=20, n_splits=3)

    assert len(seen_maxima) == 3
    assert all(value < 1_000_000.0 for value in seen_maxima)


def test_fixed_boundary_dataset_aligns_target_dates_across_windows():
    """Different lookbacks must share identical validation/test target dates."""
    import src.data_prep as data_prep

    frame = _synthetic_market_frame(n=240)
    datasets = [
        data_prep.prepare_fixed_boundary_dataset(frame, window_size=window)
        for window in (20, 30, 45, 60)
    ]

    reference_val_dates = datasets[0][-2]
    reference_test_dates = datasets[0][-1]
    for dataset in datasets[1:]:
        assert np.array_equal(dataset[-2], reference_val_dates)
        assert np.array_equal(dataset[-1], reference_test_dates)


def test_fixed_boundary_scaler_fit_is_window_independent(monkeypatch):
    """Scaler training cutoff must be fixed on raw time, not sequence count."""
    import src.data_prep as data_prep

    frame = _synthetic_market_frame(n=240)
    fit_lengths = []
    original_fit = data_prep.RobustScaler.fit

    def record_fit(self, values, *args, **kwargs):
        fit_lengths.append(len(values))
        return original_fit(self, values, *args, **kwargs)

    monkeypatch.setattr(data_prep.RobustScaler, "fit", record_fit)
    for window in (20, 30, 45, 60):
        data_prep.prepare_fixed_boundary_dataset(frame, window_size=window)

    assert len(fit_lengths) == 4
    assert len(set(fit_lengths)) == 1
