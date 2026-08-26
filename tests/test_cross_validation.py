import numpy as np
import pandas as pd
import pytest

from tbresearch.cross_validation import PurgedKFold, purge_fraction


@pytest.fixture
def overlapping_t1():
    dates = pd.bdate_range("2024-01-01", periods=100)
    # each event's window spans the next 10 days -- heavy overlap with neighbors
    t0 = dates[:90]
    t1_values = dates[10:100]
    return pd.Series(t1_values, index=t0)


def test_split_produces_no_train_test_overlap(overlapping_t1):
    n = len(overlapping_t1)
    X = np.zeros((n, 1))
    cv = PurgedKFold(n_splits=5, t1=overlapping_t1, pct_embargo=0.0)

    t0_arr = overlapping_t1.index.to_numpy()
    t1_arr = overlapping_t1.to_numpy()

    for train_idx, test_idx in cv.split(X):
        test_start = t0_arr[test_idx].min()
        test_end = t1_arr[test_idx].max()
        for i in train_idx:
            sample_overlaps = (t0_arr[i] <= test_end) and (t1_arr[i] >= test_start)
            assert not sample_overlaps, f"train sample {i} overlaps the test window"


def test_embargo_removes_samples_right_after_test_fold(overlapping_t1):
    n = len(overlapping_t1)
    X = np.zeros((n, 1))
    no_embargo = PurgedKFold(n_splits=5, t1=overlapping_t1, pct_embargo=0.0)
    with_embargo = PurgedKFold(n_splits=5, t1=overlapping_t1, pct_embargo=0.1)

    train_sizes_no_embargo = [len(train) for train, _ in no_embargo.split(X)]
    train_sizes_with_embargo = [len(train) for train, _ in with_embargo.split(X)]

    assert sum(train_sizes_with_embargo) <= sum(train_sizes_no_embargo)


def test_get_n_splits_reports_configured_value(overlapping_t1):
    cv = PurgedKFold(n_splits=4, t1=overlapping_t1)
    assert cv.get_n_splits() == 4


def test_mismatched_length_raises(overlapping_t1):
    cv = PurgedKFold(n_splits=5, t1=overlapping_t1)
    X_wrong_length = np.zeros((5, 1))
    with pytest.raises(ValueError):
        list(cv.split(X_wrong_length))


def test_purge_fraction_is_zero_when_no_overlap():
    dates = pd.bdate_range("2024-01-01", periods=50)
    # each event's window closes before the NEXT event even starts -- no overlap at all
    t0 = dates[::2][:20]
    t1 = pd.Series([t0[i] for i in range(20)], index=t0)  # zero-length windows, no overlap possible
    frac = purge_fraction(n_splits=5, t1=t1, pct_embargo=0.0)
    assert frac == pytest.approx(0.0, abs=1e-6)


def test_purge_fraction_is_positive_with_heavy_overlap(overlapping_t1):
    frac = purge_fraction(n_splits=5, t1=overlapping_t1, pct_embargo=0.0)
    assert frac > 0.0
