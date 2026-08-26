import pandas as pd

from tbresearch.sampling import cusum_filter


def test_cusum_fires_on_a_sustained_move():
    dates = pd.bdate_range("2024-01-01", periods=10)
    # flat, then a sustained run up that should trip a 3% threshold
    values = [100, 100, 100, 101, 102, 103.5, 103, 103, 103, 103]
    close = pd.Series(values, index=dates, dtype=float)

    events = cusum_filter(close, threshold=0.03)
    assert len(events) >= 1
    assert events[0] in dates


def test_cusum_stays_quiet_on_pure_noise_below_threshold():
    dates = pd.bdate_range("2024-01-01", periods=10)
    values = [100, 100.1, 99.9, 100.05, 99.95, 100.1, 99.9, 100, 100.05, 99.95]
    close = pd.Series(values, index=dates, dtype=float)

    events = cusum_filter(close, threshold=0.05)
    assert len(events) == 0


def test_cusum_resets_after_firing():
    dates = pd.bdate_range("2024-01-01", periods=20)
    # two separate run-ups separated by a flat patch
    values = (
        [100, 101, 102.5]
        + [102.5] * 5
        + [102.5, 103.5, 105.2]
        + [105.2] * 9
    )
    close = pd.Series(values[:20], index=dates, dtype=float)

    events = cusum_filter(close, threshold=0.02)
    assert len(events) >= 2  # should fire once per run-up, not just once total


def test_threshold_as_series_uses_local_bar():
    dates = pd.bdate_range("2024-01-01", periods=10)
    values = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    close = pd.Series(values, index=dates, dtype=float)
    threshold = pd.Series(0.005, index=dates)  # a tight, constant-via-Series bar

    events = cusum_filter(close, threshold)
    assert len(events) > 0
