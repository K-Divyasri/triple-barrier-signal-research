import pandas as pd
import pytest

from tbresearch.weights import average_uniqueness, num_concurrent_events


def test_non_overlapping_events_have_concurrency_one():
    dates = pd.bdate_range("2024-01-01", periods=20)
    t1 = pd.Series([dates[2], dates[7], dates[12]], index=[dates[0], dates[5], dates[10]])

    conc = num_concurrent_events(dates, t1)
    assert conc.loc[dates[0]] == 1
    assert conc.loc[dates[1]] == 1


def test_overlapping_events_raise_concurrency():
    dates = pd.bdate_range("2024-01-01", periods=10)
    t1 = pd.Series([dates[5], dates[6]], index=[dates[0], dates[3]])  # both open during dates[3..5]

    conc = num_concurrent_events(dates, t1)
    assert conc.loc[dates[4]] == 2  # both windows cover day 4
    assert conc.loc[dates[0]] == 1  # only the first window covers day 0


def test_average_uniqueness_is_one_when_fully_isolated():
    dates = pd.bdate_range("2024-01-01", periods=10)
    t1 = pd.Series([dates[2]], index=[dates[0]])
    conc = num_concurrent_events(dates, t1)
    uniq = average_uniqueness(t1, conc)
    assert uniq.iloc[0] == 1.0


def test_average_uniqueness_drops_when_overlapping():
    dates = pd.bdate_range("2024-01-01", periods=10)
    # event 2 (dates[1]..dates[5]) is fully covered by event 1 (dates[0]..dates[5]) the
    # whole time it's open, so event 2's own span has concurrency exactly 2 throughout.
    t1 = pd.Series([dates[5], dates[5]], index=[dates[0], dates[1]])

    conc = num_concurrent_events(dates, t1)
    uniq = average_uniqueness(t1, conc)
    assert uniq.loc[dates[1]] == pytest.approx(0.5)
    assert uniq.loc[dates[0]] < 1.0  # event 1 overlaps event 2 for part (not all) of its span


def test_average_uniqueness_between_zero_and_one():
    dates = pd.bdate_range("2024-01-01", periods=30)
    t1 = pd.Series([dates[i + 5] for i in range(10)], index=[dates[i] for i in range(10)])  # heavy overlap
    conc = num_concurrent_events(dates, t1)
    uniq = average_uniqueness(t1, conc)
    assert (uniq > 0).all()
    assert (uniq <= 1).all()
