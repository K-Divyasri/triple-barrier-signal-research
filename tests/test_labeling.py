import pandas as pd
import pytest

from tbresearch.labeling import apply_triple_barrier, get_labels, get_vertical_barriers


@pytest.fixture
def toy_close():
    dates = pd.bdate_range("2024-01-01", periods=10)
    # day0=100 (event), day1=101, day2=99, day3=105 (upper touch, +5%), day4=95, ...
    values = [100, 101, 99, 105, 95, 100, 100, 100, 100, 100]
    return pd.Series(values, index=dates, dtype=float)


def test_vertical_barrier_offsets_by_trading_days(toy_close):
    t_events = toy_close.index[[0, 4]]
    vb = get_vertical_barriers(t_events, toy_close.index, num_days=5)
    assert vb.iloc[0] == toy_close.index[5]
    assert vb.iloc[1] == toy_close.index[9]  # clipped to the last available day


def test_vertical_barrier_clips_at_end_of_series(toy_close):
    t_events = toy_close.index[[8]]
    vb = get_vertical_barriers(t_events, toy_close.index, num_days=5)
    assert vb.iloc[0] == toy_close.index[-1]


def test_upper_barrier_touch(toy_close):
    t_events = toy_close.index[[0]]
    target = pd.Series([0.02], index=t_events)
    vb = get_vertical_barriers(t_events, toy_close.index, num_days=5)
    events = apply_triple_barrier(toy_close, t_events, pt_sl=(1.0, 1.0), target=target, vertical_barriers=vb)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["touch_type"] == "upper"
    assert row["t1"] == toy_close.index[3]  # day3, first day return exceeds +2%


def test_lower_barrier_touch():
    dates = pd.bdate_range("2024-02-01", periods=10)
    values = [100, 100, 100, 97, 100, 100, 100, 100, 100, 100]  # day3 = -3%
    close = pd.Series(values, index=dates, dtype=float)
    t_events = close.index[[0]]
    target = pd.Series([0.02], index=t_events)
    vb = get_vertical_barriers(t_events, close.index, num_days=5)
    events = apply_triple_barrier(close, t_events, pt_sl=(1.0, 1.0), target=target, vertical_barriers=vb)

    row = events.iloc[0]
    assert row["touch_type"] == "lower"
    assert row["t1"] == close.index[3]


def test_vertical_touch_when_no_barrier_hit():
    dates = pd.bdate_range("2024-03-01", periods=10)
    values = [100, 100.5, 99.5, 100.2, 99.8, 100.1, 100, 100, 100, 100]  # stays inside +/-2%
    close = pd.Series(values, index=dates, dtype=float)
    t_events = close.index[[0]]
    target = pd.Series([0.02], index=t_events)
    vb = get_vertical_barriers(t_events, close.index, num_days=5)
    events = apply_triple_barrier(close, t_events, pt_sl=(1.0, 1.0), target=target, vertical_barriers=vb)

    row = events.iloc[0]
    assert row["touch_type"] == "vertical"
    assert row["t1"] == vb.iloc[0]


def test_get_labels_sign_matches_touch_direction(toy_close):
    t_events = toy_close.index[[0]]
    target = pd.Series([0.02], index=t_events)
    vb = get_vertical_barriers(t_events, toy_close.index, num_days=5)
    events = apply_triple_barrier(toy_close, t_events, pt_sl=(1.0, 1.0), target=target, vertical_barriers=vb)
    labels = get_labels(toy_close, events)

    assert labels.iloc[0]["label"] == 1.0
    assert labels.iloc[0]["ret"] == pytest.approx(0.05)


def test_min_ret_skips_low_volatility_events(toy_close):
    t_events = toy_close.index[[0]]
    target = pd.Series([0.0001], index=t_events)  # below min_ret
    vb = get_vertical_barriers(t_events, toy_close.index, num_days=5)
    events = apply_triple_barrier(toy_close, t_events, pt_sl=(1.0, 1.0), target=target, vertical_barriers=vb, min_ret=0.001)
    assert len(events) == 0


def test_one_sided_barrier_disables_the_other_side():
    dates = pd.bdate_range("2024-04-01", periods=10)
    values = [100, 100, 100, 97, 100, 100, 100, 100, 100, 100]  # would hit LOWER at day3
    close = pd.Series(values, index=dates, dtype=float)
    t_events = close.index[[0]]
    target = pd.Series([0.02], index=t_events)
    vb = get_vertical_barriers(t_events, close.index, num_days=5)
    # disable the lower barrier (sl=0) -- should ride to the vertical barrier instead
    events = apply_triple_barrier(close, t_events, pt_sl=(1.0, 0.0), target=target, vertical_barriers=vb)
    assert events.iloc[0]["touch_type"] == "vertical"
