import pandas as pd

from tbresearch.data import generate_synthetic_ohlcv, load_close_prices


def test_synthetic_ohlcv_shape_and_columns():
    df = generate_synthetic_ohlcv(n_days=200, seed=42)
    assert len(df) == 200
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_synthetic_ohlcv_high_low_bracket_open_close():
    df = generate_synthetic_ohlcv(n_days=100, seed=1)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_is_deterministic_given_a_seed():
    a = generate_synthetic_ohlcv(n_days=50, seed=7)
    b = generate_synthetic_ohlcv(n_days=50, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_series():
    a = generate_synthetic_ohlcv(n_days=50, seed=1)
    b = generate_synthetic_ohlcv(n_days=50, seed=2)
    assert not a["close"].equals(b["close"])


def test_load_close_prices_offline_default():
    close = load_close_prices(n_days=100, seed=42)
    assert len(close) == 100
    assert close.name == "SYNTH"
