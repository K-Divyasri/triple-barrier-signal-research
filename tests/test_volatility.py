import numpy as np
import pandas as pd

from tbresearch.volatility import daily_volatility, log_returns, realized_returns


def test_daily_volatility_is_positive_and_nan_during_warmup():
    dates = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=dates)

    vol = daily_volatility(close, span=50)
    assert vol.iloc[:10].isna().all()  # warm-up period
    assert (vol.dropna() > 0).all()


def test_higher_input_volatility_gives_higher_estimate():
    dates = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    calm = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))), index=dates)
    wild = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300))), index=dates)

    assert daily_volatility(wild).dropna().mean() > daily_volatility(calm).dropna().mean()


def test_realized_and_log_returns_are_close_for_small_moves():
    dates = pd.bdate_range("2024-01-01", periods=5)
    close = pd.Series([100, 100.1, 100.2, 100.1, 100.3], index=dates)
    simple = realized_returns(close).dropna()
    log = log_returns(close).dropna()
    assert (simple - log).abs().max() < 1e-3
