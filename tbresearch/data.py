"""Price data: a deterministic synthetic generator (default, no internet
needed) and an optional real Yahoo Finance loader.

Every notebook, lab, and test in this project runs on the synthetic series
by default -- same "offline works, real data is a flag away" pattern used
across this whole roadmap. Real market data is one function call away
(`load_close_prices(real=True)`), never required.
"""

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(n_days: int = 1000, seed: int = 42, start_price: float = 100.0) -> pd.DataFrame:
    """A regime-switching geometric Brownian motion, business-day indexed.

    Three drift/volatility regimes rotate through the series so a signal
    (and its labels) actually has directional structure to find -- a pure
    random walk gives a triple-barrier labeler nothing but noise to label.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)

    regimes = [
        (0.0006, 0.010),   # mild uptrend, normal vol
        (-0.0004, 0.018),  # mild downtrend, higher vol
        (0.0002, 0.007),   # flat/quiet
    ]
    regime_len = n_days // len(regimes) + 1

    mus = np.repeat([r[0] for r in regimes], regime_len)[:n_days]
    sigmas = np.repeat([r[1] for r in regimes], regime_len)[:n_days]

    log_returns = rng.normal(loc=mus, scale=sigmas)
    close = start_price * np.exp(np.cumsum(log_returns))

    # Build plausible OHLC around each day's close from a small intraday range.
    intraday_range = np.abs(rng.normal(loc=0.006, scale=0.003, size=n_days)).clip(0.001, None)
    open_ = np.empty(n_days)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + intraday_range * rng.uniform(0.2, 1.0, n_days))
    low = np.minimum(open_, close) * (1 - intraday_range * rng.uniform(0.2, 1.0, n_days))
    volume = rng.integers(1_000_000, 5_000_000, n_days)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def load_close_prices(
    symbol: str = "SYNTH",
    n_days: int = 1000,
    seed: int = 42,
    real: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """Return a close-price Series. `real=True` fetches `symbol` from Yahoo
    Finance via yfinance (needs internet, no API key); otherwise returns the
    synthetic series above.
    """
    if not real:
        return generate_synthetic_ohlcv(n_days=n_days, seed=seed)["close"].rename(symbol)

    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r} -- check the symbol/date range or your connection")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance can return a 1-col DataFrame depending on version
        close = close.iloc[:, 0]
    return close.rename(symbol)
