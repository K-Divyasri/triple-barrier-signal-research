"""Daily volatility estimation -- the yardstick the triple-barrier method
scales its profit-take/stop-loss barriers by.

A fixed +/-2% barrier is wrong on both a calm day and a violent one. Lopez
de Prado's fix: estimate a rolling volatility of daily returns and set each
event's barriers as a MULTIPLE of that day's own volatility, so a barrier
means "a 1-sigma move," not an arbitrary constant, on every single day.
"""

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """An EWMA estimate of daily volatility, using a lagged return series.

    Uses the return from `close[t-1]` to `close[t]`, evaluated with an
    exponentially-weighted standard deviation over `span` days -- more
    weight on recent days, without a hard cutoff.
    """
    returns = close.pct_change()
    vol = returns.ewm(span=span, min_periods=span // 4).std()
    return vol.rename("daily_vol")


def realized_returns(close: pd.Series) -> pd.Series:
    """Simple percentage returns, for convenience elsewhere in the package."""
    return close.pct_change().rename("returns")


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().rename("log_returns")
