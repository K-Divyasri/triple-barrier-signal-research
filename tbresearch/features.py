"""A small, deliberately simple technical-indicator feature set.

This project is about the LABELING and CROSS-VALIDATION machinery, not
feature engineering -- these features exist to give a classifier something
plausible to chew on, not to be a competitive alpha signal on their own.
"""

import numpy as np
import pandas as pd


def build_features(close: pd.Series) -> pd.DataFrame:
    """Momentum, moving-average, and volatility features, all computed
    causally (only using information available up to and including each
    row's own timestamp -- no look-ahead)."""
    returns = close.pct_change()

    feats = pd.DataFrame(index=close.index)
    feats["ret_1d"] = returns
    feats["ret_5d"] = close.pct_change(5)
    feats["ret_10d"] = close.pct_change(10)

    ma_fast = close.rolling(10).mean()
    ma_slow = close.rolling(50).mean()
    feats["ma_ratio"] = ma_fast / ma_slow - 1.0

    feats["vol_10d"] = returns.rolling(10).std()
    feats["vol_50d"] = returns.rolling(50).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feats["rsi_14"] = 100 - (100 / (1 + rs))

    feats["dist_from_high_20d"] = close / close.rolling(20).max() - 1.0

    return feats


def features_at_events(features: pd.DataFrame, t_events) -> pd.DataFrame:
    """Slice the feature matrix down to just the event timestamps, dropping
    any event where a feature is still NaN (the rolling-window warm-up
    period at the start of the series)."""
    aligned = features.reindex(t_events)
    return aligned.dropna()
