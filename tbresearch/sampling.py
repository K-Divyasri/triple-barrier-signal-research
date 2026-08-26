"""Event sampling: deciding WHEN to even consider placing a bet.

Labeling every single day is wasteful and statistically lazy -- most days
nothing interesting happens. The symmetric CUSUM filter instead accumulates
positive and negative return surprises and fires an "event" only once
enough has accumulated in one direction, resetting after each fire. This
gives the triple-barrier labeler a sparser, more informative set of
timestamps to actually label, closer to how a real signal would only act
when something has genuinely moved.
"""

import numpy as np
import pandas as pd


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter (Lopez de Prado, Advances in Financial Machine
    Learning, section 2.5.2.1). Fires a timestamp whenever the cumulative sum
    of returns since the last event exceeds `threshold` in either direction.

    `threshold` can be a single float (a global bar) or a Series aligned to
    `close`'s index (e.g. `k * daily_volatility(close)`, using a moving,
    data-driven bar instead of a hand-picked constant).
    """
    returns = close.pct_change().fillna(0.0)
    if isinstance(threshold, pd.Series):
        threshold = threshold.reindex(close.index).ffill().fillna(threshold.dropna().iloc[0] if not threshold.dropna().empty else 0.01)

    events = []
    pos_sum, neg_sum = 0.0, 0.0
    for t in returns.index[1:]:
        r = returns.loc[t]
        pos_sum = max(0.0, pos_sum + r)
        neg_sum = min(0.0, neg_sum + r)
        bar = threshold.loc[t] if isinstance(threshold, pd.Series) else threshold
        if pos_sum > bar:
            events.append(t)
            pos_sum = 0.0
        elif neg_sum < -bar:
            events.append(t)
            neg_sum = 0.0

    return pd.DatetimeIndex(events)
