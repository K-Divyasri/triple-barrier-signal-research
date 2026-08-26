"""Sample uniqueness: correcting for overlapping labels.

Triple-barrier labels span an INTERVAL (event start to touch time), not a
single point. When two events' intervals overlap, they're partly describing
the same underlying price move -- treating them as independent samples
(which every standard ML model assumes) double-counts that move. Average
uniqueness weights down samples that overlap heavily with others, so a
classifier isn't implicitly told the same story five times over.
"""

import pandas as pd


def num_concurrent_events(close_index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """At every timestamp in `close_index`, how many labels' [t0, t1]
    windows are currently "open" (still accumulating their outcome)."""
    count = pd.Series(0, index=close_index, dtype=int)
    for t0, t1_i in t1.items():
        count.loc[t0:t1_i] += 1
    return count


def average_uniqueness(t1: pd.Series, conc_events: pd.Series) -> pd.Series:
    """For each event, the average of 1/concurrency over its own lifespan.

    An event that never overlaps anything gets weight 1.0 (fully unique). An
    event whose window is entirely covered by 3 other concurrent events
    gets weight ~0.25 (1 / 4, itself included) for its whole span.
    """
    weights = pd.Series(index=t1.index, dtype=float)
    for t0, t1_i in t1.items():
        window = conc_events.loc[t0:t1_i]
        weights.loc[t0] = (1.0 / window).mean() if len(window) else 1.0
    return weights.rename("uniqueness")
