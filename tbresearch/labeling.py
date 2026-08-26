"""Triple-barrier labeling (Lopez de Prado, Advances in Financial Machine
Learning, chapter 3).

The idea: instead of labeling a bar by "did price go up or down over the
next N days" (arbitrary, ignores how the path got there), set THREE
barriers around each event -- an upper profit-take, a lower stop-loss, and
a vertical time limit -- and label by whichever one price touches first.
That's a much closer model of how a real position actually gets closed.
"""

import numpy as np
import pandas as pd


def get_vertical_barriers(t_events: pd.DatetimeIndex, close_index: pd.DatetimeIndex, num_days: int = 5) -> pd.Series:
    """For each event start time, the timestamp `num_days` TRADING days
    later (not calendar days) -- capped at the last available price if the
    event is too close to the end of the data.
    """
    positions = close_index.searchsorted(t_events) + num_days
    positions = positions.clip(max=len(close_index) - 1)
    vertical_times = close_index[positions]
    return pd.Series(vertical_times, index=t_events, name="t1")


def apply_triple_barrier(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    target: pd.Series,
    vertical_barriers: pd.Series,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Walk each event's price path and record which barrier it touches first.

    `target` is a per-event volatility estimate (see `volatility.daily_volatility`)
    that SCALES the barrier widths, so a barrier means "N standard deviations,"
    not a fixed percentage. `pt_sl = (profit_take_mult, stop_loss_mult)`; pass
    0 for either side to disable that barrier (a one-sided bet).

    Returns a DataFrame indexed by event start time (t0) with columns:
    `t1` (the ACTUAL touch time, <= the vertical barrier), `trgt` (the target
    width used), `touch_type` ('upper' / 'lower' / 'vertical').
    """
    pt_mult, sl_mult = pt_sl
    records = []

    for t0 in t_events:
        trgt = target.get(t0, np.nan)
        if pd.isna(trgt) or trgt < min_ret:
            continue  # not enough expected movement to bother labeling this event

        t1 = vertical_barriers.loc[t0]
        path = close.loc[t0:t1]
        if len(path) < 2:
            continue
        path_returns = path / path.iloc[0] - 1.0

        upper_hits = path_returns[path_returns > pt_mult * trgt] if pt_mult > 0 else pd.Series(dtype=float)
        lower_hits = path_returns[path_returns < -sl_mult * trgt] if sl_mult > 0 else pd.Series(dtype=float)

        upper_time = upper_hits.index[0] if len(upper_hits) else None
        lower_time = lower_hits.index[0] if len(lower_hits) else None

        candidates = [(t, "upper") for t in [upper_time] if t is not None] + [
            (t, "lower") for t in [lower_time] if t is not None
        ]
        if candidates:
            touch_time, touch_type = min(candidates, key=lambda x: x[0])
        else:
            touch_time, touch_type = t1, "vertical"

        records.append({"t0": t0, "t1": touch_time, "trgt": trgt, "touch_type": touch_type})

    if not records:
        return pd.DataFrame(columns=["t1", "trgt", "touch_type"]).rename_axis("t0")
    return pd.DataFrame(records).set_index("t0")


def get_labels(close: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Turn triple-barrier events into a return + a {-1, 0, +1} label.

    label = sign of the return realized between the event start and its
    actual touch time -- regardless of which barrier caused the touch. A
    vertical-barrier touch with a genuinely flat path is the only source
    of a 0 label; upper/lower touches are +1/-1 by construction.
    """
    ret = close.loc[events["t1"]].to_numpy() / close.loc[events.index].to_numpy() - 1.0
    label = np.sign(ret)
    return pd.DataFrame({"ret": ret, "label": label, "touch_type": events["touch_type"].to_numpy()}, index=events.index)
