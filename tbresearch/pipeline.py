"""Wires the whole pipeline together: prices -> events -> labels -> weights
-> features -> a leakage-free-vs-naive CV comparison.
"""

from dataclasses import dataclass

import pandas as pd

from .features import build_features, features_at_events
from .labeling import apply_triple_barrier, get_labels, get_vertical_barriers
from .sampling import cusum_filter
from .volatility import daily_volatility
from .weights import average_uniqueness, num_concurrent_events


@dataclass
class ResearchDataset:
    X: pd.DataFrame
    y: pd.Series
    sample_weight: pd.Series
    t1: pd.Series          # aligned to X.index: event end (touch) times
    events: pd.DataFrame   # the raw triple-barrier events (t0-indexed)
    labels: pd.DataFrame   # the raw labels (t0-indexed), before feature alignment


def build_research_dataset(
    close: pd.Series,
    cusum_k: float = 1.0,
    vol_span: int = 100,
    num_days_vertical: int = 15,
    pt_sl: tuple[float, float] = (2.5, 2.5),
    min_ret: float = 0.0,
) -> ResearchDataset:
    vol = daily_volatility(close, span=vol_span)

    threshold = cusum_k * vol
    t_events = cusum_filter(close, threshold)

    vertical_barriers = get_vertical_barriers(t_events, close.index, num_days=num_days_vertical)
    target = vol.reindex(t_events)

    events = apply_triple_barrier(close, t_events, pt_sl, target, vertical_barriers, min_ret=min_ret)
    labels = get_labels(close, events)

    conc = num_concurrent_events(close.index, events["t1"])
    uniqueness = average_uniqueness(events["t1"], conc)

    features = build_features(close)
    X = features_at_events(features, events.index)

    common_idx = X.index.intersection(labels.index)
    X = X.loc[common_idx]
    y = labels.loc[common_idx, "label"]
    sample_weight = uniqueness.loc[common_idx]
    t1 = events.loc[common_idx, "t1"]

    return ResearchDataset(X=X, y=y, sample_weight=sample_weight, t1=t1, events=events, labels=labels)
