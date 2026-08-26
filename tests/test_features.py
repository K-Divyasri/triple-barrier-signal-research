import pandas as pd

from tbresearch.data import load_close_prices
from tbresearch.features import build_features, features_at_events


def test_build_features_has_expected_columns():
    close = load_close_prices(n_days=200, seed=42)
    feats = build_features(close)
    expected = {"ret_1d", "ret_5d", "ret_10d", "ma_ratio", "vol_10d", "vol_50d", "rsi_14", "dist_from_high_20d"}
    assert expected.issubset(set(feats.columns))


def test_build_features_has_no_lookahead_warmup_nans_at_start():
    close = load_close_prices(n_days=200, seed=42)
    feats = build_features(close)
    assert feats.iloc[0].isna().any()  # the very first row can't have any rolling features yet


def test_features_at_events_drops_warmup_rows():
    close = load_close_prices(n_days=200, seed=42)
    feats = build_features(close)
    events = close.index[:5]  # the first 5 days -- guaranteed still in warm-up
    aligned = features_at_events(feats, events)
    assert len(aligned) < len(events)  # at least some warm-up rows get dropped


def test_features_at_events_keeps_only_requested_timestamps():
    close = load_close_prices(n_days=200, seed=42)
    feats = build_features(close)
    events = close.index[100:105]
    aligned = features_at_events(feats, events)
    assert set(aligned.index).issubset(set(events))
