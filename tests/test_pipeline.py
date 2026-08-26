from tbresearch.data import load_close_prices
from tbresearch.pipeline import build_research_dataset


def test_research_dataset_shapes_are_consistent():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)

    assert len(ds.X) == len(ds.y) == len(ds.sample_weight) == len(ds.t1)
    assert set(ds.X.index) == set(ds.y.index) == set(ds.t1.index)


def test_labels_are_only_plus_or_minus_one():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    assert set(ds.y.unique()).issubset({-1.0, 0.0, 1.0})


def test_sample_weights_are_between_zero_and_one():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    assert (ds.sample_weight > 0).all()
    assert (ds.sample_weight <= 1.0).all()


def test_t1_never_precedes_t0():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    assert (ds.t1.values >= ds.t1.index.values).all()


def test_more_events_with_lower_cusum_threshold():
    close = load_close_prices(n_days=1000, seed=42)
    loose = build_research_dataset(close, cusum_k=2.0)
    tight = build_research_dataset(close, cusum_k=0.5)
    assert len(tight.X) > len(loose.X)
