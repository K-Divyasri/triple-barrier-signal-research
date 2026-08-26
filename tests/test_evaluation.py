from tbresearch.data import load_close_prices
from tbresearch.evaluation import compare_naive_vs_purged
from tbresearch.pipeline import build_research_dataset


def test_naive_cv_overstates_test_accuracy_vs_purged():
    """The headline result this whole project exists to demonstrate: on the
    canonical seed/config, naive shuffled K-Fold reports HIGHER test accuracy
    than purged+embargoed K-Fold, on the exact same data and model -- because
    it's leaking overlapping-window information across the fold boundary,
    not because it's a better cross-validation scheme."""
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    reports = compare_naive_vs_purged(ds.X, ds.y, ds.sample_weight, ds.t1, n_splits=5, pct_embargo=0.02, random_state=42)

    assert reports["naive"].mean_test_accuracy > reports["purged"].mean_test_accuracy


def test_reports_have_one_fold_result_per_split():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    reports = compare_naive_vs_purged(ds.X, ds.y, ds.sample_weight, ds.t1, n_splits=5, random_state=42)

    assert len(reports["naive"].folds) == 5
    assert len(reports["purged"].folds) <= 5  # purged folds can be skipped if a fold empties out


def test_train_test_gap_is_computed_correctly():
    close = load_close_prices(n_days=1000, seed=42)
    ds = build_research_dataset(close)
    reports = compare_naive_vs_purged(ds.X, ds.y, ds.sample_weight, ds.t1, n_splits=5, random_state=42)
    report = reports["naive"]
    assert report.train_test_gap == report.mean_train_accuracy - report.mean_test_accuracy
