"""Writes a sample synthetic price series + a canonical pipeline run's
results to data/, so notebooks/labs/the hosted app can load pre-computed
results instantly instead of re-running the RandomForest fit every time.

Run: python generate_data.py
"""

import json
from pathlib import Path

from tbresearch.data import load_close_prices
from tbresearch.evaluation import compare_naive_vs_purged
from tbresearch.pipeline import build_research_dataset

DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    close = load_close_prices(n_days=1000, seed=42)
    close.to_frame(name="close").to_csv(DATA_DIR / "synthetic_prices.csv")

    dataset = build_research_dataset(close)
    reports = compare_naive_vs_purged(dataset.X, dataset.y, dataset.sample_weight, dataset.t1, n_splits=5, pct_embargo=0.02, random_state=42)

    summary = {
        "n_days": len(close),
        "n_events": len(dataset.X),
        "label_counts": {str(k): int(v) for k, v in dataset.y.value_counts().items()},
        "naive": {
            "mean_train_accuracy": reports["naive"].mean_train_accuracy,
            "mean_test_accuracy": reports["naive"].mean_test_accuracy,
            "mean_test_f1": reports["naive"].mean_test_f1,
        },
        "purged": {
            "mean_train_accuracy": reports["purged"].mean_train_accuracy,
            "mean_test_accuracy": reports["purged"].mean_test_accuracy,
            "mean_test_f1": reports["purged"].mean_test_f1,
        },
    }
    summary["naive_overstates_by"] = summary["naive"]["mean_test_accuracy"] - summary["purged"]["mean_test_accuracy"]

    with open(DATA_DIR / "canonical_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote {DATA_DIR / 'synthetic_prices.csv'} ({len(close)} rows)")
    print(f"wrote {DATA_DIR / 'canonical_run_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
