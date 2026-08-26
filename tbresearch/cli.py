"""python -m tbresearch run [--symbol SYNTH] [--real] [--n-splits 5] [--embargo 0.02]"""

import argparse
import sys

from .data import load_close_prices
from .evaluation import compare_naive_vs_purged
from .pipeline import build_research_dataset


def cmd_run(args: argparse.Namespace) -> int:
    close = load_close_prices(symbol=args.symbol, n_days=args.n_days, seed=args.seed, real=args.real)
    dataset = build_research_dataset(close, cusum_k=args.cusum_k, num_days_vertical=args.vertical_days, pt_sl=(args.pt, args.sl))

    print(f"price series: {len(close)} days ({close.index[0].date()} .. {close.index[-1].date()})")
    print(f"triple-barrier events labeled: {len(dataset.events)} ({len(dataset.X)} usable after feature warmup)")
    print(f"label distribution:\n{dataset.y.value_counts().sort_index()}")
    print()

    reports = compare_naive_vs_purged(
        dataset.X, dataset.y, dataset.sample_weight, dataset.t1, n_splits=args.n_splits, pct_embargo=args.embargo, random_state=args.seed
    )

    for name, report in reports.items():
        print(f"[{name}] {report.method}")
        print(f"  mean train accuracy: {report.mean_train_accuracy:.3f}")
        print(f"  mean test accuracy:  {report.mean_test_accuracy:.3f}")
        print(f"  mean test F1:        {report.mean_test_f1:.3f}")
        print(f"  train-test gap:      {report.train_test_gap:.3f}")
        print()

    gap_diff = reports["naive"].mean_test_accuracy - reports["purged"].mean_test_accuracy
    print(f"naive CV overstates test accuracy by {gap_diff:+.3f} vs purged+embargoed CV")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tbresearch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline and compare naive vs purged CV")
    p_run.add_argument("--symbol", default="SYNTH")
    p_run.add_argument("--real", action="store_true", help="fetch real data via yfinance instead of the synthetic series")
    p_run.add_argument("--n-days", type=int, default=1000)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--cusum-k", type=float, default=1.0)
    p_run.add_argument("--vertical-days", type=int, default=15)
    p_run.add_argument("--pt", type=float, default=2.5, help="profit-take barrier multiple of target volatility")
    p_run.add_argument("--sl", type=float, default=2.5, help="stop-loss barrier multiple of target volatility")
    p_run.add_argument("--n-splits", type=int, default=5)
    p_run.add_argument("--embargo", type=float, default=0.02)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
