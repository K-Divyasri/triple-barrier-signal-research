# Triple-Barrier Signal Research

A leakage-free signal research pipeline. `tbresearch` labels a price series
with the triple-barrier method, weights overlapping labels by their
uniqueness, and cross-validates a classifier with a purged + embargoed
K-Fold splitter that actually prevents the leakage naive K-Fold silently
allows.

> Educational project, not financial advice.

## Layout

```
triple-barrier-signal-research/
├── tbresearch/               the package
│   ├── data.py                 synthetic OHLCV generator + optional yfinance loader
│   ├── volatility.py            daily volatility estimator (scales the barriers)
│   ├── sampling.py              symmetric CUSUM filter (decides WHEN to label)
│   ├── labeling.py               triple-barrier labeling (the core method)
│   ├── weights.py                concurrency + average uniqueness
│   ├── features.py               a small technical-indicator feature set
│   ├── cross_validation.py       PurgedKFold -- the leak-free splitter
│   ├── evaluation.py             naive-vs-purged CV comparison
│   ├── pipeline.py               wires it all into one ResearchDataset
│   └── cli.py                    python -m tbresearch run ...
├── tests/                     43 pytest, all offline, no network needed
├── data/                      sample synthetic prices + a canonical run summary
├── generate_data.py           writes the files in data/
└── requirements.txt
```

## Run it

```powershell
pip install -r requirements.txt
python generate_data.py
python -m pytest -q                 # 43 passed
python -m tbresearch run            # the whole pipeline + the headline comparison
```

Expect output ending in something like:

```
[naive] naive KFold (shuffled)
  mean test accuracy:  0.627
[purged] purged + embargoed KFold
  mean test accuracy:  0.551

naive CV overstates test accuracy by +0.076 vs purged+embargoed CV
```

That gap, naive CV looking better than it should, is the entire point of
this project. See `architecture.md` for the mechanism and why it took some
real parameter tuning to make the effect this visible.

## Use real market data instead of the synthetic series

```powershell
python -m tbresearch run --real --symbol SPY
```

Needs internet (no API key needed; `yfinance` talks to Yahoo Finance directly).
Everything downstream of `load_close_prices` is unchanged; only the price
series itself is real instead of synthetic.

## Tests

```powershell
python -m pytest -q
```

43 tests: triple-barrier labeling against hand-built price paths with known
outcomes (8), volatility (3), CUSUM sampling (4), sample weights/uniqueness
(5), PurgedKFold's zero-overlap guarantee (5), the full pipeline's shape
guarantees (5), the naive-vs-purged headline comparison (3), features (4),
synthetic data generation (5), plus a small remainder.
