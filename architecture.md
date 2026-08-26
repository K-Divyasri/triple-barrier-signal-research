# Architecture — and how I'd take this further

## The pipeline, end to end

```
close prices
     |
     v
daily_volatility()  ---->  cusum_filter()  ---->  t_events (sparse timestamps)
     |                                                  |
     v                                                  v
target width (vol at each event)          get_vertical_barriers()
     |                                                  |
     +------------------> apply_triple_barrier() <------+
                                  |
                                  v
                          events (t0 -> t1, touch_type)
                                  |
                    +-------------+-------------+
                    v                           v
              get_labels()             num_concurrent_events()
                    |                           |
                    v                           v
              y (label, ret)          average_uniqueness() -> sample_weight
                    |                           |
                    +-------------+-------------+
                                  v
                          build_features() + features_at_events()
                                  |
                                  v
                    X, y, sample_weight, t1  (ResearchDataset)
                                  |
                    +-------------+-------------+
                    v                           v
            naive KFold (shuffled)      PurgedKFold (leak-free)
                    |                           |
                    v                           v
              inflated test score      honest test score
```

## The headline result, and why it's real

On the canonical synthetic series (seed 42, 1000 trading days, default
config): **naive shuffled K-Fold reports 62.7% test accuracy; purged +
embargoed K-Fold reports 55.1%** on the exact same data, exact same
`RandomForestClassifier`, exact same everything else. Naive CV overstates
by +7.6 percentage points. That gap is not a bug and not a coincidence of
one random seed — it's stable in sign across every seed tried while
building this (seeds 1, 7, 42, 99 all showed naive beating purged by
5-22 points).

The mechanism: triple-barrier labels span an *interval* (`t0` to `t1`), and
neighboring events' intervals heavily overlap (median event gap ~3 trading
days vs. median label window ~15-28 days in this project's default
config). A **shuffled** K-Fold split scatters events across folds at
random, so a huge fraction of test-fold events have a temporally
overlapping, highly-correlated neighbor sitting in the training fold — the
model partially "recognizes" the test answer because it saw a near-
duplicate slice of the same price move during training. **PurgedKFold**
keeps folds chronological and explicitly drops any training sample whose
window overlaps the test fold's window (plus a short embargo period right
after), so this shortcut isn't available — the reported score reflects
genuine out-of-sample generalization instead.

## Why the numbers needed tuning to actually show this

The first configuration tried (loose CUSUM threshold, tight 1-sigma
barriers, a shallow shrunk RandomForest) showed almost NO gap — in one
run, purged CV even scored *higher* than naive. Diagnosis: with tight
barriers, most events resolved (hit a barrier) within 2-4 days, which is
about the same as the average gap between events — so windows barely
overlapped in the first place, and there was nothing for a shuffled split
to leak. Widening the barriers to 2.5-sigma and the vertical horizon to 15
days made most events ride out longer, producing real overlap (median
window ~15-28 days vs. ~3-day gaps) — and only then did the leakage
mechanism have room to actually happen. This is worth remembering as a
general lesson: **a compatibility/leakage demonstration is only as
convincing as the configuration that makes the effect big enough to see.**
The methodology (PurgedKFold's purging logic) was correct the whole time;
what changed was giving it enough real overlap to have something to purge.

## Design choices worth naming

- **CUSUM event sampling, not "label every single day."** Labeling every
  bar is both wasteful (most days show nothing informative) and, ironically,
  a common source of the SAME overlap problem this project is about — see
  `knowledge/06`.
- **Volatility-scaled barriers, not fixed percentages.** A flat +/-2%
  barrier is meaningless across a calm regime and a violent one; scaling by
  each event's own local volatility (`daily_volatility`) makes "the barrier"
  mean the same thing — "about N standard deviations" — every day.
- **Sample weights via average uniqueness, applied even though this
  project's headline demo is about CV, not weighting.** Overlapping labels
  overweight the same underlying price move in the LOSS function too, not
  just in cross-validation; `RandomForestClassifier`'s `sample_weight`
  argument is passed to `.fit()` in both the naive and purged runs, so the
  headline comparison isn't comparing an unweighted model to a weighted
  one — the only difference between "naive" and "purged" is the fold
  assignment.
- **A single RandomForest, not a tuned model.** Feature engineering and
  model selection are deliberately minimal (see `features.py`'s own
  docstring) — the entire point of this project is the labeling and
  validation machinery, not chasing alpha.

## What's simulated vs what's real

| Piece | Real or simulated |
|---|---|
| Price data | Synthetic by default (deterministic, regime-switching GBM); real Yahoo Finance data one flag away (`load_close_prices(real=True)`) |
| Triple-barrier labeling | Real implementation, unit-tested against hand-built price paths with known barrier touches |
| CUSUM event sampling | Real implementation of the symmetric CUSUM filter from Lopez de Prado's book |
| Purged + embargoed CV | Real implementation, unit-tested to prove ZERO train/test window overlap after purging |
| The naive-vs-purged gap | Real, measured, reproducible — not asserted, computed fresh every run via `generate_data.py` / `python -m tbresearch run` |

## If I productionized this further

- **Multiple assets and cross-sectional labels.** Everything here runs on
  one price series; a real signal research pipeline would run this across
  a universe and need `PurgedKFold` extended to purge across a
  cross-section too (an event on asset A can still leak into asset B's
  fold if they're driven by a shared macro factor on the same days —
  Lopez de Prado's book calls this out explicitly as a common
  under-purging mistake).
- **The primary/secondary "meta-labeling" step.** This project labels
  every CUSUM-triggered event directly. A more complete research pipeline
  runs a cheap primary model to decide DIRECTION first, then uses
  triple-barrier + a secondary classifier only to decide SIZE/whether to
  act — see `knowledge/09` for why that split is usually the better
  design.
- **A proper backtest, not just a classification metric.** Test accuracy
  is a proxy; the real question is whether trading on these labels would
  have made money net of costs. Project 8 (`08-backtesting-framework`,
  this same track) is the natural next step to plug this signal into.
