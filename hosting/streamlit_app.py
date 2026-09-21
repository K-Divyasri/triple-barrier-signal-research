"""
An interactive dashboard for the naive-vs-purged cross-validation story --
gives you a URL to show off without anyone needing to clone the repo,
install Jupyter, or run pytest themselves.

Runs entirely OFFLINE: it uses tbresearch directly (the same package the
tests and CLI use) against the deterministic synthetic price series, so a
hosted instance on Streamlit Community Cloud works with zero secrets and
zero infrastructure.

Host it free:
  1. pip install streamlit   (locally, to test)
  2. streamlit run hosting/streamlit_app.py
  3. push to GitHub, then at share.streamlit.io point it at this file.

This file lives in hosting/ but imports tbresearch from the repo root (../),
so it works whether you run it from the repo root or from inside hosting/.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
sys.path.insert(0, str(PACKAGE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from tbresearch.data import load_close_prices  # noqa: E402
from tbresearch.volatility import daily_volatility  # noqa: E402
from tbresearch.sampling import cusum_filter  # noqa: E402
from tbresearch.pipeline import build_research_dataset  # noqa: E402
from tbresearch.evaluation import compare_naive_vs_purged  # noqa: E402
from tbresearch.cross_validation import purge_fraction  # noqa: E402

st.set_page_config(page_title="Triple-Barrier Signal Research", page_icon="\U0001F4C9", layout="wide")
st.title("Triple-Barrier Labeling + Purged/Embargoed Cross-Validation")
st.caption("Project 40 (quant track) — measuring, not just asserting, how much naive K-Fold leaks on overlapping financial labels")


@st.cache_data(show_spinner=False)
def get_close(seed: int) -> "object":
    return load_close_prices(n_days=1000, seed=seed)


@st.cache_data(show_spinner=False)
def get_events(seed: int, cusum_k: float):
    close = get_close(seed)
    vol = daily_volatility(close, span=100)
    events = cusum_filter(close, cusum_k * vol)
    return close, vol, events


@st.cache_data(show_spinner="Running the pipeline (labeling + training two CV splitters)...")
def get_reports(seed: int, cusum_k: float, vertical_days: int, pt: float, sl: float, n_splits: int, embargo: float):
    close = get_close(seed)
    ds = build_research_dataset(close, cusum_k=cusum_k, num_days_vertical=vertical_days, pt_sl=(pt, sl))
    reports = compare_naive_vs_purged(ds.X, ds.y, ds.sample_weight, ds.t1, n_splits=n_splits, pct_embargo=embargo, random_state=seed)
    frac = purge_fraction(n_splits=n_splits, t1=ds.t1, pct_embargo=embargo)
    return ds, reports, frac


tab_headline, tab_events, tab_about = st.tabs(["Naive vs Purged CV", "Price & Events", "What's real vs simulated"])

with st.sidebar:
    st.header("Pipeline configuration")
    seed = st.selectbox("Random seed (price series)", [42, 1, 7, 99], index=0)
    cusum_k = st.slider("CUSUM threshold (x volatility)", 0.5, 3.0, 1.0, 0.1)
    vertical_days = st.slider("Vertical barrier (trading days)", 3, 30, 15, 1)
    pt = st.slider("Profit-take barrier (x volatility)", 0.5, 5.0, 2.5, 0.1)
    sl = st.slider("Stop-loss barrier (x volatility)", 0.5, 5.0, 2.5, 0.1)
    n_splits = st.slider("CV folds", 3, 10, 5, 1)
    embargo = st.slider("Embargo (fraction of samples)", 0.0, 0.1, 0.02, 0.01)
    st.caption("Defaults are this project's own tuned config, verified to show a clean, robust leakage gap.")

with tab_headline:
    ds, reports, frac = get_reports(seed, cusum_k, vertical_days, pt, sl, n_splits, embargo)

    c1, c2, c3 = st.columns(3)
    c1.metric("Events labeled", len(ds.events))
    c2.metric("Usable after feature warm-up", len(ds.X))
    c3.metric("Purge fraction (this config)", f"{frac:.1%}")

    gap = reports["naive"].mean_test_accuracy - reports["purged"].mean_test_accuracy

    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["naive\n(shuffled KFold)", "purged\n(+ embargo)"]
    accs = [reports["naive"].mean_test_accuracy, reports["purged"].mean_test_accuracy]
    bars = ax.bar(names, accs, color=["crimson", "seagreen"])
    ax.set_ylabel("mean test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"naive overstates by {gap:+.1%}")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.1%}", ha="center")
    st.pyplot(fig)

    if gap > 0:
        st.success(f"Naive shuffled K-Fold overstates test accuracy by **{gap:+.3f}** vs purged + embargoed K-Fold, on this exact configuration.")
    else:
        st.warning(
            f"On this configuration the gap is {gap:+.3f} — try widening the profit-take/stop-loss "
            "barriers or the vertical horizon; tight barriers resolve too fast for much overlap to build up. "
            "See `knowledge/06_the_leakage_problem.md` for why this happens."
        )

    st.subheader("Full comparison")
    st.table(
        {
            "": ["mean train accuracy", "mean test accuracy", "mean test F1", "train-test gap"],
            "naive (shuffled KFold)": [
                f"{reports['naive'].mean_train_accuracy:.3f}",
                f"{reports['naive'].mean_test_accuracy:.3f}",
                f"{reports['naive'].mean_test_f1:.3f}",
                f"{reports['naive'].train_test_gap:.3f}",
            ],
            "purged + embargoed KFold": [
                f"{reports['purged'].mean_train_accuracy:.3f}",
                f"{reports['purged'].mean_test_accuracy:.3f}",
                f"{reports['purged'].mean_test_f1:.3f}",
                f"{reports['purged'].train_test_gap:.3f}",
            ],
        }
    )

with tab_events:
    close, vol, events = get_events(seed, cusum_k)
    st.subheader(f"CUSUM-flagged events on seed {seed} (cusum_k={cusum_k})")
    st.caption(f"{len(events)} events fired out of {len(close)} trading days ({len(events)/len(close):.1%})")

    fig2, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    close.plot(ax=axes[0], label="close", alpha=0.8)
    axes[0].scatter(events, close.reindex(events), color="red", s=14, zorder=5, label="CUSUM event")
    axes[0].set_title("price with CUSUM-flagged events")
    axes[0].legend()
    vol.plot(ax=axes[1], color="darkorange", title="EWMA daily volatility (span=100)")
    plt.tight_layout()
    st.pyplot(fig2)

with tab_about:
    st.markdown(
        """
### What's real here, and what's simulated

| Piece | Real or simulated |
|---|---|
| Price data | Synthetic by default (deterministic, regime-switching geometric Brownian motion); real Yahoo Finance data one flag away (`load_close_prices(real=True)`) |
| Triple-barrier labeling | Real implementation, unit-tested against hand-built price paths with known barrier touches |
| CUSUM event sampling | Real implementation of the symmetric CUSUM filter from Lopez de Prado's book |
| Purged + embargoed CV | Real implementation, unit-tested to prove ZERO train/test window overlap after purging |
| The naive-vs-purged gap | Real, measured fresh on every slider change above — not asserted |

This dashboard runs the exact same `tbresearch` package used by the 43
offline pytest tests in `tests/` and by
`python -m tbresearch run` on the command line — nothing here is a
separate, simplified reimplementation for the demo.

See `../knowledge/00_START_HERE.md` for the full concept walkthrough,
`../knowledge/notebooks/` for step-by-step notebooks, and
`../architecture.md` for the full pipeline diagram and
the tuning journey behind this project's default configuration.
        """
    )
