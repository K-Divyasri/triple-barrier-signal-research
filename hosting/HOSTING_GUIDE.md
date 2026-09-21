# Hosting this project (for free) and showing it off

Unlike some other projects in this repo, `tbresearch` has no infrastructure
to keep alive — no Kafka broker, no database, no message queue. It's a
pure research/ML package: pandas, numpy, scikit-learn. That makes the
hosting story simpler than usual, not more limited. What a reviewer
actually expects to see:

- Clean code on GitHub with a great README.
- A CI check that *actually runs*, on real commits, that they can click
  into and watch pass.
- Something interactive they can click on without installing Python.

This project's hosting story has two real, concrete parts.

## Step 1 — CI, actually running on GitHub (do this no matter what)

```powershell
cd 40-triple-barrier-signal-research
git init
git add .
git commit -m "Triple-barrier labeling + purged/embargoed cross-validation research pipeline"
# create an empty repo on github.com, e.g. 'triple-barrier-signal-research', then:
git remote add origin https://github.com/YOURNAME/triple-barrier-signal-research.git
git branch -M main
git push -u origin main
```

Then:

```powershell
mkdir -p .github/workflows
cp hosting/github_actions/tbresearch-ci.yml .github/workflows/tbresearch-ci.yml
git add .github
git commit -m "Add CI: run the 43-test suite and regenerate the canonical run summary"
git push
```

Open the repo's **Actions** tab. You'll see it install dependencies, run
all 43 offline tests, run `python -m tbresearch run` end to end (prints the
naive-vs-purged headline comparison right in the log), and upload the
freshly regenerated `canonical_run_summary.json` as a downloadable
artifact. That artifact is a real, verifiable, permanently-hosted proof
that the leakage gap this project claims is real, current, and
reproducible on THIS commit — not a screenshot someone has to trust.

**Turn on branch protection** (Settings -> Branches -> add a rule for
`main` -> require status checks -> select `test`) so a broken change can't
merge silently.

## Step 2 — host the interactive dashboard on Streamlit Community Cloud

`hosting/streamlit_app.py` runs entirely offline (no market data feed, no
secrets, no external service) — it uses `tbresearch` directly against the
deterministic synthetic price series already built into the package. That
makes it free to host with zero configuration:

1. Push the repo to GitHub (Step 1).
2. Go to https://share.streamlit.io , sign in with GitHub.
3. New app -> pick your repo -> set the main file to
   `hosting/streamlit_app.py` -> Deploy.

You'll get a public URL where a visitor can drag sliders for the CUSUM
threshold, barrier widths, vertical horizon, CV folds, and embargo, and
watch the naive-vs-purged accuracy gap recompute live — including seeing
the gap shrink or even flip sign if they set the barriers unrealistically
tight, which is itself an honest, interactive demonstration of `06`'s
tuning lesson (the leak isn't always big enough to see; the point is that
it's always there and this dashboard lets you find where it disappears).

## Optional — run against real market data instead of synthetic

Everything in this project supports real Yahoo Finance data one flag away
(`load_close_prices(real=True)` / `labs/07_real_market_data`). If you want
to demo this live rather than just mention it, the simplest extension to
`streamlit_app.py` is a text input for a ticker symbol and a checkbox for
"use real data" that swaps `get_close`'s implementation — left as an
exercise, since it adds a live network dependency the default dashboard
deliberately avoids for reliability.

## What to actually do (the 15-minute version)

If you only do two things: **Step 1** (CI, actually running) and **Step 2**
(the Streamlit dashboard). Together those are a complete, credible, and —
unlike a screenshot — a *verifiable* portfolio entry.

See `../README.md` for running everything locally
first, and `deploy_checklist.md` for a one-page checklist before you
consider this "shipped."
