# Deploy checklist

Before you call this "shipped" and link it on a resume:

## Code and repo hygiene

- [ ] `.gitignore` excludes `__pycache__/`, `.pytest_cache/`,
      `.venv/` (it does).
- [ ] `labs/.gitignore` excludes the generated `synthetic_ohlcv.csv` (it
      does) — commit the scripts, not their throwaway output.
- [ ] `data/`: decide whether to commit the sample
      `synthetic_prices.csv` + `canonical_run_summary.json` (useful so a
      visitor sees real numbers without running anything) or regenerate
      them fresh via CI (this project's workflow does the latter).
- [ ] README at the repo root explains what this project is in the first
      3 sentences — someone skimming your GitHub shouldn't have to open a
      file to know what they're looking at.

## Prove it actually runs

- [ ] `python -m pytest -q` passes (43 tests, no
      network needed).
- [ ] `python -m tbresearch run` prints the naive-vs-purged headline
      comparison cleanly.
- [ ] `python generate_data.py` regenerates `data/` without errors.
- [ ] `python -m tbresearch run --real --symbol SPY` works if you have
      internet handy (optional, but worth confirming at least once
      recently — not just "it worked before").

## CI, actually live

- [ ] `.github/workflows/tbresearch-ci.yml` exists in the pushed repo
      (copied from `hosting/github_actions/`).
- [ ] The Actions tab shows at least one real, green run.
- [ ] Branch protection on `main` requires the `test` check before
      merging.

## The hosted dashboard

- [ ] `hosting/streamlit_app.py` deployed to Streamlit Community Cloud,
      loads with no errors, and the default-config headline number matches
      `knowledge/06_the_leakage_problem.md` (naive ~0.627, purged ~0.551,
      gap ~+0.076).
- [ ] Drag a slider (try widening `pt`/`sl` toward 5.0) and confirm the
      dashboard actually recomputes — not just displaying a static number.

## What to actually say about it

- [ ] You can explain, without looking anything up, WHY naive shuffled
      K-Fold leaks on overlapping financial labels — not just that it does
      (see `knowledge/06` and `10_interview_questions.md`, question 6).
- [ ] You can describe the tuning journey that made the leakage gap
      visible from memory — it's the single best "I actually built and
      debugged this, I didn't just implement a known method" detail in the
      whole project (`architecture.md`).
- [ ] You know why `purge_fraction` can look small (1-6%) while the naive-
      vs-purged accuracy gap is large (5-20+ points) — a subtle point most
      people who've only read the theory get wrong (`labs/05`'s README).
- [ ] You can explain meta-labeling and why this project deliberately
      doesn't implement it, even though it's the natural next step
      (`knowledge/08`).
