# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Solar flare **nowcast** (is a flare happening now, what class?) and **15-minute forecast** (P(flare ≥ C1 / ≥ M1)) from soft X-ray light curves, built for Aditya-L1 SoLEXS/HEL1OS and validated on GOES-19 XRS against NOAA's flare list. Includes a daily continual-learning loop and the **FlareWatch** live dashboard. The repository root is a single deployable Vercel project.

## Commands

Python venv lives one level up: `"../.venv/Scripts/python.exe"` (Windows). Run everything from the repo root.

```bash
pip install -r requirements-dev.txt          # requirements.txt = runtime only (what Vercel installs)
python -m pytest -q                           # all 32 tests (pytest.ini -> tests/)
python -m pytest -q tests/test_nowcast.py     # one suite
python -m pytest -q tests/test_forecast.py -k causal   # one test by name
python serve.py [--port 8000] [--retrain-daily]        # API + page on http://127.0.0.1:8000
python -m continual seed | ingest | daily | status     # continual-learning CLI
```

Dashboard preview params: `/?demo=watch|major|flare`, `/?phase=day|dawn|night`.
Notebooks (`notebooks/*.ipynb`) run from `notebooks/` and set `ROOT = Path.cwd().parent`; the GOES archive cache goes to `data/` (git-ignored, ~100 MB, downloaded by `nowcast.ipynb`).

## Architecture

```
GOES/SoLEXS samples ─► l1_nowcast (streaming detector) ─► flare events + per-sample state
                              └─► l1_forecast (40 causal features → LightGBM+XGBoost → isotonic → alert rule)
flaremon/snapshot.py: SWPC 3-day feed → run_batch → build_features → active models → one JSON  ─► api/index.py ─► public/
continual/: DataStore (store/) → retrain on all history → gate → Registry (registry/) ─► snapshot loads active version
```

- **Streaming = batch.** `NowcastEngine` / `ForecastEngine` take one sample at a time; `run_batch` and `build_features` give identical results (tested). The dashboard is stateless and uses batch mode on each request (~3 s), cached 30 s in `api/index.py`. Keep this equivalence when changing either path.
- **Causality is a hard rule.** Features use only past data and time-based, gap-aware windows (not row counts). Forecast rows are only scored when the nowcast is *not* already in FLARE. Labels come from the causal nowcast catalogue, not whole-dataset statistics. These fix the leakage in the original prototype (`notebooks/original/`, whose 100 % TPR claim is invalid) — don't reintroduce them.
- **Nowcast** (`l1_nowcast/detector.py`): per-channel state machine QUIET→RISE→FLARE→DECAY; robust log background over 2 h of quiet samples; onset = k·σ with confirm=2 plus rise guard (`min_ratio`); NOAA half-way end rule; re-rise splits flares; two channels fused within ±5 min (BOTH / PRIMARY_ONLY / SECONDARY_ONLY). Tuned settings come from `goes_channels()` / `aditya_channels()`; tuned config in `outputs/nowcast/nowcast_config.json`.
- **Forecast** (`l1_forecast/`): `FlareForecaster` averages LightGBM + XGBoost probabilities then applies isotonic calibration stored as a lookup (sklearn only needed for training). Saved as `lgb_model.txt`, `xgb_model.json`, `forecaster.json` (includes `alert_rule` threshold/m/n and test metrics). `C1_transfer` uses only instrument-agnostic features for Aditya-L1.
- **Continual** (`continual/retrain.py:daily_cycle`): candidate trained on all history before `cutoff = ceil(T−7d)`, calibrated on the 30 days before cutoff; both champion and candidate scored on days after max(cutoff, champion train_end) + 1-day embargo; promote only if log loss ≤ champion×1.005 and CSI ≥ champion−0.05 with ≥ 3 unseen days. `Registry` version `base` = `models/<target>`; `active.json` is re-read per request so promotions need no restart. Store history: NCEI archive wins over SWPC where both exist; live cache kept 10 days.
- **Scheduling**: `serve.py --retrain-daily` (thread → subprocess at 00:30 UTC) locally; `.github/workflows/daily-retrain.yml` in CI, which commits `continual/store` and `continual/registry` (that push redeploys Vercel).
- **Frontend** (`public/`): plain HTML/CSS/JS, no build step, hand-drawn SVG charts. All colours are CSS variables on `:root` with overrides per `[data-phase="dawn"|"day"]`; the head script sets `data-phase` from the viewer's local clock before first paint, and `app.js` (`autoPhase`/`applyPhase`) re-checks every minute. Status colours (good/warning/serious/critical) are reserved for alarm state; chart series colours were validated per theme — change them per theme, not globally.

## Deployment (Vercel)

`vercel.json` serves `public/` as static output and rewrites `/api/*` to the Python function `api/index.py`. The function bundle must stay small: runtime deps only in `requirements.txt` (numpy, pandas, fastapi — **no lightgbm/xgboost**: they need `libgomp.so.1`, which Vercel lacks; `FlareForecaster.load(engine="auto")` falls back to the pure-numpy tree evaluator `l1_forecast/treeinfer.py`, kept equal to native by `test_numpy_engine_matches_native`; don't add sklearn/matplotlib/pyarrow either), and `.vercelignore` excludes `data/`, `notebooks/`, `outputs/`, `tests/`, `continual/store/`. Packages are imported from the repo root — don't add cross-folder `sys.path` hacks. The Vercel filesystem is read-only, so the snapshot path must not write files.

## Data notes

No public real-time Aditya-L1 API exists; Aditya-L1 data come as files from PRADAN (user login) via `DirectoryWatchSource`. The Aditya-L1 path is implemented and unit-tested but not yet scored on real data. Live data = NOAA SWPC GOES-19 JSON (1-minute cadence).

Docs per component: `docs/{nowcast,forecast,continual,dashboard}.md`; full write-up: `docs/FlareWatch_Project_Report.pdf`.
