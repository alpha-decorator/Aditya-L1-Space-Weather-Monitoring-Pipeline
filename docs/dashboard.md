# FlareWatch — live solar flare dashboard

A web dashboard for the nowcast and forecast engines, running on the public GOES-19 XRS real-time feed from NOAA SWPC.

* **Alarm board**: an illustrated Sun whose glow follows the alarm level, with one big status (All clear / Flare watch /
  Major watch / the class of a flare in progress) and a plain-language explanation
* Stat tiles: X-ray level, nowcast state, UTC clock, flares in the last 24 h
* X-ray brightness chart (6 h / 24 h / 48 h), flare-chance chart with alarm levels, flares per 6 h
* Ring gauges: C and M flare chance (a full ring = alarm level), flares caught and false alarms of the model
* Alarm log: every forecast alarm and C+ flare in the last 48 h, and what happened after each
* The model: active versions, alarm rules, held-out skill (caught, false alarms, false alarms per day, typical warning)
  and the daily retraining log from `continual/`
* Optional alarm sound + desktop notification; phone layout with a bottom navigation bar
* **Time-of-day themes** from the viewer's own clock: dawn (05–08), day (08–17), dusk (17–20, twilight palette),
  night (20–05). Re-checked every minute; the theme button pins Dawn / Day / Night or returns to Auto
  (preview with `/?phase=day`)
* Preview the alarm states on a quiet day: `/?demo=watch`, `/?demo=major`, `/?demo=flare`

Visual style: dark "deep space" palette (plum, magenta, orange, lavender, cream) with original flat illustrations
(sun with a zig-zag corona, orbiting satellite, sparkles). Chart colours are checked for colour-blind separation and
contrast on the dark surface.

## Run locally

```bash
pip install -r requirements-dev.txt
python serve.py                  # http://127.0.0.1:8000
python serve.py --retrain-daily  # also run the daily ingest + retrain (see docs/continual.md)
```

## How it works

```
NOAA SWPC xrays-3-day.json ─► run_batch (nowcast) ─► build_features ─► C1 / M1 models ─► JSON snapshot ─► browser
```

`flaremon/snapshot.py` is stateless: every call fetches the last 3 days, runs the nowcast and both forecast models in
batch (about 3 s), and returns one JSON document. Batch mode gives the same numbers as the streaming engine, so
nothing has to stay running between requests. `api/index.py` (FastAPI) caches the snapshot for 30 s and serves
`/api/snapshot`. The page in `public/` is plain HTML/CSS/JS with no build step, and refreshes every 30 s.

## Deploy on Vercel

The repository root is the Vercel project: import the GitHub repo, keep the framework preset "Other" and the
root directory `./`, and deploy. No build command is needed.

* `vercel.json` serves `public/` as the static site and routes `/api/*` to the Python function `api/index.py`.
* `requirements.txt` holds only what the function needs: numpy, pandas, fastapi. LightGBM and XGBoost need the
  OpenMP runtime (`libgomp.so.1`), which Vercel does not provide, so the deployed function evaluates the saved
  `lgb_model.txt` / `xgb_model.json` with numpy (`l1_forecast/treeinfer.py`; identical to the native libraries, tested).
  Training packages are in `requirements-dev.txt`.
* `.vercelignore` keeps the GOES cache, notebooks, outputs, tests and the training store out of the bundle.
* The daily retrain runs on GitHub Actions and commits the new registry. That push redeploys the site, which then
  serves the new active models.

```
api/index.py         # FastAPI: /api/snapshot (30 s cache, gzip), /api/health
flaremon/snapshot.py # fetch → nowcast → forecast → JSON
public/              # index.html, styles.css, app.js
serve.py             # local server: API + static files (+ optional daily retrain)
```
