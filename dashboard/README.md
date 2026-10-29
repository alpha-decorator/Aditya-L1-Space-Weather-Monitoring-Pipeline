# FlareWatch — live solar flare dashboard

A web dashboard for the nowcast and forecast engines, running on the public GOES-19 XRS real-time feed from NOAA SWPC.

* **Alarm board**: an illustrated Sun whose glow follows the alarm level, with one big status (All clear / Flare watch /
  Major watch / the class of a flare in progress) and a plain-language explanation
* Stat tiles: X-ray level, nowcast state, UTC clock, flares in the last 24 h
* X-ray brightness chart (6 h / 24 h / 48 h), flare-chance chart with alarm levels, flares per 6 h
* Ring gauges: C and M flare chance (a full ring = alarm level), flares caught and false alarms of the model
* Alarm log: every forecast alarm and C+ flare in the last 48 h, and what happened after each
* The model: active versions, alarm rules, held-out skill (caught, false alarms, false alarms per day, typical warning)
  and the daily retraining log from `../continual`
* Optional alarm sound + desktop notification; phone layout with a bottom navigation bar
* Preview the alarm states on a quiet day: `/?demo=watch`, `/?demo=major`, `/?demo=flare`

Visual style: dark "deep space" palette (plum, magenta, orange, lavender, cream) with original flat illustrations
(sun with a zig-zag corona, orbiting satellite, sparkles). Chart colours are checked for colour-blind separation and
contrast on the dark surface.

## Run locally

```bash
pip install -r requirements.txt
python serve.py                  # http://127.0.0.1:8000
python serve.py --retrain-daily  # also run the daily ingest + retrain (see ../continual)
```

## How it works

```
NOAA SWPC xrays-3-day.json ─► run_batch (nowcast) ─► build_features ─► C1 / M1 models ─► JSON snapshot ─► browser
```

`flaremon/snapshot.py` is stateless: every call fetches the last 3 days, runs the nowcast and both forecast models in
batch (about 3 s), and returns one JSON document. Batch mode gives the same numbers as the streaming engine, so
nothing has to stay running between requests. `api/index.py` (FastAPI) caches the snapshot for 60 s and serves
`/api/snapshot`. The page in `public/` is plain HTML/CSS/JS with no build step, and refreshes every minute.

The same layout is meant for serverless hosting: `api/index.py` exposes `app`, and `public/` is static.

```
dashboard/
├── api/index.py         # FastAPI: /api/snapshot (60 s cache, gzip), /api/health
├── flaremon/snapshot.py # fetch → nowcast → forecast → JSON
├── public/              # index.html, styles.css, app.js
├── serve.py             # local server: API + static files
└── requirements.txt
```
