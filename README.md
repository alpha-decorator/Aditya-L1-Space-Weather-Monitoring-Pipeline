# 🛰️ Aditya-L1 Space Weather Monitoring Pipeline

Streaming solar flare **nowcasting** (is a flare happening right now, and how big is it?) and **15-minute forecasting**
(will one start soon?) from soft and hard X-ray light curves — built for Aditya-L1 SoLEXS / HEL1OS and validated on
GOES-19 XRS against NOAA's official flare list.

Both engines process **one sample at a time**, so the same code runs on a live feed and in back-tests.

```
X-ray stream ─► nowcast engine ─► flare catalogue + live alerts (onset / escalation / end / fused)
                      │
                      └─► forecast engine ─► P(flare ≥ C1 in 15 min), P(flare ≥ M1 in 15 min) + alerts
```

| Component | Folder | Question it answers |
|---|---|---|
| Nowcast | [`l1_nowcast/`](docs/nowcast.md) | Is a flare in progress? When did it start? What class? |
| Forecast | [`l1_forecast/`](docs/forecast.md) | Will the nowcast confirm a flare in the next 15 minutes? |
| Continual learning | [`continual/`](docs/continual.md) | Retrain daily; promote a new model only if it is better on unseen days |
| Dashboard | [`api/`, `public/`](docs/dashboard.md) | FlareWatch: live alarms and charts on GOES-19 |
| Original notebook | `notebooks/original/` | First forecasting prototype (kept for reference; see below) |

---

## 📊 Results (held-out data, never used for tuning)

### Nowcast — tuned Jan–Apr 2026, scored May 1 – Oct 27 2026 (936 NOAA flares ≥ C1)

| Detector | Precision | Recall ≥C1 | F1 | M-class POD | X-class POD |
|---|---|---|---|---|---|
| Global `mean + 5σ` threshold (original notebook's labelling rule) | 0.96 | 0.03 | 0.05 | 0.23 | 1.00 |
| Team Phoenix-style rolling median detector | 0.75 | 0.73 | 0.74 | 0.86 | 1.00 |
| **This nowcast engine** | **0.91** | **0.91** | **0.91** | **0.98** (97/99) | **1.00** (3/3) |

* Flare class letter agrees with NOAA for 99.9 % of matched flares.
* Alerts come a median 6 min after NOAA's (retrospective) start time and 3 min before the peak.
* ~25,000 samples/s on a laptop.

### Forecast — walk-forward monthly retraining, scored Jul 1 – Oct 27 2026

| Model | Flares | POD | False-alarm ratio | CSI | False alarms / day | ROC AUC | Median warning before nowcast alarm |
|---|---|---|---|---|---|---|---|
| **≥ C1.0** | 542 | 0.66 | 0.36 | 0.47 | 1.8 | 0.79 | 1 min |
| **≥ M1.0** | 79 | 0.44 | 0.78 | 0.16 | 1.2 | 0.85 | **13 min** |
| ≥ C1.0 transfer (for Aditya-L1) | 542 | 0.69 | 0.43 | 0.44 | 2.5 | 0.77 | 1 min |
| *Baseline: rise rule, no ML (≥C1)* | 542 | 0.65 | 0.55 | 0.36 | 3.7 | – | 1 min |

* ≥C1 halves the false alarms of the rule-based baseline at the same POD, but most of its warning is the first minutes of
  the flare's own rise.
* ≥M1 shows a genuine precursor signal (57 % of hits warn before the physical start), but M-flares are rare, so its
  probability is best used as a graded risk level rather than a binary trigger.

### About the original notebook's numbers

`model.ipynb` reported **100 % TPR, 10.7 % FAR, CSI 0.89 and an 11.3 min warning** on 25 test flares. Those numbers do not
hold up: labels were built from whole-dataset statistics (including the test period), one flare was counted several times,
lead time was measured to the peak rather than the start, rows during flares were scored, LightGBM's 0/1 `predict()` was
averaged with probabilities, rolling windows counted rows across data gaps, and the test held only 25 events. Each issue is
fixed in `l1_forecast` — see the table at the top of `notebooks/forecast.ipynb`. The results above are the honest replacement.

---

## 🚀 Quick start

```bash
pip install -r requirements-dev.txt

python serve.py                              # live dashboard on http://127.0.0.1:8000 (add --retrain-daily)
jupyter notebook notebooks/nowcast.ipynb     # downloads ~100 MB of GOES data from NOAA NCEI into data/ on first run
jupyter notebook notebooks/forecast.ipynb    # needs that data cache

python -m pytest -q                          # 32 tests (nowcast 19, forecast 8, continual 5)
```

Live GOES-19 nowcast + forecast:

```python
from l1_forecast import ForecastEngine, FlareForecaster, AlertManager
from l1_nowcast import NowcastEngine, GOESLiveSource, goes_channels

models = {}
for name in ("C1", "M1"):
    m = FlareForecaster.load(f"models/{name}")
    r = m.saved["alert_rule"]
    models[name] = (m, AlertManager(r["threshold"], r["m"], r["n"]))

eng = ForecastEngine(NowcastEngine(goes_channels(), 60), models, primary="xrsb", secondary="xrsa")
for sample in GOESLiveSource(backfill="3-day"):
    rec, messages = eng.update(sample)
    print(rec["time"], rec["nowcast_state"], f"P(C1)={rec['p_C1']:.1%}", f"P(M1)={rec['p_M1']:.1%}")
```

For Aditya-L1 (1 s SoLEXS + HEL1OS), see the Aditya-L1 sections of [`docs/nowcast.md`](docs/nowcast.md) and
[`docs/forecast.md`](docs/forecast.md).

---

## 🖥️ Live dashboard and continual learning

* [`api/` + `public/`](docs/dashboard.md) — **FlareWatch**, a web dashboard on the live GOES-19 feed: alarm board, charts,
  alarm log with outcomes, model skill and versions. `python serve.py` → http://127.0.0.1:8000.
  The repository root deploys to Vercel as is (`vercel.json`, see [docs/dashboard.md](docs/dashboard.md#deploy-on-vercel))
* [`continual/`](docs/continual.md) — the forecaster keeps learning while deployed: each day's data is stored, both
  models are retrained on the full history, and a new version goes live only if it beats the current one on days
  neither has seen. Runs daily via `serve.py --retrain-daily` or `.github/workflows/daily-retrain.yml`.

---

## 📁 Repository layout

One repository, one root: the root is both the Python project and the Vercel site.

```
├── api/index.py             # Vercel serverless function (FastAPI): /api/snapshot, /api/health
├── public/                  # the FlareWatch page: index.html, styles.css, app.js (no build step)
├── flaremon/                # snapshot builder: SWPC feed → nowcast → forecast → JSON
├── l1_nowcast/              # background, detector + fusion, data sources, validation
├── l1_forecast/             # features, labels, model, walk-forward training, alerts, streaming engine
├── continual/               # daily store + retrain + champion/challenger registry
│   ├── store/               #   stored GOES-19 data (training only, not deployed)
│   └── registry/            #   model versions, active.json, decision log
├── models/{C1,M1,C1_transfer}/   # validated base models
├── notebooks/               # nowcast.ipynb, forecast.ipynb, original/ (first prototype + its plots)
├── outputs/{nowcast,forecast}/   # catalogues, validation tables, figures
├── tests/                   # 32 tests
├── docs/                    # component docs + FlareWatch_Project_Report.pdf
├── serve.py                 # local server (API + page, optional daily retrain)
├── requirements.txt         # runtime only (what Vercel installs)
├── requirements-dev.txt     # notebooks, training, tests
├── vercel.json, .vercelignore
└── .github/workflows/daily-retrain.yml
```

---

## ⚠️ Limitations

* The science validation uses GOES XRS because it has an independent flare catalogue. The Aditya-L1 path (SoLEXS/HEL1OS
  readers, channel settings, transfer model) is implemented and unit-tested but **not yet scored on real Aditya-L1 data** —
  the raw SoLEXS/HEL1OS CSVs are not in this repository. Next step once they are available: tune the nowcast on SoLEXS with
  `tune_channel`, then retrain the forecast on Aditya-L1's own nowcast catalogue with `walk_forward`.
* ISRO has no public real-time API; Aditya-L1 data arrive as files via PRADAN, consumed with `DirectoryWatchSource`.
* Forecast skill varies month to month (e.g. ≥C1 POD fell to 0.24 in the quiet September).
