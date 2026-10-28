# Forecast — 15-minute solar flare early warning on a live stream

*Will the nowcast confirm a flare in the next 15 minutes?*

```
y(t) = 1   if the nowcast engine raises a flare alarm (peak ≥ C1.0, or ≥ M1.0) in (t, t + 15 min]
```

Scored only while **no flare is in progress** (nowcast state QUIET or DECAY), because during a flare the nowcast already knows.
Labels come from the causal nowcast catalogue in `../nowcast`, which is validated against NOAA's official flare list (F1 0.91).

## Results (held-out Jul 1 – Oct 27 2026, walk-forward monthly retraining)

| Model | Flares | POD | False-alarm ratio | CSI | False alarms / day | ROC AUC | Brier skill vs climatology | Median warning before the nowcast alarm |
|---|---|---|---|---|---|---|---|---|
| **≥ C1.0** (`models/C1`) | 542 | 0.66 | 0.36 | 0.47 | 1.8 | 0.79 | +0.12 | 1 min (5 min before peak) |
| **≥ M1.0** (`models/M1`) | 79 | 0.44 | 0.78 | 0.16 | 1.2 | 0.85 | +0.05 | **13 min** (4 min before the physical start) |
| ≥ C1.0 transfer, no absolute levels (`models/C1_transfer`) | 542 | 0.69 | 0.43 | 0.44 | 2.5 | 0.77 | +0.11 | 1 min |
| *Baseline: rise rule, no ML (≥C1)* | 542 | 0.65 | 0.55 | 0.36 | 3.7 | – | – | 1 min |
| *Original notebook (claimed)* | 25 | 1.00 | 0.11 | 0.89 | – | – | – | "11.3 min" (measured to the peak) |

**How to read this**

* **≥C1**: clearly better than climatology, the activity-only model and the rise rule (half the false alarms at the same POD). Most of
  its warning, though, is the *first minutes of the flare's own rise*, caught before the nowcast is confident. Only 9 % of hits come
  before the physical start. Minute-cadence soft X-rays carry little information about a C-flare 15 minutes ahead.
* **≥M1**: a genuine precursor signal. 57 % of hits warn before the flare physically starts, a median 13 min before the nowcast alarm.
  But M-flares are rare, so 78 % of alerts are false. Use the M1 probability as a graded risk level, not a binary trigger.
  On test, the activity-only logistic regression has a slightly higher ROC AUC (0.87 vs 0.85) but lower Brier skill and PR AUC.
* **Month to month the skill varies**: in the quiet September only 68 C+ flares occurred, and POD fell to 0.24.
* **Why the original 100 % / 11.3 min was not real**: labels used whole-dataset statistics (including the test period); one flare was
  counted several times; lead time was measured to the peak; rows during flares were scored; LightGBM's 0/1 `predict()` was averaged
  with probabilities; windows counted rows across data gaps; and the test held only 25 events. Each of these is fixed here — see the table
  at the top of `forecast.ipynb`.

## Method

* **Features** (40, all causal, minute-based windows on a gap-aware grid): excess over the nowcast background and its recent maxima;
  log-flux growth over 1–30 min; volatility; climb above the recent minimum; XRS-A/XRS-B ratio (temperature proxy) and its trends;
  flaring history (onsets in the last 6/24 h, time since the last onset/end); absolute level. A test scrambles the future and checks
  that no past feature changes.
* **Model**: LightGBM + XGBoost probability average with early stopping. No scaling or PCA, no class reweighting. Isotonic
  calibration is fitted on dev out-of-sample probabilities.
* **Protocol**: for each month Mar–Oct, a fresh model is trained on data up to one day before the month. Mar–Jun (dev) is used to fit
  calibration and choose the alert rule (threshold × m-of-n persistence, maximising event CSI). Jul–Oct (test) is scored once.
* **Event scoring**: a flare is a hit if an alert occurs in `[nowcast alarm − 15 min, nowcast alarm)`. An alert episode is false if
  no qualifying alarm follows within 15 min. Lead time is reported against the nowcast alarm, the physical start, and the peak.

## Streaming engine

`ForecastEngine` = `NowcastEngine` + rolling 25 h feature buffer + one or more models + alert managers, fed one sample at a time.
On a 3-day replay it reproduces the batch probabilities exactly (max difference 0.0), at about 22 ms per minute.

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

**Aditya-L1** (1 s SoLEXS + HEL1OS): use `NowcastEngine(aditya_channels(), 1)` with `forecast_cadence_s=60`. The engine aggregates
to minute bins exactly as `records_to_frame(..., cadence="60s")` does for training. Start with `models/C1_transfer`, which uses only
dimensionless, background-relative features. Then retrain on Aditya-L1's own nowcast catalogue with `walk_forward`: the code is
unchanged, only the source and channel names differ.

## Files

```
forecast/
├── forecast.ipynb           # labels → features → walk-forward → baselines → calibration → alert tuning → test → models → streaming → live → Aditya-L1
├── l1_forecast/
│   ├── features.py          # causal gap-aware features, record aggregation, eligibility
│   ├── labels.py            # targets from the nowcast catalogue
│   ├── model.py             # LightGBM + XGBoost ensemble, isotonic calibration, save/load
│   ├── training.py          # monthly walk-forward with embargo
│   ├── alerts.py            # alert rule (batch + streaming), event scoring, tuning, probability scores
│   └── engine.py            # streaming nowcast + multi-model forecast engine
├── models/{C1,M1,C1_transfer}/   # production models (lgb_model.txt, xgb_model.json, forecaster.json with alert rule + test metrics)
├── tests/test_forecast.py   # 8 tests (targets, causality, alert rule = streaming, scoring, save/load, stream = batch at 1 min and 1 s→1 min)
└── outputs/                 # metrics tables and figures
```

```bash
pip install -r requirements.txt
jupyter notebook forecast.ipynb    # needs ../nowcast (and its data cache, which nowcast.ipynb downloads)
python -m pytest -q tests
```
