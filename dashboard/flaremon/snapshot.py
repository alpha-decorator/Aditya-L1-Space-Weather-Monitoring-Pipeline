"""
One stateless dashboard snapshot: fetch the latest GOES-19 XRS data from
NOAA SWPC, run the nowcast and both forecast models in batch, and return a
JSON-ready dict.

Batch over the last 3 days gives the same numbers as the streaming engine
(verified in ../forecast/tests), so nothing has to stay alive between
requests - the same function serves the local server and a serverless
deployment.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for sub in ("", "nowcast", "forecast"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from l1_nowcast import GOESLiveSource, goes_channels, goes_class, run_batch  # noqa: E402
from l1_forecast import (FlareForecaster, alert_episodes, alert_series,  # noqa: E402
                         build_features, eligible_mask, records_to_frame)
from continual import Registry  # noqa: E402

PRIMARY, SECONDARY = "xrsb", "xrsa"
MODEL_NAMES = ("C1", "M1")
SERIES_HOURS = 48           # the features need 25 h of history, so 3 days of input leave ~47 h of exact output
STALE_AFTER_MIN = 10

_registry = Registry()
_models: dict[Path, FlareForecaster] = {}


def _model(name: str) -> tuple[FlareForecaster, str]:
    """The model the daily retrain currently has active for this target (re-read on every call,
    so a promotion takes effect without restarting the server)."""
    version = _registry.active_version(name)
    d = _registry.model_dir(name, version)
    if d not in _models:
        _models[d] = FlareForecaster.load(d)
    return _models[d], version


def _learning(limit_runs: int = 14) -> list[dict]:
    out = []
    for e in _registry.history()[-2 * limit_runs:][::-1]:
        c, ch = e.get("candidate_metrics") or {}, e.get("champion_metrics") or {}
        out.append({"run_at": e["run_at"], "target": e["target"], "promoted": e["promoted"],
                    "reason": e["reason"], "active": e["active"], "data_end": e["data_end"],
                    "unseen_days": e["unseen_days"],
                    "logloss_new": _num(c.get("logloss"), 4), "logloss_old": _num(ch.get("logloss"), 4)})
    return out


def _ms(t) -> int | None:
    return None if t is None or pd.isna(t) else int(pd.Timestamp(t).value // 1_000_000)


def _num(x, digits: int | None = None):
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return round(x, digits) if digits is not None else x


def _arr(s: pd.Series, digits: int | None = None) -> list:
    v = s.to_numpy(dtype=float)
    if digits is not None:
        v = np.round(v, digits)
    return [None if not math.isfinite(x) else float(x) for x in v]


def _flares(res, since: pd.Timestamp) -> list[dict]:
    ev = res.primary_events(PRIMARY)
    if ev.empty:
        return []
    ev = ev[ev["end"] >= since]
    labels = {}
    if not res.fused.empty:
        labels = dict(zip(res.fused["primary_event"], res.fused["label"]))
    out = []
    for r in ev.itertuples():
        out.append({
            "id": r.event_id,
            "start": _ms(r.start), "detect": _ms(r.detect_time), "peak_time": _ms(r.peak_time),
            "end": None if r.end_reason == "stream_end" else _ms(r.end),
            "ongoing": r.end_reason == "stream_end",
            "peak": _num(r.peak), "class": r.flare_class,
            "confirmed": labels.get(r.event_id) == "BOTH",
            "detect_delay_min": _num(r.detect_delay_min, 1),
        })
    return out[::-1]


def build_snapshot() -> dict:
    t0 = time.perf_counter()
    raw = GOESLiveSource().fetch("3-day")
    if raw.empty:
        raise RuntimeError("NOAA SWPC returned no GOES XRS data")
    df = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="1min"))
    res = run_batch(df, goes_channels(), 60)
    f = records_to_frame(res.records, [PRIMARY, SECONDARY])
    X = build_features(f, PRIMARY, SECONDARY, 1.0, absolute=True)
    eligible = eligible_mask(f, PRIMARY)

    end = f.index[-1]
    since = end - pd.Timedelta(hours=SERIES_HOURS)
    win = f.index > since
    w = f.loc[win]

    forecasts, series_p = {}, {}
    for name in MODEL_NAMES:
        m, version = _model(name)
        rule = m.saved["alert_rule"]
        p = pd.Series(m.predict_proba(X), index=X.index).where(eligible)
        alert = alert_series(p, eligible, rule["threshold"], rule["m"], rule["n"])
        eps = alert_episodes(alert[win])
        tm, ps = m.saved["test_metrics_jul_oct"], m.saved["test_prob_scores"]
        forecasts[name] = {
            "target": m.saved["target"], "min_class": m.saved["min_class"],
            "horizon_min": m.saved["horizon_min"],
            "prob": _num(p.iloc[-1], 4), "alert": bool(alert.iloc[-1]),
            "threshold": rule["threshold"], "rule_m": rule["m"], "rule_n": rule["n"],
            "climatology": _num(ps["pos_rate"], 4),
            "model": {"version": version, "train_end": _ms(_registry.train_end(name, version))},
            "alerts_48h": [{"start": _ms(a.start), "end": _ms(a.end)} for a in eps.itertuples()],
            "skill": {
                "period": "Jul 1 – Oct 27 2026 (held out)",
                "flares": tm["flares"], "pod": _num(tm["POD"], 3), "far": _num(tm["FAR(ratio)"], 3),
                "csi": _num(tm["CSI"], 3), "false_alarms_per_day": _num(tm["false_alarms_per_day"], 2),
                "lead_median_min": _num(tm["lead_vs_nowcast_alarm_median_min"], 1),
                "precursor_fraction": _num(tm["precursor_fraction"], 3),
                "roc_auc": _num(ps["ROC_AUC"], 3),
            },
        }
        series_p[name] = _arr(p[win], 4)

    last = f.iloc[-1]
    flux = last[f"{PRIMARY}_value"]
    state = res.records[f"{PRIMARY}_state"].iloc[-1]
    flares = _flares(res, since)
    data_age_min = (pd.Timestamp.now(tz="UTC").tz_localize(None) - end).total_seconds() / 60.0

    return {
        "source": {"name": "GOES-19 XRS", "provider": "NOAA SWPC", "cadence_s": 60,
                   "url": "https://services.swpc.noaa.gov/json/goes/primary/"},
        "generated_at": _ms(pd.Timestamp.now(tz="UTC").tz_localize(None)),
        "data_end": _ms(end),
        "data_age_min": _num(data_age_min, 1),
        "stale": data_age_min > STALE_AFTER_MIN,
        "now": {
            "state": ["QUIET", "FLARE", "DECAY"][int(state)],
            "flux": _num(flux), "class": goes_class(flux) if math.isfinite(flux) else None,
            "background": _num(last[f"{PRIMARY}_background"]),
            "background_class": goes_class(last[f"{PRIMARY}_background"]),
            "xrsa": _num(last[f"{SECONDARY}_value"]),
            "ongoing": next((fl for fl in flares if fl["ongoing"]), None),
        },
        "forecasts": forecasts,
        "flares": flares,
        "series": {
            "t": [_ms(t) for t in w.index],
            "xrsb": _arr(w[f"{PRIMARY}_value"]),
            "xrsa": _arr(w[f"{SECONDARY}_value"]),
            "background": _arr(w[f"{PRIMARY}_background"]),
            "state": w[f"{PRIMARY}_state"].astype(int).tolist(),
            **{f"p_{k}": v for k, v in series_p.items()},
        },
        "learning": _learning(),
        "compute_s": round(time.perf_counter() - t0, 2),
    }
