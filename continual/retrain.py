"""
Daily retrain with a champion / challenger gate.

For each target (C1, M1), with T = the end of the last complete day in the store:

    ─────────── training data ───────────┤ cal ├─ embargo ─┤──── unseen evaluation window ────┤
                                    cutoff-30d  cutoff      cutoff+1d                           T
                                                  (= T - 7 days)

  * candidate: fitted on all eligible history before ``cutoff``; isotonic calibration and the alert rule
    are fitted on the 30 days before the cutoff, using a model that had not seen them (same recipe as
    forecast.ipynb)
  * evaluation window: the days after both the candidate's and the active model's training data
    (plus a 1-day embargo), so neither has seen them - a fair comparison
  * promote when the candidate's log loss on that window is no worse than the active model's
    (0.5 % tolerance) and its event CSI is not clearly worse. With fewer than 3 unseen days the
    active model is kept and the run is only logged.

The deployed model therefore lags the data by a week. That is the price of always being able to
check it on data it has not seen, and with ~10 months of history a week of extra data barely matters.
"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np
import pandas as pd

from l1_nowcast import goes_channels, run_batch
from l1_forecast import (FlareForecaster, alert_series, build_features, eligible_mask, horizon_covered,
                         make_targets, probability_scores, qualifying_events, records_to_frame,
                         score_events, tune_alert_rule)

from .registry import Registry
from .store import DataStore

TARGETS = {"C1": "C1.0", "M1": "M1.0"}
HORIZON_MIN = 15
HOLDOUT_DAYS = 7
CAL_DAYS = 30
EMBARGO = pd.Timedelta("1D")
MIN_UNSEEN_DAYS = 3
LOGLOSS_TOL = 0.005
CSI_TOL = 0.05

warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")


def build_dataset(xrs: pd.DataFrame) -> dict:
    nc = run_batch(xrs, goes_channels(), 60)
    f = records_to_frame(nc.records, ["xrsb", "xrsa"])
    X = build_features(f, "xrsb", "xrsa", step_min=1.0, absolute=True)
    eligible = eligible_mask(f, "xrsb") & horizon_covered(f.index, f.index.max(), HORIZON_MIN)
    return {"X": X, "eligible": eligible, "events": nc.events, "index": f.index}


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(model: FlareForecaster, rule: dict, ds: dict, y: pd.Series, flares: pd.DataFrame,
             window: tuple[pd.Timestamp, pd.Timestamp], base_rate: float) -> dict:
    idx = ds["index"]
    w = (idx >= window[0]) & (idx <= window[1])
    el = ds["eligible"] & w
    p = pd.Series(model.predict_proba(ds["X"][w]), index=idx[w]).where(el[w])
    ok = el[w].to_numpy() & p.notna().to_numpy()
    yy, pp = y[w].to_numpy()[ok], p.to_numpy()[ok]
    out = {"rows": int(ok.sum()), "positives": int(yy.sum()), "logloss": _logloss(yy, pp)}
    if 0 < yy.sum() < len(yy):
        s = probability_scores(yy, pp, base_rate)
        out.update(brier=s["Brier"], bss=s["BSS_vs_climatology"], roc_auc=s["ROC_AUC"])
    alert = alert_series(p, el[w], rule["threshold"], rule["m"], rule["n"])
    sc = score_events(alert, flares, HORIZON_MIN, period=window).as_dict()
    out.update(flares=sc["flares"], hits=sc["hits"], false_alarms=sc["false_alarms"],
               pod=sc["POD"], far=sc["FAR(ratio)"], csi=sc["CSI"])
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in out.items()}


def train_candidate(ds: dict, y: pd.Series, flares: pd.DataFrame, cutoff: pd.Timestamp,
                    fallback_rule: dict) -> tuple[FlareForecaster, dict, dict]:
    X, el, idx = ds["X"], ds["eligible"], ds["index"]
    cal0 = cutoff - pd.Timedelta(days=CAL_DAYS)
    # 1) a model that has not seen the calibration window -> calibration map + alert rule
    m0 = FlareForecaster(list(X.columns)).fit(X[el & (idx < cal0 - EMBARGO)], y[el & (idx < cal0 - EMBARGO)])
    cal = el & (idx >= cal0) & (idx < cutoff)
    raw = m0.predict_raw(X[cal])
    m0.fit_calibration(raw, y[cal].to_numpy())
    p_cal = pd.Series(m0.calibrate(raw), index=idx[cal])
    fl_cal = flares[(flares["detect_time"] >= cal0) & (flares["detect_time"] < cutoff)]
    rule, rule_src = dict(fallback_rule), "kept (no qualifying flares in the calibration window)"
    if len(fl_cal):
        best = tune_alert_rule(p_cal, cal[cal], fl_cal, HORIZON_MIN).iloc[0]
        if best["CSI"] > 0:
            rule = {"threshold": float(best["threshold"]), "m": int(best["m"]), "n": int(best["n"])}
            rule_src = f"tuned on {len(fl_cal)} flares (CSI {best['CSI']:.2f})"
    # 2) the candidate itself: everything before the cutoff, with that calibration
    tr = el & (idx < cutoff)
    m = FlareForecaster(list(X.columns)).fit(X[tr], y[tr])
    m.calib_x, m.calib_y = m0.calib_x, m0.calib_y
    info = {"train_rows": int(tr.sum()), "pos_rate": float(y[tr].mean()), "alert_rule_source": rule_src,
            "calibration_window": [str(cal0), str(cutoff)]}
    return m, rule, info


def decide(cand: dict, champ: dict, unseen_days: float) -> tuple[bool, str]:
    if unseen_days < MIN_UNSEEN_DAYS:
        return False, f"kept: only {unseen_days:.1f} days unseen by both models (need {MIN_UNSEEN_DAYS})"
    ll_ok = cand["logloss"] <= champ["logloss"] * (1 + LOGLOSS_TOL)
    csi_ok = cand["flares"] == 0 or (cand["csi"] or 0) >= (champ["csi"] or 0) - CSI_TOL
    if ll_ok and csi_ok:
        return True, (f"promoted: log loss {cand['logloss']:.4f} vs {champ['logloss']:.4f}, "
                      f"CSI {cand['csi'] or 0:.2f} vs {champ['csi'] or 0:.2f}")
    why = "log loss worse" if not ll_ok else "CSI worse"
    return False, (f"kept: {why} (log loss {cand['logloss']:.4f} vs {champ['logloss']:.4f}, "
                   f"CSI {cand['csi'] or 0:.2f} vs {champ['csi'] or 0:.2f})")


def daily_cycle(store: DataStore | None = None, registry: Registry | None = None, ingest: bool = True,
                verbose: bool = True) -> list[dict]:
    store, registry = store or DataStore(), registry or Registry()
    say = print if verbose else (lambda *a, **k: None)
    run_at = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("s")
    if ingest:
        say("ingest:", store.ingest_live())
    T = store.data_end()
    if T is None:
        raise RuntimeError("the data store is empty - run `python -m continual seed` first")
    t0 = time.time()
    ds = build_dataset(store.history(end=T))
    say(f"dataset through {T}: {len(ds['index']):,} minutes, built in {time.time() - t0:.0f}s")
    cutoff = (T - pd.Timedelta(days=HOLDOUT_DAYS)).ceil("1D")
    version = f"{T:%Y%m%d}"
    results = []
    for target, min_class in TARGETS.items():
        t1 = time.time()
        flares = qualifying_events(ds["events"], "xrsb", min_class)
        y = make_targets(ds["index"], flares["detect_time"], HORIZON_MIN)
        champ_v = registry.active_version(target)
        champ = FlareForecaster.load(registry.model_dir(target, champ_v))
        champ_rule = champ.saved["alert_rule"]
        cand, rule, info = train_candidate(ds, y, flares, cutoff, champ_rule)

        start = max(cutoff, registry.train_end(target, champ_v)) + EMBARGO
        window = (start, T)
        unseen = max(0.0, (T - start).total_seconds() / 86400)
        base_rate = info["pos_rate"]
        m_cand = evaluate(cand, rule, ds, y, flares, window, base_rate) if unseen > 0 else None
        m_champ = evaluate(champ, champ_rule, ds, y, flares, window, base_rate) if unseen > 0 else None
        promote, reason = decide(m_cand, m_champ, unseen) if unseen > 0 else (False, "kept: no unseen days yet")

        d = registry.new_version_dir(target, version)
        cand.save(d, extra={
            "target": f"nowcast alarm for a flare >= {min_class} within {HORIZON_MIN} min",
            "horizon_min": HORIZON_MIN, "min_class": min_class, "primary": "xrsb", "secondary": "xrsa",
            "train_end": str(cutoff - pd.Timedelta("1min")),
            "trained_on": f"{ds['index'].min()} .. {cutoff - pd.Timedelta('1min')} (GOES-19 XRS, 1 min)",
            "alert_rule": rule, "evaluation": {"window": [str(window[0]), str(window[1])], "metrics": m_cand},
            # the dashboard's skill panel reads these two keys
            "test_metrics_jul_oct": champ.saved.get("test_metrics_jul_oct"),
            "test_prob_scores": champ.saved.get("test_prob_scores"),
            **info,
        })
        if promote:
            registry.promote(target, version)
        removed = registry.prune(target)
        entry = {"run_at": str(run_at), "target": target, "data_end": str(T), "candidate": version,
                 "candidate_train_end": str(cutoff - pd.Timedelta("1min")), "champion": champ_v,
                 "window": [str(window[0]), str(window[1])], "unseen_days": round(unseen, 2),
                 "candidate_metrics": m_cand, "champion_metrics": m_champ, "alert_rule": rule,
                 "promoted": promote, "reason": reason, "active": registry.active_version(target),
                 "seconds": round(time.time() - t1, 1), "pruned": removed}
        registry.log(entry)
        results.append(entry)
        say(f"{target}: {reason}  [{entry['seconds']}s]")
    return results
