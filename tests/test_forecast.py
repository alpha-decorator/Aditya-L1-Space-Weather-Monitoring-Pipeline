"""Unit tests for l1_forecast. Run from the repository root:  python -m pytest -q"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from l1_forecast import (  # noqa: E402
    AlertManager, FlareForecaster, ForecastEngine, alert_series, build_features, eligible_mask,
    make_targets, records_to_frame, score_events,
)
from l1_nowcast import FrameSource, NowcastEngine, goes_channels, run_batch  # noqa: E402


def synthetic_goes(days: int = 3, seed: int = 0) -> pd.DataFrame:
    """Quiet Sun with a flare every ~3 h of random size."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2026-03-01", periods=n, freq="1min")
    b = 1e-6 * 10 ** (0.02 * rng.standard_normal(n))
    m = np.arange(n, dtype=float)
    for t0 in range(120, n - 60, 180):
        t0 += int(rng.integers(-40, 40))
        peak = 10 ** rng.uniform(-6, -4.5)
        d = m - t0
        b += np.where(d < 0, 0, np.where(d < 8, peak * d / 8, peak * np.exp(-(d - 8) / 15)))
    df = pd.DataFrame({"xrsb": b, "xrsa": b / 15}, index=idx)
    df.iloc[2000:2030] = np.nan  # a data gap
    return df


@pytest.fixture(scope="module")
def nowcast_result():
    return run_batch(synthetic_goes(), goes_channels(), 60)


def test_make_targets_window():
    idx = pd.date_range("2026-01-01 00:00", periods=40, freq="1min")
    y = make_targets(idx, [pd.Timestamp("2026-01-01 00:20")], 15)
    assert y["2026-01-01 00:04"] == 0          # 16 min before
    assert y["2026-01-01 00:05"] == 1          # exactly 15 min before
    assert y["2026-01-01 00:19"] == 1
    assert y["2026-01-01 00:20"] == 0          # the event itself is not in (t, t+H]


def test_features_are_causal(nowcast_result):
    f = records_to_frame(nowcast_result.records, ["xrsb", "xrsa"])
    X_full = build_features(f, "xrsb", "xrsa")
    cut = 2500
    f2 = f.copy()
    f2.iloc[cut:] = f2.iloc[cut:].sample(frac=1.0, random_state=1).to_numpy()  # scramble the future
    X_cut = build_features(f2, "xrsb", "xrsa")
    pd.testing.assert_frame_equal(X_full.iloc[:cut], X_cut.iloc[:cut])


def test_eligibility_excludes_ongoing_flares(nowcast_result):
    f = records_to_frame(nowcast_result.records, ["xrsb", "xrsa"])
    el = eligible_mask(f, "xrsb")
    assert not el[f["xrsb_state"] == 1].any()
    assert not el[f["xrsb_value"].isna()].any()


def test_alert_manager_matches_batch_rule():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=500, freq="1min")
    p = pd.Series(rng.random(500), index=idx)
    el = pd.Series(rng.random(500) > 0.1, index=idx)
    batch = alert_series(p, el, 0.7, m=2, n=3)
    am = AlertManager(0.7, 2, 3)
    stream = [am.update(t, p[t] if el[t] else np.nan, el[t])[0] for t in idx]
    assert batch.tolist() == stream


def test_score_events_hits_misses_false_alarms():
    idx = pd.date_range("2026-01-01", periods=300, freq="1min")
    alert = pd.Series(False, index=idx)
    alert["2026-01-01 00:50":"2026-01-01 00:55"] = True    # 10 min before flare A -> hit
    alert["2026-01-01 03:00":"2026-01-01 03:02"] = True    # nothing follows -> false alarm
    flares = pd.DataFrame({
        "detect_time": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00"]),
        "start": pd.to_datetime(["2026-01-01 00:57", "2026-01-01 01:58"]),
        "peak_time": pd.to_datetime(["2026-01-01 01:05", "2026-01-01 02:06"]),
    })
    sc = score_events(alert, flares, 15)
    assert (sc.hits, sc.misses, sc.false_alarms, sc.episodes) == (1, 1, 1, 2)
    assert sc.lead_detect[0] == 10 and sc.lead_start[0] == 7
    assert sc.precursor_fraction == 1.0


def _quick_model(nowcast_result):
    f = records_to_frame(nowcast_result.records, ["xrsb", "xrsa"])
    X = build_features(f, "xrsb", "xrsa")
    ev = nowcast_result.events
    ev = ev[ev.channel == "xrsb"]
    y = make_targets(f.index, ev["detect_time"], 15)
    el = eligible_mask(f, "xrsb")
    mdl = FlareForecaster(list(X.columns), lgb_params={"n_estimators": 60}, xgb_params={"n_estimators": 60})
    mdl.fit(X[el], y[el])
    return mdl, f, X


def test_model_save_load_roundtrip(tmp_path, nowcast_result):
    mdl, f, X = _quick_model(nowcast_result)
    raw = np.clip(mdl.predict_raw(X.iloc[:500]), 0, 1)
    mdl.fit_calibration(raw, (raw > 0.5).astype(int))
    mdl.save(tmp_path)
    back = FlareForecaster.load(tmp_path)
    np.testing.assert_allclose(back.predict_proba(X.iloc[:500]), mdl.predict_proba(X.iloc[:500]), atol=1e-6)


def test_streaming_engine_matches_batch(nowcast_result):
    mdl, f, X = _quick_model(nowcast_result)
    df = synthetic_goes()
    eng = ForecastEngine(NowcastEngine(goes_channels(), 60), {"C1": (mdl, AlertManager(0.5))}, "xrsb", "xrsa")
    probs = {}
    for s in FrameSource(df):
        rec, _ = eng.update(s)
        probs[rec["time"]] = rec["p_C1"]
    stream = pd.Series(probs)
    el = eligible_mask(f, "xrsb")
    batch = pd.Series(mdl.predict_proba(X), index=X.index).where(el)
    both = stream.notna() & batch.notna()
    assert both.sum() > 3000
    assert (stream.isna() == batch.isna()).all()
    np.testing.assert_allclose(stream[both], batch[both], atol=1e-6)


def test_streaming_engine_aggregates_1s_input_to_1min_forecasts():
    """Aditya-L1 path: 1 s SoLEXS/HEL1OS samples, forecast every 60 s."""
    from l1_nowcast import aditya_channels
    rng = np.random.default_rng(2)
    n = 6 * 3600
    idx = pd.date_range("2026-03-01", periods=n, freq="1s")
    sec = np.arange(n)
    soft = rng.poisson(9, n).astype(float)
    hard = rng.poisson(0.2, n).astype(float)
    for t0 in (7200, 14400):
        soft += np.where(sec >= t0, 600 * np.exp(-np.maximum(sec - t0 - 300, 0) / 500) * np.minimum((sec - t0) / 300, 1), 0)
        hard += np.where((sec >= t0) & (sec < t0 + 200), rng.poisson(30, n), 0)
    df = pd.DataFrame({"solexs_counts": soft, "hel1os_cdte": hard}, index=idx)

    res = run_batch(df, aditya_channels(), 1)
    f = records_to_frame(res.records, ["solexs_counts", "hel1os_cdte"], cadence="60s")
    X = build_features(f, "solexs_counts", "hel1os_cdte", step_min=1.0)
    y = pd.Series((np.arange(len(X)) % 7 == 0).astype(int), index=X.index)  # any labels: equality test only
    mdl = FlareForecaster(list(X.columns), use_xgb=False, lgb_params={"n_estimators": 20, "min_child_samples": 5})
    mdl.fit(X, y)

    eng = ForecastEngine(NowcastEngine(aditya_channels(), 1), {"C1": (mdl, AlertManager(0.5))},
                         "solexs_counts", "hel1os_cdte", forecast_cadence_s=60)
    probs = {}
    for s in FrameSource(df):
        rec, _ = eng.update(s)
        if rec is not None:
            probs[rec["time"]] = rec["p_C1"]
    stream = pd.Series(probs)
    el = eligible_mask(f, "solexs_counts")
    batch = pd.Series(mdl.predict_proba(X), index=X.index).where(el).reindex(stream.index)
    assert len(stream) == len(X) - 1          # the last, still-open bin is not emitted
    assert (stream.isna() == batch.isna()).all()
    both = stream.notna()
    np.testing.assert_allclose(stream[both], batch[both], atol=1e-6)
