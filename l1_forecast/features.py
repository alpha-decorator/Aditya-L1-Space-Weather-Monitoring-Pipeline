"""
Causal, gap-aware forecast features.

Input is the per-sample record stream produced by the nowcast engine
(value, background, state, event id per channel). Every feature at time t
uses only rows <= t, and every window is defined in *minutes* on a
regular grid, so a data gap shows up as NaN instead of silently stretching
a window across hours (the row-count windows of the original notebook did
exactly that across SoLEXS gaps).

The same ``build_features`` function serves batch training and live
streaming: the streaming engine keeps the last ``required_history_min()``
minutes in a buffer and evaluates the function on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from l1_nowcast import DECAY, FLARE

DIFF_LAGS_MIN = (1, 2, 3, 5, 10, 15, 30)
EXC_MAX_WINDOWS_MIN = (30, 60, 180, 360, 1440)
VOL_WINDOWS_MIN = (10, 30, 60)
ABOVE_MIN_WINDOWS_MIN = (10, 30)
SEC_DIFF_LAGS_MIN = (1, 3, 5, 10)
ACTIVITY_WINDOWS_MIN = (360, 1440)
SINCE_CAP_MIN = 1440.0


def required_history_min() -> int:
    """Minutes of history the features need for an exact (batch-identical) value."""
    return int(max(max(EXC_MAX_WINDOWS_MIN), max(ACTIVITY_WINDOWS_MIN), SINCE_CAP_MIN)) + 60


def _log10(s: pd.Series) -> pd.Series:
    v = s.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.Series(np.where(v > 0, np.log10(v), np.nan), index=s.index)


def records_to_frame(records: pd.DataFrame, channels: list[str], cadence: str | None = None) -> pd.DataFrame:
    """Reduce nowcast records to what the forecaster needs, optionally
    aggregated to a coarser forecast cadence (e.g. 1 s Aditya-L1 -> 1 min).

    Per channel: ``value`` and ``background`` (mean), ``state`` (last),
    ``onset`` (a nowcast flare was confirmed in this bin), ``end`` (a flare
    closed in this bin).
    """
    out = {}
    for ch in channels:
        eid = records[f"{ch}_event_id"]
        prev = eid.shift(1)
        has = eid.notna()
        onset = has & (eid != prev)
        ended = prev.notna() & (eid != prev)
        out[f"{ch}_value"] = records[f"{ch}_value"].astype(float)
        out[f"{ch}_background"] = records[f"{ch}_background"].astype(float)
        out[f"{ch}_state"] = records[f"{ch}_state"].astype(int)
        out[f"{ch}_onset"] = onset.astype(int)
        out[f"{ch}_end"] = ended.astype(int)
    f = pd.DataFrame(out, index=records.index)
    if cadence is None:
        return f
    rules = {}
    for ch in channels:
        rules.update({f"{ch}_value": "mean", f"{ch}_background": "mean", f"{ch}_state": "last",
                      f"{ch}_onset": "max", f"{ch}_end": "max"})
    g = f.resample(cadence, label="left", closed="left").agg(rules)
    return g


def _run_length(mask: np.ndarray, cap: int) -> np.ndarray:
    """Length of the current run of True values ending at each row (capped)."""
    out = np.zeros(len(mask), dtype=float)
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        out[i] = min(run, cap)
    return out


def _minutes_since(flag: np.ndarray, step_min: float, cap: float) -> np.ndarray:
    idx = np.arange(len(flag), dtype=float)
    last = np.where(flag > 0, idx, np.nan)
    last = pd.Series(last).ffill().to_numpy()
    out = (idx - last) * step_min
    out = np.where(np.isfinite(out), out, cap)
    return np.minimum(out, cap)


def build_features(f: pd.DataFrame, primary: str, secondary: str | None = None,
                   step_min: float = 1.0, absolute: bool = True) -> pd.DataFrame:
    """Feature matrix from a regular-cadence frame made by ``records_to_frame``.

    Groups (all causal):
      * primary excess over the nowcast background, its recent maxima
      * primary growth over 1..30 min (log-flux differences) and volatility
      * how far flux has climbed above its recent minimum, current rise run
      * secondary channel excess / growth, and the secondary/primary ratio
        (for GOES: XRS-A/XRS-B, a plasma-temperature proxy - pre-flare
        heating is one of the few known soft X-ray precursors)
      * flaring activity: nowcast onsets in the last 6/24 h, minutes since
        the last onset / flare end, decay-phase flag
      * absolute level (optional: off when transferring between instruments
        with different units, e.g. GOES flux -> SoLEXS counts)
    """
    def n(minutes: float) -> int:
        return max(1, int(round(minutes / step_min)))

    out: dict[str, pd.Series | np.ndarray] = {}
    p = _log10(f[f"{primary}_value"])
    b = _log10(f[f"{primary}_background"])
    exc = p - b
    out["p_exc"] = exc
    for k in DIFF_LAGS_MIN:
        out[f"p_d{k}"] = p - p.shift(n(k))
    for w in EXC_MAX_WINDOWS_MIN:
        out[f"p_exc_max{w}"] = exc.rolling(n(w), min_periods=1).max()
    d1 = p - p.shift(1)
    for w in VOL_WINDOWS_MIN:
        out[f"p_vol{w}"] = d1.rolling(n(w), min_periods=max(2, n(w) // 2)).std()
    for w in ABOVE_MIN_WINDOWS_MIN:
        out[f"p_above_min{w}"] = p - p.rolling(n(w), min_periods=1).min()
    out["p_rise_run"] = _run_length((d1 > 0).to_numpy(), cap=10)
    out["p_exc_d3"] = exc - exc.shift(n(3))

    if secondary is not None and f"{secondary}_value" in f:
        s = _log10(f[f"{secondary}_value"])
        sb = _log10(f[f"{secondary}_background"])
        out["s_exc"] = s - sb
        for k in SEC_DIFF_LAGS_MIN:
            out[f"s_d{k}"] = s - s.shift(n(k))
        tr = s - p
        out["tr"] = tr
        for k in SEC_DIFF_LAGS_MIN:
            out[f"tr_d{k}"] = tr - tr.shift(n(k))
        out["tr_dev120"] = tr - tr.rolling(n(120), min_periods=n(30)).median()

    state = f[f"{primary}_state"].to_numpy()
    onset = f[f"{primary}_onset"].to_numpy()
    ended = f[f"{primary}_end"].to_numpy()
    out["in_decay"] = (state == DECAY).astype(float)
    onset_s = pd.Series(onset.astype(float), index=f.index)
    for w in ACTIVITY_WINDOWS_MIN:
        out[f"onsets_{w // 60}h"] = onset_s.rolling(n(w), min_periods=1).sum()
    out["min_since_onset"] = _minutes_since(onset, step_min, SINCE_CAP_MIN)
    out["min_since_end"] = _minutes_since(ended, step_min, SINCE_CAP_MIN)
    out["valid_frac60"] = p.notna().astype(float).rolling(n(60), min_periods=1).mean()

    if absolute:
        out["p_level"] = p
        out["b_level"] = b
        out["p_max1440"] = p.rolling(n(1440), min_periods=1).max()

    X = pd.DataFrame(out, index=f.index)
    return X.astype("float32")


def eligible_mask(f: pd.DataFrame, primary: str) -> pd.Series:
    """Rows where a forecast is meaningful: valid data and no flare in progress.

    Rows inside a nowcast FLARE are excluded from training and scoring - at
    those times the nowcast already knows, and including them is how a
    forecaster ends up "predicting" flares that are already happening.
    """
    v = f[f"{primary}_value"]
    return (v.notna() & (f[f"{primary}_state"] != FLARE)).rename("eligible")
