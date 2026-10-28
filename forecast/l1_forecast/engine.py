"""
Streaming forecast engine: nowcast + forecast on one live sample stream.

    sample ─► NowcastEngine ─► record ─► (aggregate to forecast cadence)
                                   │
                                   └► rolling buffer ─► build_features ─► model(s) ─► AlertManager(s)

Several forecasters can run side by side (e.g. ``"C1"`` for any C-class or
larger flare, ``"M1"`` for major flares); features are computed once per
step and each model picks the columns it was trained on.

Each forecast step returns one combined record (flux, background, nowcast
state, one probability and alert flag per model) plus the messages from
both stages (nowcast onset / escalation / end / fused, and
forecast_alert_on / forecast_alert_off tagged with the model name) - the
feed a dashboard subscribes to.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from l1_nowcast import FLARE, STATE_NAMES, NowcastEngine, Sample

from .alerts import AlertManager
from .features import build_features, required_history_min
from .model import FlareForecaster


class ForecastEngine:
    def __init__(self, nowcast: NowcastEngine,
                 models: dict[str, tuple[FlareForecaster, AlertManager]],
                 primary: str, secondary: str | None = None,
                 forecast_cadence_s: float | None = None, horizon_min: float = 15.0):
        if not models:
            raise ValueError("need at least one model")
        self.nowcast = nowcast
        self.models = models
        self.primary, self.secondary = primary, secondary
        self.channels = [primary] + ([secondary] if secondary else [])
        self.native_s = nowcast.cadence_s
        self.cadence_s = float(forecast_cadence_s or nowcast.cadence_s)
        if self.cadence_s % self.native_s:
            raise ValueError("forecast cadence must be a multiple of the nowcast cadence")
        self.step_min = self.cadence_s / 60.0
        self.horizon_min = horizon_min
        self.buffer_rows = int(math.ceil(required_history_min() / self.step_min)) + 2
        self._native_rows: list[dict] = []
        self._bin: float | None = None
        self._frame_rows: deque = deque(maxlen=self.buffer_rows)
        self._prev_eid: dict[str, str | None] = {c: None for c in self.channels}
        self.last: dict | None = None

    def _native_to_row(self, rec: dict) -> dict:
        row = {}
        for ch in self.channels:
            eid = rec[f"{ch}_event_id"]
            prev = self._prev_eid[ch]
            row[f"{ch}_value"] = rec[f"{ch}_value"]
            row[f"{ch}_background"] = rec[f"{ch}_background"]
            row[f"{ch}_state"] = rec[f"{ch}_state"]
            row[f"{ch}_onset"] = int(eid is not None and eid != prev)
            row[f"{ch}_end"] = int(prev is not None and eid != prev)
            self._prev_eid[ch] = eid
        return row

    def _aggregate(self, rows: list[dict]) -> dict:
        """Same reduction as ``records_to_frame(..., cadence=...)``."""
        if len(rows) == 1:
            return rows[0]
        out = {}
        for ch in self.channels:
            for k in ("value", "background"):
                xs = [r[f"{ch}_{k}"] for r in rows if np.isfinite(r[f"{ch}_{k}"])]
                out[f"{ch}_{k}"] = float(np.mean(xs)) if xs else math.nan
            out[f"{ch}_state"] = rows[-1][f"{ch}_state"]
            out[f"{ch}_onset"] = max(r[f"{ch}_onset"] for r in rows)
            out[f"{ch}_end"] = max(r[f"{ch}_end"] for r in rows)
        return out

    def update(self, sample: Sample) -> tuple[dict | None, list[dict]]:
        """Feed one native-cadence sample. Returns a forecast record whenever
        a forecast-cadence bin completes (every sample when cadences match),
        else ``None``, plus all messages produced by this sample."""
        rec, msgs = self.nowcast.update(sample)
        t = pd.Timestamp(sample.time)
        if self.cadence_s == self.native_s:
            self._native_rows = [self._native_to_row(rec)]
            return self._emit(t, msgs), msgs
        b = math.floor(t.value / 1e9 / self.cadence_s) * self.cadence_s
        out = None
        if self._bin is not None and b != self._bin and self._native_rows:
            out = self._emit(pd.Timestamp(int(self._bin * 1e9)), msgs)
            self._native_rows = []
        self._bin = b
        self._native_rows.append(self._native_to_row(rec))
        return out, msgs

    def _emit(self, t: pd.Timestamp, msgs: list[dict]) -> dict:
        row = self._aggregate(self._native_rows)
        self._frame_rows.append((t, row))
        idx = pd.DatetimeIndex([r[0] for r in self._frame_rows])
        f = pd.DataFrame([r[1] for r in self._frame_rows], index=idx)
        X = build_features(f, self.primary, self.secondary, self.step_min, absolute=True).iloc[[-1]]
        state = row[f"{self.primary}_state"]
        eligible = bool(np.isfinite(row[f"{self.primary}_value"]) and state != FLARE)
        out = {
            "time": t,
            "value": row[f"{self.primary}_value"],
            "background": row[f"{self.primary}_background"],
            "secondary_value": row.get(f"{self.secondary}_value", math.nan) if self.secondary else math.nan,
            "nowcast_state": STATE_NAMES[state],
            "horizon_min": self.horizon_min,
        }
        for name, (model, alert) in self.models.items():
            p = float(model.predict_proba(X)[0]) if eligible else math.nan
            active, amsgs = alert.update(t, p, eligible)
            for m in amsgs:
                m["target"] = name
            msgs.extend(amsgs)
            out[f"p_{name}"] = p
            out[f"alert_{name}"] = bool(active)
        self.last = out
        return out

    def run(self, stream):
        for s in stream:
            rec, msgs = self.update(s)
            if rec is not None or msgs:
                yield rec, msgs
