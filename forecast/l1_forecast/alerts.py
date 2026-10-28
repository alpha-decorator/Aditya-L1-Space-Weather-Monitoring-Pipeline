"""
Turning probabilities into operational alerts, and scoring them as events.

Alert rule: at least ``m`` of the last ``n`` eligible samples have
probability >= ``threshold`` (persistence filter against single-sample
spikes). ``alert_series`` (batch) and ``AlertManager`` (streaming) are the
same rule.

Event scoring (all relative to the nowcast catalogue):
  * a qualifying flare is a HIT if an alert sample occurs in
    [detect - H, detect), i.e. strictly before the nowcast alarm;
  * an alert episode is FALSE if no qualifying nowcast alarm follows within
    H of any of its samples;
  * lead time is measured from the first alert sample in the window to the
    nowcast alarm, to the physical start, and to the peak. Lead before the
    *start* is reported separately: those are true precursor warnings, the
    rest are early detections of the rise phase.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


def alert_series(prob: pd.Series, eligible: pd.Series, threshold: float, m: int = 1, n: int = 1) -> pd.Series:
    cross = ((prob >= threshold) & eligible.reindex(prob.index, fill_value=False)).astype(int)
    return (cross.rolling(n, min_periods=1).sum() >= m).rename("alert")


class AlertManager:
    """Streaming version of ``alert_series``."""

    def __init__(self, threshold: float, m: int = 1, n: int = 1):
        self.threshold, self.m, self.n = float(threshold), int(m), int(n)
        self._hist: deque[int] = deque(maxlen=self.n)
        self.active = False
        self.since: pd.Timestamp | None = None

    def update(self, t: pd.Timestamp, prob: float, eligible: bool) -> tuple[bool, list[dict]]:
        cross = int(bool(eligible) and np.isfinite(prob) and prob >= self.threshold)
        self._hist.append(cross)
        new = sum(self._hist) >= self.m
        msgs = []
        if new and not self.active:
            self.since = t
            msgs.append({"type": "forecast_alert_on", "time": t, "prob": prob})
        elif self.active and not new:
            msgs.append({"type": "forecast_alert_off", "time": t, "prob": prob, "since": self.since})
            self.since = None
        self.active = new
        return new, msgs


def alert_episodes(alert: pd.Series) -> pd.DataFrame:
    a = alert.to_numpy(dtype=bool)
    if not a.any():
        return pd.DataFrame(columns=["start", "end"])
    d = np.diff(np.concatenate([[0], a.astype(int), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    idx = alert.index
    return pd.DataFrame({"start": idx[starts], "end": idx[ends]})


@dataclass
class EventScore:
    hits: int
    misses: int
    false_alarms: int
    episodes: int
    days: float
    lead_detect: np.ndarray
    lead_start: np.ndarray
    lead_peak: np.ndarray
    hit_flags: np.ndarray

    @property
    def pod(self):
        return self.hits / max(1, self.hits + self.misses)

    @property
    def far(self):
        """False alarm RATIO: fraction of alert episodes not followed by a flare."""
        return self.false_alarms / max(1, self.episodes)

    @property
    def csi(self):
        return self.hits / max(1, self.hits + self.misses + self.false_alarms)

    @property
    def precursor_fraction(self):
        """Fraction of hits whose first alert came before the physical start."""
        return float(np.mean(self.lead_start > 0)) if len(self.lead_start) else math.nan

    def as_dict(self) -> dict:
        med = lambda a: float(np.median(a)) if len(a) else math.nan
        return {
            "flares": self.hits + self.misses, "hits": self.hits, "misses": self.misses,
            "alert_episodes": self.episodes, "false_alarms": self.false_alarms,
            "POD": self.pod, "FAR(ratio)": self.far, "CSI": self.csi,
            "false_alarms_per_day": self.false_alarms / max(self.days, 1e-9),
            "lead_vs_nowcast_alarm_median_min": med(self.lead_detect),
            "lead_vs_start_median_min": med(self.lead_start),
            "lead_vs_peak_median_min": med(self.lead_peak),
            "precursor_fraction": self.precursor_fraction,
        }


def score_events(alert: pd.Series, flares: pd.DataFrame, horizon_min: float,
                 period: tuple | None = None) -> EventScore:
    """Score an alert series against qualifying nowcast flares
    (``detect_time``, ``start``, ``peak_time`` columns)."""
    idx = alert.index
    if period is not None:
        lo, hi = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    else:
        lo, hi = idx.min(), idx.max()
    fl = flares[(flares["detect_time"] >= lo) & (flares["detect_time"] <= hi)].reset_index(drop=True)
    a = alert.reindex(idx).fillna(False).to_numpy(dtype=bool)
    t = idx.to_numpy(dtype="datetime64[ns]")
    alert_times = t[a]
    H = np.timedelta64(int(horizon_min * 60), "s")

    hit_flags = np.zeros(len(fl), dtype=bool)
    lead_d, lead_s, lead_p = [], [], []
    for i, r in fl.iterrows():
        D = np.datetime64(r["detect_time"])
        j0 = np.searchsorted(alert_times, D - H, side="left")
        j1 = np.searchsorted(alert_times, D, side="left")
        if j1 > j0:
            hit_flags[i] = True
            first = alert_times[j0]
            lead_d.append((D - first) / np.timedelta64(1, "m"))
            lead_s.append((np.datetime64(r["start"]) - first) / np.timedelta64(1, "m"))
            lead_p.append((np.datetime64(r["peak_time"]) - first) / np.timedelta64(1, "m"))

    eps = alert_episodes(pd.Series(a, index=idx))
    det = np.sort(pd.to_datetime(flares["detect_time"]).to_numpy(dtype="datetime64[ns]"))
    fa = 0
    for e in eps.itertuples(index=False):
        s, en = np.datetime64(e.start), np.datetime64(e.end)
        k = np.searchsorted(det, s, side="right")
        if not (k < len(det) and det[k] <= en + H):
            fa += 1
    days = (hi - lo).total_seconds() / 86400.0
    return EventScore(hits=int(hit_flags.sum()), misses=int((~hit_flags).sum()), false_alarms=fa,
                      episodes=len(eps), days=days, lead_detect=np.array(lead_d),
                      lead_start=np.array(lead_s), lead_peak=np.array(lead_p), hit_flags=hit_flags)


def tune_alert_rule(prob: pd.Series, eligible: pd.Series, flares: pd.DataFrame, horizon_min: float,
                    thresholds=None, persistence=((1, 1), (2, 2), (2, 3), (3, 3), (3, 5)),
                    objective: str = "CSI") -> pd.DataFrame:
    """Grid-search threshold x persistence on (out-of-sample) probabilities."""
    if thresholds is None:
        q = prob[eligible.reindex(prob.index, fill_value=False)].dropna()
        thresholds = np.unique(np.round(np.quantile(q, np.linspace(0.80, 0.995, 40)), 4))
    rows = []
    for (m, n) in persistence:
        for th in thresholds:
            al = alert_series(prob, eligible, th, m, n)
            sc = score_events(al, flares, horizon_min)
            rows.append({"threshold": float(th), "m": m, "n": n, **sc.as_dict()})
    return pd.DataFrame(rows).sort_values(objective, ascending=False).reset_index(drop=True)


def probability_scores(y: np.ndarray, p: np.ndarray, base_rate: float | None = None) -> dict:
    """Sample-level skill: ROC AUC, PR AUC, Brier score and Brier skill score
    against climatology (``base_rate``: the training positive rate)."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    ok = np.isfinite(p)
    y, p = np.asarray(y)[ok], np.asarray(p)[ok]
    clim = float(np.mean(y)) if base_rate is None else float(base_rate)
    bs = brier_score_loss(y, p)
    bs_ref = brier_score_loss(y, np.full_like(p, clim, dtype=float))
    return {"n": int(len(y)), "pos_rate": float(np.mean(y)), "ROC_AUC": roc_auc_score(y, p),
            "PR_AUC": average_precision_score(y, p), "Brier": bs, "BSS_vs_climatology": 1 - bs / bs_ref}


def tss_at(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """True Skill Statistic (POD - POFD) per sample at a threshold."""
    ok = np.isfinite(p)
    y, pred = np.asarray(y)[ok].astype(bool), np.asarray(p)[ok] >= threshold
    tp = np.sum(pred & y); fn = np.sum(~pred & y); fp = np.sum(pred & ~y); tn = np.sum(~pred & ~y)
    return tp / max(1, tp + fn) - fp / max(1, fp + tn)
