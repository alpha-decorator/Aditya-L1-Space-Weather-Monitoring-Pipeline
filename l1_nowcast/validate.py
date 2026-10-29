"""
Validation of a nowcast catalogue against a reference flare list
(NOAA's official GOES XRS flare summary), and parameter tuning.
"""

from __future__ import annotations

import itertools
import math
import time as _time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .detector import ChannelConfig, class_letter, class_to_flux, run_batch

CLASS_ORDER = ["A", "B", "C", "M", "X"]


def _peak_flux_of(df: pd.DataFrame, flux_col: str, class_col: str) -> np.ndarray:
    f = df[flux_col].to_numpy(dtype=float) if flux_col in df else np.full(len(df), np.nan)
    if class_col in df:
        from_cls = df[class_col].map(class_to_flux).to_numpy(dtype=float)
        f = np.where(np.isfinite(f), f, from_cls)
    return f


def match_events(ours: pd.DataFrame, ref: pd.DataFrame, tolerance_s: float = 300.0) -> pd.DataFrame:
    """One-to-one matching of detected flares to reference flares.

    A detected flare is a candidate for a reference flare when its
    [start, end] interval overlaps [ref.start - tol, ref.end + tol]. Reference
    flares are processed largest-first and each takes the unused candidate
    with the closest peak time, so a big flare is never "stolen" by a small
    neighbour.

    Returns the reference table with ``match_idx`` (row in ``ours`` or -1)
    and timing columns.
    """
    tol = pd.Timedelta(seconds=tolerance_s)
    ref = ref.reset_index(drop=True).copy()
    ours = ours.reset_index(drop=True)
    o_start = ours["start"].to_numpy(dtype="datetime64[ns]")
    o_end = ours["end"].to_numpy(dtype="datetime64[ns]")
    o_peak = ours["peak_time"].to_numpy(dtype="datetime64[ns]")
    used = np.zeros(len(ours), dtype=bool)
    match = np.full(len(ref), -1, dtype=int)
    order = np.argsort(-_peak_flux_of(ref, "peak_flux", "flare_class"))
    for i in order:
        r = ref.iloc[i]
        lo = np.datetime64(r["start"] - tol)
        hi = np.datetime64(r["end"] + tol)
        cand = np.flatnonzero((o_start <= hi) & (o_end >= lo) & ~used)
        if cand.size == 0:
            continue
        d = np.abs((o_peak[cand] - np.datetime64(r["peak"])).astype("timedelta64[s]").astype(float))
        j = cand[np.argmin(d)]
        used[j] = True
        match[i] = j
    ref["match_idx"] = match
    ok = match >= 0
    for col, src in (("det_start", "start"), ("det_detect", "detect_time"),
                     ("det_peak_time", "peak_time"), ("det_class", "flare_class"), ("det_peak", "peak")):
        vals = ours[src].to_numpy() if src in ours else np.full(len(ours), None)
        out = np.full(len(ref), None, dtype=object)
        out[ok] = vals[match[ok]]
        ref[col] = out
    ref["det_detect"] = pd.to_datetime(ref["det_detect"])
    ref["det_peak_time"] = pd.to_datetime(ref["det_peak_time"])
    ref["det_start"] = pd.to_datetime(ref["det_start"])
    ref["alert_latency_min"] = (ref["det_detect"] - ref["start"]).dt.total_seconds() / 60
    ref["peak_offset_min"] = (ref["det_peak_time"] - ref["peak"]).dt.total_seconds() / 60
    ref["alert_before_peak_min"] = (ref["peak"] - ref["det_detect"]).dt.total_seconds() / 60
    return ref


@dataclass
class ValidationReport:
    matched: pd.DataFrame          # reference flares with match info
    by_class: pd.DataFrame         # POD per reference class letter
    precision: float               # fraction of detections >= min_class matching a reference flare
    recall: float                  # POD for reference flares >= min_class
    f1: float
    n_detected: int
    n_reference: int
    timing: dict
    class_agreement: float         # same class letter, matched >= min_class flares

    def summary(self) -> pd.Series:
        return pd.Series({"precision": self.precision, "recall(POD)": self.recall, "F1": self.f1,
                          "detected(>=min)": self.n_detected, "reference(>=min)": self.n_reference,
                          "class_letter_agreement": self.class_agreement, **self.timing})


def evaluate(ours: pd.DataFrame, ref: pd.DataFrame, min_class: str = "C1.0",
             tolerance_s: float = 300.0, exclude_days: pd.DatetimeIndex | None = None) -> ValidationReport:
    """Score a detected catalogue against a reference list.

    ``min_class`` sets the population that counts: reference flares at or
    above it for recall, detections at or above it for precision (a C-class
    detection matched to a reference B-flare still counts as correct).
    ``exclude_days`` drops days where the reference list is unavailable.
    """
    fmin = class_to_flux(min_class)
    if exclude_days is not None and len(exclude_days):
        bad = set(pd.DatetimeIndex(exclude_days).normalize())
        ours = ours[~ours["start"].dt.normalize().isin(bad)]
        ref = ref[~ref["start"].dt.normalize().isin(bad)]
    ours = ours.reset_index(drop=True)
    ref = ref.reset_index(drop=True)
    m = match_events(ours, ref, tolerance_s)
    ref_flux = _peak_flux_of(m, "peak_flux", "flare_class")
    ref_big = ref_flux >= fmin * (1 - 1e-9)
    our_flux = _peak_flux_of(ours, "peak", "flare_class")
    our_big = our_flux >= fmin * (1 - 1e-9)
    matched_ours = set(m.loc[m["match_idx"] >= 0, "match_idx"].tolist())
    our_big_idx = np.flatnonzero(our_big)
    tp_prec = sum(1 for j in our_big_idx if j in matched_ours)
    precision = tp_prec / max(1, len(our_big_idx))
    recall = float((m.loc[ref_big, "match_idx"] >= 0).mean()) if ref_big.any() else math.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    m["ref_letter"] = [class_letter(c) if isinstance(c, str) else class_letter(f)
                       for c, f in zip(m["flare_class"], ref_flux)]
    by = (m.assign(detected=m["match_idx"] >= 0)
           .groupby("ref_letter")["detected"].agg(["count", "sum", "mean"])
           .rename(columns={"count": "n_reference", "sum": "n_detected", "mean": "POD"}))
    by = by.reindex([c for c in CLASS_ORDER if c in by.index])

    hit = m[(m["match_idx"] >= 0) & ref_big]
    timing = {
        "alert_latency_median_min": float(hit["alert_latency_min"].median()) if len(hit) else math.nan,
        "alert_latency_p90_min": float(hit["alert_latency_min"].quantile(0.9)) if len(hit) else math.nan,
        "alert_before_peak_median_min": float(hit["alert_before_peak_min"].median()) if len(hit) else math.nan,
        "peak_offset_median_min": float(hit["peak_offset_min"].median()) if len(hit) else math.nan,
    }
    agree = [class_letter(a) == class_letter(b) for a, b in zip(hit["flare_class"], hit["det_class"])
             if isinstance(a, str) and isinstance(b, str)]
    return ValidationReport(matched=m, by_class=by, precision=precision, recall=recall, f1=f1,
                            n_detected=int(our_big.sum()), n_reference=int(ref_big.sum()),
                            timing=timing, class_agreement=float(np.mean(agree)) if agree else math.nan)


def tune_channel(df: pd.DataFrame, ref: pd.DataFrame, base: ChannelConfig, grid: dict[str, list],
                 cadence_s: float, min_class: str = "C1.0", exclude_days=None,
                 objective: str = "f1", verbose: bool = True) -> pd.DataFrame:
    """Grid-search detection parameters for one channel against a reference list.

    Only the channel in ``base`` is run (fusion is irrelevant for tuning), so
    each configuration costs one pass of the streaming engine over ``df``.
    Returns one row per configuration, best first.
    """
    keys = list(grid)
    rows = []
    combos = list(itertools.product(*(grid[k] for k in keys)))
    t0 = _time.time()
    for n, combo in enumerate(combos, 1):
        cfg = ChannelConfig(**{**base.to_dict(), **dict(zip(keys, combo))})
        res = run_batch(df[[cfg.name]], [cfg], cadence_s, keep_records=False)
        rep = evaluate(res.events, ref, min_class=min_class, exclude_days=exclude_days)
        rows.append({**dict(zip(keys, combo)), "precision": rep.precision, "recall": rep.recall,
                     "f1": rep.f1, "n_detected": rep.n_detected,
                     "latency_med_min": rep.timing["alert_latency_median_min"]})
        if verbose and (n == 1 or n % 10 == 0 or n == len(combos)):
            print(f"  [{n:3d}/{len(combos)}] {dict(zip(keys, combo))} -> F1={rep.f1:.3f} "
                  f"P={rep.precision:.3f} R={rep.recall:.3f}  ({_time.time() - t0:.0f}s)")
    out = pd.DataFrame(rows).sort_values([objective, "latency_med_min"], ascending=[False, True])
    return out.reset_index(drop=True)
