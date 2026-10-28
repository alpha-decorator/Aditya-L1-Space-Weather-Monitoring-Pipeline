"""
Streaming flare nowcaster.

Per channel, every incoming sample goes through:

    background (trailing, flare-resistant)  ->  threshold  ->  state machine

and the state machine moves between three states:

    QUIET  --(confirm consecutive samples above threshold)-->  FLARE
    FLARE  --(flux decays to bg + end_fraction*(peak-bg))--->  DECAY  (flare closed)
    FLARE  --(below threshold longer than max_dropout_s)---->  QUIET  (flare closed)
    FLARE  --(sustained re-rise after the peak)------------->  FLARE  (old flare closed,
                                                                 new one opened)
    DECAY  --(rise_samples consecutive rises, >= rise_ratio)->  FLARE  (new flare
                                                                 on a decaying tail)
    DECAY  --(back below threshold)------------------------->  QUIET

The end rule and the "new flare during decay" rule mirror the NOAA GOES
flare definition (end = half-way back to pre-flare flux; start = sustained
rise), so the catalogue is directly comparable with NOAA's official list.

Events from a soft (primary) and a hard/short-wavelength (secondary)
channel are then fused: a flare seen by both is high-confidence.

There is exactly ONE implementation: batch processing is just this
engine fed row by row, so batch back-tests and the live stream always
agree.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .background import AdaptiveBackground
from .sources import Sample, to_epoch_s

QUIET, FLARE, DECAY = 0, 1, 2
STATE_NAMES = {QUIET: "QUIET", FLARE: "FLARE", DECAY: "DECAY"}


# ---------------------------------------------------------------------------
# GOES classes
# ---------------------------------------------------------------------------

_CLASS_BASES = (("X", 1e-4), ("M", 1e-5), ("C", 1e-6), ("B", 1e-7), ("A", 1e-8))


def goes_class(flux: float | None) -> str | None:
    """Peak 1-8 A flux in W/m^2 -> GOES class string, e.g. 1.53e-4 -> 'X1.5'.

    The magnitude is truncated (not rounded) to one decimal, so a class never
    crosses a letter boundary by rounding (9.97e-6 is C9.9, not C10.0).
    """
    if flux is None or not np.isfinite(flux) or flux <= 0:
        return None
    for letter, base in _CLASS_BASES:
        if flux >= base:
            return f"{letter}{math.floor(flux / base * 10 + 1e-9) / 10:.1f}"
    return f"A{math.floor(flux / 1e-8 * 10 + 1e-9) / 10:.1f}"


def class_to_flux(cls: str | None) -> float:
    """'M2.3' -> 2.3e-5. Returns NaN for unparseable input."""
    if not cls or not isinstance(cls, str):
        return math.nan
    bases = dict(_CLASS_BASES)
    letter = cls[0].upper()
    if letter not in bases:
        return math.nan
    try:
        return float(cls[1:]) * bases[letter]
    except ValueError:
        return math.nan


def class_letter(cls_or_flux) -> str | None:
    if isinstance(cls_or_flux, str):
        return cls_or_flux[0].upper() if cls_or_flux else None
    c = goes_class(cls_or_flux)
    return c[0] if c else None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    """Detection settings for one instrument channel.

    model
        ``log_robust``  median / IQR of log10(value)   - fluxes (GOES XRS)
        ``robust``      median / IQR of value          - count rates (SoLEXS)
        ``poisson``     mean + k*sqrt(mean)            - sparse photon counts
                                                         (HEL1OS, mostly zeros)
    """
    name: str
    model: str = "log_robust"
    role: str = "primary"            # "primary" (soft) or "secondary" (hard / corroboration)
    window_s: float = 7200.0         # trailing background window
    min_count: int = 60              # samples needed in the window before detecting
    k_sigma: float = 4.0             # threshold = center + k * spread ...
    min_ratio: float = 1.3           # ... and at least center * min_ratio (log/robust models)
    sigma_floor: float = 0.0         # lower bound on spread, model units
    abs_floor: float = 0.0           # poisson: minimum excess above mean, in counts
    confirm: int = 2                 # consecutive above-threshold samples to open a flare
    max_dropout_s: float = 180.0     # flare survives this long below threshold / without data
    end_fraction: float = 0.5        # flare ends at bg + end_fraction * (peak - bg)
    rise_samples: int = 3            # consecutive increases ...
    rise_ratio: float = 1.4          # ... totalling at least this factor -> flare start
    rise_in_quiet: bool = True       # rise rule may also open a flare from QUIET (else DECAY only)
    split_on_rerise: bool = True     # a sustained re-rise after the peak starts a new flare
    onset_min_rise: float = 1.0      # level onset also needs value >= this x min(last onset_lookback)
    onset_lookback: int = 5          #   samples (1.0 = off); rejects noise on slowly decaying tails
    stale_s: float = 6 * 3600.0      # re-seed background if no quiet sample for this long
    smooth_n: int = 1                # moving average over the last n samples (noisy 1 s counts)
    is_goes_flux: bool = False       # peak value is a GOES 1-8 A flux -> class from it
    units: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def goes_channels(**overrides) -> list[ChannelConfig]:
    """GOES-R XRS at 1-minute cadence: XRS-B (1-8 A) primary, XRS-A (0.5-4 A)
    secondary. Defaults are the values tuned in nowcast.ipynb on Jan-Apr 2026
    against NOAA's official flare list (XRS-B: best dev F1 = 0.909; the onset
    rise guard and the stricter rise rule lift precision from ~0.70 to ~0.90. XRS-A: stricter ratio
    and no quiet-state rise trigger, because near its noise floor small
    fluctuations look like rises). Pass overrides to experiment, e.g.
    ``goes_channels(xrsb={"k_sigma": 3})``."""
    common = dict(model="log_robust", window_s=7200.0, min_count=60, k_sigma=1.5,
                  min_ratio=1.3, confirm=2, max_dropout_s=180.0, end_fraction=0.5,
                  rise_samples=2, rise_ratio=1.6, rise_in_quiet=True, stale_s=6 * 3600.0,
                  onset_min_rise=1.3, onset_lookback=5, units="W/m^2")
    p = {**common, **overrides.get("xrsb", {})}
    s = {**common, "k_sigma": 3.0, "min_ratio": 1.5, "rise_in_quiet": False, "onset_min_rise": 1.0,
         **overrides.get("xrsa", {})}
    return [
        ChannelConfig(name="xrsb", role="primary", is_goes_flux=True, **p),
        ChannelConfig(name="xrsa", role="secondary", **s),
    ]


def aditya_channels(**overrides) -> list[ChannelConfig]:
    """Aditya-L1 at 1-second cadence: SoLEXS soft X-ray counts primary,
    HEL1OS CdTe hard X-ray counts secondary.

    SoLEXS uses the robust (median/IQR) model the team validated; HEL1OS CdTe
    is zero-inflated, so it uses Poisson statistics with an absolute floor
    (median/MAD collapses to zero there - see phase2_background1.py).
    These are starting values; re-tune with ``tune_channel`` once SoLEXS/HEL1OS
    data is loaded.
    """
    soft = dict(model="robust", window_s=7200.0, min_count=600, k_sigma=5.0, min_ratio=1.5,
                confirm=10, max_dropout_s=60.0, end_fraction=0.5, rise_samples=20,
                rise_ratio=1.4, rise_in_quiet=False, smooth_n=10, units="counts/s")
    hard = dict(model="poisson", window_s=7200.0, min_count=600, k_sigma=5.0, abs_floor=5.0,
                confirm=5, max_dropout_s=60.0, end_fraction=0.5, rise_samples=10,
                rise_ratio=2.0, rise_in_quiet=False, smooth_n=5, units="counts/s")
    soft.update(overrides.get("solexs_counts", {}))
    hard.update(overrides.get("hel1os_cdte", {}))
    return [
        ChannelConfig(name="solexs_counts", role="primary", **soft),
        ChannelConfig(name="hel1os_cdte", role="secondary", **hard),
    ]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class FlareEvent:
    event_id: str
    channel: str
    start: pd.Timestamp              # first sample of the flare (retrospective start)
    detect_time: pd.Timestamp        # when the nowcaster confirmed it (causal alert time)
    background: float                # pre-flare level (linear units)
    trigger: str                     # "level" or "rise" (new flare on a decaying tail)
    peak_time: pd.Timestamp = None
    peak: float = -math.inf
    end: pd.Timestamp | None = None
    end_reason: str | None = None
    fluence: float = 0.0             # integrated excess over background (units * s)
    n_samples: int = 0
    flare_class: str | None = None   # GOES class (flux channels or calibrated counts)

    def as_record(self) -> dict:
        d = asdict(self)
        d["duration_min"] = ((self.end - self.start).total_seconds() / 60.0) if self.end is not None else math.nan
        d["detect_delay_min"] = (self.detect_time - self.start).total_seconds() / 60.0
        d["rise_min"] = (self.peak_time - self.start).total_seconds() / 60.0 if self.peak_time is not None else math.nan
        return d


# ---------------------------------------------------------------------------
# Per-channel detector
# ---------------------------------------------------------------------------

class ChannelDetector:
    def __init__(self, cfg: ChannelConfig, cadence_s: float,
                 calibration: Callable[[float], float] | None = None):
        self.cfg = cfg
        self.cadence_s = float(cadence_s)
        self.calibration = calibration
        self.bg = AdaptiveBackground(cfg.window_s, cfg.min_count, cfg.stale_s, cfg.model)
        self.state = QUIET
        self.event: FlareEvent | None = None
        self.closed: list[FlareEvent] = []
        self._n = 0
        self._run = 0
        self._cand: tuple[pd.Timestamp, float] | None = None
        self._last_above = -math.inf
        self._last_valid = -math.inf
        self._prev_v = math.nan
        self._prev_ts: pd.Timestamp | None = None
        self._rise_run = 0
        self._rise_base: tuple[pd.Timestamp, float] | None = None
        self._smooth: deque[float] = deque(maxlen=max(1, cfg.smooth_n))
        self._recent: deque[float] = deque(maxlen=max(1, cfg.onset_lookback))
        self._class_letter: str | None = None
        self._pending_escalation: tuple[pd.Timestamp, str] | None = None

    # -- model space -------------------------------------------------------
    def _to_model(self, v: float) -> float:
        return math.log10(v) if self.cfg.model == "log_robust" else v

    def _to_linear(self, x: float) -> float:
        return 10.0 ** x if self.cfg.model == "log_robust" else x

    def _threshold(self, center: float, spread: float) -> float:
        c = self.cfg
        if c.model == "poisson":
            return center + max(c.k_sigma * spread, c.abs_floor)
        s = max(spread, c.sigma_floor)
        if c.model == "log_robust":
            return center + max(c.k_sigma * s, math.log10(c.min_ratio))
        return max(center + c.k_sigma * s, center * c.min_ratio)

    def _classify(self, peak: float) -> str | None:
        if self.cfg.is_goes_flux:
            return goes_class(peak)
        if self.calibration is not None:
            return goes_class(self.calibration(peak))
        return None

    # -- main step ---------------------------------------------------------
    def update(self, t: float, ts: pd.Timestamp, raw: float) -> tuple[dict, list[dict]]:
        cfg = self.cfg
        msgs: list[dict] = []
        valid = raw is not None and np.isfinite(raw) and (cfg.model != "log_robust" or raw > 0)
        if valid:
            self._smooth.append(raw)
            v = sum(self._smooth) / len(self._smooth)
        else:
            v = math.nan

        center, spread, held = self.bg.stats(t)
        warm = np.isfinite(center)
        thr_x = self._threshold(center, spread) if warm else math.nan
        bg_lin = self._to_linear(center) if warm else math.nan
        x = self._to_model(v) if valid else math.nan
        above = bool(valid and warm and x > thr_x)

        if valid:
            self._last_valid = t
            if above:
                self._last_above = t
            rise_fire = self._track_rise(ts, v)
            if self.state == QUIET:
                self._step_quiet(t, ts, v, above, warm, bg_lin, rise_fire, msgs)
            elif self.state == FLARE:
                self._step_flare(t, ts, v, above, msgs, rise_fire)
            else:
                self._step_decay(t, ts, v, above, rise_fire, msgs)
            self._prev_v, self._prev_ts = v, ts
            self._recent.append(v)
        else:
            if self.state == FLARE and t - self._last_above > cfg.max_dropout_s:
                self._close(ts, "data_gap", msgs)
                self.state = QUIET
            elif self.state == DECAY and t - self._last_valid > cfg.max_dropout_s:
                self.state = QUIET
            self._run = 0
            self._cand = None
            self._reset_rise()
            self._prev_v, self._prev_ts = math.nan, None  # never measure a rise across a gap

        # Background admission: only quiet samples (or anything while warming up).
        if valid:
            quiet = (not warm) or (not above and self.state == QUIET)
            self.bg.observe(t, x, quiet)

        self._n += 1
        rec = {
            "value": v,
            "background": bg_lin,
            "threshold": self._to_linear(thr_x) if warm else math.nan,
            "above": above,
            "state": self.state,
            "event_id": self.event.event_id if self.event is not None else None,
            "bg_held": bool(held),
        }
        return rec, msgs

    # -- sustained-rise tracker (NOAA-style start rule) ---------------------
    def _reset_rise(self):
        self._rise_run = 0
        self._rise_base = None

    def _track_rise(self, ts, v) -> bool:
        """Count consecutive increases; True once the rise is long and large
        enough to count as a flare start (``rise_samples`` / ``rise_ratio``)."""
        if np.isfinite(self._prev_v) and v > self._prev_v:
            if self._rise_run == 0:
                self._rise_base = (self._prev_ts, self._prev_v)
            self._rise_run += 1
            base_v = self._rise_base[1]
            return self._rise_run >= self.cfg.rise_samples and v >= base_v * self.cfg.rise_ratio
        self._reset_rise()
        return False

    def _step_quiet(self, t, ts, v, above, warm, bg_lin, rise_fire, msgs):
        if above:
            self._run += 1
            if self._run == 1:
                self._cand = (ts, bg_lin)
        else:
            self._run = 0
            self._cand = None
        level_fire = self._run >= self.cfg.confirm
        if level_fire and self.cfg.onset_min_rise > 1.0 and self._recent:
            level_fire = v >= self.cfg.onset_min_rise * min(self._recent)
        rise_ok = rise_fire and self.cfg.rise_in_quiet and warm and v > bg_lin
        if level_fire or rise_ok:
            # Retrospective start: the beginning of the ongoing rise if there is
            # one, else the first above-threshold sample.
            if self._rise_base is not None and self._rise_base[0] is not None:
                start, base_v = self._rise_base
                if self._cand is not None and self._cand[0] < start:
                    start = self._cand[0]
                bg = min(bg_lin, base_v) if np.isfinite(bg_lin) else base_v
            else:
                start, bg = self._cand if self._cand is not None else (ts, bg_lin)
            self._open(start, ts, bg, "level" if level_fire else "rise", msgs)
            self._accumulate(ts, v)
            self.state = FLARE
            self._run = 0
            self._cand = None
            self._reset_rise()

    def _step_flare(self, t, ts, v, above, msgs, rise_fire=False):
        ev = self.event
        # A sustained re-rise after the peak is a new flare on the decay of
        # this one (NOAA ends the old event when a new one begins).
        if (rise_fire and self.cfg.split_on_rerise and self._rise_base is not None
                and self._rise_base[0] is not None and self._rise_base[0] > ev.peak_time):
            base_ts, base_v = self._rise_base
            self._close(base_ts, "new_flare", msgs)
            self._open(base_ts, ts, base_v, "rise", msgs)
            self._accumulate(ts, v)
            self._reset_rise()
            return
        self._accumulate(ts, v)
        end_level = ev.background + self.cfg.end_fraction * (ev.peak - ev.background)
        if ts > ev.peak_time and v <= end_level:
            self._close(ts, "decay", msgs)
            self.state = DECAY if above else QUIET
            self._reset_rise()
        elif not above and t - self._last_above > self.cfg.max_dropout_s:
            self._close(ts, "below_threshold", msgs)
            self.state = QUIET
            self._reset_rise()

    def _step_decay(self, t, ts, v, above, rise_fire, msgs):
        if not above:
            self.state = QUIET
            return
        if rise_fire:
            base_ts, base_v = self._rise_base
            self._open(base_ts, ts, base_v, "rise", msgs)
            self._accumulate(ts, v)
            self.state = FLARE
            self._reset_rise()

    def _open(self, start, detect, bg, trigger, msgs):
        eid = f"{self.cfg.name}-{len(self.closed) + 1:05d}"
        self.event = FlareEvent(event_id=eid, channel=self.cfg.name, start=start,
                                detect_time=detect, background=bg, trigger=trigger,
                                peak_time=detect)
        self._class_letter = None
        msgs.append({"type": "onset", "time": detect, "channel": self.cfg.name,
                     "event_id": eid, "start": start, "trigger": trigger, "background": bg})

    def _accumulate(self, ts, v):
        ev = self.event
        ev.n_samples += 1
        ev.fluence += max(v - ev.background, 0.0) * self.cadence_s
        if v > ev.peak:
            ev.peak, ev.peak_time = v, ts
            cls = self._classify(v)
            ev.flare_class = cls
            letter = cls[0] if cls else None
            if letter and letter != self._class_letter:
                if self._class_letter is not None:
                    self._pending_escalation = (ts, cls)
                self._class_letter = letter

    def _close(self, ts, reason, msgs):
        ev = self.event
        ev.end = ts
        ev.end_reason = reason
        ev.flare_class = self._classify(ev.peak)
        self.closed.append(ev)
        self.event = None
        msgs.append({"type": "end", "time": ts, "channel": self.cfg.name,
                     "event_id": ev.event_id, "event": ev})

    def finalize(self, ts: pd.Timestamp) -> list[dict]:
        msgs: list[dict] = []
        if self.event is not None:
            self._close(ts, "stream_end", msgs)
        self.state = QUIET
        return msgs

    def pop_escalation(self, ts, msgs):
        """Emit an 'escalation' message when a running flare crosses into a
        higher GOES class letter (e.g. C -> M) - the operationally important
        moment for a nowcast alert."""
        esc = self._pending_escalation
        if esc is not None and self.event is not None:
            msgs.append({"type": "escalation", "time": ts, "channel": self.cfg.name,
                         "event_id": self.event.event_id, "flare_class": esc[1]})
        self._pending_escalation = None


# ---------------------------------------------------------------------------
# Fusion of primary (soft) and secondary (hard) events
# ---------------------------------------------------------------------------

@dataclass
class FusedFlare:
    fused_id: str
    label: str                       # BOTH / PRIMARY_ONLY / SECONDARY_ONLY
    start: pd.Timestamp
    detect_time: pd.Timestamp
    end: pd.Timestamp
    peak_time: pd.Timestamp
    flare_class: str | None
    primary_event: str | None
    secondary_events: list[str] = field(default_factory=list)
    peak_lag_s: float = math.nan     # primary peak - secondary peak (Neupert timing)
    primary_peak: float = math.nan
    secondary_peak: float = math.nan


class EventFuser:
    """Match primary and secondary events in a stream.

    Two events are the same flare if ``[P.start - tol, P.end + tol]`` overlaps
    the secondary event. A fused record is emitted once enough time has
    passed after the primary event closed (``tol``) that no further matching
    secondary event can still start.
    """

    def __init__(self, tolerance_s: float = 300.0):
        self.tol = float(tolerance_s)
        self.pending_primary: list[FlareEvent] = []
        self.secondary: list[FlareEvent] = []
        self.used_secondary: set[str] = set()
        self.fused: list[FusedFlare] = []

    def add(self, ev: FlareEvent, role: str) -> None:
        if role == "primary":
            self.pending_primary.append(ev)
        else:
            self.secondary.append(ev)

    def _matches(self, p: FlareEvent, s: FlareEvent) -> bool:
        lo = p.start - pd.Timedelta(seconds=self.tol)
        hi = p.end + pd.Timedelta(seconds=self.tol)
        s_end = s.end if s.end is not None else s.peak_time
        return s.start <= hi and s_end >= lo

    def tick(self, now: pd.Timestamp, open_secondary: Iterable[FlareEvent] = (),
             open_primary: Iterable[FlareEvent] = (), force: bool = False) -> list[dict]:
        msgs = []
        tol = pd.Timedelta(seconds=self.tol)
        still = []
        for p in self.pending_primary:
            if not force and now < p.end + tol:
                still.append(p)
                continue
            # A long secondary event may corroborate several primary flares
            # (e.g. two soft peaks inside one hard/short-wavelength episode).
            cands = [s for s in list(self.secondary) + list(open_secondary) if self._matches(p, s)]
            for s in cands:
                self.used_secondary.add(s.event_id)
            msgs.append(self._emit(p, cands))
        self.pending_primary = still
        # Secondary events that matched nothing and can no longer match anything
        # are reported on their own; matched ones are kept while they could
        # still corroborate a pending primary.
        # A primary that is still open, or closed but waiting, may yet match.
        live_starts = [p.start for p in self.pending_primary] + [p.start for p in open_primary]
        keep = []
        for s in self.secondary:
            could_match = any(ps - tol <= s.end for ps in live_starts)
            expired = force or (now >= s.end + tol and not could_match)
            if not expired:
                keep.append(s)
            elif s.event_id not in self.used_secondary:
                self.used_secondary.add(s.event_id)
                msgs.append(self._emit(None, [s]))
        self.secondary = keep
        return msgs

    def _emit(self, p: FlareEvent | None, secs: list[FlareEvent]) -> dict:
        n = len(self.fused) + 1
        if p is not None:
            s_best = max(secs, key=lambda s: s.peak) if secs else None
            f = FusedFlare(
                fused_id=f"F{n:05d}", label="BOTH" if secs else "PRIMARY_ONLY",
                start=min([p.start] + [s.start for s in secs]),
                detect_time=min([p.detect_time] + [s.detect_time for s in secs]),
                end=p.end, peak_time=p.peak_time, flare_class=p.flare_class,
                primary_event=p.event_id, secondary_events=[s.event_id for s in secs],
                peak_lag_s=(p.peak_time - s_best.peak_time).total_seconds() if s_best else math.nan,
                primary_peak=p.peak, secondary_peak=s_best.peak if s_best else math.nan)
        else:
            s = secs[0]
            f = FusedFlare(fused_id=f"F{n:05d}", label="SECONDARY_ONLY", start=s.start,
                           detect_time=s.detect_time, end=s.end, peak_time=s.peak_time,
                           flare_class=None, primary_event=None, secondary_events=[s.event_id],
                           secondary_peak=s.peak)
        self.fused.append(f)
        return {"type": "fused", "time": f.end, "fused": f}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NowcastEngine:
    """Multi-channel streaming nowcaster.

    >>> eng = NowcastEngine(goes_channels(), cadence_s=60)
    >>> for sample in GOESLiveSource():
    ...     record, messages = eng.update(sample)
    ...     for m in messages:
    ...         print(m["type"], m["time"], m.get("channel"))
    """

    def __init__(self, channels: list[ChannelConfig], cadence_s: float,
                 fusion_tolerance_s: float = 300.0,
                 calibration: Callable[[float], float] | None = None):
        if not channels:
            raise ValueError("need at least one channel")
        self.channels = channels
        self.cadence_s = float(cadence_s)
        self.detectors = {c.name: ChannelDetector(c, cadence_s, calibration if c.role == "primary" else None)
                          for c in channels}
        prim = [c for c in channels if c.role == "primary"]
        self.primary = prim[0].name if prim else channels[0].name
        self.fuser = EventFuser(fusion_tolerance_s)
        self.last_time: pd.Timestamp | None = None
        self.n_samples = 0

    def update(self, sample: Sample) -> tuple[dict, list[dict]]:
        t = to_epoch_s(sample.time)
        ts = pd.Timestamp(sample.time)
        rec: dict = {"time": ts}
        msgs: list[dict] = []
        for name, det in self.detectors.items():
            r, m = det.update(t, ts, sample.values.get(name, math.nan))
            det.pop_escalation(ts, m)
            for k, v in r.items():
                rec[f"{name}_{k}"] = v
            for mm in m:
                if mm["type"] == "end":
                    self.fuser.add(mm["event"], det.cfg.role)
            msgs.extend(m)
        open_sec = [d.event for d in self.detectors.values()
                    if d.cfg.role != "primary" and d.event is not None]
        open_prim = [d.event for d in self.detectors.values()
                     if d.cfg.role == "primary" and d.event is not None]
        msgs.extend(self.fuser.tick(ts, open_sec, open_prim))
        prim = self.detectors[self.primary]
        rec["status"] = STATE_NAMES[prim.state]
        self.last_time = ts
        self.n_samples += 1
        return rec, msgs

    def finalize(self) -> list[dict]:
        msgs: list[dict] = []
        if self.last_time is None:
            return msgs
        for det in self.detectors.values():
            for m in det.finalize(self.last_time):
                self.fuser.add(m["event"], det.cfg.role)
                msgs.append(m)
        msgs.extend(self.fuser.tick(self.last_time, force=True))
        return msgs

    def run(self, stream: Iterable[Sample]) -> Iterator[tuple[dict, list[dict]]]:
        for s in stream:
            yield self.update(s)

    # -- catalogues -------------------------------------------------------
    def events(self, channel: str | None = None) -> pd.DataFrame:
        evs = []
        for name, det in self.detectors.items():
            if channel is None or name == channel:
                evs.extend(e.as_record() for e in det.closed)
        df = pd.DataFrame(evs)
        return df.sort_values("start").reset_index(drop=True) if not df.empty else df

    def fused(self) -> pd.DataFrame:
        df = pd.DataFrame([asdict(f) for f in self.fuser.fused])
        return df.sort_values("start").reset_index(drop=True) if not df.empty else df


@dataclass
class NowcastResult:
    records: pd.DataFrame
    events: pd.DataFrame
    fused: pd.DataFrame
    messages: list[dict]

    def primary_events(self, channel: str) -> pd.DataFrame:
        e = self.events
        return e[e["channel"] == channel].reset_index(drop=True) if not e.empty else e


def run_batch(df: pd.DataFrame, channels: list[ChannelConfig], cadence_s: float,
              fusion_tolerance_s: float = 300.0, calibration=None,
              keep_records: bool = True) -> NowcastResult:
    """Run the streaming engine over a whole gridded DataFrame.

    Records are written into preallocated numpy columns, so this scales to
    the ~10 M one-second rows of a full Aditya-L1 season.
    """
    eng = NowcastEngine(channels, cadence_s, fusion_tolerance_s, calibration)
    names = [c.name for c in channels]
    n = len(df)
    vals = {c: df[c].to_numpy(dtype=float) if c in df else np.full(n, np.nan) for c in names}
    times = df.index
    cols: dict[str, np.ndarray] = {}
    if keep_records:
        for c in names:
            cols[f"{c}_value"] = np.full(n, np.nan)
            cols[f"{c}_background"] = np.full(n, np.nan)
            cols[f"{c}_threshold"] = np.full(n, np.nan)
            cols[f"{c}_above"] = np.zeros(n, dtype=bool)
            cols[f"{c}_state"] = np.zeros(n, dtype=np.int8)
            cols[f"{c}_event_id"] = np.empty(n, dtype=object)
    messages: list[dict] = []
    for i in range(n):
        rec, msgs = eng.update(Sample(times[i], {c: vals[c][i] for c in names}))
        if msgs:
            messages.extend(msgs)
        if keep_records:
            for c in names:
                cols[f"{c}_value"][i] = rec[f"{c}_value"]
                cols[f"{c}_background"][i] = rec[f"{c}_background"]
                cols[f"{c}_threshold"][i] = rec[f"{c}_threshold"]
                cols[f"{c}_above"][i] = rec[f"{c}_above"]
                cols[f"{c}_state"][i] = rec[f"{c}_state"]
                cols[f"{c}_event_id"][i] = rec[f"{c}_event_id"]
    messages.extend(eng.finalize())
    records = pd.DataFrame(cols, index=times) if keep_records else pd.DataFrame(index=times)
    return NowcastResult(records=records, events=eng.events(), fused=eng.fused(), messages=messages)


# ---------------------------------------------------------------------------
# Counts -> flux calibration (for SoLEXS classes)
# ---------------------------------------------------------------------------

def fit_counts_to_flux(peak_counts: np.ndarray, peak_flux: np.ndarray) -> Callable[[float], float]:
    """Fit log10(flux) = a + b*log10(counts) on flares seen by both SoLEXS
    and GOES. Returns a callable usable as ``calibration=`` in the engine, so
    SoLEXS events get real GOES classes instead of a counts-only ranking.

    Until the SoLEXS RMF is available this cross-calibration against GOES is
    the most defensible way to assign A/B/C/M/X to Aditya-L1 flares.
    """
    c = np.asarray(peak_counts, float)
    f = np.asarray(peak_flux, float)
    ok = (c > 0) & (f > 0) & np.isfinite(c) & np.isfinite(f)
    if ok.sum() < 3:
        raise ValueError("need at least 3 matched flares to calibrate")
    b, a = np.polyfit(np.log10(c[ok]), np.log10(f[ok]), 1)

    def calib(counts: float) -> float:
        return 10 ** (a + b * math.log10(counts)) if counts and counts > 0 else math.nan

    calib.coefficients = (a, b)  # type: ignore[attr-defined]
    return calib
