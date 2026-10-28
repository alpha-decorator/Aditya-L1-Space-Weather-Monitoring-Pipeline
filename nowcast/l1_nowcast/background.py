"""
Trailing (causal) rolling statistics that can be updated one sample at a time.

Everything here is incremental: a sample is added, old samples expire as
time advances, and the statistics of the *current* window are available
at any moment in O(1) (quantiles) or O(1) amortised (mean).

The same objects are used for live streaming and for batch processing
(batch = feeding every row of a DataFrame through the same code), so the
two modes can never disagree.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque


class RollingWindow:
    """Values observed in the trailing time window (t - window_s, t].

    Keeps a time-ordered deque (for expiry) and a sorted list (for exact
    quantiles). Insert/remove into the sorted list is a memmove of at most
    `n` floats, which is fast for the window sizes used here (a few
    thousand samples).
    """

    __slots__ = ("window_s", "_times", "_vals", "_sorted", "_sum")

    def __init__(self, window_s: float):
        self.window_s = float(window_s)
        self._times: deque[float] = deque()
        self._vals: deque[float] = deque()
        self._sorted: list[float] = []
        self._sum = 0.0

    def __len__(self) -> int:
        return len(self._sorted)

    def expire(self, t: float) -> None:
        """Drop samples that are no longer inside (t - window_s, t]."""
        cutoff = t - self.window_s
        times, vals, srt = self._times, self._vals, self._sorted
        while times and times[0] <= cutoff:
            times.popleft()
            v = vals.popleft()
            del srt[bisect_left(srt, v)]
            self._sum -= v
        if not srt:
            self._sum = 0.0  # kill accumulated float drift whenever empty

    def add(self, t: float, v: float) -> None:
        self._times.append(t)
        self._vals.append(v)
        insort(self._sorted, v)
        self._sum += v

    def clear(self) -> None:
        self._times.clear()
        self._vals.clear()
        self._sorted.clear()
        self._sum = 0.0

    def items(self):
        return zip(self._times, self._vals)

    def last_time(self) -> float | None:
        return self._times[-1] if self._times else None

    def quantile(self, q: float) -> float:
        """Linear-interpolated quantile (same definition as numpy's default)."""
        s = self._sorted
        n = len(s)
        if n == 0:
            return math.nan
        pos = q * (n - 1)
        lo = int(pos)
        frac = pos - lo
        if lo + 1 >= n:
            return s[-1]
        return s[lo] + (s[lo + 1] - s[lo]) * frac

    def median(self) -> float:
        return self.quantile(0.5)

    def iqr_sigma(self) -> float:
        """Robust standard-deviation estimate: IQR / 1.349.

        Used instead of MAD because it is an exact function of the sorted
        window (no second pass over |x - median|), so it is cheap to keep
        up to date in a stream and immune to the outliers we are hunting.
        """
        return (self.quantile(0.75) - self.quantile(0.25)) / 1.349

    def mean(self) -> float:
        n = len(self._sorted)
        return self._sum / n if n else math.nan


class AdaptiveBackground:
    """Flare-resistant trailing background for one channel.

    Two windows are maintained:

    * ``accepted`` - only samples that were judged *quiet* when they
      arrived (below threshold, not inside a flare). This is the causal
      equivalent of the team's "two-pass" background: a long flare can
      never pull its own background up, because flare samples are never
      admitted.
    * ``raw`` - every valid sample. Used only to re-seed ``accepted`` if
      it goes stale (e.g. the Sun's baseline steps up permanently and every
      new sample looks like a flare). Without this, the detector could
      lock itself in a never-ending event.

    Statistics are computed in "model space" (log10 for flux-like
    channels, linear for counts) by the caller.
    """

    def __init__(self, window_s: float, min_count: int, stale_s: float, model: str):
        if model not in ("log_robust", "robust", "poisson"):
            raise ValueError(f"unknown background model {model!r}")
        self.model = model
        self.window_s = float(window_s)
        self.min_count = int(min_count)
        self.stale_s = float(stale_s)
        self.accepted = RollingWindow(window_s)
        self.raw = RollingWindow(window_s)
        self._held: tuple[float, float] | None = None  # (center, spread)
        self._held_time: float | None = None
        self.reseeds = 0

    def stats(self, t: float) -> tuple[float, float, bool]:
        """Return (center, spread, held) for the window ending just before t.

        ``held`` is True when the live window is too thin and the last good
        estimate is being carried forward (e.g. during a long flare).
        center/spread are NaN while warming up.
        """
        self.accepted.expire(t)
        self.raw.expire(t)

        if len(self.accepted) >= self.min_count:
            center, spread = self._compute(self.accepted)
            self._held = (center, spread)
            self._held_time = t
            return center, spread, False

        if self._held is not None and self._held_time is not None:
            if t - self._held_time <= self.stale_s:
                return self._held[0], self._held[1], True
            # Stale: the quiet baseline has not been seen for too long.
            # Re-seed from everything recent so detection can resume.
            if len(self.raw) >= self.min_count:
                self.accepted.clear()
                for ti, vi in self.raw.items():
                    self.accepted.add(ti, vi)
                self.reseeds += 1
                center, spread = self._compute(self.accepted)
                self._held = (center, spread)
                self._held_time = t
                return center, spread, False

        return math.nan, math.nan, False

    def _compute(self, w: RollingWindow) -> tuple[float, float]:
        if self.model == "poisson":
            mu = w.mean()
            return mu, math.sqrt(max(mu, 0.0))
        return w.median(), w.iqr_sigma()

    def observe(self, t: float, x: float, quiet: bool) -> None:
        """Record a sample after the detection decision was made."""
        self.raw.add(t, x)
        if quiet:
            self.accepted.add(t, x)
