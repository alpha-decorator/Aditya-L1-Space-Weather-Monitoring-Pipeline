"""
Data sources: everything that turns an instrument feed into a stream of
regularly-gridded samples ``(time, {channel: value})``.

Supported inputs
----------------
Aditya-L1 (ISRO)
    * SoLEXS / HEL1OS CSV exports in the format used by Team Phoenix
      (``combined_solexs_dataset.csv``, ``combined_hel1os_dataset_full.csv``).
    * OGIP FITS light curves (the format ISSDC/PRADAN Level-1 light-curve
      products use: a BINTABLE with TIME + COUNTS/RATE columns and
      MJDREF/TIMEZERO header keywords).
    * A watched directory: new files dropped into a folder (e.g. by a
      PRADAN sync job or a ground-station pipeline) are picked up and
      streamed as they arrive.

    ISRO does not publish a real-time Aditya-L1 API; data reaches users
    as files. The directory watcher is how a continuous Aditya-L1 feed is
    consumed in practice.

GOES-R XRS (NOAA) - real-time and archive soft X-ray flux
    * ``GOESLiveSource`` polls the public SWPC JSON feed (1-minute cadence,
      updated every minute). This is the only openly accessible *live*
      solar X-ray stream and is physically the same measurement class as
      SoLEXS (thermal soft X-rays).
    * ``download_goes_archive`` / ``load_goes_avg1m`` read the NCEI
      science-quality 1-minute archive, and ``load_goes_flare_summary``
      reads NOAA's official flare list (used as ground truth).

All timestamps are timezone-naive UTC ``pandas.Timestamp`` objects.
"""

from __future__ import annotations

import datetime as dt
import glob
import heapq
import json
import math
import os
import time as _time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sample:
    time: pd.Timestamp
    values: dict


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def to_epoch_s(t) -> float:
    """Timestamp -> float seconds since 1970 (naive UTC)."""
    return pd.Timestamp(t).value / 1e9


def regularize(df: pd.DataFrame, cadence: str, agg: dict[str, str] | None = None) -> pd.DataFrame:
    """Put a time-indexed frame on a complete regular grid.

    Rows are binned into ``cadence`` bins (left-labelled) and aggregated per
    column with ``agg`` (default ``mean``). Bins with no data become NaN rows,
    so gaps are explicit instead of silently collapsing - the failure mode
    the Phase-1 audit found in the raw SoLEXS export.
    """
    if df.empty:
        return df
    agg = agg or {}
    rules = {c: agg.get(c, "mean") for c in df.columns}
    # min_count=1 so that an empty bin sums to NaN, not a fake 0 count.
    out = {}
    binned = df.resample(cadence, label="left", closed="left")
    for col, rule in rules.items():
        if rule == "sum":
            out[col] = binned[col].sum(min_count=1)
        else:
            out[col] = getattr(binned[col], rule)()
    out = pd.DataFrame(out)
    full = pd.date_range(out.index.min(), out.index.max(), freq=cadence)
    return out.reindex(full)


def frame_to_samples(df: pd.DataFrame, columns: Iterable[str] | None = None) -> Iterator[Sample]:
    cols = list(columns) if columns is not None else list(df.columns)
    arr = df[cols].to_numpy(dtype=float)
    for t, row in zip(df.index, arr):
        yield Sample(pd.Timestamp(t), {c: float(v) for c, v in zip(cols, row)})


class FrameSource:
    """Stream the rows of a DataFrame.

    ``speed=None`` streams as fast as possible (batch / back-test).
    ``speed=60`` replays one minute of data per real second, etc. -
    useful for demos of the live engines on archived events.
    """

    def __init__(self, df: pd.DataFrame, columns: Iterable[str] | None = None,
                 speed: float | None = None):
        self.df = df
        self.columns = list(columns) if columns is not None else list(df.columns)
        self.speed = speed

    def __iter__(self) -> Iterator[Sample]:
        wall0 = data0 = None
        for s in frame_to_samples(self.df, self.columns):
            if self.speed:
                if wall0 is None:
                    wall0, data0 = _time.monotonic(), s.time
                due = wall0 + (s.time - data0).total_seconds() / self.speed
                delay = due - _time.monotonic()
                if delay > 0:
                    _time.sleep(delay)
            yield s


def merge_sources(*sources: Iterable[Sample]) -> Iterator[Sample]:
    """Time-ordered merge of several sample streams (lazy, so it also works
    for live sources)."""
    return heapq.merge(*sources, key=lambda s: s.time)


class StreamAligner:
    """Bin a raw (irregular, multi-instrument) sample stream onto a grid.

    Samples from several instruments arrive interleaved and at different
    cadences (HEL1OS is ~1.6 Hz, SoLEXS 1 Hz). Each channel is aggregated
    into ``cadence``-second bins with its own rule (``sum`` for photon
    counts, ``mean`` for ratios), and a bin is emitted once a sample at
    least ``lateness_s`` past the end of the bin has been seen. Empty bins
    are emitted as all-NaN rows so downstream code always sees explicit
    gaps.
    """

    def __init__(self, cadence_s: float, rules: dict[str, str], lateness_s: float = 0.0,
                 max_gap_fill_s: float = 6 * 3600):
        self.cadence = float(cadence_s)
        self.rules = dict(rules)
        self.lateness = float(lateness_s)
        self.max_gap_fill = float(max_gap_fill_s)
        self._bins: dict[float, dict[str, list[float]]] = {}
        self._next_emit: float | None = None

    def _bin_of(self, t: float) -> float:
        return math.floor(t / self.cadence) * self.cadence

    def push(self, sample: Sample) -> list[Sample]:
        t = to_epoch_s(sample.time)
        b = self._bin_of(t)
        if self._next_emit is None:
            self._next_emit = b
        if b < self._next_emit:
            return []  # too late: that bin was already emitted
        slot = self._bins.setdefault(b, {})
        for ch, v in sample.values.items():
            if ch in self.rules and v is not None and np.isfinite(v):
                slot.setdefault(ch, []).append(float(v))
        return self._drain(watermark=t - self.lateness)

    def flush(self) -> list[Sample]:
        return self._drain(watermark=math.inf)

    def _drain(self, watermark: float) -> list[Sample]:
        out = []
        while self._next_emit is not None and self._next_emit + self.cadence <= watermark:
            if watermark == math.inf and not self._bins:
                break  # flushing and nothing left
            b = self._next_emit
            slot = self._bins.pop(b, None)
            vals = {}
            for ch, rule in self.rules.items():
                xs = (slot or {}).get(ch)
                if not xs:
                    vals[ch] = math.nan
                elif rule == "sum":
                    vals[ch] = float(sum(xs))
                elif rule == "max":
                    vals[ch] = float(max(xs))
                else:
                    vals[ch] = float(sum(xs) / len(xs))
            out.append(Sample(pd.Timestamp(b * 1e9), vals))
            self._next_emit = b + self.cadence
            # Jump over very long gaps instead of emitting millions of NaN rows.
            if self._bins:
                nxt = min(self._bins)
                if nxt - self._next_emit > self.max_gap_fill:
                    self._next_emit = nxt
        return out

    def run(self, stream: Iterable[Sample]) -> Iterator[Sample]:
        for s in stream:
            yield from self.push(s)
        yield from self.flush()


# ---------------------------------------------------------------------------
# Aditya-L1: SoLEXS + HEL1OS (CSV exports, Team Phoenix format)
# ---------------------------------------------------------------------------

def read_solexs_csv(path: str | os.PathLike) -> pd.DataFrame:
    """SoLEXS export with columns ``time, counts, hardness`` (1 Hz).

    Returns a 1-second regular grid with columns ``solexs_counts`` and
    ``solexs_hardness``; GTI gaps and missing chunks become NaN rows.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df.rename(columns={"counts": "solexs_counts", "hardness": "solexs_hardness"})
    df = df[[c for c in ("solexs_counts", "solexs_hardness") if c in df.columns]]
    return regularize(df, "1s", {"solexs_counts": "sum", "solexs_hardness": "mean"})


HEL1OS_RULES = {
    "hel1os_cdte": "sum", "hel1os_czt": "sum",
    "hel1os_cdte_hardness": "mean", "hel1os_czt_hardness": "mean",
}


def read_hel1os_csv(path: str | os.PathLike) -> pd.DataFrame:
    """HEL1OS export with ``time, cdte_broad, cdte_hardness, czt_broad,
    czt_hardness`` at the native ~1.6 Hz cadence.

    Exact duplicate rows are dropped, then readings are binned to 1 Hz:
    broadband counts are SUMMED (photons are additive), hardness ratios
    are AVERAGED. Seconds with no reading become NaN, never 0.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    df = df.rename(columns={
        "cdte_broad": "hel1os_cdte", "czt_broad": "hel1os_czt",
        "cdte_hardness": "hel1os_cdte_hardness", "czt_hardness": "hel1os_czt_hardness",
    })
    df = df[[c for c in HEL1OS_RULES if c in df.columns]]
    return regularize(df, "1s", HEL1OS_RULES)


def align_frames(*frames: pd.DataFrame, cadence: str = "1s") -> pd.DataFrame:
    """Outer-join several gridded instrument frames onto one master grid."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    start = min(f.index.min() for f in frames)
    end = max(f.index.max() for f in frames)
    grid = pd.date_range(start, end, freq=cadence)
    return pd.concat([f.reindex(grid) for f in frames], axis=1)


# ---------------------------------------------------------------------------
# Aditya-L1: OGIP FITS light curves (ISSDC / PRADAN Level-1 products)
# ---------------------------------------------------------------------------

_TIME_COLS = ("TIME", "time", "Time")
_VALUE_COLS = ("COUNTS", "RATE", "counts", "rate", "COUNT_RATE")


def read_ogip_lightcurve(path: str | os.PathLike, channel: str,
                         value_col: str | None = None, hdu: int | str = 1,
                         per_second: bool = False) -> pd.DataFrame:
    """Read an OGIP-style FITS light curve into a time-indexed frame.

    Time conversion follows the OGIP convention:
        t_utc = MJDREF (MJDREFI + MJDREFF) + (TIMEZERO + TIME) / 86400
    If the header carries no MJDREF, TIME is assumed to be Unix seconds.

    ``per_second=True`` converts a COUNTS column to a rate using TIMEDEL.
    """
    from astropy.io import fits  # imported lazily: only needed for FITS input

    with fits.open(path, memmap=False) as hd:
        h = hd[hdu].header
        data = hd[hdu].data
        names = list(data.columns.names)
        tcol = next((c for c in _TIME_COLS if c in names), None)
        if tcol is None:
            raise KeyError(f"{path}: no TIME column in {names}")
        vcol = value_col or next((c for c in _VALUE_COLS if c in names), None)
        if vcol is None:
            raise KeyError(f"{path}: no COUNTS/RATE column in {names}; pass value_col=")
        tt = np.asarray(data[tcol], dtype=float)
        vv = np.asarray(data[vcol], dtype=float)
        if vv.ndim > 1:  # spectral light curve: sum over energy channels
            vv = vv.sum(axis=tuple(range(1, vv.ndim)))
        tz = float(h.get("TIMEZERO", 0.0))
        if "MJDREFI" in h or "MJDREF" in h:
            mjdref = float(h.get("MJDREFI", h.get("MJDREF", 0.0))) + float(h.get("MJDREFF", 0.0))
            mjd = mjdref + (tt + tz) / 86400.0
            times = pd.to_datetime((mjd - 40587.0) * 86400.0, unit="s")
        else:
            times = pd.to_datetime(tt + tz, unit="s")
        if per_second:
            vv = vv / float(h.get("TIMEDEL", 1.0))
    df = pd.DataFrame({channel: vv}, index=pd.DatetimeIndex(times, name="time"))
    return df[~df.index.duplicated()].sort_index()


class DirectoryWatchSource:
    """Continuous stream from files arriving in a directory.

    Each new file matching ``pattern`` is parsed with ``reader(path) ->
    DataFrame`` and its rows are streamed in time order; rows not newer than
    what was already emitted are skipped, so overlapping files are safe.
    With ``poll_s`` set the source keeps watching for new files (live
    mode); ``idle_timeout_s`` ends the stream after a quiet period.
    """

    def __init__(self, directory: str | os.PathLike, pattern: str,
                 reader: Callable[[str], pd.DataFrame], poll_s: float | None = None,
                 idle_timeout_s: float | None = None):
        self.directory = Path(directory)
        self.pattern = pattern
        self.reader = reader
        self.poll_s = poll_s
        self.idle_timeout_s = idle_timeout_s
        self._seen: set[str] = set()
        self._last: pd.Timestamp | None = None

    def _new_files(self) -> list[str]:
        files = sorted(glob.glob(str(self.directory / self.pattern)))
        new = [f for f in files if f not in self._seen]
        self._seen.update(new)
        return new

    def __iter__(self) -> Iterator[Sample]:
        idle_since = _time.monotonic()
        while True:
            new = self._new_files()
            for f in new:
                df = self.reader(f)
                if self._last is not None:
                    df = df[df.index > self._last]
                for s in frame_to_samples(df):
                    self._last = s.time
                    yield s
            if new:
                idle_since = _time.monotonic()
            if self.poll_s is None:
                return
            if self.idle_timeout_s is not None and _time.monotonic() - idle_since > self.idle_timeout_s:
                return
            _time.sleep(self.poll_s)


# ---------------------------------------------------------------------------
# GOES-R XRS (NOAA): live feed, science archive, official flare list
# ---------------------------------------------------------------------------

SWPC_BASE = "https://services.swpc.noaa.gov/json/goes/primary"
NCEI_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes"
_ENERGY_TO_CHANNEL = {"0.1-0.8nm": "xrsb", "0.05-0.4nm": "xrsa"}


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "aditya-l1-nowcast/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_swpc_xrays(payload: bytes | str) -> pd.DataFrame:
    """SWPC ``xrays-*.json`` -> frame with ``xrsa``, ``xrsb`` (W/m^2)."""
    rows = json.loads(payload)
    recs: dict[pd.Timestamp, dict[str, float]] = {}
    for r in rows:
        ch = _ENERGY_TO_CHANNEL.get(r.get("energy"))
        if ch is None:
            continue
        t = pd.Timestamp(r["time_tag"])
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        f = r.get("flux")
        recs.setdefault(t, {})[ch] = float(f) if f is not None and f > 0 else math.nan
    df = pd.DataFrame.from_dict(recs, orient="index").sort_index()
    df.index.name = "time"
    for ch in ("xrsa", "xrsb"):
        if ch not in df:
            df[ch] = math.nan
    return df[["xrsa", "xrsb"]]


class GOESLiveSource:
    """Real-time GOES XRS stream from NOAA SWPC (public, no key).

    On start it back-fills ``backfill`` worth of history (enough to warm the
    2-hour background), then polls every ``poll_s`` seconds and yields only
    minutes it has not emitted yet. Network errors are retried on the next
    poll instead of ending the stream.
    """

    def __init__(self, poll_s: float = 60.0, backfill: str = "6-hour",
                 max_polls: int | None = None, duration_s: float | None = None,
                 on_error: Callable[[Exception], None] | None = None):
        if backfill not in ("6-hour", "1-day", "3-day", "7-day"):
            raise ValueError("backfill must be one of 6-hour, 1-day, 3-day, 7-day")
        self.poll_s = poll_s
        self.backfill = backfill
        self.max_polls = max_polls
        self.duration_s = duration_s
        self.on_error = on_error
        self.last_time: pd.Timestamp | None = None

    def fetch(self, span: str) -> pd.DataFrame:
        return parse_swpc_xrays(_http_get(f"{SWPC_BASE}/xrays-{span}.json"))

    def __iter__(self) -> Iterator[Sample]:
        t0 = _time.monotonic()
        polls = 0
        span = self.backfill
        while True:
            try:
                df = self.fetch(span)
                if self.last_time is not None:
                    df = df[df.index > self.last_time]
                if not df.empty:
                    # fill any minute the feed skipped with an explicit NaN row
                    start = df.index.min() if self.last_time is None else self.last_time + pd.Timedelta("1min")
                    df = df.reindex(pd.date_range(start, df.index.max(), freq="1min"))
                    for s in frame_to_samples(df):
                        self.last_time = s.time
                        yield s
                span = "6-hour"
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
                if self.on_error:
                    self.on_error(e)
            polls += 1
            if self.max_polls is not None and polls >= self.max_polls:
                return
            if self.duration_s is not None and _time.monotonic() - t0 >= self.duration_s:
                return
            _time.sleep(self.poll_s)


def goes_archive_urls(start: dt.date, end: dt.date, satellite: int = 19,
                      products: Iterable[str] = ("avg1m", "flsum"),
                      version: str = "v2-2-2") -> list[tuple[str, str, str]]:
    """(product, url, filename) for every day in [start, end]."""
    out = []
    d = start
    while d <= end:
        for p in products:
            fn = f"sci_xrsf-l2-{p}_g{satellite}_d{d:%Y%m%d}_{version}.nc"
            url = f"{NCEI_BASE}/goes{satellite}/l2/data/xrsf-l2-{p}_science/{d:%Y/%m}/{fn}"
            out.append((p, url, fn))
        d += dt.timedelta(days=1)
    return out


def download_goes_archive(dest: str | os.PathLike, start: dt.date, end: dt.date,
                          satellite: int = 19, products: Iterable[str] = ("avg1m", "flsum"),
                          workers: int = 8, version: str = "v2-2-2") -> dict[str, int]:
    """Download NCEI GOES XRS science files into ``dest/<product>/``.

    Files already present are skipped, so this is safe to re-run. Some days
    legitimately have no flare-summary file; those are counted as missing.
    """
    dest = Path(dest)
    jobs = goes_archive_urls(start, end, satellite, products, version)
    stats = {"downloaded": 0, "cached": 0, "missing": 0}

    def fetch(job):
        p, url, fn = job
        target = dest / p / fn
        if target.exists() and target.stat().st_size > 0:
            return "cached"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = _http_get(url, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "missing"
            raise
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        return "downloaded"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(fetch, jobs):
            stats[r] += 1
    return stats


_GOES_EPOCH = pd.Timestamp("2000-01-01 12:00:00")


def _h5_times(h) -> pd.DatetimeIndex:
    secs = h["time"][:].astype(float)
    secs[secs == -9999.0] = np.nan
    return pd.DatetimeIndex(_GOES_EPOCH + pd.to_timedelta(secs, unit="s"), name="time")


def _h5_strings(ds) -> list[str | None]:
    out = []
    for s in ds[:]:
        if isinstance(s, bytes):
            s = s.decode()
        out.append(s if isinstance(s, str) and s.strip() else None)
    return out


def _read_avg1m_file(path: str) -> pd.DataFrame:
    """One NCEI avg1m file, read straight from HDF5 (~100x faster than xarray)."""
    import h5py
    with h5py.File(path, "r") as h:
        idx = _h5_times(h)
        cols = {}
        for ch in ("xrsa", "xrsb"):
            v = h[f"{ch}_flux"][:].astype(float)
            v[v == -9999.0] = np.nan
            cols[ch] = v
            cols[f"{ch}_flag"] = h[f"{ch}_flag"][:].astype(int)  # fill 255 -> bad bits set
    return pd.DataFrame(cols, index=idx)


def _day_of(path) -> pd.Timestamp:
    return pd.Timestamp(Path(path).name.split("_d")[1][:8])


def load_goes_avg1m(directory: str | os.PathLike, start=None, end=None,
                    cache: bool = True) -> pd.DataFrame:
    """Load NCEI 1-minute XRS files into one regular 1-minute frame.

    Quality control:
      * flag bit 1 (eclipse) or bit 2 (bad data) -> NaN
      * non-positive / fill flux -> NaN
    The result has columns ``xrsa`` (0.05-0.4 nm) and ``xrsb`` (0.1-0.8 nm)
    in W/m^2 and an explicit NaN row for every missing minute.

    With ``cache=True`` the merged frame is stored as parquet next to the
    files and reused while the set of files is unchanged.
    """
    directory = Path(directory)
    files = sorted(glob.glob(str(directory / "*avg1m*.nc")))
    if not files:
        raise FileNotFoundError(f"no avg1m files in {directory}")
    cache_path = directory / "_avg1m_cache.parquet"
    key_path = directory / "_avg1m_cache.key"
    key = f"{len(files)}|{Path(files[0]).name}|{Path(files[-1]).name}"
    df = None
    if cache and cache_path.exists() and key_path.exists() and key_path.read_text() == key:
        df = pd.read_parquet(cache_path)
    if df is None:
        df = pd.concat([_read_avg1m_file(f) for f in files]).sort_index()
        df = df[df.index.notna()]
        for ch in ("xrsa", "xrsb"):
            bad = (df[f"{ch}_flag"].to_numpy() & 0b11) != 0
            v = df[ch].to_numpy()
            df[ch] = np.where(bad | ~(v > 0), np.nan, v)
        df = df[["xrsa", "xrsb"]]
        df.index = df.index.floor("1min")
        df = df[~df.index.duplicated(keep="first")]
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1min"))
        df.index.name = "time"
        if cache:
            df.to_parquet(cache_path)
            key_path.write_text(key)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def load_goes_flare_summary(directory: str | os.PathLike) -> pd.DataFrame:
    """NOAA's official GOES flare list (XRS L2 ``flsum``) as one row per flare.

    Columns: ``start, peak, end, flare_class, peak_flux, background_flux``.
    Flares that span midnight are stitched across daily files by walking the
    EVENT_START / EVENT_PEAK / EVENT_END records in time order.
    """
    import h5py
    files = sorted(glob.glob(str(Path(directory) / "*flsum*.nc")))
    if not files:
        raise FileNotFoundError(f"no flsum files in {directory}")
    recs = []
    for f in files:
        with h5py.File(f, "r") as h:
            if h["time"].shape[0] == 0:
                continue
            bkg = h["background_flux"][:].astype(float)
            flux = h["xrsb_flux"][:].astype(float)
            bkg[bkg == -9999.0] = np.nan
            flux[flux == -9999.0] = np.nan
            recs.append(pd.DataFrame({"time": _h5_times(h), "status": _h5_strings(h["status"]),
                                      "flux": flux, "bkg": bkg, "cls": _h5_strings(h["flare_class"])}))
    d = pd.concat(recs).sort_values("time", kind="stable").drop_duplicates(["time", "status"])
    flares, cur = [], None
    for r in d.itertuples(index=False):
        if r.status == "EVENT_START":
            cur = {"start": r.time, "background_flux": r.bkg}
        elif r.status == "EVENT_PEAK" and cur is not None:
            cur.update(peak=r.time, peak_flux=r.flux, flare_class=r.cls)
        elif r.status == "EVENT_END" and cur is not None:
            if "peak" in cur:
                cur["end"] = r.time
                flares.append(cur)
            cur = None
    out = pd.DataFrame(flares, columns=["start", "peak", "end", "flare_class", "peak_flux", "background_flux"])
    return out.reset_index(drop=True)


def missing_reference_days(directory: str | os.PathLike, start, end) -> pd.DatetimeIndex:
    """Days in [start, end] with no flare-summary file (exclude them from scoring)."""
    have = set(available_days(directory, "flsum"))
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return pd.DatetimeIndex([d for d in days if d not in have])


def available_days(directory: str | os.PathLike, product: str) -> pd.DatetimeIndex:
    files = glob.glob(str(Path(directory) / f"*{product}*.nc"))
    return pd.DatetimeIndex(sorted(pd.Timestamp(Path(f).name.split("_d")[1][:8]) for f in files))
