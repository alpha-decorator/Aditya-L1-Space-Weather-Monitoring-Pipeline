"""
Data store for continual learning.

    store/
    ├── history/xrs_YYYY-MM.parquet   permanent 1-minute XRS-A/XRS-B history, one file per month
    └── live/YYYY-MM-DD.parquet       raw copy of each recent day from the real-time feed (rolling, 10 days)

History is what the models train on. It is never truncated: flares of M class and above are rare,
so a model that only saw the last few days would forget what a big flare looks like.
Where a minute exists from both sources, the science-quality NCEI archive wins over the
real-time SWPC feed; the feed only fills minutes the archive does not have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from . import ROOT

STORE = ROOT / "continual" / "store"
CHANNELS = ["xrsa", "xrsb"]


class DataStore:
    def __init__(self, root: str | Path = STORE, live_days: int = 10):
        self.root = Path(root)
        self.hist_dir = self.root / "history"
        self.live_dir = self.root / "live"
        self.live_days = live_days

    # -- reading ----------------------------------------------------------
    def history(self, start=None, end=None) -> pd.DataFrame:
        """All stored minutes as one regular 1-minute frame (NaN rows for gaps)."""
        files = sorted(self.hist_dir.glob("xrs_*.parquet"))
        if not files:
            return pd.DataFrame(columns=CHANNELS, index=pd.DatetimeIndex([], name="time"))
        df = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
        df = df[~df.index.duplicated(keep="first")].astype("float64")
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1min"))
        df.index.name = "time"
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    def data_end(self) -> pd.Timestamp | None:
        """Last minute of the last complete UTC day in history."""
        v = self.history().dropna(how="all")
        if v.empty:
            return None
        last = v.index.max()
        day_end = last.normalize() + pd.Timedelta("1D") - pd.Timedelta("1min")
        return day_end if last == day_end else day_end - pd.Timedelta("1D")

    # -- writing ----------------------------------------------------------
    def merge(self, df: pd.DataFrame, prefer_existing: bool = True) -> int:
        """Merge minutes into history. Returns how many previously missing minutes were filled."""
        df = df[CHANNELS].astype("float32")
        df = df[df.notna().any(axis=1)]
        if df.empty:
            return 0
        self.hist_dir.mkdir(parents=True, exist_ok=True)
        filled = 0
        for month, part in df.groupby(df.index.to_period("M")):
            path = self.hist_dir / f"xrs_{month}.parquet"
            if path.exists():
                old = pd.read_parquet(path)
                before = int(old.notna().any(axis=1).sum())
                new = old.combine_first(part) if prefer_existing else part.combine_first(old)
            else:
                before, new = 0, part
            new = new.sort_index().astype("float32")
            new.index.name = "time"
            filled += int(new.notna().any(axis=1).sum()) - before
            new.to_parquet(path)
        return filled

    def seed_from_archive(self, avg1m_dir: str | Path) -> int:
        """One-off import of the NCEI science archive (downloaded by nowcast.ipynb)."""
        from l1_nowcast import load_goes_avg1m
        return self.merge(load_goes_avg1m(avg1m_dir), prefer_existing=False)

    def ingest_live(self, now: pd.Timestamp | None = None,
                    fetch: Callable[[str], pd.DataFrame] | None = None) -> dict:
        """Store every complete UTC day in the real-time 7-day feed, then prune the live cache.

        Running this at least once a week loses nothing; running it daily keeps history current.
        """
        if fetch is None:
            from l1_nowcast import GOESLiveSource
            fetch = GOESLiveSource().fetch
        now = pd.Timestamp.now(tz="UTC").tz_localize(None) if now is None else pd.Timestamp(now)
        today = now.normalize()
        raw = fetch("7-day")
        raw = raw[raw.index < today]
        self.live_dir.mkdir(parents=True, exist_ok=True)
        days = []
        for day, part in raw.groupby(raw.index.normalize()):
            part = part.reindex(pd.date_range(day, day + pd.Timedelta("1D") - pd.Timedelta("1min"), freq="1min"))
            part.index.name = "time"
            part[CHANNELS].astype("float32").to_parquet(self.live_dir / f"{day:%Y-%m-%d}.parquet")
            days.append(f"{day:%Y-%m-%d}")
        filled = self.merge(raw, prefer_existing=True)
        pruned = self.prune_live(today)
        return {"days_in_feed": days, "minutes_added": filled, "live_files_pruned": pruned}

    def prune_live(self, today: pd.Timestamp) -> list[str]:
        cutoff = pd.Timestamp(today).normalize() - pd.Timedelta(days=self.live_days)
        gone = []
        for f in sorted(self.live_dir.glob("*.parquet")):
            if pd.Timestamp(f.stem) < cutoff:
                f.unlink()
                gone.append(f.stem)
        return gone

    def status(self) -> dict:
        h = self.history()
        v = h.dropna(how="all")
        return {
            "history_start": None if v.empty else str(v.index.min()),
            "history_end": None if v.empty else str(v.index.max()),
            "history_minutes": int(len(v)),
            "coverage": None if h.empty else round(len(v) / len(h), 4),
            "live_days": sorted(f.stem for f in self.live_dir.glob("*.parquet")) if self.live_dir.exists() else [],
        }
