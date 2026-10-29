import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continual import DataStore, Registry  # noqa: E402
from continual.retrain import MIN_UNSEEN_DAYS, decide  # noqa: E402


def frame(start, minutes, value):
    idx = pd.date_range(start, periods=minutes, freq="1min", name="time")
    return pd.DataFrame({"xrsa": value / 10, "xrsb": value}, index=idx)


def fake_feed(df):
    return lambda span: df


def test_merge_prefers_existing_and_fills_gaps(tmp_path):
    s = DataStore(tmp_path)
    archive = frame("2026-10-01", 1440, 1e-6)
    archive.iloc[100:110] = np.nan                       # a gap in the science archive
    s.merge(archive, prefer_existing=False)
    filled = s.merge(frame("2026-10-01", 1440, 5e-6))    # real-time feed for the same day
    h = s.history()
    assert filled == 10
    assert h["xrsb"].iloc[0] == pytest.approx(1e-6)     # archive kept
    assert h["xrsb"].iloc[105] == pytest.approx(5e-6)   # gap filled from the feed


def test_ingest_stores_complete_days_only_and_prunes_live(tmp_path):
    s = DataStore(tmp_path, live_days=3)
    feed = frame("2026-10-20", 1440 * 7 + 300, 2e-7)    # 7 complete days + 5 h of "today"
    now = pd.Timestamp("2026-10-27 05:00")
    out = s.ingest_live(now=now, fetch=fake_feed(feed))
    assert out["days_in_feed"][-1] == "2026-10-26"
    assert s.history().index.max() == pd.Timestamp("2026-10-26 23:59")
    assert s.data_end() == pd.Timestamp("2026-10-26 23:59")
    assert sorted(f.stem for f in s.live_dir.glob("*.parquet")) == ["2026-10-24", "2026-10-25", "2026-10-26"]


def test_data_end_skips_incomplete_last_day(tmp_path):
    s = DataStore(tmp_path)
    s.merge(frame("2026-10-01", 1440 + 60, 1e-7))
    assert s.data_end() == pd.Timestamp("2026-10-01 23:59")


def test_registry_promote_train_end_and_prune(tmp_path):
    base = tmp_path / "base"
    (base / "C1").mkdir(parents=True)
    (base / "C1" / "forecaster.json").write_text(json.dumps({"trained_on": "2026-01-01 00:00:00 .. 2026-10-27 23:59:00 (GOES)"}))
    r = Registry(tmp_path / "reg", base)
    assert r.active_version("C1") == "base"
    assert r.train_end("C1") == pd.Timestamp("2026-10-27 23:59")
    for v in ("20261101", "20261102", "20261103", "20261104"):
        d = r.new_version_dir("C1", v)
        d.mkdir(parents=True)
        (d / "forecaster.json").write_text(json.dumps({"train_end": f"2026-10-{int(v[-2:]) + 20} 23:59"}))
    r.promote("C1", "20261101")
    assert r.train_end("C1") == pd.Timestamp("2026-10-21 23:59")
    removed = r.prune("C1", keep=2)
    assert removed == ["20261102"]
    assert sorted(p.name for p in (tmp_path / "reg" / "C1").iterdir()) == ["20261101", "20261103", "20261104"]


def test_decide():
    good = {"logloss": 0.10, "csi": 0.5, "flares": 4}
    bad = {"logloss": 0.12, "csi": 0.5, "flares": 4}
    assert decide(good, bad, 7)[0] is True
    assert decide(bad, good, 7)[0] is False
    assert decide(good, bad, MIN_UNSEEN_DAYS - 0.5)[0] is False               # too little unseen data
    assert decide({**good, "csi": 0.2}, {**bad, "csi": 0.5}, 7)[0] is False  # better log loss, clearly worse alarms
    assert decide({**good, "flares": 0, "csi": 0}, {**bad, "flares": 0, "csi": 0}, 7)[0] is True
