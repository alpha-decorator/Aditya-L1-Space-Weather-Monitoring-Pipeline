"""Unit tests for l1_nowcast. Run from the repository root:  python -m pytest -q"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from l1_nowcast import (  # noqa: E402
    ChannelConfig, DirectoryWatchSource, EventFuser, FlareEvent, FrameSource, NowcastEngine,
    RollingWindow, Sample, StreamAligner, aditya_channels, class_to_flux, fit_counts_to_flux,
    goes_channels, goes_class, parse_swpc_xrays, read_hel1os_csv, read_ogip_lightcurve,
    read_solexs_csv, run_batch,
)


# ---------------------------------------------------------------- helpers

def quiet_series(minutes: int, level: float = 1e-6, noise: float = 0.02, seed: int = 0,
                 start: str = "2026-03-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=minutes, freq="1min")
    v = level * 10 ** (noise * rng.standard_normal(minutes))
    return pd.DataFrame({"xrsb": v, "xrsa": v / 20}, index=idx)


def add_flare(df: pd.DataFrame, at: str, peak: float, rise_min: int = 8, decay_min: int = 20):
    t0 = pd.Timestamp(at)
    m = (df.index - t0).total_seconds().to_numpy() / 60.0
    shape = np.where(m < 0, 0.0, np.where(m < rise_min, m / rise_min, np.exp(-(m - rise_min) / decay_min)))
    out = df.copy()
    out["xrsb"] = out["xrsb"] + peak * shape
    out["xrsa"] = out["xrsa"] + peak / 5 * shape
    return out


# ---------------------------------------------------------------- statistics

def test_rolling_window_quantiles_match_numpy():
    rng = np.random.default_rng(1)
    w = RollingWindow(100.0)
    xs = rng.standard_normal(500)
    for i, x in enumerate(xs):
        w.expire(float(i))
        w.add(float(i), float(x))
    w.expire(500.0)
    live = xs[-99:]  # window (400, 500] excludes t=400
    assert len(w) == len(live)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert w.quantile(q) == pytest.approx(np.quantile(live, q), abs=1e-12)
    assert w.mean() == pytest.approx(live.mean(), abs=1e-9)


def test_goes_class_formatting():
    assert goes_class(1.527e-4) == "X1.5"
    assert goes_class(9.97e-6) == "C9.9"          # truncation, never rounds up a letter
    assert goes_class(1e-5) == "M1.0"
    assert goes_class(2.3e-3) == "X23.0"
    assert goes_class(5e-9) == "A0.5"
    assert goes_class(float("nan")) is None
    assert class_to_flux("M2.3") == pytest.approx(2.3e-5)
    assert math.isnan(class_to_flux("Q1"))


# ---------------------------------------------------------------- detection

def test_detects_single_flare_with_correct_class_and_start():
    df = add_flare(quiet_series(600), "2026-03-01 05:00", peak=2e-5)
    res = run_batch(df, goes_channels(), 60)
    ev = res.primary_events("xrsb")
    assert len(ev) == 1
    e = ev.iloc[0]
    assert e["flare_class"][0] == "M"
    assert abs((e["start"] - pd.Timestamp("2026-03-01 05:00")).total_seconds()) <= 120
    assert e["detect_time"] <= pd.Timestamp("2026-03-01 05:04")
    assert e["peak_time"] == pd.Timestamp("2026-03-01 05:08")


def test_quiet_sun_gives_no_events():
    res = run_batch(quiet_series(3000, seed=3), goes_channels(), 60)
    assert len(res.primary_events("xrsb")) == 0


def test_long_flare_does_not_mask_itself():
    # 4-hour flare: a naive trailing background would rise to meet it.
    df = add_flare(quiet_series(900), "2026-03-01 04:00", peak=3e-5, rise_min=60, decay_min=150)
    res = run_batch(df, goes_channels(), 60)
    bg = res.records["xrsb_background"]
    assert bg["2026-03-01 06:00":"2026-03-01 08:00"].max() < 1.5e-6
    ev = res.primary_events("xrsb")
    assert (ev["flare_class"].str[0] == "M").sum() == 1


def test_second_flare_on_decaying_tail_is_separate_event():
    df = add_flare(quiet_series(900), "2026-03-01 04:00", peak=5e-5, decay_min=60)
    df = add_flare(df, "2026-03-01 04:40", peak=4e-5, rise_min=6, decay_min=15)
    res = run_batch(df, goes_channels(), 60)
    ev = res.primary_events("xrsb")
    assert len(ev) == 2
    assert ev.iloc[1]["trigger"] == "rise"
    assert abs((ev.iloc[1]["start"] - pd.Timestamp("2026-03-01 04:40")).total_seconds()) <= 180


def test_gap_closes_event_and_creates_no_false_onset():
    df = add_flare(quiet_series(600), "2026-03-01 05:00", peak=2e-5)
    df.loc["2026-03-01 05:10":"2026-03-01 05:40"] = np.nan
    res = run_batch(df, goes_channels(), 60)
    ev = res.primary_events("xrsb")
    assert ev.iloc[0]["end_reason"] == "data_gap"
    assert len(ev) <= 2  # data returning on the tail may reopen once, never more


def test_streaming_equals_batch():
    df = add_flare(quiet_series(800, seed=5), "2026-03-01 06:00", peak=8e-6)
    df = add_flare(df, "2026-03-01 09:30", peak=3e-5)
    batch = run_batch(df, goes_channels(), 60)
    eng = NowcastEngine(goes_channels(), 60)
    msgs = []
    for s in FrameSource(df):
        msgs.extend(eng.update(s)[1])
    msgs.extend(eng.finalize())
    pd.testing.assert_frame_equal(eng.events(), batch.events)
    pd.testing.assert_frame_equal(eng.fused(), batch.fused)
    assert [m["type"] for m in msgs] == [m["type"] for m in batch.messages]


def test_messages_include_onset_end_and_fused():
    df = add_flare(quiet_series(600), "2026-03-01 05:00", peak=2e-5)
    res = run_batch(df, goes_channels(), 60)
    types = {m["type"] for m in res.messages}
    assert {"onset", "end", "fused"} <= types


# ---------------------------------------------------------------- fusion

def _ev(eid, ch, start, end, peak_t, peak):
    e = FlareEvent(event_id=eid, channel=ch, start=pd.Timestamp(start), detect_time=pd.Timestamp(start),
                   background=1e-7, trigger="level", peak_time=pd.Timestamp(peak_t), peak=peak)
    e.end = pd.Timestamp(end)
    return e


def test_fusion_secondary_closing_before_primary_still_matches():
    fu = EventFuser(300)
    s = _ev("s1", "xrsa", "2026-01-01 00:02", "2026-01-01 00:10", "2026-01-01 00:05", 1e-6)
    p = _ev("p1", "xrsb", "2026-01-01 00:01", "2026-01-01 00:30", "2026-01-01 00:07", 1e-5)
    fu.add(s, "secondary")
    # secondary has long expired, but the primary is still open -> must wait
    out = fu.tick(pd.Timestamp("2026-01-01 00:20"), open_primary=[p])
    assert out == []
    fu.add(p, "primary")
    out = fu.tick(pd.Timestamp("2026-01-01 00:36"))
    assert len(out) == 1 and out[0]["fused"].label == "BOTH"
    assert out[0]["fused"].peak_lag_s == 120.0


def test_fusion_labels_single_instrument_events():
    fu = EventFuser(300)
    fu.add(_ev("p1", "xrsb", "2026-01-01 01:00", "2026-01-01 01:20", "2026-01-01 01:05", 1e-5), "primary")
    fu.add(_ev("s1", "xrsa", "2026-01-01 05:00", "2026-01-01 05:05", "2026-01-01 05:02", 1e-7), "secondary")
    out = fu.tick(pd.Timestamp("2026-01-01 06:00"))
    assert sorted(m["fused"].label for m in out) == ["PRIMARY_ONLY", "SECONDARY_ONLY"]


# ---------------------------------------------------------------- Aditya-L1 inputs

def test_hel1os_csv_dedup_and_1hz_binning(tmp_path):
    t = pd.Timestamp("2026-03-01 00:00:00.733")
    times = [t + pd.Timedelta(seconds=0.6 * i) for i in range(10)]
    df = pd.DataFrame({"time": times, "cdte_broad": 1, "cdte_hardness": 0.5,
                       "czt_broad": 2, "czt_hardness": 0.4})
    df = pd.concat([df, df.iloc[:3]])  # exact duplicate rows, as in the real export
    p = tmp_path / "hel1os.csv"
    df.to_csv(p, index=False)
    out = read_hel1os_csv(p)
    assert out["hel1os_cdte"].sum() == 10          # duplicates removed, counts summed
    assert out.index.freq == pd.tseries.frequencies.to_offset("1s")
    assert set(out["hel1os_cdte_hardness"].dropna()) == {0.5}


def test_solexs_csv_gaps_become_nan(tmp_path):
    idx = list(pd.date_range("2026-03-01", periods=5, freq="1s")) + \
          list(pd.date_range("2026-03-01 00:00:10", periods=5, freq="1s"))
    p = tmp_path / "solexs.csv"
    pd.DataFrame({"time": idx, "counts": 9.0, "hardness": 1.6}).to_csv(p, index=False)
    out = read_solexs_csv(p)
    assert len(out) == 15
    assert out["solexs_counts"].isna().sum() == 5


def test_stream_aligner_sums_counts_and_emits_gaps():
    al = StreamAligner(1.0, {"c": "sum", "h": "mean"})
    t0 = pd.Timestamp("2026-03-01")
    samples = [Sample(t0 + pd.Timedelta(seconds=x), {"c": 1.0, "h": h})
               for x, h in [(0.1, 1.0), (0.7, 3.0), (1.3, 2.0), (4.2, 5.0)]]
    out = list(al.run(samples))
    assert [o.values["c"] for o in out[:2]] == [2.0, 1.0]
    assert out[0].values["h"] == 2.0
    assert math.isnan(out[2].values["c"]) and math.isnan(out[3].values["c"])
    assert out[-1].values["c"] == 1.0


def test_ogip_fits_reader(tmp_path):
    from astropy.io import fits
    tt = np.arange(0, 10, 1.0)
    cols = fits.ColDefs([fits.Column(name="TIME", format="D", array=tt),
                         fits.Column(name="COUNTS", format="E", array=np.full(10, 7.0))])
    h = fits.BinTableHDU.from_columns(cols)
    h.header["MJDREFI"] = 61100   # 2026-03-01 is MJD 61100
    h.header["MJDREFF"] = 0.0
    h.header["TIMEDEL"] = 1.0
    p = tmp_path / "solexs_lc.fits"
    fits.HDUList([fits.PrimaryHDU(), h]).writeto(p)
    out = read_ogip_lightcurve(p, "solexs_counts")
    assert out.index[0] == pd.Timestamp("2026-03-01")
    assert (out["solexs_counts"] == 7.0).all()


def test_directory_watch_streams_new_files_in_order(tmp_path):
    def write(name, start, n):
        idx = pd.date_range(start, periods=n, freq="1s")
        pd.DataFrame({"time": idx, "counts": range(n), "hardness": 1.0}).to_csv(tmp_path / name, index=False)
    write("a.csv", "2026-03-01 00:00:00", 5)
    write("b.csv", "2026-03-01 00:00:03", 5)  # overlaps a.csv by 2 s
    src = DirectoryWatchSource(tmp_path, "*.csv", read_solexs_csv)
    times = [s.time for s in src]
    assert times == sorted(times) and len(times) == len(set(times)) == 8


def test_aditya_channels_detect_soft_and_hard_flare():
    rng = np.random.default_rng(0)
    n = 4 * 3600
    idx = pd.date_range("2026-03-01", periods=n, freq="1s")
    sec = np.arange(n)
    soft = rng.poisson(9, n).astype(float)
    hard = rng.poisson(0.2, n).astype(float)
    t0 = 3 * 3600
    soft += np.where(sec >= t0, 800 * np.exp(-np.maximum(sec - t0 - 300, 0) / 600) * np.minimum((sec - t0) / 300, 1), 0)
    hard += np.where((sec >= t0) & (sec < t0 + 240), rng.poisson(40, n), 0)
    df = pd.DataFrame({"solexs_counts": soft, "hel1os_cdte": hard}, index=idx)
    res = run_batch(df, aditya_channels(), 1)
    assert (res.fused["label"] == "BOTH").sum() == 1


# ---------------------------------------------------------------- GOES live parsing / calibration

def test_parse_swpc_json():
    payload = json.dumps([
        {"time_tag": "2026-10-28T11:50:00Z", "satellite": 19, "flux": 6e-9, "energy": "0.05-0.4nm"},
        {"time_tag": "2026-10-28T11:50:00Z", "satellite": 19, "flux": 4e-7, "energy": "0.1-0.8nm"},
        {"time_tag": "2026-10-28T11:51:00Z", "satellite": 19, "flux": -1, "energy": "0.1-0.8nm"},
    ])
    df = parse_swpc_xrays(payload)
    assert df.index[0] == pd.Timestamp("2026-10-28 11:50")
    assert df.loc["2026-10-28 11:50", "xrsb"] == 4e-7
    assert math.isnan(df.loc["2026-10-28 11:51", "xrsb"])


def test_counts_to_flux_calibration_recovers_power_law():
    c = np.array([50, 200, 1000, 5000, 15000.0])
    f = 10 ** (-9.0 + 0.9 * np.log10(c))
    cal = fit_counts_to_flux(c, f)
    a, b = cal.coefficients
    assert a == pytest.approx(-9.0) and b == pytest.approx(0.9)
    assert cal(1000) == pytest.approx(10 ** (-9 + 2.7))
