# Nowcast — streaming solar flare detection (Aditya-L1 SoLEXS/HEL1OS · GOES-R XRS)

*Is a flare happening right now, when did it start, and how big is it?*

`l1_nowcast` is a streaming engine: it processes **one sample at a time**, so the same code runs on a live feed and in
back-tests (batch mode is the engine in a loop — verified identical in the notebook and the tests).
It is the improved successor of Team Phoenix's six-script nowcast pipeline.

## Results (held-out, never used for tuning)

Validated against **NOAA's official GOES flare list**. Parameters were tuned on Jan–Apr 2026 and the scores below are from
**May 1 – Oct 27 2026** (936 NOAA flares ≥ C1).

| Detector (same held-out period) | Precision | Recall ≥C1 | F1 | M-class POD | X-class POD |
|---|---|---|---|---|---|
| Global `mean + 5σ` threshold (labelling rule of the original forecast notebook) | 0.96 | 0.03 | 0.05 | 0.23 | 1.00 |
| Team Phoenix-style rolling median detector | 0.75 | 0.73 | 0.74 | 0.86 | 1.00 |
| **This engine** | **0.91** | **0.91** | **0.91** | **0.98** (97/99) | **1.00** (3/3) |

* Class letter agrees with NOAA for **99.9 %** of matched flares (no M flare called C, no X called M).
* Alerts come a median **6 min after NOAA's flare start**, a median **3 min before the peak**.
  NOAA's start time is itself retrospective — it is set to the start of a 4-minute rise once that rise has happened.
* All 4 X-flares NOAA lists inside the team's Aditya-L1 window are detected with the right class
  (X1.5 Mar 30, X2.5 Apr 24 00:51, X2.5 Apr 24 08:01, X1.0 Jun 3).
  Note: `calib_check.py` expects an "X2.4 on Apr 23", but NOAA's GOES-19 list has no X-flare that day (its largest is M5.0).
* Throughput: ~25,000 samples/s on a laptop, about 25,000× faster than Aditya-L1's 1 Hz cadence.

## How it works

```
sample ─► background ─► threshold ─► state machine ─► events ─► fusion (soft + hard)
```

* **Background**: trailing 2 h window, median + IQR/1.349 (log-flux for GOES, counts for SoLEXS, Poisson mean for HEL1OS).
  Only samples judged *quiet* on arrival are admitted. This is the causal form of the team's two-pass fix: a long flare can never
  raise its own background. If no quiet sample is seen for 6 h, the window re-seeds so a permanent baseline step can't lock the detector.
* **Onset**: the threshold is crossed (`confirm` samples, plus a *rise guard*: flux ≥ 1.3 × its 5-sample minimum, which rejects noise on
  decaying tails). Alternatively, a NOAA-style sustained rise opens a flare.
* **End**: the NOAA rule, flux back half-way to the pre-flare level → DECAY. A sustained re-rise after the peak closes the flare
  and opens a new one, so overlapping flares are catalogued separately.
* **Classes**: real GOES classes from peak flux. For SoLEXS counts, `fit_counts_to_flux` cross-calibrates against GOES on flares
  seen by both instruments.
* **Fusion**: primary (XRS-B / SoLEXS) and secondary (XRS-A / HEL1OS CdTe) events within ±5 min → `BOTH` / `PRIMARY_ONLY` /
  `SECONDARY_ONLY`. 93 % of M-flares and 100 % of X-flares are confirmed by both GOES channels.
* **Messages** for a UI or alerting: `onset`, `escalation` (e.g. C → M during the rise), `end`, `fused`.

## Data sources

| Source | Class / function | Notes |
|---|---|---|
| GOES-19 XRS **live** (NOAA SWPC JSON) | `GOESLiveSource` | public, 1-minute, updated every minute — the real-time stream |
| GOES-19 XRS archive + NOAA flare list (NCEI) | `download_goes_archive`, `load_goes_avg1m`, `load_goes_flare_summary` | training and validation |
| Aditya-L1 SoLEXS / HEL1OS CSV (Team Phoenix export format) | `read_solexs_csv`, `read_hel1os_csv`, `align_frames` | HEL1OS: duplicates dropped, ~1.6 Hz → 1 Hz (counts summed, hardness averaged) |
| Aditya-L1 OGIP FITS light curves (ISSDC / PRADAN) | `read_ogip_lightcurve` | MJDREF/TIMEZERO time convention |
| New files dropped into a folder | `DirectoryWatchSource` | how a continuous Aditya-L1 feed is consumed. ISRO has no public real-time API; data are released as files via PRADAN |
| Multi-instrument raw streams | `merge_sources`, `StreamAligner` | per-channel aggregation onto a 1 s grid, with explicit gaps |

## Usage

```python
from l1_nowcast import NowcastEngine, GOESLiveSource, goes_channels

eng = NowcastEngine(goes_channels(), cadence_s=60)
for sample in GOESLiveSource():                 # backfills, then polls every minute
    record, messages = eng.update(sample)
    for m in messages:
        print(m["type"], m["time"], m.get("channel"), m.get("event_id"))
```

Aditya-L1:

```python
from l1_nowcast import aditya_channels, align_frames, read_solexs_csv, read_hel1os_csv, run_batch
df = align_frames(read_solexs_csv("combined_solexs_dataset.csv"), read_hel1os_csv("combined_hel1os_dataset_full.csv"))
res = run_batch(df[["solexs_counts", "hel1os_cdte"]], aditya_channels(), cadence_s=1)
res.fused            # catalogue with BOTH / SOFT-only / HARD-only labels and soft-hard peak lag
```

## Files

```
notebooks/nowcast.ipynb      # download → audit → tune (Jan-Apr) → validate (May-Oct) → baselines → streaming → live → Aditya-L1 → export
├── l1_nowcast/
│   ├── background.py        # incremental rolling window + flare-masked adaptive background
│   ├── detector.py          # channel state machine, events, fusion, engine, GOES classes, counts→flux calibration
│   ├── sources.py           # GOES live/archive, NOAA flare list, SoLEXS/HEL1OS CSV, OGIP FITS, folder watcher, aligner
│   └── validate.py          # one-to-one matching vs a reference catalogue, scores, parameter tuning
tests/test_nowcast.py        # 19 tests (synthetic flares, streaming = batch, fusion, readers, aligner, parsing)
outputs/nowcast/             # catalogues, validation tables, tuned config (nowcast_config.json), figures
data/                        # GOES archive cache (git-ignored, re-downloaded by the notebook)
```

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/nowcast.ipynb   # first run downloads ~100 MB from NOAA NCEI into data/
python -m pytest -q tests/test_nowcast.py
```

## Limitations

* The science validation uses GOES XRS because it comes with an independent flare catalogue. The Aditya-L1 path is implemented and
  unit-tested, but has not been scored on real SoLEXS/HEL1OS data here (the raw CSVs are not in either repository).
  `aditya_channels()` holds starting values; run `tune_channel` on SoLEXS once the data is loaded.
* Precision is measured against NOAA's list. Part of the remaining 9 % "false" detections are sub-peaks that NOAA merges into
  one flare, or small C1 events NOAA does not list.
