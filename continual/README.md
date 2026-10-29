# Continual learning — the forecaster keeps learning while deployed

Every day the system stores the new day of GOES-19 XRS data, retrains both forecasters (≥C1, ≥M1),
and switches to the new model **only if it does better on recent days that neither model has seen**.

```
00:30 UTC daily
  ingest   NOAA SWPC 7-day feed ─► store/live/<day>.parquet (raw, kept 10 days)
                                └► store/history/xrs_<month>.parquet (permanent)
  retrain  all history ─► nowcast ─► features ─► candidate model (data up to 7 days ago)
  gate     candidate vs active model on the unseen days ─► promote, or keep the active model
  log      registry/log.jsonl ─► dashboard "Model learning" card
```

## Why it trains on all history, not only the last few days

M-class flares are rare: a typical week has zero to a handful. A model trained only on the last 5–10 days
would forget what a big flare looks like after a quiet week, and raise false alarms everywhere after a
busy one. So the last 10 days are kept as a raw cache (for monitoring and catching up on missed runs),
while training always uses the full, growing history. The model improves because the history grows by one
day, every day.

## The gate (why it can't silently get worse)

* The candidate is trained on data up to `T − 7 days`. Its probability calibration and alert threshold
  are fitted on the 30 days before that, with a model that had not seen them (same recipe as
  `forecast/forecast.ipynb`).
* Both models are scored on the days after **both** of their training periods (plus a 1-day embargo),
  so neither has seen them.
* Promote if the candidate's log loss is no worse (0.5 % tolerance) and its alarm CSI is not clearly
  worse (0.05). With fewer than 3 unseen days, keep the active model and only log the run.

The starting champion is the validated model in `forecast/models/` (version `base`), trained through
Oct 27 2026. The first fair comparison is therefore possible from about Nov 1.

Replayed on past data (champion trained through Sep 8, candidate through Sep 23, scored on Sep 25–30):
the C1 candidate was promoted (log loss 0.129 vs 0.142, caught 2 of 16 flares vs 0), and the M1 candidate
was rejected (log loss 0.0037 vs 0.0013): the gate blocks a worse model.

## Commands

```bash
python -m continual seed       # once: import the NCEI archive downloaded by nowcast.ipynb (Jan 1 – Oct 27 2026)
python -m continual daily      # ingest + retrain + gate (~30 s)
python -m continual status     # store coverage, active versions, recent decisions
python -m pytest -q continual/tests
```

Scheduling:

* **Locally**: `python dashboard/serve.py --retrain-daily` runs the cycle at 00:30 UTC (and at start-up if
  it hasn't run today). The dashboard picks up a promoted model on its next refresh, no restart needed.
* **Deployed**: `.github/workflows/daily-retrain.yml` runs the cycle on GitHub Actions at 00:30 UTC and
  commits `continual/store` and `continual/registry`. The push redeploys the site with the new active models.

## Layout

```
continual/
├── store.py      DataStore: history (permanent) + live cache (rolling 10 days), ingest, merge
├── registry.py   versioned models, active.json, log.jsonl, pruning (active + 2 newest kept)
├── retrain.py    dataset build, candidate training, evaluation, promotion gate, daily_cycle
├── store/        data (~8 MB for 10 months, grows ~25 KB/day)
└── registry/     models and the decision log
```

Data note: history from Jan 1 to Oct 27 2026 is the NCEI science archive; later days come from the
SWPC real-time feed (the same data the dashboard sees live). Where both exist, the archive is kept.
