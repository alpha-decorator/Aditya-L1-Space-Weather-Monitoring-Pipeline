"""Local dev server: API + static dashboard on http://127.0.0.1:8000

    python serve.py                   # dashboard only
    python serve.py --retrain-daily   # also run the daily ingest + retrain at 00:30 UTC (and once at
                                      # start-up if it has not run today)
"""

import argparse
import datetime as dt
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

from api.index import app

REPO = Path(__file__).resolve().parent
RUN_AT_UTC = dt.time(0, 30)

app.mount("/", StaticFiles(directory=REPO / "public", html=True), name="public")


def _ran_today() -> bool:
    log = REPO / "continual" / "registry" / "log.jsonl"
    if not log.exists():
        return False
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    return bool(lines) and f'"run_at": "{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}' in lines[-1]


def _run_daily():
    print(f"[retrain] starting daily cycle at {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC", flush=True)
    # a separate process, so training never blocks or bloats the web server
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "continual", "daily"], cwd=REPO,
                       capture_output=True, text=True)
    print(r.stdout[-2000:] or r.stderr[-2000:], flush=True)


def _scheduler():
    if not _ran_today():
        _run_daily()
    while True:
        now = dt.datetime.now(dt.timezone.utc)
        nxt = dt.datetime.combine(now.date(), RUN_AT_UTC, dt.timezone.utc)
        if nxt <= now:
            nxt += dt.timedelta(days=1)
        time.sleep((nxt - now).total_seconds())
        _run_daily()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--retrain-daily", action="store_true")
    a = ap.parse_args()
    if a.retrain_daily:
        threading.Thread(target=_scheduler, daemon=True, name="retrain").start()
    uvicorn.run(app, host=a.host, port=a.port)
