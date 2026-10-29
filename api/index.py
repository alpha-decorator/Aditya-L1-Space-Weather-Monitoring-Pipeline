"""
HTTP API for the dashboard.

    GET /api/snapshot   latest nowcast + forecast snapshot (recomputed at most every 30 s)
    GET /api/health

Run locally with ``python serve.py`` (which also serves public/). On Vercel
this module is the serverless function (``app``) and public/ is served as static files.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flaremon.snapshot import build_snapshot  # noqa: E402

TTL_S = 30.0

app = FastAPI(title="Solar flare monitor API")
app.add_middleware(GZipMiddleware, minimum_size=1024)
_lock = threading.Lock()
_cache: dict = {"at": 0.0, "data": None, "error": None}


def _get() -> tuple[dict | None, str | None]:
    with _lock:
        if _cache["data"] is None or time.monotonic() - _cache["at"] > TTL_S:
            try:
                _cache["data"], _cache["error"] = build_snapshot(), None
            except Exception as e:  # keep serving the last good snapshot
                _cache["error"] = f"{type(e).__name__}: {e}"
            _cache["at"] = time.monotonic()
        return _cache["data"], _cache["error"]


@app.get("/api/snapshot")
def snapshot():
    data, error = _get()
    if data is None:
        return JSONResponse({"error": error or "no data yet"}, status_code=503)
    return JSONResponse({**data, "error": error},
                        headers={"Cache-Control": "public, max-age=15, s-maxage=30"})


@app.get("/api/health")
def health():
    return {"ok": True, "cached": _cache["data"] is not None, "error": _cache["error"]}


# On Vercel, vercel.json rewrites /api/<route> to this function as /api/index?route=<route>, and the path
# the app then sees is not always the public one. Dispatch on ?route= (or the last path segment) instead.
def _dispatch(request: Request):
    route = request.query_params.get("route") or request.url.path.rstrip("/").rsplit("/", 1)[-1]
    return health() if route == "health" else snapshot()


@app.get("/api/{rest:path}")
def api_fallback(request: Request, rest: str = ""):
    return _dispatch(request)


if os.environ.get("VERCEL"):  # locally, "/" must stay free for the static page mounted by serve.py
    @app.get("/{rest:path}")
    def vercel_fallback(request: Request, rest: str = ""):
        return _dispatch(request)
