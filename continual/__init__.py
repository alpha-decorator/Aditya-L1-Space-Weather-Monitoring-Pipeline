"""
continual - keep the deployed flare forecaster learning from new data.

Every day:
  1. ``DataStore.ingest_live``   store the new complete days of GOES-19 XRS (rolling 10-day raw cache
                                 + permanent history)
  2. ``retrain.daily_cycle``     retrain each forecaster on ALL stored history, score the candidate and the
                                 active model on recent days neither has seen, and promote the candidate
                                 only if it is at least as good
  3. ``Registry``                versioned models, the active model per target, and a log of every decision

    python -m continual seed      # once: import the NCEI archive already downloaded by nowcast.ipynb
    python -m continual daily     # every day (cron / GitHub Actions / serve.py --retrain-daily)
    python -m continual status
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from .registry import Registry  # noqa: E402
from .store import DataStore  # noqa: E402

__all__ = ["DataStore", "Registry", "ROOT"]
