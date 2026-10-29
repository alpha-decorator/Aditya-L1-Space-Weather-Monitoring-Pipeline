"""
l1_forecast - 15-minute solar flare forecasting on top of the l1_nowcast
streaming engine (Aditya-L1 SoLEXS/HEL1OS or GOES-R XRS input).
"""

import l1_nowcast  # noqa: F401

from .alerts import (AlertManager, alert_episodes, alert_series, probability_scores,
                     score_events, tss_at, tune_alert_rule)
from .engine import ForecastEngine
from .features import build_features, eligible_mask, records_to_frame, required_history_min
from .labels import horizon_covered, make_targets, qualifying_events
from .model import FlareForecaster
from .training import month_starts, walk_forward

__version__ = "1.0.0"
