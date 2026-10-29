"""
Forecast targets derived from the nowcast catalogue.

    y(t) = 1  if the nowcast will confirm a qualifying flare in (t, t + H]

"Qualifying" means peak at or above ``min_class`` (C1.0 by default, M1.0
for the major-flare model). Labels come from the *same* causal nowcast
that runs operationally, so the forecaster is trained to do exactly what
the system needs: warn before the nowcast alarm goes off.

This replaces the original notebook's target (counts > global mean +
5 sigma), which used statistics of the whole dataset - including the test
period - and split one flare into several "events" whenever counts
dipped across the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from l1_nowcast import class_to_flux


def qualifying_events(events: pd.DataFrame, channel: str, min_class: str = "C1.0") -> pd.DataFrame:
    e = events[events["channel"] == channel] if "channel" in events else events
    fmin = class_to_flux(min_class)
    peak = e["peak"].astype(float)
    if "flare_class" in e and e["flare_class"].notna().any():
        peak = peak.where(e["flare_class"].isna(), e["flare_class"].map(class_to_flux))
    return e[peak >= fmin * (1 - 1e-9)].sort_values("detect_time").reset_index(drop=True)


def make_targets(index: pd.DatetimeIndex, event_times: pd.Series | np.ndarray,
                 horizon_min: float) -> pd.Series:
    """1 where an event time falls in (t, t + horizon]."""
    times = np.sort(pd.to_datetime(pd.Series(event_times)).to_numpy(dtype="datetime64[ns]"))
    t = index.to_numpy(dtype="datetime64[ns]")
    nxt = np.searchsorted(times, t, side="right")
    has = nxt < len(times)
    nxt_time = np.where(has, times[np.minimum(nxt, len(times) - 1)], np.datetime64("NaT"))
    h = np.timedelta64(int(horizon_min * 60), "s")
    y = has & (nxt_time - t <= h)
    return pd.Series(y.astype(np.int8), index=index, name="y")


def horizon_covered(index: pd.DatetimeIndex, data_end: pd.Timestamp, horizon_min: float) -> pd.Series:
    """False for the last ``horizon`` of the data, where the label is unknown."""
    return pd.Series(index <= data_end - pd.Timedelta(minutes=horizon_min), index=index)
