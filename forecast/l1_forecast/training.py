"""
Walk-forward (expanding window) training: the honest way to evaluate a
time-series forecaster.

For each test month, a fresh model is trained only on data that ends
``embargo`` before the month starts, then predicts that month. Stitching
the months together gives out-of-sample probabilities for the whole
evaluation span - exactly what an operational system retrained monthly
would have produced. The embargo prevents a label window (t, t+H] in the
training set from overlapping the test month.
"""

from __future__ import annotations

import time as _time
from typing import Callable

import numpy as np
import pandas as pd

from .model import FlareForecaster


def month_starts(first: str | pd.Timestamp, last: str | pd.Timestamp) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(first).normalize().replace(day=1), pd.Timestamp(last), freq="MS"))


def walk_forward(X: pd.DataFrame, y: pd.Series, mask: pd.Series, test_starts: list[pd.Timestamp],
                 data_end: pd.Timestamp, make_model: Callable[[list[str]], FlareForecaster],
                 embargo: str = "1D", verbose: bool = True) -> tuple[pd.Series, pd.DataFrame]:
    """Return (out-of-sample raw probabilities, per-fold summary)."""
    oos = pd.Series(np.nan, index=X.index, name="p_raw")
    folds = []
    bounds = list(test_starts) + [pd.Timestamp(data_end) + pd.Timedelta("1min")]
    for i in range(len(test_starts)):
        t0, t1 = bounds[i], bounds[i + 1]
        tr = mask & (X.index < t0 - pd.Timedelta(embargo))
        te = (X.index >= t0) & (X.index < t1)
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        clock = _time.time()
        mdl = make_model(list(X.columns)).fit(X[tr], y[tr])
        oos[te] = mdl.predict_raw(X[te])
        te_m = te & mask
        folds.append({"test_month": t0.strftime("%Y-%m"), "train_rows": int(tr.sum()),
                      "train_pos_rate": float(y[tr].mean()), "test_rows": int(te_m.sum()),
                      "test_pos_rate": float(y[te_m].mean()) if te_m.any() else np.nan,
                      "lgb_trees": getattr(getattr(mdl.lgb_model, "booster_", None), "best_iteration", None),
                      "seconds": round(_time.time() - clock, 1)})
        if verbose:
            f = folds[-1]
            print(f"  fold {f['test_month']}: train={f['train_rows']:,} rows "
                  f"(pos {f['train_pos_rate']:.3f})  test={f['test_rows']:,}  {f['seconds']}s")
    return oos, pd.DataFrame(folds)
