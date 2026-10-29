"""
Gradient-boosted flare forecaster: LightGBM + XGBoost probability average,
with optional isotonic calibration.

Changes from the original ensemble:
  * no StandardScaler / PCA - trees don't need scaling, and PCA hides which
    physical signal drives a forecast (feature importances are reported
    instead);
  * every member contributes a probability (the original averaged LightGBM's
    hard 0/1 ``predict`` with the others' probabilities);
  * no ``scale_pos_weight`` - reweighting distorts probabilities; the
    operating point is chosen afterwards by tuning the alert threshold;
  * early stopping on the most recent slice of the training period.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LGB = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   verbose=-1, n_jobs=-1, random_state=42)
DEFAULT_XGB = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=20,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                   eval_metric="logloss", n_jobs=-1, random_state=42, early_stopping_rounds=100)


class FlareForecaster:
    def __init__(self, feature_names: list[str], use_lgb: bool = True, use_xgb: bool = True,
                 lgb_params: dict | None = None, xgb_params: dict | None = None):
        if not (use_lgb or use_xgb):
            raise ValueError("enable at least one model")
        self.feature_names = list(feature_names)
        self.use_lgb, self.use_xgb = use_lgb, use_xgb
        self.lgb_params = {**DEFAULT_LGB, **(lgb_params or {})}
        self.xgb_params = {**DEFAULT_XGB, **(xgb_params or {})}
        self.lgb_model = None
        self.xgb_model = None
        self.calib_x: np.ndarray | None = None
        self.calib_y: np.ndarray | None = None
        self.meta: dict = {}

    # -- training --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, es_fraction: float = 0.15) -> "FlareForecaster":
        """Fit on a chronologically ordered frame; the last ``es_fraction``
        of rows is held out for early stopping."""
        X = X[self.feature_names]
        cut = int(len(X) * (1 - es_fraction))
        Xtr, ytr, Xes, yes = X.iloc[:cut], y.iloc[:cut], X.iloc[cut:], y.iloc[cut:]
        if self.use_lgb:
            import lightgbm as lgb
            self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
            self.lgb_model.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
        if self.use_xgb:
            import xgboost as xgb
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            self.xgb_model.fit(Xtr, ytr, eval_set=[(Xes, yes)], verbose=False)
        self.meta.update(n_train=int(len(Xtr)), n_es=int(len(Xes)), pos_rate=float(y.mean()))
        return self

    # -- inference ------------------------------------------------------
    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        ps = []
        if self.lgb_model is not None:
            ps.append(self.lgb_model.predict_proba(X)[:, 1])
        if self.xgb_model is not None:
            ps.append(self.xgb_model.predict_proba(X)[:, 1])
        return np.mean(ps, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrate(self.predict_raw(X))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calib_x is None:
            return raw
        return np.interp(raw, self.calib_x, self.calib_y)

    def fit_calibration(self, raw: np.ndarray, y: np.ndarray) -> None:
        """Isotonic map raw -> calibrated probability (stored as a lookup
        table so it can be saved to JSON and applied with np.interp)."""
        from sklearn.isotonic import IsotonicRegression
        ok = np.isfinite(raw)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw[ok], np.asarray(y)[ok])
        self.calib_x = np.asarray(iso.X_thresholds_, float)
        self.calib_y = np.asarray(iso.y_thresholds_, float)

    def feature_importance(self) -> pd.DataFrame:
        cols = {}
        if self.lgb_model is not None:
            g = self.lgb_model.booster_.feature_importance(importance_type="gain")
            cols["lgb_gain"] = g / g.sum()
        if self.xgb_model is not None:
            sc = self.xgb_model.get_booster().get_score(importance_type="gain")
            g = np.array([sc.get(f, 0.0) for f in self.feature_names])
            cols["xgb_gain"] = g / g.sum() if g.sum() else g
        df = pd.DataFrame(cols, index=self.feature_names)
        df["mean"] = df.mean(axis=1)
        return df.sort_values("mean", ascending=False)

    # -- persistence -----------------------------------------------------
    def save(self, directory: str | Path, extra: dict | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.lgb_model is not None:
            self.lgb_model.booster_.save_model(str(d / "lgb_model.txt"))
        if self.xgb_model is not None:
            self.xgb_model.get_booster().save_model(str(d / "xgb_model.json"))
        meta = {
            "feature_names": self.feature_names,
            "use_lgb": self.lgb_model is not None, "use_xgb": self.xgb_model is not None,
            "calib_x": None if self.calib_x is None else self.calib_x.tolist(),
            "calib_y": None if self.calib_y is None else self.calib_y.tolist(),
            "meta": self.meta, **(extra or {}),
        }
        (d / "forecaster.json").write_text(json.dumps(meta, indent=2, default=str))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "FlareForecaster":
        d = Path(directory)
        meta = json.loads((d / "forecaster.json").read_text())
        obj = cls(meta["feature_names"], use_lgb=meta["use_lgb"], use_xgb=meta["use_xgb"])
        if meta["use_lgb"]:
            import lightgbm as lgb
            obj.lgb_model = _LgbBoosterWrapper(lgb.Booster(model_file=str(d / "lgb_model.txt")))
        if meta["use_xgb"]:
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(d / "xgb_model.json"))
            obj.xgb_model = _XgbBoosterWrapper(booster)
        if meta.get("calib_x") is not None:
            obj.calib_x = np.asarray(meta["calib_x"], float)
            obj.calib_y = np.asarray(meta["calib_y"], float)
        obj.meta = meta.get("meta", {})
        obj.saved = meta
        return obj


class _LgbBoosterWrapper:
    """Minimal predict_proba / booster_ interface over a loaded lgb.Booster."""

    def __init__(self, booster):
        self.booster_ = booster

    def predict_proba(self, X):
        p = self.booster_.predict(X)
        return np.column_stack([1 - p, p])


class _XgbBoosterWrapper:
    def __init__(self, booster):
        self._b = booster

    def get_booster(self):
        return self._b

    def predict_proba(self, X):
        import xgboost as xgb
        p = self._b.predict(xgb.DMatrix(X, feature_names=list(X.columns)))
        return np.column_stack([1 - p, p])
