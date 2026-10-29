"""
Pure-numpy inference for the saved LightGBM (``lgb_model.txt``) and XGBoost
(``xgb_model.json``) models.

The native libraries need OpenMP (``libgomp.so.1``), which serverless hosts
such as Vercel do not provide, and they are large. Prediction is only a walk
down each tree, so the deployed site evaluates the saved files directly; the
native libraries are needed for training only. Results match the native
``predict`` to floating-point rounding (tests/test_forecast.py).

Supports what the trainer produces: binary logistic objective, numerical
splits, NaN/zero missing-value handling. Categorical splits are rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ZERO = 1e-35  # LightGBM's kZeroThreshold


class _Forest:
    """Trees flattened into arrays. Child pointers >= 0 are internal nodes; a
    negative pointer ``c`` is the leaf ``~c``."""

    def __init__(self, trees: list[dict], le: bool, dtype):
        self.trees, self.le, self.dtype = trees, le, dtype

    def margin(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=self.dtype)
        n = len(X)
        out = np.zeros(n)
        rows = np.arange(n)
        for t in self.trees:
            if t["root_leaf"] is not None:
                out += t["root_leaf"]
                continue
            cur = np.zeros(n, dtype=np.int64)
            live = rows
            while live.size:
                node = cur[live]
                x = X[live, t["feat"][node]]
                nan = np.isnan(x)
                mt = t["mtype"][node]
                if self.le:  # LightGBM: NaN counts as 0 unless missing_type is NaN
                    x = np.where(nan & (mt != 2), 0.0, x)
                    missing = ((mt == 1) & (np.abs(x) <= _ZERO)) | ((mt == 2) & nan)
                    go_left = np.where(missing, t["dleft"][node], x <= t["thr"][node])
                else:        # XGBoost: NaN follows the default direction, else x < threshold
                    go_left = np.where(nan, t["dleft"][node], x < t["thr"][node])
                nxt = np.where(go_left, t["left"][node], t["right"][node])
                cur[live] = nxt
                live = live[nxt >= 0]
            out += t["leaf"][~cur]
        return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class NumpyLgbModel:
    """Drop-in for the loaded LightGBM booster wrapper (``predict_proba``)."""

    def __init__(self, path: str | Path):
        blocks = [b.strip() for b in Path(path).read_text().split("\n\n")]
        head = dict(l.split("=", 1) for l in blocks[0].splitlines() if "=" in l)
        obj = head.get("objective", "")
        if not obj.startswith("binary"):
            raise ValueError(f"unsupported LightGBM objective {obj!r}")
        self.sigmoid = float(obj.split("sigmoid:")[1].split()[0]) if "sigmoid:" in obj else 1.0
        self.feature_names = head["feature_names"].split()
        trees = []
        for b in blocks[1:]:
            if b.startswith("end of trees"):
                break
            if not b.startswith("Tree="):
                continue
            kv = dict(l.split("=", 1) for l in b.splitlines() if "=" in l)
            leaf = np.array(kv["leaf_value"].split(), float)
            if int(kv["num_leaves"]) == 1:
                trees.append({"root_leaf": float(leaf[0])})
                continue
            if int(kv.get("num_cat", 0)):
                raise ValueError("categorical splits are not supported")
            dt = np.array(kv["decision_type"].split(), int)
            trees.append({
                "root_leaf": None, "leaf": leaf,
                "feat": np.array(kv["split_feature"].split(), int),
                "thr": np.array(kv["threshold"].split(), float),
                "left": np.array(kv["left_child"].split(), int),
                "right": np.array(kv["right_child"].split(), int),
                "dleft": (dt & 2) > 0, "mtype": (dt >> 2) & 3,
            })
        self.forest = _Forest(trees, le=True, dtype=np.float64)

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.sigmoid * self.forest.margin(X))
        return np.column_stack([1 - p, p])


class NumpyXgbModel:
    """Drop-in for the loaded XGBoost booster wrapper (``predict_proba``)."""

    def __init__(self, path: str | Path):
        d = json.loads(Path(path).read_text())
        learner = d["learner"]
        if learner["objective"]["name"] != "binary:logistic":
            raise ValueError(f"unsupported XGBoost objective {learner['objective']['name']!r}")
        base = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
        self.base_margin = float(np.log(base / (1 - base)))
        self.feature_names = learner.get("feature_names") or []
        trees = []
        for t in learner["gradient_booster"]["model"]["trees"]:
            if any(t.get("split_type", [])):
                raise ValueError("categorical splits are not supported")
            left = np.array(t["left_children"], int)
            right = np.array(t["right_children"], int)
            cond = np.array(t["split_conditions"], np.float32)
            is_leaf = left == -1
            if is_leaf[0]:
                trees.append({"root_leaf": float(cond[0])})
                continue
            ids = np.arange(len(left))
            enc = np.where(is_leaf, ~ids, ids)   # pointer to a leaf node -> negative code
            trees.append({
                "root_leaf": None, "leaf": cond.astype(np.float64),
                "feat": np.array(t["split_indices"], int), "thr": cond,
                "left": np.where(left >= 0, enc[np.maximum(left, 0)], 0),
                "right": np.where(right >= 0, enc[np.maximum(right, 0)], 0),
                "dleft": np.array(t["default_left"], bool), "mtype": np.zeros(len(left), int),
            })
        self.forest = _Forest(trees, le=False, dtype=np.float32)

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.base_margin + self.forest.margin(X))
        return np.column_stack([1 - p, p])
