"""
Model registry.

    registry/
    ├── active.json            {"C1": "C1/20261112", "M1": "base"}   which version serves forecasts
    ├── log.jsonl              one line per target per daily run: metrics of both models + decision
    └── <target>/<version>/    lgb_model.txt, xgb_model.json, forecaster.json

Version ``base`` is the hand-validated model in models/<target>, the starting champion.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from . import ROOT

REGISTRY = ROOT / "continual" / "registry"
BASE_MODELS = ROOT / "models"


class Registry:
    def __init__(self, root: str | Path = REGISTRY, base_models: str | Path = BASE_MODELS):
        self.root = Path(root)
        self.base = Path(base_models)

    # -- active models -----------------------------------------------------
    def _active(self) -> dict:
        p = self.root / "active.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def active_version(self, target: str) -> str:
        return self._active().get(target, "base").split("/")[-1]

    def model_dir(self, target: str, version: str | None = None) -> Path:
        version = version or self.active_version(target)
        return self.base / target if version == "base" else self.root / target / version

    def active_dir(self, target: str) -> Path:
        return self.model_dir(target)

    def train_end(self, target: str, version: str | None = None) -> pd.Timestamp:
        """Last minute of data the model was trained on (nothing after it is in-sample)."""
        meta = json.loads((self.model_dir(target, version) / "forecaster.json").read_text())
        if meta.get("train_end"):
            return pd.Timestamp(meta["train_end"])
        return pd.Timestamp(meta["trained_on"].split("..")[-1].split("(")[0].strip())

    def promote(self, target: str, version: str) -> None:
        a = self._active()
        a[target] = "base" if version == "base" else f"{target}/{version}"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "active.json").write_text(json.dumps(a, indent=2) + "\n")

    # -- candidates ----------------------------------------------------------
    def new_version_dir(self, target: str, version: str) -> Path:
        d = self.root / target / version
        if d.exists():
            shutil.rmtree(d)
        return d

    def prune(self, target: str, keep: int = 2) -> list[str]:
        """Keep the active version plus the ``keep`` newest others (model files are a few MB each)."""
        d = self.root / target
        if not d.exists():
            return []
        active = self.active_version(target)
        versions = sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        others = [p for p in versions if p.name != active]
        removed = []
        for p in others[keep:]:
            shutil.rmtree(p)
            removed.append(p.name)
        return removed

    # -- decision log ---------------------------------------------------------
    def log(self, entry: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def history(self, limit: int | None = None) -> list[dict]:
        p = self.root / "log.jsonl"
        if not p.exists():
            return []
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-limit:] if limit else rows
