"""Artifact loader for the UI.

Reads the persisted product artifacts without re-running any models. If the
artifacts are missing, :func:`load_product` raises a clear error naming the
exact build command, and :func:`ensure_artifacts` can auto-build them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fas.product import ARTIFACT_FILES

BUILD_COMMAND = "python -m fas.cli product-build --no-download"


@dataclass(slots=True)
class Product:
    """In-memory bundle of all product artifacts."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables[name]

    def get(self, name: str, default=None):
        return self.tables.get(name, default)


def artifacts_present(data_root: str | Path = "data") -> bool:
    out = Path(data_root) / "processed"
    return all((out / f).exists() for f in ARTIFACT_FILES)


def ensure_artifacts(data_root: str | Path = "data", *, allow_download: bool = False,
                     seed: int = 7, verbose: bool = False) -> None:
    """Build artifacts if any are missing (used by the UI launch path)."""
    if artifacts_present(data_root):
        return
    from fas.product.build import product_build

    product_build(data_root=data_root, allow_download=allow_download,
                  seed=seed, verbose=verbose)


def load_product(data_root: str | Path = "data") -> Product:
    """Load all artifacts into a :class:`Product`. Raises if not built yet."""
    out = Path(data_root) / "processed"
    missing = [f for f in ARTIFACT_FILES if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"product artifacts not found in {out.resolve()} (missing {missing}). "
            f"Build them first:\n    {BUILD_COMMAND}")

    tables: dict[str, pd.DataFrame] = {}
    for f in ARTIFACT_FILES:
        if f.endswith(".parquet"):
            df = pd.read_parquet(out / f)
            if list(df.columns) == ["_empty"]:
                df = df.iloc[0:0]
            tables[f[:-len(".parquet")]] = df
    summary = json.loads((out / "product_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return Product(tables=tables, summary=summary, manifest=manifest)
