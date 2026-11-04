"""Tidy CSV output (SOP 2.7): one row per item, full precision for Eh."""

from __future__ import annotations

import csv
from pathlib import Path


def fmt(key: str, v: object) -> object:
    """CSV cell: full precision for hartree energies, 6 decimals otherwise."""
    if isinstance(v, float):
        return f"{v:.10f}" if key.endswith("_hartree") else f"{v:.6f}"
    return "" if v is None else v


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows (dicts with equal keys) as CSV."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not rows:
            return
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: fmt(k, v) for k, v in r.items()} for r in rows)
