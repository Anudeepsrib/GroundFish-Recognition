"""
Metrics parsing utilities for Ultralytics YOLO11 `results.csv`.

Extracts mAP50, mAP50-95, precision, recall from the last (best) row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def parse_results_csv(csv_path: Path) -> dict[str, Any]:
    """
    Read Ultralytics results.csv and return normalized metrics.

    Columns typically include:
      epoch, time, train/box_loss, train/cls_loss, train/dfl_loss,
      metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), ...
    """
    if not csv_path.exists():
        return {"error": f"results.csv not found: {csv_path}"}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"Failed to read {csv_path}: {e}"}

    if df.empty:
        return {"error": "results.csv is empty"}

    # Normalize column names (Ultralytics sometimes uses (B) for boxes)
    cols = {c.strip(): c for c in df.columns}

    def get_col(*candidates: str) -> str | None:
        for c in candidates:
            if c in cols:
                return cols[c]
        return None

    mAP50_col = get_col("metrics/mAP50(B)", "metrics/mAP50", "mAP50")
    mAP_col = get_col("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95")
    prec_col = get_col("metrics/precision(B)", "metrics/precision", "precision")
    rec_col = get_col("metrics/recall(B)", "metrics/recall", "recall")

    last = df.iloc[-1].to_dict()

    def safe_float(key: str | None) -> float | None:
        if key and key in last:
            try:
                return float(last[key])
            except (ValueError, TypeError):
                pass
        return None

    metrics = {
        "map50": safe_float(mAP50_col),
        "map50_95": safe_float(mAP_col),
        "precision": safe_float(prec_col),
        "recall": safe_float(rec_col),
        "epochs_trained": int(last.get("epoch", len(df))) if "epoch" in last else len(df),
        "csv_path": str(csv_path),
    }
    # Also keep the raw best row if user wants everything
    metrics["last_row"] = {
        k: (float(v) if isinstance(v, (int, float)) else v) for k, v in last.items()
    }
    return metrics


def metrics_to_json(metrics: dict[str, Any], path: Path) -> None:
    """Write metrics dict as JSON (used by runners and notebooks)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
