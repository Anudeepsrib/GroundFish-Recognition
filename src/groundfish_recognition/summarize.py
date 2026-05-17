"""
Aggregate metrics across all experiments into summary.csv and summary.md.

Tolerates missing experiment results (marks them "unavailable").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import RESULTS_DIR, ensure_dir


def find_experiment_metrics(results_dir: Path = RESULTS_DIR) -> list[dict[str, Any]]:
    """Scan for * _metrics.json files written by runners/notebooks."""
    items = []
    for p in sorted(results_dir.glob("*_metrics.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_source_file"] = str(p.name)
            items.append(data)
        except Exception as e:
            items.append({"experiment_id": p.stem.replace("_metrics", ""), "error": str(e)})
    return items


def build_summary_table(metrics_list: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for m in metrics_list:
        exp = m.get("experiment_id") or m.get("exp_id") or "unknown"
        rows.append(
            {
                "experiment_id": exp,
                "experiment_name": m.get("experiment_name", exp),
                "model": m.get("model", m.get("model_name", "")),
                "map50": m.get("map50"),
                "map50_95": m.get("map50_95"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "epochs": m.get("epochs_trained") or m.get("epochs"),
                "status": "ok" if "map50" in m and m.get("map50") is not None else "unavailable",
                "source": m.get("_source_file", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=["experiment_id", "status", "map50", "map50_95", "precision", "recall"]
        )
    return df


def write_summary(
    metrics_list: list[dict[str, Any]],
    output_csv: Path,
    output_md: Path,
) -> None:
    df = build_summary_table(metrics_list)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)

    # Markdown
    lines = [
        "# GroundFish-Recognition Experiment Summary (YOLO11)",
        "",
        f"Generated from {len(df)} experiment result files.",
        "",
        "| Experiment | Model | mAP50 | mAP50-95 | Precision | Recall | Epochs | Status |",
        "|------------|-------|-------|----------|-----------|--------|--------|--------|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['experiment_id']} | {r.get('model','')} | "
            f"{r.get('map50','')} | {r.get('map50_95','')} | "
            f"{r.get('precision','')} | {r.get('recall','')} | "
            f"{r.get('epochs','')} | {r.get('status','')} |"
        )
    lines += [
        "",
        "> Note: `unavailable` means the experiment has not been run or its metrics JSON was not found.",
        "> Exact numbers vary with dataset version, Ultralytics release, GPU, seed, and batch size.",
    ]
    ensure_dir(output_md.parent)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[summarize] Wrote {output_csv} and {output_md}")
