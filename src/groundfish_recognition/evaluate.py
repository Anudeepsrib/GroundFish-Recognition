"""
Validation, prediction, and metrics extraction for YOLO11 runs.

Can operate on an existing run directory (post-training) without re-running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .metrics import parse_results_csv


def run_validation(
    cfg: ExperimentConfig,
    data_yaml: Path,
    weights_path: Path,
    *,
    dry_run: bool = False,
    split: str = "val",
) -> dict[str, Any] | None:
    """Run model.val() and return metrics dict."""
    if dry_run:
        print(f"[evaluate] DRY-RUN val on {weights_path} with {data_yaml}")
        return {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "dry_run": True}

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    print(f"[evaluate] Validating {weights_path} on {split} split...")
    results = model.val(
        data=str(data_yaml), split=split, imgsz=cfg.imgsz, batch=cfg.batch, device=cfg.device
    )
    # Ultralytics results have .box.map, .box.map50, etc.
    try:
        metrics = {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
    except Exception:
        metrics = {"raw_results": str(results)}
    return metrics


def run_prediction(
    cfg: ExperimentConfig,
    source: Path,
    weights_path: Path,
    *,
    dry_run: bool = False,
    conf: float = 0.25,
    save: bool = True,
) -> Path | None:
    """Run inference and return the directory where predictions were saved."""
    if dry_run:
        print(f"[evaluate] DRY-RUN predict on {source} using {weights_path}")
        return None

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    print(f"[evaluate] Predicting on {source} (conf={conf}) ...")
    results = model.predict(
        source=str(source), save=save, conf=conf, imgsz=cfg.imgsz, device=cfg.device
    )
    # Return the save dir if possible
    save_dir = getattr(results[0], "save_dir", None) if results else None
    return Path(save_dir) if save_dir else None


def evaluate_existing_run(run_dir: Path, cfg: ExperimentConfig | None = None) -> dict[str, Any]:
    """
    Parse an existing Ultralytics run directory for metrics (no model execution).
    Looks for results.csv and extracts best mAP etc.
    """
    csv = run_dir / "results.csv"
    if not csv.exists():
        # Try common sub-locations
        candidates = list(run_dir.glob("**/results.csv"))
        if candidates:
            csv = candidates[0]
        else:
            return {"error": f"No results.csv found under {run_dir}"}

    metrics = parse_results_csv(csv)
    metrics["run_dir"] = str(run_dir)
    return metrics
