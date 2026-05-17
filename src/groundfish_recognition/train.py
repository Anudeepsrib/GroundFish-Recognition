"""
YOLO11 training wrapper for GroundFish-Recognition.

Uses the modern Ultralytics Python API (not the old `!yolo` CLI).
Supports dry-run (no GPU, no Roboflow, no training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .paths import ensure_dir, get_experiment_run_dir


def train_yolo11(
    cfg: ExperimentConfig,
    data_yaml: Path,
    *,
    dry_run: bool = False,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Run `model = YOLO(cfg.model_name); model.train(...)`.

    Returns the Ultralytics Results object (or dict in some versions) or None for dry-run.
    """
    if dry_run:
        print("[train] DRY-RUN: would execute")
        print(f"  model     = {cfg.model_name}")
        print(f"  data      = {data_yaml}")
        print(f"  epochs    = {cfg.epochs}")
        print(f"  imgsz     = {cfg.imgsz}")
        print(f"  batch     = {cfg.batch}")
        print(f"  device    = {cfg.device}")
        print(f"  seed      = {cfg.seed}")
        print(f"  project   = {cfg.project}")
        print(f"  name      = {cfg.run_name}")
        print(f"  output    = {cfg.output_dir}")
        return None

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics not installed. pip install -r requirements.txt") from e

    # Ensure output dir exists (Ultralytics will also create it)
    run_dir = get_experiment_run_dir(cfg.experiment_id, cfg.run_name, Path(cfg.output_dir))
    ensure_dir(run_dir.parent)

    print(f"[train] Loading YOLO11 model: {cfg.model_name}")
    model = YOLO(cfg.model_name)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.run_name,
        seed=cfg.seed,
        patience=cfg.patience,
        workers=cfg.workers,
        exist_ok=cfg.exist_ok,
        pretrained=cfg.pretrained,
        verbose=cfg.verbose,
        plots=cfg.plots,
        save=cfg.save,
    )
    if extra_kwargs:
        train_kwargs.update(extra_kwargs)

    print(f"[train] Starting training for {cfg.experiment_id} / {cfg.run_name} ...")
    results = model.train(**train_kwargs)
    print("[train] Training complete.")
    return results


def maybe_resume_or_finetune(
    cfg: ExperimentConfig,
    data_yaml: Path,
    weights_path: Path,
    *,
    dry_run: bool = False,
    finetune_epochs: int | None = None,
) -> dict[str, Any] | None:
    """
    Used by transfer experiments (Exp4/Exp5). Loads previous best.pt and continues.
    """
    if dry_run:
        print(f"[train] DRY-RUN finetune from {weights_path}")
        return None

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    epochs = finetune_epochs or cfg.epochs
    print(f"[train] Fine-tuning from {weights_path} for {epochs} epochs on {data_yaml}")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=f"{cfg.run_name}_finetune",
        seed=cfg.seed,
        exist_ok=cfg.exist_ok,
        verbose=cfg.verbose,
        plots=cfg.plots,
    )
    return results
