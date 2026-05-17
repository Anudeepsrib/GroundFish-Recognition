"""
Configuration loader for GroundFish-Recognition experiments.

Merges default.yaml + experiment-specific yaml, validates required fields,
and resolves Roboflow settings from environment when .env is present.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .paths import REPO_ROOT, ensure_dir


@dataclass
class ExperimentConfig:
    # Identity
    experiment_id: str = "unknown"
    experiment_name: str = "Unnamed Experiment"
    description: str = ""

    # Domains
    train_domain: str | None = None
    test_domain: str | None = None
    dataset_mode: str = "single"  # single | mixed | transfer

    # YOLO11
    model_name: str = "yolo11n.pt"
    imgsz: int = 640
    epochs: int = 5
    batch: int = 16
    seed: int = 42
    device: str = "auto"

    # Output
    project: str = "runs"
    run_name: str = "run"
    output_dir: str = "runs"

    # Roboflow (may be None until .env loaded)
    roboflow_workspace: str | None = None
    roboflow_project: str | None = None
    roboflow_version: int | None = None

    # Extra Ultralytics kwargs
    patience: int = 20
    workers: int = 4
    exist_ok: bool = True
    pretrained: bool = True
    verbose: bool = True
    plots: bool = True
    save: bool = True

    notes: str = ""

    # Internal
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_raw", None)
        return d

    def validate(self) -> None:
        """Fail fast on obviously bad configs (used by dry-run and real runs)."""
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.model_name or not self.model_name.endswith(".pt"):
            raise ValueError(f"model_name must be a .pt file (got {self.model_name})")
        if self.imgsz <= 0 or self.epochs <= 0 or self.batch <= 0:
            raise ValueError("imgsz, epochs, batch must be positive")
        if self.dataset_mode not in {"single", "mixed", "transfer"}:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")
        # Roboflow fields are allowed to be None for dry-run


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_config(
    experiment_yaml: str | Path,
    default_yaml: str | Path | None = None,
    load_dotenv_first: bool = True,
) -> ExperimentConfig:
    """
    Load and merge configs.

    Priority (lowest to highest):
        1. configs/default.yaml
        2. the experiment-specific yaml
        3. environment variables (ROBOFLOW_*)
    """
    if load_dotenv_first:
        load_dotenv(REPO_ROOT / ".env", override=False)

    default_path = Path(default_yaml) if default_yaml else (REPO_ROOT / "configs" / "default.yaml")
    exp_path = Path(experiment_yaml)

    defaults = _load_yaml(default_path)
    exp = _load_yaml(exp_path)

    merged = {**defaults, **exp}

    # Resolve Roboflow from env if not explicitly set in yaml
    ws = merged.get("roboflow_workspace")
    proj = merged.get("roboflow_project")
    ver = merged.get("roboflow_version")

    # Map from our .env naming convention when the yaml left them null
    train_domain = merged.get("train_domain")
    if ws is None:
        if train_domain == "conveyor" or merged.get("dataset_mode") == "mixed":
            ws = os.getenv("ROBOFLOW_WORKSPACE_CONVEYOR")
        elif train_domain == "underwater":
            ws = os.getenv("ROBOFLOW_WORKSPACE_UNDERWATER")
        merged["roboflow_workspace"] = ws

    if proj is None:
        if train_domain == "conveyor":
            proj = os.getenv("ROBOFLOW_PROJECT_CONVEYOR")
        elif train_domain == "underwater":
            proj = os.getenv("ROBOFLOW_PROJECT_UNDERWATER")
        elif merged.get("dataset_mode") == "mixed":
            # mixed will be handled specially in datasets.py
            proj = os.getenv("ROBOFLOW_PROJECT_UNDERWATER")  # placeholder
        merged["roboflow_project"] = proj

    if ver is None:
        if train_domain == "conveyor":
            v = os.getenv("ROBOFLOW_VERSION_CONVEYOR")
            merged["roboflow_version"] = int(v) if v else None
        elif train_domain == "underwater":
            v = os.getenv("ROBOFLOW_VERSION_UNDERWATER")
            merged["roboflow_version"] = int(v) if v else None

    # Coerce types
    for k in ("imgsz", "epochs", "batch", "seed", "patience", "workers", "roboflow_version"):
        if k in merged and merged[k] is not None:
            try:
                merged[k] = int(merged[k])
            except (ValueError, TypeError):
                pass

    cfg = ExperimentConfig(**{k: v for k, v in merged.items() if hasattr(ExperimentConfig, k)})
    cfg._raw = merged
    cfg.validate()
    return cfg


def save_config(cfg: ExperimentConfig, path: Path) -> None:
    """Write the effective config to a yaml (useful for audit)."""
    ensure_dir(path.parent)
    d = cfg.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, sort_keys=False)
