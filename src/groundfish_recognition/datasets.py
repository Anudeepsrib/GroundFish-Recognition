"""
Dataset management for GroundFish-Recognition.

- Loads Roboflow credentials exclusively from environment (never prints keys).
- Supports download-only mode.
- Fails with clear messages when key or workspace is missing for real runs.
- Dry-run paths never call Roboflow.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .paths import DATASETS_DIR, ensure_dir, resolve_data_yaml


class DatasetError(RuntimeError):
    pass


def _require_env(var: str) -> str:
    val = os.getenv(var)
    if not val or val.startswith("your_") or "****" in val:
        raise DatasetError(
            f"Missing or placeholder environment variable: {var}\n"
            f"Copy .env.example to .env and fill in your Roboflow credentials.\n"
            f"Dry-run does not require this variable."
        )
    return val


def get_roboflow_credentials() -> tuple[str, str, str, int | None]:
    """
    Return (api_key, workspace, project, version) from .env.
    Never logs or returns the key in plaintext to callers that might print it.
    """
    load_dotenv(override=False)
    key = _require_env("ROBOFLOW_API_KEY")
    # We do not validate workspace/project here; that happens at download time
    return key, os.getenv("ROBOFLOW_WORKSPACE"), os.getenv("ROBOFLOW_PROJECT"), None


def download_dataset(
    workspace: str | None = None,
    project: str | None = None,
    version: int | None = None,
    *,
    export_format: str = "yolov8",  # still compatible with YOLO11
    location: Path | None = None,
    dry_run: bool = False,
) -> Path | None:
    """
    Download (or return cached) a Roboflow dataset.

    Returns the dataset root directory containing data.yaml, or None in dry-run.
    Raises DatasetError with actionable messages on missing credentials.
    """
    if dry_run:
        print("[datasets] DRY-RUN: skipping Roboflow download")
        return None

    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise DatasetError("roboflow package not installed. pip install roboflow") from e

    key = _require_env("ROBOFLOW_API_KEY")
    ws = workspace or _require_env("ROBOFLOW_WORKSPACE_CONVEYOR")  # fallback
    proj = project or _require_env("ROBOFLOW_PROJECT_CONVEYOR")
    ver = version

    if not ws or not proj:
        raise DatasetError(
            "Roboflow workspace and project must be provided via args or .env "
            "(ROBOFLOW_WORKSPACE_*, ROBOFLOW_PROJECT_*)."
        )

    print(f"[datasets] Connecting to Roboflow workspace='{ws}' project='{proj}' (key hidden)")
    rf = Roboflow(api_key=key)
    project_obj = rf.workspace(ws).project(proj)

    if ver is None:
        # Use latest available version if not pinned
        ver = project_obj.version  # may be int or require .version( )

    print(f"[datasets] Downloading version {ver} in '{export_format}' format...")
    dataset = project_obj.version(ver).download(export_format)

    ds_path = Path(dataset.location)
    ensure_dir(DATASETS_DIR)  # keep our canonical location too
    # Roboflow downloads to cwd or specified; we just report the location
    data_yaml = resolve_data_yaml(ds_path)
    if not data_yaml.exists():
        raise DatasetError(f"Download succeeded but data.yaml not found at {ds_path}")

    print(f"[datasets] Dataset ready: {ds_path}")
    print(f"[datasets] data.yaml: {data_yaml}")
    return ds_path


def get_data_yaml_for_domain(domain: str, dry_run: bool = False) -> Path | None:
    """
    Convenience wrapper used by runners.
    domain: "conveyor" | "underwater"
    """
    if dry_run:
        return None

    ws_var = (
        "ROBOFLOW_WORKSPACE_CONVEYOR" if domain == "conveyor" else "ROBOFLOW_WORKSPACE_UNDERWATER"
    )
    proj_var = (
        "ROBOFLOW_PROJECT_CONVEYOR" if domain == "conveyor" else "ROBOFLOW_PROJECT_UNDERWATER"
    )
    ver_var = "ROBOFLOW_VERSION_CONVEYOR" if domain == "conveyor" else "ROBOFLOW_VERSION_UNDERWATER"

    ws = os.getenv(ws_var)
    proj = os.getenv(proj_var)
    ver = int(os.getenv(ver_var)) if os.getenv(ver_var) else None

    return download_dataset(ws, proj, ver, dry_run=dry_run)
