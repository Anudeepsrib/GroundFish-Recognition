"""
Centralized path management for GroundFish-Recognition.

Prevents accidental writes outside repo-controlled directories.
All training outputs, downloads, and results must go through here.
"""

from __future__ import annotations

from pathlib import Path

# Repo root (this file is in src/groundfish_recognition/paths.py)
# parents[0]=paths.py, [1]=groundfish_recognition/, [2]=src/, [3]=repo root -> use parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard directories (created on demand)
DATASETS_DIR = REPO_ROOT / "datasets"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
CURATED_DIR = RESULTS_DIR / "curated"
REPORTS_DIR = REPO_ROOT / "reports"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

# Roboflow cache (ignored)
ROBOFLOW_CACHE = REPO_ROOT / "roboflow"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if missing. Return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_experiment_run_dir(experiment_id: str, run_name: str, base: Path | None = None) -> Path:
    """Return (and create) the Ultralytics project/run directory for an experiment."""
    base = base or RUNS_DIR
    run_dir = ensure_dir(base / experiment_id / run_name)
    return run_dir


def get_results_json_path(experiment_id: str) -> Path:
    """Normalized metrics JSON written after evaluate or summarize."""
    ensure_dir(RESULTS_DIR)
    return RESULTS_DIR / f"{experiment_id}_metrics.json"


def get_curated_path(filename: str) -> Path:
    """Path inside results/curated/ for final figures/tables."""
    ensure_dir(CURATED_DIR)
    return CURATED_DIR / filename


def resolve_data_yaml(dataset_location: Path) -> Path:
    """Given a Roboflow download folder, return the data.yaml path."""
    p = dataset_location / "data.yaml"
    if not p.exists():
        # Some Roboflow exports put it one level deeper
        candidates = list(dataset_location.glob("**/data.yaml"))
        if candidates:
            return candidates[0]
    return p


def safe_resolve(path_str: str) -> Path:
    """Resolve a user-supplied path and ensure it stays inside the repo root."""
    p = Path(path_str).resolve()
    if not str(p).startswith(str(REPO_ROOT)):
        raise ValueError(f"Refusing to operate outside repo root: {p}")
    return p


# For scripts that want a quick "where is everything"
def print_path_summary() -> None:
    print(f"Repo root:     {REPO_ROOT}")
    print(f"Datasets:      {DATASETS_DIR}")
    print(f"Runs:          {RUNS_DIR}")
    print(f"Results:       {RESULTS_DIR}")
    print(f"Curated:       {CURATED_DIR}")
