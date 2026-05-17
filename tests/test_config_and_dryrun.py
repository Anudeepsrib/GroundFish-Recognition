"""Tests for config loading, validation, and dry-run safety (no GPU / no key required)."""

from __future__ import annotations

import pytest

from groundfish_recognition.config import ExperimentConfig, load_config
from groundfish_recognition.paths import REPO_ROOT


def test_all_five_experiment_configs_exist():
    for i in range(1, 6):
        p = REPO_ROOT / "configs" / f"experiment{i}.yaml"
        assert p.exists(), f"Missing {p}"


def test_default_config_loads_and_validates():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", load_dotenv_first=False)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model_name == "yolo11n.pt"
    assert cfg.seed == 42
    cfg.validate()


def test_experiment1_config_merges_and_has_correct_domains():
    cfg = load_config(REPO_ROOT / "configs" / "experiment1.yaml", load_dotenv_first=False)
    assert cfg.experiment_id == "exp1"
    assert cfg.train_domain == "conveyor"
    assert cfg.test_domain == "underwater"
    assert "Conveyor Belt" in cfg.description or "conveyor" in cfg.description.lower()


def test_missing_required_field_fails():
    bad = {"experiment_id": "", "experiment_name": "", "model_name": "yolo11n.pt", "epochs": 1}
    with pytest.raises(ValueError):
        c = ExperimentConfig(**bad)
        c.validate()


def test_dry_run_does_not_require_real_roboflow_key(monkeypatch):
    """The most important CI safety test."""
    from groundfish_recognition.datasets import DatasetError, _require_env

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(DatasetError):
        _require_env("ROBOFLOW_API_KEY")  # real path would fail

    # But dry-run code paths must never call _require_env
    # (we test the runner entrypoint below)


def test_dry_run_runner_entrypoint_does_not_need_key_or_gpu(repo_root, tmp_path, monkeypatch):
    """End-to-end: run_experiment.py --dry-run must succeed with zero secrets and zero GPU."""
    import subprocess
    import sys

    env = os.environ.copy()
    # Ensure no key
    for k in list(env.keys()):
        if "ROBOFLOW" in k:
            del env[k]

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_experiment.py"),
        "--config",
        str(repo_root / "configs" / "experiment1.yaml"),
        "--dry-run",
        "--results-dir",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"dry-run failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "DRY-RUN MODE" in proc.stdout
    assert "Dry-run finished successfully" in proc.stdout


import os  # needed for the env copy in the test above
