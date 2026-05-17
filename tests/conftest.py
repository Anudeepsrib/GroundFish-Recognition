"""Pytest fixtures for GroundFish-Recognition tests (CI-safe, no GPU, no Roboflow)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture
def yolo_results_csv(fixtures_dir: Path) -> Path:
    return fixtures_dir / "yolo_results.csv"


@pytest.fixture
def temp_results_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure no real keys leak into tests."""
    for var in [
        "ROBOFLOW_API_KEY",
        "ROBOFLOW_WORKSPACE_CONVEYOR",
        "ROBOFLOW_PROJECT_CONVEYOR",
        "ROBOFLOW_VERSION_CONVEYOR",
        "ROBOFLOW_WORKSPACE_UNDERWATER",
        "ROBOFLOW_PROJECT_UNDERWATER",
        "ROBOFLOW_VERSION_UNDERWATER",
    ]:
        monkeypatch.delenv(var, raising=False)
    # Provide safe placeholders so .env loading in tests doesn't accidentally read a real .env
    monkeypatch.setenv("ROBOFLOW_API_KEY", "DUMMY_FOR_TESTS_ONLY")
    yield
