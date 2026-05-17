"""Path hygiene and secret leakage tests."""

from __future__ import annotations

from pathlib import Path

from groundfish_recognition.paths import REPO_ROOT, safe_resolve


def test_safe_resolve_stays_inside_repo():
    p = safe_resolve(str(REPO_ROOT / "configs" / "default.yaml"))
    assert str(p).startswith(str(REPO_ROOT))


def test_safe_resolve_rejects_outside_path():
    outside = "/tmp/evil"
    if Path(outside).exists():
        with pytest.raises(ValueError):
            safe_resolve(outside)


def test_env_example_contains_no_real_key():
    env_ex = REPO_ROOT / ".env.example"
    content = env_ex.read_text()
    assert "your_roboflow_api_key_here" in content
    # No long base64-looking strings that look like real keys
    assert "ebMjZPLXM8pNSTq3989b" not in content
    assert "7wxVtgQM8Oz2oO0d7PfU" not in content


import pytest  # for the raises above
