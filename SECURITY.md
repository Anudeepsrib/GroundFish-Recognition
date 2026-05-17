# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security issue (especially accidental commit of Roboflow keys, tokens, or private dataset URLs):

1. **Do not open a public issue.**
2. Email the maintainer or open a private security advisory on GitHub if available.
3. Include: what was exposed, when, and any affected commits.

## Critical Rules for This Repository

### Roboflow API Keys
- **NEVER** commit a real `ROBOFLOW_API_KEY`.
- The original notebooks (pre-2026 upgrade) contained hardcoded keys (`ebMjZPLXM8pNSTq3989b`, `7wxVtgQM8Oz2oO0d7PfU`). These have been **removed** in the YOLO11 upgrade.
- If you ever see a key in the repo history or a PR, **rotate it immediately** in the Roboflow dashboard and open a security advisory.
- All real runs now load keys exclusively via `.env` (loaded by `python-dotenv`). Dry-run mode and CI never require or read the key.

### Dataset Licensing & Access
- All datasets are downloaded from Roboflow on-demand.
- You must have legitimate access rights to the workspace/project/version you configure in `.env`.
- **Do not redistribute** private or licensed Roboflow datasets.
- The authors of this research repo do not grant redistribution rights to third-party datasets.

### Trained Model Weights
- `*.pt`, `*.onnx`, `*.engine` and all `runs/` outputs are gitignored.
- Only share weights if the dataset license + your training run permit redistribution.
- Default policy: weights are **not** committed and are considered research artifacts for the original experimenter only.

### Local Paths & Colab
- All notebooks and scripts have been cleaned of personal Google Drive paths, `/content/`, and absolute Windows/Mac paths.
- If you add new notebooks, use `pathlib.Path` relative to the repo root.

## Safe Development Checklist

- [ ] `cp .env.example .env` (and fill)
- [ ] `git status` shows `.env` is untracked (never add it)
- [ ] Before PR: `git grep -i "api_key\|sk-\|AIza\|roboflow.*[a-z0-9]{20}" -- '*.py' '*.ipynb' '*.yaml' '*.md'` returns nothing
- [ ] `pip-audit` and `ruff` pass in CI

## History Note

This upgrade (2026) was performed specifically to eliminate the P0 key leakage and YOLOv8-era fragility. The audit report (AUDIT_REPORT.md) documents the exact locations that were cleaned.

If you are a reviewer or recruiter: the presence of any key in a fork or PR is a red flag and should be reported.

---

*Maintained as part of the reproducible MLOps upgrade for GroundFish-Recognition.*