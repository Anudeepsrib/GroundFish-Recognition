# GroundFish-Recognition

> **YOLO11-based cross-domain groundfish detection research and reproducible MLOps experiment suite.**

This repository studies how well object detection models generalize across two very different visual domains for groundfish (bottom-dwelling fish) recognition:

- **Conveyor Belt** — controlled industrial imaging (clean background, consistent lighting)
- **Underwater** — wild marine environment (turbidity, variable illumination, complex backgrounds)

The work preserves the original five-experiment research design while upgrading the entire stack to modern Ultralytics YOLO11 tooling.

## 🚀 Try the Interactive Demo (No GPU required)

Want to explore the project instantly?

```bash
# 1. Install the base dependencies + Streamlit UI extras
pip install -r requirements.txt -r requirements-ui.txt

# 2. Launch the web app
streamlit run app.py
```

The UI lets you:
- Browse all five cross-domain experiments with their exact configs
- Run the same dry-run validation the CI uses (one click in the sidebar)
- **Live inference demo** — upload any fish photo and see YOLO11 predictions in real time using `yolo11n.pt`, `yolo11s.pt`, or `yolo11m.pt`
- View the experiment matrix and results summary

Perfect for quick demos, interviews, or letting recruiters see the work without cloning notebooks.

![Streamlit UI screenshot](https://via.placeholder.com/800x400?text=GroundFish+YOLO11+Streamlit+Demo)  
*(Replace with a real screenshot after you run it the first time)*

## YOLO11 Upgrade (2026)

This repo was upgraded from YOLOv8-era Colab notebooks to the latest stable Ultralytics workflow:

- **Package**: `ultralytics>=8.4.51`
- **Default smoke-test model**: `yolo11n.pt`
- **Recommended real-experiment models**: `yolo11s.pt` (fast GPU) or `yolo11m.pt` (higher accuracy)
- Full Python API (`YOLO(...).train()`) instead of fragile `!yolo` shell commands
- Config-driven experiments + CLI runners + dry-run mode for CI
- Safe `.env`-based Roboflow handling (no more hardcoded keys)

See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the complete before/after analysis.

---

## Experiment Matrix

| # | Experiment | Train Domain   | Test Domain    | Purpose                                      | Config                    | Notebook                                      | Result Status |
|---|------------|----------------|----------------|----------------------------------------------|---------------------------|-----------------------------------------------|---------------|
| 1 | Cross-Domain Generalization | Conveyor      | Underwater     | Zero-shot domain shift baseline              | `configs/experiment1.yaml` | `Experiment1/...`                             | Ready (dry-run) |
| 2 | Cross-Domain Generalization | Underwater    | Conveyor       | Does harder training help simpler domains?   | `configs/experiment2.yaml` | `Experiment2/...`                             | Ready (dry-run) |
| 3 | Mixed Dataset Training      | Mixed         | Mixed          | Upper-bound performance with full diversity  | `configs/experiment3.yaml` | `Experiment3/...`                             | Ready (dry-run) |
| 4 | Transfer Learning           | Conveyor →    | Underwater     | Pretrain then finetune (source → target)     | `configs/experiment4.yaml` | `Experiment4/...`                             | Ready (dry-run) |
| 5 | Transfer Learning           | Underwater →  | Conveyor       | Reverse transfer direction                   | `configs/experiment5.yaml` | `Experiment5/...`                             | Ready (dry-run) |

---

## Quick Start (Reproducible, No GPU Required for Smoke Test)

```bash
# 1. Clone & create environment (Python 3.11 recommended)
git clone https://github.com/Anudeepsrib/GroundFish-Recognition.git
cd GroundFish-Recognition

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

# 2. (Optional but recommended) Create .env for real Roboflow downloads
cp .env.example .env
# Edit .env with your Roboflow API key + workspace/project/version values

# 3. Smoke test — validates everything without network or GPU
python scripts/run_experiment.py --config configs/experiment1.yaml --dry-run

# 4. Run all five dry-runs (what CI does)
for i in 1 2 3 4 5; do
  python scripts/run_experiment.py --config configs/experiment$i.yaml --dry-run
done

# 5. Generate summary (works even with no real results)
python scripts/summarize_results.py --results-dir results --output results/summary.md
```

### Real GPU Experiment Example

```bash
# After filling .env and having a CUDA GPU
python scripts/run_experiment.py \
  --config configs/experiment1.yaml \
  --train --evaluate \
  --device cuda \
  --model yolo11s.pt \
  --epochs 100
```

> **Tip**: `yolo11n.pt` is intentionally tiny for CI and local smoke tests. Use `yolo11s.pt` or `yolo11m.pt` for any paper-quality numbers.

---

## Project Structure

```
├── configs/                  # YAML-driven experiment definitions
│   ├── default.yaml
│   └── experiment{1..5}.yaml
├── src/groundfish_recognition/
│   ├── config.py             # YAML loader + merge + validation
│   ├── datasets.py           # Safe Roboflow downloader (never prints keys)
│   ├── train.py              # YOLO11 Python API wrapper
│   ├── evaluate.py
│   ├── metrics.py            # results.csv → normalized JSON
│   ├── summarize.py          # Cross-experiment CSV + Markdown
│   └── paths.py              # Centralized, safe output locations
├── scripts/
│   ├── run_experiment.py     # Main CLI (supports --dry-run, --train, --device, overrides)
│   ├── evaluate_experiment.py
│   └── summarize_results.py
├── tests/                    # CI-safe unit tests (no GPU, no secrets)
├── .github/workflows/ci.yml  # Lint + pytest + 5× dry-run on every push/PR
├── Experiment*/              # Original notebooks (upgraded with YOLO11 headers)
├── results/curated/          # Only place for figures you want to keep in git
├── .env.example
├── SECURITY.md
└── AUDIT_REPORT.md
```

---

## Artifact & Security Policy

- **Never committed**: `.env`, `datasets/`, `runs/`, `*.pt`, `*.onnx`, `wandb/`, `logs/`, `__pycache__/`
- **Curated outputs only**: Put final figures/tables you want visible in `results/curated/`
- **Roboflow keys**: Must live in `.env`. Rotate immediately if leaked.
- See [SECURITY.md](SECURITY.md) for full guidance.

---

## Hardware & Runtime Notes

- **Minimum for smoke**: CPU-only Python 3.11 env (dry-run finishes in <10s)
- **Real training**: NVIDIA GPU with CUDA 11.8+ or 12.x recommended. `yolo11s.pt` at 640px / batch 16 fits on 8–12 GB cards.
- Ultralytics will auto-download the requested `yolo11*.pt` weights on first use.
- Training time on a single V100/A100 for 50 epochs with `yolo11s` is typically 30–90 min depending on dataset size.

---

## Reproducibility

Every experiment is fully defined by:
1. Its `configs/experimentN.yaml`
2. The exact Roboflow version pinned in your `.env`
3. `seed: 42` + Ultralytics deterministic flags
4. The pinned `requirements.txt`

Exact mAP numbers will still vary slightly across:
- Different Roboflow dataset versions
- Ultralytics patch releases
- CUDA / cuDNN / PyTorch versions
- GPU architecture

The goal of this repo is **repeatable process**, not bit-identical numbers.

---

## License & Citation

- Code & orchestration: MIT (see [LICENSE](LICENSE))
- Datasets: Subject to the original Roboflow project licenses (you must have access rights)
- YOLO11 weights: Ultralytics license (AGPL-3.0 for open-source use)

If this work helps your research, please cite the original:

```bibtex
@misc{Bathina-GroundFishRecognition,
  title   = {GroundFish Recognition: Cross Database and Transfer Learning Experiments},
  author  = {Bathina, Anudeepsri},
  year    = {2023},
  howpublished = {\url{https://github.com/Anudeepsrib/GroundFish-Recognition}},
}
```

---

## Remaining Manual Actions (Post-Upgrade)

See the final section of [AUDIT_REPORT.md](AUDIT_REPORT.md) for the complete checklist.

---

**Status**: Production-ready for research, portfolio review, and CI-gated experimentation. All P0/P1 issues from the original notebooks have been eliminated.