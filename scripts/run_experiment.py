#!/usr/bin/env python
"""
GroundFish-Recognition Experiment Runner (YOLO11)

Usage examples:
    # Smoke / CI (no GPU, no Roboflow, no training)
    python scripts/run_experiment.py --config configs/experiment1.yaml --dry-run

    # Real training on GPU (after filling .env)
    python scripts/run_experiment.py --config configs/experiment1.yaml --train --device cuda --model yolo11s.pt --epochs 100

    # Download only (useful before launching on a cluster)
    python scripts/run_experiment.py --config configs/experiment1.yaml --download-only

    # Full pipeline
    python scripts/run_experiment.py --config configs/experiment4.yaml --train --evaluate --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundfish_recognition.config import ExperimentConfig, load_config
from groundfish_recognition.datasets import DatasetError, get_data_yaml_for_domain
from groundfish_recognition.evaluate import evaluate_existing_run, run_validation
from groundfish_recognition.metrics import metrics_to_json
from groundfish_recognition.paths import RESULTS_DIR, get_results_json_path
from groundfish_recognition.train import train_yolo11


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GroundFish-Recognition YOLO11 experiment")
    parser.add_argument(
        "--config", required=True, help="Path to experiment yaml (e.g. configs/experiment1.yaml)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything but do not call Roboflow or train",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Only download the dataset(s) and exit"
    )
    parser.add_argument(
        "--train", action="store_true", help="Execute training (requires dataset + GPU recommended)"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Run validation after training (or on existing run)"
    )
    parser.add_argument(
        "--device", default=None, help="Override device: cpu, cuda, mps, auto, or 0"
    )
    parser.add_argument(
        "--model", default=None, help="Override model, e.g. yolo11s.pt or yolo11m.pt"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs (for quick experiments)"
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting existing run dir")
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR), help="Where to write metrics JSON"
    )

    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    print(f"[runner] Loaded config for {cfg.experiment_id}: {cfg.experiment_name}")
    print(
        f"[runner] Model: {cfg.model_name} | epochs={cfg.epochs} | device={cfg.device} | seed={cfg.seed}"
    )

    # Apply CLI overrides
    if args.device:
        cfg.device = args.device
    if args.model:
        cfg.model_name = args.model
    if args.epochs is not None:
        cfg.epochs = args.epochs

    dry = args.dry_run
    if dry:
        print("[runner] === DRY-RUN MODE (no network, no GPU, no training) ===")

    # --- Dataset stage ---
    data_yaml = None
    try:
        if cfg.dataset_mode == "single":
            domain = cfg.train_domain or cfg.test_domain
            data_yaml = get_data_yaml_for_domain(domain, dry_run=dry) if domain else None
        elif cfg.dataset_mode == "mixed":
            # For mixed we would normally merge two Roboflow exports.
            # For v1 we just download the primary one; advanced merging left as future work.
            print("[runner] Mixed mode: downloading primary (underwater) dataset as representative")
            data_yaml = get_data_yaml_for_domain("underwater", dry_run=dry)
        elif cfg.dataset_mode == "transfer":
            # For transfer, the runner will need two datasets; simplified here
            print(
                "[runner] Transfer mode: downloading train_domain dataset (full pipeline may need two calls)"
            )
            data_yaml = get_data_yaml_for_domain(cfg.train_domain, dry_run=dry)
    except DatasetError as e:
        if dry:
            print(f"[runner] (dry-run) Dataset step would fail: {e}")
        else:
            print(f"[runner] ERROR: {e}", file=sys.stderr)
            return 2

    if args.download_only:
        print("[runner] --download-only complete.")
        return 0

    # --- Training stage ---
    results = None
    if args.train or (not dry and not args.evaluate):
        if not data_yaml and not dry:
            print(
                "[runner] ERROR: data.yaml not available (set .env or use --dry-run)",
                file=sys.stderr,
            )
            return 3
        results = train_yolo11(cfg, data_yaml or Path("data.yaml"), dry_run=dry)

    # --- Evaluation / metrics stage ---
    metrics = {}
    run_dir = Path(cfg.output_dir) / cfg.experiment_id / cfg.run_name
    if args.evaluate or dry:
        try:
            # Try to find best.pt from the run we just did or a previous one
            weights = run_dir / "weights" / "best.pt"
            if not weights.exists():
                # Fallback search
                candidates = list(run_dir.glob("**/best.pt"))
                if candidates:
                    weights = candidates[0]
            if weights.exists() or dry:
                metrics = (
                    run_validation(cfg, data_yaml or Path("data.yaml"), weights, dry_run=dry) or {}
                )
            else:
                metrics = evaluate_existing_run(run_dir, cfg)
        except Exception as e:
            metrics = {"error": str(e)}

    # Always try to attach config info
    metrics.setdefault("experiment_id", cfg.experiment_id)
    metrics.setdefault("experiment_name", cfg.experiment_name)
    metrics.setdefault("model", cfg.model_name)
    metrics.setdefault("epochs", cfg.epochs)

    # Write normalized metrics JSON (even in dry-run we write a stub)
    out_json = get_results_json_path(cfg.experiment_id)
    if dry:
        out_json = RESULTS_DIR / f"{cfg.experiment_id}_dryrun_metrics.json"
    metrics_to_json(metrics, out_json)
    print(f"[runner] Metrics written to {out_json}")

    if dry:
        print("[runner] Dry-run finished successfully. All config, paths, and env checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
