#!/usr/bin/env python
"""
Standalone evaluation script for an existing YOLO11 run.

Example:
    python scripts/evaluate_experiment.py --run-dir runs/exp1/exp1_conveyor_to_underwater --config configs/experiment1.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundfish_recognition.config import load_config
from groundfish_recognition.evaluate import evaluate_existing_run
from groundfish_recognition.metrics import metrics_to_json
from groundfish_recognition.paths import get_results_json_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=None, help="Explicit JSON output path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config) if args.config else None

    metrics = evaluate_existing_run(run_dir, cfg)
    out = (
        Path(args.output)
        if args.output
        else get_results_json_path(metrics.get("experiment_id", run_dir.name))
    )
    metrics_to_json(metrics, out)
    print(f"Metrics -> {out}")
    print(json.dumps(metrics, indent=2, default=str)[:800])
    return 0


if __name__ == "__main__":
    import json

    sys.exit(main())
