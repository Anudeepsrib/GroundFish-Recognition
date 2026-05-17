#!/usr/bin/env python
"""
Aggregate all experiment *_metrics.json files into results/summary.{csv,md}

Example:
    python scripts/summarize_results.py --results-dir results --output results/summary.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundfish_recognition.paths import RESULTS_DIR
from groundfish_recognition.summarize import find_experiment_metrics, write_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--output", default=str(RESULTS_DIR / "summary.md"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_list = find_experiment_metrics(results_dir)
    if not metrics_list:
        print(
            f"No *_metrics.json files found under {results_dir}. Run experiments first (or use fixtures)."
        )
        # Still write an empty summary so CI doesn't fail
        metrics_list = [{"experiment_id": "none", "status": "unavailable"}]

    csv_path = results_dir / "summary.csv"
    md_path = Path(args.output)
    write_summary(metrics_list, csv_path, md_path)
    print("Summary generation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
