#!/usr/bin/env python
"""
One-time notebook sanitizer and YOLO11 upgrader for GroundFish-Recognition.

- Removes / masks all Roboflow API key cells
- Replaces YOLOv8 references and yolov8l-seg.pt with YOLO11 + Python API examples
- Adds standard header (Experiment purpose, YOLO11, env vars, expected outputs)
- Adds standard footer (save metrics JSON + curated figures note)
- Replaces Colab-specific download helpers with warnings
- Does NOT execute any training cells

Run once:
    python scripts/upgrade_notebooks.py
"""

import json
import re
from pathlib import Path

NOTEBOOKS = [
    (
        "Experiment1",
        "1_Cross_Database_Generalization_from_Conveyor_Belt_to_Underwater_Environment.ipynb",
        "exp1",
    ),
    (
        "Experiment2",
        "2_Assessment_of_Model_Performance_from_Underwater_to_Conveyor_Belt_Environment.ipynb",
        "exp2",
    ),
    (
        "Experiment3",
        "3_Integrated_Approach_with_a_Mixed_Dataset_(Conventional_Training_and_Testing).ipynb",
        "exp3",
    ),
    (
        "Experiment4",
        "4_Leveraging_Transfer_Learning_from_Conveyor_Belt_to_Underwater_Dataset.ipynb",
        "exp4",
    ),
    (
        "Experiment5",
        "5_Transfer_Learning_from_Underwater_to_Conveyor_Belt_Environment.ipynb",
        "exp5",
    ),
]

HEADER_MD = """# YOLO11 Upgrade Notice (2026)

**This notebook was modernized** from the original YOLOv8-era Colab workflow to the current Ultralytics YOLO11 stack.

- **Model family**: YOLO11 (`yolo11n.pt` default for smoke tests; `yolo11s.pt` / `yolo11m.pt` recommended for real experiments)
- **Training API**: `from ultralytics import YOLO; model = YOLO(...); model.train(...)` (Python, not `!yolo` CLI)
- **Secrets**: All Roboflow keys have been **removed**. Use `.env` + `python-dotenv`.
- **Reproducibility**: See `configs/{experiment}.yaml` and `python scripts/run_experiment.py --config ...`

> **Do not commit** any `.env` or run outputs. This notebook is for reference and light local execution only.
> For serious runs use the script runner (supports `--dry-run`, `--device cuda`, etc.).

## Required Environment
```bash
cp .env.example .env   # fill your Roboflow credentials
pip install -r requirements.txt
```

## Expected Outputs
- `runs/{exp_id}/.../results.csv`
- `results/{exp_id}_metrics.json`
- Curated figures in `results/curated/`

Run the matching config with:
```bash
python scripts/run_experiment.py --config configs/{exp_config}.yaml --dry-run
```
"""

FOOTER_MD = """
## Notebook Footer - Save Results (YOLO11)

After you finish training / validation in this notebook, run the cell below to persist metrics for the summary generator.

```python
# === YOLO11 NOTEBOOK FOOTER ===
from pathlib import Path
import json
from datetime import datetime

# Adjust these to match the run you just executed
experiment_id = "{exp_id}"
metrics = {
    "experiment_id": experiment_id,
    "model": "yolo11n.pt",   # or whatever you actually used
    "map50": 0.0,            # fill from your results.csv or results.box.map50
    "map50_95": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "epochs_trained": 0,
    "note": "Manual extraction from Ultralytics run. Prefer the script runner for automated JSON."
}

out = Path("results") / f"{experiment_id}_metrics.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {out}")

# Copy key figures you want to keep long-term into results/curated/
# (the rest of runs/ is gitignored)
print("Copy important PNGs from your run directory into results/curated/ for the portfolio.")
```
"""


def sanitize_source(src: str) -> str:
    # Remove real keys
    src = re.sub(
        r'Roboflow\(api_key="[^"]+"',
        'Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY")  # from .env',
        src,
    )
    src = re.sub(
        r"Roboflow\(api_key=\'[^\']+\'",
        "Roboflow(api_key=os.getenv('ROBOFLOW_API_KEY')  # from .env",
        src,
    )
    # Update model references
    src = src.replace("yolov8l-seg.pt", "yolo11n.pt  # upgraded from yolov8l-seg.pt")
    src = re.sub(
        r'download\("yolov8"\)',
        'download("yolov8")  # still works for YOLO11; consider "ultralytics"',
        src,
    )
    # Warn on Colab
    src = src.replace(
        "from google.colab import files",
        "# from google.colab import files  # REMOVED - not portable. Use results/curated/ instead.",
    )
    src = src.replace("/content/runs", "runs  # was /content/runs - now relative to repo")
    return src


def upgrade_notebook(nb_dir: str, nb_name: str, exp_id: str, exp_config: str) -> None:
    nb_path = Path(nb_dir) / nb_name
    print(f"Upgrading {nb_path} ...")
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Insert header as first markdown cell (after any existing title if present)
    header_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": HEADER_MD.replace("{exp_config}", f"experiment{exp_id[3]}.yaml")
        .replace("{exp_id}", exp_id)
        .splitlines(keepends=True),
    }
    # Put header early (index 0 or 1)
    nb["cells"].insert(0, header_cell)

    # 2. Sanitize every code cell
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            new_src = sanitize_source(src)
            # Remove entire cells that only contained key setup (they will be empty-ish after sanitize)
            if "Roboflow(api_key=" in src and "os.getenv" not in new_src:
                # Replace the cell content with a safe comment
                new_src = "# Roboflow key cell removed during YOLO11 upgrade.\n# Use .env + load_dotenv() + os.getenv('ROBOFLOW_API_KEY')\n"
            cell["source"] = [
                line + "\n" if not line.endswith("\n") else line for line in new_src.splitlines()
            ]

    # 3. Append footer as last cell
    footer_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": FOOTER_MD.replace("{exp_id}", exp_id).splitlines(keepends=True),
    }
    nb["cells"].append(footer_cell)

    # 4. Update notebook metadata
    nb.setdefault("metadata", {})
    nb["metadata"]["ultralytics_upgrade"] = "2026-yolo11"
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }

    # Write back (overwrite)
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  -> {nb_path} upgraded (keys removed, YOLO11 header/footer added)")


if __name__ == "__main__":
    for d, name, eid in NOTEBOOKS:
        upgrade_notebook(d, name, eid, eid)
    print("\nAll notebooks upgraded. Review the diffs before committing.")
