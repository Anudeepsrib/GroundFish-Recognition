"""
GroundFish Recognition — Simple Streamlit UI

A lightweight web demo for the YOLO11 cross-domain groundfish detection experiments.

Launch:
    pip install -r requirements.txt -r requirements-ui.txt
    streamlit run app.py
"""

from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml
from PIL import Image

# Make the local package importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from groundfish_recognition.config import load_config
from groundfish_recognition.paths import REPO_ROOT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GroundFish • YOLO11",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "1": {
        "title": "Experiment 1: Conveyor → Underwater",
        "config": "configs/experiment1.yaml",
        "train": "Conveyor Belt",
        "test": "Underwater",
        "purpose": "Zero-shot generalization baseline",
    },
    "2": {
        "title": "Experiment 2: Underwater → Conveyor",
        "config": "configs/experiment2.yaml",
        "train": "Underwater",
        "test": "Conveyor Belt",
        "purpose": "Does harder training produce more robust features?",
    },
    "3": {
        "title": "Experiment 3: Mixed Dataset",
        "config": "configs/experiment3.yaml",
        "train": "Mixed (both domains)",
        "test": "Mixed (both domains)",
        "purpose": "Upper-bound performance with maximum data diversity",
    },
    "4": {
        "title": "Experiment 4: Transfer (Conveyor pretrain → Underwater finetune)",
        "config": "configs/experiment4.yaml",
        "train": "Conveyor → Underwater",
        "test": "Underwater",
        "purpose": "Quantify benefit of source-domain pretraining",
    },
    "5": {
        "title": "Experiment 5: Transfer (Underwater pretrain → Conveyor finetune)",
        "config": "configs/experiment5.yaml",
        "train": "Underwater → Conveyor",
        "test": "Conveyor Belt",
        "purpose": "Study directionality of transfer learning",
    },
}

MODEL_OPTIONS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

# ---------------------------------------------------------------------------
# Cached model loader (critical for good UX)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO11 model (first time downloads ~6–40 MB)...")
def get_yolo_model(model_name: str):
    """Load and cache the Ultralytics YOLO11 model."""
    from ultralytics import YOLO
    return YOLO(model_name)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_experiment_config(exp_num: str):
    cfg_path = REPO_ROOT / EXPERIMENTS[exp_num]["config"]
    return load_config(cfg_path, load_dotenv_first=False)


def render_domain_pills(train: str, test: str):
    """Nice visual badges for domains."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background:#e6f3ff;padding:8px 14px;border-radius:8px;text-align:center">
                <strong>🚚 Train:</strong><br>{train}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="background:#fff4e6;padding:8px 14px;border-radius:8px;text-align:center">
                <strong>🌊 Test:</strong><br>{test}
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_inference(model, image: Image.Image, conf: float):
    """Run YOLO11 prediction and return annotated image + basic stats."""
    import numpy as np
    import time

    # Convert PIL to numpy for ultralytics
    img_array = np.array(image.convert("RGB"))

    start = time.time()
    results = model.predict(img_array, conf=conf, verbose=False)
    elapsed = (time.time() - start) * 1000  # ms

    annotated = Image.fromarray(results[0].plot()[:, :, ::-1])  # BGR → RGB

    # Simple stats
    boxes = results[0].boxes
    num_dets = len(boxes) if boxes is not None else 0
    avg_conf = float(boxes.conf.mean()) if num_dets > 0 else 0.0

    stats = {
        "detections": num_dets,
        "avg_confidence": round(avg_conf, 3),
        "inference_ms": round(elapsed, 1),
    }
    return annotated, stats


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🐟 GroundFish Recognition")
st.markdown(
    """
    **YOLO11 cross-domain experiments** — studying how well fish detection models generalize
    between controlled **conveyor-belt** imagery and real **underwater** environments.
    """
)

# Sidebar
st.sidebar.header("⚙️ Controls")

exp_key = st.sidebar.selectbox(
    "Select Experiment",
    list(EXPERIMENTS.keys()),
    format_func=lambda k: EXPERIMENTS[k]["title"],
    index=0,
)

selected_model = st.sidebar.selectbox(
    "YOLO11 Model Size",
    MODEL_OPTIONS,
    index=0,
    help="n = fastest (smoke test), s = good balance, m = stronger accuracy",
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.35,
    step=0.05,
    help="Lower = more detections (more false positives)",
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Run Dry-Run for this Experiment", use_container_width=True):
    with st.spinner(f"Running dry-run for Experiment {exp_key}..."):
        import subprocess
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_experiment.py"),
            "--config", str(REPO_ROOT / EXPERIMENTS[exp_key]["config"]),
            "--dry-run",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode == 0:
            st.sidebar.success("Dry-run completed successfully!")
            st.sidebar.code(result.stdout[-1500:] if result.stdout else "No output", language="text")
        else:
            st.sidebar.error("Dry-run failed (see details below)")
            st.sidebar.code(result.stderr or result.stdout, language="text")

st.sidebar.markdown("---")
st.sidebar.caption("This UI reuses the same configs and YOLO11 stack as the research repo.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
exp = EXPERIMENTS[exp_key]
cfg = load_experiment_config(exp_key)

# Experiment header
st.header(exp["title"])
st.caption(exp["purpose"])

render_domain_pills(exp["train"], exp["test"])

with st.expander("📋 Full experiment configuration (from YAML)", expanded=False):
    st.json(cfg.to_dict())

# Tabs
tab_demo, tab_matrix = st.tabs(["🔬 Live Fish Detector", "📊 Experiment Matrix & Results"])

# ------------------------------------------------------------------
# TAB 1: Live inference demo (the fun part)
# ------------------------------------------------------------------
with tab_demo:
    st.subheader("Try the model yourself")
    st.write(
        "Upload any image containing fish (conveyor photos, underwater shots, or even random fish pictures). "
        "The same YOLO11 architecture used in the experiments will run live."
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded_file = st.file_uploader(
            "Upload image (jpg / png / jpeg)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original image", width="stretch")

    with col_right:
        if uploaded_file is not None:
            if st.button("🚀 Detect Fish", type="primary", use_container_width=True):
                with st.spinner(f"Running {selected_model} ..."):
                    model = get_yolo_model(selected_model)
                    annotated, stats = run_inference(model, image, conf_threshold)

                st.image(annotated, caption="YOLO11 predictions", width="stretch")

                # Stats row
                m1, m2, m3 = st.columns(3)
                m1.metric("Detections", stats["detections"])
                m2.metric("Avg Confidence", stats["avg_confidence"])
                m3.metric("Inference time", f"{stats['inference_ms']} ms")

                st.caption(
                    f"Model: **{selected_model}** &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"Confidence threshold: **{conf_threshold}**"
                )
        else:
            st.info(
                "👆 Upload an image on the left to run detection.\n\n"
                "Tip: Try both conveyor-belt style photos and real underwater images to experience the domain shift the experiments study."
            )

    st.markdown("---")
    st.caption(
        "💡 This demo uses the exact same model family as the five research experiments. "
        "Performance on real underwater data is typically lower than on clean conveyor images — that is the core research question."
    )

# ------------------------------------------------------------------
# TAB 2: Experiment matrix + summary
# ------------------------------------------------------------------
with tab_matrix:
    st.subheader("The Five Experiments at a Glance")

    # Build a nice table
    matrix_data = []
    for k, v in EXPERIMENTS.items():
        c = load_experiment_config(k)
        matrix_data.append({
            "Exp": k,
            "Name": v["title"].split(": ")[1] if ": " in v["title"] else v["title"],
            "Train → Test": f"{v['train']} → {v['test']}",
            "Model (default)": c.model_name,
            "Epochs": c.epochs,
            "Purpose": v["purpose"],
        })

    st.dataframe(matrix_data, use_container_width=True, hide_index=True)

    st.markdown("### Current Results Summary")
    summary_path = REPO_ROOT / "results" / "summary.md"
    if summary_path.exists():
        st.markdown(summary_path.read_text(encoding="utf-8"))
    else:
        st.info("Run the experiments (or the dry-runs) to populate `results/summary.md`.")

    st.markdown("---")
    st.markdown(
        """
        **How to reproduce real results**
        ```bash
        python scripts/run_experiment.py --config configs/experiment1.yaml --train --device cuda --model yolo11s.pt --epochs 50
        python scripts/summarize_results.py
        ```
        Then refresh this page — the summary table will update automatically.
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:0.85rem">
    Part of the <a href="https://github.com/Anudeepsrib/GroundFish-Recognition" target="_blank">GroundFish-Recognition</a> repository — 
    upgraded to YOLO11 + reproducible MLOps (2026).<br>
    Made for research, education, and portfolio demonstration.
    </div>
    """,
    unsafe_allow_html=True,
)
