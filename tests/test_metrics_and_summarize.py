"""Tests for metrics parser and summary generator using fixture data."""

from __future__ import annotations

import json

from groundfish_recognition.metrics import metrics_to_json, parse_results_csv
from groundfish_recognition.summarize import build_summary_table, write_summary


def test_metrics_parser_on_fixture(yolo_results_csv):
    m = parse_results_csv(yolo_results_csv)
    assert "error" not in m
    assert m["map50"] is not None and 0.5 < m["map50"] < 0.95
    assert m["map50_95"] is not None
    assert m["precision"] is not None
    assert m["recall"] is not None
    assert m["epochs_trained"] >= 10


def test_metrics_to_json_roundtrip(yolo_results_csv, tmp_path):
    m = parse_results_csv(yolo_results_csv)
    out = tmp_path / "metrics.json"
    metrics_to_json(m, out)
    loaded = json.loads(out.read_text())
    assert loaded["map50"] == m["map50"]


def test_summary_handles_missing_and_present(tmp_path, yolo_results_csv):
    # Simulate two experiments: one with data, one missing
    good = parse_results_csv(yolo_results_csv)
    good["experiment_id"] = "exp1"
    good["experiment_name"] = "Exp1 Test"
    good["model"] = "yolo11n.pt"

    missing = {"experiment_id": "exp2", "status": "unavailable"}

    df = build_summary_table([good, missing])
    assert len(df) == 2
    assert set(df["status"]) == {"ok", "unavailable"}

    csv_p = tmp_path / "summary.csv"
    md_p = tmp_path / "summary.md"
    write_summary([good, missing], csv_p, md_p)
    assert csv_p.exists()
    assert "unavailable" in md_p.read_text()
