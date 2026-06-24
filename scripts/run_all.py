#!/usr/bin/env python3
"""Run the whole analysis pipeline end-to-end (assumes data/dataset.csv already built).

    micromamba run -n dist-shift-diagnosis python scripts/run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ("diagnosis numbers", "scripts/run_baseline.py"),
    ("figures", "scripts/make_figures.py"),
    ("monitor smoke + figure", "scripts/run_monitor.py"),
    ("0DTE attribution", "scripts/run_attribution.py"),
]
if os.path.exists(os.path.join(HERE, "data", "dataset_qqq.csv")):
    STEPS.append(("QQQ replication", "scripts/run_qqq.py"))

if not os.path.exists(os.path.join(HERE, "data", "dataset.csv")):
    sys.exit("data/dataset.csv missing — build it first:\n"
             "  micromamba run -n intraday-momentum python scripts/build_dataset.py")

for label, script in STEPS:
    print(f"\n{'='*70}\n# {label}  ({script})\n{'='*70}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)])
    if r.returncode != 0:
        sys.exit(f"step failed: {script}")

print(f"\n{'='*70}\nALL DONE. Re-render the report with:\n"
      "  cd reports && pandoc report.md -o report.html --standalone --embed-resources "
      "--css style.css --metadata title=\"Model-Reliability Monitor\"\n"
      "Launch the dashboard with:  streamlit run app.py")
