#!/bin/bash
# R12 chunk runner (2026-10-28). Each chunk stops after 7000 s; the runner resumes from the saved runs.
# Usage: ./run_r12.sh  (log to /home/claude/wm/logs/r12_chunkN.log)
PY=/home/claude/wm/runtime/venv/bin/python
cd /home/claude/wm/r12
sha256sum r12_frozen_scale.py r12_cohort.json r12_analyze.py
timeout 7000 $PY -I r12_frozen_scale.py --out /home/claude/wm/r12/out --threads 2
echo "exit code $?"
