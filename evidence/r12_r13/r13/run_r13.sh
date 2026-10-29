#!/bin/bash
# R13 chunk runner (2026-10-29). Each chunk stops after 7000 s; the runner resumes from the saved runs.
PY=/home/claude/wm/runtime/venv/bin/python
cd /home/claude/wm/r13
sha256sum r13_zfull.py r13_analyze.py /home/claude/wm/r12/r12_cohort.json
timeout 7000 $PY -I r13_zfull.py --cohort /home/claude/wm/r12/r12_cohort.json --out /home/claude/wm/r13/out --threads 2
echo "exit code $?"
