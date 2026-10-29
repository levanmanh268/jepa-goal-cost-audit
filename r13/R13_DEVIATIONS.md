# R13 deviations

- **Container restart (infrastructure), 29 Oct 2026, about 00:05–00:07 UTC.**
  - The session's container restarted during chunk 1 and killed the run in progress.
  - Four runs (starts 0–3) had been saved atomically. Start 4, which was in progress, was lost and will be rerun
    from the start.
  - Code, lock hashes and the cohort are unchanged (checked against `lock_hashes_R13.txt` after the restart). The runner
    resumes from the saved runs after its config, code and identity checks.
  - Runs are deterministic given the seed, so the rerun of start 4 is the same computation.
  - No outcome was inspected; the logs print timing and α0 only.
