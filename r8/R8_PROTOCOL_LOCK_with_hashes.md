# R8 Protocol LOCK: new goals from the official TRAIN split (Claude, 27/10/2026)

Locked before any R8 data exist. Purpose: the first evaluation of the R6/R7 scores on start/goal pairs that no
score, head or hyperparameter has seen. Development-grade, not confirmation: the trajectories come from the split the
JEPA-WM checkpoint was trained on (seen by the checkpoint, unseen by every score), the CEM budget is reduced (64 x 8)
and n is small.

## Cohort

- Source: official PushT `pusht_noise/train` (18,685 trajectories), fetched by HTTP range from the DINO-WM OSF archive;
  SHA256 in `train_extract_sha256.json`. No val trajectory's first state appears in the train split (checked).
- Trajectories in this fixed order (numpy `default_rng(38000000).choice(18685, 16, replace=False)`):
  3635, 2466, 3108, 16642, 3665, 18150, 15360, 12440, 1941, 10673, 12946, 4828, 6331, 10136, 13906, 10710.
- Collector: `r8_collect.py`, which is `r6_collect.py` with only the data path changed. Mode fresh, `--seed 38000000`:
  - offset, env seed and CEM seed are keyed by `default_rng([38000000, trajectory])`;
  - JEPA-WM CEM 64 x 8, 10 elites, H = 6, official cost;
  - calibration and evaluation banks of 64 plans each from the final Gaussian, with streams [38000000, k, 1/2].
- CPU cap 2 h for collection. Starts are processed in the order above; the cohort is whatever completes within the
  cap (at least 8 for a decision). No start is ever dropped by outcome. The collector prints no outcomes.

## Primary (replication of R6-A on unseen goals)

D = margin(B1) − margin(B0) on each start's 64-plan evaluation bank (argmin, lowest index on ties). B1 is the frozen
R6 head (`R7_FROZEN_R6_HEAD.json`, no refit); B0 is the exact official cost.
- REPLICATED (development): mean D <= −0.10 AND B1 successes >= B0 successes AND mean D without the single most
  favourable start <= 0.
- NOT_REPLICATED (for this budget): otherwise. This is not evidence of zero effect.
- INCOMPLETE: fewer than 8 starts or a provenance failure.

## Secondary (frozen, all reported, none decides)

H_prop lambda = 10; H_prop-std (z-scored vis + proprio within the bank); TPE k = 2 and POW q = 8 (R7, NO_SIGNAL on
the R6 test banks); FIXEDMASK 128; encoded-true E-B0 and E-B1; best in bank O; random plan mean. Also reported:
component errors (agent position, block position, block angle) of each selected plan, the official success, and a
block-only criterion (block within 20 units and angle within 20 degrees). All secondary rows are descriptive.

## Addendum slot

A new label-free candidate may be added as a SEPARATE pre-registered hypothesis in `R8_ADDENDUM_LOCK.md`. That
addendum must be written to the project before anyone opens any R8 outcome. The candidate's design may use only the
26 development starts of R6/R7.

## Analysis

`r8_analyze.py` is hashed into `lock_hashes_R8.txt` before any R8 outcome is opened. Bootstrap over starts: 20,000
resamples, seed 38, descriptive only. Exact paired success gains/losses are reported.

```
9090e731159401ffa008b2ced2178e14bd7d4203cf80cc087a8a262d7bf5e5ba  R8_PROTOCOL_LOCK.md
6c6a2e858b5e3bbce4b2d85aaeb272583307d929fe12cfe827e60415db72d588  r8_collect.py
2d290f5ce1cadad410a32a88ad259708345ccc957e88af313c27cd58a3b25217  /home/claude/wm/r6/r6_collect.py
39b342acba94edfab9b9966d27cd807c5b40da809893824b3fd9a026ba5a1a9a  train_extract_sha256.json
5502e1aa79fe97649259ea1817250b06cae0413d11711c59ad7bc0e28982e117  /home/claude/wm/r7/R7_FROZEN_R6_HEAD.json
24bcdec1ac4ee6684bbeab9ccaaa50203d9cae1488935023ba059e78a10b95cd  /home/claude/wm/r7/R7_TPE_LOCK.json
Tue Oct 27 02:32:04 UTC 2026
```
