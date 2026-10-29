# R10 Protocol LOCK: cross-model replication of VELDROP-z on DINO-WM (Claude, 27/10/2026)

Written before any R10 data exist. Written after the R8 outcomes were opened (R8: VELDROP-z SUPPORTED on JEPA-WM). R10
asks one question: does the label-free VELDROP-z correction, frozen exactly as locked in R8_ADDENDUM_LOCK.md, also
improve fixed-bank selection for a second published model? The model is DINO-WM PushT
(dino_wm_pusht.pth.tar, sha256 8ec9cb05…). It has the same frozen DINOv2 encoder, its own predictor, a Linear(4 -> 20)
proprio encoder, and the same official cost (visual + 0.1 proprio). Nothing about VELDROP-z is re-tuned. Development
grade: the trajectories come from the official train split (seen by the checkpoint), the CEM budget is reduced
(64 x 8), and n is small.

## Cohort
- 16 trajectories of the official PushT train split, disjoint from the R8 cohort. Pre-declared rule:
  `default_rng(39000000).choice(18685, 32, replace=False)`, keep the first 16 that are not R8 trajectories and have
  length >= 31. Result, in this fixed order:
  15720, 1021, 16600, 12404, 15364, 17531, 10308, 16094, 7913, 15537, 10157, 4859, 5725, 14537, 2288, 5586.
- Collector `r10_collect.py` = `r6_collect_v2.py` with only the data path switched to the train split; mode fresh,
  `--model dino --seed 39000000`. Offset, env seed and CEM seed are keyed by `default_rng([39000000, trajectory])`.
  - CEM: DINO-WM CEM 64 x 8, 10 elites, H = 6, official cost.
  - Banks: calibration and evaluation banks of 64 plans each from the final Gaussian.
- CPU cap 7000 s. The cohort is whatever completes, in order (at least 8 for a decision). No start is dropped by
  outcome. Starts after R9 ends.

## Primary
D = margin(VELDROP-z) − margin(B0) on each start's 64-plan evaluation bank (argmin, lowest index on ties). VELDROP-z
uses DINO-WM's own proprio encoder: decoded r = pinv(W)(mean_token(emb) − b), prop_pos = mean_k (W[:, :2](r_xy −
r_xy_goal))_k^2, score = z(vis) + z(prop_pos) within the bank.
- **SUPPORTED** (development) iff mean D <= −0.10 AND VELDROP-z successes >= B0 successes AND mean D without the most
  favourable start <= 0.
- **NOT_SUPPORTED** (for this budget) otherwise.
- **INCOMPLETE**: fewer than 8 starts or a provenance failure.

## Mechanism check M2
REPLICATED iff mean within-bank Spearman(prop_pos, true final agent distance) >= 0.80 AND mean Spearman(official proprio
term, agent distance) <= 0.60. NOT_REPLICATED otherwise.

## Descriptive (none decides)
- H_prop lambda = 10, H_prop-std, POS-only, VELDROP @ 0.1, TPE k = 2, POW q = 8, FIXEDMASK 128.
- E-B0 on encoded true endpoints; O; random plan mean; component errors; block-only criterion.
- The JEPA-trained frozen B1 head applied cross-model (B1, E-B1).

## Analysis
`r10_analyze.py` (hash below) is adapted from the locked `r8_analyze.py`. On the R8 cohort with the JEPA checkpoint it
reproduces every R8 mean margin, success count and mechanism statistic exactly (dry run). Bootstrap over starts:
20,000 resamples, seed 39, descriptive.

## What would refute
- NOT_SUPPORTED: the R8 result is specific to the JEPA-WM predictor at this budget.
- M2 NOT_REPLICATED: the DINO-WM proprio term does not carry the velocity nuisance in the same way.

## Hashes
