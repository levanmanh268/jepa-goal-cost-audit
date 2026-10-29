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
```
f29544790c76bf51044757e553f355c099252e0b61514b8037cf576a47d0c383  r10_collect.py
78847b049a50baa8b7efaf25a46765c6a772db8048f23865518acca58e4fa227  r10_analyze.py
edf8e6951cc7df245f49cfad2f0c2ec0ba692eb0be7b8d256267cdabf94fcc29  run_r10.sh
2d4c2baa863933c54e41b6d7bde859736132c67620b59cbadf60c168f49bba74  R10_PROTOCOL_LOCK.md
39b342acba94edfab9b9966d27cd807c5b40da809893824b3fd9a026ba5a1a9a  ../r8/train_extract_sha256.json
8ec9cb05f22812d7f12e3c216b0637f41641055c0653e503e2746edb981b550f  /home/claude/wm/runtime/checkpoints/dino_wm_pusht.pth.tar
5502e1aa79fe97649259ea1817250b06cae0413d11711c59ad7bc0e28982e117  /home/claude/wm/r7/R7_FROZEN_R6_HEAD.json
24bcdec1ac4ee6684bbeab9ccaaa50203d9cae1488935023ba059e78a10b95cd  /home/claude/wm/r7/R7_TPE_LOCK.json
Tue Oct 27 04:57:46 UTC 2026
```

# R10 Amendment 1 (Claude, 27/10/2026), written before any R10 data exist

A DINO-WM smoke test on this CPU (scoring 4 plans took 8.5 s on one contended thread) is consistent with the R5
measurement of about 22 minutes per DINO-WM start. The R10 cap of 7000 s would then yield about 5 starts, below the
8-start minimum, i.e. a guaranteed INCOMPLETE. Change, decided before any R10 data and independent of any outcome:

- The R10 cohort is fixed at the **first 8 starts of the locked order** (15720, 1021, 16600, 12404, 15364, 17531,
  10308, 16094). Collection runs in resumable chunks of at most 7000 s and stops as soon as 8 starts are recorded
  (count-based). The collector command, seed and episode list are unchanged, so the provenance constants of
  `r10_analyze.py` still apply.
- The decision rules, the analysis script and the mechanism check are unchanged. With n = 8 the primary decision is
  weak; M2 (within-bank Spearman over 64 plans per start) is the more precise part.
- New hash of the launcher `run_r10.sh` below; all other hashes as in R10_PROTOCOL_LOCK.md.
```
aa23315ae3afa67128fa7edb23083afa10adbce24da42c43b0ed02faaf1c4f92  run_r10.sh
c24ddf8c1ce938b68845fd2bbd46dbc4a410b449e773f367ba82ecf6fa0522bd  R10_AMENDMENT_1.md
f29544790c76bf51044757e553f355c099252e0b61514b8037cf576a47d0c383  r10_collect.py
78847b049a50baa8b7efaf25a46765c6a772db8048f23865518acca58e4fa227  r10_analyze.py
Tue Oct 27 05:00:15 UTC 2026
```
