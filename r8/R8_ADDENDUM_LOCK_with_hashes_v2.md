# R8 Addendum LOCK + R9 in-loop protocol (Claude, 27/10/2026)

Written and hashed before any R8 outcome was opened. At the time of writing the R8 collector (`r8_collect.py`, launched
under `R8_PROTOCOL_LOCK.md`) has written its first starts; it prints only identities, shapes, provenance checks and
timings. No analysis has read any R8 NPZ. Nothing below changes the R8 primary (frozen B1 vs B0).

## 1. Origin of the addendum candidate (development data only: the 26 R6/R7 starts)

- `r7/dev_decompose.py`: per evaluation bank, the predicted proprio matches the encoded true proprio almost exactly
  (Spearman 0.996), yet the official proprio term ranks plans poorly by the true final agent distance (Spearman 0.39).
- Mechanism (code facts, checked in jepa-wms 13cf1d9c): for PushT the proprio is `[x, y, vx, vy]`
  (`with_velocity=True`, `app/plan_common/datasets/pusht_dset.py`, `__init__.py`: "2 pos + 2 vel"), normalised by fixed
  stats and embedded by `Linear(4 -> 16)` (checkpoint key `proprio_encoder.module.patch_embed`), tiled over 256 tokens.
  Goals are expert frames 30 steps ahead (median agent speed 79 units/s on the dev goals), so the official terminal cost
  asks the plan's final agent VELOCITY to match the goal frame's velocity. The official success criterion ignores velocity.
- `r7/dev_velocity.py` (variants pre-declared in its docstring, all printed): velocity carries 73% of the proprio term on
  the dev evaluation banks; Spearman with the true final agent distance is 0.956 for the position-only term vs 0.398
  for the official proprio term. Selection on the 26 dev starts: official 3.845 (7 successes), VELDROP-z 3.183 (10),
  B1-type signed head under leave-one-trajectory-out CV 3.468 (7), H_prop-std 3.316 (6). On the 10 R6 test starts
  (development by now): VELDROP-z 3.745 (4 successes) vs B0 4.363 (2) and frozen B1 3.604 (4).
- Other dev screens run today, all disclosed and all development (logs in `logs/dev_*.log`): calibrated factor energies
  (`dev_cfe.py`), monotone nonnegative channel heads and two entity-balanced pooling energies (`dev_screen2.py`).
  None is carried forward.
- The idea was formed on these 26 starts, so the dev numbers are optimistic by construction. R8 is the first test.

## 2. Hypothesis A1 (addendum, label-free): VELDROP-z

Score on each plan of the 64-plan evaluation bank:
- `vis` = official visual term (mean squared error over 256 patches x 384 channels);
- `r = pinv(W) (mean_over_tokens(emb) - b)` decodes the embedded proprio with the model's own encoder `(W, b)`;
- `prop_pos = mean_k ( W[:, :2] (r_xy - r_xy_goal) )_k^2`, i.e. the official proprio term with the decoded velocity
  difference set to zero;
- `VELDROP-z = z(vis) + z(prop_pos)`, z-scores within the bank.

No outcome labels are used. The only information beyond the official cost is the model's proprio encoder and the
documented proprio layout (which entries are velocities).

D = margin(VELDROP-z) − margin(B0), argmin with lowest index on ties.
- **SUPPORTED** (development) iff mean D <= −0.10 AND VELDROP-z successes >= B0 successes AND mean D without the single
  most favourable start <= 0.
- **NOT_SUPPORTED** (for this budget) otherwise. Not evidence of zero effect.
- **INCOMPLETE**: fewer than 8 starts or a provenance failure.

## 3. Mechanism check M1

REPLICATED iff, over R8 starts, mean within-bank Spearman(prop_pos, true final agent distance) >= 0.80 AND mean
Spearman(official proprio term, true final agent distance) <= 0.60. NOT_REPLICATED otherwise; INCOMPLETE as above.

## 4. Descriptive (none decides)

VELDROP-z − B1 (bootstrap), paired success gains/losses VELDROP-z vs B1, POS-only, VELDROP@0.1 (vis + 0.1 prop_pos),
component errors, block-only criterion, velocity share of the proprio term, goal-frame agent speed.

## 5. R9: in-loop test on the R8 starts (locked now; run after the R8 analysis)

- Script `r9_inloop.py` (hash below). Starts: every R8 start, in order. Same identities, env seeds and CEM seeds as R8.
- Arms: `veldrop_z` (z-scores within each 64-sample CEM population) and `b1` (frozen R6 head). CEM 64 x 8, 10 elites,
  H = 6, sample 0 = mean, torch generator seeded with the identity's `cem_seed`. The final CEM mean is executed open
  loop (30 primitive actions). B0 = the R8 collector's own official-cost CEM with the same seed, i.e. the stored
  `proposal_mean`, executed; a raw-arm rerun on start 0 must reproduce it exactly (max abs difference 0), otherwise
  R9 is INCOMPLETE.
- Order: start-major; resumable chunks; the R9 cohort is the prefix of starts for which every arm completed. No
  outcome-dependent stopping.
- **R9 primary**: D = executed margin(VELDROP-z CEM) − executed margin(B0 CEM). SUPPORTED iff mean D <= −0.10 AND
  successes not lower AND mean D without the most favourable start <= 0; NOT_SUPPORTED otherwise; INCOMPLETE if fewer
  than 8 starts or a provenance failure.
- **R9 secondary**: the same rule for B1-in-CEM vs B0 (labels SIGNAL / NO_SIGNAL), plus exploitation diagnostics per
  iteration (fraction of negative B1 scores, elite raw cost, elite position term, action magnitudes).
- Analysis script `r9_analyze.py` will be written and hashed before `r9_inloop.json` is opened; it may only implement
  the rules above.

## 6. Multiplicity and what would refute

Pre-specified decisions: R8 primary (B1), A1, M1, R9 primary, R9 secondary. No correction; all reported. A1
NOT_SUPPORTED would say the dev gain does not transfer to new goals at this budget; M1 NOT_REPLICATED would say the
velocity mechanism is specific to the dev goals; R9 NOT_SUPPORTED with A1 SUPPORTED would say the selection gain does
not survive optimisation. Novelty of removing velocity from goal costs is unchecked at the time of this lock.

## Hashes
```
306a179cf0c82d4762bae0e5af57a53620eb4fa7a1b7666bdca717e65a94f898  r8_analyze.py
a9a5fd323dd6dc53dcbf7f377568d1b78957282e0b028d6c6f09dc65c24b54ce  r9_inloop.py
9090e731159401ffa008b2ced2178e14bd7d4203cf80cc087a8a262d7bf5e5ba  R8_PROTOCOL_LOCK.md
a9828d86d16d2e08e1b288ed933673072df5eda8d5225f0d902d0ea4c6753fbe  R8_ADDENDUM_LOCK.md
535f54b636c14b2f83acd1964383b993d423f8f2d8c1865c2f3b78cdefc2f8cb  /home/claude/wm/r7/energies.py
d91418eb225a8dad138eb97e796d068a3797b8f8d12cc0b3bc1ea1f2eeb7fb93  /home/claude/wm/r7/tail_energies.py
5502e1aa79fe97649259ea1817250b06cae0413d11711c59ad7bc0e28982e117  /home/claude/wm/r7/R7_FROZEN_R6_HEAD.json
24bcdec1ac4ee6684bbeab9ccaaa50203d9cae1488935023ba059e78a10b95cd  /home/claude/wm/r7/R7_TPE_LOCK.json
06939c360998e1771648ebbebed41ce9afaa50846eb8b468af343f51af299990  /home/claude/wm/r7/dev_velocity.py
085b6fe04f37cb0040a9908d7aa4aff080e21c44e28f7d62b60e6b5df97377c0  /home/claude/wm/r7/dev_decompose.py
865d0308416b30d50ab27babe7889ec6f6854bb193ec6a9ab467b4fb3d80bb91  /home/claude/wm/r7/dev_screen2.py
66041f9f8126e1e68cbd58bea288f5abed742d6088808a457fee1147d1854732  /home/claude/wm/r7/dev_cfe.py
Tue Oct 27 02:52:24 UTC 2026
```

## Receipt 2 (appended before any R9 output exists)
```
d6285035258bd72ec55f60f905b14f95ecd392bbb3fc2d375f891179fb95157a  r9_analyze.py
Tue Oct 27 02:54:57 UTC 2026
```
