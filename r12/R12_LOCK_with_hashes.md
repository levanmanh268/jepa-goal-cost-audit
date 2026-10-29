# R12 LOCK: full versus position-only at one frozen weight, on a fresh cohort (28/10/2026)

Status: locked before any R12 cohort outcome exists. The only model runs before this lock are infrastructure checks on
old data (R8 start 0): a regression of the official-cost CEM and a 2-iteration smoke test of the two fixed arms.
The protocol follows ChatGPT's `R12_FROZEN_SCALE_PROTOCOL.md`, with one pre-declared secondary question added (Q2).
This is exploratory development evidence, not confirmation.

## Questions and estimands
- **Q1 (primary).** On a fresh cohort, does keeping only the position energy improve the executed margin when both arms
  use the same weight α0, frozen for the whole CEM run?
  - D = margin(fixed_pos) − margin(fixed_full), averaged over starts.
  - This changes the retained quantity, which also drops the cross, residual and token-variation terms. It is not a
    pure-velocity causal effect.
- **Q2 (pre-declared secondary).** On the same fresh cohort, D = margin(veldrop_z) − margin(raw). This is a fresh-cohort
  repeat of the R9 in-search contrast (VELDROP-z against the official cost).
- **Descriptive only:** fixed_full − raw, fixed_pos − raw, veldrop_z − fixed_pos, veldrop_z − fixed_full; the α0
  distribution; per-arm agent, block and angle errors.

## Cohort (`make_r12_cohort.py` → `r12_cohort.json`)
- 32 PushT trajectories from the training split.
  - Drawn with `default_rng(42000000).choice(18685, 128, replace=False)`, keeping the first 32 that are not in the R8
    list or the R10 list (32 trajectories excluded).
  - The script asserts this, and re-derives the R8 list from its own seed.
- Offset, env seed and CEM seed are keyed by `default_rng([42000000, trajectory])`, exactly as `r8_collect.py` does.
- Label: fresh evaluation cohort within the checkpoint's training split. It is not held out from pretraining and not
  out of distribution.

## Planner and arms (`r12_frozen_scale.py`)
- Same JEPA-WM checkpoint and runtime as R8 to R11.
- CEM settings: 64 samples, 8 iterations, 10 elites, horizon 6 macro-actions of dimension 10. Candidate 0 is the mean.
  One keyed generator per start (its cem_seed), so the first population Q0 is identical in every arm.
- The final mean is executed open loop with the release criterion. One CEM seed per start.
- α0 = (sd(L_vis(Q0)) + 1e-12) / (sd(Π_full(Q0)) + 1e-12), with population sd (ddof = 0).
  - Computed once from Q0 and frozen for all iterations.
  - It is never re-estimated from later populations, from one arm, or from outcomes.
- The arms:

  | Arm | Cost | Question |
  |---|---|---|
  | fixed_full | L_vis + α0·Π_full | Q1 |
  | fixed_pos | L_vis + α0·Π_pos | Q1 |
  | raw | the model's official cost | Q2 |
  | veldrop_z | z(L_vis) + z(Π_pos) | Q2 |

- Each run logs: α0, both sd values, the sha256 of Q0, per-iteration sds and elite means, the executed plan's predicted
  terms, and timing.
- Order: the Q1 arms for all 32 starts, then the Q2 arms for all 32 starts.
- Execution is in chunks of at most 7000 s, with atomic per-run saves and resume after config, code and identity checks.
- Cohort-mode logs print timing and α0 only. Outcomes are not looked at until the analysis runs: before then, the
  results file is only counted by keys.

## Analysis (`r12_analyze.py`), run once
- **When it runs:** after all 128 runs exist, or at the paper freeze if collection is unfinished (that question is then
  labelled INCOMPLETE, as below).
- **Integrity checks:** margins and successes recomputed from final_state; α0 and Q0 identical across a start's arms;
  α0 constant across iterations.
- **Per question:** all 32 paired differences, mean and median, and a percentile bootstrap 95% CI.
  - The bootstrap uses one index matrix, `default_rng(12001).integers(0, 32, (20000, 32))`, shared by all contrasts.
  - Also reported: improved/worsened/unchanged counts, paired success gains and losses, exact two-sided McNemar test,
    two-sided sign test, mean without the most favourable start, and the number of plan changes.
- **Operational gate (priority rule, not a type-I-error-controlled test):** mean D ≤ −0.10 AND successes(new) ≥
  successes(reference) AND mean D without the most favourable start ≤ 0.
  - Q1: POSITION_HELPS or NOT_SHOWN.
  - Q2: VELDROP_Z_HELPS or NOT_SHOWN.
  - INCOMPLETE if any of the 32 pairs is missing; available pairs are shown but not gated.
- **Interpretation:**
  - A CI containing 0 is uncertain, neither "confirmed" nor "no effect".
  - Success changes are reported separately and not inferred from the margin CI.
  - n = 32 is an exploratory budget, not a power calculation.
  - No extra starts, seeds, α values or subsets will be added after seeing results.

## Pre-lock checks (old data only: R8 start 0 and R11 process logs)
- **Regression.** The official-cost CEM on R8 start 0 reproduced the stored R8 proposal mean exactly (max abs 0.0) in
  457.8 s (`checks/r12_regression.json`).
- **Smoke test** (2 iterations, `checks/r12_smoke.json`):
  - fixed_full and fixed_pos have the same α0 (0.246668) and the same Q0 hash, both also equal to the regression run's;
  - α0 is constant across iterations.
- **Regime of α0, from R11 process statistics, not outcomes.**
  - On the R11 cohort, the Q0 scale ratio had median 1.15 (range 0.25–3.30). That is roughly 11 times the official 0.1
    at the median.
  - Under population-wise standardisation the adaptive weight grew to a median of about 21 by the last iteration
    (mean 9.88 over all populations).
  - So Q1 tests the retained component at a fixed intermediate weight, not at the adaptive weights' scale. A null Q1
    would not rule out an effect at larger weights.
- **Runtime estimate:** about 450 s per run, so Q1 ≈ 8 h and Q2 ≈ 8 h of CPU in this container.

## Failures, deviations, freeze
- **Infrastructure crash:** a run that crashes (traceback, OOM, kill) is rerun with unchanged code. The deterministic
  seeds make the rerun identical. Every crash is recorded in `R12_DEVIATIONS.md`.
- **Code fix:** if code must change, the change is recorded there before continuing. Only crash fixes are allowed;
  nothing that touches the scores.
- **Paper freeze:**
  - If Q1 or Q2 is not complete by **29 Oct 2026, 20:00 Vietnam time**, the ICAI paper reports only the completed
    question(s) and labels the rest INCOMPLETE.
  - Any R12 text enters the paper only through a new revision with a changelog, whatever the result.
- **Not claimed:** a pure-velocity causal effect, a new architecture, or a confirmed success-rate gain.

## Hashes (sha256) recorded at lock time
```
2f093d588b668100d45805592fbfea863fde9c6c6b66451b94da6c7bc09e88b1  R12_LOCK.md
9c3ebe2e218b3e3ac822ecc280b7664990de3d2bc07207a183e680f65cdd2c35  r12_frozen_scale.py
b4e4678596e008dee6f17e092385d040cee8fdf88897421ca872b64c8f7b82c2  r12_analyze.py
a44c3a9ac25bc386d22f30410b01991aca2cba0aa221ca73fe48cb84ab04877e  make_r12_cohort.py
91ba9441cd6f777267b26cb84a66f73ff6a2c60ac00c20b624154b62af9cff68  r12_cohort.json
04c7a3465d3c99d9189fd6bbb8117424290a67089644cb753bc5f5d502cd9915  run_r12.sh
dcf3ea4107399500b28e368aef9201f362c1ed5378f79f34c886cc9075068880  checks/r12_regression.json
86303b5af4521cf3d751b0cc78549bd2a3bc1d99877411e7360e1dabed2d9019  checks/r12_smoke.json
0ad42f0af306617141cabbdde7bad219f88c040aa0fcc45049bfed8652efa496  ../d3/official_tasks.py
4f941022c4866a777bef34db06bdda9a963a1b65f52aaeced5bc344d0b2f5cc6  ../d3/wm_costs.py
b3effe85c1e62912759251a9265118766d58bf3763d54b33c50b4f0ff85f4279  ../v06_reconstructed/jepa_adapter.py
locked_at_utc 2026-10-28T07:54:10Z
```
