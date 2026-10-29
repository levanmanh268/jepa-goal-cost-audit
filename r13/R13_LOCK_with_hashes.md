# R13 LOCK: position versus full under standardisation, on the fresh R12 cohort (29/10/2026)

Status: locked after the R12 analysis and before any R13 run. R12's results are known, including VELDROP-z on these 32
starts. This is exploratory: the arm was chosen because R12 lacked it, as stated in the R12 report before this lock.

## Question
On the 32 locked R12 starts (same identities, seeds and first population), does keeping only the position energy
improve the executed margin under population-wise standardisation?
- Primary D = margin(veldrop_z, from R12) − margin(hprop_std, from R13), where hprop_std = z(L_vis) + z(Π_full).
- This repeats R11's primary contrast (C11 − C01) on new goals.
- Like R11, it changes the retained component and the standardisation denominator at once. It is not a pure-velocity
  effect.
- Descriptive only: hprop_std − raw, with raw being R12's official cost.

## Method
- **Runner** `r13_zfull.py`: `r12_frozen_scale.py` (unchanged in r12/) plus the single arm hprop_std. It runs on the
  locked `r12_cohort.json` with CEM 64 × 8, 10 elites and horizon 6, one seed per start (the R12 cem_seed), and the
  final mean executed open loop. The diff is in `r13_runner_vs_r12.diff`.
- **Integrity:** the Q0 sha256 of every hprop_std run must equal that of the R12 veldrop_z run on the same start.
  Margins and successes are recomputed from final_state.
- **Analysis** `r13_analyze.py`, run once after all 32 runs:
  - bootstrap with `default_rng(13001).integers(0, 32, (20000, 32))`, percentile 95% CI;
  - reported: improved/worsened counts, success gains and losses, exact McNemar, sign test, mean without the most
    favourable start.
- **Gate (priority rule, not a type-I-error-controlled test):** mean D ≤ −0.10 AND successes(veldrop_z) ≥
  successes(hprop_std) AND mean D without the most favourable start ≤ 0.
  - Label DROP_HELPS_FRESH if met, otherwise NOT_SHOWN.
  - INCOMPLETE if any pair is missing.
- **Logs** print timing and α0 only. Outcomes are first seen at the analysis.
- **Freeze:** if R13 is incomplete at 18:00 on 29 Oct (Vietnam time), it is reported as INCOMPLETE. The paper changes
  only through a new revision with a changelog, whatever the result.

## Hashes (sha256) at lock time
```
cebaae13f33fd45ea7edc611704399d7a8bea77759e7526969fa98f03124093e  R13_LOCK.md
4c45a86cfe236ca8f9d00577e561994a11d73e8ec3bfe5c3449e266612695be6  r13_zfull.py
97d9d931adf4caa7f14e00777eeb9dc95db58013974c2d59f7a36c793bbd8cee  r13_analyze.py
95ce42b31f4edb252f52952b5cb63d371faa7b4eb0d82cec52cd146ee2546a10  run_r13.sh
a2b4036dd416b46f9e7b67e27bf9c958caed0ff3f27fb101f89c03aafab14494  r13_runner_vs_r12.diff
91ba9441cd6f777267b26cb84a66f73ff6a2c60ac00c20b624154b62af9cff68  ../r12/r12_cohort.json
2ba48a567a6d183676356723b43f0c2883a85e59c3da7c222c9be3cb6f9550f1  ../r12/out/r12_inloop.json
locked_at_utc 2026-10-28T23:41:19Z
```
