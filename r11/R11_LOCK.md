# R11 LOCK: separating velocity removal from rescaling inside CEM (Claude, 28/10/2026)

Written and hashed before any R11 output exists, and before the R11 smoke test.
- **Origin:** ChatGPT's review of R8–R10 (`R11_MECHANISM_PROTOCOL_chatgpt_proposal.md`, priority 1), adopted with the
  specifics below.
- **Status:** development. The 16 starts are the R8/R9 cohort, whose R9 outcomes are known. The contrasts were chosen
  after reading R8/R9, so nothing here is confirmation.

## Question

R9 changed two things at once: it replaced the full proprioceptive term with its position-only part, and it replaced
the fixed weight 0.1 with within-population z-scoring. Which change carries the R9 gain?

## Cells (2 × 2, CEM 64 × 8, 10 elites, H = 6, sample 0 = mean, same keyed seeds, final mean executed open loop)

| Cell | Proprio quantity | Weight rule | Source |
|---|---|---|---|
| C00 | full | visual + 0.1 full (official) | R9 `b0` (R8 proposal mean, executed) |
| C10 | position only | visual + 0.1 position | **R11 `veldrop_01` (new)** |
| C01 | full | z(visual) + z(full) within each population | **R11 `hprop_std` (new)** |
| C11 | position only | z(visual) + z(position) | R9 `veldrop_z` |

The runner is `r11_scale_ablation.py`, written by ChatGPT. Claude diffed it line by line against the locked
`r9_inloop.py`. The only changes are:
- the two arms;
- the output file name;
- the code-hash key;
- the extra score terms;
- logged population standard deviations and the implied scales.

The raw (official-cost) arm is rerun on start 0 and must reproduce the R8 proposal mean exactly (max abs 0.0).
Otherwise R11 is INCOMPLETE.

## Analysis (`r11_analyze.py`, hash below)

- **PRIMARY:** D1 = margin(C11) − margin(C01), paired by start.
  - **DROP_HELPS** iff mean D1 ≤ −0.10 AND successes(C11) ≥ successes(C01) AND mean D1 without the most favourable
    start ≤ 0.
  - **DROP_NOT_SHOWN** otherwise.
  - **INCOMPLETE** if n < 8 or a provenance failure.
  - This is an operational priority rule with the same thresholds as R8–R10, not an alpha-controlled test.
- **Descriptive:**
  - C10 − C00;
  - the interaction (C11 − C01) − (C10 − C00);
  - the scale effects C01 − C00 and C11 − C10;
  - per cell: successes, block-only successes, component errors;
  - bootstrap over starts (20,000 resamples, seed 4001);
  - exact McNemar and sign tests, descriptive only;
  - mean in-loop alpha.

## Pre-specified readings

- **DROP_HELPS:** under the same z-scored search, using the position-only term instead of the full term improves the
  executed plans. This supports velocity removal as an active ingredient of the R9 gain, on this development cohort.
- **DROP_NOT_SHOWN with C01 close to C11:** the R9 gain is attributable to rescaling, not to dropping velocity. The
  paper must then say so.
- **C10 − C00:** from the R8 banks (C10 ≈ C00 at weight 0.1), a near-zero difference is expected. It does not decide
  anything.
- **What would refute the velocity mechanism in-loop:** DROP_NOT_SHOWN, together with C01 matching C11 in both margin
  and successes.

## Schedule and integrity

- One model process at a time.
- Chunks of at most 7000 s, resumable, with a separate log per chunk.
- All 16 starts; no outcome-dependent stopping.
- Status is reported only from ps, logs and JSON.

## Hashes
```
ed608b0a42d88e4a8e30782bb15717259035681ddf8763c20bfbc5677ce9dcf0  r11_scale_ablation.py
ed9c59b72b9ddcec3f4a0fb5ef5deb13e2f10d490e1ea7e5934bc3a65db12448  r11_analyze.py
4fa03ade3770f0bcf0980e656cf56be51c1279f64f96e4cbf23a76119d253984  run_r11.sh
d58985ca8ef88d76b5b841d7dc74ded2d96e0df3027b9bec9cf7e36684f6ba7d  r9_inloop.json
535f54b636c14b2f83acd1964383b993d423f8f2d8c1865c2f3b78cdefc2f8cb  energies.py
0ad42f0af306617141cabbdde7bad219f88c040aa0fcc45049bfed8652efa496  official_tasks.py
4f941022c4866a777bef34db06bdda9a963a1b65f52aaeced5bc344d0b2f5cc6  wm_costs.py
b3effe85c1e62912759251a9265118766d58bf3763d54b33c50b4f0ff85f4279  jepa_adapter.py
```
