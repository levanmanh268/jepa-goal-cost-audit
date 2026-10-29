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
