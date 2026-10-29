# R9 / R10 process deviations (Claude, 27/10/2026)

Written at 11:19 UTC, after R9 chunk 3 started and before any R9 or R10 analysis. None of these changes a cohort, an arm,
a seed, a stopping rule or an analysis script. Each was locked and hashed earlier: R8_ADDENDUM_LOCK.md at 02:52/02:55 UTC,
R10_PROTOCOL_LOCK.md at 04:57/05:00 UTC.

## R9 (in-loop CEM on the 16 R8 starts)

- **R9-D1. Chunk 1 log overwritten.**
  - Chunk 1 ran 04:50–06:47 UTC and ended at the 7000 s cap (exit 124), as designed.
  - It recorded starts 0–11 in full (b0, veldrop_z, b1; plus raw on start 0) and 12:b0.
  - Its stdout log was overwritten when chunk 2 was launched with the same log path. Only the last three lines survive,
    in `logs/r9_inloop_chunk1_tail.txt` (sha256 3883bf13…).
  - No result is lost. Every run record in `r9_inloop.json` keeps its own wall time (`seconds`), identity, plan and
    final state.
- **R9-D2. Chunk 2 recorded nothing.**
  - Chunk 2 was launched at 06:50 UTC as a detached shell job, concurrently with R10 attempt 1 (see R10-D1).
  - It recorded no run: the run set and the 12:b0 record are unchanged.
  - It was no longer running when the container restarted at about 11:06 UTC. The cause cannot be verified after the
    restart. Possible causes: detached jobs are stopped after a session turn ends, or memory pressure from two models on
    8 GB without swap.
  - Log: `logs/r9_inloop_chunk2_killed.log` (sha256 56a0d7be…).
- **R9-D3. Chunk 3 resume.**
  - Before resuming, `verify_r9_independent.py --provenance-only` re-checked the 38 existing runs. All passed:
    - code hash equals the lock;
    - identities and NPZ hashes match the R8 index;
    - the stored margin equals the recomputed official margin (max deviation 0.0);
    - the b0 plans equal the R8 proposal means;
    - the raw-arm rerun reproduces it (max abs 0.0).
  - A snapshot of the JSON before chunk 3 is kept: `r9/r9_inloop_snapshot_before_chunk3.json`
    (sha256 748f6bcb7b2c5a61cce95ed45ca1a24785cbcea7a0b1333fad578d26e7af5604).
  - Chunk 3 started at 11:13 UTC as a tracked background job, alone on the machine. Log:
    `logs/r9_inloop_chunk3.log`.
- **R9-D4. Interim look at two records.**
  - At about 11:16 UTC, while inspecting the JSON schema, Claude printed two individual run records, `0:raw` and
    `1:veldrop_z`, with their margins and success flags.
  - No summary statistic, D or decision was computed.
  - Arms, order, stopping (all 16 starts, start-major) and `r9_analyze.py` were fixed and hashed beforehand, so the look
    cannot change any decision. Disclosed for completeness.

## R10 (DINO-WM cross-model check, 8 starts)

- **R10-D1. Attempt 1 started early and recorded nothing.**
  - Attempt 1 was launched at 06:50 UTC, concurrently with R9 chunk 2. This is contrary to the lock ("starts after R9
    ends").
  - The collector process was killed (exit 137) before any start was recorded: `cohort/` is empty and no index exists.
    No R10 data or outcome existed or was seen.
  - Log: `logs/r10_collect_attempt1_killed.log` (sha256 339f7d64…).
  - R10 is relaunched alone after R9 completes. The locked command, seed (39000000) and episode list are unchanged, and
    so are `r10_collect.py`, `r10_analyze.py` and `run_r10.sh` (hashes as in R10_PROTOCOL_LOCK.md and Amendment 1).

## Independent checkers written before the analyses

- `verify_r9_independent.py`:
  - numpy only;
  - re-derives every margin from the stored final state and the R8 goal state with the official PushT criterion;
  - checks provenance and recomputes both R9 decisions.
- `r10/verify_r10_independent.py`:
  - generalises `verify_r8_independent.py` to any proprio width;
  - dry run on the R8 cohort with the JEPA checkpoint reproduces every R8 decision number: B0 2.947 (8), VELDROP-z
    2.320 (9), B1 2.909 (6), D −0.627, leave-best-out −0.371, mechanism 0.433 / 0.954. Agreement: ALL.

## Timing note (not a deviation)

- **Chunk 3 is slower than chunk 1.** Chunk 3 runs about 1.75 times slower per CEM iteration: 58–71 s, against
  34–36 s in chunk 1 (start 12 veldrop_z took 504.9 s; starts 10–11 took 276–286 s).
- **Cause: the restarted container, not page faults.**
  - The restarted container (Intel Xeon @ 2.10 GHz, 2 vCPU, 8 GB, no swap) spends about 100 s of user CPU per
    64-sample scoring call, against about 66 s implied by chunk 1. It also has many minor page faults.
  - A check in the scratchpad (`bench_malloc.py`, start 0, one CEM population) gave bit-identical outputs with and
    without `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (sha256 d4d21633…).
  - That setting cut minor faults 113-fold (8.05 M to 71 k) but wall time only from 122 s to 119 s under contention.
- **Decision.** The setting was not used, so the runtime environment of R9 and R10 stays as locked.
- **Effect.** No result depends on speed; only the delivery time moves. Expected: R9 completes about 12:25 UTC; R10 at
  about 31–37 min per start, so about 4–5 h after it starts.

## Communication error (no effect on data)

- At 12:21 Vietnam time (05:21 UTC) Claude scheduled a message to Mạnh for 18:00 whose text announced that R9, R10 and
  the packet were finished.
- Between 13:50 and 18:06 Vietnam time no experiment was running, and the status reported to Mạnh in that window was
  not verified.
- Corrected at 18:11 Vietnam time. From now on status is reported only from `ps`, logs and JSON contents.
