# Repeating collection

The one-command path in the README uses saved results. This document describes the additional dependencies for new
CPU collection. The portable collection scripts have been syntax-checked; they have not been run against checkpoints
in this packaging environment. CUDA and the official-budget MPC evaluation have not been tested here.

## Pinned external sources

- JEPA-WMs: https://github.com/facebookresearch/jepa-wms at
  `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- DINOv2: https://github.com/facebookresearch/dinov2 at
  `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Checkpoint URLs, byte sizes and hashes: `v06_reconstructed/checkpoints.json`.
- The DINOv2 ViT-S/14 weight hash is pinned in `v06_reconstructed/jepa_adapter.py`.
- PushT data: use the dataset instructions in the pinned JEPA-WMs README. It attributes these data to DINO-WM and
  links its official hosted release. The four input-file hashes used for the fresh cohort are in
  `r12/r12_cohort.json`, field `train_sha256`.

The saved collection provenance records PyTorch `2.7.0+cu126` and the runtime export records NumPy `2.3.5`. The
collection scripts explicitly select CPU. This is not a complete recreation of the original environment lockfile;
the artifact does not assert identical physics across arbitrary Python, PyMunk or BLAS versions.

## Paths

Set `WM_ROOT` to an absolute path containing this repository's folders. If it is unset, portable scripts use the
repository root. Add these external paths underneath it:

| Path | Required content |
| --- | --- |
| `runtime/jepa-wms/` | Git checkout at the pinned JEPA-WMs commit |
| `runtime/dinov2/` | Git checkout at the pinned DINOv2 commit |
| `runtime/checkpoints/jepa_wm_pusht.pth.tar` | JEPA checkpoint under the exact local filename |
| `runtime/checkpoints/dino_wm_pusht.pth.tar` | DINO checkpoint when repeating DINO runs |
| `d3/data_official/pusht_noise/train/` | `states.pth`, `velocities.pth`, `rel_actions.pth`, `seq_lengths.pkl` |

Use the official upstream environment instructions. The adapter validates source revisions, checkpoint hashes and
parameter loading. Checkpoint deserialisation should only use the official files whose hashes match these records.
The dataset and model weights are not included in this repository, and their upstream conditions still apply.

## R12 and R13 collection

The following commands perform substantial CPU work; they are separate from saved-result reproduction:

```bash
export WM_ROOT="$(pwd)"
python r12/r12_frozen_scale.py --cohort r12/r12_cohort.json --out outputs/new_r12 --threads 2
python r13/r13_zfull.py --cohort r12/r12_cohort.json --out outputs/new_r13 --threads 2
```

Always use new output directories. Portable source has new hashes, so it must not resume or overwrite the original
recorded collection. The runner intentionally rejects code/config changes when resuming.
Do not treat rerunning the now-exposed cohort as a new holdout or a newly registered test.

The old-start `--smoke` and `--regress` modes require the original R8 `start_000.npz` cache, which is not bundled.
Earlier R8/R10 full-cache analyses require their corresponding per-start prediction banks. Repeating the R9 supervised
head comparison also needs the R6 training cache. These are limitations of the current lightweight release. Their
compact saved-result comparisons are runnable through `scripts/reproduce.py`.

## Provenance and code changes

`provenance/PORTABILITY_CHANGES.json` maps source hashes and packaging changes. Original source files are preserved
under `provenance/original_code/`, and the scientific R12/R13 source subset is additionally preserved in `evidence/`.
Environment-specific paths inside original evidence records are historical metadata. They are not paths required by
the portable reproduction entry point.

The R13 metadata key `code_sha256.r12_frozen_scale.py` contains the R13 runner hash, despite retaining the old key
name. The independent audit deliberately checks it against `r13_zfull.py`. This naming issue does not change the
recorded computations.
