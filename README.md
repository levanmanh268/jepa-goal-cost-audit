# JEPA goal-cost audit

Research artifact for **Matching the Wrong Thing: Goal-Frame Velocity in the Planning Cost of a JEPA World Model**.

**Author:** Le Van Manh, Faculty of Information Technology and Communications, CMC University, Ha Noi, Viet Nam.
Contact: BAI250042@st.cmcu.edu.vn.

This repository contains a PushT audit of the released JEPA-WM planning cost and VELDROP-z, a correction that retains
position information in the proprioceptive goal term. VELDROP-z changes the planning objective without retraining the
world model. The accompanying manuscript is prepared for ICAI 2026.

## Main observations

The final CEM mean is executed open loop, with 64 samples, 8 iterations, 10 elites and one seed per start.
All evaluation trajectories are within the checkpoints' training split.

| Comparison on the 32-start cohort | Mean paired margin difference [bootstrap 95% interval] | Successes | Status |
| --- | --- | --- | --- |
| VELDROP-z minus official cost | -1.495 [-2.445, -0.724] | 13 to 20, gains/losses 7/0 | Pre-declared secondary R12 comparison |
| Position minus full, same frozen weight | -0.169 [-0.430, 0.091] | 14 to 14, gains/losses 1/1 | Primary R12 comparison, uncertain margin effect |
| Standardised position minus standardised full | -0.324 [-0.808, 0.073] | 14 to 20, gains/losses 6/0 | R13 added after observing R12 results |

R13 changes the retained components and the standardisation denominator simultaneously. These comparisons do not
identify a pure-velocity effect. Operational labels such as `DROP_HELPS_FRESH` are priority rules, not error-controlled
statistical declarations. The nominal McNemar values are unadjusted across the adaptive sequence of questions.
See the manuscript for the earlier 16-start JEPA-WM results and the 8-bank DINO-WM boundary case.

## Recompute the saved evidence without a checkpoint

The checked analysis environment is Python 3.12.14 with the versions in `requirements-analysis.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/reproduce.py
```

This command recomputes the compact R8 and R10 bank decisions, R9 through R13 saved-result analyses, the independent
160-record R12/R13 audit, 523 numerical LaTeX macros, and Figures 1 and 2. R9, R11, R12 and R13 analysis JSON and the
five macro files must match the supplied references byte for byte. Results go into `outputs/reproduced/`.
Figure PDF metadata may differ, so figures are not compared byte for byte; the Figure 2 script checks its per-start
counts against the locked R12 analysis before saving.

This is a saved-result check. It does not rerun checkpoint inference, physics collection, model training or CUDA.
The historical R6 macro file is a legacy source record, not a claim that its experiment is reproduced by this command.

## Build the manuscript

With TeX Live and `latexmk` installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main_ieee.tex
```

`ICAI2026_LeVanManh_submit_flat.tex` is the equivalent flattened source and requires `fig_audit.pdf` and `fig_fresh.pdf`
in the same directory.

## Where the paper's numbers come from

Table entries, intervals, test values and most counts in the text are LaTeX macros written by these scripts from
saved records. Protocol facts (budgets, lock times, run and chunk counts) and a few small counts in the prose are
typed and can be checked against the same records.

| Paper item | Saved records | Generator in `scripts/` |
| --- | --- | --- |
| Section IV and Fig. 1 | `r11/r8_proprio_decomposition.json`, `r11/r8_proprio_diag.npz` | `make_numbers_extra.py`, `make_fig_audit.py` |
| Development numbers (Section V, Limitations) | `r8/dev_numbers.json` | `make_numbers_r8.py`, `make_numbers_v4.py` |
| Table I and the bank ablations | `r8/r8_analysis.json`, `r8/r8_components.npz` | `make_numbers_r8.py`, `make_numbers_extra.py`, `make_numbers_v4.py` |
| Table II | `r8/r9_inloop.json`, `r8/r9_analysis.json` | `make_numbers_r8.py` |
| Table III | `r11/r11_inloop.json`, `r11/r11_analysis.json` | `make_numbers_extra.py` |
| Table IV and Fig. 2 | `r12/out/r12_inloop.json`, `r13/out/r13_inloop.json`, `r12/r12_analysis.json`, `r13/r13_analysis.json` | `make_numbers_r12.py`, `make_numbers_r13.py`, `make_fig_fresh.py` |
| DINO-WM (Section VI-D) | `r10/r10_analysis.json`, `r10/r10_components.npz`, `r11/r10_proprio_decomposition.json` | `make_numbers_r8.py`, `make_numbers_extra.py` |

## Artifact map

| Path | Contents |
| --- | --- |
| `r8/`, `r10/` | Compact fixed-bank components, cohort indices, original recorded analyses and collection scripts |
| `r8/r9_inloop.json`, `r11/r11_inloop.json` | Earlier executed-plan records |
| `r12/`, `r13/` | Fresh-cohort identities, executed terminal states, analysis scripts, locks and deviations |
| `scripts/` | Portable reproduction entry point, independent audit and paper generators |
| `evidence/r12_r13/` | Unmodified scientific subset of the incoming R12/R13 packet, with a subset manifest |
| `provenance/original_code/` | Original source before portability edits |
| `provenance/logs/` | Supplied execution and verification logs |
| `paper/` | Manuscript sources, PDF and figures |
| `COLLECTION.md` | External dependencies and collection limits |

Large checkpoints, the PushT dataset and per-plan latent caches are not redistributed. Protocol timestamps and logs
are provenance records, not an immutable public preregistration. The R13 infrastructure restart remains documented.

## Rights and upstream attribution

See `THIRD_PARTY_NOTICES.md`. Upstream dependencies retain their own terms. A blanket open-source license has not
been selected for the author's original material, so this artifact must not be described as MIT-licensed or
unrestricted for commercial reuse. Public availability is separate from a license grant.

## Citation

Use `CITATION.cff` for the research artifact and cite the underlying JEPA-WMs and DINO-WM papers. The paper's status is
an unpublished manuscript until an actual publication is recorded.
