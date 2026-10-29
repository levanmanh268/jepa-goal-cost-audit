# Third-party sources and licensing

The author of the manuscript is Le Van Manh. The released JEPA-WM and DINO-WM models, their datasets, standard
algorithms and third-party software remain the work of their respective authors. This artifact does not claim their
authorship or relicense their material.

## JEPA-WMs

Source: https://github.com/facebookresearch/jepa-wms, pinned commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.

The root license and README specify Creative Commons Attribution-NonCommercial 4.0, with separately licensed
third-party components. The exact root license is retained as `licenses/JEPA_WMS_CC_BY_NC_4.0.txt`.
Relevant upstream source headers were checked during packaging:

- `evals/simu_env_planning/envs/pusht_env/pusht_env.py`
- `evals/simu_env_planning/envs/pusht_gym_wrap.py`

Both say `Copyright (c) Facebook, Inc. and its affiliates.`, credit the DINO-WM implementation, and state MIT License.
The evaluation criterion and fast-rollout adaptation in `d3/official_tasks.py` follow those files. Their attribution
is retained here, and the MIT permission notice is included in `licenses/PUSHT_MIT.txt`.
The fast-rollout adaptation skips intermediate rendering and coverage calculation; its earlier verification is a
saved project result, not a newly executed physics check in this artifact build.

The CEM and latent-cost implementation in `d3/wm_costs.py` follows the released planner. The main JEPA-WMs terms
continue to apply to any upstream-derived portions outside the separately MIT-marked environment files. Packaging
adds path portability and terminal-state audit support; it does not claim the original planner as a new algorithm.

## DINOv2

External dependency: https://github.com/facebookresearch/dinov2 at
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, whose LICENSE is Apache 2.0.
The DINOv2 implementation and weights are not bundled here.

## Data and models

Official checkpoint locations and hashes are in `v06_reconstructed/checkpoints.json`. The pinned JEPA-WMs README
attributes the PushT dataset to the DINO-WM project. Download those assets from the official releases under their
own terms; this artifact does not redistribute them.

## IEEEtran

`paper/IEEEtran.cls` is the unmodified standard IEEEtran v1.8b class. Its copyright and LaTeX Project Public License
notice remain in the file. Its inclusion is for formatting and does not imply IEEE publication or endorsement.

## Original material

No blanket license has been selected for the author's original manuscript, analysis additions or saved results.
Their presence in a public repository should not be represented as an unrestricted license grant. Existing
third-party rights and permissions are unaffected. Please contact the author about reuse beyond applicable permissions.
