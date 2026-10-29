"""Post hoc DESCRIPTIVE facts explaining the R10 mechanism result (Claude, 2026-10-27; written after r10_analysis.json
was opened, so nothing here is pre-registered).
1. Column norms of each checkpoint's proprio encoder W (inputs x, y, vx, vy): how much each model's goal term can
   depend on velocity at all.
2. On the R10 and R8 compact components: per start, does VELDROP-z select the same plan as the standardised full
   proprio rule z(vis) + z(prop_full) (H_prop-std)?
Usage: python3 -I r10_mechanism_facts.py out.json
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys
import numpy as np
import torch

CK = {'jepa': str(_WM_ROOT / 'runtime/checkpoints/jepa_wm_pusht.pth.tar'), 'dino': str(_WM_ROOT / 'runtime/checkpoints/dino_wm_pusht.pth.tar')}
COMP = {'jepa_r8': str(_WM_ROOT / 'r8/r8_components.npz'), 'dino_r10': str(_WM_ROOT / 'r10/r10_components.npz')}


def first_min(c):
    c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1.0, abs(c.min())))[0])


def z(a):
    return (a - a.mean()) / a.std()


out = {'encoder_column_norms': {}, 'veldrop_vs_hprop_std_same_selection': {}}
for n, p in CK.items():
    W = torch.load(p, map_location='cpu', weights_only=False)['proprio_encoder']['module.patch_embed.weight'][:, :, 0].double()
    cn = W.norm(dim=0).tolist()
    out['encoder_column_norms'][n] = {'x': cn[0], 'y': cn[1], 'vx': cn[2], 'vy': cn[3],
                                      'velocity_to_position_norm_ratio': float(np.hypot(cn[2], cn[3]) / np.hypot(cn[0], cn[1]))}
for n, p in COMP.items():
    Z = np.load(p); same = []
    for k in range(Z['vis'].shape[0]):
        jv = first_min(z(Z['vis'][k]) + z(Z['prop_pos'][k])); jh = first_min(z(Z['vis'][k]) + z(Z['prop_full'][k]))
        same.append(jv == jh)
    out['veldrop_vs_hprop_std_same_selection'][n] = {'same': int(sum(same)), 'n': len(same)}
json.dump(out, open(sys.argv[1], 'w'), indent=1); print(json.dumps(out, indent=1))
