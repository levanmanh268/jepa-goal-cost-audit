"""DEVELOPMENT (26 R6/R7 starts): what limits success after VELDROP-z? Counts starts where the bank contains a success
(O succeeds) and whether VELDROP-z / official pick it. Usage: python3 -I dev_gap_after_veldrop.py"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(_WM_ROOT / 'r7'))
from dev_velocity import load_W
from dev_screen2 import pick
ROOT = Path(str(_WM_ROOT / 'r7/banks')); W, b = load_W(); Wi = np.linalg.pinv(W)
zs = lambda a: (a - a.mean()) / (a.std() + 1e-12)
tab = {'O_succ': 0, 'veldrop_succ': 0, 'official_succ': 0, 'O_succ_veldrop_miss': 0, 'O_succ_official_miss': 0, 'n_success_plans': []}
for cohort in ['train', 'test']:
    d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
    for k in sorted(idx['starts'], key=int):
        z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
        P = z['pv16'][ev].astype(float).reshape(n, -1, C); G = z['gv'].astype(float).reshape(-1, C)
        vis = ((P - G[None]) ** 2).mean((1, 2)); pp = z['pp16'][ev].astype(float); gp = z['gp'].astype(float)
        r = (pp.mean(1) - b) @ Wi.T; rg = (gp.mean(0) - b) @ Wi.T; pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
        s = z['success'][ev].astype(bool); jv = pick(zs(vis) + zs(pos)); j0 = pick(z['model_cost'][ev])
        tab['O_succ'] += int(s.any()); tab['veldrop_succ'] += int(s[jv]); tab['official_succ'] += int(s[j0])
        tab['O_succ_veldrop_miss'] += int(s.any() and not s[jv]); tab['O_succ_official_miss'] += int(s.any() and not s[j0])
        tab['n_success_plans'].append(int(s.sum()))
print(json.dumps({k: v for k, v in tab.items() if k != 'n_success_plans'}), 'successful plans per bank (of 64):', tab['n_success_plans'])
