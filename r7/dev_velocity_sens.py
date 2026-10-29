"""DEVELOPMENT sensitivity (26 R6/R7 starts; written after the R8 addendum lock, cannot change it).
(a) weight lambda in z(vis) + lambda * z(prop_pos); (b) a zero-goal-velocity target (stationarity, LEAP-like) instead of
dropping velocity: prop_v0 = (1/16)||W_p dp + W_v v_hat||^2. All variants printed. Usage: python3 -I dev_velocity_sens.py"""

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
ROOT = Path(str(_WM_ROOT / 'r7/banks'))
W, b = load_W(); Wi = np.linalg.pinv(W)
S = []
for cohort in ['train', 'test']:
    d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
    for k in sorted(idx['starts'], key=int):
        z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
        P = z['pv16'][ev].astype(float).reshape(n, -1, C); G = z['gv'].astype(float).reshape(-1, C)
        vis = ((P - G[None]) ** 2).mean((1, 2)); pp = z['pp16'][ev].astype(float); gp = z['gp'].astype(float)
        r = (pp.mean(1) - b) @ Wi.T; rg = (gp.mean(0) - b) @ Wi.T
        pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
        v0 = (((r[:, :2] - rg[:2]) @ W[:, :2].T + r[:, 2:] @ W[:, 2:].T) ** 2).mean(1)
        S.append(dict(vis=vis, pos=pos, v0=v0, m=z['margin'][ev]))
zs = lambda a: (a - a.mean()) / (a.std() + 1e-12)
def run(fn):
    m = np.array([s['m'][pick(fn(s))] for s in S]); return m.mean(), int((m < 1).sum())
print('official reference: see dev_velocity.log (3.845, 7)')
for lam in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    print(f'z(vis) + {lam:4.2f} z(prop_pos): mean %.3f succ %d' % run(lambda s, l=lam: zs(s['vis']) + l * zs(s['pos'])))
print('z(vis) + z(prop with ZERO goal velocity): mean %.3f succ %d' % run(lambda s: zs(s['vis']) + zs(s['v0'])))
print('vis + 0.1 * prop with ZERO goal velocity: mean %.3f succ %d' % run(lambda s: s['vis'] + 0.1 * s['v0']))
