"""EXPLORATORY, written after the R8 addendum lock (cannot change R8/R9). R6 test banks only (10 starts; frozen B1 is
out-of-sample there, but these banks are development data). Does combining B1's visual part with the position-only
proprio term help? Rules: B1 (frozen), B1vis = channel part of B1 only, z(B1vis)+z(prop_pos), z(B1)+z(prop_pos).
Usage: python3 -I dev_combo_b1vis_pos.py"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(_WM_ROOT / 'r7'))
import energies as E
from dev_velocity import load_W
from dev_screen2 import pick
W, b = load_W(); Wi = np.linalg.pinv(W); w = np.array(json.load(open(str(_WM_ROOT / 'r7/R7_FROZEN_R6_HEAD.json')))['weights'])
zs = lambda a: (a - a.mean()) / (a.std() + 1e-12)
d = Path(str(_WM_ROOT / 'r7/banks/test')); idx = json.loads((d / 'r7_index.json').read_text()); res = {}
for k in sorted(idx['starts'], key=int):
    z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
    P = z['pv16'][ev].astype(float).reshape(n, -1, C); G = z['gv'].astype(float).reshape(-1, C)
    p = z['pp16'][ev].astype(float).reshape(n, -1); gp = z['gp'].astype(float).reshape(-1)
    ch, pr = E.channel_energies(P, G, p, gp); b1vis = ch @ w[:-1]; b1 = b1vis + pr * w[-1]
    vis = ((P - G[None]) ** 2).mean((1, 2)); r = (p.reshape(n, -1, 16).mean(1) - b) @ Wi.T; rg = (gp.reshape(-1, 16).mean(0) - b) @ Wi.T
    pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1); m = z['margin'][ev]
    for name, sc in [('B0', z['model_cost'][ev]), ('B1', b1), ('B1vis only', b1vis), ('VELDROP-z', zs(vis) + zs(pos)),
                     ('z(B1vis)+z(prop_pos)', zs(b1vis) + zs(pos)), ('z(B1)+z(prop_pos)', zs(b1) + zs(pos))]:
        res.setdefault(name, []).append(m[pick(sc)])
for name, v in res.items():
    v = np.array(v); print(f'{name:24s} mean {v.mean():.3f} succ {(v < 1).sum()}')
