"""POST HOC, descriptive (written after R8 outcomes were opened; decides nothing). R8 cohort, JEPA-WM.
Sensitivity of VELDROP-z to the position weight lambda in z(vis) + lambda z(prop_pos), and the zero-goal-velocity
variant, mirroring r7/dev_velocity_sens.py. Usage: python3 -I posthoc_r8_sensitivity.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
d = Path('/home/claude/wm/r8/cohort'); idx = json.loads((d / 'r6_index.json').read_text())
pe = torch.load('/home/claude/wm/runtime/checkpoints/jepa_wm_pusht.pth.tar', map_location='cpu', weights_only=False)['proprio_encoder']
W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy(); Wi = np.linalg.pinv(W)
zs = lambda a: (a - a.mean()) / (a.std() + 1e-12)
def pick(c): c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1., abs(c.min())))[0])
S = []
for k in sorted(idx['starts'], key=int):
    z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
    P = z['pv16'][ev].astype(float).reshape(n, -1, C); G = z['gv'].astype(float).reshape(-1, C)
    vis = ((P - G[None]) ** 2).mean((1, 2)); pp = z['pp16'][ev].astype(float); gp = z['gp'].astype(float)
    r = (pp.mean(1) - b) @ Wi.T; rg = (gp.mean(0) - b) @ Wi.T
    pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1); v0 = (((r[:, :2] - rg[:2]) @ W[:, :2].T + r[:, 2:] @ W[:, 2:].T) ** 2).mean(1)
    S.append(dict(vis=vis, pos=pos, v0=v0, m=z['margin'][ev], s=z['success'][ev]))
def run(fn):
    js = [pick(fn(s)) for s in S]; m = np.array([s['m'][j] for s, j in zip(S, js)]); return m.mean(), int(sum(s['s'][j] for s, j in zip(S, js)))
for lam in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    print(f'z(vis) + {lam:4.2f} z(prop_pos): mean %.3f succ %d' % run(lambda s, l=lam: zs(s['vis']) + l * zs(s['pos'])))
print('z(vis) + z(prop, zero goal velocity): mean %.3f succ %d' % run(lambda s: zs(s['vis']) + zs(s['v0'])))
