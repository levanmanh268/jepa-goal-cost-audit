"""DEVELOPMENT: exact split of the official proprio term into position, velocity and cross parts (26 R6/R7 starts).
Pi = (1/16)||W_p dp + W_v dv||^2 = Pi_pos + Pi_vel + Pi_cross. Reports mean shares, within-bank variance shares,
anisotropy of the position quadratic form in simulator (pixel) units, and the Gaussian selection-attenuation factor
rho = corr(Pi_pos, Pi) per bank. Usage: python3 -I dev_velocity_terms.py"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/home/claude/wm/r7')
from dev_velocity import load_W, STD
ROOT = Path('/home/claude/wm/r7/banks')
W, b = load_W(); Wp = np.linalg.pinv(W); Wpos, Wvel = W[:, :2], W[:, 2:]
M = Wpos.T @ Wpos / 16; Dinv = np.diag(1 / STD[:2]); Mpix = Dinv @ M @ Dinv; ev = np.linalg.eigvalsh(Mpix)
print('position quadratic form in pixel units: eigenvalues', ev, 'condition number', ev.max() / ev.min())
print('cosines between position and velocity columns of W:', np.round((Wpos / np.linalg.norm(Wpos, axis=0)).T @ (Wvel / np.linalg.norm(Wvel, axis=0)), 3))
sh = []; vs = []; rho = []; rho_full = []
for cohort in ['train', 'test']:
    d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
    for k in sorted(idx['starts'], key=int):
        z = np.load(d / f'start_{int(k):03d}.npz'); ev_ = z['split'] == 1
        pp = z['pp16'].astype(np.float64)[ev_]; gp = z['gp'].astype(np.float64)
        r = (pp.mean(1) - b) @ Wp.T; rg = (gp.mean(0) - b) @ Wp.T; dp = r[:, :2] - rg[:2]; dv = r[:, 2:] - rg[2:]
        pos = ((dp @ Wpos.T) ** 2).mean(1); vel = ((dv @ Wvel.T) ** 2).mean(1); cross = 2 * ((dp @ Wpos.T) * (dv @ Wvel.T)).mean(1)
        tot = ((pp.reshape(len(pp), -1) - gp.reshape(-1)) ** 2).mean(1)
        sh.append([pos.mean() / tot.mean(), vel.mean() / tot.mean(), cross.mean() / tot.mean()])
        vs.append(np.var(pos) / np.var(tot)); rho.append(np.corrcoef(pos, pos + vel + cross)[0, 1])
        st = z['official_state'][ev_]; a = np.linalg.norm(st[:, :2] - z['goal_state'][:2], axis=1)
        rho_full.append(np.corrcoef(np.argsort(np.argsort(pos)), np.argsort(np.argsort(a)))[0, 1])
sh = np.array(sh)
print(f'mean shares of the official proprio term (eval banks): position {sh[:,0].mean():.3f}, velocity {sh[:,1].mean():.3f}, cross {sh[:,2].mean():.3f}; '
      f'reconstruction residual share {1 - sh.sum(1).mean():+.4f}')
print(f'within-bank variance ratio Var(Pi_pos)/Var(Pi): median {np.median(vs):.3f}; corr(Pi_pos, Pi) per bank: mean {np.mean(rho):.3f}, median {np.median(rho):.3f}')
