"""Independent recomputation of the R8 decisions (Claude, 2026-10-27). Does NOT import r8_analyze.py, energies.py or
tail_energies.py; everything is re-derived from the NPZ arrays, the frozen head JSON and the checkpoint's proprio encoder.
Checks: B0 / B1 / VELDROP-z selections, margins, successes, decisions, mechanism Spearman means.
Usage: python3 -I verify_r8_independent.py <cohort dir> <index json name> <r8_analysis.json to compare against>
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys
from pathlib import Path
import numpy as np
import torch


def rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    # average ties
    xs = x[o]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
        if j > i: r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spear(a, b):
    return float(np.corrcoef(rank(np.asarray(a, float)), rank(np.asarray(b, float)))[0, 1])


def first_min(c):
    c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1.0, abs(c.min())))[0])


def main(cohort, index_name, ref):
    d = Path(cohort); idx = json.loads((d / index_name).read_text())
    w = np.array(json.load(open(str(_WM_ROOT / 'r7/R7_FROZEN_R6_HEAD.json')))['weights'], float)
    pe = torch.load(str(_WM_ROOT / 'runtime/checkpoints/jepa_wm_pusht.pth.tar'), map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy(); Wi = np.linalg.pinv(W)
    rows = []
    for k in sorted(idx['starts'], key=int):
        z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum())
        P16 = z['pv16'][ev].reshape(n, -1, z['pv16'].shape[-1]); G = z['gv'].astype(float).reshape(-1, P16.shape[-1])
        p16 = z['pp16'][ev].reshape(n, -1); gp = z['gp'].astype(float).reshape(-1)
        D = P16.astype(float) - G[None]; ch = (D * D).mean(1)                       # channel energies (float16 inputs, as R6)
        pr16 = ((p16.astype(float) - gp[None]) ** 2).mean(1)
        b1 = ch @ w[:-1] + pr16 * w[-1]
        vis = (D * D).mean((1, 2))
        pt = p16.astype(float).reshape(n, -1, 16).mean(1); r = (pt - b) @ Wi.T; rg = (gp.reshape(-1, 16).mean(0) - b) @ Wi.T
        pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
        vz = (vis - vis.mean()) / vis.std() + (pos - pos.mean()) / pos.std()
        m = z['margin'][ev]; s = z['success'][ev].astype(bool); st = z['official_state'][ev]
        agent = np.linalg.norm(st[:, :2] - z['goal_state'][:2], axis=1)
        prop_full = ((z['pp16'][ev].astype(float).reshape(n, -1) - gp[None]) ** 2).mean(1)
        j0, j1, jv = first_min(z['model_cost'][ev]), first_min(b1), first_min(vz)
        rows.append(dict(k=int(k), m0=m[j0], m1=m[j1], mv=m[jv], s0=s[j0], s1=s[j1], sv=s[jv], sp_prop=spear(prop_full, agent), sp_pos=spear(pos, agent)))
    n = len(rows); M0 = np.array([r['m0'] for r in rows]); M1 = np.array([r['m1'] for r in rows]); MV = np.array([r['mv'] for r in rows])
    S0 = sum(r['s0'] for r in rows); S1 = sum(r['s1'] for r in rows); SV = sum(r['sv'] for r in rows)

    def dec(D, sn, so, yes, no):
        if n < 8: return 'INCOMPLETE'
        return yes if (D.mean() <= -0.10 and sn >= so and np.sort(D)[1:].mean() <= 0) else no
    out = {'n': n, 'B0': [float(M0.mean()), int(S0)], 'B1': [float(M1.mean()), int(S1)], 'VELDROP_Z': [float(MV.mean()), int(SV)],
           'primary': dec(M1 - M0, S1, S0, 'REPLICATED', 'NOT_REPLICATED'), 'addendum': dec(MV - M0, SV, S0, 'SUPPORTED', 'NOT_SUPPORTED'),
           'mech_prop': float(np.mean([r['sp_prop'] for r in rows])), 'mech_pos': float(np.mean([r['sp_pos'] for r in rows]))}
    print(json.dumps(out, indent=1))
    if ref:
        R = json.load(open(ref))['summary']
        chk = {'B0_mean': abs(R['mean_selected_margin']['B0'][0] - out['B0'][0]) < 1e-9, 'B1_mean': abs(R['mean_selected_margin']['B1'][0] - out['B1'][0]) < 1e-9,
               'VZ_mean': abs(R['mean_selected_margin']['VELDROP_Z'][0] - out['VELDROP_Z'][0]) < 1e-9,
               'succ': R['success_counts']['B0'] == out['B0'][1] and R['success_counts']['B1'] == out['B1'][1] and R['success_counts']['VELDROP_Z'] == out['VELDROP_Z'][1],
               'primary': R['PRIMARY_B1_vs_B0']['decision'] == out['primary'] or R['PRIMARY_B1_vs_B0']['decision'] == 'INCOMPLETE',
               'addendum': R['ADDENDUM_VELDROP_Z_vs_B0']['decision'] == out['addendum'] or R['ADDENDUM_VELDROP_Z_vs_B0']['decision'] == 'INCOMPLETE',
               'mech': abs(R['MECHANISM']['mean_spearman_proppos_agent'] - out['mech_pos']) < 1e-9 and abs(R['MECHANISM']['mean_spearman_prop_agent'] - out['mech_prop']) < 1e-9}
        print('agreement with locked analysis:', chk)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
