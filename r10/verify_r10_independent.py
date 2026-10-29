"""Independent recomputation of the R10 decisions (Claude, 2026-10-27).

Generalises r8/verify_r8_independent.py to any checkpoint (proprio width taken from W). It does NOT import
r10_analyze.py, r8_analyze.py, energies.py or tail_energies.py. Everything is re-derived from the NPZ arrays, the frozen
B1 head JSON and the checkpoint's own proprio encoder (W, b).

Checks:
  - NPZ sha256 vs the index;
  - cohort = the first 8 starts of the locked order (R10 Amendment 1);
  - B0 / B1 / VELDROP-z selections (argmin, lowest index on ties), margins and successes;
  - primary decision (VELDROP-z vs B0) and mechanism M2 (Spearman of prop_pos and of the official proprio term with the
    true final agent distance);
  - agreement with r10_analysis.json.

Usage:
  python3 -I verify_r10_independent.py <cohort dir> <checkpoint> [<analysis json>] [--r8-dry-run]
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

LOCKED_ORDER = [15720, 1021, 16600, 12404, 15364, 17531, 10308, 16094]


def rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x)); xs = x[o]; i = 0
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


def main(argv):
    r8_dry = '--r8-dry-run' in argv; argv = [a for a in argv if a != '--r8-dry-run']
    d = Path(argv[0]); ckpt = argv[1]; ref = argv[2] if len(argv) > 2 else None
    idx = json.loads((d / 'r6_index.json').read_text()); issues = []
    w = np.array(json.load(open(str(_WM_ROOT / 'r7/R7_FROZEN_R6_HEAD.json')))['weights'], float)
    pe = torch.load(ckpt, map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy()
    Wi = np.linalg.pinv(W); dp = W.shape[0]
    keys = sorted(idx['starts'], key=int)
    if not r8_dry:
        if len(keys) > 8: issues.append(f'{len(keys)} starts recorded; using the first 8 (Amendment 1)')
        keys = keys[:8]
        eps = [idx['starts'][k]['identity']['val_episode'] for k in keys]
        if eps != LOCKED_ORDER[:len(eps)]: issues.append(f'episode order {eps} differs from the locked order')
    rows = []
    for k in keys:
        f = d / f'start_{int(k):03d}.npz'
        if hashlib.sha256(f.read_bytes()).hexdigest() != idx['starts'][k]['npz_sha256']: issues.append(f'npz sha256 start {k}')
        z = np.load(f); ev = z['split'] == 1; n = int(ev.sum())
        P16 = z['pv16'][ev].reshape(n, -1, z['pv16'].shape[-1]); G = z['gv'].astype(float).reshape(-1, P16.shape[-1])
        p16 = z['pp16'][ev].reshape(n, -1).astype(float); gp = z['gp'].astype(float).reshape(-1)
        D = P16.astype(float) - G[None]; ch = (D * D).mean(1); vis = (D * D).mean((1, 2))
        prop_full = ((p16 - gp[None]) ** 2).mean(1)
        b1 = ch @ w[:-1] + prop_full * w[-1]
        pt = p16.reshape(n, -1, dp).mean(1); r = (pt - b) @ Wi.T; rg = (gp.reshape(-1, dp).mean(0) - b) @ Wi.T
        pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
        vz = (vis - vis.mean()) / vis.std() + (pos - pos.mean()) / pos.std()
        m = z['margin'][ev]; s = z['success'][ev].astype(bool); st = z['official_state'][ev]
        agent = np.linalg.norm(st[:, :2] - z['goal_state'][:2], axis=1)
        # the stored margins must equal the official criterion recomputed from the stored states
        pos_d = np.linalg.norm(st[:, :4] - z['goal_state'][None, :4], axis=1); ang = np.abs(st[:, 4] - z['goal_state'][4]); ang = np.minimum(ang, 2 * np.pi - ang)
        if np.max(np.abs(np.maximum(pos_d / 20, ang / (np.pi / 9)) - m)) > 1e-9: issues.append(f'margin recomputation start {k}')
        j0, j1, jv = first_min(z['model_cost'][ev]), first_min(b1), first_min(vz)
        rows.append(dict(k=int(k), m0=m[j0], m1=m[j1], mv=m[jv], s0=s[j0], s1=s[j1], sv=s[jv],
                         sp_prop=spear(prop_full, agent), sp_pos=spear(pos, agent)))
    n = len(rows); M0, M1, MV = (np.array([r[c] for r in rows]) for c in ['m0', 'm1', 'mv'])
    S0, S1, SV = (int(sum(r[c] for r in rows)) for c in ['s0', 's1', 'sv'])
    sp_prop = float(np.mean([r['sp_prop'] for r in rows])); sp_pos = float(np.mean([r['sp_pos'] for r in rows]))
    ok = not [i for i in issues if not i.endswith('(Amendment 1)')]
    DV = MV - M0
    prim = 'INCOMPLETE' if (n < 8 or not ok) else ('SUPPORTED' if (DV.mean() <= -0.10 and SV >= S0 and np.sort(DV)[1:].mean() <= 0) else 'NOT_SUPPORTED')
    mech = 'INCOMPLETE' if (n < 8 or not ok) else ('REPLICATED' if (sp_pos >= 0.80 and sp_prop <= 0.60) else 'NOT_REPLICATED')
    out = {'n': n, 'issues': issues, 'B0': [float(M0.mean()), S0], 'B1': [float(M1.mean()), S1], 'VELDROP_Z': [float(MV.mean()), SV],
           'primary': prim, 'mean_D': float(DV.mean()), 'D_without_most_favourable': float(np.sort(DV)[1:].mean()) if n > 1 else None,
           'improved': int((DV < 0).sum()), 'worsened': int((DV > 0).sum()), 'mechanism': mech, 'mech_prop': sp_prop, 'mech_pos': sp_pos}
    print(json.dumps(out, indent=1))
    if ref:
        R = json.load(open(ref))['summary']
        P = R.get('PRIMARY_VELDROP_Z_vs_B0') or R.get('ADDENDUM_VELDROP_Z_vs_B0'); Mm = R.get('MECHANISM')
        chk = {'n': R['n'] == n,
               'B0_mean': abs(R['mean_selected_margin']['B0'][0] - out['B0'][0]) < 1e-9,
               'B1_mean': abs(R['mean_selected_margin']['B1'][0] - out['B1'][0]) < 1e-9,
               'VZ_mean': abs(R['mean_selected_margin']['VELDROP_Z'][0] - out['VELDROP_Z'][0]) < 1e-9,
               'succ': all(R['success_counts'][a] == out[a][1] for a in ['B0', 'B1', 'VELDROP_Z']),
               'primary': P['decision'] == prim,
               'mech_decision': Mm['decision'] == mech,
               'mech_values': abs(Mm['mean_spearman_proppos_agent'] - sp_pos) < 1e-9 and abs(Mm['mean_spearman_prop_agent'] - sp_prop) < 1e-9}
        print('agreement with locked analysis:', chk, 'ALL' if all(chk.values()) else 'MISMATCH')


if __name__ == '__main__':
    main(sys.argv[1:])
