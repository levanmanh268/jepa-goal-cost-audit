"""Recompute R8 / R10 fixed-bank decisions and R9 in-loop decisions from the compact packet files (Claude, 2026-10-27).
numpy only; no project code. Intended for an external reviewer working from the zip.

  python3 -I recompute_from_components.py bank  r8/r8_components.npz            # R8: B0, B1, VELDROP-z, M1
  python3 -I recompute_from_components.py bank  r10/r10_components.npz          # R10: B0, VELDROP-z, M2 (B1 descriptive)
  python3 -I recompute_from_components.py inloop r8/r9_inloop.json r8/r8_components.npz   # R9: both decisions

Rules (R8_ADDENDUM_LOCK.md; R10_PROTOCOL_LOCK.md): argmin with the lowest index on ties; SUPPORTED iff mean D <= -0.10
AND successes not lower AND mean D without the most favourable start <= 0; mechanism REPLICATED iff mean within-bank
Spearman(prop_pos, agent distance) >= 0.80 AND mean Spearman(prop_full, agent distance) <= 0.60; n >= 8.
"""
import json, sys
import numpy as np


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


def rule(D, s_new, s_ref, yes, no):
    if len(D) < 8: return 'INCOMPLETE'
    return yes if (D.mean() <= -0.10 and s_new >= s_ref and np.sort(D)[1:].mean() <= 0) else no


def official(goal, cur):
    pos = float(np.linalg.norm(np.asarray(goal[:4]) - np.asarray(cur[:4]))); ang = abs(float(goal[4]) - float(cur[4]))
    ang = min(ang, 2 * np.pi - ang); return max(pos / 20.0, ang / (np.pi / 9)), bool(pos < 20 and ang < np.pi / 9)


def bank(path):
    from pathlib import Path
    Z = np.load(path); w = None
    here = Path(__file__).resolve().parent
    for cand in [Path('R7_FROZEN_R6_HEAD.json'), here / 'R7_FROZEN_R6_HEAD.json', here.parent / 'r7' / 'R7_FROZEN_R6_HEAD.json']:
        if cand.exists():
            w = np.array(json.load(open(cand))['weights'], float); break
    n = Z['vis'].shape[0]; rows = []
    for k in range(n):
        vis, pos, pf, m, s = Z['vis'][k], Z['prop_pos'][k], Z['prop_full'][k], Z['margin'][k], Z['success'][k].astype(bool)
        vz = (vis - vis.mean()) / vis.std() + (pos - pos.mean()) / pos.std()
        agent = np.linalg.norm(Z['official_state'][k][:, :2] - Z['goal_state'][k][:2], axis=1)
        j0, jv = first_min(Z['model_cost'][k]), first_min(vz)
        row = dict(m0=m[j0], mv=m[jv], s0=s[j0], sv=s[jv], sp_prop=spear(pf, agent), sp_pos=spear(pos, agent), O=m.min())
        if w is not None:
            j1 = first_min(Z['ch'][k] @ w[:-1] + pf * w[-1]); row.update(m1=m[j1], s1=s[j1])
        rows.append(row)
    M0, MV = np.array([r['m0'] for r in rows]), np.array([r['mv'] for r in rows]); S0, SV = sum(r['s0'] for r in rows), sum(r['sv'] for r in rows)
    out = {'n': n, 'B0': [float(M0.mean()), int(S0)], 'VELDROP_Z': [float(MV.mean()), int(SV)],
           'VELDROP_Z_vs_B0': rule(MV - M0, SV, S0, 'SUPPORTED', 'NOT_SUPPORTED'), 'mean_D': float((MV - M0).mean()),
           'D_without_most_favourable': float(np.sort(MV - M0)[1:].mean()), 'improved': int((MV < M0).sum()), 'worsened': int((MV > M0).sum()),
           'mech_prop_full': float(np.mean([r['sp_prop'] for r in rows])), 'mech_prop_pos': float(np.mean([r['sp_pos'] for r in rows])),
           'O_best_in_bank': [float(np.mean([r['O'] for r in rows])), int(sum(r['O'] < 1 for r in rows))]}
    out['mechanism'] = 'INCOMPLETE' if n < 8 else ('REPLICATED' if (out['mech_prop_pos'] >= 0.80 and out['mech_prop_full'] <= 0.60) else 'NOT_REPLICATED')
    if w is not None:
        M1 = np.array([r['m1'] for r in rows]); S1 = sum(r['s1'] for r in rows)
        out['B1'] = [float(M1.mean()), int(S1)]; out['B1_vs_B0'] = rule(M1 - M0, S1, S0, 'REPLICATED', 'NOT_REPLICATED')
    print(json.dumps(out, indent=1))


def inloop(runs_path, comp_path):
    R = json.load(open(runs_path))['runs']; Z = np.load(comp_path); arms = ['b0', 'veldrop_z', 'b1']; n = 0
    while all(f'{n}:{a}' in R for a in arms): n += 1
    M = {a: np.zeros(n) for a in arms}; S = {a: np.zeros(n, bool) for a in arms}
    for k in range(n):
        for a in arms:
            M[a][k], S[a][k] = official(Z['goal_state'][k], R[f'{k}:{a}']['final_state'])
    raw = [v for v in R.values() if v['arm'] == 'raw']
    repro = float(np.max(np.abs(np.array(raw[0]['final_mean']) - Z['proposal_mean'][raw[0]['identity']['k']]))) if raw else None
    b0_ok = all(float(np.max(np.abs(np.array(R[f'{k}:b0']['final_mean']) - Z['proposal_mean'][k]))) == 0.0 for k in range(n))
    out = {'n': n, 'repro_maxabs': repro, 'b0_plans_equal_R8_proposal_mean': b0_ok,
           'mean_margin': {a: float(M[a].mean()) for a in arms}, 'successes': {a: int(S[a].sum()) for a in arms}}
    ok = repro == 0.0 and b0_ok
    for a, yes, no in [('veldrop_z', 'SUPPORTED', 'NOT_SUPPORTED'), ('b1', 'SIGNAL', 'NO_SIGNAL')]:
        D = M[a] - M['b0']
        out[a + '_vs_b0'] = {'decision': rule(D, S[a].sum(), S['b0'].sum(), yes, no) if ok else 'INCOMPLETE', 'mean_D': float(D.mean()),
                             'D_without_most_favourable': float(np.sort(D)[1:].mean()), 'improved': int((D < 0).sum()), 'worsened': int((D > 0).sum())}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'bank': bank(sys.argv[2])
    else: inloop(sys.argv[2], sys.argv[3])
