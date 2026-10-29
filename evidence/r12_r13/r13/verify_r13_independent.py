"""Independent re-computation of the R13 decision (2026-10-29), written before any R13 outcome was seen.
numpy only; does not import r13_analyze.py.
Usage: python3 -I verify_r13_independent.py r12_cohort.json r12_inloop.json r13_inloop.json r13_analysis.json
"""
import json, sys
from math import comb
import numpy as np


def margin_success(goal, cur):
    g = np.asarray(goal, float); c = np.asarray(cur, float)
    pos = float(np.sqrt(((g[0:2] - c[0:2]) ** 2).sum() + ((g[2:4] - c[2:4]) ** 2).sum()))
    dth = (float(g[4]) - float(c[4])) % (2 * np.pi); ang = min(dth, 2 * np.pi - dth)
    return max(pos / 20.0, ang / (np.pi / 9)), (pos < 20) and (ang < np.pi / 9)


def main(cp, r12p, r13p, ap):
    coh = json.load(open(cp)); R12 = json.load(open(r12p))['runs']; R13 = json.load(open(r13p))['runs']; A = json.load(open(ap))['summary']
    n = len(coh['starts']); d = []; sv = []; sh = []
    for s in coh['starts']:
        k = s['k']; a = R12[f'{k}:veldrop_z']; b = R13[f'{k}:hprop_std']
        assert a['q0_sha256'] == b['q0_sha256'] and b['identity']['cem_seed'] == s['cem_seed']
        ma, sa = margin_success(a['goal_state'], a['final_state']); mb, sb = margin_success(b['goal_state'], b['final_state'])
        d.append(ma - mb); sv.append(sa); sh.append(sb)
    d = np.array(d); sv = np.array(sv); sh = np.array(sh)
    lo, hi = np.quantile(d[np.random.default_rng(13001).integers(0, n, (20000, n))].mean(axis=1), [0.025, 0.975])
    g = int((sv & ~sh).sum()); l = int((sh & ~sv).sum()); lbo = float(np.sort(d)[1:].mean())
    dec = 'DROP_HELPS_FRESH' if (d.mean() <= -0.10 and sv.sum() >= sh.sum() and lbo <= 0) else 'NOT_SHOWN'
    mc = min(1.0, 2 * sum(comb(g + l, i) for i in range(min(g, l) + 1)) / 2 ** (g + l)) if g + l else 1.0
    L = A['PRIMARY']
    agree = (abs(L['mean_ci'][0] - d.mean()) < 1e-12 and abs(L['mean_ci'][1] - lo) < 1e-9 and abs(L['mean_ci'][2] - hi) < 1e-9
             and L['success_gains_losses'] == [g, l] and L['decision'] == dec and abs(L['without_most_favourable'] - lbo) < 1e-12)
    print(json.dumps({'mean': float(d.mean()), 'ci': [float(lo), float(hi)], 'gains_losses': [g, l], 'succ_veldrop_hprop': [int(sv.sum()), int(sh.sum())],
                      'lbo': lbo, 'mcnemar': mc, 'decision': dec}, indent=1))
    print('ALL' if agree else 'DISAGREEMENT')


if __name__ == '__main__':
    main(*sys.argv[1:5])
