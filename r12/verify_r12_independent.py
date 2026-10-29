"""Independent re-computation of R12 decisions (2026-10-29), written before the locked analysis was run.
numpy only; does not import r12_analyze.py. Recomputes every margin from final_state and goal_state with its own
implementation of the release criterion, rebuilds Q1 and Q2 from scratch, and compares with r12_analysis.json.
Usage: python3 -I verify_r12_independent.py r12_cohort.json r12_inloop.json r12_analysis.json
"""
import json, sys
from math import comb
import numpy as np


def margin_success(goal, cur):
    g = np.asarray(goal, float); c = np.asarray(cur, float)
    d_agent = g[0:2] - c[0:2]; d_block = g[2:4] - c[2:4]
    pos = float(np.sqrt(d_agent @ d_agent + d_block @ d_block))
    dth = (float(g[4]) - float(c[4])) % (2 * np.pi); ang = min(dth, 2 * np.pi - dth)
    return max(pos / 20.0, ang / (np.pi / 9)), (pos < 20) and (ang < np.pi / 9)


def exact_two_sided(a, b):
    n = a + b
    if n == 0: return 1.0
    tail = sum(comb(n, i) for i in range(0, min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(cp, rp, ap):
    coh = json.load(open(cp)); runs = json.load(open(rp))['runs']; A = json.load(open(ap))['summary']
    n = len(coh['starts']); M = {}; S = {}
    for s in coh['starts']:
        k = s['k']
        for arm in ['fixed_full', 'fixed_pos', 'raw', 'veldrop_z']:
            r = runs[f'{k}:{arm}']
            assert r['identity']['val_episode'] == s['val_episode'] and r['identity']['cem_seed'] == s['cem_seed']
            M[k, arm], S[k, arm] = margin_success(r['goal_state'], r['final_state'])
    idx = np.random.default_rng(12001).integers(0, n, (20000, n)); out = {}
    for q, new, ref in [('Q1', 'fixed_pos', 'fixed_full'), ('Q2', 'veldrop_z', 'raw')]:
        d = np.array([M[k, new] - M[k, ref] for k in range(n)])
        bm = d[idx].mean(axis=1); lo, hi = np.quantile(bm, [0.025, 0.975])
        g = sum(S[k, new] and not S[k, ref] for k in range(n)); l = sum(S[k, ref] and not S[k, new] for k in range(n))
        drop_best = np.sort(d)[1:].mean()
        sn = sum(S[k, new] for k in range(n)); sr = sum(S[k, ref] for k in range(n))
        dec = ('POSITION_HELPS' if q == 'Q1' else 'VELDROP_Z_HELPS') if (d.mean() <= -0.10 and sn >= sr and drop_best <= 0) else 'NOT_SHOWN'
        out[q] = {'mean': float(d.mean()), 'ci': [float(lo), float(hi)], 'gains_losses': [int(g), int(l)], 'succ_new_ref': [int(sn), int(sr)],
                  'lbo': float(drop_best), 'mcnemar': exact_two_sided(int(g), int(l)), 'decision': dec}
        L = A[q]
        out[q]['agrees_with_locked'] = bool(abs(L['mean_ci'][0] - d.mean()) < 1e-12 and abs(L['mean_ci'][1] - lo) < 1e-9 and abs(L['mean_ci'][2] - hi) < 1e-9
                                            and L['success_gains_losses'] == [g, l] and L['decision'] == dec and abs(L['without_most_favourable'] - drop_best) < 1e-12)
    print(json.dumps(out, indent=1)); print('ALL' if all(v['agrees_with_locked'] for v in out.values()) else 'DISAGREEMENT')


if __name__ == '__main__':
    main(*sys.argv[1:4])
