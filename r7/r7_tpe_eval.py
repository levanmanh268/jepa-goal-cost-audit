"""R7-TPE step 2 (locked, run once): tail patch energy on the R6 TEST banks (Claude, 2026-10-27). Development.
Hyperparameters only from R7_TPE_LOCK.json (training cohort). Per start, 64-plan evaluation bank, argmin with
lowest-index ties: B0 (official exact cost), TPE = topk(k*), POW = powmean(q*), FIXEDMASK (K = 128, label-free
goal-vs-start mask), B1 (frozen R6 head; supervised reference, selections checked against R6), E-B0 and E-TPE on
encoded true endpoints (privileged), O (best in bank), random plan mean.
Primary: D = margin(TPE) - margin(B0). PROCEED iff mean D <= -0.10 AND TPE successes >= B0 successes AND mean D
without the most favourable start <= 0. Otherwise NO_SIGNAL (for this budget). Secondary descriptive only.
Usage: python3 -I r7_tpe_eval.py banks/test R7_TPE_LOCK.json R7_FROZEN_R6_HEAD.json r6/analysis_r6a.json out.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import energies as E, tail_energies as T

DELTA = 0.10


def pick(c):
    c = np.asarray(c, float); lo = c.min(); return int(np.flatnonzero(c <= lo + 1e-12 * max(1., abs(lo)))[0])


def boot(x, B=20000, seed=36):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main(bank_dir, lockp, headp, r6p, outp):
    L = json.load(open(lockp)); k, q = L['k_star'], L['q_star']
    w = np.array(json.load(open(headp))['weights']); R6 = json.load(open(r6p)); r6sel = {r['k']: r for r in R6['per_test_start']}
    d = Path(bank_dir); idx = json.loads((d / 'r7_index.json').read_text()); rows = []
    for kk, rec in sorted(idx['starts'].items(), key=lambda kv: int(kv[0])):
        z = np.load(d / f'start_{int(kk):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
        P = z['pv16'][ev].astype(np.float64).reshape(n, -1, C); p = z['pp16'][ev].astype(np.float64).reshape(n, -1)
        Tv = z['tv16'][ev].astype(np.float64).reshape(n, -1, C); tp = z['tp16'][ev].astype(np.float64).reshape(n, -1)
        G = z['gv'].astype(np.float64).reshape(-1, C); gp = z['gp'].astype(np.float64).reshape(-1)
        Z0 = z['z0v'].astype(np.float64).reshape(-1, C); m = z['margin'][ev]; mask = T.fixedmask_indices(G, Z0, 128)
        sc = {'B0': z['model_cost'][ev], 'TPE': T.topk(P, G, p, gp, k), 'POW': T.powmean(P, G, p, gp, q),
              'FIXEDMASK': T.fixedmask(P, G, p, gp, mask), 'B1': E.head(P, G, p, gp, w),
              'E_B0': E.official(Tv, G, tp, gp), 'E_TPE': T.topk(Tv, G, tp, gp, k)}
        sel = {name: pick(v) for name, v in sc.items()}
        rows.append({'k': int(kk), 'traj': rec['traj'], **{f'{nm}_margin': float(m[j]) for nm, j in sel.items()}, 'selected': sel,
                     'O': float(m.min()), 'random_mean': float(m.mean()),
                     'spearman': {nm: float(spearmanr(v, m).statistic) for nm, v in sc.items()},
                     'b1_matches_r6': bool(sel['B1'] == r6sel[int(kk)]['B1']['selected']), 'b0_matches_r6': bool(sel['B0'] == r6sel[int(kk)]['B0']['selected'])})
        print('start', kk, 'scored', flush=True)
    names = ['B0', 'TPE', 'POW', 'FIXEDMASK', 'B1', 'E_B0', 'E_TPE']
    M = {nm: np.array([r[f'{nm}_margin'] for r in rows]) for nm in names}
    D = M['TPE'] - M['B0']; Ds = np.sort(D); succ = {nm: int((M[nm] < 1).sum()) for nm in names}
    proceed = bool(D.mean() <= -DELTA and succ['TPE'] >= succ['B0'] and Ds[1:].mean() <= 0)
    gains = int(((M['TPE'] < 1) & (M['B0'] >= 1)).sum()); losses = int(((M['TPE'] >= 1) & (M['B0'] < 1)).sum())
    S = {'n': len(rows), 'k_star': k, 'q_star': q, 'decision': 'PROCEED (development signal)' if proceed else 'NO_SIGNAL (for this budget)',
         'primary_D_TPE_minus_B0': boot(D), 'D_improved': int((D < 0).sum()), 'D_worsened': int((D > 0).sum()), 'D_mean_excluding_best': float(Ds[1:].mean()),
         'paired_success_gains_losses': [gains, losses], 'success_counts': succ, 'mean_selected_margin': {nm: boot(M[nm]) for nm in names},
         'O_mean': float(np.mean([r['O'] for r in rows])), 'random_mean': float(np.mean([r['random_mean'] for r in rows])),
         'secondary': {f'TPE_minus_{nm}': boot(M['TPE'] - M[nm]) for nm in ['POW', 'FIXEDMASK', 'B1']} | {'E_TPE_minus_E_B0': boot(M['E_TPE'] - M['E_B0'])},
         'within_bank_spearman_mean': {nm: float(np.mean([r['spearman'][nm] for r in rows])) for nm in names},
         'heterogeneity_spearman_D_vs_B0_margin': float(spearmanr(D, M['B0']).statistic),
         'checks': {'b1_selections_match_r6': all(r['b1_matches_r6'] for r in rows), 'b0_selections_match_r6': all(r['b0_matches_r6'] for r in rows)}}
    json.dump({'summary': S, 'per_start': rows}, open(outp, 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:6])
