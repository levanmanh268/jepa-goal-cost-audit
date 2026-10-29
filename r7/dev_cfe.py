"""DEVELOPMENT (R6/R7 cohorts only, 26 starts; both cohorts are development data by now). Claude, R7, 2026-10-27.

Candidate CFE (calibrated factor energy): E = sum_f w_f * E_f with w_f >= 0 over a few LABEL-FREE factor energies of
the predicted endpoint (official visual mean, proprio term, tail mean of patch errors). Only the K nonnegative weights
use labels. Nonnegative weights keep E a nonnegative sum of nonnegative energies (monotone in every factor), unlike
the signed 385-weight B1 head. Questions: (1) does CFE match B1 under leave-one-trajectory-out CV? (2) how many
labelled starts does it need? Every variant tried is printed (forking paths disclosed). Nothing here touches R8.
Usage: python3 -I dev_cfe.py
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, str(_WM_ROOT / 'r7'))
import energies as E, tail_energies as T

ROOT = Path(str(_WM_ROOT / 'r7/banks'))


def pick(c):
    lo = c.min(); return int(np.flatnonzero(c <= lo + 1e-12 * max(1., abs(lo)))[0])


def load_all():
    S = []
    for cohort in ['train', 'test']:
        d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
        for k, rec in sorted(idx['starts'].items(), key=lambda kv: int(kv[0])):
            z = np.load(d / f'start_{int(k):03d}.npz'); n = len(z['margin']); C = z['pv16'].shape[-1]
            P = z['pv16'].astype(np.float64).reshape(n, -1, C); p = z['pp16'].astype(np.float64).reshape(n, -1)
            G = z['gv'].astype(np.float64).reshape(-1, C); gp = z['gp'].astype(np.float64).reshape(-1)
            e = T.patch_errors(P, G); es = -np.sort(-e, 1)
            ch, pr = E.channel_energies(P, G, p, gp)
            feats = {'vis': e.mean(1), 'prop': E.proprio_term(p, gp), 'tail2': es[:, :2].mean(1), 'tail8': es[:, :8].mean(1),
                     'pow8': np.mean(e ** 8, 1) ** (1 / 8)}
            S.append({'cohort': cohort, 'k': int(k), 'traj': rec['traj'], 'split': z['split'], 'margin': z['margin'], 'b0': z['model_cost'],
                      'F': feats, 'FA': np.column_stack([ch, pr])})
    return S


def fit_nnls(train, names):
    X = np.concatenate([np.column_stack([s['F'][n] for n in names]) - np.column_stack([s['F'][n] for n in names]).mean(0) for s in train])
    y = np.concatenate([s['margin'] - s['margin'].mean() for s in train])
    sc = X.std(0); sc[sc < 1e-12] = 1
    w, _ = nnls(X / sc, y)
    return w / sc


def fit_ridge_fa(train, lam=1.0):
    X = np.concatenate([s['FA'] - s['FA'].mean(0) for s in train]); y = np.concatenate([s['margin'] - s['margin'].mean() for s in train])
    sd = X.std(0); sd[sd < 1e-12] = 1; Z = X / sd
    return np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ y) / sd


def evaluate(S, score_fn):
    out = []
    for s in S:
        ev = s['split'] == 1; j = pick(score_fn(s)[ev]); out.append(s['margin'][ev][j])
    return np.array(out)


def main():
    S = load_all(); trajs = sorted({s['traj'] for s in S})
    print(f'{len(S)} development starts, {len(trajs)} trajectories (leave-one-trajectory-out CV)')
    res = {}
    res['official'] = evaluate(S, lambda s: s['b0'])
    res['hprop_std (label-free)'] = evaluate(S, lambda s: (s['F']['vis'] - s['F']['vis'][s['split'] == 1].mean()) / (s['F']['vis'][s['split'] == 1].std() + 1e-12)
                                             + (s['F']['prop'] - s['F']['prop'][s['split'] == 1].mean()) / (s['F']['prop'][s['split'] == 1].std() + 1e-12))
    families = {'CFE[vis,prop]': ['vis', 'prop'], 'CFE[vis,prop,tail2]': ['vis', 'prop', 'tail2'], 'CFE[vis,prop,tail8]': ['vis', 'prop', 'tail8'],
                'CFE[vis,prop,tail2,tail8,pow8]': ['vis', 'prop', 'tail2', 'tail8', 'pow8']}
    weights = {}
    for name, names in list(families.items()) + [('B1-type signed FA ridge (lam=1)', None)]:
        sel = np.zeros(len(S)); wl = []
        for t in trajs:
            tr = [s for s in S if s['traj'] != t]; te = [i for i, s in enumerate(S) if s['traj'] == t]
            if names is None:
                w = fit_ridge_fa(tr); fn = lambda s, w=w: s['FA'] @ w
            else:
                w = fit_nnls(tr, names); wl.append(w); fn = lambda s, w=w, names=names: np.column_stack([s['F'][n] for n in names]) @ w
            for i in te:
                s = S[i]; ev = s['split'] == 1; sel[i] = s['margin'][ev][pick(fn(s)[ev])]
        res[name + ' (LOTO-CV)'] = sel
        if wl: weights[name] = np.mean(wl, 0).round(4).tolist()
    print(f"{'rule':44s} {'all26':>7s} {'succ':>4s} | {'R6 train16':>10s} {'R6 test10':>9s}")
    coh = np.array([s['cohort'] for s in S])
    for n, v in res.items():
        print(f"{n:44s} {v.mean():7.3f} {(v < 1).sum():4d} | {v[coh == 'train'].mean():10.3f} {v[coh == 'test'].mean():9.3f}")
    print('mean CFE weights over folds (natural units):', json.dumps(weights))
    # label-budget curve for CFE[vis,prop,tail8]: fit on m random training trajectories, evaluate on all others
    rng = np.random.default_rng(271)
    print('label budget (CFE[vis,prop,tail8], fit on m trajectories, evaluated on the remaining starts; 50 draws):')
    for m in [1, 2, 4, 8]:
        vals = []
        for _ in range(50):
            fit_t = set(rng.choice(trajs, m, replace=False).tolist())
            tr = [s for s in S if s['traj'] in fit_t]; te = [s for s in S if s['traj'] not in fit_t]
            w = fit_nnls(tr, ['vis', 'prop', 'tail8'])
            v = evaluate(te, lambda s, w=w: np.column_stack([s['F'][n] for n in ['vis', 'prop', 'tail8']]) @ w)
            v0 = evaluate(te, lambda s: s['b0'])
            vals.append((v.mean() - v0.mean(), (v < 1).sum() - (v0 < 1).sum()))
        vals = np.array(vals)
        print(f'  m={m}: mean D vs official {vals[:, 0].mean():+.3f} (5-95%: {np.quantile(vals[:, 0], .05):+.3f}, {np.quantile(vals[:, 0], .95):+.3f}); mean success diff {vals[:, 1].mean():+.2f}')


if __name__ == '__main__':
    main()
