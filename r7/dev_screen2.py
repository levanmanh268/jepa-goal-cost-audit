"""DEVELOPMENT screen 2 (26 R6/R7 starts, all development). Claude, R7, 2026-10-27. Nothing here touches R8.

Pre-declared variants (all printed, forking paths disclosed):
  MCH-0 / MCH-1: monotone channel head = NONNEGATIVE weights on the 385 B1 features (384 channel energies + proprio),
                 fitted by NNLS (lam=0) or NNLS with ridge rows (lam=1) on standardized, within-bank-centred features,
                 target within-bank-centred margin. Leave-one-trajectory-out CV.
  B1-26: signed ridge (the B1 recipe) refitted under the same CV (reference).
  ENT-1: label-free entity-balanced visual energy: goal patches split into foreground/background by Otsu on the distance
         of each goal patch to the goal's median patch; E_vis = 0.5 mean_FG(e) + 0.5 mean_BG(e).
  ENT-2: same, foreground split further into 4-connected components; E_vis = mean over groups {components, BG} of the
         group-mean error.
  Each ENT variant is combined with proprio in two pre-declared ways: official weight (+0.1 prop) and within-bank z-score sum.
Also prints component errors of the selected plans (agent position, block position, block angle), mean over starts.
Usage: python3 -I dev_screen2.py
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls
from scipy import ndimage
sys.path.insert(0, str(_WM_ROOT / 'r7'))
import energies as E, tail_energies as T

ROOT = Path(str(_WM_ROOT / 'r7/banks'))


def pick(c):
    lo = c.min(); return int(np.flatnonzero(c <= lo + 1e-12 * max(1., abs(lo)))[0])


def otsu(x, nb=64):
    h, edges = np.histogram(x, nb); c = 0.5 * (edges[1:] + edges[:-1]); w = h / h.sum()
    best, thr = -1, c[0]
    for t in range(1, nb):
        w0, w1 = w[:t].sum(), w[t:].sum()
        if w0 == 0 or w1 == 0: continue
        m0, m1 = (w[:t] * c[:t]).sum() / w0, (w[t:] * c[t:]).sum() / w1
        v = w0 * w1 * (m0 - m1) ** 2
        if v > best: best, thr = v, edges[t]
    return thr


def groups_for_goal(G):
    ld = np.log(((G - np.median(G, 0)) ** 2).sum(1) + 1e-12); fg = ld > otsu(ld)
    lab, ncomp = ndimage.label(fg.reshape(16, 16))
    comps = [np.flatnonzero(lab.reshape(-1) == c) for c in range(1, ncomp + 1)]
    return fg, comps


def comp_err(fs, gs):
    a = np.linalg.norm(fs[:, 0:2] - gs[0:2], axis=1); b = np.linalg.norm(fs[:, 2:4] - gs[2:4], axis=1)
    th = np.abs(fs[:, 4] - gs[4]) % (2 * np.pi); th = np.minimum(th, 2 * np.pi - th)
    return a, b, np.degrees(th)


def zsum(a, b, ev):
    return (a - a[ev].mean()) / (a[ev].std() + 1e-12) + (b - b[ev].mean()) / (b[ev].std() + 1e-12)


def load_all():
    S = []
    for cohort in ['train', 'test']:
        d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
        for k, rec in sorted(idx['starts'].items(), key=lambda kv: int(kv[0])):
            z = np.load(d / f'start_{int(k):03d}.npz'); n = len(z['margin']); C = z['pv16'].shape[-1]
            P = z['pv16'].astype(np.float64).reshape(n, -1, C); p = z['pp16'].astype(np.float64).reshape(n, -1)
            G = z['gv'].astype(np.float64).reshape(-1, C); gp = z['gp'].astype(np.float64).reshape(-1)
            e = T.patch_errors(P, G); fg, comps = groups_for_goal(G)
            ent1 = 0.5 * e[:, fg].mean(1) + 0.5 * e[:, ~fg].mean(1)
            ent2 = np.mean([e[:, c].mean(1) for c in comps] + [e[:, ~fg].mean(1)], 0)
            ch, pr = E.channel_energies(P, G, p, gp)
            a, b, th = comp_err(z['official_state'], z['goal_state'])
            S.append({'cohort': cohort, 'traj': rec['traj'], 'split': z['split'], 'margin': z['margin'], 'b0': z['model_cost'],
                      'prop': E.proprio_term(p, gp), 'ent1': ent1, 'ent2': ent2, 'FA': np.column_stack([ch, pr]),
                      'nfg': int(fg.sum()), 'ncomp': len(comps), 'comp': np.column_stack([a, b, th])})
    return S


def design(train):
    X = np.concatenate([s['FA'] - s['FA'].mean(0) for s in train]); y = np.concatenate([s['margin'] - s['margin'].mean() for s in train])
    sd = X.std(0); sd[sd < 1e-12] = 1
    return X / sd, y, sd


def fit_signed(train, lam=1.0):
    Z, y, sd = design(train)
    return np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ y) / sd


def fit_nonneg(train, lam):
    Z, y, sd = design(train)
    if lam > 0:
        Z = np.vstack([Z, np.sqrt(lam * len(Z)) * np.eye(Z.shape[1])]); y = np.concatenate([y, np.zeros(Z.shape[1])])
    w, _ = nnls(Z, y, maxiter=20000)
    return w / sd


def main():
    S = load_all(); trajs = sorted({s['traj'] for s in S}); coh = np.array([s['cohort'] for s in S])
    print(f'{len(S)} starts, {len(trajs)} trajectories; goal foreground patches mean {np.mean([s["nfg"] for s in S]):.1f}, '
          f'components mean {np.mean([s["ncomp"] for s in S]):.2f} (min {min(s["ncomp"] for s in S)}, max {max(s["ncomp"] for s in S)})')
    rules = {
        'official': lambda s, ev: s['b0'],
        'hprop_std': lambda s, ev: zsum(s['b0'] - 0.1 * s['prop'], s['prop'], ev),
        'ENT-1 +0.1prop': lambda s, ev: s['ent1'] + 0.1 * s['prop'],
        'ENT-1 zsum': lambda s, ev: zsum(s['ent1'], s['prop'], ev),
        'ENT-2 +0.1prop': lambda s, ev: s['ent2'] + 0.1 * s['prop'],
        'ENT-2 zsum': lambda s, ev: zsum(s['ent2'], s['prop'], ev),
    }
    sel = {}
    for name, fn in rules.items():
        sel[name] = []
        for s in S:
            ev = s['split'] == 1; sc = fn(s, ev); j = np.flatnonzero(ev)[pick(sc[ev])]; sel[name].append(j)
    for name, fitter in [('B1-26 signed ridge', lambda tr: fit_signed(tr, 1.0)), ('MCH-0 nonneg', lambda tr: fit_nonneg(tr, 0.0)),
                         ('MCH-1 nonneg+ridge', lambda tr: fit_nonneg(tr, 1.0))]:
        sel[name] = [None] * len(S); nneg = []
        for t in trajs:
            tr = [s for s in S if s['traj'] != t]; w = fitter(tr); nneg.append(int((w > 0).sum()))
            for i, s in enumerate(S):
                if s['traj'] != t: continue
                ev = s['split'] == 1; sc = s['FA'] @ w; sel[name][i] = np.flatnonzero(ev)[pick(sc[ev])]
        print(f'{name}: nonzero/positive weights per fold, mean {np.mean(nneg):.1f}')
    print(f"\n{'rule':24s} {'all26':>7s} {'succ':>4s} | {'tr16':>6s} {'te10':>6s} | {'agent':>6s} {'block':>6s} {'angle':>6s}")
    for name, js in sel.items():
        m = np.array([S[i]['margin'][j] for i, j in enumerate(js)]); c = np.array([S[i]['comp'][j] for i, j in enumerate(js)])
        print(f"{name:24s} {m.mean():7.3f} {(m < 1).sum():4d} | {m[coh == 'train'].mean():6.3f} {m[coh == 'test'].mean():6.3f} | "
              f"{c[:, 0].mean():6.1f} {c[:, 1].mean():6.1f} {c[:, 2].mean():6.1f}")
    orc = [int(np.flatnonzero(s['split'] == 1)[pick(s['margin'][s['split'] == 1])]) for s in S]
    m = np.array([S[i]['margin'][j] for i, j in enumerate(orc)]); c = np.array([S[i]['comp'][j] for i, j in enumerate(orc)])
    print(f"{'O (best in bank)':24s} {m.mean():7.3f} {(m < 1).sum():4d} | {m[coh == 'train'].mean():6.3f} {m[coh == 'test'].mean():6.3f} | "
          f"{c[:, 0].mean():6.1f} {c[:, 1].mean():6.1f} {c[:, 2].mean():6.1f}")
    # which term decides the margin in the evaluation banks
    fr = []; agent_share = []
    for s in S:
        ev = s['split'] == 1; c = s['comp'][ev]; pos = np.sqrt(c[:, 0] ** 2 + c[:, 1] ** 2) / 20; ang = c[:, 2] / 20
        fr.append(np.mean(pos >= ang)); agent_share.append(np.median(c[:, 0] ** 2 / (c[:, 0] ** 2 + c[:, 1] ** 2 + 1e-12)))
    print(f'\nevaluation banks: fraction of plans whose margin is set by position (not angle): {np.mean(fr):.3f}; '
          f'median agent share of the squared position error: {np.median(agent_share):.3f}')


if __name__ == '__main__':
    main()
