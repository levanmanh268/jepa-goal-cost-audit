"""DEVELOPMENT decomposition on the 26 R6/R7 starts (all development). Claude, R7, 2026-10-27. Nothing touches R8.

Where does the selection gap (official cost vs best plan in bank) come from? Rules compared on each 64-plan evaluation bank:
  official (predicted latents), E-official (true encoded latents), prop-pred (predicted proprio term only),
  prop-true (true encoded proprio term only), agent-oracle (true final agent distance, privileged),
  agent+block oracle (true position norm, privileged), O (true margin, privileged).
Also: MCH-0 fitted on all 26 starts (weights printed: which channels, proprio weight), for the record.
Usage: python3 -I dev_decompose.py
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
from dev_screen2 import pick, comp_err, design

ROOT = Path(str(_WM_ROOT / 'r7/banks'))


def main():
    S = []
    for cohort in ['train', 'test']:
        d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
        for k, rec in sorted(idx['starts'].items(), key=lambda kv: int(kv[0])):
            z = np.load(d / f'start_{int(k):03d}.npz'); n = len(z['margin']); C = z['pv16'].shape[-1]
            P = z['pv16'].astype(np.float64).reshape(n, -1, C); p = z['pp16'].astype(np.float64).reshape(n, -1)
            Tv = z['tv16'].astype(np.float64).reshape(n, -1, C); tp = z['tp16'].astype(np.float64).reshape(n, -1)
            G = z['gv'].astype(np.float64).reshape(-1, C); gp = z['gp'].astype(np.float64).reshape(-1)
            a, b, th = comp_err(z['official_state'], z['goal_state'])
            ch, pr = E.channel_energies(P, G, p, gp)
            S.append(dict(cohort=cohort, traj=rec['traj'], split=z['split'], margin=z['margin'], b0=z['model_cost'],
                          eb0=E.official(Tv, G, tp, gp), ppred=E.proprio_term(p, gp), ptrue=E.proprio_term(tp, gp),
                          perr=np.sqrt(((p - tp) ** 2).mean(1)), pspread=np.sqrt(((p - p.mean(0)) ** 2).mean(1)).mean(),
                          agent=a, posn=np.sqrt(a ** 2 + b ** 2), comp=np.column_stack([a, b, th]), FA=np.column_stack([ch, pr])))
    coh = np.array([s['cohort'] for s in S])
    rules = {'official': 'b0', 'E-official (true latents)': 'eb0', 'prop-pred only': 'ppred', 'prop-true only': 'ptrue',
             'agent-oracle (privileged)': 'agent', 'agent+block pos oracle (priv.)': 'posn', 'O = margin oracle': 'margin'}
    print(f"{'rule':32s} {'all26':>7s} {'succ':>4s} | {'tr16':>6s} {'te10':>6s} | {'agent':>6s} {'block':>6s} {'angle':>6s}")
    for name, key in rules.items():
        js = [int(np.flatnonzero(s['split'] == 1)[pick(s[key][s['split'] == 1])]) for s in S]
        m = np.array([S[i]['margin'][j] for i, j in enumerate(js)]); c = np.array([S[i]['comp'][j] for i, j in enumerate(js)])
        print(f"{name:32s} {m.mean():7.3f} {(m < 1).sum():4d} | {m[coh == 'train'].mean():6.3f} {m[coh == 'test'].mean():6.3f} | "
              f"{c[:, 0].mean():6.1f} {c[:, 1].mean():6.1f} {c[:, 2].mean():6.1f}")
    pe = np.concatenate([s['perr'] for s in S]); ps = np.array([s['pspread'] for s in S])
    print(f"\nproprio prediction RMSE (pred vs true encoded, per plan): median {np.median(pe):.4f}; "
          f"mean within-bank spread of predicted proprio {ps.mean():.4f}")
    rho = [np.corrcoef(np.argsort(np.argsort(s['ppred'])), np.argsort(np.argsort(s['ptrue'])))[0, 1] for s in S]
    rho_a = [np.corrcoef(np.argsort(np.argsort(s['ptrue'])), np.argsort(np.argsort(s['agent'])))[0, 1] for s in S]
    print(f"Spearman(prop-pred, prop-true) per bank: mean {np.mean(rho):.3f}; Spearman(prop-true, true agent distance): mean {np.mean(rho_a):.3f}")
    Z, y, sd = design(S); w, _ = nnls(Z, y, maxiter=20000); w = w / sd
    nz = np.flatnonzero(w > 0)
    print(f"\nMCH-0 fitted on all 26 starts: {len(nz)} positive weights; proprio feature index 384 weight {w[384]:.4g}; "
          f"channels {[int(i) for i in nz if i < 384]}")
    contrib = np.array([np.concatenate([s['FA'][:, i] for s in S]).std() * w[i] for i in nz])
    print('share of score spread by feature:', {int(i): round(float(c / contrib.sum()), 3) for i, c in zip(nz, contrib)})
    json.dump({'w': w.tolist()}, open(str(_WM_ROOT / 'r7/dev_mch0_all26.json'), 'w'))


if __name__ == '__main__':
    main()
