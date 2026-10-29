"""Five-term decomposition of the official proprio goal term on the stored evaluation banks (Claude, 2026-10-28;
post hoc diagnostic requested by ChatGPT's R8-R10 review; descriptive, no decision).

For each plan: d_i = predicted - goal proprio token i (T tokens, dim m), h = mean_i d_i, dr = pinv(W) h,
u = W_p dr_p, v = W_v dr_v, rho = h - W dr (orthogonal to range(W)). Then
  official term  Pi = mean_i ||d_i||^2 / m = (||u||^2 + ||v||^2 + 2<u,v> + ||rho||^2 + mean_i ||d_i - h||^2) / m.
VELDROP position term = ||u||^2 / m. Checks the identity numerically and reports mean shares, sign of the cross term and
within-bank Spearman of each part with the true final agent distance.
Usage: python3 -I proprio_decomposition.py <cohort dir> <checkpoint> <out.json> [--first N]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr


def main(argv):
    first = None
    if '--first' in argv: i = argv.index('--first'); first = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    d, ckpt, outp = Path(argv[0]), argv[1], argv[2]
    idx = json.loads((d / 'r6_index.json').read_text()); keys = sorted(idx['starts'], key=int)
    if first: keys = keys[:first]
    W = torch.load(ckpt, map_location='cpu', weights_only=False)['proprio_encoder']['module.patch_embed.weight'][:, :, 0].double().numpy()
    m = W.shape[0]; Wi = np.linalg.pinv(W); Wp, Wv = W[:, :2], W[:, 2:]
    rows = []; maxid = 0.0
    for k in keys:
        f = d / f'start_{int(k):03d}.npz'
        assert hashlib.sha256(f.read_bytes()).hexdigest() == idx['starts'][k]['npz_sha256']
        z = np.load(f); ev = z['split'] == 1; n = int(ev.sum())
        P = z['pp16'][ev].astype(np.float64).reshape(n, -1, m); G = z['gp'].astype(np.float64).reshape(-1, m)
        D = P - G[None]; h = D.mean(1); dr = h @ Wi.T
        u = dr[:, :2] @ Wp.T; v = dr[:, 2:] @ Wv.T; rho = h - dr @ W.T
        tok = ((D - h[:, None, :]) ** 2).sum(2).mean(1)
        parts = {'pos': (u ** 2).sum(1) / m, 'vel': (v ** 2).sum(1) / m, 'cross': 2 * (u * v).sum(1) / m, 'resid': (rho ** 2).sum(1) / m, 'token_var': tok / m}
        full = (D ** 2).sum(2).mean(1) / m
        maxid = max(maxid, float(np.max(np.abs(sum(parts.values()) - full) / np.maximum(full, 1e-30))))
        st = z['official_state'][ev]; agent = np.linalg.norm(st[:, :2] - z['goal_state'][:2], axis=1)
        row = {'k': int(k), 'mean_full': float(full.mean())}
        for nm, a in parts.items():
            row[f'share_{nm}'] = float(a.mean() / full.mean())
            row[f'spearman_{nm}'] = float(spearmanr(a, agent).statistic) if np.std(a) > 0 else None
        row['spearman_full'] = float(spearmanr(full, agent).statistic)
        row['net_removed_fraction'] = float(1 - parts['pos'].mean() / full.mean())
        rows.append(row)
    keysum = [c for c in rows[0] if c.startswith('share_') or c.startswith('spearman_') or c == 'net_removed_fraction']
    summ = {c: float(np.mean([r[c] for r in rows if r[c] is not None])) for c in keysum}
    out = {'checkpoint': Path(ckpt).name, 'n_starts': len(rows), 'max_rel_identity_error': maxid, 'mean_over_starts': summ, 'per_start': rows}
    json.dump(out, open(outp, 'w'), indent=1); print(json.dumps({k: out[k] for k in ['checkpoint', 'n_starts', 'max_rel_identity_error', 'mean_over_starts']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
