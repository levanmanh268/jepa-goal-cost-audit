"""Compact per-plan proprio diagnostics for external review (Claude, 2026-10-28; requested by ChatGPT's R8-R10 review).
For each evaluation plan of each start: decoded (x, y, vx, vy) of the predicted and of the encoded-true endpoint (through
pinv(W) of the model's own encoder, from the stored 16-bit tokens), the decoded goal, the five energy parts of the
official proprio term (position, velocity, cross, residual, token variation), the full term, margin, success and the
true final agent distance. Also stores W and b. Usage: python3 -I export_proprio_diag.py <cohort> <checkpoint> <out.npz>
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch


def main(cohort, ckpt, outp):
    d = Path(cohort); idx = json.loads((d / 'r6_index.json').read_text()); keys = sorted(idx['starts'], key=int)
    W = torch.load(ckpt, map_location='cpu', weights_only=False)['proprio_encoder']
    b = W['module.patch_embed.bias'].double().numpy(); W = W['module.patch_embed.weight'][:, :, 0].double().numpy()
    m = W.shape[0]; Wi = np.linalg.pinv(W); cols = {c: [] for c in ['dec_pred', 'dec_true', 'dec_goal', 'e_pos', 'e_vel', 'e_cross', 'e_resid', 'e_tokvar',
                                                                     'e_full', 'margin', 'success', 'agent_dist']}
    for k in keys:
        f = d / f'start_{int(k):03d}.npz'; assert hashlib.sha256(f.read_bytes()).hexdigest() == idx['starts'][k]['npz_sha256']
        z = np.load(f); ev = z['split'] == 1; n = int(ev.sum())
        P = z['pp16'][ev].astype(np.float64).reshape(n, -1, m); Tt = z['tp16'][ev].astype(np.float64).reshape(n, -1, m); G = z['gp'].astype(np.float64).reshape(-1, m)
        dec = lambda X: (X.mean(-2) - b) @ Wi.T
        D = P - G[None]; h = D.mean(1); dr = h @ Wi.T; u = dr[:, :2] @ W[:, :2].T; w = dr[:, 2:] @ W[:, 2:].T; rho = h - dr @ W.T
        st = z['official_state'][ev]
        for c, v in [('dec_pred', dec(P)), ('dec_true', dec(Tt)), ('dec_goal', dec(G)), ('e_pos', (u ** 2).sum(1) / m), ('e_vel', (w ** 2).sum(1) / m),
                     ('e_cross', 2 * (u * w).sum(1) / m), ('e_resid', (rho ** 2).sum(1) / m), ('e_tokvar', ((D - h[:, None]) ** 2).sum(2).mean(1) / m),
                     ('e_full', (D ** 2).sum(2).mean(1) / m), ('margin', z['margin'][ev]), ('success', z['success'][ev]),
                     ('agent_dist', np.linalg.norm(st[:, :2] - z['goal_state'][:2], axis=1))]:
            cols[c].append(np.asarray(v))
    np.savez_compressed(outp, **{c: np.stack(v) for c, v in cols.items()}, W=W, b=b, checkpoint=np.array(Path(ckpt).name))
    print(outp, {c: np.stack(v).shape for c, v in cols.items()})


if __name__ == '__main__':
    main(*sys.argv[1:4])
