"""Extract compact per-plan score components from the full cohort NPZs (Claude, 2026-10-27).

The full NPZs (about 50 MB per start) are too large for the review packet. This file stores, for each start's
64-plan evaluation bank, the scalar ingredients of every selection rule, so a reviewer can recompute selections,
decisions and the mechanism check without the latents. Formulas are those of verify_r8_independent.py, with float16
latents cast to float64 as in R6.

Fields, per start k and evaluation plan j:
  model_cost[k,j]  official planning cost (as stored by the collector)
  vis[k,j]         mean squared visual error over patches and channels
  ch[k,j,:]        per-channel visual energies (384), inputs of the frozen B1 head
  prop_full[k,j]   official proprio term without the 0.1 weight
  prop_pos[k,j]    position-only proprio term
  margin, success, official_state; and per start goal_state, init_state, proposal_mean, identity.

Usage: python3 -I make_compact_components.py <cohort dir> <checkpoint> <out.npz> [--first N]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch


def main(argv):
    first = None
    if '--first' in argv: i = argv.index('--first'); first = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    d, ckpt, out = Path(argv[0]), argv[1], argv[2]
    idx = json.loads((d / 'r6_index.json').read_text()); keys = sorted(idx['starts'], key=int)
    if first: keys = keys[:first]
    pe = torch.load(ckpt, map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy()
    Wi = np.linalg.pinv(W); dp = W.shape[0]
    cols = {c: [] for c in ['model_cost', 'vis', 'ch', 'prop_full', 'prop_pos', 'margin', 'success', 'official_state',
                            'goal_state', 'init_state', 'proposal_mean']}
    ids, shas = [], []
    for k in keys:
        f = d / f'start_{int(k):03d}.npz'; h = hashlib.sha256(f.read_bytes()).hexdigest()
        assert h == idx['starts'][k]['npz_sha256'], f'sha mismatch {k}'
        z = np.load(f); ev = z['split'] == 1; n = int(ev.sum())
        P = z['pv16'][ev].reshape(n, -1, z['pv16'].shape[-1]).astype(float); G = z['gv'].astype(float).reshape(-1, P.shape[-1])
        Dl = P - G[None]; ch = (Dl * Dl).mean(1); vis = (Dl * Dl).mean((1, 2))
        p = z['pp16'][ev].reshape(n, -1).astype(float); gp = z['gp'].astype(float).reshape(-1)
        prop_full = ((p - gp[None]) ** 2).mean(1)
        r = (p.reshape(n, -1, dp).mean(1) - b) @ Wi.T; rg = (gp.reshape(-1, dp).mean(0) - b) @ Wi.T
        prop_pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
        for c, v in [('model_cost', z['model_cost'][ev]), ('vis', vis), ('ch', ch), ('prop_full', prop_full), ('prop_pos', prop_pos),
                     ('margin', z['margin'][ev]), ('success', z['success'][ev]), ('official_state', z['official_state'][ev]),
                     ('goal_state', z['goal_state']), ('init_state', z['init_state']), ('proposal_mean', z['proposal_mean'])]:
            cols[c].append(np.asarray(v))
        ids.append(idx['starts'][k]['identity']); shas.append(h)
    arrs = {c: np.stack(v) for c, v in cols.items()}
    np.savez_compressed(out, **arrs, identity_json=np.array(json.dumps(ids)), source_npz_sha256=np.array(shas),
                        checkpoint=np.array(Path(ckpt).name), proprio_W=W, proprio_b=b)
    print(out, {c: a.shape for c, a in arrs.items()})


if __name__ == '__main__':
    main(sys.argv[1:])
