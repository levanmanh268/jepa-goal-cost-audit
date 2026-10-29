"""DEVELOPMENT check of a mechanism found in dev_decompose.py (26 R6/R7 starts, all development). Claude, R7, 2026-10-27.

Finding to explain: per bank, Spearman(true encoded proprio term, true final agent distance) is only 0.39 although the
predicted proprio matches the true encoded proprio (Spearman 0.996). In jepa-wms PushT the proprio is
[agent_x, agent_y, agent_vx, agent_vy] (with_velocity=True), normalized by fixed stats and embedded by a linear map
W (16x4) + b, tiled over 256 tokens. The official terminal cost therefore asks the final agent VELOCITY to match the goal
frame's velocity, which the pose-based success criterion ignores.

Label-free fix (uses only the model's own proprio encoder and the documented proprio layout, no outcome labels):
decode the embedded proprio by least squares r = pinv(W)(emb - b) and drop the velocity block.
  prop_pos = mean_k (W[:, :2] (r_pos - r_pos_goal))_k^2   (the official proprio term with the velocity difference set to 0)
Pre-declared variants: VELDROP (vis + 0.1 prop_pos, i.e. the official weights, velocity removed), VELDROP-z (within-bank
z-score sum of vis and prop_pos), POS-only, VEL-only (diagnostic). All printed. Nothing here touches R8.
Usage: python3 -I dev_velocity.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/home/claude/wm/r7')
import energies as E, tail_energies as T
from dev_screen2 import pick, comp_err, zsum

ROOT = Path('/home/claude/wm/r7/banks')
MEAN = np.array([236.61549377441406, 264.5674133300781, -2.9303202629089355, 2.543079137802124])
STD = np.array([101.12020111083984, 87.01119995117188, 74.8455581665039, 74.14009094238281])


def load_W():
    ck = torch.load('/home/claude/wm/runtime/checkpoints/jepa_wm_pusht.pth.tar', map_location='cpu', weights_only=False)
    pe = ck['proprio_encoder']
    return pe['module.patch_embed.weight'][:, :, 0].double().numpy(), pe['module.patch_embed.bias'].double().numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return np.corrcoef(ra, rb)[0, 1]


def main():
    W, b = load_W(); Wp = np.linalg.pinv(W)
    print('W', W.shape, 'singular values', np.round(np.linalg.svd(W, compute_uv=False), 4))
    G_pos = W[:, :2]; G_vel = W[:, 2:]
    print('column-block norms: position', np.round(np.linalg.norm(G_pos, axis=0), 4), 'velocity', np.round(np.linalg.norm(G_vel, axis=0), 4))
    S = []; dec_err = []; resid = []
    for cohort in ['train', 'test']:
        d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
        for k, rec in sorted(idx['starts'].items(), key=lambda kv: int(kv[0])):
            z = np.load(d / f'start_{int(k):03d}.npz'); n = len(z['margin']); C = z['pv16'].shape[-1]
            P = z['pv16'].astype(np.float64).reshape(n, -1, C); G = z['gv'].astype(np.float64).reshape(-1, C)
            pp = z['pp16'].astype(np.float64); tp = z['tp16'].astype(np.float64); gp = z['gp'].astype(np.float64)
            r_pred = (pp.mean(1) - b) @ Wp.T; r_true = (tp.mean(1) - b) @ Wp.T; r_goal = (gp.mean(0) - b) @ Wp.T
            st = z['official_state']; r_state = (st[:, [0, 1, 5, 6]] - MEAN) / STD
            dec_err.append(np.abs(r_true - r_state).max()); resid.append(np.abs(pp.mean(1) - (r_pred @ W.T + b)).max())
            vis = T.patch_errors(P, G).mean(1); prop = E.proprio_term(pp.reshape(n, -1), gp.reshape(-1))
            dpos = r_pred[:, :2] - r_goal[:2]; dvel = r_pred[:, 2:] - r_goal[2:]
            prop_pos = ((dpos @ G_pos.T) ** 2).mean(1); prop_vel = ((dvel @ G_vel.T) ** 2).mean(1)
            a, bb, th = comp_err(st, z['goal_state'])
            S.append(dict(cohort=cohort, split=z['split'], margin=z['margin'], b0=z['model_cost'], vis=vis, prop=prop,
                          prop_pos=prop_pos, prop_vel=prop_vel, agent=a, comp=np.column_stack([a, bb, th]),
                          goal_speed=float(np.linalg.norm(z['goal_state'][5:7]))))
    print(f'decode check: max |pinv-decoded true proprio - simulator proprio| over all plans {max(dec_err):.2e} (normalized units); '
          f'max residual of predicted embedding outside range(W) {max(resid):.3e}')
    print(f"goal-frame agent speed (simulator units): median {np.median([s['goal_speed'] for s in S]):.1f}, "
          f"min {min(s['goal_speed'] for s in S):.1f}, max {max(s['goal_speed'] for s in S):.1f}")
    sh = [np.mean(s['prop_vel'][s['split'] == 1]) / np.mean(s['prop'][s['split'] == 1]) for s in S]
    print(f'velocity share of the official proprio term (eval banks, mean over starts): {np.mean(sh):.3f}')
    for key in ['prop', 'prop_pos', 'prop_vel']:
        r = [spearman(s[key][s['split'] == 1], s['agent'][s['split'] == 1]) for s in S]
        print(f'Spearman({key}, true final agent distance) per eval bank: mean {np.mean(r):.3f}, median {np.median(r):.3f}')
    coh = np.array([s['cohort'] for s in S])
    rules = {'official': lambda s, ev: s['b0'],
             'hprop_std (R6)': lambda s, ev: zsum(s['vis'], s['prop'], ev),
             'VELDROP = vis + 0.1 prop_pos': lambda s, ev: s['vis'] + 0.1 * s['prop_pos'],
             'VELDROP-z = z(vis)+z(prop_pos)': lambda s, ev: zsum(s['vis'], s['prop_pos'], ev),
             'POS-only (prop_pos)': lambda s, ev: s['prop_pos'],
             'VEL-only (diagnostic)': lambda s, ev: s['prop_vel'],
             'prop-only (R6)': lambda s, ev: s['prop']}
    print(f"\n{'rule':32s} {'all26':>7s} {'succ':>4s} | {'tr16':>6s} {'te10':>6s} | {'agent':>6s} {'block':>6s} {'angle':>6s}")
    out = {}
    for name, fn in rules.items():
        js = []
        for s in S:
            ev = s['split'] == 1; js.append(int(np.flatnonzero(ev)[pick(fn(s, ev)[ev])]))
        m = np.array([S[i]['margin'][j] for i, j in enumerate(js)]); c = np.array([S[i]['comp'][j] for i, j in enumerate(js)])
        out[name] = m
        print(f"{name:32s} {m.mean():7.3f} {(m < 1).sum():4d} | {m[coh == 'train'].mean():6.3f} {m[coh == 'test'].mean():6.3f} | "
              f"{c[:, 0].mean():6.1f} {c[:, 1].mean():6.1f} {c[:, 2].mean():6.1f}")
    for name in ['VELDROP = vis + 0.1 prop_pos', 'VELDROP-z = z(vis)+z(prop_pos)']:
        D = out[name] - out['official']
        print(f'{name}: D vs official mean {D.mean():+.3f}; improved {int((D < 0).sum())}, worsened {int((D > 0).sum())}; '
              f'without most favourable start {np.sort(D)[1:].mean():+.3f}')


if __name__ == '__main__':
    main()
