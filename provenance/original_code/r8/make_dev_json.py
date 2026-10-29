"""Recompute the development numbers used in the R8 manuscript section into one JSON (Claude, 2026-10-27).
26 R6/R7 starts (r7/banks), all development. Mirrors dev_decompose.py, dev_velocity.py, dev_velocity_terms.py,
dev_velocity_sens.py and dev_gap_after_veldrop.py (same definitions), so the manuscript reads numbers from JSON.
Usage: python3 -I make_dev_json.py dev_numbers.json
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/home/claude/wm/r7/banks')
pe = torch.load('/home/claude/wm/runtime/checkpoints/jepa_wm_pusht.pth.tar', map_location='cpu', weights_only=False)['proprio_encoder']
W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy(); Wi = np.linalg.pinv(W)
STD = np.array([101.12020111083984, 87.01119995117188])


def pick(c):
    c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1., abs(c.min())))[0])


def zs(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def spear(a, b_):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b_)); return float(np.corrcoef(ra, rb)[0, 1])


def main(outp):
    S = []
    for cohort in ['train', 'test']:
        d = ROOT / cohort; idx = json.loads((d / 'r7_index.json').read_text())
        for k in sorted(idx['starts'], key=int):
            z = np.load(d / f'start_{int(k):03d}.npz'); ev = z['split'] == 1; n = int(ev.sum()); C = z['pv16'].shape[-1]
            P = z['pv16'][ev].astype(float).reshape(n, -1, C); Tv = z['tv16'][ev].astype(float).reshape(n, -1, C)
            G = z['gv'].astype(float).reshape(-1, C); gp = z['gp'].astype(float)
            pp = z['pp16'][ev].astype(float); tp = z['tp16'][ev].astype(float)
            vis = ((P - G[None]) ** 2).mean((1, 2)); prop = ((pp.reshape(n, -1) - gp.reshape(-1)) ** 2).mean(1)
            vis_t = ((Tv - G[None]) ** 2).mean((1, 2)); prop_t = ((tp.reshape(n, -1) - gp.reshape(-1)) ** 2).mean(1)
            r = (pp.mean(1) - b) @ Wi.T; rt = (tp.mean(1) - b) @ Wi.T; rg = (gp.mean(0) - b) @ Wi.T
            dp = r[:, :2] - rg[:2]; dv = r[:, 2:] - rg[2:]
            pos = ((dp @ W[:, :2].T) ** 2).mean(1); vel = ((dv @ W[:, 2:].T) ** 2).mean(1); cross = 2 * ((dp @ W[:, :2].T) * (dv @ W[:, 2:].T)).mean(1)
            v0 = ((dp @ W[:, :2].T + r[:, 2:] @ W[:, 2:].T) ** 2).mean(1)
            st = z['official_state'][ev]; gs = z['goal_state']
            a = np.linalg.norm(st[:, :2] - gs[:2], axis=1); bl = np.linalg.norm(st[:, 2:4] - gs[2:4], axis=1)
            th = np.abs(st[:, 4] - gs[4]) % (2 * np.pi); th = np.degrees(np.minimum(th, 2 * np.pi - th))
            S.append(dict(cohort=cohort, m=z['margin'][ev], s=z['success'][ev].astype(bool), b0=z['model_cost'][ev], vis=vis, prop=prop,
                          eb0=vis_t + 0.1 * prop_t, prop_t=prop_t, pos=pos, vel=vel, cross=cross, v0=v0, a=a, bl=bl, th=th,
                          resid=float(np.abs(pp.mean(1) - (r @ W.T + b)).max()), goal_speed=float(np.linalg.norm(gs[5:7]))))
    coh = np.array([s['cohort'] for s in S])

    def rule(fn):
        js = [pick(fn(s)) for s in S]; m = np.array([s['m'][j] for s, j in zip(S, js)]); sc = np.array([s['s'][j] for s, j in zip(S, js)])
        return {'mean': float(m.mean()), 'succ': int(sc.sum()), 'train16': float(m[coh == 'train'].mean()), 'test10': float(m[coh == 'test'].mean()),
                'agent': float(np.mean([s['a'][j] for s, j in zip(S, js)])), 'block': float(np.mean([s['bl'][j] for s, j in zip(S, js)])),
                'angle': float(np.mean([s['th'][j] for s, j in zip(S, js)])), 'per_start': m.tolist()}
    R = {'official': rule(lambda s: s['b0']), 'E_official': rule(lambda s: s['eb0']), 'prop_pred': rule(lambda s: s['prop']),
         'prop_true': rule(lambda s: s['prop_t']), 'agent_oracle': rule(lambda s: s['a']), 'agent_block_oracle': rule(lambda s: np.sqrt(s['a'] ** 2 + s['bl'] ** 2)),
         'O': rule(lambda s: s['m']), 'hprop_std': rule(lambda s: zs(s['vis']) + zs(s['prop'])), 'veldrop_z': rule(lambda s: zs(s['vis']) + zs(s['pos'])),
         'veldrop_01': rule(lambda s: s['vis'] + 0.1 * s['pos']), 'pos_only': rule(lambda s: s['pos']), 'vel_only': rule(lambda s: s['vel']),
         'zero_goal_velocity_z': rule(lambda s: zs(s['vis']) + zs(s['v0']))}
    D = np.array(R['veldrop_z']['per_start']) - np.array(R['official']['per_start'])
    sens = {str(l): rule(lambda s, l=l: zs(s['vis']) + l * zs(s['pos'])) for l in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]}
    out = {'n': len(S), 'rules': R, 'veldrop_z_minus_official': {'mean': float(D.mean()), 'improved': int((D < 0).sum()), 'worsened': int((D > 0).sum()),
                                                                  'without_most_favourable': float(np.sort(D)[1:].mean())},
           'sensitivity_lambda': {k: {'mean': v['mean'], 'succ': v['succ']} for k, v in sens.items()},
           'mechanism': {'share_pos': float(np.mean([s['pos'].mean() / s['prop'].mean() for s in S])), 'share_vel': float(np.mean([s['vel'].mean() / s['prop'].mean() for s in S])),
                         'share_cross': float(np.mean([s['cross'].mean() / s['prop'].mean() for s in S])),
                         'var_ratio_pos_median': float(np.median([np.var(s['pos']) / np.var(s['prop']) for s in S])),
                         'corr_pos_prop_mean': float(np.mean([np.corrcoef(s['pos'], s['prop'])[0, 1] for s in S])),
                         'spearman_prop_agent': float(np.mean([spear(s['prop'], s['a']) for s in S])),
                         'spearman_pos_agent': float(np.mean([spear(s['pos'], s['a']) for s in S])),
                         'spearman_vel_agent': float(np.mean([spear(s['vel'], s['a']) for s in S])),
                         'spearman_pred_true_prop': float(np.mean([spear(s['prop'], s['prop_t']) for s in S])),
                         'max_residual_outside_range': float(max(s['resid'] for s in S)),
                         'goal_speed_median': float(np.median([s['goal_speed'] for s in S])),
                         'pos_form_condition_number_pixels': float((lambda e: e.max() / e.min())(np.linalg.eigvalsh(np.diag(1 / STD) @ (W[:, :2].T @ W[:, :2] / 16) @ np.diag(1 / STD))))},
           'bank_success_availability': {'banks_with_success': int(sum(s['s'].any() for s in S)),
                                         'veldrop_hits': int(sum(s['s'][pick(zs(s['vis']) + zs(s['pos']))] for s in S)),
                                         'official_hits': int(sum(s['s'][pick(s['b0'])] for s in S))}}
    json.dump(out, open(outp, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rules'}, indent=1))
    print({k: (round(v['mean'], 3), v['succ']) for k, v in R.items()})


if __name__ == '__main__':
    main(sys.argv[1])
