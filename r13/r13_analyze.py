"""R13 locked analysis (2026-10-29). Run once, after all 32 hprop_std runs exist. numpy only.
Usage: python3 -I r13_analyze.py r12_cohort.json r12_inloop.json r13_inloop.json r13_analysis.json
Primary: D = margin(veldrop_z, R12) - margin(hprop_std, R13) on the 32 locked R12 starts, same seeds and first population.
This is the R11 primary contrast (position versus full under population-wise standardisation) repeated on new goals.
Gate (priority rule, not a type-I-error-controlled test): mean D <= -0.10 AND successes(veldrop_z) >= successes(hprop_std)
AND mean D without the most favourable start <= 0 -> DROP_HELPS_FRESH, else NOT_SHOWN; INCOMPLETE if a pair is missing.
Descriptive: hprop_std - raw (R12 official cost). Bootstrap: default_rng(13001).integers(0, 32, (20000, 32)), percentile.
"""
import json, math, sys
import numpy as np


def official(goal, cur):
    goal = np.asarray(goal, np.float64); cur = np.asarray(cur, np.float64)
    pos = float(np.linalg.norm(goal[:4] - cur[:4])); ang = abs(float(goal[4]) - float(cur[4])); ang = min(ang, 2 * np.pi - ang)
    return max(pos / 20.0, ang / (np.pi / 9)), bool(pos < 20 and ang < np.pi / 9)


def binom2(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def contrast(M, S, new, ref, ks, IDX, label=None):
    d = np.array([M[new][k] - M[ref][k] for k in ks]); sn = np.array([S[new][k] for k in ks]); sr = np.array([S[ref][k] for k in ks])
    lo, hi = np.percentile(d[IDX].mean(1), [2.5, 97.5]); imp, wor = int((d < -1e-12).sum()), int((d > 1e-12).sum())
    g, l = int((sn & ~sr).sum()), int((sr & ~sn).sum()); lbo = float(np.delete(d, int(np.argmin(d))).mean())
    out = {'new': new, 'reference': ref, 'n': len(ks), 'mean_ci': [float(d.mean()), float(lo), float(hi)], 'median': float(np.median(d)),
           'D_by_start': {str(k): float(x) for k, x in zip(ks, d)}, 'improved': imp, 'worsened': wor, 'success_gains_losses': [g, l],
           'successes_new_ref': [int(sn.sum()), int(sr.sum())], 'mcnemar_exact_two_sided': binom2(g, l), 'sign_test_two_sided': binom2(imp, wor),
           'without_most_favourable': lbo}
    if label: out['decision'] = label if (d.mean() <= -0.10 and sn.sum() >= sr.sum() and lbo <= 0) else 'NOT_SHOWN'
    return out


def main(coh_p, r12_p, r13_p, out_p):
    coh = json.load(open(coh_p)); R12 = json.load(open(r12_p))['runs']; R13 = json.load(open(r13_p)); runs13 = R13['runs']; issues = []
    if R13['config']['groups'] != [['hprop_std']] or R13['config']['iters'] != 8 or R13['config']['mode'] != 'cohort': issues.append('unexpected R13 config')
    starts = {s['k']: {kk: s[kk] for kk in ['k', 'val_episode', 'offset', 'env_seed', 'cem_seed']} for s in coh['starts']}
    M = {a: {} for a in ['hprop_std', 'veldrop_z', 'raw']}; S = {a: {} for a in M}; dev = 0.0; q0 = 0
    for k, ident in starts.items():
        for a, src in [('hprop_std', runs13), ('veldrop_z', R12), ('raw', R12)]:
            r = src.get(f'{k}:{a}')
            if r is None: continue
            if r['identity'] != ident: issues.append(f'identity mismatch {k}:{a}'); continue
            m, s = official(r['goal_state'], r['final_state']); dev = max(dev, abs(m - r['margin']))
            if s != r['success']: issues.append(f'success mismatch {k}:{a}')
            M[a][k], S[a][k] = m, s
        if f'{k}:hprop_std' in runs13 and f'{k}:veldrop_z' in R12:
            if runs13[f'{k}:hprop_std']['q0_sha256'] != R12[f'{k}:veldrop_z']['q0_sha256']: issues.append(f'first population differs at {k}')
            else: q0 += 1
    if dev > 1e-9: issues.append(f'stored margins deviate ({dev:.2e})')
    N = len(starts); IDX = np.random.default_rng(13001).integers(0, N, (20000, N)); ks = sorted(k for k in starts if k in M['hprop_std'] and k in M['veldrop_z'])
    summary = {'n_cohort': N, 'issues': issues, 'max_margin_dev': dev, 'q0_matches': q0, 'hprop_std_runs': len(M['hprop_std'])}
    summary['PRIMARY'] = contrast(M, S, 'veldrop_z', 'hprop_std', ks, IDX, 'DROP_HELPS_FRESH') if len(ks) == N else {'decision': 'INCOMPLETE', 'pairs': len(ks)}
    if len(ks) == N: summary['hprop_std_minus_raw'] = contrast(M, S, 'hprop_std', 'raw', ks, IDX)
    summary['cell_hprop_std'] = {'mean_margin': float(np.mean(list(M['hprop_std'].values()))) if M['hprop_std'] else None, 'successes': int(sum(S['hprop_std'].values())),
                                 'mean_agent_err': float(np.mean([runs13[f'{k}:hprop_std']['agent_err'] for k in M['hprop_std']])) if M['hprop_std'] else None,
                                 'mean_block_err': float(np.mean([runs13[f'{k}:hprop_std']['block_err'] for k in M['hprop_std']])) if M['hprop_std'] else None,
                                 'block_only_successes': int(sum(runs13[f'{k}:hprop_std']['block_only_success'] for k in M['hprop_std']))}
    json.dump({'summary': summary}, open(out_p, 'w'), indent=1)
    print(json.dumps({'decision': summary['PRIMARY'].get('decision'), 'mean_ci': summary['PRIMARY'].get('mean_ci')}), 'issues', issues)


if __name__ == '__main__':
    main(*sys.argv[1:5])
