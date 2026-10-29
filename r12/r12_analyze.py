"""R12 locked analysis (2026-10-28). Run once, after all runs exist (or to report INCOMPLETE). numpy only.
Usage: python3 -I r12_analyze.py r12_cohort.json r12_inloop.json r12_analysis.json
Q1 (primary): D = margin(fixed_pos) - margin(fixed_full), same alpha0 frozen from the shared first population.
Q2 (pre-declared secondary): D = margin(veldrop_z) - margin(raw), a fresh-cohort repeat of the R9 in-search contrast.
Operational gate per question (priority rule, not a type-I-error-controlled test):
  mean D <= -0.10 AND successes(new) >= successes(reference) AND mean D without the most favourable start <= 0.
Labels: Q1 POSITION_HELPS / NOT_SHOWN, Q2 VELDROP_Z_HELPS / NOT_SHOWN; INCOMPLETE if any of the 32 pairs is missing.
Bootstrap: one index matrix default_rng(12001).integers(0, n, (20000, n)) shared by all contrasts; percentile 95% CI.
Margins and successes are recomputed from final_state and goal_state with the release criterion and must match storage.
"""
import json, math, sys
import numpy as np

ARMS = ['fixed_full', 'fixed_pos', 'raw', 'veldrop_z']
QUESTIONS = {'Q1': ('fixed_pos', 'fixed_full', 'POSITION_HELPS'), 'Q2': ('veldrop_z', 'raw', 'VELDROP_Z_HELPS')}
DESCRIPTIVE = [('fixed_full', 'raw'), ('fixed_pos', 'raw'), ('veldrop_z', 'fixed_pos'), ('veldrop_z', 'fixed_full')]


def official(goal, cur):
    goal = np.asarray(goal, np.float64); cur = np.asarray(cur, np.float64)
    pos = float(np.linalg.norm(goal[:4] - cur[:4])); ang = abs(float(goal[4]) - float(cur[4])); ang = min(ang, 2 * np.pi - ang)
    return max(pos / 20.0, ang / (np.pi / 9)), bool(pos < 20 and ang < np.pi / 9)


def binom2(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def contrast(M, S, new, ref, ks, IDX, E, label=None):
    d = np.array([M[new][k] - M[ref][k] for k in ks]); sn = np.array([S[new][k] for k in ks]); sr = np.array([S[ref][k] for k in ks])
    boot = d[IDX].mean(1); lo, hi = np.percentile(boot, [2.5, 97.5])
    imp, wor = int((d < -1e-12).sum()), int((d > 1e-12).sum()); g, l = int((sn & ~sr).sum()), int((sr & ~sn).sum())
    lbo = float(np.delete(d, int(np.argmin(d))).mean()) if len(d) > 1 else float('nan')
    out = {'new': new, 'reference': ref, 'n': len(ks), 'mean_ci': [float(d.mean()), float(lo), float(hi)], 'median': float(np.median(d)),
           'D_by_start': {str(k): float(x) for k, x in zip(ks, d)}, 'improved': imp, 'worsened': wor, 'unchanged': len(d) - imp - wor,
           'success_gains_losses': [g, l], 'successes_new_ref': [int(sn.sum()), int(sr.sum())], 'mcnemar_exact_two_sided': binom2(g, l),
           'sign_test_two_sided': binom2(imp, wor), 'without_most_favourable': lbo,
           'plan_changed': int(sum(E[new][k] for k in ks))}
    if label:
        ok = d.mean() <= -0.10 and sn.sum() >= sr.sum() and lbo <= 0
        out['decision'] = label if ok else 'NOT_SHOWN'
    return out


def main(coh_p, res_p, out_p):
    coh = json.load(open(coh_p)); R = json.load(open(res_p)); runs = R['runs']; cfg = R['config']
    issues = []
    if cfg['mode'] != 'cohort' or cfg['groups'] != [['fixed_full', 'fixed_pos'], ['raw', 'veldrop_z']] or cfg['iters'] != 8:
        issues.append('unexpected config')
    starts = {s['k']: {kk: s[kk] for kk in ['k', 'val_episode', 'offset', 'env_seed', 'cem_seed']} for s in coh['starts']}
    N = len(starts); M = {a: {} for a in ARMS}; S = {a: {} for a in ARMS}; F = {a: {} for a in ARMS}; dev = 0.0
    alpha = {}; per = {}
    for k, ident in starts.items():
        per[k] = {}
        for a in ARMS:
            r = runs.get(f'{k}:{a}')
            if r is None: continue
            if r['identity'] != ident: issues.append(f'identity mismatch {k}:{a}'); continue
            m, s = official(r['goal_state'], r['final_state']); dev = max(dev, abs(m - r['margin']))
            if s != r['success']: issues.append(f'success mismatch {k}:{a}')
            M[a][k], S[a][k], F[a][k] = m, s, np.asarray(r['final_mean'])
            if a.startswith('fixed') and any(h['alpha_used'] != r['alpha0'] for h in r['history']): issues.append(f'alpha not frozen {k}:{a}')
            per[k][a] = {'margin': m, 'success': s, 'agent_err': r['agent_err'], 'block_err': r['block_err'], 'angle_err_deg': r['angle_err_deg'],
                         'block_only_success': r['block_only_success'], 'alpha0': r['alpha0'], 'q0_sha256': r['q0_sha256'], 'seconds': r['seconds']}
        al = {per[k][a]['alpha0'] for a in per[k]}; qs = {per[k][a]['q0_sha256'] for a in per[k]}
        if len(al) > 1: issues.append(f'alpha0 differs across arms at start {k}')
        if len(qs) > 1: issues.append(f'first population differs across arms at start {k}')
        if al: alpha[k] = next(iter(al))
    if dev > 1e-9: issues.append(f'stored margins deviate from recomputed ({dev:.2e})')
    IDX = np.random.default_rng(12001).integers(0, N, (20000, N))
    E = {a: {} for a in ARMS}
    for a in ARMS:
        ref = 'fixed_full' if a == 'fixed_pos' else ('raw' if a == 'veldrop_z' else None)
        for k in F[a]: E[a][k] = bool(ref is not None and k in F[ref] and not np.array_equal(F[a][k], F[ref][k]))
    summary = {'n_cohort': N, 'issues': issues, 'max_margin_dev': dev, 'arm_counts': {a: len(M[a]) for a in ARMS}}
    for q, (new, ref, lab) in QUESTIONS.items():
        ks = sorted(k for k in starts if k in M[new] and k in M[ref])
        if len(ks) < N:
            summary[q] = {'decision': 'INCOMPLETE', 'pairs': len(ks), 'note': 'fewer than 32 complete pairs; available pairs reported'}
            if len(ks) >= 2: summary[q]['available'] = contrast(M, S, new, ref, ks, np.random.default_rng(12001).integers(0, len(ks), (20000, len(ks))), E)
        else:
            summary[q] = contrast(M, S, new, ref, ks, IDX, E, lab)
    summary['descriptive'] = {}
    for new, ref in DESCRIPTIVE:
        ks = sorted(k for k in starts if k in M[new] and k in M[ref])
        if len(ks) == N: summary['descriptive'][f'{new}_minus_{ref}'] = contrast(M, S, new, ref, ks, IDX, {**E, new: {k: False for k in ks}})
    summary['cells'] = {a: {'n': len(M[a]), 'mean_margin': float(np.mean(list(M[a].values()))) if M[a] else None,
                            'successes': int(sum(S[a].values())),
                            'mean_agent_err': float(np.mean([per[k][a]['agent_err'] for k in per if a in per[k]])) if M[a] else None,
                            'mean_block_err': float(np.mean([per[k][a]['block_err'] for k in per if a in per[k]])) if M[a] else None,
                            'block_only_successes': int(sum(per[k][a]['block_only_success'] for k in per if a in per[k]))} for a in ARMS}
    av = np.array(list(alpha.values())) if alpha else np.array([np.nan])
    summary['alpha0'] = {'mean': float(av.mean()), 'median': float(np.median(av)), 'min': float(av.min()), 'max': float(av.max()), 'n': len(alpha)}
    json.dump({'summary': summary, 'per_start': {str(k): v for k, v in per.items()}}, open(out_p, 'w'), indent=1)
    print(json.dumps({q: (summary[q].get('decision'), summary[q].get('mean_ci')) for q in QUESTIONS}), 'issues', issues)


if __name__ == '__main__':
    main(*sys.argv[1:4])
