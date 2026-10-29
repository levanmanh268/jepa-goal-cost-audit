"""R9 locked analysis (Claude, 2026-10-27). Implements only the rules of R8_ADDENDUM_LOCK.md section 5.
Written and hashed before r9_inloop.json exists. Development grade.

Input: r9_inloop.json from r9_inloop.py (runs keyed 'k:arm', arms b0 / veldrop_z / b1 / raw).
R9 cohort = the prefix of R8 starts 0..n-1 for which b0, veldrop_z and b1 all completed (start-major order).
  R9 PRIMARY  : D = executed margin(veldrop_z) - executed margin(b0). SUPPORTED iff mean D <= -0.10 AND
                successes(veldrop_z) >= successes(b0) AND mean D without the most favourable start <= 0;
                NOT_SUPPORTED otherwise; INCOMPLETE if n < 8 or a provenance failure.
  R9 SECONDARY: the same rule for b1 vs b0 (SIGNAL / NO_SIGNAL / INCOMPLETE).
  Provenance  : the raw-arm rerun on the repro start must reproduce the R8 proposal mean exactly (max abs 0.0); config
                code hashes must match the hashes recorded in lock_hashes_R8.txt for r9_inloop.py.
Descriptive : component errors, block-only criterion, paired success gains/losses, exploitation diagnostics for b1
              (fraction of negative b1 scores per iteration, elite raw cost at the last iteration), bootstrap CIs
              (20,000 resamples over starts, seed 39).
Usage: python3 -I r9_analyze.py r9/r9_inloop.json r9_analysis.json
"""
import json, sys
import numpy as np

R9_INLOOP_SHA = 'a9a5fd323dd6dc53dcbf7f377568d1b78957282e0b028d6c6f09dc65c24b54ce'
DELTA = 0.10; N_MIN = 8


def boot(x, B=20000, seed=39):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def decide(D, s_new, s_ref, n, ok, yes, no):
    if n < N_MIN or not ok: return 'INCOMPLETE'
    return yes if (D.mean() <= -DELTA and s_new >= s_ref and np.sort(D)[1:].mean() <= 0) else no


def main(inp, outp):
    R = json.load(open(inp)); runs = R['runs']; cfg = R['config']; fails = []
    if cfg['code_sha256'].get('r9_inloop.py') != R9_INLOOP_SHA: fails.append('r9_inloop.py hash differs from the lock')
    if cfg.get('smoke'): fails.append('smoke run')
    rep = [v for v in runs.values() if v['arm'] == 'raw']
    if not rep: fails.append('no raw-arm reproduction run')
    elif any(v.get('repro_maxabs_vs_r8_proposal_mean', 1.0) != 0.0 for v in rep): fails.append('raw-arm rerun does not reproduce the R8 proposal mean')
    n = 0
    while all(f'{n}:{a}' in runs for a in ['b0', 'veldrop_z', 'b1']): n += 1
    arms = ['b0', 'veldrop_z', 'b1']
    M = {a: np.array([runs[f'{k}:{a}']['margin'] for k in range(n)]) for a in arms}
    S = {a: np.array([bool(runs[f'{k}:{a}']['success']) for k in range(n)]) for a in arms}
    ok = len(fails) == 0
    Dv = M['veldrop_z'] - M['b0']; Db = M['b1'] - M['b0']

    def block(D, a):
        return {'D': boot(D) if n else None, 'improved': int((D < 0).sum()), 'worsened': int((D > 0).sum()),
                'D_without_most_favourable': float(np.sort(D)[1:].mean()) if n > 1 else None,
                'successes': [int(S[a].sum()), int(S['b0'].sum())],
                'paired_success_gains_losses': [int((S[a] & ~S['b0']).sum()), int((S['b0'] & ~S[a]).sum())]}

    summ = {'n': n, 'provenance_failures': fails,
            'R9_PRIMARY_veldrop_z_vs_b0': {'decision': decide(Dv, S['veldrop_z'].sum(), S['b0'].sum(), n, ok, 'SUPPORTED', 'NOT_SUPPORTED'), **block(Dv, 'veldrop_z')},
            'R9_SECONDARY_b1_vs_b0': {'decision': decide(Db, S['b1'].sum(), S['b0'].sum(), n, ok, 'SIGNAL', 'NO_SIGNAL'), **block(Db, 'b1')},
            'mean_executed_margin': {a: boot(M[a]) if n else None for a in arms},
            'success_counts': {a: int(S[a].sum()) for a in arms},
            'block_only_success_counts': {a: int(sum(runs[f'{k}:{a}']['block_only_success'] for k in range(n))) for a in arms},
            'mean_component_errors': {a: {c: float(np.mean([runs[f'{k}:{a}'][c] for k in range(n)])) if n else None
                                          for c in ['agent_err', 'block_err', 'angle_err_deg']} for a in arms},
            'veldrop_z_minus_b1': boot(M['veldrop_z'] - M['b1']) if n else None,
            'b1_exploitation': {'frac_negative_b1_last_iter_mean': float(np.mean([runs[f'{k}:b1']['history'][-1]['frac_negative_b1'] for k in range(n)])) if n else None,
                                'elite_raw_cost_last_iter': {a: float(np.mean([runs[f'{k}:{a}']['history'][-1]['raw_elite_mean'] for k in range(n)])) if n else None
                                                             for a in ['veldrop_z', 'b1']},
                                'executed_pred_raw_cost': {a: float(np.mean([runs[f'{k}:{a}']['executed_pred']['raw'] for k in range(n)])) if n else None
                                                           for a in ['veldrop_z', 'b1']},
                                'action_abs_mean_last_iter': {a: float(np.mean([runs[f'{k}:{a}']['history'][-1]['action_abs_mean'] for k in range(n)])) if n else None
                                                              for a in ['veldrop_z', 'b1']}}}
    per = [{'k': k, **{f'{a}_margin': float(M[a][k]) for a in arms}, **{f'{a}_success': bool(S[a][k]) for a in arms}} for k in range(n)]
    json.dump({'summary': summ, 'per_start': per}, open(outp, 'w'), indent=1)
    print(json.dumps({k: summ[k] for k in ['n', 'provenance_failures', 'R9_PRIMARY_veldrop_z_vs_b0', 'R9_SECONDARY_b1_vs_b0', 'success_counts']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
