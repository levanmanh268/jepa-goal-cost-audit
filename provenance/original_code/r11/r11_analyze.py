"""R11 locked analysis (Claude, 2026-10-28). Written and hashed BEFORE r11_inloop.json exists. Development grade.

Design: 2 x 2 factorial inside CEM on the 16 R8/R9 starts (same identities, seeds, budget), cells
  C00 = visual + 0.1 full proprio (official cost)   -> R9 'b0' (stored R8 proposal mean, executed)
  C10 = visual + 0.1 position-only proprio            -> R11 'veldrop_01'   (NEW)
  C01 = z(visual) + z(full proprio) within population -> R11 'hprop_std'    (NEW)
  C11 = z(visual) + z(position-only) within population-> R9 'veldrop_z'
Provenance: R11 config code hash = locked r11_scale_ablation.py hash; the R11 raw-arm rerun on start 0 reproduces
the R8 proposal mean exactly (max abs 0.0); every R11 run identity equals the R9 identity of the same start; R9 inputs
are the locked R9 results (sha256 checked).
Cohort: prefix of starts 0..n-1 with all four cells present. INCOMPLETE if n < 8 or any provenance failure.
PRIMARY (priority decision, operational thresholds, not an alpha-controlled test):
  D1 = margin(C11) - margin(C01).  DROP_HELPS iff mean D1 <= -0.10 AND successes(C11) >= successes(C01) AND mean D1
  without the single most favourable start <= 0; otherwise DROP_NOT_SHOWN.
SECONDARY (descriptive): D2 = C10 - C00; interaction (C11 - C01) - (C10 - C00); scale effects C01 - C00 and C11 - C10.
Reported for every contrast: mean, median, all per-start differences, bootstrap 95% interval (20,000 resamples over
starts, seed 4001), improved/worsened/unchanged counts, paired success gains/losses, exact two-sided McNemar and sign
test (descriptive only). Also: success, block-only success and component errors per cell; mean in-loop alpha values.
Usage: python3 -I r11_analyze.py <r9_inloop.json> <r11_inloop.json> <out.json>
"""
import hashlib, json, math, sys
import numpy as np

R11_SCRIPT_SHA = 'ed608b0a42d88e4a8e30782bb15717259035681ddf8763c20bfbc5677ce9dcf0'
R9_JSON_SHA = None  # filled from the lock file at run time (see main)
DELTA = 0.10; N_MIN = 8


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def boot(x, B=20000, seed=4001):
    x = np.asarray(x, float); r = np.random.default_rng(seed); m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def binom2(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def contrast(Ma, Mb, Sa, Sb):
    D = Ma - Mb
    return {'per_start': [float(d) for d in D], 'mean_ci': boot(D), 'median': float(np.median(D)),
            'improved': int((D < 0).sum()), 'worsened': int((D > 0).sum()), 'unchanged': int((D == 0).sum()),
            'without_most_favourable': float(np.sort(D)[1:].mean()) if len(D) > 1 else None,
            'success_gains_losses': [int((Sa & ~Sb).sum()), int((Sb & ~Sa).sum())],
            'mcnemar_exact_two_sided': binom2(int((Sa & ~Sb).sum()), int((Sb & ~Sa).sum())),
            'sign_test_two_sided': binom2(int((D < 0).sum()), int((D > 0).sum()))}


def main(r9p, r11p, outp):
    lock = open('/home/claude/wm/r11/R11_LOCK.md').read()
    r9_sha_locked = [l.split()[0] for l in lock.splitlines() if l.strip().endswith(' r9_inloop.json')][0]
    R9 = json.load(open(r9p)); R11 = json.load(open(r11p)); fails = []
    if sha(r9p) != r9_sha_locked: fails.append('r9_inloop.json differs from the locked R9 results')
    if R11['config']['code_sha256'].get('r11_scale_ablation.py') != R11_SCRIPT_SHA: fails.append('r11_scale_ablation.py hash differs from the lock')
    if R11['config'].get('smoke'): fails.append('smoke run')
    rep = [v for v in R11['runs'].values() if v['arm'] == 'raw']
    if not rep: fails.append('no raw-arm reproduction run')
    elif any(v.get('repro_maxabs_vs_r8_proposal_mean', 1.0) != 0.0 for v in rep): fails.append('raw-arm rerun does not reproduce the R8 proposal mean')
    a9, a11 = R9['runs'], R11['runs']
    cell = {'C00': (a9, 'b0'), 'C11': (a9, 'veldrop_z'), 'C01': (a11, 'hprop_std'), 'C10': (a11, 'veldrop_01')}
    n = 0
    while all(f'{n}:{arm}' in src for src, arm in cell.values()): n += 1
    for k in range(n):
        for src, arm in [(a11, 'hprop_std'), (a11, 'veldrop_01')]:
            if src[f'{k}:{arm}']['identity'] != a9[f'{k}:veldrop_z']['identity']: fails.append(f'identity mismatch {k}:{arm}')
    ok = not fails
    M = {c: np.array([src[f'{k}:{arm}']['margin'] for k in range(n)]) for c, (src, arm) in cell.items()}
    S = {c: np.array([bool(src[f'{k}:{arm}']['success']) for k in range(n)]) for c, (src, arm) in cell.items()}
    BO = {c: int(sum(bool(src[f'{k}:{arm}']['block_only_success']) for k in range(n))) for c, (src, arm) in cell.items()}
    CE = {c: {e: float(np.mean([src[f'{k}:{arm}'][e] for k in range(n)])) if n else None for e in ['agent_err', 'block_err', 'angle_err_deg']}
          for c, (src, arm) in cell.items()}
    P = contrast(M['C11'], M['C01'], S['C11'], S['C01'])
    dec = 'INCOMPLETE' if (n < N_MIN or not ok) else ('DROP_HELPS' if (P['mean_ci'][0] <= -DELTA and S['C11'].sum() >= S['C01'].sum()
                                                                       and P['without_most_favourable'] <= 0) else 'DROP_NOT_SHOWN')
    inter = (M['C11'] - M['C01']) - (M['C10'] - M['C00'])
    alphas = {arm: {key: float(np.mean([np.mean([h[key] for h in a11[f'{k}:{arm}']['history']]) for k in range(n)])) if n else None
                    for key in ['alpha_z_full', 'alpha_z_pos']} for arm in ['hprop_std', 'veldrop_01']}
    summ = {'n': n, 'provenance_failures': fails,
            'PRIMARY_C11_minus_C01': {'decision': dec, **P},
            'SECONDARY_C10_minus_C00': contrast(M['C10'], M['C00'], S['C10'], S['C00']),
            'scale_effect_full_C01_minus_C00': contrast(M['C01'], M['C00'], S['C01'], S['C00']),
            'scale_effect_pos_C11_minus_C10': contrast(M['C11'], M['C10'], S['C11'], S['C10']),
            'interaction': {'per_start': [float(x) for x in inter], 'mean_ci': boot(inter)},
            'cells': {c: {'mean_margin_ci': boot(M[c]) if n else None, 'successes': int(S[c].sum()), 'block_only_successes': BO[c],
                          'component_errors': CE[c]} for c in cell},
            'mean_inloop_alpha': alphas,
            'r11_raw_repro_maxabs': [v.get('repro_maxabs_vs_r8_proposal_mean') for v in rep]}
    json.dump({'summary': summ}, open(outp, 'w'), indent=1)
    print(json.dumps({k: summ[k] for k in ['n', 'provenance_failures', 'cells']}, indent=1))
    print('PRIMARY', dec, P['mean_ci'], P['success_gains_losses'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
