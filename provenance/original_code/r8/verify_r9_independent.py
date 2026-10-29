"""Independent recomputation of the R9 decisions (Claude, 2026-10-27).

Written while R9 was still collecting. It does NOT import r9_analyze.py or any project module (numpy and json only).

Re-derived from raw fields, against R8_ADDENDUM_LOCK.md section 5:
  - every run's margin and success from its stored final_state and the R8 cohort goal_state, using the official PushT
    criterion re-implemented here (pos = ||goal[:4] - cur[:4]||, wrapped angle; margin = max(pos/20, ang/(pi/9)));
  - run identities equal the R8 index identities, and each R8 NPZ matches its recorded sha256;
  - the B0 plan equals the stored R8 proposal_mean exactly;
  - the raw-arm rerun reproduces that proposal mean exactly (max abs diff 0);
  - r9_inloop.py on disk and in the config match the locked hash;
  - the R9 cohort prefix, D, leave-best-out mean, success counts and both decisions;
  - agreement with r9_analysis.json.

Usage:
  python3 -I verify_r9_independent.py r9/r9_inloop.json cohort r9_analysis.json
  python3 -I verify_r9_independent.py r9/r9_inloop.json cohort --provenance-only   (prints no outcomes)
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path('/home/claude/wm/r8')
LOCK = HERE / 'R8_ADDENDUM_LOCK_with_hashes_v2.md'
ARMS = ['b0', 'veldrop_z', 'b1']


def official(goal, cur):
    goal = np.asarray(goal, float); cur = np.asarray(cur, float)
    pos = float(np.linalg.norm(goal[:4] - cur[:4]))
    ang = abs(float(goal[4]) - float(cur[4])); ang = min(ang, 2 * np.pi - ang)
    return max(pos / 20.0, ang / (np.pi / 9)), bool(pos < 20 and ang < np.pi / 9)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance(R, cohort):
    runs, cfg = R['runs'], R['config']; issues = []
    lock_hash = [l.split()[0] for l in LOCK.read_text().splitlines() if l.strip().endswith(' r9_inloop.py')][0]
    if cfg['code_sha256'].get('r9_inloop.py') != lock_hash: issues.append('config r9_inloop.py hash differs from lock')
    if sha(HERE / 'r9_inloop.py') != lock_hash: issues.append('r9_inloop.py on disk differs from lock')
    if cfg.get('smoke'): issues.append('smoke run')
    idx = json.loads((Path(cohort) / 'r6_index.json').read_text())
    checked_npz, max_dev, repro = set(), 0.0, None
    for key, run in runs.items():
        k = int(key.split(':')[0]); f = Path(cohort) / f'start_{k:03d}.npz'; ent = idx['starts'][str(k)]
        if k not in checked_npz:
            if sha(f) != ent['npz_sha256']: issues.append(f'npz sha256 start {k}')
            checked_npz.add(k)
        for fld, v in ent['identity'].items():
            if fld in run['identity'] and run['identity'][fld] != v: issues.append(f'identity {key} {fld}')
        z = np.load(f); m, s = official(z['goal_state'], run['final_state'])
        max_dev = max(max_dev, abs(m - run['margin']))
        if s != bool(run['success']): issues.append(f'success flag {key}')
        if run['arm'] == 'b0' and float(np.max(np.abs(np.array(run['final_mean']) - z['proposal_mean']))) != 0.0:
            issues.append(f'b0 plan differs from R8 proposal_mean at {key}')
        if run['arm'] == 'raw':
            repro = float(np.max(np.abs(np.array(run['final_mean']) - z['proposal_mean'])))
    if repro is None: issues.append('no raw-arm reproduction run')
    elif repro != 0.0: issues.append(f'raw-arm rerun differs from R8 proposal_mean (max abs {repro})')
    if max_dev > 1e-9: issues.append(f'stored margin differs from recomputed official margin (max {max_dev:.3e})')
    return issues, {'runs': len(runs), 'npz_checked': len(checked_npz), 'max_margin_deviation': max_dev, 'repro_maxabs': repro}


def decide(D, s_new, s_ref, n, ok, yes, no):
    if n < 8 or not ok: return 'INCOMPLETE'
    return yes if (D.mean() <= -0.10 and s_new >= s_ref and np.sort(D)[1:].mean() <= 0) else no


def main():
    runs_path, cohort = sys.argv[1], sys.argv[2]; rest = sys.argv[3:]
    R = json.load(open(runs_path)); issues, info = provenance(R, cohort)
    print('provenance:', 'OK' if not issues else issues, info)
    if '--provenance-only' in rest: return
    runs = R['runs']; n = 0
    while all(f'{n}:{a}' in runs for a in ARMS): n += 1
    z = {}
    for k in range(n):
        g = np.load(Path(cohort) / f'start_{k:03d}.npz')['goal_state']
        for a in ARMS: z[(k, a)] = official(g, runs[f'{k}:{a}']['final_state'])   # recomputed, not the stored margin
    M = {a: np.array([z[(k, a)][0] for k in range(n)]) for a in ARMS}
    S = {a: np.array([z[(k, a)][1] for k in range(n)]) for a in ARMS}
    ok = not issues; out = {'n': n}
    for a, yes, no in [('veldrop_z', 'SUPPORTED', 'NOT_SUPPORTED'), ('b1', 'SIGNAL', 'NO_SIGNAL')]:
        D = M[a] - M['b0']
        out[a] = {'decision': decide(D, S[a].sum(), S['b0'].sum(), n, ok, yes, no), 'mean_D': float(D.mean()),
                  'D_without_most_favourable': float(np.sort(D)[1:].mean()) if n > 1 else None,
                  'improved': int((D < 0).sum()), 'worsened': int((D > 0).sum()), 'successes': [int(S[a].sum()), int(S['b0'].sum())],
                  'gains_losses': [int((S[a] & ~S['b0']).sum()), int((S['b0'] & ~S[a]).sum())]}
    out['mean_margin'] = {a: float(M[a].mean()) for a in ARMS}
    print(json.dumps(out, indent=1))
    print('per start (k, b0, veldrop_z, b1 margins; successes):')
    for k in range(n):
        print(f'  {k:2d}  ' + '  '.join(f'{M[a][k]:7.3f}{"*" if S[a][k] else " "}' for a in ARMS))
    if rest and not rest[0].startswith('--'):
        T = json.load(open(rest[0]))['summary']; P = T['R9_PRIMARY_veldrop_z_vs_b0']; Q = T['R9_SECONDARY_b1_vs_b0']
        chk = {'n': T['n'] == n,
               'primary_decision': P['decision'] == out['veldrop_z']['decision'],
               'secondary_decision': Q['decision'] == out['b1']['decision'],
               'mean_D_primary': abs(P['D'][0] - out['veldrop_z']['mean_D']) < 1e-9,
               'mean_D_secondary': abs(Q['D'][0] - out['b1']['mean_D']) < 1e-9,
               'lbo_primary': abs(P['D_without_most_favourable'] - out['veldrop_z']['D_without_most_favourable']) < 1e-9,
               'success_counts': all(T['success_counts'][a] == int(S[a].sum()) for a in ARMS),
               'mean_margins': all(abs(T['mean_executed_margin'][a][0] - out['mean_margin'][a]) < 1e-9 for a in ARMS),
               'provenance_failures_agree': (len(T['provenance_failures']) == 0) == ok}
        print('agreement with locked analysis:', chk, 'ALL' if all(chk.values()) else 'MISMATCH')


if __name__ == '__main__':
    main()
