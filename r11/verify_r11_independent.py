"""Independent recomputation of the R11 decision (Claude, 2026-10-28; written before r11_inloop.json is complete).
numpy only; does not import r11_analyze.py. Recomputes every margin of the four cells from the stored final_state and the
R8 goal_state (official PushT criterion of the release), checks the raw-arm reproduction and identities, recomputes the
primary contrast C11 - C01 and its operational decision, and compares with r11_analysis.json.
Usage: python3 -I verify_r11_independent.py r9_inloop.json r11_inloop.json r8_components.npz r11_analysis.json
"""
import json, sys
import numpy as np


def official(goal, cur):
    goal = np.asarray(goal, float); cur = np.asarray(cur, float)
    pos = float(np.linalg.norm(goal[:4] - cur[:4])); ang = abs(float(goal[4]) - float(cur[4])); ang = min(ang, 2 * np.pi - ang)
    return max(pos / 20.0, ang / (np.pi / 9)), bool(pos < 20 and ang < np.pi / 9)


def main(r9p, r11p, compp, refp):
    R9 = json.load(open(r9p))['runs']; R11 = json.load(open(r11p))['runs']; Z = np.load(compp)
    cells = {'C00': (R9, 'b0'), 'C11': (R9, 'veldrop_z'), 'C01': (R11, 'hprop_std'), 'C10': (R11, 'veldrop_01')}
    n = 0
    while all(f'{n}:{a}' in src for src, a in cells.values()): n += 1
    issues = []; M = {c: np.zeros(n) for c in cells}; S = {c: np.zeros(n, bool) for c in cells}; dev = 0.0
    for k in range(n):
        for c, (src, a) in cells.items():
            run = src[f'{k}:{a}']; m, s = official(Z['goal_state'][k], run['final_state'])
            dev = max(dev, abs(m - run['margin'])); M[c][k], S[c][k] = m, s
            if s != bool(run['success']): issues.append(f'success flag {k}:{a}')
            if run['identity'] != R9[f'{k}:veldrop_z']['identity']: issues.append(f'identity {k}:{a}')
    raw = [v for v in R11.values() if v['arm'] == 'raw']
    repro = float(np.max(np.abs(np.array(raw[0]['final_mean']) - Z['proposal_mean'][raw[0]['identity']['k']]))) if raw else None
    if repro != 0.0: issues.append(f'raw repro {repro}')
    if dev > 1e-9: issues.append(f'stored margins deviate from recomputed ({dev:.2e})')
    D = M['C11'] - M['C01']; g = int((S['C11'] & ~S['C01']).sum()); l = int((S['C01'] & ~S['C11']).sum())
    dec = 'INCOMPLETE' if (n < 8 or issues) else ('DROP_HELPS' if (D.mean() <= -0.10 and S['C11'].sum() >= S['C01'].sum() and np.sort(D)[1:].mean() <= 0) else 'DROP_NOT_SHOWN')
    out = {'n': n, 'issues': issues, 'repro_maxabs': repro, 'max_margin_dev': dev, 'decision': dec, 'mean_D': float(D.mean()),
           'without_most_favourable': float(np.sort(D)[1:].mean()), 'gains_losses': [g, l],
           'cells': {c: [float(M[c].mean()), int(S[c].sum())] for c in cells}}
    print(json.dumps(out, indent=1))
    if refp:
        T = json.load(open(refp))['summary']; P = T['PRIMARY_C11_minus_C01']
        chk = {'n': T['n'] == n, 'decision': P['decision'] == dec, 'mean_D': abs(P['mean_ci'][0] - out['mean_D']) < 1e-9,
               'lbo': abs(P['without_most_favourable'] - out['without_most_favourable']) < 1e-9, 'gains_losses': P['success_gains_losses'] == [g, l],
               'cells': all(abs(T['cells'][c]['mean_margin_ci'][0] - out['cells'][c][0]) < 1e-9 and T['cells'][c]['successes'] == out['cells'][c][1] for c in cells)}
        print('agreement with locked analysis:', chk, 'ALL' if all(chk.values()) else 'MISMATCH')


if __name__ == '__main__':
    main(*sys.argv[1:5])
