"""Independent saved-result audit. No checkpoint inference or simulator execution.

Usage: python audit_r12_r13.py PACKET OUTPUT_JSON
Reads terminal states directly; does not import the supplied analysis scripts.
"""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(x, y):
    if not np.allclose(x, y, atol=1e-11, rtol=1e-11):
        raise AssertionError((x, y))


def endpoint(rec):
    goal = np.asarray(rec['goal_state'], dtype=float)
    state = np.asarray(rec['final_state'], dtype=float)
    assert goal.shape == state.shape == (7,)
    assert np.isfinite(goal).all() and np.isfinite(state).all()
    assert 0 <= goal[4] < 2 * math.pi and 0 <= state[4] < 2 * math.pi
    d = state - goal
    agent = math.hypot(*d[:2])
    block = math.hypot(*d[2:4])
    position = math.hypot(agent, block)
    angle = abs(math.atan2(math.sin(d[4]), math.cos(d[4])))
    margin = max(position / 20, angle / (math.pi / 9))
    success = position < 20 and angle < math.pi / 9
    block_only = block < 20 and angle < math.pi / 9
    close(margin, rec['margin'])
    close(position, rec['pos_diff'])
    close(agent, rec['agent_err'])
    close(block, rec['block_err'])
    close(math.degrees(angle), rec['angle_err_deg'])
    assert success == rec['success'] and block_only == rec['block_only_success']
    full = np.asarray(rec['final_full_state'])
    # The release observation is float32; the full simulator snapshot is float64.
    close(state, full[[0, 1, 4, 5, 6, 2, 3]].astype(np.float32).astype(float))
    return dict(margin=margin, success=success, agent=agent, block=block,
                angle_deg=math.degrees(angle), block_only=block_only)


def test_p(a, b):
    return float(binomtest(a, a + b).pvalue) if a + b else 1.0


def comparison(cells, runs, new, ref, seed):
    d = np.array([a['margin'] - b['margin'] for a, b in zip(cells[new], cells[ref])])
    sn = np.array([a['success'] for a in cells[new]], bool)
    sr = np.array([a['success'] for a in cells[ref]], bool)
    n = len(d)
    idx = np.random.default_rng(seed).integers(0, n, size=(20000, n))
    lo, hi = np.quantile(d[idx].mean(axis=1), [.025, .975])
    imp, worse = int((d < -1e-12).sum()), int((d > 1e-12).sum())
    gains, losses = np.where(sn & ~sr)[0].tolist(), np.where(sr & ~sn)[0].tolist()
    drop_best = float(np.delete(d, d.argmin()).mean())
    return dict(new=new, reference=ref, n=n, mean_ci=[float(d.mean()), float(lo), float(hi)],
                median=float(np.median(d)), improved=imp, worsened=worse,
                success_gains_losses=[len(gains), len(losses)], success_gain_start_ids=gains,
                success_loss_start_ids=losses, successes_new_ref=[int(sn.sum()), int(sr.sum())],
                mcnemar_exact_two_sided=test_p(len(gains), len(losses)),
                sign_test_two_sided=test_p(imp, worse), without_most_favourable=drop_best,
                operational_gate=bool(d.mean() <= -.1 and sn.sum() >= sr.sum() and drop_best <= 0),
                plan_changed=int(sum(not np.array_equal(runs[new][k]['final_mean'], runs[ref][k]['final_mean']) for k in range(n))),
                D_by_start=d.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.packet
    read = lambda p: json.loads((root / p).read_text())
    manifest_count = 0
    for line in (root / 'MANIFEST_SHA256.txt').read_text().splitlines():
        expected, name = line.split(None, 1)
        target = (root / name.strip()).resolve()
        assert target.is_relative_to(root.resolve())
        assert digest(target) == expected, name
        manifest_count += 1
    cohort = read('r12/r12_cohort.json')
    r12 = read('r12/out/r12_inloop.json')
    r13 = read('r13/out/r13_inloop.json')
    stored12 = read('r12/r12_analysis.json')['summary']
    stored13 = read('r13/r13_analysis.json')['summary']
    assert len(cohort['starts']) == len(set(cohort['episodes'])) == cohort['n'] == 32
    assert not set(cohort['episodes']) & set(cohort['excluded_episodes'])
    draw = np.random.default_rng(cohort['seed']).choice(18685, 128, replace=False)
    kept = [int(i) for i in draw if int(i) not in cohort['excluded_episodes']][:32]
    assert kept == cohort['episodes']
    assert len(r12['runs']) == 128 and len(r13['runs']) == 32
    assert r12['provenance'] | {'load_seconds': 0} == r13['provenance'] | {'load_seconds': 0}
    for runset, script in [(r12, 'r12/r12_frozen_scale.py'), (r13, 'r13/r13_zfull.py')]:
        cfg = runset['config']
        assert cfg['samples'] == 64 and cfg['iters'] == 8 and cfg['elites'] == 10 and cfg['horizon'] == 6
        assert cfg['cohort_sha256'] == digest(root / 'r12/r12_cohort.json')
        # R13 retains the historical metadata key r12_frozen_scale.py.
        assert cfg['code_sha256']['r12_frozen_scale.py'] == digest(root / script)
    cells, runs, scales = {}, {}, {}
    for arm in ['raw', 'fixed_full', 'fixed_pos', 'veldrop_z', 'hprop_std']:
        source = r13 if arm == 'hprop_std' else r12
        runs[arm] = [source['runs'][f'{k}:{arm}'] for k in range(32)]
        cells[arm] = []
        scales[arm] = []
        for k, rec in enumerate(runs[arm]):
            assert rec['identity'] == cohort['starts'][k]
            assert rec['arm'] == arm and len(rec['history']) == 8
            assert np.shape(rec['final_mean']) == (6, 10)
            assert np.isfinite(rec['final_mean']).all()
            assert np.isfinite(rec['final_std']).all()
            assert (np.asarray(rec['final_std']) >= 0).all()
            close(rec['alpha0'], (rec['sd_vis_q0'] + 1e-12) / (rec['sd_full_q0'] + 1e-12))
            close(rec['history'][0]['sd_vis'], rec['sd_vis_q0'])
            close(rec['history'][0]['sd_prop_full'], rec['sd_full_q0'])
            for j, h in enumerate(rec['history']):
                assert h['iter'] == j
                if arm.startswith('fixed'):
                    assert h['alpha_used'] == rec['alpha0']
                else:
                    assert h['alpha_used'] is None
            cells[arm].append(endpoint(rec))
            denom = 'sd_prop_pos' if arm == 'veldrop_z' else 'sd_prop_full'
            scales[arm].append([(h['sd_vis'] + 1e-12) / (h[denom] + 1e-12) for h in rec['history']])
    for k in range(32):
        assert len({runs[a][k]['q0_sha256'] for a in runs}) == 1
        assert len({runs[a][k]['alpha0'] for a in runs}) == 1
        for arm in runs:
            assert np.array_equal(runs[arm][k]['goal_state'], runs['raw'][k]['goal_state'])
    contrasts = {}
    configs = [('Q1', 'fixed_pos', 'fixed_full', 12001, stored12['Q1']),
               ('Q2', 'veldrop_z', 'raw', 12001, stored12['Q2']),
               ('R13', 'veldrop_z', 'hprop_std', 13001, stored13['PRIMARY'])]
    for name, new, ref, seed, stored in configs:
        result = comparison(cells, runs, new, ref, seed)
        for key in ['mean_ci', 'median', 'improved', 'worsened', 'success_gains_losses',
                    'successes_new_ref', 'mcnemar_exact_two_sided', 'sign_test_two_sided', 'without_most_favourable']:
            close(result[key], stored[key])
        assert result['operational_gate']
        contrasts[name] = result
    for key, stored in stored12['descriptive'].items():
        result = comparison(cells, runs, stored['new'], stored['reference'], 12001)
        close(result['mean_ci'], stored['mean_ci'])
        contrasts[key] = result
    result = comparison(cells, runs, 'hprop_std', 'raw', 13001)
    close(result['mean_ci'], stored13['hprop_std_minus_raw']['mean_ci'])
    contrasts['hprop_std_minus_raw'] = result
    summary = {a: dict(mean_margin=float(np.mean([x['margin'] for x in c])),
                      successes=sum(x['success'] for x in c),
                      success_start_ids=[k for k, x in enumerate(c) if x['success']],
                      agent_error=float(np.mean([x['agent'] for x in c])),
                      block_error=float(np.mean([x['block'] for x in c]))) for a, c in cells.items()}
    result = dict(scope='Saved-result audit only; no checkpoint or simulator rerun.',
                  manifest_files_verified=manifest_count, terminal_records_verified=160,
                  same_goals_and_q0_hashes_across_all_five_arms=True,
                  frozen_alpha_verified_for_all_64_fixed_weight_runs=True,
                  cohort_draw_reproduced=True, summaries=summary, contrasts=contrasts,
                  adaptive_scale_medians={a: np.median(np.asarray(scales[a]), axis=0).tolist()
                                         for a in ['veldrop_z', 'hprop_std']},
                  python=platform.python_version(), numpy=np.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: contrasts[k]['mean_ci'] for k in ['Q1', 'Q2', 'R13']}))
    print(f'PASS: 160 endpoint records; five-arm pairing; frozen scales; {manifest_count} manifest files.')


if __name__ == '__main__':
    main()
