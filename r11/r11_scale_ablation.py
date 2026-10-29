"""R11 proposed two-arm CEM ablation; not yet run in the model/simulator environment.
Adapted from the supplied R9 collector. Original remains unchanged.
Copy this file into Claude's environment and retain its existing R7/runtime/D3 dependencies.
Separate output directory required, e.g.:
  python3 -I r11_scale_ablation.py --cohort /home/claude/wm/r8/cohort --out /home/claude/wm/r11_scale
Both arms, all locked R8 identities, CPU, same seeds. Raw start0 reproduction is required.
Important: lock the executable hash BEFORE running; this file alone is not registration.
No CUDA execution has been implemented or tested here.
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

WM = Path(str(_WM_ROOT / '')); V06 = WM / 'v06_reconstructed'; D3 = WM / 'd3'; RUNTIME = WM / 'runtime'; R6 = WM / 'r6'; R7 = WM / 'r7'
TRAIN = D3 / 'data_official/pusht_noise/train'; VAL = D3 / 'data_official/pusht_noise/val'
HEAD = R7 / 'R7_FROZEN_R6_HEAD.json'; CKPT = RUNTIME / 'checkpoints/jepa_wm_pusht.pth.tar'


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


def full_state(env):
    a, b = env.agent, env.block
    return np.array([a.position[0], a.position[1], a.velocity[0], a.velocity[1], b.position[0], b.position[1],
                     b.angle % (2 * np.pi), b.velocity[0], b.velocity[1], b.angular_velocity], np.float64)


def zs(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def components(st, gs):
    st = np.asarray(st, np.float64); gs = np.asarray(gs, np.float64)
    a = float(np.linalg.norm(st[0:2] - gs[0:2])); bl = float(np.linalg.norm(st[2:4] - gs[2:4]))
    th = abs(float(st[4]) - float(gs[4])) % (2 * np.pi); th = min(th, 2 * np.pi - th)
    return {'agent_err': a, 'block_err': bl, 'angle_err_deg': float(np.degrees(th)), 'block_only_success': bool(bl < 20 and np.degrees(th) < 20)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default=None); ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='hprop_std,veldrop_01'); ap.add_argument('--repro-start', type=int, default=0)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); arms = a.arms.split(',')
    if set(arms) != {'hprop_std', 'veldrop_01'} or len(arms) != 2:
        raise SystemExit('R11 requires exactly hprop_std,veldrop_01; do not alter arms after locking')
    sys.path.insert(0, str(V06)); sys.path.insert(0, str(D3)); sys.path.insert(0, str(R7))
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc, energies as E
    w = np.array(json.loads(HEAD.read_text())['weights'], np.float64)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)['proprio_encoder']
    Wpe = ck['module.patch_embed.weight'][:, :, 0].double().numpy(); bpe = ck['module.patch_embed.bias'].double().numpy(); Wp = np.linalg.pinv(Wpe)
    env = make_env(RUNTIME / 'jepa-wms'); model, pre, prov = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', 'jepa', 'cpu')
    T = ot.FRAMESKIP * ot.GOAL_H + 1
    if a.smoke:
        states, acts_d, lens = ot.load_val(VAL); iters = 1
        with np.load(R6 / 'r6a_cache_train.npz', allow_pickle=False) as zc:
            meta = {m['k']: m for m in json.loads(zc['meta_json'].tobytes())}
        jobs = [(0, meta[0]['identity'], None)]; index_cfg_sha = None
    else:
        states, acts_d, lens = ot.load_val(TRAIN); iters = 8
        cohort = Path(a.cohort); index = json.loads((cohort / 'r6_index.json').read_text()); index_cfg_sha = index['config_sha256']
        ks = sorted(int(k) for k in index['starts'])
        assert ks == list(range(len(ks))), ks
        jobs = [(k, index['starts'][str(k)]['identity'], index['starts'][str(k)]['npz_sha256']) for k in ks]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res_path = out / ('r11_smoke.json' if a.smoke else 'r11_inloop.json')
    cfg = {'arms': arms, 'repro_start': a.repro_start, 'samples': 64, 'iters': iters, 'elites': 10, 'horizon': 6, 'smoke': a.smoke,
           'r8_index_config_sha256': index_cfg_sha,
           'code_sha256': {'r11_scale_ablation.py': sha_file(__file__), 'energies.py': sha_file(R7 / 'energies.py'),
                           'official_tasks.py': sha_file(D3 / 'official_tasks.py'), 'wm_costs.py': sha_file(D3 / 'wm_costs.py'),
                           'jepa_adapter.py': sha_file(V06 / 'jepa_adapter.py')},
           'head_sha256': sha_file(HEAD), 'checkpoint_sha256': prov.get('checkpoint_sha256'), 'torch': torch.__version__}
    res = json.loads(res_path.read_text()) if res_path.exists() else {'config': cfg, 'provenance': prov, 'runs': {}}
    if res['config'] != cfg: raise SystemExit('config/code changed since previous run; refusing to resume')

    def save():
        tmp = Path(str(res_path) + '.tmp'); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, res_path)

    for k, ident, npz_sha in jobs:
        i, off, es, cs = ident['val_episode'], ident['offset'], ident['env_seed'], ident['cem_seed']
        init = states[i, off].copy(); expert = acts_d[i, off:off + T - 1].copy()
        obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, expert)
        gstate = np.asarray(gstate, np.float64)
        r8 = None
        if not a.smoke:
            f = cohort / f'start_{k:03d}.npz'
            if sha_file(f) != npz_sha: raise SystemExit(f'R8 npz hash mismatch at start {k}')
            with np.load(f) as zr:
                if not np.array_equal(zr['goal_state'], gstate): raise SystemExit(f'goal state mismatch at start {k}')
                r8 = {'proposal_mean': zr['proposal_mean'].copy()}
        sc = wc.ModelScorer(model, obs0, gobs)
        gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy(); C = gv.shape[-1]
        rg = (gp.reshape(-1, Wpe.shape[0]).mean(0) - bpe) @ Wp.T

        def terms(c_raw, pv, pp):
            P = pv.double().numpy(); p = pp.double().numpy(); B = P.shape[0]
            P2 = P.reshape(B, -1, C); p2 = p.reshape(B, -1)
            vis = ((P2.reshape(B, -1) - gv.reshape(-1)) ** 2).mean(1)
            r = (p.reshape(B, -1, Wpe.shape[0]).mean(1) - bpe) @ Wp.T
            pos = (((r[:, :2] - rg[:2]) @ Wpe[:, :2].T) ** 2).mean(1)
            full = ((p2 - gp.reshape(-1)) ** 2).mean(1)
            return {'hprop_std': zs(vis) + zs(full), 'veldrop_01': vis + 0.1 * pos, 'prop_full': full, 'raw': np.asarray(c_raw, np.float64), 'b1': E.head(P2, gv.reshape(-1, C), p2, gp.reshape(-1), w),
                    'veldrop_z': zs(vis) + zs(pos), 'vis': vis, 'prop_pos': pos}

        def execute(plan):
            _, st = ot.fast_rollout(env, es, init, ot.norm_to_env(plan), render_final=False); fs = full_state(env)
            r = ot.eval_state(gstate, st)
            return {'final_state': np.asarray(st).tolist(), 'final_full_state': fs.tolist(), 'success': r['success'], 'margin': r['margin'],
                    'pos_diff': r['pos_diff'], 'angle_diff': r['angle_diff'], **components(st, gstate)}

        key0 = f'{k}:b0'
        if r8 is not None and key0 not in res['runs']:
            res['runs'][key0] = {'identity': ident, 'arm': 'b0', 'source': 'R8 proposal_mean (official-cost CEM, same seed)',
                                 'final_mean': r8['proposal_mean'].tolist(), **execute(r8['proposal_mean'])}
            save()
        run_arms = arms + (['raw'] if (k == a.repro_start and not a.smoke) else [])
        for arm in run_arms:
            key = f'{k}:{arm}'
            if key in res['runs']: continue
            t0 = time.perf_counter(); hist = []
            g = torch.Generator().manual_seed(int(cs)); mean = torch.zeros(6, 10); std = torch.ones(6, 10)
            for it in range(iters):
                q = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(6, 64, 10, generator=g); q[:, 0, :] = mean
                c_raw, pv, pp = sc(q); tt = terms(c_raw, pv, pp); cost = tt[arm]
                el = torch.topk(-torch.from_numpy(np.asarray(cost, np.float64)), 10).indices; eln = el.numpy()
                qa = q.permute(1, 0, 2).reshape(64, -1).numpy()
                hist.append({'sd_vis': float(tt['vis'].std()), 'sd_prop_full': float(tt['prop_full'].std()), 'sd_prop_pos': float(tt['prop_pos'].std()), 'alpha_z_full': float((tt['vis'].std()+1e-12)/(tt['prop_full'].std()+1e-12)), 'alpha_z_pos': float((tt['vis'].std()+1e-12)/(tt['prop_pos'].std()+1e-12)), 'iter': it, 'arm_score_min': float(np.min(cost)), 'arm_score_elite_mean': float(np.mean(cost[eln])),
                             'raw_elite_mean': float(tt['raw'][eln].mean()), 'vis_elite_mean': float(tt['vis'][eln].mean()),
                             'prop_pos_elite_mean': float(tt['prop_pos'][eln].mean()), 'b1_elite_mean': float(tt['b1'][eln].mean()),
                             'frac_negative_b1': float((tt['b1'] < 0).mean()), 'action_abs_mean': float(np.abs(qa).mean()),
                             'frac_action_abs_gt3': float((np.abs(qa) > 3).mean()), 'std_mean': float(std.mean()), 'seconds': time.perf_counter() - t0})
                mean = q[:, el].mean(1); std = q[:, el].std(1)
            plan = mean.numpy().astype(np.float64)
            rec = {'identity': ident, 'arm': arm, 'final_mean': plan.tolist(), 'final_std': std.numpy().astype(np.float64).tolist(), 'history': hist,
                   **execute(plan)}
            c_e, pv_e, pp_e = sc(torch.from_numpy(plan[None]).permute(1, 0, 2).float()); te = terms(c_e, pv_e, pp_e)
            rec['executed_pred'] = {'raw': float(te['raw'][0]), 'b1': float(te['b1'][0]), 'vis': float(te['vis'][0]), 'prop_pos': float(te['prop_pos'][0])}
            if arm == 'raw' and r8 is not None:
                rec['repro_maxabs_vs_r8_proposal_mean'] = float(np.max(np.abs(plan - r8['proposal_mean'])))
            rec['seconds'] = time.perf_counter() - t0
            res['runs'][key] = rec; save()
            print('start', k, 'arm', arm, 'done', round(rec['seconds'], 1),
                  ('repro_maxabs ' + str(rec['repro_maxabs_vs_r8_proposal_mean'])) if 'repro_maxabs_vs_r8_proposal_mean' in rec else '', flush=True)
    print('done', len(res['runs']), flush=True)


if __name__ == '__main__':
    main()
