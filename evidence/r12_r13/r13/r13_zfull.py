"""R13 runner (2026-10-29): r12_frozen_scale.py (kept unchanged in r12/) with one added arm, hprop_std = z(L_vis) + z(Pi_full),
run on the locked R12 cohort with the same seeds. Original R12 docstring follows.
R12 frozen-scale CEM runner (2026-10-28), derived from r11_scale_ablation.py (unchanged original kept in r11/).
Changes from R11: fresh cohort read from r12_cohort.json (no fixed banks); arms
  fixed_full : L_vis + alpha0 * Pi_full     (Q1, primary pair)
  fixed_pos  : L_vis + alpha0 * Pi_pos      (Q1, primary pair)
  raw        : the model's official cost   (Q2, secondary pair; same as the R8/R11 official-cost CEM)
  veldrop_z  : z(L_vis) + z(Pi_pos)         (Q2, secondary pair)
alpha0 = (sd(L_vis(Q0)) + 1e-12) / (sd(Pi_full(Q0)) + 1e-12), ddof = 0, computed once from the first CEM population Q0,
which is identical for every arm (same keyed generator, zero mean, unit std, candidate 0 = mean); it is frozen for all
8 iterations. The supervised head is no longer loaded. Simulator, action transform, CEM update and execution unchanged.
Modes:
  --regress : old R8 start 0, official-cost CEM; must reproduce the stored R8 proposal mean exactly (infrastructure check).
  --smoke   : old R8 start 0, fixed_full and fixed_pos with 2 iterations; checks alpha0 equality and constancy.
  default   : the locked R12 cohort; Q1 arms for every start first, then Q2 arms; resumable, atomic saves.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

WM = Path('/home/claude/wm'); V06 = WM / 'v06_reconstructed'; D3 = WM / 'd3'; RUNTIME = WM / 'runtime'
TRAIN = D3 / 'data_official/pusht_noise/train'; CKPT = RUNTIME / 'checkpoints/jepa_wm_pusht.pth.tar'
R8C = WM / 'r8/cohort'; EPS = 1e-12
Q1 = ['fixed_full', 'fixed_pos']; Q2 = ['raw', 'veldrop_z']


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
    return (a - a.mean()) / (a.std() + EPS)


def components(st, gs):
    st = np.asarray(st, np.float64); gs = np.asarray(gs, np.float64)
    a = float(np.linalg.norm(st[0:2] - gs[0:2])); bl = float(np.linalg.norm(st[2:4] - gs[2:4]))
    th = abs(float(st[4]) - float(gs[4])) % (2 * np.pi); th = min(th, 2 * np.pi - th)
    return {'agent_err': a, 'block_err': bl, 'angle_err_deg': float(np.degrees(th)), 'block_only_success': bool(bl < 20 and np.degrees(th) < 20)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', default=str(WM / 'r12/r12_cohort.json')); ap.add_argument('--out', required=True)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--regress', action='store_true'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, str(V06)); sys.path.insert(0, str(D3))
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)['proprio_encoder']
    Wpe = ck['module.patch_embed.weight'][:, :, 0].double().numpy(); bpe = ck['module.patch_embed.bias'].double().numpy(); Wp = np.linalg.pinv(Wpe)
    env = make_env(RUNTIME / 'jepa-wms'); model, pre, prov = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', 'jepa', 'cpu')
    T = ot.FRAMESKIP * ot.GOAL_H + 1
    states, acts_d, lens = ot.load_val(str(TRAIN))
    old_mean = None
    if a.regress or a.smoke:
        idx = json.loads((R8C / 'r6_index.json').read_text()); rec0 = idx['starts']['0']
        if sha_file(R8C / 'start_000.npz') != rec0['npz_sha256']: raise SystemExit('R8 start 0 hash mismatch')
        with np.load(R8C / 'start_000.npz') as zr: old_mean = zr['proposal_mean'].copy(); old_goal = zr['goal_state'].copy()
        jobs = [(0, rec0['identity'])]; iters = 8 if a.regress else 2
        groups = [['raw']] if a.regress else [Q1]; name = 'r12_regression.json' if a.regress else 'r12_smoke.json'; cohort_sha = None
    else:
        coh = json.loads(Path(a.cohort).read_text()); cohort_sha = sha_file(a.cohort)
        jobs = [(s['k'], {kk: s[kk] for kk in ['k', 'val_episode', 'offset', 'env_seed', 'cem_seed']}) for s in coh['starts']]
        assert [k for k, _ in jobs] == list(range(len(jobs)))
        iters = 8; groups = [['hprop_std']]; name = 'r13_inloop.json'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res_path = out / name
    cfg = {'mode': 'regress' if a.regress else ('smoke' if a.smoke else 'cohort'), 'groups': groups, 'samples': 64, 'iters': iters,
           'elites': 10, 'horizon': 6, 'eps': EPS, 'cohort_sha256': cohort_sha,
           'code_sha256': {'r12_frozen_scale.py': sha_file(__file__), 'official_tasks.py': sha_file(D3 / 'official_tasks.py'),
                           'wm_costs.py': sha_file(D3 / 'wm_costs.py'), 'jepa_adapter.py': sha_file(V06 / 'jepa_adapter.py')},
           'checkpoint_sha256': prov.get('checkpoint_sha256'), 'torch': torch.__version__}
    res = json.loads(res_path.read_text()) if res_path.exists() else {'config': cfg, 'provenance': prov, 'runs': {}}
    if res['config'] != cfg: raise SystemExit('config/code changed since previous run; refusing to resume')

    def save():
        tmp = Path(str(res_path) + '.tmp'); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, res_path)

    cache = {}

    def setup(k, ident):
        if k in cache: return cache[k]
        i, off, es = ident['val_episode'], ident['offset'], ident['env_seed']
        init = states[i, off].copy(); expert = acts_d[i, off:off + T - 1].copy()
        obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, expert)
        gstate = np.asarray(gstate, np.float64)
        if old_mean is not None and not np.array_equal(gstate, old_goal): raise SystemExit('goal state mismatch on R8 start 0')
        sc = wc.ModelScorer(model, obs0, gobs)
        gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy()
        rg = (gp.reshape(-1, Wpe.shape[0]).mean(0) - bpe) @ Wp.T
        cache.clear(); cache[k] = (init, es, gstate, sc, gv, gp, rg); return cache[k]

    for group in groups:
        for k, ident in jobs:
            for arm in group:
                key = f'{k}:{arm}'
                if key in res['runs']:
                    if res['runs'][key]['identity'] != ident: raise SystemExit(f'identity mismatch at {key}')
                    continue
                init, es, gstate, sc, gv, gp, rg = setup(k, ident); C = gv.shape[-1]

                def terms(c_raw, pv, pp):
                    P = pv.double().numpy(); p = pp.double().numpy(); B = P.shape[0]
                    vis = ((P.reshape(B, -1) - gv.reshape(-1)) ** 2).mean(1)
                    r = (p.reshape(B, -1, Wpe.shape[0]).mean(1) - bpe) @ Wp.T
                    pos = (((r[:, :2] - rg[:2]) @ Wpe[:, :2].T) ** 2).mean(1)
                    full = ((p.reshape(B, -1) - gp.reshape(-1)) ** 2).mean(1)
                    return {'raw': np.asarray(c_raw, np.float64), 'vis': vis, 'prop_full': full, 'prop_pos': pos}

                def arm_cost(tt, alpha0):
                    if arm == 'raw': return tt['raw']
                    if arm == 'veldrop_z': return zs(tt['vis']) + zs(tt['prop_pos'])
                    if arm == 'hprop_std': return zs(tt['vis']) + zs(tt['prop_full'])
                    if arm == 'fixed_full': return tt['vis'] + alpha0 * tt['prop_full']
                    if arm == 'fixed_pos': return tt['vis'] + alpha0 * tt['prop_pos']
                    raise SystemExit(arm)

                def execute(plan):
                    _, st = ot.fast_rollout(env, es, init, ot.norm_to_env(plan), render_final=False); fs = full_state(env)
                    r = ot.eval_state(gstate, st)
                    return {'final_state': np.asarray(st).tolist(), 'final_full_state': fs.tolist(), 'goal_state': gstate.tolist(),
                            'success': r['success'], 'margin': r['margin'], 'pos_diff': r['pos_diff'], 'angle_diff': r['angle_diff'],
                            **components(st, gstate)}

                t0 = time.perf_counter(); hist = []; alpha0 = None; q0_sha = None
                g = torch.Generator().manual_seed(int(ident['cem_seed'])); mean = torch.zeros(6, 10); std = torch.ones(6, 10)
                for it in range(iters):
                    q = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(6, 64, 10, generator=g); q[:, 0, :] = mean
                    c_raw, pv, pp = sc(q); tt = terms(c_raw, pv, pp)
                    if it == 0:
                        q0_sha = hashlib.sha256(q.numpy().astype(np.float32).tobytes()).hexdigest()
                        sd_v0, sd_f0 = float(tt['vis'].std()), float(tt['prop_full'].std()); alpha0 = (sd_v0 + EPS) / (sd_f0 + EPS)
                    cost = arm_cost(tt, alpha0)
                    el = torch.topk(-torch.from_numpy(np.asarray(cost, np.float64)), 10).indices; eln = el.numpy()
                    qa = q.permute(1, 0, 2).reshape(64, -1).numpy()
                    hist.append({'iter': it, 'alpha_used': alpha0 if arm.startswith('fixed') else None,
                                 'sd_vis': float(tt['vis'].std()), 'sd_prop_full': float(tt['prop_full'].std()), 'sd_prop_pos': float(tt['prop_pos'].std()),
                                 'arm_score_min': float(np.min(cost)), 'arm_score_elite_mean': float(np.mean(cost[eln])),
                                 'raw_elite_mean': float(tt['raw'][eln].mean()), 'vis_elite_mean': float(tt['vis'][eln].mean()),
                                 'prop_full_elite_mean': float(tt['prop_full'][eln].mean()), 'prop_pos_elite_mean': float(tt['prop_pos'][eln].mean()),
                                 'action_abs_mean': float(np.abs(qa).mean()), 'frac_action_abs_gt3': float((np.abs(qa) > 3).mean()),
                                 'std_mean': float(std.mean()), 'seconds': time.perf_counter() - t0})
                    mean = q[:, el].mean(1); std = q[:, el].std(1)
                plan = mean.numpy().astype(np.float64)
                rec = {'identity': ident, 'arm': arm, 'alpha0': alpha0, 'sd_vis_q0': sd_v0, 'sd_full_q0': sd_f0, 'q0_sha256': q0_sha,
                       'final_mean': plan.tolist(), 'final_std': std.numpy().astype(np.float64).tolist(), 'history': hist, **execute(plan)}
                c_e, pv_e, pp_e = sc(torch.from_numpy(plan[None]).permute(1, 0, 2).float()); te = terms(c_e, pv_e, pp_e)
                rec['executed_pred'] = {'raw': float(te['raw'][0]), 'vis': float(te['vis'][0]), 'prop_full': float(te['prop_full'][0]),
                                        'prop_pos': float(te['prop_pos'][0]), 'fixed_full': float(te['vis'][0] + alpha0 * te['prop_full'][0]),
                                        'fixed_pos': float(te['vis'][0] + alpha0 * te['prop_pos'][0])}
                if old_mean is not None and arm == 'raw':
                    rec['repro_maxabs_vs_r8_proposal_mean'] = float(np.max(np.abs(plan - old_mean)))
                rec['seconds'] = time.perf_counter() - t0
                res['runs'][key] = rec; save()
                # outcomes are printed only for the old-data checks; cohort runs log timing and alpha0 only
                shown = f"margin {round(rec['margin'], 4)} success {rec['success']}" if old_mean is not None else ''
                print('start', k, 'arm', arm, 'alpha0', round(alpha0, 6), 'sec', round(rec['seconds'], 1), shown,
                      ('repro_maxabs ' + str(rec['repro_maxabs_vs_r8_proposal_mean'])) if 'repro_maxabs_vs_r8_proposal_mean' in rec else '',
                      flush=True)
    print('done', len(res['runs']), flush=True)


if __name__ == '__main__':
    main()
