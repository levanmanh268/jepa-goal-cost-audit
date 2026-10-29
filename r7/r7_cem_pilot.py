"""R7 CPU pilot: scores INSIDE the CEM loop, final CEM mean executed open loop (Claude, 2026-10-27). Development.

Follows ChatGPT's R7_NEXT_EXPERIMENT.md (sections A-D) with one declared addition (an exploratory third arm, TPE):
  * starts: R6 training cohort k = 0,1,2,3 (trajectories 18, 9, 15, 2), exact R6/R5 identities; development data,
    already used to fit B1, so this is an integration / optimisation-failure check, not a generalisation test;
  * arms : raw (official latent cost), b1 (frozen R6 head, R7_FROZEN_R6_HEAD.json), tpe (tail patch energy, top-k with
    k fixed in R7_TPE_LOCK.json; exploratory third arm; the PAE arm of R7_ARCH_LOCK was dropped after PAE failed
    its pre-registered mechanism probe, deviation D2);
  * CEM 64 x 8, 10 elites, H = 6, macro-action 10, identical torch generator seed (identity cem_seed) for every arm,
    sample 0 = current mean, no clipping (copied from r6_collect.cem_full);
  * the FINAL CEM MEAN is executed (30 primitive actions, single-plan open loop, no replanning);
  * per iteration: min/mean/elite cost, signed visual and proprio contributions (b1), fraction of negative scores,
    action norms and fraction of |a| > 3 (normalised units), proposal mean/std norms, runtime;
  * after execution: final state, official success and margin, predicted and encoded-true raw/b1/tpe scores of the
    executed plan; a 16-plan diagnostic bank drawn from each arm's final Gaussian (stream [37000000, k, arm]) with
    margins and scores. The diagnostic bank never feeds back into any choice.
Raw-arm reproduction check: R5 drew the evaluation/calibration banks of these starts from its final CEM Gaussian, so the
cached bank plans must equal mean + std * Z for the regenerated R5 normal draws if the raw CEM reproduces R5 exactly.
Usage: python r7_cem_pilot.py --out <dir> --arms raw,b1[,tpe] [--starts 0,1,2,3]
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

WM = Path(str(_WM_ROOT / '')); V06 = WM / 'v06_reconstructed'; D3 = WM / 'd3'; RUNTIME = WM / 'runtime'
VAL = D3 / 'data_official/pusht_noise/val'; R6 = WM / 'r6'; R7 = WM / 'r7'
HEAD = R7 / 'R7_FROZEN_R6_HEAD.json'; TPE_LOCK = R7 / 'R7_TPE_LOCK.json'


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def full_state(env):
    a, b = env.agent, env.block
    return np.array([a.position[0], a.position[1], a.velocity[0], a.velocity[1], b.position[0], b.position[1],
                     b.angle % (2 * np.pi), b.velocity[0], b.velocity[1], b.angular_velocity], np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--arms', default='raw,b1')
    ap.add_argument('--starts', default='0,1,2,3'); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(); arms = a.arms.split(','); starts = [int(x) for x in a.starts.split(',')]
    sys.path.insert(0, str(V06)); sys.path.insert(0, str(D3)); sys.path.insert(0, str(R7))
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc, energies as E, tail_energies as TE
    w = np.array(json.loads(HEAD.read_text())['weights'], np.float64)
    tpe_k = json.loads(TPE_LOCK.read_text())['k_star'] if 'tpe' in arms else None
    env = make_env(RUNTIME / 'jepa-wms'); model, pre, prov = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', 'jepa', 'cpu')
    states, acts_val, lens = ot.load_val(VAL); T = ot.FRAMESKIP * ot.GOAL_H + 1
    with np.load(R6 / 'r6a_cache_train.npz', allow_pickle=False) as z:
        meta = {m['k']: m for m in json.loads(z['meta_json'].tobytes())}; cached_plans = {m: z[f's{m:03d}_plans'].copy() for m in meta}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); res_path = out / 'r7_pilot.json'
    cfg = {'arms': arms, 'starts': starts, 'tpe_k': tpe_k, 'samples': 64, 'iters': 8, 'elites': 10, 'horizon': 6,
           'code_sha256': {'r7_cem_pilot.py': sha_file(__file__), 'energies.py': sha_file(R7 / 'energies.py'), 'tail_energies.py': sha_file(R7 / 'tail_energies.py'),
                           'official_tasks.py': sha_file(D3 / 'official_tasks.py'), 'wm_costs.py': sha_file(D3 / 'wm_costs.py'),
                           'jepa_adapter.py': sha_file(V06 / 'jepa_adapter.py')},
           'head_sha256': sha_file(HEAD), 'tpe_lock_sha256': sha_file(TPE_LOCK) if TPE_LOCK.exists() else None,
           'checkpoint_sha256': prov.get('checkpoint_sha256'), 'torch': torch.__version__}
    res = json.loads(res_path.read_text()) if res_path.exists() else {'config': cfg, 'provenance': prov, 'starts': {}}
    if res['config'] != cfg: raise SystemExit('config/code changed since previous run; refusing to resume')

    for k in starts:
        m = meta[k]; ident = m['identity']; i, off, es, cs = ident['val_episode'], ident['offset'], ident['env_seed'], ident['cem_seed']
        init = states[i, off].copy(); expert = acts_val[i, off:off + T - 1].copy()
        obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, expert)
        sc = wc.ModelScorer(model, obs0, gobs)
        gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy()
        def terms(pv, pp):
            """pv,pp torch terminal latents -> dict of numpy scores for every arm (same float16 convention as R6)."""
            P = pv.double().numpy(); p = pp.double().numpy(); B = P.shape[0]; C = P.shape[-1]
            P2 = P.reshape(B, -1, C); p2 = p.reshape(B, -1)
            ch, pr = E.channel_energies(P2, gv.reshape(-1, C), p2, gp.reshape(-1))
            out_ = {'b1_visual': ch @ w[:-1], 'b1_proprio': pr * w[-1]}; out_['b1'] = out_['b1_visual'] + out_['b1_proprio']
            if tpe_k is not None: out_['tpe'] = TE.topk(P2, gv.reshape(-1, C), p2, gp.reshape(-1), tpe_k)
            return out_
        for arm in arms:
            key = f'{k}:{arm}'
            if key in res['starts']: continue
            t0 = time.perf_counter(); hist = []
            g = torch.Generator().manual_seed(int(cs)); mean = torch.zeros(6, 10); std = torch.ones(6, 10)
            for it in range(8):
                q = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(6, 64, 10, generator=g); q[:, 0, :] = mean
                c_raw, pv, pp = sc(q); tt = terms(pv, pp)
                cost = c_raw if arm == 'raw' else tt[arm]
                el = torch.topk(-torch.from_numpy(np.asarray(cost, np.float64)), 10).indices; eln = el.numpy()
                qa = q.permute(1, 0, 2).reshape(64, -1).numpy()
                hist.append({'iter': it, 'min_cost': float(np.min(cost)), 'mean_cost': float(np.mean(cost)), 'elite_mean_cost': float(np.mean(cost[eln])),
                             'raw_cost_elite_mean': float(c_raw[eln].mean()), 'b1_elite_mean': float(tt['b1'][eln].mean()),
                             'b1_visual_elite_mean': float(tt['b1_visual'][eln].mean()), 'b1_proprio_elite_mean': float(tt['b1_proprio'][eln].mean()),
                             'frac_negative_b1': float((tt['b1'] < 0).mean()), 'action_abs_mean': float(np.abs(qa).mean()),
                             'frac_action_abs_gt3': float((np.abs(qa) > 3).mean()), 'mean_norm': float(mean.norm()), 'std_mean': float(std.mean()),
                             'tpe_elite_mean': float(tt['tpe'][eln].mean()) if 'tpe' in tt else None, 'seconds': time.perf_counter() - t0})
                mean = q[:, el].mean(1); std = q[:, el].std(1)
            plan = mean.numpy().astype(np.float64); stdn = std.numpy().astype(np.float64)
            _, st = ot.fast_rollout(env, es, init, ot.norm_to_env(plan), render_final=False); fs = full_state(env)
            r = ot.eval_state(gstate, st)
            obs_e, _ = ot.fast_rollout(env, es, init, ot.norm_to_env(plan), render_final=True)
            c_e, pv_e, pp_e = sc(torch.from_numpy(plan[None]).permute(1, 0, 2).float()); te = terms(pv_e, pp_e)
            zt = wc.encode_obs_list(model, [obs_e]); tt_true = terms(zt['visual'][:, 0], zt['proprio'][:, 0])
            true_raw = float(wc.latent_cost(zt['visual'][:, 0], zt['proprio'][:, 0], sc.zg).double().numpy()[0])
            # diagnostic bank from this arm's final Gaussian (never used for any choice)
            rng = np.random.default_rng([37000000, k, arms.index(arm)])
            bank = plan[None] + stdn[None] * rng.standard_normal((16, 6, 10))
            cb, pvb, ppb = sc(torch.from_numpy(bank).permute(1, 0, 2).float()); tb = terms(pvb, ppb)
            mb = [ot.eval_state(gstate, ot.fast_rollout(env, es, init, ot.norm_to_env(b), render_final=False)[1])['margin'] for b in bank]
            rec = {'identity': ident, 'arm': arm, 'final_mean': plan.tolist(), 'final_std': stdn.tolist(), 'history': hist,
                   'final_state': np.asarray(st).tolist(), 'final_full_state': fs.tolist(), 'success': r['success'], 'margin': r['margin'],
                   'pos_diff': r['pos_diff'], 'angle_diff': r['angle_diff'],
                   'executed_scores_pred': {'raw': float(c_e[0]), 'b1': float(te['b1'][0]), 'tpe': float(te['tpe'][0]) if 'tpe' in te else None},
                   'executed_scores_true': {'raw': true_raw, 'b1': float(tt_true['b1'][0]), 'tpe': float(tt_true['tpe'][0]) if 'tpe' in tt_true else None},
                   'diag_bank': {'margin': [float(x) for x in mb], 'raw': cb.tolist(), 'b1': tb['b1'].tolist(), 'tpe': tb['tpe'].tolist() if 'tpe' in tb else None},
                   'seconds': time.perf_counter() - t0}
            if arm == 'raw':
                # exact reproduction check of the R5 final Gaussian through the cached bank plans
                cp = cached_plans[k].reshape(-1, 6, 10)
                chk = {}
                for name, seed in [('r5_seed_35000000', 35000000)]:
                    zc = np.random.default_rng([seed, k, 1]).standard_normal((64, 6, 10)); ze = np.random.default_rng([seed, k, 2]).standard_normal((64, 6, 10))
                    recon = np.concatenate([plan[None] + stdn[None] * zc, plan[None] + stdn[None] * ze])
                    chk[name] = float(np.max(np.abs(recon - cp)))
                rec['r5_bank_reconstruction_maxabs'] = chk
            res['starts'][key] = rec
            tmp = Path(str(res_path) + '.tmp'); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, res_path)
            print('start', k, 'arm', arm, 'done', round(rec['seconds'], 1), 'repro' if arm == 'raw' else '', rec.get('r5_bank_reconstruction_maxabs', ''), flush=True)
    print('done', len(res['starts']), flush=True)


if __name__ == '__main__':
    main()
