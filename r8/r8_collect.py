"""R8 = r6_collect.py with the data source switched to the official PushT TRAIN split (Claude, 2026-10-27).
Only the VAL path constant differs from the R6 collector (fresh mode used; reuse mode not applicable).

Original docstring: R6-A data collection in COMMON latent coordinates (Claude, 2026-10-26). Development. CPU.

Purpose: Protocol A (shared scoring head across starts/goals) needs predicted, encoded-true and goal latents in the
model's own coordinate system for every start. R5 stored per-start QR coordinates, whose bases differ across starts,
so they cannot be pooled. This collector stores the original latents (visual patches x channels, proprio) as float16,
plus exact float64 latent costs, simulator labels and provenance.

Two modes:
  reuse : re-collect an existing R5 start (same identity, stored plans; no CEM). Every stored simulator quantity
          (goal state, endpoint full states, official states, margins) must match the R5 NPZ exactly, and the model
          cost must match R5's model_cost to 1e-9 relative, otherwise the start is refused.
  fresh : new start from the official val split (trajectory, offset, env seed and CEM seed drawn from a keyed RNG),
          JEPA-WM CEM 64 x 8 exactly as R5 (r5_runner.cem_full logic, copied below), then a calibration and an
          evaluation bank of 64 plans each from the final Gaussian, with RNG streams [seed, k, 1] and [seed, k, 2].
Output per start: NPZ (atomic) + index JSON with identity, hashes, timing; resume verifies identity, code and data hashes.
Usage:
  python -I r6_collect.py --mode reuse --r5run <run_dir> --out <dir> [--only k]
  python -I r6_collect.py --mode fresh --episodes 0,1,3 --seed 36000000 --out <dir> [--only k]
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

V06 = str(_WM_ROOT / 'v06_reconstructed'); D3 = str(_WM_ROOT / 'd3'); RUNTIME = str(_WM_ROOT / 'runtime')
VAL = str(_WM_ROOT / 'd3/data_official/pusht_noise/train')   # R8: official TRAIN split (unseen by heads/energies; seen by the checkpoint)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def full_state(env):
    a, b = env.agent, env.block
    return np.array([a.position[0], a.position[1], a.velocity[0], a.velocity[1], b.position[0], b.position[1],
                     b.angle % (2 * np.pi), b.velocity[0], b.velocity[1], b.angular_velocity], np.float64)


def cem_full(score, n, iters, elites, seed, horizon=6, adim=10):
    """Copied from r5_runner.py (same sampling and update as D3 wm_costs.cem)."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    mean = torch.zeros(horizon, adim); std = torch.ones(horizon, adim); hist = []
    for it in range(iters):
        acts = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(horizon, n, adim, generator=g)
        acts[:, 0, :] = mean
        cost = score(acts); el = torch.topk(-torch.from_numpy(cost), elites).indices
        hist.append({'iter': it, 'min_cost': float(cost.min()), 'elite_mean_cost': float(cost[el.numpy()].mean())})
        mean = acts[:, el].mean(1); std = acts[:, el].std(1)
    return mean.numpy().astype(np.float64), std.numpy().astype(np.float64), hist


def atomic_save_npz(path, **arrays):
    tmp = Path(str(path) + '.tmp.npz'); np.savez_compressed(tmp, **arrays); os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['reuse', 'fresh'], required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--r5run', default=None); ap.add_argument('--episodes', default=None); ap.add_argument('--seed', type=int, default=36000000)
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--only', type=int, default=None)
    a = ap.parse_args()
    sys.path.insert(0, V06); sys.path.insert(0, D3)
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc
    R = Path(RUNTIME); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = make_env(R / 'jepa-wms'); model, pre, prov = load_model(R / 'jepa-wms', R / 'dinov2', R / 'checkpoints', 'jepa', 'cpu')
    states, acts, lens = ot.load_val(VAL); T = ot.FRAMESKIP * ot.GOAL_H + 1
    cfg = {'mode': a.mode, 'r5run': a.r5run, 'episodes': a.episodes, 'seed': a.seed, 'samples': 64, 'iters': 8, 'elites': 10, 'n_cal': 64, 'n_eval': 64,
           'code_sha256': {'r6_collect.py': sha_file(__file__), 'official_tasks.py': sha_file(Path(D3) / 'official_tasks.py'),
                           'wm_costs.py': sha_file(Path(D3) / 'wm_costs.py'), 'jepa_adapter.py': sha_file(Path(V06) / 'jepa_adapter.py')},
           'val_sha256': {f.name: sha_file(f) for f in sorted(Path(VAL).iterdir())}, 'checkpoint_sha256': prov.get('checkpoint_sha256')}
    if a.mode == 'reuse':
        r5 = Path(a.r5run); r5idx = json.loads((r5 / 'r5_index.json').read_text())
        cfg['r5_index_sha256'] = sha_file(r5 / 'r5_index.json')
        jobs = []
        for k, rec in sorted(r5idx['starts'].items(), key=lambda kv: int(kv[0])):
            f = r5 / f'start_{int(k):03d}.npz'; assert sha_file(f) == rec['npz_sha256'], f
            jobs.append((int(k), rec['identity'], f))
    else:
        eps = [int(x) for x in a.episodes.split(',')]; jobs = []
        for k, i in enumerate(eps):
            assert lens[i] >= T, i
            rng = np.random.default_rng([a.seed, i]); off = int(rng.integers(0, lens[i] - T + 1))
            es = int(rng.integers(0, 2 ** 31 - 1)); cs = int(rng.integers(0, 2 ** 31 - 1))
            jobs.append((k, {'k': k, 'val_episode': i, 'offset': off, 'env_seed': es, 'cem_seed': cs}, None))
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    idx_path = out / 'r6_index.json'
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {'config': cfg, 'config_sha256': cfg_hash, 'provenance': prov, 'starts': {}}
    if index['config_sha256'] != cfg_hash: raise SystemExit('config/code/data changed since previous run; refusing to resume')
    for k, ident, r5f in jobs:
        if a.only is not None and k != a.only: continue
        f = out / f'start_{k:03d}.npz'
        if str(k) in index['starts'] and f.exists():
            if index['starts'][str(k)]['identity'] != ident or index['starts'][str(k)]['npz_sha256'] != sha_file(f):
                raise SystemExit(f'identity or hash mismatch for start {k}; refusing to resume')
            continue
        T0 = time.perf_counter(); tim = {}
        i, off, es, cs = ident['val_episode'], ident['offset'], ident['env_seed'], ident['cem_seed']
        init = states[i, off].copy(); expert = acts[i, off:off + T - 1].copy()
        obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, expert)
        sc = wc.ModelScorer(model, obs0, gobs)
        checks = {}
        if a.mode == 'reuse':
            old = np.load(r5f)
            assert np.array_equal(np.asarray(gstate), old['goal_state']) and np.array_equal(init, old['init_state'])
            plans = old['plans'].copy(); split = old['split'].copy(); mean, std = old['proposal_mean'], old['proposal_std']; hist = None
        else:
            t = time.perf_counter(); mean, std, hist = cem_full(lambda x: sc(x)[0], 64, 8, 10, cs); tim['cem'] = time.perf_counter() - t
            rc = np.random.default_rng([a.seed, k, 1]); re_ = np.random.default_rng([a.seed, k, 2])
            plans = np.concatenate([mean[None] + std[None] * rc.standard_normal((64, 6, 10)), mean[None] + std[None] * re_.standard_normal((64, 6, 10))], 0)
            split = np.array([0] * 64 + [1] * 64)
        t = time.perf_counter(); obs_f, fs, offs, marg, succ = [], [], [], [], []
        for p in plans:
            o, st = ot.fast_rollout(env, es, init, ot.norm_to_env(p), render_final=True)
            obs_f.append(o); fs.append(full_state(env)); offs.append(st); r = ot.eval_state(gstate, st); marg.append(r['margin']); succ.append(r['success'])
        tim['sim'] = time.perf_counter() - t
        fs = np.array(fs); offs = np.array(offs); marg = np.array(marg); succ = np.array(succ)
        t = time.perf_counter(); pcs, pvs, pps = [], [], []
        for s0 in range(0, len(plans), 64):
            P_t = torch.from_numpy(plans[s0:s0 + 64]).permute(1, 0, 2).float(); pc, pv, pp = sc(P_t)
            pcs.append(pc); pvs.append(pv.double().numpy()); pps.append(pp.double().numpy())
        tim['predict'] = time.perf_counter() - t
        pc = np.concatenate(pcs); pv = np.concatenate(pvs); pp = np.concatenate(pps)
        t = time.perf_counter(); zt = wc.encode_obs_list(model, obs_f); z0 = wc.encode_obs_list(model, [obs0]); tim['encode'] = time.perf_counter() - t
        tv = zt['visual'][:, 0].double().numpy(); tp = zt['proprio'][:, 0].double().numpy()
        gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy()
        z0v = z0['visual'][0, 0].double().numpy(); z0p = z0['proprio'][0, 0].double().numpy()
        # exact costs in float64 from the float64 latents (official objective: mean visual sq + 0.1 mean proprio sq)
        n = len(plans); pvf = pv.reshape(n, -1); ppf = pp.reshape(n, -1)
        cost64 = ((pvf - gv.reshape(-1)) ** 2).mean(1) + 0.1 * ((ppf - gp.reshape(-1)) ** 2).mean(1)
        true64 = ((tv.reshape(n, -1) - gv.reshape(-1)) ** 2).mean(1) + 0.1 * ((tp.reshape(n, -1) - gp.reshape(-1)) ** 2).mean(1)
        checks['model_cost_vs_scorer_rel'] = float(np.max(np.abs(cost64 - pc) / np.maximum(np.abs(pc), 1e-12)))
        pv16 = pv.astype(np.float16); pp16 = pp.astype(np.float16); tv16 = tv.astype(np.float16); tp16 = tp.astype(np.float16)
        c16 = ((pv16.reshape(n, -1).astype(np.float64) - gv.reshape(-1)) ** 2).mean(1) + 0.1 * ((pp16.reshape(n, -1).astype(np.float64) - gp.reshape(-1)) ** 2).mean(1)
        checks['float16_cost_rel_err_max'] = float(np.max(np.abs(c16 - cost64) / cost64))
        checks['float16_argmin_eval_equal'] = bool(np.argmin(c16[split == 1]) == np.argmin(cost64[split == 1]))
        if a.mode == 'reuse':
            checks['margin_equal'] = bool(np.array_equal(marg, old['margin'])); checks['full_state_equal'] = bool(np.array_equal(fs, old['full_state']))
            checks['official_state_equal'] = bool(np.array_equal(offs, old['official_state']))
            checks['model_cost_vs_r5_rel'] = float(np.max(np.abs(pc - old['model_cost']) / np.abs(old['model_cost'])))
            checks['true_cost_vs_r5_rel'] = float(np.max(np.abs(true64 - old['true_latent_cost']) / np.abs(old['true_latent_cost'])))
            if not (checks['margin_equal'] and checks['full_state_equal'] and checks['official_state_equal'] and checks['model_cost_vs_r5_rel'] < 1e-6):
                raise SystemExit(f'reuse start {k} does not reproduce R5: {checks}')
        atomic_save_npz(f, pv16=pv16, pp16=pp16, tv16=tv16, tp16=tp16, gv=gv.astype(np.float32), gp=gp.astype(np.float32),
                        z0v=z0v.astype(np.float32), z0p=z0p.astype(np.float32), model_cost=pc, cost64=cost64, true_cost64=true64,
                        plans=plans, split=split, full_state=fs, official_state=offs, margin=marg, success=succ,
                        goal_state=np.asarray(gstate), init_state=np.asarray(init), proposal_mean=mean, proposal_std=std)
        rec = {'identity': ident, 'npz_sha256': sha_file(f), 'timing_s': tim, 'checks': checks, 'cem_history': hist,
               'visual_shape': list(pv.shape[1:]), 'proprio_shape': list(pp.shape[1:]), 'seconds': time.perf_counter() - T0}
        index['starts'][str(k)] = rec; atomic_write_json(idx_path, index)
        print('start', k, 'ep', i, 'off', off, 'vis', list(pv.shape[1:]), 'pro', list(pp.shape[1:]),
              {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in checks.items()}, {kk: round(v, 1) for kk, v in tim.items()},
              round(rec['seconds'], 1), flush=True)
    print('done', len(index['starts']), flush=True)


if __name__ == '__main__':
    main()
