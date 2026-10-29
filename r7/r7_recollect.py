"""R7 re-collection of the R6 banks with FULL latent grids, rebuilt from the R6 compact caches (Claude, 2026-10-27).

Why: the R6 compact caches keep only channel/spatial energies, so any energy that needs patch positions or
cross-channel structure (transport energies, PSD metrics) cannot be evaluated from them. The original raw NPZ files
(48 MB/start) stayed in the R6 session container, which is not reachable from here.

What: for every start in r6a_cache_{train,test}.npz, rebuild the episode from its recorded identity (val_episode,
offset, env_seed) and the official val split, replay the stored 128 plans (no CEM, no new plans), and store the
predicted, encoded-true and goal latents. Provenance checks per start (the start is REFUSED if any fails):
  * every simulator margin equals the cached margin exactly (np.array_equal);
  * the model's official cost reproduces the cached exact B0 cost to <= 1e-6 relative (R6 used the same bar vs R5);
  * the channel energies recomputed from the stored float16 tokens equal the cached `ch` features (<= 1e-9 abs);
This prints identities, check values and timing only, never outcomes of any new energy.
Usage: python r7_recollect.py --cohort train|test --out <dir> [--only k] [--threads 2]
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

WM = Path(str(_WM_ROOT / '')); V06 = WM / 'v06_reconstructed'; D3 = WM / 'd3'; RUNTIME = WM / 'runtime'
VAL = D3 / 'data_official/pusht_noise/val'; R6 = WM / 'r6'


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def full_state(env):
    a, b = env.agent, env.block
    return np.array([a.position[0], a.position[1], a.velocity[0], a.velocity[1], b.position[0], b.position[1],
                     b.angle % (2 * np.pi), b.velocity[0], b.velocity[1], b.angular_velocity], np.float64)


def load_cache(name):
    with np.load(R6 / f'r6a_cache_{name}.npz', allow_pickle=False) as z:
        meta = json.loads(z['meta_json'].tobytes())
        return [{**m, **{k.split('_', 1)[1]: z[k].copy() for k in z.files if k.startswith(f"s{m['k']:03d}_")}} for m in meta]


def atomic_save_npz(path, **arrays):
    tmp = Path(str(path) + '.tmp.npz'); np.savez(tmp, **arrays); os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', choices=['train', 'test'], required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--only', type=int, default=None); ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, str(V06)); sys.path.insert(0, str(D3))
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = make_env(RUNTIME / 'jepa-wms'); model, pre, prov = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', 'jepa', 'cpu')
    states, acts, lens = ot.load_val(VAL); T = ot.FRAMESKIP * ot.GOAL_H + 1
    cache = load_cache(a.cohort)
    cfg = {'cohort': a.cohort, 'source_cache_sha256': sha_file(R6 / f'r6a_cache_{a.cohort}.npz'),
           'code_sha256': {'r7_recollect.py': sha_file(__file__), 'official_tasks.py': sha_file(D3 / 'official_tasks.py'),
                           'wm_costs.py': sha_file(D3 / 'wm_costs.py'), 'jepa_adapter.py': sha_file(V06 / 'jepa_adapter.py')},
           'val_sha256': {f.name: sha_file(f) for f in sorted(VAL.iterdir())}, 'checkpoint_sha256': prov.get('checkpoint_sha256'),
           'torch': torch.__version__, 'numpy': np.__version__, 'threads': a.threads}
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    idx_path = out / 'r7_index.json'
    index = json.loads(idx_path.read_text()) if idx_path.exists() else {'config': cfg, 'config_sha256': cfg_hash, 'provenance': prov, 'starts': {}}
    if index['config_sha256'] != cfg_hash: raise SystemExit('config/code/data changed since previous run; refusing to resume')
    for s in cache:
        k = int(s['k'])
        if a.only is not None and k != a.only: continue
        f = out / f'start_{k:03d}.npz'
        if str(k) in index['starts'] and f.exists():
            if index['starts'][str(k)]['npz_sha256'] != sha_file(f): raise SystemExit(f'hash mismatch for start {k}; refusing to resume')
            continue
        T0 = time.perf_counter(); tim = {}; ident = s['identity']
        i, off, es = ident['val_episode'], ident['offset'], ident['env_seed']
        init = states[i, off].copy(); expert = acts[i, off:off + T - 1].copy()
        obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, expert)
        sc = wc.ModelScorer(model, obs0, gobs)
        n = len(s['margin']); plans = s['plans'].reshape(n, ot.GOAL_H, 2 * ot.FRAMESKIP).astype(np.float64)
        # simulator: replay the stored plans, final frames rendered for encoding
        t = time.perf_counter(); obs_f, fs, offs, marg, succ = [], [], [], [], []
        for p in plans:
            o, st = ot.fast_rollout(env, es, init, ot.norm_to_env(p), render_final=True)
            obs_f.append(o); fs.append(full_state(env)); offs.append(st); r = ot.eval_state(gstate, st); marg.append(r['margin']); succ.append(r['success'])
        tim['sim'] = time.perf_counter() - t
        marg = np.array(marg); succ = np.array(succ)
        # predictor: same batching as R6 (64 per call)
        t = time.perf_counter(); pcs, pvs, pps = [], [], []
        for s0 in range(0, n, 64):
            P_t = torch.from_numpy(plans[s0:s0 + 64]).permute(1, 0, 2).float(); pc, pv, pp = sc(P_t)
            pcs.append(pc); pvs.append(pv.double().numpy()); pps.append(pp.double().numpy())
        tim['predict'] = time.perf_counter() - t
        pc = np.concatenate(pcs); pv = np.concatenate(pvs); pp = np.concatenate(pps)
        t = time.perf_counter(); zt = wc.encode_obs_list(model, obs_f); tim['encode'] = time.perf_counter() - t
        tv = zt['visual'][:, 0].double().numpy(); tp = zt['proprio'][:, 0].double().numpy()
        gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy()
        z0v = sc.z0['visual'][0, 0].double().numpy(); z0p = sc.z0['proprio'][0, 0].double().numpy()
        pv16 = pv.astype(np.float16); pp16 = pp.astype(np.float16); tv16 = tv.astype(np.float16); tp16 = tp.astype(np.float16)
        # provenance checks against the R6 cache
        C = pv.shape[-1]
        dv = pv16.reshape(n, -1, C).astype(np.float64) - gv.astype(np.float32).astype(np.float64).reshape(-1, C)
        ch = (dv * dv).mean(1)
        checks = {'margin_equal': bool(np.array_equal(marg, s['margin'])),
                  'success_equal': bool(np.array_equal(succ, s['success'])),
                  'model_cost_vs_cache_rel': float(np.max(np.abs(pc - s['b0']) / np.abs(s['b0']))),
                  'ch_vs_cache_abs': float(np.max(np.abs(ch - s['ch'])))}
        ok = checks['margin_equal'] and checks['success_equal'] and checks['model_cost_vs_cache_rel'] < 1e-6 and checks['ch_vs_cache_abs'] < 1e-9
        if not ok: raise SystemExit(f'start {k} does not reproduce the R6 cache: {checks}')
        atomic_save_npz(f, pv16=pv16, pp16=pp16, tv16=tv16, tp16=tp16, gv=gv.astype(np.float32), gp=gp.astype(np.float32),
                        z0v=z0v.astype(np.float32), z0p=z0p.astype(np.float32), model_cost=pc, margin=marg, success=succ,
                        plans=plans, split=s['split'], full_state=np.array(fs), official_state=np.array(offs),
                        goal_state=np.asarray(gstate), init_state=np.asarray(init))
        rec = {'identity': ident, 'traj': int(s['traj']), 'npz_sha256': sha_file(f), 'timing_s': tim, 'checks': checks,
               'visual_shape': list(pv.shape[1:]), 'proprio_shape': list(pp.shape[1:]), 'seconds': time.perf_counter() - T0}
        index['starts'][str(k)] = rec; atomic_write_json(idx_path, index)
        print('start', k, 'traj', s['traj'], {kk: (round(v, 9) if isinstance(v, float) else v) for kk, v in checks.items()},
              {kk: round(v, 1) for kk, v in tim.items()}, round(rec['seconds'], 1), flush=True)
    print('done', len(index['starts']), flush=True)


if __name__ == '__main__':
    main()
