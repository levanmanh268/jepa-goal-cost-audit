"""Engineering audit (Claude, 2026-10-28; requested by ChatGPT's R8-R10 review): do 16-bit stored predictions change any
selection? Re-scores each stored evaluation bank (the same batch of 64 plans the collector scored) with the released
model in float32, then
  1. checks that the float32 official cost equals the stored model_cost and that rounding the new float32 tokens to
     float16 reproduces the stored float16 tokens exactly (so the new inference is the original one);
  2. recomputes vis, full and position-only proprio terms from float32 and from float16 tokens, and compares the
     argmin of VELDROP-z, of the standardised full term and of the official cost, plus decoded-position differences.
Does not modify any locked output. Usage: python3 -I fp32_check.py <cohort> <jepa|dino> <out.json> [--first N]
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys, time
from pathlib import Path
import numpy as np

V06 = str(_WM_ROOT / 'v06_reconstructed'); D3 = str(_WM_ROOT / 'd3'); RUNTIME = Path(str(_WM_ROOT / 'runtime'))
TRAIN = str(_WM_ROOT / 'd3/data_official/pusht_noise/train')


def fm(c):
    c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1.0, abs(c.min())))[0])


def z(a):
    return (a - a.mean()) / a.std()


def main(argv):
    first = None
    if '--first' in argv: i = argv.index('--first'); first = int(argv[i + 1]); argv = argv[:i] + argv[i + 2:]
    cohort, kind, outp = Path(argv[0]), argv[1], argv[2]
    sys.path.insert(0, V06); sys.path.insert(0, D3)
    import torch; torch.set_num_threads(2)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc
    env = make_env(RUNTIME / 'jepa-wms'); model, _, _ = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', kind, 'cpu')
    pe = torch.load(RUNTIME / f'checkpoints/{kind}_wm_pusht.pth.tar', map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy(); Wi = np.linalg.pinv(W); m = W.shape[0]
    states, acts, lens = ot.load_val(TRAIN); T = ot.FRAMESKIP * ot.GOAL_H + 1
    idx = json.loads((cohort / 'r6_index.json').read_text()); keys = sorted(idx['starts'], key=int)
    if first: keys = keys[:first]
    rows = []
    for k in keys:
        t0 = time.perf_counter(); ident = idx['starts'][k]['identity']; i, off, es = ident['val_episode'], ident['offset'], ident['env_seed']
        init = states[i, off].copy(); obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, acts[i, off:off + T - 1].copy())
        zf = np.load(cohort / f'start_{int(k):03d}.npz'); ev = zf['split'] == 1; plans = zf['plans'][ev]
        assert np.array_equal(np.asarray(gstate, np.float64), zf['goal_state'])
        sc = wc.ModelScorer(model, obs0, gobs)
        pc, pv, pp = sc(torch.from_numpy(plans).permute(1, 0, 2).float())
        pv32 = pv.double().numpy(); pp32 = pp.double().numpy(); n = plans.shape[0]
        gv = sc.zg['visual'][0, 0].double().numpy().reshape(-1, pv32.shape[-1]); gp = sc.zg['proprio'][0, 0].double().numpy().reshape(-1)
        same_cost = float(np.max(np.abs(np.asarray(pc, np.float64) - zf['model_cost'][ev])))
        same16_v = bool(np.array_equal(pv32.astype(np.float16), zf['pv16'][ev])); same16_p = bool(np.array_equal(pp32.astype(np.float16), zf['pp16'][ev]))

        def comps(PV, PP):
            P = PV.reshape(n, -1, PV.shape[-1]); D = P - gv[None]; vis = (D * D).mean((1, 2))
            p = PP.reshape(n, -1); full = ((p - gp[None]) ** 2).mean(1)
            r = (p.reshape(n, -1, m).mean(1) - b) @ Wi.T; rg = (gp.reshape(-1, m).mean(0) - b) @ Wi.T
            pos = (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1)
            return vis, full, pos, r[:, :2]
        v32, f32, p32, x32 = comps(pv32, pp32); v16, f16, p16, x16 = comps(zf['pv16'][ev].astype(np.float64), zf['pp16'][ev].astype(np.float64))
        sel = {}
        for nm, (a32, a16) in {'veldrop_z': (z(v32) + z(p32), z(v16) + z(p16)), 'hprop_std': (z(v32) + z(f32), z(v16) + z(f16)),
                               'official': (v32 + 0.1 * f32, v16 + 0.1 * f16)}.items():
            j32, j16 = fm(a32), fm(a16)
            sel[nm] = {'argmin32': j32, 'argmin16': j16, 'same': j32 == j16, 'margin32': float(zf['margin'][ev][j32]), 'margin16': float(zf['margin'][ev][j16])}
        rows.append({'k': int(k), 'max_abs_cost_diff_vs_stored': same_cost, 'fp16_roundtrip_equals_stored_vis': same16_v, 'fp16_roundtrip_equals_stored_prop': same16_p,
                     'max_rel_pos_energy_diff': float(np.max(np.abs(p32 - p16) / np.maximum(np.abs(p32), 1e-30))),
                     'max_decoded_norm_pos_diff': float(np.max(np.abs(x32 - x16))), 'selections': sel, 'seconds': time.perf_counter() - t0})
        print(k, 'done', round(rows[-1]['seconds'], 1), flush=True)
    summ = {'n': len(rows), 'all_roundtrips_equal': all(r['fp16_roundtrip_equals_stored_vis'] and r['fp16_roundtrip_equals_stored_prop'] for r in rows),
            'max_cost_diff': max(r['max_abs_cost_diff_vs_stored'] for r in rows),
            'selection_changes': {nm: int(sum(not r['selections'][nm]['same'] for r in rows)) for nm in ['veldrop_z', 'hprop_std', 'official']},
            'mean_margin_32_vs_16': {nm: [float(np.mean([r['selections'][nm]['margin32'] for r in rows])), float(np.mean([r['selections'][nm]['margin16'] for r in rows]))]
                                     for nm in ['veldrop_z', 'hprop_std', 'official']},
            'max_rel_pos_energy_diff': max(r['max_rel_pos_energy_diff'] for r in rows)}
    json.dump({'model': kind, 'summary': summ, 'rows': rows}, open(outp, 'w'), indent=1); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
