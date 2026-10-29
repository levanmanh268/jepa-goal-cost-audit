"""R9 post hoc DESCRIPTIVE analysis (not part of any locked decision; written before r9_inloop.json exists).
Optimism gap of each arm's executed plan: re-simulate the executed plan with rendering, encode the true final
observation, and compare the arm's own score on predicted vs encoded-true endpoints (raw cost, B1 head, and the
position-only proprio term). A planner that exploits model error under its score shows predicted << true.
Usage: python3 -I r9_optimism.py --cohort cohort --runs r9/r9_inloop.json --out r9_optimism.json
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import argparse, json, sys
from pathlib import Path
import numpy as np

WM = Path(str(_WM_ROOT / '')); V06 = WM / 'v06_reconstructed'; D3 = WM / 'd3'; RUNTIME = WM / 'runtime'; R7 = WM / 'r7'
TRAIN = D3 / 'data_official/pusht_noise/train'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', required=True); ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--threads', type=int, default=2); a = ap.parse_args()
    sys.path.insert(0, str(V06)); sys.path.insert(0, str(D3)); sys.path.insert(0, str(R7))
    import torch; torch.set_num_threads(a.threads)
    from jepa_adapter import make_env, load_model
    import official_tasks as ot, wm_costs as wc, energies as E
    w = np.array(json.loads((R7 / 'R7_FROZEN_R6_HEAD.json').read_text())['weights'])
    pe = torch.load(RUNTIME / 'checkpoints/jepa_wm_pusht.pth.tar', map_location='cpu', weights_only=False)['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy(); Wi = np.linalg.pinv(W)
    env = make_env(RUNTIME / 'jepa-wms'); model, _, _ = load_model(RUNTIME / 'jepa-wms', RUNTIME / 'dinov2', RUNTIME / 'checkpoints', 'jepa', 'cpu')
    states, acts, lens = ot.load_val(TRAIN); T = ot.FRAMESKIP * ot.GOAL_H + 1
    R = json.load(open(a.runs))['runs']; out = []
    for key, run in sorted(R.items(), key=lambda kv: (int(kv[0].split(':')[0]), kv[0])):
        if run['arm'] == 'raw': continue
        ident = run['identity']; i, off, es = ident['val_episode'], ident['offset'], ident['env_seed']
        init = states[i, off].copy(); obs0, _ = ot.prepare(env, es, init); gobs, gstate = ot.fast_rollout(env, es, init, acts[i, off:off + T - 1].copy())
        sc = wc.ModelScorer(model, obs0, gobs); gv = sc.zg['visual'][0, 0].double().numpy(); gp = sc.zg['proprio'][0, 0].double().numpy(); C = gv.shape[-1]
        plan = np.array(run['final_mean'])
        obs_e, st = ot.fast_rollout(env, es, init, ot.norm_to_env(plan), render_final=True)
        assert abs(ot.eval_state(gstate, st)['margin'] - run['margin']) < 1e-9, key
        c_p, pv, pp = sc(torch.from_numpy(plan[None]).permute(1, 0, 2).float()); zt = wc.encode_obs_list(model, [obs_e])

        def scores(Pv, Pp):
            P = Pv.double().numpy().reshape(1, -1, C); p = Pp.double().numpy().reshape(1, -1)
            raw = float(((P.reshape(1, -1) - gv.reshape(-1)) ** 2).mean() + 0.1 * ((p - gp.reshape(-1)) ** 2).mean())
            r = (p.reshape(1, -1, W.shape[0]).mean(1) - b) @ Wi.T; rg = (gp.reshape(-1, W.shape[0]).mean(0) - b) @ Wi.T
            pos = float((((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean())
            return {'raw': raw, 'b1': float(E.head(P, gv.reshape(-1, C), p, gp.reshape(-1), w)[0]), 'prop_pos': pos}
        sp, stt = scores(pv, pp), scores(zt['visual'][:, 0], zt['proprio'][:, 0])
        out.append({'key': key, 'arm': run['arm'], 'margin': run['margin'], 'pred': sp, 'true': stt})
        print(key, 'done', flush=True)
    summ = {}
    for arm in sorted({o['arm'] for o in out}):
        rows = [o for o in out if o['arm'] == arm]
        summ[arm] = {s: {'pred_mean': float(np.mean([o['pred'][s] for o in rows])), 'true_mean': float(np.mean([o['true'][s] for o in rows])),
                         'median_true_over_pred': float(np.median([o['true'][s] / max(o['pred'][s], 1e-12) for o in rows]))} for s in ['raw', 'b1', 'prop_pos']}
    json.dump({'summary': summ, 'rows': out}, open(a.out, 'w'), indent=1); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
