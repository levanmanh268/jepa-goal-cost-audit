"""R8 locked analysis (Claude, 2026-10-27). Run ONCE on the R8 cohort after collection ends. Development grade.

Protocol: R8_PROTOCOL_LOCK.md (primary: frozen R6 head B1 vs official cost B0 on unseen goals from the official PushT
TRAIN split) and R8_ADDENDUM_LOCK.md (label-free addendum: VELDROP-z vs B0, plus a mechanism check). Every row is
computed on each start's 64-plan EVALUATION bank (split == 1); argmin with the lowest index on ties.

Rules
  B0         official cost as returned by the model scorer (`model_cost`), as in R6/R7
  B1         frozen R6 head (R7_FROZEN_R6_HEAD.json), channel energies through float16 as in R6
  HPROP10    vis + 10 * prop                         (R6 H_prop lambda*)
  HPROP_STD  z(vis) + z(prop), z-scores within the bank   (R6 H_prop-std)
  TPE        top-2 mean of patch errors + 0.1 prop    (R7, k* = 2)
  POW        power mean q = 8 of patch errors + 0.1 prop (R7, q* = 8)
  FIXEDMASK  mean patch error over the 128 patches with the largest goal-vs-start change + 0.1 prop
  E_B0, E_B1 the same B0/B1 scores on ENCODED TRUE endpoints (privileged)
  VELDROP_Z  addendum: z(vis) + z(prop_pos), where prop_pos is the official proprio term with the decoded velocity
             difference set to zero. Decoding uses only the model's proprio encoder (Linear 4->16, from the checkpoint)
             and the documented proprio layout [x, y, vx, vy]: r = pinv(W)(mean_token(emb) - b);
             prop_pos = mean_k (W[:, :2] (r_xy - r_xy_goal))_k^2.
  POS_ONLY, VELDROP_01 (vis + 0.1 prop_pos): addendum descriptive rows.
  O          best plan in the bank (privileged); RANDOM: mean over the bank.
Decisions
  PRIMARY (B1 vs B0):  REPLICATED iff mean D <= -0.10 AND successes(B1) >= successes(B0) AND mean D without the single
                       most favourable start <= 0; NOT_REPLICATED otherwise; INCOMPLETE if n < 8 or provenance fails.
  ADDENDUM (VELDROP_Z vs B0): SUPPORTED / NOT_SUPPORTED with the same three conditions; INCOMPLETE as above.
  MECHANISM: REPLICATED iff mean within-bank Spearman(prop_pos, true final agent distance) >= 0.80 AND mean
             Spearman(prop, true final agent distance) <= 0.60; NOT_REPLICATED otherwise.
Bootstrap over starts: 20,000 resamples, seed 38, descriptive only. Exact paired success gains/losses reported.

Usage (locked):  python3 -I r8_analyze.py cohort r8_analysis.json
Dry run on development banks (code check only, no R8 data): python3 -I r8_analyze.py --dry-run <r7 bank dir> <out.json>
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

R7 = Path(str(_WM_ROOT / 'r7'))
sys.path.insert(0, str(R7))
import energies as E, tail_energies as T

HEAD = R7 / 'R7_FROZEN_R6_HEAD.json'; TPE_LOCK = R7 / 'R7_TPE_LOCK.json'
CKPT = Path(str(_WM_ROOT / 'runtime/checkpoints/jepa_wm_pusht.pth.tar'))
DELTA = 0.10; N_MIN = 8; BOOT = 20000; BOOT_SEED = 38
LOCKED = {
    'seed': 38000000, 'samples': 64, 'iters': 8, 'elites': 10, 'n_cal': 64, 'n_eval': 64, 'mode': 'fresh',
    'episodes': '3635,2466,3108,16642,3665,18150,15360,12440,1941,10673,12946,4828,6331,10136,13906,10710',
    'code_sha256': {'r6_collect.py': '6c6a2e858b5e3bbce4b2d85aaeb272583307d929fe12cfe827e60415db72d588',
                    'official_tasks.py': '0ad42f0af306617141cabbdde7bad219f88c040aa0fcc45049bfed8652efa496',
                    'wm_costs.py': '4f941022c4866a777bef34db06bdda9a963a1b65f52aaeced5bc344d0b2f5cc6',
                    'jepa_adapter.py': 'b3effe85c1e62912759251a9265118766d58bf3763d54b33c50b4f0ff85f4279'},
    'data_sha256': {'states.pth': 'e539d89d18637f5d35cf429c13ba04ce9a578d208e3e8a8d93043ea60213042e',
                    'velocities.pth': 'b953d994ed736c85ecc64975192d6365be136866656019f76c19d7569f5292b8',
                    'rel_actions.pth': '99a023838d0a9215848c1640cedf3e53c3ae241cfc5365bc642d508c02dcf53c',
                    'seq_lengths.pkl': '60b98f3e84b05717f088948b895a8c37debc93a0f46f16887bd749fbb87681a1'},
    'checkpoint_sha256': '9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb',
    'head_sha256': '5502e1aa79fe97649259ea1817250b06cae0413d11711c59ad7bc0e28982e117',
    'tpe_lock_sha256': '24bcdec1ac4ee6684bbeab9ccaaa50203d9cae1488935023ba059e78a10b95cd'}
RULES = ['B0', 'B1', 'HPROP10', 'HPROP_STD', 'TPE', 'POW', 'FIXEDMASK', 'E_B0', 'E_B1', 'VELDROP_Z', 'POS_ONLY', 'VELDROP_01']


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()


def pick(c):
    c = np.asarray(c, float); lo = c.min(); return int(np.flatnonzero(c <= lo + 1e-12 * max(1., abs(lo)))[0])


def boot(x):
    x = np.asarray(x, float); r = np.random.default_rng(BOOT_SEED); m = x[r.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def z(a):
    return (a - a.mean()) / (a.std() + 1e-12)


def proprio_encoder():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False); pe = ck['proprio_encoder']
    W = pe['module.patch_embed.weight'][:, :, 0].double().numpy(); b = pe['module.patch_embed.bias'].double().numpy()
    return W, b, np.linalg.pinv(W)


def prop_pos_term(p_tok, gp_tok, W, b, Wp):
    """p_tok [n, tokens, 16] predicted proprio embeddings, gp_tok [tokens, 16]. Official proprio term with the decoded
    velocity difference removed (position block of the linear proprio encoder only)."""
    r = (p_tok.mean(1) - b) @ Wp.T; rg = (gp_tok.mean(0) - b) @ Wp.T
    return (((r[:, :2] - rg[:2]) @ W[:, :2].T) ** 2).mean(1), r, rg


def components(fs, gs):
    a = np.linalg.norm(fs[:, 0:2] - gs[0:2], axis=1); bl = np.linalg.norm(fs[:, 2:4] - gs[2:4], axis=1)
    th = np.abs(fs[:, 4] - gs[4]) % (2 * np.pi); th = np.minimum(th, 2 * np.pi - th)
    return a, bl, np.degrees(th)


def provenance(cohort, index):
    cfg = index['config']; fails = []
    for key in ['seed', 'samples', 'iters', 'elites', 'n_cal', 'n_eval', 'mode', 'episodes']:
        if cfg.get(key) != LOCKED[key]: fails.append(f'config {key}: {cfg.get(key)} != {LOCKED[key]}')
    for name, h in LOCKED['code_sha256'].items():
        if cfg['code_sha256'].get(name) != h: fails.append(f'code hash {name}')
    for name, h in LOCKED['data_sha256'].items():
        if cfg['val_sha256'].get(name) != h: fails.append(f'data hash {name}')
    if cfg.get('checkpoint_sha256') != LOCKED['checkpoint_sha256']: fails.append('checkpoint hash')
    if sha_file(HEAD) != LOCKED['head_sha256']: fails.append('frozen head hash')
    if sha_file(TPE_LOCK) != LOCKED['tpe_lock_sha256']: fails.append('TPE lock hash')
    for k, rec in index['starts'].items():
        if sha_file(Path(cohort) / f'start_{int(k):03d}.npz') != rec['npz_sha256']: fails.append(f'npz hash start {k}')
        if not rec['checks'].get('float16_argmin_eval_equal', False): fails.append(f'float16 argmin check start {k}')
    eps = [int(x) for x in LOCKED['episodes'].split(',')]
    ks = sorted(int(k) for k in index['starts'])
    if ks != list(range(len(ks))): fails.append(f'completed starts are not a prefix of the locked order: {ks}')
    for k in ks:
        if index['starts'][str(k)]['identity']['val_episode'] != eps[k]: fails.append(f'episode order start {k}')
    return fails


def score_start(zf, w, k_tpe, q_pow, W, b, Wp):
    ev = zf['split'] == 1; n = int(ev.sum()); C = zf['pv16'].shape[-1]
    P = zf['pv16'][ev].astype(np.float64).reshape(n, -1, C); p = zf['pp16'][ev].astype(np.float64).reshape(n, -1)
    Tv = zf['tv16'][ev].astype(np.float64).reshape(n, -1, C); tp = zf['tp16'][ev].astype(np.float64).reshape(n, -1)
    G = zf['gv'].astype(np.float64).reshape(-1, C); gp = zf['gp'].astype(np.float64).reshape(-1)
    Z0 = zf['z0v'].astype(np.float64).reshape(-1, C)
    vis = E.visual_official(P, G); prop = E.proprio_term(p, gp)
    pos, r, rg = prop_pos_term(p.reshape(n, -1, W.shape[0]), gp.reshape(-1, W.shape[0]), W, b, Wp)
    mask = T.fixedmask_indices(G, Z0, 128)
    sc = {'B0': zf['model_cost'][ev], 'B1': E.head(P, G, p, gp, w), 'HPROP10': vis + 10.0 * prop, 'HPROP_STD': z(vis) + z(prop),
          'TPE': T.topk(P, G, p, gp, k_tpe), 'POW': T.powmean(P, G, p, gp, q_pow), 'FIXEDMASK': T.fixedmask(P, G, p, gp, mask),
          'E_B0': E.official(Tv, G, tp, gp), 'E_B1': E.head(Tv, G, tp, gp, w),
          'VELDROP_Z': z(vis) + z(pos), 'POS_ONLY': pos, 'VELDROP_01': vis + 0.1 * pos}
    m = zf['margin'][ev]; succ = zf['success'][ev].astype(bool)
    a, bl, th = components(zf['official_state'][ev], zf['goal_state'])
    assert np.array_equal(succ, m < 1), 'success flag disagrees with margin < 1'
    sel = {nm: pick(v) for nm, v in sc.items()}
    row = {'selected': sel, 'margin': {nm: float(m[j]) for nm, j in sel.items()}, 'success': {nm: bool(succ[j]) for nm, j in sel.items()},
           'agent_err': {nm: float(a[j]) for nm, j in sel.items()}, 'block_err': {nm: float(bl[j]) for nm, j in sel.items()},
           'angle_err_deg': {nm: float(th[j]) for nm, j in sel.items()},
           'block_only_success': {nm: bool(bl[j] < 20 and th[j] < 20) for nm, j in sel.items()},
           'O': float(m.min()), 'O_success': bool(succ.any()), 'random_mean': float(m.mean()), 'random_success_rate': float(succ.mean()),
           'spearman_with_margin': {nm: float(spearmanr(v, m).statistic) for nm, v in sc.items()},
           'mechanism': {'spearman_prop_agent': float(spearmanr(prop, a).statistic), 'spearman_proppos_agent': float(spearmanr(pos, a).statistic),
                         'velocity_share_of_prop': float(1 - pos.mean() / prop.mean()) if prop.mean() > 0 else None,
                         'goal_speed': float(np.linalg.norm(zf['goal_state'][5:7]))}}
    if 'cost64' in zf.files: row['b0_argmin_equals_cost64_argmin'] = bool(sel['B0'] == pick(zf['cost64'][ev]))
    return row


def decide(D, s_new, s_ref, n, prov_ok, yes, no):
    if n < N_MIN or not prov_ok: return 'INCOMPLETE'
    Ds = np.sort(D)
    return yes if (D.mean() <= -DELTA and s_new >= s_ref and Ds[1:].mean() <= 0) else no


def main(argv):
    dry = argv[0] == '--dry-run'
    if dry: argv = argv[1:]
    cohort, outp = Path(argv[0]), Path(argv[1])
    w = np.array(json.load(open(HEAD))['weights']); L = json.load(open(TPE_LOCK)); W, b, Wp = proprio_encoder()
    index = json.loads((cohort / ('r7_index.json' if dry else 'r6_index.json')).read_text())
    fails = [] if dry else provenance(cohort, index)
    rows = []
    for k in sorted(index['starts'], key=int):
        zf = np.load(cohort / f'start_{int(k):03d}.npz'); row = score_start(zf, w, L['k_star'], L['q_star'], W, b, Wp)
        ident = index['starts'][k].get('identity', {}); row['k'] = int(k); row['episode'] = ident.get('val_episode', index['starts'][k].get('traj'))
        rows.append(row); print('start', k, 'scored', flush=True)
    n = len(rows); M = {nm: np.array([r['margin'][nm] for r in rows]) for nm in RULES}
    succ = {nm: int(sum(r['success'][nm] for r in rows)) for nm in RULES}
    D1 = M['B1'] - M['B0']; DV = M['VELDROP_Z'] - M['B0']; prov_ok = len(fails) == 0
    sp_prop = np.array([r['mechanism']['spearman_prop_agent'] for r in rows]); sp_pos = np.array([r['mechanism']['spearman_proppos_agent'] for r in rows])

    def paired(a_, b_):
        g = sum(r['success'][a_] and not r['success'][b_] for r in rows); l = sum(r['success'][b_] and not r['success'][a_] for r in rows)
        return [int(g), int(l)]

    S = {'n': n, 'dry_run': dry, 'provenance_failures': fails,
         'PRIMARY_B1_vs_B0': {'decision': decide(D1, succ['B1'], succ['B0'], n, prov_ok, 'REPLICATED', 'NOT_REPLICATED'),
                              'D': boot(D1), 'improved': int((D1 < 0).sum()), 'worsened': int((D1 > 0).sum()),
                              'D_without_most_favourable': float(np.sort(D1)[1:].mean()) if n > 1 else None,
                              'paired_success_gains_losses': paired('B1', 'B0')},
         'ADDENDUM_VELDROP_Z_vs_B0': {'decision': decide(DV, succ['VELDROP_Z'], succ['B0'], n, prov_ok, 'SUPPORTED', 'NOT_SUPPORTED'),
                                     'D': boot(DV), 'improved': int((DV < 0).sum()), 'worsened': int((DV > 0).sum()),
                                     'D_without_most_favourable': float(np.sort(DV)[1:].mean()) if n > 1 else None,
                                     'paired_success_gains_losses': paired('VELDROP_Z', 'B0')},
         'MECHANISM': {'decision': ('REPLICATED' if (sp_pos.mean() >= 0.80 and sp_prop.mean() <= 0.60) else 'NOT_REPLICATED') if n >= N_MIN and prov_ok else 'INCOMPLETE',
                       'mean_spearman_prop_agent': float(sp_prop.mean()), 'mean_spearman_proppos_agent': float(sp_pos.mean()),
                       'mean_velocity_share_of_prop': float(np.mean([r['mechanism']['velocity_share_of_prop'] for r in rows]))},
         'success_counts': succ, 'block_only_success_counts': {nm: int(sum(r['block_only_success'][nm] for r in rows)) for nm in RULES},
         'mean_selected_margin': {nm: boot(M[nm]) for nm in RULES},
         'mean_component_errors': {nm: {'agent': float(np.mean([r['agent_err'][nm] for r in rows])), 'block': float(np.mean([r['block_err'][nm] for r in rows])),
                                        'angle_deg': float(np.mean([r['angle_err_deg'][nm] for r in rows]))} for nm in RULES},
         'O': {'mean': float(np.mean([r['O'] for r in rows])), 'successes': int(sum(r['O_success'] for r in rows))},
         'random_plan': {'mean': float(np.mean([r['random_mean'] for r in rows])), 'success_rate': float(np.mean([r['random_success_rate'] for r in rows]))},
         'descriptive_contrasts': {'VELDROP_Z_minus_B1': boot(M['VELDROP_Z'] - M['B1']), 'E_B1_minus_E_B0': boot(M['E_B1'] - M['E_B0']),
                                   'HPROP_STD_minus_B0': boot(M['HPROP_STD'] - M['B0']), 'HPROP10_minus_B0': boot(M['HPROP10'] - M['B0']),
                                   'TPE_minus_B0': boot(M['TPE'] - M['B0']), 'POS_ONLY_minus_B0': boot(M['POS_ONLY'] - M['B0']),
                                   'paired_success_VELDROP_Z_vs_B1': paired('VELDROP_Z', 'B1')},
         'within_bank_spearman_mean': {nm: float(np.mean([r['spearman_with_margin'][nm] for r in rows])) for nm in RULES},
         'checks': {'b0_argmin_equals_cost64_argmin_all': all(r.get('b0_argmin_equals_cost64_argmin', True) for r in rows)}}
    json.dump({'summary': S, 'per_start': rows}, open(outp, 'w'), indent=1)
    print(json.dumps({k: S[k] for k in ['n', 'provenance_failures', 'PRIMARY_B1_vs_B0', 'ADDENDUM_VELDROP_Z_vs_B0', 'MECHANISM', 'success_counts']}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
