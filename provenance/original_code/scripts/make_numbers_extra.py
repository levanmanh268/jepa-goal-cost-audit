"""Extra LaTeX macros for the ICAI-final manuscript revision (2026-10-28), generated from data files only.
Inputs: r11/r8_proprio_decomposition.json, r11/r10_proprio_decomposition.json, r11/r8_proprio_diag.npz,
r11/r10_proprio_diag.npz, r8/r8_components.npz, r10/r10_components.npz, checkpoints (proprio encoders),
r8/r9_optimism.json, r8/r9_analysis.json, r11/r11_analysis.json, r11/out/r11_inloop.json.
Changes from the R11 snapshot: adds per-bank Spearman range and pooled velocity share (Dec*SpVelMin/Max, Dec*VelPooled)
and the range of the adaptive weight over the full-standardised search populations (RElAlphaFullMin/Max, RElNPop).
Usage: python3 -I make_numbers_extra.py numbers_extra.tex
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch

WM = Path('/home/claude/wm'); out = []; PEND = '\\textbf{[pending]}'


def fmt(x, d=3):
    return f'{x:.{d}f}'.replace('-', '\\ensuremath{-}')


def mac(n, v):
    out.append(f'\\newcommand{{\\{n}}}{{{v}}}')


def pct(x, d=1):
    return f'{100 * x:.{d}f}\\%'


def binom2(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fm(c):
    c = np.asarray(c, float); return int(np.flatnonzero(c <= c.min() + 1e-12 * max(1.0, abs(c.min())))[0])


def z(a):
    return (a - a.mean()) / a.std()


# five-term decomposition
for tag, f in [('J', 'r11/r8_proprio_decomposition.json'), ('Dn', 'r11/r10_proprio_decomposition.json')]:
    S = json.load(open(WM / f))['mean_over_starts']
    mac(f'Dec{tag}Pos', pct(S['share_pos'])); mac(f'Dec{tag}Vel', pct(S['share_vel'])); mac(f'Dec{tag}Cross', pct(S['share_cross']))
    mac(f'Dec{tag}ResTok', pct(S['share_resid'] + S['share_token_var'], 2)); mac(f'Dec{tag}NetRemoved', pct(S['net_removed_fraction']))
    mac(f'Dec{tag}SpVel', fmt(S['spearman_vel'], 2)); mac(f'Dec{tag}SpPos', fmt(S['spearman_pos'], 3)); mac(f'Dec{tag}SpFull', fmt(S['spearman_full'], 3))
    mac(f'Dec{tag}IdErr', f"{json.load(open(WM / f))['max_rel_identity_error']:.0e}".replace('e-', '\\times10^{-').replace('e+', '\\times10^{') + '}')
    rv = [p['spearman_vel'] for p in json.load(open(WM / f))['per_start']]
    mac(f'Dec{tag}SpVelMin', fmt(min(rv), 3)); mac(f'Dec{tag}SpVelMax', fmt(max(rv), 3))
    Zd = np.load(WM / f.replace('_decomposition.json', '_diag.npz'))
    mac(f'Dec{tag}VelPooled', pct(float(Zd['e_vel'].sum() / Zd['e_full'].sum())))
# encoder geometry
for tag, ck in [('J', 'jepa'), ('Dn', 'dino')]:
    W = torch.load(WM / f'runtime/checkpoints/{ck}_wm_pusht.pth.tar', map_location='cpu', weights_only=False)['proprio_encoder']['module.patch_embed.weight'][:, :, 0].double().numpy()
    Qp, _ = np.linalg.qr(W[:, :2]); Qv, _ = np.linalg.qr(W[:, 2:]); smax = np.linalg.svd(Qp.T @ Qv, compute_uv=False).max()
    ang = float(np.degrees(np.arccos(min(1.0, smax)))); A = W[:, :2] @ np.hstack([np.eye(2), np.zeros((2, 2))]) @ np.linalg.pinv(W)
    mac(f'Cond{tag}', fmt(np.linalg.cond(W), 1 if tag == 'Dn' else 2)); mac(f'Ang{tag}', fmt(ang, 1)); mac(f'Gain{tag}', fmt(np.linalg.norm(A, 2), 2))
# R8 post hoc factorial on the fixed banks
for tag, f in [('Ra', 'r8/r8_components.npz'), ('Rt', 'r10/r10_components.npz')]:
    Z = np.load(WM / f); n = Z['vis'].shape[0]
    rules = {'C00': lambda k: Z['model_cost'][k], 'C10': lambda k: Z['vis'][k] + 0.1 * Z['prop_pos'][k],
             'C01': lambda k: z(Z['vis'][k]) + z(Z['prop_full'][k]), 'C11': lambda k: z(Z['vis'][k]) + z(Z['prop_pos'][k]),
             'SA': lambda k: Z['vis'][k] + (Z['vis'][k].std() / Z['prop_full'][k].std()) * Z['prop_pos'][k]}
    MS = {}; LET = {'C00': 'Czz', 'C10': 'Coz', 'C01': 'Czo', 'C11': 'Coo', 'SA': 'SA'}  # TeX macro names cannot contain digits
    for c, r in rules.items():
        j = [fm(r(k)) for k in range(n)]; m = np.array([Z['margin'][k][j[k]] for k in range(n)]); s = np.array([bool(Z['success'][k][j[k]]) for k in range(n)])
        MS[c] = (m, s); mac(f'Fac{tag}{LET[c]}', fmt(m.mean())); mac(f'Fac{tag}{LET[c]}Succ', int(s.sum()))
    (m11, s11), (m01, s01) = MS['C11'], MS['C01']
    g, l = int((s11 & ~s01).sum()), int((s01 & ~s11).sum())
    mac(f'Fac{tag}DropD', fmt((m11 - m01).mean())); mac(f'Fac{tag}DropGain', g); mac(f'Fac{tag}DropLoss', l); mac(f'Fac{tag}DropMcN', fmt(binom2(g, l), 4))
# statistics on locked results (descriptive)
r8 = json.load(open(WM / 'r8/r8_analysis.json'))['summary']; r9 = json.load(open(WM / 'r8/r9_analysis.json'))['summary']; r10 = json.load(open(WM / 'r10/r10_analysis.json'))['summary']
gA = r8['ADDENDUM_VELDROP_Z_vs_B0']['paired_success_gains_losses']; mac('McNREight', fmt(binom2(*gA), 3))
gP = r9['R9_PRIMARY_veldrop_z_vs_b0']; mac('McNRNine', fmt(binom2(*gP['paired_success_gains_losses']), 3)); mac('SignRNine', fmt(binom2(gP['improved'], gP['worsened']), 4))
gB = r9['R9_SECONDARY_b1_vs_b0']; mac('McNRNineBone', fmt(binom2(*gB['paired_success_gains_losses']), 3)); mac('RNineBoneGainN', gB['paired_success_gains_losses'][0]); mac('RNineBoneLossN', gB['paired_success_gains_losses'][1])
gT = r10['PRIMARY_VELDROP_Z_vs_B0']; mac('SignRTen', fmt(binom2(gT['improved'], gT['worsened']), 2)); mac('BootZeroRTen', fmt((5 / 8) ** 8, 4))
# B1 own predicted score: B1-CEM plan vs B0 plan
O = json.load(open(WM / 'r8/r9_optimism.json'))['rows']
pb = {int(r['key'].split(':')[0]): r['pred']['b1'] for r in O if r['arm'] == 'b0'}; p1 = {int(r['key'].split(':')[0]): r['pred']['b1'] for r in O if r['arm'] == 'b1'}
ks = sorted(set(pb) & set(p1))
mac('BoneOwnBone', fmt(np.mean([p1[k] for k in ks]))); mac('BoneOwnBzero', fmt(np.mean([pb[k] for k in ks]))); mac('BoneOwnWorse', int(sum(p1[k] > pb[k] for k in ks))); mac('BoneOwnN', len(ks))
# R11
p = WM / 'r11/r11_analysis.json'
names = ['RElN', 'RElDecision', 'RElD', 'RElDLo', 'RElDHi', 'RElImp', 'RElWor', 'RElUnch', 'RElDexcl', 'RElGain', 'RElLoss', 'RElMcN',
         'RElCzz', 'RElCzzSucc', 'RElCoz', 'RElCozSucc', 'RElCzo', 'RElCzoSucc', 'RElCoo', 'RElCooSucc', 'RElSecD', 'RElSecDLo', 'RElSecDHi',
         'RElInter', 'RElInterLo', 'RElInterHi', 'RElScaleFull', 'RElScaleFullLo', 'RElScaleFullHi', 'RElScalePos', 'RElAlphaFull', 'RElAlphaPos',
         'RElCzzAgent', 'RElCzoAgent', 'RElCozAgent', 'RElCooAgent', 'RElCzzBO', 'RElCzoBO', 'RElCozBO', 'RElCooBO', 'RElMedian', 'RElSign',
         'RElScaleFullGain', 'RElScaleFullLoss', 'RElScaleFullImp', 'RElScaleFullWor', 'RElSecGain', 'RElSecLoss', 'RElCzzBlock', 'RElCooBlock', 'RElEffFold',
         'RElVOnePosRatio', 'RElVOnePosRatioMin', 'RElAlphaFullMin', 'RElAlphaFullMax', 'RElNPop']
if p.exists():
    U = json.load(open(p))['summary']; P = U['PRIMARY_C11_minus_C01']; C = U['cells']
    mac('RElN', U['n']); mac('RElDecision', P['decision'].replace('_', '\\_')); mac('RElD', fmt(P['mean_ci'][0])); mac('RElDLo', fmt(P['mean_ci'][1])); mac('RElDHi', fmt(P['mean_ci'][2]))
    mac('RElImp', P['improved']); mac('RElWor', P['worsened']); mac('RElUnch', P['unchanged']); mac('RElDexcl', fmt(P['without_most_favourable']))
    mac('RElGain', P['success_gains_losses'][0]); mac('RElLoss', P['success_gains_losses'][1]); mac('RElMcN', fmt(P['mcnemar_exact_two_sided'], 3))
    for c, nm in [('C00', 'Czz'), ('C10', 'Coz'), ('C01', 'Czo'), ('C11', 'Coo')]:
        mac(f'REl{nm}', fmt(C[c]['mean_margin_ci'][0])); mac(f'REl{nm}Succ', C[c]['successes'])
        mac(f'REl{nm}Agent', fmt(C[c]['component_errors']['agent_err'], 1)); mac(f'REl{nm}BO', C[c]['block_only_successes'])
    X = U['SECONDARY_C10_minus_C00']['mean_ci']; mac('RElSecD', fmt(X[0])); mac('RElSecDLo', fmt(X[1])); mac('RElSecDHi', fmt(X[2]))
    X = U['interaction']['mean_ci']; mac('RElInter', fmt(X[0])); mac('RElInterLo', fmt(X[1])); mac('RElInterHi', fmt(X[2]))
    X = U['scale_effect_full_C01_minus_C00']['mean_ci']; mac('RElScaleFull', fmt(X[0])); mac('RElScaleFullLo', fmt(X[1])); mac('RElScaleFullHi', fmt(X[2]))
    mac('RElScalePos', fmt(U['scale_effect_pos_C11_minus_C10']['mean_ci'][0]))
    mac('RElMedian', fmt(P['median'])); mac('RElSign', fmt(P['sign_test_two_sided'], 2))
    sf = U['scale_effect_full_C01_minus_C00']; mac('RElScaleFullGain', sf['success_gains_losses'][0]); mac('RElScaleFullLoss', sf['success_gains_losses'][1])
    mac('RElScaleFullImp', sf['improved']); mac('RElScaleFullWor', sf['worsened'])
    sd = U['SECONDARY_C10_minus_C00']; mac('RElSecGain', sd['success_gains_losses'][0]); mac('RElSecLoss', sd['success_gains_losses'][1])
    mac('RElCzzBlock', fmt(C['C00']['component_errors']['block_err'], 1)); mac('RElCooBlock', fmt(C['C11']['component_errors']['block_err'], 1))
    mac('RElEffFold', fmt(U['mean_inloop_alpha']['hprop_std']['alpha_z_full'] / 0.1, 0))  # effective proprio weight / official 0.1
    mac('RElAlphaFull', fmt(U['mean_inloop_alpha']['hprop_std']['alpha_z_full'], 2)); mac('RElAlphaPos', fmt(U['mean_inloop_alpha']['hprop_std']['alpha_z_pos'], 2))
    # C10 arm (vis + 0.1 pos): spread of the visual term over the spread of the weighted position term, per population.
    mac('RElVOnePosRatio', fmt(U['mean_inloop_alpha']['veldrop_01']['alpha_z_pos'] / 0.1, 0))
    raw = json.load(open(WM / 'r11/out/r11_inloop.json'))['runs']
    mac('RElVOnePosRatioMin', fmt(min(e['alpha_z_pos'] for r in raw.values() if r['arm'] == 'veldrop_01' for e in r['history']) / 0.1, 1))
    af = [e['alpha_z_full'] for r in raw.values() if r['arm'] == 'hprop_std' for e in r['history']]
    mac('RElAlphaFullMin', fmt(min(af), 3)); mac('RElAlphaFullMax', fmt(max(af), 3)); mac('RElNPop', len(af))
else:
    for nm in names: mac(nm, PEND)
# fp32 vs fp16 audit
for tag, f in [('Dn', 'r11/fp32_dino_r10.json'), ('J', 'r11/fp32_jepa_r8_first4.json')]:
    q = WM / f
    if q.exists():
        S = json.load(open(q))['summary']
        mac(f'Fp{tag}N', S['n']); mac(f'Fp{tag}Changes', sum(S['selection_changes'].values())); mac(f'Fp{tag}MaxRel', pct(S['max_rel_pos_energy_diff']))
        mac(f'Fp{tag}Exact', 'yes' if (S['all_roundtrips_equal'] and S['max_cost_diff'] == 0.0) else 'no')
    else:
        for nm in ['N', 'Changes', 'MaxRel', 'Exact']: mac(f'Fp{tag}{nm}', PEND)
Path(sys.argv[1]).write_text('% Auto-generated by make_numbers_extra.py. Do not edit by hand.\n' + '\n'.join(out) + '\n')
print(len(out), 'macros;', 'r11' if p.exists() else 'no r11')
