"""Generate LaTeX number macros for the R8 working draft from JSON outputs (Claude, 2026-10-27).
Inputs: dev_numbers.json (development, 26 starts), r8_analysis.json (locked R8), and, if present, ../r8/r9_analysis.json,
../r8/r9_optimism.json and ../r10/r10_analysis.json. Missing inputs give a visible [pending] macro.
Usage: python3 -I make_numbers_r8.py numbers_r8.tex
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys
from pathlib import Path

HERE = Path(str(_WM_ROOT / 'r8'))
out = []


def fmt(x, d=3):
    s = f'{x:.{d}f}'
    return s.replace('-', '\\ensuremath{-}')


def mac(name, val):
    out.append(f'\\newcommand{{\\{name}}}{{{val}}}')


def load(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


dev = load(HERE / 'dev_numbers.json'); r8 = load(HERE / 'r8_analysis.json')
r9 = load(HERE / 'r9_analysis.json'); r9o = load(HERE / 'r9_optimism.json'); r10 = load(str(_WM_ROOT / 'r10/r10_analysis.json'))
PEND = '\\textbf{[pending]}'

# development (26 starts)
R = dev['rules']
mac('DevN', dev['n'])
for key, nm in [('official', 'Off'), ('E_official', 'EOff'), ('prop_pred', 'PropPred'), ('prop_true', 'PropTrue'), ('agent_oracle', 'AgentOr'),
                ('agent_block_oracle', 'ABOr'), ('O', 'Oracle'), ('hprop_std', 'Hstd'), ('veldrop_z', 'VZ'), ('veldrop_01', 'VOne'), ('pos_only', 'PosOnly'),
                ('vel_only', 'VelOnly'), ('zero_goal_velocity_z', 'Vzero')]:
    mac(f'Dev{nm}', fmt(R[key]['mean'])); mac(f'Dev{nm}Succ', R[key]['succ'])
    mac(f'Dev{nm}Agent', fmt(R[key]['agent'], 1)); mac(f'Dev{nm}Block', fmt(R[key]['block'], 1)); mac(f'Dev{nm}Angle', fmt(R[key]['angle'], 1))
d = dev['veldrop_z_minus_official']
mac('DevVZD', fmt(d['mean'])); mac('DevVZImp', d['improved']); mac('DevVZWor', d['worsened']); mac('DevVZDexcl', fmt(d['without_most_favourable']))
m = dev['mechanism']
mac('DevSharePos', fmt(m['share_pos'], 2)); mac('DevShareVel', fmt(m['share_vel'], 2)); mac('DevShareCross', fmt(m['share_cross'], 2))
mac('DevVarRatio', fmt(m['var_ratio_pos_median'], 2)); mac('DevCorrPosProp', fmt(m['corr_pos_prop_mean'], 2))
mac('DevSpProp', fmt(m['spearman_prop_agent'], 2)); mac('DevSpPos', fmt(m['spearman_pos_agent'], 3)); mac('DevSpVel', fmt(m['spearman_vel_agent'], 2))
mac('DevSpPredTrue', fmt(m['spearman_pred_true_prop'], 3)); mac('DevResid', fmt(m['max_residual_outside_range'], 3))
mac('DevGoalSpeed', fmt(m['goal_speed_median'], 0)); mac('DevCond', fmt(m['pos_form_condition_number_pixels'], 2))
ba = dev['bank_success_availability']
mac('DevBanksWithSucc', ba['banks_with_success']); mac('DevVZHits', ba['veldrop_hits']); mac('DevOffHits', ba['official_hits'])
sl = dev['sensitivity_lambda']
mac('DevSensMinMean', fmt(min(v['mean'] for k, v in sl.items() if float(k) >= 0.75)))
mac('DevSensMaxMean', fmt(max(v['mean'] for k, v in sl.items() if float(k) >= 0.75)))

# R8 (locked)
S = r8['summary']; ms = S['mean_selected_margin']; sc = S['success_counts']
mac('REightN', S['n'])
for key, nm in [('B0', 'Bzero'), ('B1', 'Bone'), ('VELDROP_Z', 'VZ'), ('HPROP10', 'Hten'), ('HPROP_STD', 'Hstd'), ('TPE', 'Tpe'), ('POW', 'Pow'),
                ('FIXEDMASK', 'Fix'), ('POS_ONLY', 'PosOnly'), ('VELDROP_01', 'VOne'), ('E_B0', 'EBzero'), ('E_B1', 'EBone')]:
    mac(f'REight{nm}', fmt(ms[key][0])); mac(f'REight{nm}Lo', fmt(ms[key][1])); mac(f'REight{nm}Hi', fmt(ms[key][2])); mac(f'REight{nm}Succ', sc[key])
    mac(f'REight{nm}Sp', fmt(S['within_bank_spearman_mean'][key], 3))
mac('REightOracle', fmt(S['O']['mean'])); mac('REightOracleSucc', S['O']['successes']); mac('REightRandom', fmt(S['random_plan']['mean']))
A = S['ADDENDUM_VELDROP_Z_vs_B0']; P = S['PRIMARY_B1_vs_B0']
mac('REightVZDecision', A['decision'].replace('_', '\\_')); mac('REightVZD', fmt(A['D'][0])); mac('REightVZDLo', fmt(A['D'][1])); mac('REightVZDHi', fmt(A['D'][2]))
mac('REightVZImp', A['improved']); mac('REightVZWor', A['worsened']); mac('REightVZDexcl', fmt(A['D_without_most_favourable']))
mac('REightVZGain', A['paired_success_gains_losses'][0]); mac('REightVZLoss', A['paired_success_gains_losses'][1])
mac('REightBoneDecision', P['decision'].replace('_', '\\_')); mac('REightBoneD', fmt(P['D'][0])); mac('REightBoneDLo', fmt(P['D'][1])); mac('REightBoneDHi', fmt(P['D'][2]))
mac('REightBoneImp', P['improved']); mac('REightBoneWor', P['worsened']); mac('REightBoneDexcl', fmt(P['D_without_most_favourable']))
M = S['MECHANISM']
mac('REightMechDecision', M['decision'].replace('_', '\\_')); mac('REightSpProp', fmt(M['mean_spearman_prop_agent'], 2)); mac('REightSpPos', fmt(M['mean_spearman_proppos_agent'], 3))
mac('REightShareVel', fmt(M['mean_velocity_share_of_prop'], 2))
c = S['descriptive_contrasts']['VELDROP_Z_minus_B1']
mac('REightVZminusBone', fmt(c[0])); mac('REightVZminusBoneLo', fmt(c[1])); mac('REightVZminusBoneHi', fmt(c[2]))
mac('REightVZvsBoneGain', S['descriptive_contrasts']['paired_success_VELDROP_Z_vs_B1'][0]); mac('REightVZvsBoneLoss', S['descriptive_contrasts']['paired_success_VELDROP_Z_vs_B1'][1])

# R7 (locked TPE test on R6 test banks; PAE mechanism probe)
t7 = load(str(_WM_ROOT / 'r7/r7_tpe_test.json'))['summary']; l7 = load(str(_WM_ROOT / 'r7/landscape_train_goals.json'))
mac('RSevenTpeD', fmt(t7['primary_D_TPE_minus_B0'][0])); mac('RSevenTpeDLo', fmt(t7['primary_D_TPE_minus_B0'][1])); mac('RSevenTpeDHi', fmt(t7['primary_D_TPE_minus_B0'][2]))
mac('RSevenTpeImp', t7['D_improved']); mac('RSevenTpeWor', t7['D_worsened'])
mac('RSevenProbeN', l7['n_states']); mac('RSevenOffSp', fmt(l7['per_energy']['official']['spearman_with_margin_mean'], 3))
mac('RSevenPaeSp', fmt(l7['per_energy'][l7['H_M2_best_pae']]['spearman_with_margin_mean'], 3)); mac('RSevenOffRot', fmt(l7['H_M1_rotation_spearman_official'], 3))

# R9 (locked, in-loop)
if r9:
    T = r9['summary']; mac('RNineN', T['n'])
    for key, nm in [('R9_PRIMARY_veldrop_z_vs_b0', 'VZ'), ('R9_SECONDARY_b1_vs_b0', 'Bone')]:
        X = T[key]; mac(f'RNine{nm}Decision', X['decision'].replace('_', '\\_')); mac(f'RNine{nm}D', fmt(X['D'][0])); mac(f'RNine{nm}DLo', fmt(X['D'][1]))
        mac(f'RNine{nm}DHi', fmt(X['D'][2])); mac(f'RNine{nm}Imp', X['improved']); mac(f'RNine{nm}Wor', X['worsened'])
        mac(f'RNine{nm}Dexcl', fmt(X['D_without_most_favourable'])); mac(f'RNine{nm}Gain', X['paired_success_gains_losses'][0]); mac(f'RNine{nm}Loss', X['paired_success_gains_losses'][1])
    for a, nm in [('b0', 'Bzero'), ('veldrop_z', 'VZ'), ('b1', 'Bone')]:
        mac(f'RNine{nm}Margin', fmt(T['mean_executed_margin'][a][0])); mac(f'RNine{nm}Succ', T['success_counts'][a])
        mac(f'RNine{nm}BlockOnly', T['block_only_success_counts'][a])
    mac('RNineVZminusBone', fmt(T['veldrop_z_minus_b1'][0])); mac('RNineVZminusBoneLo', fmt(T['veldrop_z_minus_b1'][1])); mac('RNineVZminusBoneHi', fmt(T['veldrop_z_minus_b1'][2]))
    for a, nm in [('b0', 'Bzero'), ('veldrop_z', 'VZ'), ('b1', 'Bone')]:
        ce = T['mean_component_errors'][a]
        mac(f'RNine{nm}Agent', fmt(ce['agent_err'], 1)); mac(f'RNine{nm}Block', fmt(ce['block_err'], 1)); mac(f'RNine{nm}Angle', fmt(ce['angle_err_deg'], 1))
        mac(f'RNine{nm}MarginLo', fmt(T['mean_executed_margin'][a][1])); mac(f'RNine{nm}MarginHi', fmt(T['mean_executed_margin'][a][2]))
    X = T['b1_exploitation']
    mac('RNineEliteRawVZ', fmt(X['elite_raw_cost_last_iter']['veldrop_z'])); mac('RNineEliteRawBone', fmt(X['elite_raw_cost_last_iter']['b1']))
    mac('RNineFracNegBone', fmt(100 * X['frac_negative_b1_last_iter_mean'], 1))
else:
    for nm in ['RNineN', 'RNineVZDecision', 'RNineVZD', 'RNineVZDLo', 'RNineVZDHi', 'RNineVZImp', 'RNineVZWor', 'RNineVZDexcl', 'RNineVZGain', 'RNineVZLoss',
               'RNineBoneDecision', 'RNineBoneD', 'RNineBoneDLo', 'RNineBoneDHi', 'RNineBoneImp', 'RNineBoneWor', 'RNineBoneDexcl', 'RNineBoneGain', 'RNineBoneLoss',
               'RNineBzeroMargin', 'RNineBzeroSucc', 'RNineBzeroBlockOnly', 'RNineVZMargin', 'RNineVZSucc', 'RNineVZBlockOnly', 'RNineBoneMargin', 'RNineBoneSucc',
               'RNineBoneBlockOnly', 'RNineVZminusBone', 'RNineVZminusBoneLo', 'RNineVZminusBoneHi']:
        mac(nm, PEND)
if r9o:
    for a, nm in [('veldrop_z', 'VZ'), ('b1', 'Bone'), ('b0', 'Bzero')]:
        if a in r9o['summary']:
            mac(f'OptRaw{nm}', fmt(r9o['summary'][a]['raw']['median_true_over_pred'], 2)); mac(f'OptBone{nm}', fmt(r9o['summary'][a]['b1']['median_true_over_pred'], 2))
            mac(f'OptPos{nm}', fmt(r9o['summary'][a]['prop_pos']['median_true_over_pred'], 2))
# R10 (locked, DINO-WM)
if r10:
    U = r10['summary']; mac('RTenN', U['n']); X = U['PRIMARY_VELDROP_Z_vs_B0']
    mac('RTenVZDecision', X['decision'].replace('_', '\\_')); mac('RTenVZD', fmt(X['D'][0])); mac('RTenVZDLo', fmt(X['D'][1])); mac('RTenVZDHi', fmt(X['D'][2]))
    mac('RTenVZImp', X['improved']); mac('RTenVZWor', X['worsened']); mac('RTenVZDexcl', fmt(X['D_without_most_favourable']))
    for key, nm in [('B0', 'Bzero'), ('VELDROP_Z', 'VZ'), ('HPROP_STD', 'Hstd'), ('HPROP10', 'Hten'), ('B1', 'Bone')]:
        mac(f'RTen{nm}', fmt(U['mean_selected_margin'][key][0])); mac(f'RTen{nm}Succ', U['success_counts'][key])
    mac('RTenOracle', fmt(U['O']['mean'])); mac('RTenOracleSucc', U['O']['successes'])
    M2 = U['MECHANISM']; mac('RTenMechDecision', M2['decision'].replace('_', '\\_')); mac('RTenSpProp', fmt(M2['mean_spearman_prop_agent'], 2))
    mac('RTenSpPos', fmt(M2['mean_spearman_proppos_agent'], 3)); mac('RTenShareVel', fmt(M2['mean_velocity_share_of_prop'], 3))
    mac('RTenDexcl', fmt(X['D_without_most_favourable'])); mac('RTenUnchanged', U['n'] - X['improved'] - X['worsened'])
    for key, nm in [('POS_ONLY', 'PosOnly'), ('VELDROP_01', 'VOne')]:
        mac(f'RTen{nm}', fmt(U['mean_selected_margin'][key][0]))
    mac('RTenRandom', fmt(U['random_plan']['mean']))
    Y = U['SECONDARY_B1_vs_B0_crossmodel_descriptive']; mac('RTenBoneD', fmt(Y['D'][0]))
    F = load(str(_WM_ROOT / 'r10/r10_mechanism_facts.json'))
    if F:
        e = F['encoder_column_norms']
        mac('EncJepaRatio', fmt(e['jepa']['velocity_to_position_norm_ratio'], 2)); mac('EncDinoRatio', fmt(e['dino']['velocity_to_position_norm_ratio'], 3))
        mac('EncDinoVx', fmt(e['dino']['vx'], 3)); mac('EncDinoVy', fmt(e['dino']['vy'], 3)); mac('EncDinoX', fmt(e['dino']['x'], 2)); mac('EncDinoY', fmt(e['dino']['y'], 2))
        mac('EncJepaVx', fmt(e['jepa']['vx'], 2)); mac('EncJepaVy', fmt(e['jepa']['vy'], 2)); mac('EncJepaX', fmt(e['jepa']['x'], 2)); mac('EncJepaY', fmt(e['jepa']['y'], 2))
        s = F['veldrop_vs_hprop_std_same_selection']
        mac('SameSelJepa', s['jepa_r8']['same']); mac('SameSelJepaN', s['jepa_r8']['n']); mac('SameSelDino', s['dino_r10']['same']); mac('SameSelDinoN', s['dino_r10']['n'])
else:
    for nm in ['RTenN', 'RTenVZDecision', 'RTenVZD', 'RTenVZDLo', 'RTenVZDHi', 'RTenVZImp', 'RTenVZWor', 'RTenVZDexcl', 'RTenBzero', 'RTenBzeroSucc', 'RTenVZ',
               'RTenVZSucc', 'RTenHstd', 'RTenHstdSucc', 'RTenHten', 'RTenHtenSucc', 'RTenBone', 'RTenBoneSucc', 'RTenOracle', 'RTenOracleSucc', 'RTenMechDecision',
               'RTenSpProp', 'RTenSpPos', 'RTenShareVel']:
        mac(nm, PEND)

Path(sys.argv[1]).write_text('% Auto-generated by make_numbers_r8.py from JSON outputs. Do not edit by hand.\n' + '\n'.join(out) + '\n')
print(len(out), 'macros;', 'r9' if r9 else 'no r9', 'r10' if r10 else 'no r10')
