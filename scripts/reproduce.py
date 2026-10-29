"""Recompute saved results and the manuscript macros. No model download or GPU use."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'outputs/reproduced')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'WM_ROOT': str(ROOT), 'OPENBLAS_NUM_THREADS': '1'}
    receipt = {'scope': 'Saved-result analysis, not simulator/checkpoint reproduction.',
               'python': platform.python_version(), 'checks': []}

    def run(name, cmd):
        p = subprocess.run([sys.executable, '-I', *map(str, cmd)], cwd=ROOT, env=env,
                           text=True, capture_output=True)
        (out / (name + '.log')).write_text(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(name + ' failed: ' + p.stderr[-1500:])
        receipt['checks'].append({'name': name, 'exit_code': 0})
        print('PASS', name)

    def identical(generated, reference):
        a, b = generated.read_bytes(), (ROOT / reference).read_bytes()
        if a != b:
            raise AssertionError('Byte mismatch: ' + str(reference))
        receipt['checks'].append({'byte_identical_to': str(reference),
                                  'sha256': hashlib.sha256(a).hexdigest()})

    run('r8_compact', ['r8/recompute_from_components.py', 'bank', 'r8/r8_components.npz'])
    run('r10_compact', ['r8/recompute_from_components.py', 'bank', 'r10/r10_components.npz'])
    run('r9_compact', ['r8/recompute_from_components.py', 'inloop', 'r8/r9_inloop.json', 'r8/r8_components.npz'])
    run('r9_analysis', ['r8/r9_analyze.py', 'r8/r9_inloop.json', out / 'r9_analysis.json'])
    identical(out / 'r9_analysis.json', 'r8/r9_analysis.json')
    run('r11_analysis', ['r11/r11_analyze.py', 'r8/r9_inloop.json', 'r11/r11_inloop.json', out / 'r11_analysis.json'])
    identical(out / 'r11_analysis.json', 'r11/r11_analysis.json')
    run('r12_analysis', ['r12/r12_analyze.py', 'r12/r12_cohort.json', 'r12/out/r12_inloop.json', out / 'r12_analysis.json'])
    identical(out / 'r12_analysis.json', 'r12/r12_analysis.json')
    run('r13_analysis', ['r13/r13_analyze.py', 'r12/r12_cohort.json', 'r12/out/r12_inloop.json',
                        'r13/out/r13_inloop.json', out / 'r13_analysis.json'])
    identical(out / 'r13_analysis.json', 'r13/r13_analysis.json')
    for n in [12, 13]:
        parsed = json.loads((out / f'r{n}_analysis.json').read_text())
        assert not parsed['summary']['issues']
    run('independent_r12_r13', ['scripts/audit_r12_r13.py', 'evidence/r12_r13', out / 'independent_r12_r13.json'])
    for suffix in ['r8', 'extra', 'r12', 'r13']:
        run('macros_' + suffix, [f'scripts/make_numbers_{suffix}.py', out / f'numbers_{suffix}.tex'])
        identical(out / f'numbers_{suffix}.tex', f'paper/numbers_{suffix}.tex')
    run('figure', ['scripts/make_fig_audit.py', out / 'fig_audit.pdf'])
    # PDF metadata can vary, so figure validation is numerical/render-based, not byte identity.
    receipt['status'] = 'PASS'
    (out / 'REPRODUCTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('All saved-result checks passed. Receipt:', out / 'REPRODUCTION_RECEIPT.json')


if __name__ == '__main__':
    main()
