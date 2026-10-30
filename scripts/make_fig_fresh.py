"""Figure 2: per-start executed margin on the fresh cohort, official cost versus VELDROP-z (CEM objective).
Reads only saved records: r12/out/r12_inloop.json (runs) and r12/r12_analysis.json (locked analysis, used as a check).
Usage: python3 -I make_fig_fresh.py fig_fresh.pdf
"""

# Resolve external assets relative to this checkout unless WM_ROOT is set.
import os as _os
from pathlib import Path as _Path
_WM_ROOT = _Path(_os.environ.get("WM_ROOT", str(_Path(__file__).resolve().parents[1]))).resolve()
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
CATS = [('success gained', '#2a78d6', 'o'), ('success under both', '#eb6834', '^'), ('success under neither', '#1baf7a', 's')]


def main(out):
    R = json.loads((_WM_ROOT / 'r12/out/r12_inloop.json').read_text())['runs']
    n = len({k.split(':')[0] for k in R})
    x = np.array([R[f'{k}:raw']['margin'] for k in range(n)]); y = np.array([R[f'{k}:veldrop_z']['margin'] for k in range(n)])
    sx = np.array([R[f'{k}:raw']['success'] for k in range(n)]); sy = np.array([R[f'{k}:veldrop_z']['success'] for k in range(n)])
    assert not np.any(sx & ~sy), 'a lost success would need its own category'
    groups = [sy & ~sx, sx & sy, ~sx & ~sy]
    # Consistency with the locked R12 analysis (Q2 = VELDROP-z versus the official cost).
    Q2 = json.loads((_WM_ROOT / 'r12/r12_analysis.json').read_text())['summary']['Q2']
    assert n == Q2['n'] and int(groups[0].sum()) == Q2['success_gains_losses'][0]
    assert [int(sy.sum()), int(sx.sum())] == Q2['successes_new_ref'] and int((y < x).sum()) == Q2['improved']
    plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6,
                         'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    lo, hi = 0.05, 20
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
    ax.axhline(1, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0); ax.axvline(1, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for (label, color, marker), g in zip(CATS, groups):
        ax.scatter(x[g], y[g], s=22, c=color, marker=marker, edgecolors='white', linewidths=0.6, zorder=3,
                   label=f'{label} ({int(g.sum())})')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('executed margin, official cost'); ax.set_ylabel('executed margin, VELDROP-z')
    ax.text(14, 0.07, 'below the diagonal:\nVELDROP-z better', ha='right', va='bottom', color=MUTED, fontsize=6.5)
    ax.text(0.06, 1.12, 'success threshold', color=MUTED, fontsize=6.5, va='bottom')
    ax.grid(False)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    leg = ax.legend(loc='upper left', frameon=False, fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
    for t in leg.get_texts(): t.set_color(INK)
    fig.tight_layout(pad=0.3); fig.savefig(out)
    print('saved', out, 'n', n, [int(g.sum()) for g in groups], 'below diagonal', int((y < x).sum()))


if __name__ == '__main__':
    main(sys.argv[1])
