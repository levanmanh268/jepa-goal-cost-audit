"""Figure 1 (post hoc, descriptive): within-bank rank of each proprio energy part vs rank of the true final agent
distance, all 16 R8 evaluation banks x 64 plans (JEPA-WM). Data: r11/r8_proprio_diag.npz. Claude, 2026-10-28.
Usage: python3 -I make_fig_audit.py fig_audit.pdf
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

Z = np.load('/home/claude/wm/r11/r8_proprio_diag.npz')
K, N = Z['agent_dist'].shape
INK, INK2, GRID, S1 = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
plt.rcParams.update({'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'font.family': 'serif'})
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True, constrained_layout=True)
for ax, part, name in [(axes[0], 'e_pos', 'position part'), (axes[1], 'e_vel', 'velocity part')]:
    xs, ys, rhos = [], [], []
    for k in range(K):
        a, d = Z[part][k], Z['agent_dist'][k]
        xs.append((rankdata(a) - 1) / (N - 1)); ys.append((rankdata(d) - 1) / (N - 1)); rhos.append(spearmanr(a, d).statistic)
    ax.scatter(np.concatenate(xs), np.concatenate(ys), s=3, c=S1, alpha=0.35, linewidths=0, rasterized=True)
    ax.set_title(f'{name}\nmean Spearman {np.mean(rhos):.2f}'.replace('-', '−'), color=INK, fontsize=7)
    ax.set_xlabel('within-bank rank of energy', color=INK)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.grid(True, color=GRID, linewidth=0.5); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
axes[0].set_ylabel('rank of final agent distance', color=INK)
fig.savefig(sys.argv[1], dpi=300); print('saved', sys.argv[1])
