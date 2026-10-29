"""Unit checks for energies.py (Claude, R7). Run: python3 -I test_energies.py
Checks only mathematical identities of the implementation (no data, no outcomes)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import energies as E

rng = np.random.default_rng(0)
out = {}

# 1. PAE limit: beta large -> identity permutation -> official visual term exactly.
S, C = 4, 8; N = S * S
G = rng.normal(size=(N, C)); P = G[None] + 0.3 * rng.normal(size=(3, N, C))
ref = E.visual_official(P, G)
val, perm = E.pae_visual(P, G, beta=1e6, return_perm=True)
out['pae_limit_identity'] = {'max_abs_err': float(np.max(np.abs(val - ref))), 'identity_perm': bool(np.all(perm == np.arange(N)))}
assert np.allclose(val, ref, rtol=1e-12, atol=1e-12) and np.all(perm == np.arange(N))

# 2. PAE is zero at the goal, nonnegative, and never above the official visual term (identity is feasible).
v0 = E.pae_visual(G[None].copy(), G, beta=0.1); assert abs(v0[0]) < 1e-12
for beta in [0.0, 1e-3, 1e-1, 10.0]:
    v = E.pae_visual(P, G, beta=beta); assert np.all(v >= -1e-12) and np.all(v <= ref + 1e-12)
out['pae_le_official'] = True

# 3. PAE is monotone non-decreasing in beta (feasible set fixed, objective increases with beta).
vs = [E.pae_visual(P, G, beta=b) for b in [0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e3]]
assert all(np.all(vs[i] <= vs[i + 1] + 1e-12) for i in range(len(vs) - 1))
out['pae_monotone_in_beta'] = True

# 4. Sinkhorn divergence (entropic variant): ~0 at P = G, positive otherwise.
v0s = E.pte_visual(G[None].copy(), G, beta=1.0, eps=0.05); assert abs(v0s[0]) < 1e-8
assert np.all(E.pte_visual(P, G, beta=1.0, eps=0.05) > 0)

# 5. Blur with sigma 0 equals the official term; pooled energy is close to a very wide blur.
p0 = np.zeros((3, 2)); gp0 = np.zeros(2)
assert np.allclose(E.blur(P, G, p0, gp0, 0.0), E.visual_official(P, G))
out['pool_vs_wide_blur_max_rel'] = float(np.max(np.abs(E.blur(P, G, p0, gp0, 50.0) - E.pool(P, G, p0, gp0)) / E.pool(P, G, p0, gp0)))

# 6. Channel energies: official visual term = mean over channels of ch.
ch, pr = E.channel_energies(P, G, rng.normal(size=(3, 5)), rng.normal(size=5), reproduce_float16=False)
assert np.allclose(ch.mean(1), E.visual_official(P, G))

# 7. Translated blob on a 16 x 16 lattice (descriptive only): official vs PAE profiles.
S, C = 16, 4; N = S * S
def blob_grid(cx, cy, r=1.5):
    X = E.lattice(S); g = np.zeros((N, C)); w = np.exp(-0.5 * (((X[:, 0] - cx) ** 2 + (X[:, 1] - cy) ** 2) / r ** 2))
    g[:, 0] = w; g[:, 1] = 0.5 * w; return g
Gb = blob_grid(5, 5); shifts = [0, 1, 2, 3, 4, 6, 8]
Pb = np.stack([blob_grid(5 + d, 5) for d in shifts])
out['blob_translation_descriptive'] = {'shift': shifts, 'official': np.round(E.visual_official(Pb, Gb), 6).tolist(),
                                       **{f'pae_beta_{b:g}': np.round(E.pae_visual(Pb, Gb, b), 6).tolist() for b in [1e-3, 1e-2, 1e-1]}}
print(json.dumps(out, indent=1))
print('PASS energies unit checks')
