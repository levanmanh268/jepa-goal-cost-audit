"""Candidate planning energies for patch-grid latent world models (Claude, R7, 2026-10-27). NumPy only.

All energies compare a batch of endpoint grids P [B,N,C] (predicted or encoded) with one goal grid G [N,C] on a
square patch lattice (N = S*S), plus the official proprio term.  Lower is better (an energy, used for argmin).

  official   : mean_ic (P_ic - G_ic)^2 + 0.1 * mean (p - g_p)^2                 (JEPA-WM PushT objective)
  pte        : Sinkhorn divergence between the patch measures mu_P = (1/N) sum_i delta_(x_i, P_i) and mu_G with
               ground cost c((x,z),(x',z')) = ||z - z'||^2 / C + beta * ||x - x'||^2  (x in patch units),
               entropic scale eps; + the same 0.1 * proprio term.  Label-free and training-free.
               Limits: beta -> inf and eps -> 0 recovers the official visual term exactly (identity coupling).
  blur       : official term after an isotropic Gaussian blur (sigma patches) of both grids (control for 'any
               spatial smoothing widens the basin')
  pool       : squared distance of mean-pooled grids (sigma -> inf limit of blur)
  chamfer    : symmetric soft-min alignment without mass conservation (control for 'transport' vs 'matching')
  head       : R6 frozen signed channel-energy head (supervised reference), from the channel energies of P - G

Nothing here touches labels; hyperparameters are passed in explicitly by the caller.
"""
import numpy as np

PROPRIO_W = 0.1


def lattice(S):
    ii, jj = np.meshgrid(np.arange(S, dtype=np.float64), np.arange(S, dtype=np.float64), indexing='ij')
    return np.stack([ii.ravel(), jj.ravel()], 1)                      # [N,2] patch units, row-major like the ViT


def proprio_term(p, gp):
    B = p.shape[0]
    return ((p.reshape(B, -1).astype(np.float64) - gp.reshape(-1).astype(np.float64)) ** 2).mean(1)


def official(P, G, p, gp):
    B = P.shape[0]
    vis = ((P.reshape(B, -1).astype(np.float64) - G.reshape(-1).astype(np.float64)) ** 2).mean(1)
    return vis + PROPRIO_W * proprio_term(p, gp)


def visual_official(P, G):
    B = P.shape[0]
    return ((P.reshape(B, -1).astype(np.float64) - G.reshape(-1).astype(np.float64)) ** 2).mean(1)


def feature_cost(A, Bm):
    """A [B,N,C], Bm [B,M,C] or [M,C] -> ||a_i - b_j||^2 / C, shape [B,N,M]."""
    A = A.astype(np.float64); Bm = Bm.astype(np.float64); C = A.shape[-1]
    if Bm.ndim == 2: Bm = np.broadcast_to(Bm, (A.shape[0],) + Bm.shape)
    a2 = (A * A).sum(-1)[:, :, None]; b2 = (Bm * Bm).sum(-1)[:, None, :]
    cost = a2 + b2 - 2.0 * np.einsum('bnc,bmc->bnm', A, Bm, optimize=True)
    return np.maximum(cost, 0.0) / C


def _sinkhorn_cost(Cm, eps, iters=300, tol=1e-9, symmetric=False):
    """Entropic OT value (dual form) for uniform marginals, log-domain, batched. Cm [B,N,M]."""
    B, N, M = Cm.shape
    la = -np.log(N); lb = -np.log(M)
    f = np.zeros((B, N)); g = np.zeros((B, M))
    def lse(x, axis):
        m = x.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)
    for it in range(iters):
        f_new = -eps * lse(lb + (g[:, None, :] - Cm) / eps, 2)
        if symmetric:
            f_new = 0.5 * (f + f_new); g_new = f_new
        else:
            g_new = -eps * lse(la + (f_new[:, :, None] - Cm) / eps, 1)
        delta = max(np.abs(f_new - f).max(), np.abs(g_new - g).max())
        f, g = f_new, g_new
        if delta < tol * max(1.0, np.abs(f).max()):
            break
    return f.mean(1) + g.mean(1), it + 1


def pte_visual(P, G, beta, eps, iters=300, return_iters=False):
    """Sinkhorn divergence S = OT(P,G) - OT(P,P)/2 - OT(G,G)/2 with positional ground cost. P [B,N,C], G [N,C]."""
    B, N, C = P.shape; S = int(round(np.sqrt(N))); X = lattice(S)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)                # [N,N]
    c_pg = feature_cost(P, G) + beta * D2[None]
    c_pp = feature_cost(P, P) + beta * D2[None]
    c_gg = feature_cost(G[None], G[None]) + beta * D2[None]
    ot_pg, n1 = _sinkhorn_cost(c_pg, eps, iters)
    ot_pp, n2 = _sinkhorn_cost(c_pp, eps, iters, symmetric=True)
    ot_gg, n3 = _sinkhorn_cost(c_gg, eps, iters, symmetric=True)
    val = ot_pg - 0.5 * ot_pp - 0.5 * ot_gg[0]
    return (val, (n1, n2, n3)) if return_iters else val


def pte(P, G, p, gp, beta, eps, iters=300):
    return pte_visual(P, G, beta, eps, iters) + PROPRIO_W * proprio_term(p, gp)


def pae_visual(P, G, beta, return_perm=False):
    """Exact (unregularised) W2^2 between the uniform patch measures on the joint (position, feature) space:
    (1/N) min over permutations sigma of sum_i ||P_i - G_sigma(i)||^2 / C + beta * ||x_i - x_sigma(i)||^2.
    For equal-size uniform measures the optimal coupling is a permutation (Birkhoff), so the assignment
    problem is exact.  beta -> inf gives the identity permutation, i.e. the official visual term."""
    from scipy.optimize import linear_sum_assignment
    B, N, C = P.shape; S = int(round(np.sqrt(N))); X = lattice(S)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    Cm = feature_cost(P, G) + beta * D2[None]
    vals = np.empty(B); perms = []
    for b in range(B):
        r, c = linear_sum_assignment(Cm[b]); vals[b] = Cm[b, r, c].mean(); perms.append(c)
    return (vals, np.array(perms)) if return_perm else vals


def pae(P, G, p, gp, beta):
    return pae_visual(P, G, beta) + PROPRIO_W * proprio_term(p, gp)


def gaussian_blur_grid(P, sigma):
    """Separable Gaussian blur over the S x S lattice with reflect padding. P [B,N,C]."""
    if sigma <= 0: return P.astype(np.float64)
    B, N, C = P.shape; S = int(round(np.sqrt(N)))
    r = int(np.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    X = P.astype(np.float64).reshape(B, S, S, C)
    for axis in (1, 2):
        pad = [(0, 0)] * 4; pad[axis] = (r, r)
        Xp = np.pad(X, pad, mode='reflect')
        X = sum(k[t] * np.take(Xp, np.arange(t, t + S), axis=axis) for t in range(2 * r + 1))
    return X.reshape(B, N, C)


def blur(P, G, p, gp, sigma):
    Pb = gaussian_blur_grid(P, sigma); Gb = gaussian_blur_grid(G[None], sigma)[0]
    return visual_official(Pb, Gb) + PROPRIO_W * proprio_term(p, gp)


def pool(P, G, p, gp):
    d = P.astype(np.float64).mean(1) - G.astype(np.float64).mean(0)[None]
    return (d * d).mean(1) + PROPRIO_W * proprio_term(p, gp)


def chamfer(P, G, p, gp, beta, tau):
    """Symmetric soft-min alignment: mean_i softmin_j c_ij + mean_j softmin_i c_ij (temperature tau)."""
    B, N, C = P.shape; S = int(round(np.sqrt(N))); X = lattice(S)
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    c = feature_cost(P, G) + beta * D2[None]
    def softmin(x, axis):
        m = x.min(axis=axis, keepdims=True)
        return (m - tau * np.log(np.exp(-(x - m) / tau).mean(axis=axis, keepdims=True))).squeeze(axis)
    return 0.5 * (softmin(c, 2).mean(1) + softmin(c, 1).mean(1)) + PROPRIO_W * proprio_term(p, gp)


def channel_energies(P, G, p, gp, reproduce_float16=True):
    """R6 features: ch [B,C] = mean over patches of (P-G)^2 (P cast through float16 as in R6), pr [B]."""
    B = P.shape[0]; C = P.shape[-1]
    Pq = P.astype(np.float16) if reproduce_float16 else P
    pq = p.astype(np.float16) if reproduce_float16 else p
    d = Pq.reshape(B, -1, C).astype(np.float64) - G.reshape(-1, C).astype(np.float64)
    pr = ((pq.reshape(B, -1).astype(np.float64) - gp.reshape(-1).astype(np.float64)) ** 2).mean(1)
    return (d * d).mean(1), pr


def head(P, G, p, gp, w):
    ch, pr = channel_energies(P, G, p, gp)
    return ch @ np.asarray(w[:-1]) + pr * w[-1]


def label_free_beta(G, quantile=0.5):
    """Exchange rate fixed from the GOAL grid only: moving one patch costs the median between-patch feature cost.
    beta = median_{i != j} ||g_i - g_j||^2 / C  (per goal; no labels, no predictions)."""
    c = feature_cost(G[None], G[None])[0]; N = c.shape[0]
    off = c[~np.eye(N, dtype=bool)]
    return float(np.quantile(off, quantile))
