"""Conjunctive (tail) aggregation of per-patch goal errors for patch-grid latent planners (Claude, R7, 2026-10-27).

Per-patch error e_i = mean_c (z_ic - g_ic)^2 (same units as the official visual term, which is mean_i e_i).
  topk(k)    : mean of the k largest e_i (an empirical CVaR of the patch errors at level k/N)
  powmean(q) : (mean_i e_i^q)^(1/q)   (q = 1 is the official visual term; q -> inf is max_i e_i)
  fixedmask  : mean of e_i over a FIXED subset of K patches chosen per start before planning, label-free, from the
               goal-vs-start difference ||g_i - z0_i||^2 (largest K) -- a cost-side analogue of goal-conditioned
               token selection (e.g. COSTGRAD selects tokens by cost-gradient norm; this is NOT that method).
All variants add the unchanged official proprio term 0.1 * mean (p - g_p)^2.
Rationale (hypothesis, development): the mean over patches weights each entity by its footprint, so small entities
(the agent) and boundary-local changes (rotation) are diluted; success criteria are conjunctive over entities.
"""
import numpy as np
from energies import proprio_term, PROPRIO_W


def patch_errors(P, G):
    """P [B,N,C] float, G [N,C] -> e [B,N]."""
    d = P.astype(np.float64) - G.astype(np.float64)[None]
    return (d * d).mean(2)


def topk(P, G, p, gp, k):
    e = patch_errors(P, G); es = -np.sort(-e, 1)
    return es[:, :k].mean(1) + PROPRIO_W * proprio_term(p, gp)


def powmean(P, G, p, gp, q):
    e = patch_errors(P, G)
    return np.mean(e ** q, 1) ** (1.0 / q) + PROPRIO_W * proprio_term(p, gp)


def fixedmask_indices(G, Z0, K):
    dg = ((G.astype(np.float64) - Z0.astype(np.float64)) ** 2).mean(1)
    return np.argsort(-dg, kind='stable')[:K]


def fixedmask(P, G, p, gp, idx):
    e = patch_errors(P, G)
    return e[:, idx].mean(1) + PROPRIO_W * proprio_term(p, gp)
