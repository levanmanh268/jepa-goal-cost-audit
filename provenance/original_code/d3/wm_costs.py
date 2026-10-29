"""Latent costs and a faithful CEM re-implementation for the published JEPA-WM / DINO-WM PushT checkpoints.
Claude, 2026-10-25. Cost = official ReprTargetDistMPCObjective, L2, alpha=0.1, terminal step only.
CEM mirrors jepa-wms planning/planner.py:CEMPlanner (diagonal Gaussian in normalized 10-d macro-action space,
mean init 0, std init var_scale=1, mean-sample inclusion, top-k elites, unbiased std, no momentum, no clipping)."""
import time
import numpy as np

def encode_obs_list(model, obs_list):
    import torch
    vis = torch.from_numpy(np.stack([np.asarray(o['visual']).copy() for o in obs_list])).permute(0, 3, 1, 2)[:, None]
    pro = torch.from_numpy(np.stack([np.asarray(o['proprio'], np.float32).copy() for o in obs_list]))[:, None]
    with torch.inference_mode():
        z = model.encode({'visual': vis, 'proprio': pro})
    return z

def latent_cost(zv, zp, zg):
    """zv [B,...], zp [B,...]; zg encoded goal (batch 1, time 1)."""
    return ((zv - zg['visual'][0, 0]) ** 2).flatten(1).mean(1) + 0.1 * ((zp - zg['proprio'][0, 0]) ** 2).flatten(1).mean(1)

class ModelScorer:
    def __init__(self, model, obs0, goal_obs):
        import torch
        self.model = model; self.torch = torch
        self.z0 = encode_obs_list(model, [obs0]); self.zg = encode_obs_list(model, [goal_obs])
        self.n_calls = 0; self.seconds = 0.0
    def __call__(self, actions_norm):
        """actions_norm: torch [6,B,10] normalized. Returns numpy [B] predicted costs and terminal latents."""
        t0 = time.perf_counter()
        with self.torch.inference_mode():
            roll = self.model.unroll(self.z0, actions_norm.to(self.model.device))
            c = latent_cost(roll['visual'][-1], roll['proprio'][-1], self.zg)
        self.n_calls += actions_norm.shape[1]; self.seconds += time.perf_counter() - t0
        return c.cpu().numpy().astype(np.float64), roll['visual'][-1], roll['proprio'][-1]

def cem(score_fn, n_samples, n_iters, n_elites, seed, horizon=6, adim=10, var_scale=1.0, keep_last=True):
    """score_fn(actions_norm torch [H,B,A]) -> numpy costs [B]. Returns plan [H,A], history, last population."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    mean = torch.zeros(horizon, adim); std = var_scale * torch.ones(horizon, adim); hist = []; last = None
    for it in range(n_iters):
        acts = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(horizon, n_samples, adim, generator=g)
        acts[:, 0, :] = mean
        cost = score_fn(acts); cost_t = torch.from_numpy(cost)
        el = torch.topk(-cost_t, n_elites).indices
        hist.append({'iter': it, 'min_cost': float(cost.min()), 'elite_mean_cost': float(cost[el.numpy()].mean())})
        if keep_last and it == n_iters - 1:
            last = {'actions': acts.permute(1, 0, 2).numpy().copy(), 'cost': cost.copy()}
        mean = acts[:, el].mean(1); std = acts[:, el].std(1)
    return mean.numpy().copy(), hist, last
