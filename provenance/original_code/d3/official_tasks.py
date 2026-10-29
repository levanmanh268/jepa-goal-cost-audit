"""Official-style PushT episodes for JEPA-WM checkpoints (Claude, 2026-10-25).

Follows jepa-wms commit 13cf1d9c, evals/simu_env_planning/planning/plan_evaluator.py (goal_source=dset):
  * a segment of length frameskip*goal_H+1 = 31 states is drawn from the official VALIDATION split
    (pusht_noise/val, 21 episodes; states with agent velocities, 7-d);
  * init_state = segment state 0; the 30 recorded actions are replayed in the simulator from init_state;
  * goal observation/state = last frame of that replay; init observation = frame after env.prepare.
  * success = envs/pusht_gym_wrap.py:eval_state: ||goal[:4]-cur[:4]|| < 20 and wrapped |angle diff| < pi/9.
Our sampling seed differs from the official evaluation seeds (not reproducible without their seed shift);
the distribution of (init, goal) is the official one. The expert (dataset) actions are kept OUT of every
planner and candidate bank; they are used only as a labelled privileged control.
"""
import io, pickle
from pathlib import Path
import numpy as np

FRAMESKIP, GOAL_H = 5, 6
ACTION_MEAN = np.array([-0.0087, 0.0068]); ACTION_STD = np.array([0.2019, 0.2002])

class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {('numpy.core.multiarray', 'scalar'), ('numpy', 'dtype'), ('numpy._core.multiarray', 'scalar')}
    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            import importlib
            return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')

def load_val(root):
    import torch
    root = Path(root)
    states = torch.load(root / 'states.pth', weights_only=True).double().numpy()
    vel = torch.load(root / 'velocities.pth', weights_only=True).double().numpy()
    acts = torch.load(root / 'rel_actions.pth', weights_only=True).double().numpy() / 100.0  # env units (env multiplies by 100)
    lens = [int(x) for x in _SafeUnpickler(io.BytesIO((root / 'seq_lengths.pkl').read_bytes())).load()]
    return np.concatenate([states, vel], -1), acts, lens

def sample_episode(states, acts, lens, rng):
    T = FRAMESKIP * GOAL_H + 1
    ok = [i for i, L in enumerate(lens) if L >= T]
    i = int(rng.choice(ok)); off = int(rng.integers(0, lens[i] - T + 1))
    return {'val_episode': i, 'offset': off, 'init_state': states[i, off].copy(), 'expert_actions': acts[i, off:off + T - 1].copy()}

def prepare(env, seed, state):
    env.seed(int(seed)); env.reset_to_state = np.asarray(state, np.float64); obs, st = env.reset(); env.reset_to_state = None
    return obs, np.asarray(st, np.float64)

def rollout(env, seed, init_state, actions, disturb=None):
    """Replay raw env-unit actions [30,2]; disturb: optional [30,2] additive exogenous noise (keyed)."""
    obs, st = prepare(env, seed, init_state)
    for t, a in enumerate(np.asarray(actions, np.float64)):
        if disturb is not None: a = a + disturb[t]
        obs, _, _, info = env.step(a); st = np.asarray(info['state'], np.float64)
    return obs, st

def eval_state(goal_state, cur_state):
    """Exact copy of the official criterion, plus the continuous margin max(pos/20, ang/(pi/9)) (success iff < 1)."""
    pos = float(np.linalg.norm(np.asarray(goal_state[:4]) - np.asarray(cur_state[:4])))
    ang = abs(float(goal_state[4]) - float(cur_state[4])); ang = min(ang, 2 * np.pi - ang)
    return {'success': bool(pos < 20 and ang < np.pi / 9), 'pos_diff': pos, 'angle_diff': ang,
            'margin': max(pos / 20.0, ang / (np.pi / 9)), 'state_dist': float(np.linalg.norm(np.asarray(goal_state) - np.asarray(cur_state)))}

def norm_to_env(plan_norm):
    """[6,10] normalized macro-actions -> [30,2] env-unit raw actions (inverse of the adapter's packing)."""
    p = np.asarray(plan_norm, np.float64).reshape(GOAL_H * FRAMESKIP, 2)
    return p * ACTION_STD + ACTION_MEAN

def env_to_norm(raw):
    r = (np.asarray(raw, np.float64) - ACTION_MEAN) / ACTION_STD
    return r.reshape(GOAL_H, FRAMESKIP * 2)

def fast_rollout(env, seed, init_state, actions, disturb=None, render_final=True):
    """Same physics as PushTEnv.step (PD control + space.step, sim_hz/control_hz substeps) but without the
    per-step rendering and coverage computation. Verified bitwise against rollout() in verify_fast_rollout.py."""
    from pymunk.vec2d import Vec2d
    prepare(env, seed, init_state)
    dt = 1.0 / env.sim_hz; n_steps = env.sim_hz // env.control_hz
    for t, a in enumerate(np.asarray(actions, np.float64)):
        if disturb is not None: a = a + disturb[t]
        act = np.array(a) * env.action_scale
        if env.relative: act = env.agent.position + act
        env.latest_action = act
        for _ in range(n_steps):
            acc = env.k_p * (act - env.agent.position) + env.k_v * (Vec2d(0, 0) - env.agent.velocity)
            env.agent.velocity += acc * dt
            env.space.step(dt)
    st = np.asarray(env._get_obs(), np.float64)
    obs = None
    if render_final:
        vis = env._render_frame('rgb_array'); s32 = env._get_obs(); pro = s32[:2]
        if env.with_velocity: pro = np.concatenate((pro, s32[5:]))
        obs = {'visual': vis, 'proprio': pro}
    return obs, st
