from __future__ import annotations
try:
    import gymnasium as gym
except Exception:  # last-resort fallback
    import gym  # type: ignore

def reset_env(env, **kwargs):
    out = env.reset(**kwargs)
    return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})

def step_env(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, term, trunc, info = out
        return obs, r, bool(term or trunc), info
    else:
        obs, r, done, info = out
        return obs, r, bool(done), info

__all__ = ["gym", "reset_env", "step_env"]
