from __future__ import annotations
import argparse
import os
import torch
from stable_baselines3 import PPO
from src.training.env_factory import make_env
from src.training.ppo_sb3 import make_vec, policy_kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_dir", type=str, default="models/checkpoints")
    ap.add_argument("--tensorboard", type=str, default="runs/ssl_v2")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    env_fns = [make_env(seed=args.seed + i) for i in range(args.envs)]
    vec = make_vec(env_fns)

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=max(2048 // args.envs, 128),
        batch_size=4096,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=6,
        tensorboard_log=args.tensorboard,
        policy_kwargs=policy_kwargs(),
        verbose=1,
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.learn(total_timesteps=args.steps)
    model.save(os.path.join(args.ckpt_dir, "best_sb3.zip"))


if __name__ == "__main__":
    main()
