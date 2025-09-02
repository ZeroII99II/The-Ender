from __future__ import annotations
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb3", action="store_true", help="Export Stable-Baselines3 PPO policy")
    ap.add_argument("--ckpt", type=str, help="Path to SB3 .zip checkpoint")
    ap.add_argument("--out", type=str, help="Output TorchScript path")
    ap.add_argument("--obs_dim", type=int, default=107)
    args = ap.parse_args()

    if args.sb3:
        from stable_baselines3 import PPO
        import torch
        import torch.nn as nn

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = PPO.load(args.ckpt, device=dev)
        pol = m.policy

        class SB3PolicyExport(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor):
                # SB3 expects batch; ensure 2D
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)
                latent = self.policy.extract_features(obs)
                latent_pi, _ = self.policy.mlp_extractor(latent)
                # Continuous head (5) with tanh
                cont_logits = self.policy.action_net(latent_pi)
                a_cont = torch.tanh(cont_logits[..., :5])
                # Discrete logits (3)
                if cont_logits.shape[-1] >= 8:
                    disc_logits = cont_logits[..., 5:8]
                else:
                    disc_logits = torch.zeros((obs.shape[0], 3), device=obs.device)
                return a_cont, disc_logits

        mod = SB3PolicyExport(pol).to(dev).eval()
        example = torch.zeros(args.obs_dim, device=dev)
        ts = torch.jit.trace(mod, example)
        ts.save(args.out)
        print(f"Saved TorchScript → {args.out}")
        sys.exit(0)

    raise SystemExit("No export path specified")


if __name__ == "__main__":
    main()
