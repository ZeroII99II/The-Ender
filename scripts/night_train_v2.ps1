$ErrorActionPreference="Stop"
$envs = 8
$steps = 1000000
while ($true) {
  ..\.venv\Scripts\python -m src.training.train_v2 --envs $envs --steps $steps --ckpt_dir models\checkpoints --tensorboard runs\ssl_v2
  ..\.venv\Scripts\python -m src.inference.export --sb3 --ckpt models\checkpoints\best_sb3.zip --out models\exported\ssl_policy.ts --obs_dim 107
  Start-Sleep -Seconds 5
}
