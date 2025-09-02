# The Ender

SkyForgeBot is an RLBot-compatible Rocket League bot powered by TorchScript models.
This document describes installing dependencies, training new models, and
packaging the bot for RLBot Championship submissions.

## Dependencies and Installation

1. Install [Python](https://www.python.org/) 3.8+.
2. Install runtime dependencies:
   ```bash
   pip install -r SkyForgeBot/requirements.txt
   ```
3. (Optional) For training, also install:
   ```bash
   pip install stable-baselines3 rlgym
   ```

## Training new weights

Run the provided training script to produce a TorchScript actor:
```bash
python training/train.py
```
The script saves `necto-model.pt` directly inside `SkyForgeBot/`.  Since
`bot.cfg` already references this path, RLBot will automatically load the newly
trained model without any renaming or file moves.
The script writes the resulting model to the location specified by the
`SKYFORGEBOT_MODEL_PATH` environment variable. When the variable is unset, the
file defaults to `SkyForgeBot/trained-model.pt`. Update
`SkyForgeBot/bot.cfg`'s `model_path` or set `SKYFORGEBOT_MODEL_PATH` when
running RLBot so the fresh weights are picked up. Rename the file if you want
to replace the shipped `necto-model.pt`.

You can also launch the training process through RLBot by using
`training/bot.cfg` directly or referencing it from `rlbot.cfg`.

## Packaging for RLBot Championship

1. Ensure the following files are present in the repository:
   - `SkyForgeBot/bot.cfg`
   - `SkyForgeBot/appearance.cfg`
   - `SkyForgeBot/logo.png`
   - `SkyForgeBot/requirements.txt`
   - `bob.toml` and `bot.toml`
   - the model file referenced by `bot.cfg`
2. Install the packager and build a release zip:
   ```bash
   pip install rlbotpackager
   python -m rlbotpackager package
   ```
   The resulting archive in `dist/` is ready for RLBot Championship submission.

## 120 Hz execution tips

- `SkyForgeBot/bot.cfg` sets `maximum_tick_rate_preference = 120`.
- In the RLBot GUI or runner, verify the match is configured for 120 tick
  rate; lower rates will slow decision making.

## Verifying 2v2 compatibility

`training/train.py` builds a 2v2 environment (`team_size=2`), so the agent
defaults to supporting two cars per team.  To confirm it works in game:

1. Launch RLBot and configure a match with two `SkyForgeBot` instances per team.
2. Start the match and ensure all four bots spawn and respond without errors.

