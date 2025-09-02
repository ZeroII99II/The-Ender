# SkyForgeBot

This package contains the RLBot implementation of SkyForgeBot.

## Training

Training uses the rlgym 2.0 environment driven through the RLBot v5 runner.
Install the tooling and launch a session with:

```bash
pip install rlbot rlgym==2.* rlgym-compat stable-baselines3
rlbot run -c training/bot.cfg
```

The same configuration can be loaded from the RLBot GUI if you prefer a
graphical interface.  Checkpoints written by the training script can be swapped
in without restarting RLBot.

## Model selection

The policy network is stored as a TorchScript file. By default the bot uses
`necto-model.pt` shipped with this repository. To load a different checkpoint:

* Pass the path when constructing `Agent`, e.g. `Agent("path/to/model.pt")`
* Set the `SKYFORGEBOT_MODEL_PATH` environment variable
* Edit `bot.cfg` and change the `model_path` under the `[Locations]` section

Relative paths in `bot.cfg` are resolved from the configuration file's
location.

## Replacing the model for RLBot matches

To play matches with a new policy, place the checkpoint where `bot.cfg` expects
it or overwrite the `model_path` value. RLBot also honours
`SKYFORGEBOT_MODEL_PATH`, so you can point to a model at launch time:

```bash
export SKYFORGEBOT_MODEL_PATH=/path/to/checkpoint.pt
rlbot run -c SkyForgeBot/bot.cfg
```
