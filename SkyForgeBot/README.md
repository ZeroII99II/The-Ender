# SkyForgeBot

This package contains the RLBot implementation of SkyForgeBot.

## Training

Training uses [rlgym](https://github.com/RLBot/RLGym) 2.0 and Stable-Baselines3.
Install the dependencies in your Python environment:

```
pip install rlgym==2.* stable-baselines3 torch
```

Run the training script from the project root to produce a TorchScript model:

```
python training/train.py
```

The resulting `trained-model.pt` is written into the `SkyForgeBot` directory.
Set `SKYFORGEBOT_MODEL_PATH` or update `bot.cfg` to point to this file when
running the bot.

## Model selection

The policy network is stored as a TorchScript file. By default the bot uses
`necto-model.pt` shipped with this repository. To swap in freshly trained
weights, either:

* Pass the path when constructing `Agent`, e.g. `Agent("path/to/model.pt")`, or
* Set the `SKYFORGEBOT_MODEL_PATH` environment variable, or
* Edit `bot.cfg` and change `model_path` under the `[Locations]` section.

Relative paths in `bot.cfg` are resolved from the configuration file's
location.

## RLBot requirements

SkyForgeBot is built for RLBot matches running at 120 Hz with two cars per
team. Using different tick rates or team sizes may lead to unexpected
behaviour.
