# SkyForgeBot

This package contains the RLBot implementation of SkyForgeBot.

## Model selection

The policy network is stored as a TorchScript file. By default the bot uses
`necto-model.pt` shipped with this repository. To swap in a freshly trained
model, either:

* Pass the path when constructing `Agent`, e.g. `Agent("path/to/model.pt")`, or
* Set the `SKYFORGEBOT_MODEL_PATH` environment variable, or
* Edit `bot.cfg` and change the `model_path` under the `[Locations]` section.

Relative paths in `bot.cfg` are resolved from the configuration file's
location.
