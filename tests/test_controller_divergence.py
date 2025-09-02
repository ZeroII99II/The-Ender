import importlib.util
import pathlib
import numpy as np

# Dynamically load EnderAgent from the directory with a space in its name
ender_path = pathlib.Path(__file__).resolve().parents[1] / 'The Ender' / 'ender_agent.py'
spec = importlib.util.spec_from_file_location('ender_agent', ender_path)
ender_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ender_module)
EnderAgent = ender_module.EnderAgent

class NectoStub:
    """Minimal stub mimicking Necto's interface."""
    def act(self, obs):
        # return a different constant action
        return np.zeros(8)

def test_controller_outputs_diverge():
    bot = EnderAgent()
    necto = NectoStub()
    for i in range(5):
        obs = np.random.random(3)  # dummy observation
        bot_action = bot.act(obs)
        necto_action = necto.act(obs)
        print(f"case {i}: bot {bot_action} necto {necto_action}")
        assert not np.allclose(bot_action, necto_action)
