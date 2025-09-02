import importlib.util
import pathlib
import sys
import types
from unittest.mock import patch

# Provide minimal stubs so agent.py can be imported without heavy dependencies.
numpy_stub = types.ModuleType("numpy")
sys.modules.setdefault("numpy", numpy_stub)

torch_stub = types.ModuleType("torch")
torch_stub.jit = types.SimpleNamespace(load=lambda f: None)
torch_stub.set_num_threads = lambda n: None
sys.modules.setdefault("torch", torch_stub)

nn = types.ModuleType("nn")
nn.functional = types.ModuleType("functional")
sys.modules.setdefault("torch.nn", nn)
sys.modules.setdefault("torch.nn.functional", nn.functional)

dists = types.ModuleType("distributions")
dists.Categorical = object
sys.modules.setdefault("torch.distributions", dists)

spec = importlib.util.spec_from_file_location(
    "agent", pathlib.Path(__file__).resolve().parents[1] / "SkyForgeBot" / "agent.py"
)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
Agent = agent_module.Agent


def test_constructor_overrides_path(tmp_path):
    model_file = tmp_path / "dummy.pt"
    model_file.write_bytes(b"stub")
    with patch("torch.jit.load") as mock_load:
        Agent(model_path=str(model_file))
        assert mock_load.call_args[0][0].name == str(model_file)


def test_env_var_used_when_no_arg(tmp_path, monkeypatch):
    model_file = tmp_path / "env.pt"
    model_file.write_bytes(b"stub")
    monkeypatch.setenv("SKYFORGEBOT_MODEL_PATH", str(model_file))
    try:
        with patch("torch.jit.load") as mock_load:
            Agent()
            assert mock_load.call_args[0][0].name == str(model_file)
    finally:
        monkeypatch.delenv("SKYFORGEBOT_MODEL_PATH", raising=False)
