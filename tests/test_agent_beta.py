import sys
import types
import pathlib
import importlib.util
import numpy as np

# --- Torch stubs ---
class TensorWrapper:
    def __init__(self, array):
        self.array = np.array(array, dtype=float)
    def float(self):
        return self.array.astype(float)
    @property
    def shape(self):
        return self.array.shape

def from_numpy(arr):
    return TensorWrapper(arr)

class NoGrad:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def stack(lst):
    arr = np.stack([getattr(a, 'array', a) for a in lst])
    class StackResult:
        def __init__(self, arr):
            self.arr = arr
        def swapdims(self, a, b):
            return np.swapaxes(self.arr, a, b)
    return StackResult(arr)

def pad(array, pad, value):
    before, after = pad
    return np.pad(array, ((0,0), (before, after)), constant_values=value)

class Categorical:
    def __init__(self, logits):
        logits = np.array(logits, dtype=float)
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        self.probs = exp / exp.sum(axis=-1, keepdims=True)
    def sample(self):
        samples = [np.random.choice(len(p), p=p) for p in self.probs]
        class Result:
            def __init__(self, arr):
                self._arr = np.array(arr)
            def numpy(self):
                return self._arr
        return Result(samples)

class StubActor:
    def __call__(self, state):
        outputs = [
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[1.0, 2.0, 0.1]]),
            np.array([[0.5, 1.5, 0.7]]),
            np.array([[2.0, 0.0, 1.0]]),
            np.array([[0.1, 0.9, 0.5]]),
        ]
        return outputs, None

torch_stub = types.ModuleType('torch')
torch_stub.from_numpy = from_numpy
torch_stub.no_grad = NoGrad
torch_stub.stack = stack
torch_stub.isfinite = np.isfinite
torch_stub.set_num_threads = lambda n: None
torch_stub.tensor = lambda data: TensorWrapper(data)
torch_stub.jit = types.SimpleNamespace(load=lambda f: StubActor())

nn = types.ModuleType('nn')
nn.functional = types.SimpleNamespace(pad=pad)

dists = types.ModuleType('distributions')
dists.Categorical = Categorical

sys.modules['torch'] = torch_stub
sys.modules['torch.nn'] = nn
sys.modules['torch.nn.functional'] = nn.functional
sys.modules['torch.distributions'] = dists
# --- End torch stubs ---

# Import Agent after stubs are in place
agent_path = pathlib.Path(__file__).resolve().parents[1] / 'SkyForgeBot' / 'agent.py'
spec = importlib.util.spec_from_file_location('agent', agent_path)
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
Agent = agent_module.Agent


def test_act_respects_beta(tmp_path):
    model_file = tmp_path / 'dummy.pt'
    model_file.write_bytes(b'stub')
    agent = Agent(model_path=str(model_file))

    state = (
        np.zeros((1,1,29)),
        np.zeros((1,1,24)),
        np.zeros((1,1)),
    )

    # beta=1 -> argmax
    act1, _ = agent.act(state, beta=1)
    assert act1[0] == 1 and act1[1] == 0

    # beta=-1 -> argmin
    actm1, _ = agent.act(state, beta=-1)
    assert actm1[0] == -1 and actm1[1] == 1

    # beta=0 -> uniform random but deterministic with seeding
    np.random.seed(0)
    act0_a, _ = agent.act(state, beta=0)
    np.random.seed(0)
    act0_b, _ = agent.act(state, beta=0)
    assert np.array_equal(act0_a, act0_b)
    assert not np.array_equal(act0_a[:2], act1[:2])
    assert not np.array_equal(act0_a[:2], actm1[:2])
