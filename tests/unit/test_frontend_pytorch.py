import numpy as np
import torch
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir import OpCode
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


def test_release_module_storage_keeps_exported_weights():
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=True)
    weight = model.weight.detach().cpu().numpy().copy()
    bias = model.bias.detach().cpu().numpy().copy()
    exported = export_torch_model(
        model,
        (torch.randn(2, 8),),
        model_name="linear",
        optimize=False,
        release_module_storage=True,
    )
    arrays = {
        tensor.name: tensor.data
        for tensor in exported.main_graph.tensors.values()
        if getattr(tensor, "data", None) is not None
    }

    def _matches(target: np.ndarray) -> bool:
        return any(
            arr.shape == target.shape and np.allclose(arr, target) for arr in arrays.values()
        )

    assert _matches(weight)
    assert _matches(bias)
    assert model.weight.numel() == 0
    assert model.bias.numel() == 0


def test_export_leaves_module_parameters_intact():
    model = nn.Linear(8, 4, bias=True)
    before = model.weight.detach().cpu().numpy().copy()
    export_torch_model(model, (torch.randn(2, 8),), model_name="linear", optimize=False)
    assert model.weight.shape == before.shape
    assert np.allclose(model.weight.detach().cpu().numpy(), before)


def test_export_simple_model():
    model = SimpleModel()
    x = torch.randn(2, 32)
    m = export_torch_model(model, (x,), model_name="simple")

    assert m.name == "simple"
    g = m.main_graph
    assert len(g.inputs) == 1
    assert len(g.outputs) == 1
    assert len(g.parameters) == 2  # weight and bias

    # Check ops in graph
    opcodes = [op.opcode for op in g.nodes]
    assert OpCode.MATMUL in opcodes or OpCode.LINEAR in opcodes
    assert OpCode.RELU in opcodes


class DynamicModel(nn.Module):
    def forward(self, a, b):
        return (a + b) * 2.0


def test_dynamic_shape_export():
    m = DynamicModel()
    a = torch.randn(2, 16)
    b = torch.randn(2, 16)
    dim_b = torch.export.Dim("batch", min=1, max=32)
    dynamic_shapes = {"a": {0: dim_b}, "b": {0: dim_b}}

    model = export_torch_model(m, (a, b), dynamic_shapes=dynamic_shapes, model_name="dynamic_add")
    g = model.main_graph
    assert len(g.inputs) == 2
    assert len(g.outputs) == 1

    in_a = g.get_tensor(g.inputs[0])
    assert not in_a.shape[0].is_static()
    assert len(in_a.shape[0].free_symbols()) == 1
