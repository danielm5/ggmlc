import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.frontend.pytorch import export_torch_model
from torch import nn


class TinyCausalDWConv1d(nn.Module):
    def __init__(self, channels: int = 8, k: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, k, groups=channels, padding=k - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[-1]
        return self.conv(x)[:, :, :s]


def test_conv1d_dw_flags_is_1d_and_lowers_dw():
    model = TinyCausalDWConv1d().eval()
    x = torch.randn(2, 8, 16)
    exported = export_torch_model(model, (x,), model_name="tiny_conv1d", enable_fusion=False)
    convs = [n for n in exported.main_graph.nodes if n.opcode.value == "conv2d"]
    assert convs, [n.opcode for n in exported.main_graph.nodes]
    assert all(c.attributes.get("is_1d") == 1 for c in convs)
    ggml = lower_to_ggml(exported.main_graph, enable_fusion=False)
    dw = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_CONV_2D_DW]
    assert dw, [op.opcode for op in ggml.nodes]
    assert all(op.attributes.get("is_1d") == 1 for op in dw)
