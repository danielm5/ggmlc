import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.frontend.pytorch import export_torch_model
from torch import nn


class GroupedMultiplierConv(nn.Module):
    """Grouped conv with channel multiplier: groups=80, C_in=80, C_out=160.

    Not depthwise, so ggml has no single op for it: the importer must
    decompose it into per-group convs instead of emitting CONV_2D_DW.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(80, 160, 7, padding=3, groups=80, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TrueDepthwiseConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def test_grouped_multiplier_decomposes_without_dw():
    model = GroupedMultiplierConv().eval()
    x = torch.randn(1, 80, 16, 16)
    exported = export_torch_model(model, (x,), model_name="tiny_grouped", enable_fusion=False)
    ggml = lower_to_ggml(exported.main_graph, enable_fusion=False)
    dw = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_CONV_2D_DW]
    assert not dw, [op.name for op in dw]
    convs = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_CONV_2D]
    assert len(convs) == 80, [op.name for op in convs]


def test_true_depthwise_stays_single_dw():
    model = TrueDepthwiseConv().eval()
    x = torch.randn(1, 16, 16, 16)
    exported = export_torch_model(model, (x,), model_name="tiny_dw", enable_fusion=False)
    ggml = lower_to_ggml(exported.main_graph, enable_fusion=False)
    dw = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_CONV_2D_DW]
    assert len(dw) == 1, [op.opcode for op in ggml.nodes]
    convs = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_CONV_2D]
    assert not convs
