import torch
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.frontend.pytorch import export_torch_model
from ggmlc.ir.op import OpCode
from ggmlc.ops import gated_delta_net
from torch import nn


class TinyGDN(nn.Module):
    def forward(self, q, k, v, g, beta, state):
        return gated_delta_net(q, k, v, g, beta, state)


def test_gated_delta_net_imports_and_lowers():
    b, s, h, d = 1, 4, 2, 8
    model = TinyGDN().eval()
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    v = torch.randn(b, s, h, d)
    g = torch.randn(b, s, h, 1)
    beta = torch.rand(b, s, h, 1)
    state = torch.zeros(b, h, d, d)
    exported = export_torch_model(
        model, (q, k, v, g, beta, state), model_name="tiny_gdn", enable_fusion=False
    )
    opcodes = [n.opcode for n in exported.main_graph.nodes]
    assert OpCode.GATED_DELTA_NET in opcodes
    ggml = lower_to_ggml(exported.main_graph, enable_fusion=False)
    gdn_ops = [op for op in ggml.nodes if op.opcode == GGMLOpCode.GGML_OP_GATED_DELTA_NET]
    assert len(gdn_ops) == 1
    assert gdn_ops[0].attributes.get("K", 1) == 1


def test_gated_delta_net_keeps_dynamic_batch():
    class TinyGDN(nn.Module):
        def forward(self, q, k, v, g, beta, state):
            return gated_delta_net(q, k, v, g, beta, state)

    b, s, h, d = 2, 4, 2, 8
    model = TinyGDN().eval()
    q = torch.randn(b, s, h, d)
    k = torch.randn(b, s, h, d)
    v = torch.randn(b, s, h, d)
    g = torch.randn(b, s, h, 1)
    beta = torch.rand(b, s, h, 1)
    state = torch.zeros(b, h, d, d)
    dim_b = torch.export.Dim("b", min=1, max=8)
    dyn = ({0: dim_b}, {0: dim_b}, {0: dim_b}, {0: dim_b}, {0: dim_b}, {0: dim_b})
    exported = export_torch_model(
        model,
        (q, k, v, g, beta, state),
        model_name="tiny_gdn_dyn",
        enable_fusion=False,
        dynamic_shapes=dyn,
    )
    gdn = [n for n in exported.main_graph.nodes if n.opcode == OpCode.GATED_DELTA_NET]
    assert gdn
    out = exported.main_graph.get_tensor(gdn[0].outputs[0])
    v_t = exported.main_graph.get_tensor(gdn[0].inputs[2])
    assert str(out.shape.dims[0]) == str(v_t.shape.dims[0])
    assert not out.shape.dims[0].is_static()
