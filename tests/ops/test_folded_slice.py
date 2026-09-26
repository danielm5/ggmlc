"""5D stacked slices (e.g. QKV getitem) must lower to valid 4D views.

Outer dims fold into ggml ne[3], so the slice lands on ggml dim 3 with an
offset multiplier - never an out-of-range stride index like nb[4].
"""

from pathlib import Path

from ggmlc.codegen import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass


def _stack_graph(batch):
    g = Graph(name="qkv_split")
    stacked = g.add_tensor(
        "stacked", Shape.from_tuple((3, batch, 2, 4, 8)), DType.F32, StorageClass.INPUT
    )
    outs = []
    for i in range(3):
        o = g.add_tensor(
            f"part{i}", Shape.from_tuple((batch, 2, 4, 8)), DType.F32, StorageClass.OUTPUT
        )
        outs.append(o.id)
    g.inputs = [stacked.id]
    g.outputs = outs
    for i, oid in enumerate(outs):
        g.add_op(
            OpCode.SLICE,
            [stacked.id],
            [oid],
            attributes={"dim": 0, "start": i, "end": i + 1, "step": 1},
            name=f"getitem_{i}",
        )
    return g


def _lowered_views(batch):
    ggml_graph = lower_to_ggml(_stack_graph(batch), enable_fusion=False)
    views = [op for op in ggml_graph.nodes if op.opcode == GGMLOpCode.GGML_OP_VIEW]
    assert len(views) == 3, [op.opcode for op in ggml_graph.nodes]
    return ggml_graph, views


def test_folded_slice_stays_within_4d():
    _, views = _lowered_views(batch=1)
    for op in views:
        assert op.attributes["ggml_dim"] <= 3, op.attributes
        assert op.attributes.get("offset_mult", 1) == 1, op.attributes


def test_folded_slice_offset_multiplier():
    _, views = _lowered_views(batch=2)
    for op in views:
        assert op.attributes["ggml_dim"] == 3, op.attributes
        assert op.attributes["offset_mult"] == 2, op.attributes


def test_folded_slice_codegen_has_no_oob_stride(tmp_path):
    ggml_graph, _ = _lowered_views(batch=1)
    proj_dir = Path(tmp_path) / "proj"
    generate_cpp_project(ggml_graph, proj_dir, model_name="FoldedSlice")
    text = (proj_dir / "FoldedSlice.h").read_text(encoding="utf-8")
    assert "nb[4]" not in text
    assert "->nb[3]" in text
