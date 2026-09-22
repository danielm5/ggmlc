"""Qwen3 zero-centered RMSNorm (rms(x) * (1 + weight)) must fuse to RMS_NORM."""

from __future__ import annotations

import numpy as np
from ggmlc.ir.dtype import DType
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import StorageClass
from ggmlc.transforms.dce import DeadCodeEliminationPass
from ggmlc.transforms.fusion import FusionOptions, fuse_operations


def _zero_centered_graph(weight: np.ndarray, eps: float = 1e-6) -> Graph:
    g = Graph(name="rms_zero")
    dim = int(weight.shape[0])
    x = g.add_tensor("x", Shape.from_tuple((2, 4, dim)), DType.F32, StorageClass.INPUT)
    w = g.add_tensor(
        "weight",
        Shape.from_tuple((dim,)),
        DType.F32,
        StorageClass.PARAMETER,
        data=weight.astype(np.float32),
    )
    ones = g.add_tensor(
        "ones",
        Shape.from_tuple(()),
        DType.F32,
        StorageClass.CONSTANT,
        data=np.float32(1.0),
    )
    eps_t = g.add_tensor(
        "eps",
        Shape.from_tuple(()),
        DType.F32,
        StorageClass.CONSTANT,
        data=np.float32(eps),
    )
    pow_o = g.add_tensor("pow", Shape.from_tuple((2, 4, dim)), DType.F32, StorageClass.ACTIVATION)
    mean_o = g.add_tensor("mean", Shape.from_tuple((2, 4, 1)), DType.F32, StorageClass.ACTIVATION)
    add_eps = g.add_tensor(
        "add_eps", Shape.from_tuple((2, 4, 1)), DType.F32, StorageClass.ACTIVATION
    )
    rsqrt_o = g.add_tensor("rsqrt", Shape.from_tuple((2, 4, 1)), DType.F32, StorageClass.ACTIVATION)
    mul_x = g.add_tensor("mul_x", Shape.from_tuple((2, 4, dim)), DType.F32, StorageClass.ACTIVATION)
    add_w = g.add_tensor("add_w", Shape.from_tuple((dim,)), DType.F32, StorageClass.ACTIVATION)
    out = g.add_tensor("out", Shape.from_tuple((2, 4, dim)), DType.F32, StorageClass.OUTPUT)

    g.inputs = [x.id]
    g.parameters = [w.id]
    g.outputs = [out.id]
    g.add_op(OpCode.POW, [x.id], [pow_o.id], {"exponent": 2.0})
    g.add_op(OpCode.MEAN, [pow_o.id], [mean_o.id], {"dim": -1, "keepdim": 1})
    g.add_op(OpCode.ADD, [mean_o.id, eps_t.id], [add_eps.id])
    g.add_op(OpCode.RSQRT, [add_eps.id], [rsqrt_o.id])
    g.add_op(OpCode.MUL, [x.id, rsqrt_o.id], [mul_x.id])
    g.add_op(OpCode.ADD, [w.id, ones.id], [add_w.id])
    g.add_op(OpCode.MUL, [mul_x.id, add_w.id], [out.id], name="mul_1")
    return g


def test_zero_centered_rms_fuses_one_plus_weight():
    weight = np.array([0.0, 0.25, -0.5, 1.5], dtype=np.float32)
    g = _zero_centered_graph(weight)
    fuse_operations(
        g,
        FusionOptions(
            enable_rope=False,
            enable_bake_rms_into_linear=False,
            enable_bake_affine=False,
            enable_horizontal_mlp=False,
            enable_horizontal_qkv=False,
        ),
    )
    rms = [n for n in g.nodes if n.opcode == OpCode.RMS_NORM]
    assert len(rms) == 1
    assert (
        rms[0].attributes["eps"] == np.float32(1e-6) or abs(rms[0].attributes["eps"] - 1e-6) < 1e-12
    )
    gamma = g.get_tensor(rms[0].inputs[1])
    np.testing.assert_allclose(np.asarray(gamma.data).reshape(-1), 1.0 + weight)

    pruned = DeadCodeEliminationPass().run(g).graph
    assert any(n.opcode == OpCode.RMS_NORM for n in pruned.nodes)
    assert not any(n.opcode == OpCode.RSQRT for n in pruned.nodes)
    # Original weight is unused once gamma = 1+w is the RMS input.
    assert all(pruned.get_tensor(pid).name != "weight" for pid in pruned.parameters)


def test_plain_rms_gamma_still_fuses():
    g = Graph(name="rms_plain")
    dim = 4
    x = g.add_tensor("x", Shape.from_tuple((2, dim)), DType.F32, StorageClass.INPUT)
    w = g.add_tensor(
        "weight",
        Shape.from_tuple((dim,)),
        DType.F32,
        StorageClass.PARAMETER,
        data=np.ones(dim, dtype=np.float32),
    )
    eps_t = g.add_tensor(
        "eps", Shape.from_tuple(()), DType.F32, StorageClass.CONSTANT, data=np.float32(1e-5)
    )
    pow_o = g.add_tensor("pow", Shape.from_tuple((2, dim)), DType.F32, StorageClass.ACTIVATION)
    mean_o = g.add_tensor("mean", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    add_eps = g.add_tensor("add_eps", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    rsqrt_o = g.add_tensor("rsqrt", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    mul_x = g.add_tensor("mul_x", Shape.from_tuple((2, dim)), DType.F32, StorageClass.ACTIVATION)
    out = g.add_tensor("out", Shape.from_tuple((2, dim)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.parameters = [w.id]
    g.outputs = [out.id]
    g.add_op(OpCode.POW, [x.id], [pow_o.id], {"exponent": 2.0})
    g.add_op(OpCode.MEAN, [pow_o.id], [mean_o.id], {"dim": -1, "keepdim": 1})
    g.add_op(OpCode.ADD, [mean_o.id, eps_t.id], [add_eps.id])
    g.add_op(OpCode.RSQRT, [add_eps.id], [rsqrt_o.id])
    g.add_op(OpCode.MUL, [x.id, rsqrt_o.id], [mul_x.id])
    g.add_op(OpCode.MUL, [mul_x.id, w.id], [out.id])

    fuse_operations(
        g,
        FusionOptions(
            enable_rope=False,
            enable_bake_rms_into_linear=False,
            enable_bake_affine=False,
            enable_horizontal_mlp=False,
            enable_horizontal_qkv=False,
        ),
    )
    rms = [n for n in g.nodes if n.opcode == OpCode.RMS_NORM]
    assert len(rms) == 1
    assert rms[0].inputs[1] == w.id


def test_l2_norm_is_not_rms():
    """x * rsqrt(sum(x^2)+eps) has no gamma and must stay expanded."""
    g = Graph(name="l2")
    dim = 4
    x = g.add_tensor("x", Shape.from_tuple((2, dim)), DType.F32, StorageClass.INPUT)
    eps_t = g.add_tensor(
        "eps", Shape.from_tuple(()), DType.F32, StorageClass.CONSTANT, data=np.float32(1e-6)
    )
    sqr = g.add_tensor("sqr", Shape.from_tuple((2, dim)), DType.F32, StorageClass.ACTIVATION)
    sum_o = g.add_tensor("sum", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    add_o = g.add_tensor("add", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    rsqrt_o = g.add_tensor("rsqrt", Shape.from_tuple((2, 1)), DType.F32, StorageClass.ACTIVATION)
    out = g.add_tensor("out", Shape.from_tuple((2, dim)), DType.F32, StorageClass.OUTPUT)
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.add_op(OpCode.MUL, [x.id, x.id], [sqr.id])
    g.add_op(OpCode.SUM, [sqr.id], [sum_o.id], {"dim": -1, "keepdim": 1})
    g.add_op(OpCode.ADD, [sum_o.id, eps_t.id], [add_o.id])
    g.add_op(OpCode.RSQRT, [add_o.id], [rsqrt_o.id])
    g.add_op(OpCode.MUL, [x.id, rsqrt_o.id], [out.id])

    fuse_operations(
        g,
        FusionOptions(
            enable_rope=False,
            enable_bake_rms_into_linear=False,
            enable_bake_affine=False,
            enable_horizontal_mlp=False,
            enable_horizontal_qkv=False,
        ),
    )
    assert not any(n.opcode == OpCode.RMS_NORM for n in g.nodes)
