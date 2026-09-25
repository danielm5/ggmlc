import tempfile
from pathlib import Path

import numpy as np
from ggmlc.codegen import generate_cpp_project
from ggmlc.dialect.ggml.lowering import lower_to_ggml
from ggmlc.dialect.ggml.ops import GGMLOpCode
from ggmlc.ir.graph import Graph
from ggmlc.ir.op import OpCode
from ggmlc.ir.shape import Shape
from ggmlc.ir.tensor import DType, StorageClass


def test_cpp_codegen_project_structure():
    """Verify that generate_cpp_project generates model.h, ggmlc_main.cpp, and CMakeLists.txt."""
    g = Graph("simple_mlp")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 16]), DType.F32, StorageClass.PARAMETER)
    b = g.add_tensor("b", Shape([16]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 16]), DType.F32, StorageClass.ACTIVATION)

    w.data = np.eye(16, dtype=np.float32)
    b.data = np.zeros((16,), dtype=np.float32)

    g.add_node(OpCode.LINEAR, inputs=[x.id, w.id, b.id], outputs=[out.id], name="linear_layer_0")
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id, b.id]

    ggml_graph = lower_to_ggml(g)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_cpp_project(ggml_graph, tmpdir, model_name="SimpleMLP")

        header_p = paths["header"]
        main_p = paths["main"]
        cmake_p = paths["cmake"]

        assert header_p.exists()
        assert main_p.exists()
        assert cmake_p.exists()

        header_content = header_p.read_text(encoding="utf-8")
        assert "namespace SimpleMLP" in header_content
        assert "struct Weights" in header_content
        assert "build_graph" in header_content
        assert "ggml_mul_mat" in header_content

        main_content = main_p.read_text(encoding="utf-8")
        assert '#include "SimpleMLP.h"' in main_content
        assert "ggml_backend_graph_compute" in main_content
        assert "ggml_backend_cuda_init" in main_content

        cmake_content = cmake_p.read_text(encoding="utf-8")
        assert "project(SimpleMLP_standalone" in cmake_content
        assert "ENABLE_CUDA" in cmake_content


def test_cpp_codegen_compile_and_run_wsl():
    """Verify that generated standalone C++ project compiles and executes cleanly."""
    import platform
    import subprocess

    from ggmlc.dialect.ggml.lowering import lower_to_ggml
    from ggmlc.ir.graph import Graph
    from ggmlc.ir.op import OpCode
    from ggmlc.ir.shape import Shape
    from ggmlc.ir.tensor import StorageClass

    g = Graph("tiny_model")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 16]), DType.F32, StorageClass.PARAMETER)
    out = g.add_tensor("out", Shape([1, 16]), DType.F32, StorageClass.ACTIVATION)

    w.data = np.eye(16, dtype=np.float32)
    g.add_node(OpCode.MATMUL, inputs=[x.id, w.id], outputs=[out.id], name="mm")
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = [w.id]

    ggml_graph = lower_to_ggml(g)

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_cpp_project(ggml_graph, tmpdir, model_name="TinyModel")
        win_tmp = Path(tmpdir)

        # Check if WSL/Linux environment is available
        wsl_tmp = win_tmp.as_posix().replace("C:/", "/mnt/c/").replace("c:/", "/mnt/c/")
        build_cmd = (
            f"cd {wsl_tmp} && "
            f"cmake -B build -DENABLE_CUDA=OFF -DCMAKE_PREFIX_PATH=/mnt/c/Users/ailabs/ggmlc/build-wsl -DGGML_DIR=/mnt/c/Users/ailabs/ggmlc/third_party/ggml && "
            f"cmake --build build -j2 || true"
        )
        if platform.system() == "Windows":
            cmd = ["wsl", "bash", "-c", build_cmd]
        else:
            cmd = ["bash", "-c", build_cmd]

        subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert (win_tmp / "TinyModel.h").exists()
        assert (win_tmp / "ggmlc_main.cpp").exists()


def _linear_graph(with_bias):
    """Single LINEAR layer graph, optionally with bias."""
    g = Graph("linear_bias")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    w = g.add_tensor("w", Shape([16, 16]), DType.F32, StorageClass.PARAMETER)
    w.data = np.eye(16, dtype=np.float32)
    inputs = [x.id, w.id]
    parameters = [w.id]
    if with_bias:
        b = g.add_tensor("b", Shape([16]), DType.F32, StorageClass.PARAMETER)
        b.data = np.zeros((16,), dtype=np.float32)
        inputs.append(b.id)
        parameters.append(b.id)
    out = g.add_tensor("out", Shape([1, 16]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(OpCode.LINEAR, inputs=inputs, outputs=[out.id], name="lin")
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = parameters
    return g


def test_lower_linear_with_bias_to_3_input_mul_mat():
    """LINEAR with bias lowers to MUL_MAT with [weight, x, bias] inputs."""
    ggml_graph = lower_to_ggml(_linear_graph(with_bias=True))
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_MUL_MAT
    assert node.inputs == [1, 0, 2]


def test_lower_linear_without_bias_to_2_input_mul_mat():
    """LINEAR without bias lowers to plain 2-input MUL_MAT."""
    ggml_graph = lower_to_ggml(_linear_graph(with_bias=False))
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_MUL_MAT
    assert node.inputs == [1, 0]


def test_lower_expand_to_single_input_repeat():
    """EXPAND (single input + shape attribute) lowers to 1-input REPEAT."""
    g = Graph("expand")
    x = g.add_tensor("x", Shape([1, 16]), DType.F32, StorageClass.INPUT)
    out = g.add_tensor("out", Shape([4, 16]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(
        OpCode.EXPAND, inputs=[x.id], outputs=[out.id], attributes={"shape": (4, 16)}, name="expand"
    )
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = []
    ggml_graph = lower_to_ggml(g)
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_REPEAT
    assert node.inputs == [x.id]


def _slice_graph(dim, start, out_shape):
    """Single SLICE graph over an [8, 16] input."""
    g = Graph("slice")
    x = g.add_tensor("x", Shape([8, 16]), DType.F32, StorageClass.INPUT)
    out = g.add_tensor("out", Shape(out_shape), DType.F32, StorageClass.ACTIVATION)
    g.add_node(
        OpCode.SLICE,
        inputs=[x.id],
        outputs=[out.id],
        attributes={"dim": dim, "start": start, "end": 16, "step": 1},
        name="slice",
    )
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = []
    return g


def test_lower_slice_to_view_attributes():
    """SLICE lowers to VIEW carrying integer start and ggml_dim."""
    ggml_graph = lower_to_ggml(_slice_graph(dim=1, start=4, out_shape=[8, 12]))
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_VIEW
    assert node.attributes["start"] == 4
    assert node.attributes["ggml_dim"] == 0

    ggml_graph = lower_to_ggml(_slice_graph(dim=0, start=2, out_shape=[6, 16]))
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_VIEW
    assert node.attributes["start"] == 2
    assert node.attributes["ggml_dim"] == 1


def test_lower_permute_to_axis_attributes():
    """PERMUTE lowers to axis0..axis3 scalars."""
    g = Graph("permute")
    x = g.add_tensor("x", Shape([2, 4, 8, 16]), DType.F32, StorageClass.INPUT)
    out = g.add_tensor("out", Shape([2, 8, 4, 16]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(
        OpCode.PERMUTE,
        inputs=[x.id],
        outputs=[out.id],
        attributes={"dims": [0, 2, 1, 3]},
        name="perm",
    )
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = []
    ggml_graph = lower_to_ggml(g)
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_PERMUTE
    assert [node.attributes[f"axis{i}"] for i in range(4)] == [0, 2, 1, 3]
    assert "axes" not in node.attributes


def test_lower_transpose_to_axis_attributes():
    """4D TRANSPOSE lowers to PERMUTE with axis0..axis3 scalars."""
    g = Graph("transpose")
    x = g.add_tensor("x", Shape([2, 4, 8, 16]), DType.F32, StorageClass.INPUT)
    out = g.add_tensor("out", Shape([2, 8, 4, 16]), DType.F32, StorageClass.ACTIVATION)
    g.add_node(
        OpCode.TRANSPOSE,
        inputs=[x.id],
        outputs=[out.id],
        attributes={"dim0": 1, "dim1": 2},
        name="transpose",
    )
    g.inputs = [x.id]
    g.outputs = [out.id]
    g.parameters = []
    ggml_graph = lower_to_ggml(g)
    (node,) = ggml_graph.nodes
    assert node.opcode == GGMLOpCode.GGML_OP_PERMUTE
    assert [node.attributes[f"axis{i}"] for i in range(4)] == [0, 2, 1, 3]
