"""Unused Module buffers lifted by torch.export must survive compile into GGUF."""

from __future__ import annotations

import io
import struct

import numpy as np
import torch
from ggmlc import compile as ggmlc_compile
from ggmlc.frontend.pytorch.importer import import_exported_program
from ggmlc.transforms import create_standard_optimization_pipeline
from torch import nn


class TinyWithSidecar(nn.Module):
    """Linear plus a codebook buffer never read by the compute graph."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer(
            "embedding_matrix", torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _gguf_tensor_names(path_or_bytes) -> set[str]:
    """Minimal GGUF v3 tensor-name scan (no full loader dependency)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        data = bytes(path_or_bytes)
    else:
        data = path_or_bytes.read_bytes()
    bio = io.BytesIO(data)
    magic = bio.read(4)
    assert magic == b"GGUF"
    _version = struct.unpack("<I", bio.read(4))[0]
    n_tensors, n_kv = struct.unpack("<QQ", bio.read(16))

    def read_string() -> str:
        (n,) = struct.unpack("<Q", bio.read(8))
        return bio.read(n).decode("utf-8")

    def skip_value(vtype: int) -> None:
        sizes = {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 8,
            10: 8,
            11: 1,
            12: 8,
        }
        if vtype == 8:  # string
            read_string()
            return
        if vtype == 9:  # array
            (etype,) = struct.unpack("<I", bio.read(4))
            (n,) = struct.unpack("<Q", bio.read(8))
            for _ in range(n):
                skip_value(etype)
            return
        bio.read(sizes[vtype])

    for _ in range(n_kv):
        read_string()
        (vtype,) = struct.unpack("<I", bio.read(4))
        skip_value(vtype)

    names: set[str] = set()
    for _ in range(n_tensors):
        names.add(read_string())
        n_dims = struct.unpack("<I", bio.read(4))[0]
        bio.read(8 * n_dims)
        bio.read(4)  # ggml type
        bio.read(8)  # offset
    return names


class TinyWithDeadBufferUsers(nn.Module):
    """Buffer is read in forward but the result never reaches the output.

    Mirrors PlaidQ: ``return_reconst=False`` still builds ``cat(E, E.detach())``
    then discards it, so FX keeps users while DCE would drop an unmarked buffer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer(
            "embedding_matrix", torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Touch the buffer so FX keeps users, but cancel the contribution.
        dead = self.embedding_matrix[0, 0] * 0.0
        return self.lin(x) + dead


def test_buffer_with_dead_fx_users_survives_dce_and_gguf():
    m = TinyWithDeadBufferUsers().eval()
    ep = torch.export.export(m, (torch.randn(1, 4),))
    emb_ph = next(n for n in ep.graph.nodes if n.name == "b_embedding_matrix")
    assert len(emb_ph.users) > 0

    g = import_exported_program(ep, graph_name="tiny_dead")
    emb = next(t for t in g.tensors.values() if t.name == "embedding_matrix")
    assert emb.export_required is True

    optimized = create_standard_optimization_pipeline().run(g).graph
    assert any(t.name == "embedding_matrix" for t in optimized.tensors.values())

    gguf = ggmlc_compile(m, (torch.zeros(1, 4),), model_name="tiny_dead_buf", quantize=None)
    assert "embedding_matrix" in _gguf_tensor_names(gguf)


def test_unused_embedding_matrix_marked_export_required():
    m = TinyWithSidecar().eval()
    ep = torch.export.export(m, (torch.randn(1, 4),))
    g = import_exported_program(ep, graph_name="tiny")
    emb = next(t for t in g.tensors.values() if t.name == "embedding_matrix")
    assert emb.export_required is True
    assert emb.storage.name == "CONSTANT"
    assert emb.id in g.parameters

    optimized = create_standard_optimization_pipeline().run(g).graph
    assert any(t.name == "embedding_matrix" for t in optimized.tensors.values())


def test_unused_embedding_matrix_survives_ggmlc_compile_gguf():
    m = TinyWithSidecar().eval()
    x = torch.zeros(1, 4)
    gguf = ggmlc_compile(m, (x,), model_name="tiny_sidecar", quantize=None)
    names = _gguf_tensor_names(gguf)
    assert "embedding_matrix" in names

    # Payload must match the registered buffer (F32, row-major).
    # Locate tensor data via gguf names already checked; use numpy round-trip
    # through a second compile-to-graph path for value check.
    from ggmlc.frontend.pytorch import export_torch_model

    g = export_torch_model(m, (x,), model_name="tiny_sidecar").main_graph
    emb = next(t for t in g.tensors.values() if t.name == "embedding_matrix")
    assert np.allclose(emb.data, np.arange(32, dtype=np.float32).reshape(8, 4))
