from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.export import export

from ggmlc.frontend.pytorch.importer import import_exported_program
from ggmlc.ir.model import Model
from ggmlc.transforms import create_standard_optimization_pipeline

_RELEASED_FILES: list[Path] = []


def cleanup_released_storage() -> None:
    """Delete weight memmaps created by ``release_module_storage``."""
    gc.collect()
    for path in _RELEASED_FILES:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    _RELEASED_FILES.clear()


def _owned_weight(array: np.ndarray) -> np.ndarray:
    """Copy a weight into memory the module does not own.

    Large copies go to a file-backed memmap so the extra buffer does not need
    another private allocation while the torch parameters are still resident.
    """
    if array.nbytes < 1 << 20:
        return np.array(array, copy=True, order="C")
    fd, name = tempfile.mkstemp(prefix="ggmlc-w-", suffix=".bin")
    os.close(fd)
    path = Path(name)
    _RELEASED_FILES.append(path)
    mapped = np.memmap(path, dtype=array.dtype, mode="w+", shape=tuple(int(n) for n in array.shape))
    if array.flags.c_contiguous:
        mapped[...] = array
    else:
        mapped[...] = np.ascontiguousarray(array)
    mapped.flush()
    return mapped


def _torch_owner(array: np.ndarray) -> torch.Tensor | None:
    """Return the torch tensor that owns this NumPy view, if any."""
    base: Any = array
    seen: set[int] = set()
    while base is not None and id(base) not in seen:
        seen.add(id(base))
        if isinstance(base, torch.Tensor):
            return base
        base = getattr(base, "base", None)
    return None


def _release_module_storage(model: torch.nn.Module, graph: Any) -> None:
    """Copy IR weights into owned arrays, then drop the live module storages.

    ``tensor.numpy()`` shares storage with the module. Fusion then needs another
    buffer for concatenated weights. Copying one parameter at a time and freeing
    that storage keeps the extra peak to a single tensor. The module is empty
    afterwards; callers must not run it again.
    """
    grouped: dict[int, list[tuple[Any, torch.Tensor | None]]] = {}
    for tensor in graph.tensors.values():
        data = getattr(tensor, "data", None)
        if not isinstance(data, np.ndarray) or data.nbytes == 0:
            continue
        owner = _torch_owner(data)
        if owner is None:
            continue
        grouped.setdefault(owner.data_ptr(), []).append((tensor, owner))

    held = set(grouped)
    for param in model.parameters():
        if param.numel() > 0 and param.data_ptr() not in held:
            param.data = torch.empty(0, dtype=param.dtype, device="cpu")
    for buf in model.buffers():
        if buf.numel() > 4096 and buf.data_ptr() not in held:
            buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")
    gc.collect()
    empty_host = getattr(torch._C, "_host_emptyCache", None)
    if empty_host is not None:
        empty_host()

    def _nbytes(items: list[tuple[Any, torch.Tensor | None]]) -> int:
        return sum(t.data.nbytes for t, _ in items)

    for ptr, items in sorted(grouped.items(), key=lambda kv: _nbytes(kv[1])):
        for tensor, _owner in items:
            tensor.data = _owned_weight(tensor.data)
        for _tensor, owner in items:
            if owner is not None and owner.numel() > 0 and owner.data_ptr() == ptr:
                owner.data = torch.empty(0, dtype=owner.dtype, device="cpu")
        for param in model.parameters():
            if param.numel() > 0 and param.data_ptr() == ptr:
                param.data = torch.empty(0, dtype=param.dtype, device="cpu")
        for buf in model.buffers():
            if buf.numel() > 0 and buf.data_ptr() == ptr:
                buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")
        gc.collect()
        if empty_host is not None:
            empty_host()


def export_torch_model(
    model: torch.nn.Module,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    model_name: str = "model",
    optimize: bool = True,
    enable_fusion: bool = True,
    fusion_options: Any | None = None,
    release_module_storage: bool = False,
) -> Model:
    """Exports a PyTorch model into a ggmlc Model containing Canonical IR graphs.

    Args:
        enable_fusion: When optimize=True, whether OperatorFusionPass runs.
        fusion_options: Optional FusionOptions (or dict) controlling fusion passes.
            Must be threaded from ``ggmlc.compile`` so A/B flags are not overwritten
            by a default-options export pass.
        release_module_storage: Copy IR weights out of the module and drop its
            parameter storage before fusion. The module cannot be run afterwards.
    """
    model.eval()
    ep = export(
        model,
        args=example_args,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )
    g = import_exported_program(ep, graph_name="main")
    if release_module_storage:
        del ep
        _release_module_storage(model, g)
    if optimize:
        pipeline = create_standard_optimization_pipeline(
            enable_fusion=enable_fusion, options=fusion_options
        )
        g = pipeline(g)
    m = Model(name=model_name)
    m.add_graph(g, is_main=True)
    return m
