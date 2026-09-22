"""File-backed tensor payloads so quantization does not hold a second full copy."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Private copies smaller than this stay in memory. Larger ones go to a temp file.
SPILL_MIN_BYTES = 1 << 20
_CHUNK_BYTES = 8 << 20

SPILL_FILES: list[Path] = []


class SpilledBytes:
    """Length-aware handle for a tensor payload stored in a temporary file."""

    def __init__(self, path: Path, size: int) -> None:
        self.path = path
        self.size = size

    def __len__(self) -> int:
        return self.size

    def write_to(self, out: Any, chunk: int = _CHUNK_BYTES) -> None:
        with open(self.path, "rb") as src:
            while True:
                buf = src.read(chunk)
                if not buf:
                    return
                out.write(buf)


def cleanup_spills() -> None:
    """Delete quantization spill files created during this process."""
    for path in SPILL_FILES:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    SPILL_FILES.clear()


def _chunk_elements(arr: np.ndarray) -> int:
    elems = max(32, _CHUNK_BYTES // max(int(arr.dtype.itemsize), 1))
    return elems - (elems % 32)


def _iter_chunks(arr: np.ndarray, chunk_elems: int):
    if arr.flags.c_contiguous:
        flat = arr.reshape(-1)
        for start in range(0, int(flat.size), chunk_elems):
            yield flat[start : start + chunk_elems]
        return
    rows = arr.reshape(int(arr.shape[0]), -1)
    row_elems = int(rows.shape[1])
    rows_per = max(1, chunk_elems // max(row_elems, 1))
    if row_elems % 32 != 0:
        rows_per = 1
    for start in range(0, int(rows.shape[0]), rows_per):
        piece = np.ascontiguousarray(rows[start : start + rows_per])
        yield piece.reshape(-1)


def _encode_chunk(piece: np.ndarray, kind: str) -> bytes:
    if kind == "f16":
        return np.ascontiguousarray(piece, dtype=np.float16).tobytes()
    if kind == "f32":
        return np.ascontiguousarray(piece, dtype=np.float32).tobytes()
    from ggmlc.quantization.quantize import quantize_q4_0, quantize_q8_0

    if kind == "q8_0":
        return quantize_q8_0(piece)
    if kind == "q4_0":
        return quantize_q4_0(piece)
    raise ValueError(f"unsupported spill kind {kind}")


def payload_from_float32(
    arr: np.ndarray, kind: str, spill_min: int = SPILL_MIN_BYTES
) -> bytes | SpilledBytes:
    """Encode a float32 weight, spilling to disk when the source is large."""
    if int(arr.nbytes) < spill_min:
        if arr.flags.c_contiguous:
            return _encode_chunk(arr.reshape(-1), kind)
        return _encode_chunk(np.ascontiguousarray(arr).reshape(-1), kind)

    fd, name = tempfile.mkstemp(prefix="ggmlc-q-", suffix=".bin")
    os.close(fd)
    path = Path(name)
    SPILL_FILES.append(path)
    total = 0
    with open(path, "wb") as dst:
        for piece in _iter_chunks(arr, _chunk_elements(arr)):
            raw = _encode_chunk(piece, kind)
            dst.write(raw)
            total += len(raw)
    return SpilledBytes(path, total)
