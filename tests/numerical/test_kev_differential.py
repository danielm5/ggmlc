"""Differential numerical parity for Kev DecisionModel vs clean trunk vs compiled GGUF."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "laya"))

kev_available = False
try:
    from kev.api import Choice, Noul, SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions

    kev_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not kev_available, reason="kev package not installed")

EMAIL = SystemOneRequest(
    state={
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": (
            "Hi, we were billed twice for March. Please refund the duplicate "
            "today or we will cancel our plan."
        ),
    },
    questions={
        "category": Choice(
            type="choice",
            instructions="Which department should handle this request?",
            criteria={
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
                "other": "everything else",
            },
        ),
        "refund": Noul(type="noul", instructions="Does the customer ask for money back?"),
    },
)


def _gguf_logits(gguf: Path, arrays, enable_arena_reuse=True, device: str = "cpu"):
    from ggmlc.runtime.runner import ModelRunner

    runner = ModelRunner(str(gguf), device=device, n_threads=4)
    ggml = runner(
        *[t.detach().cpu().numpy() if hasattr(t, "detach") else t for t in arrays],
        n_threads=4,
        enable_arena_reuse=enable_arena_reuse,
    )
    if isinstance(ggml, dict):
        outs = []
        seen = set()
        for tid in runner.outputs:
            t = runner.tensor_info.get(tid)
            key = t.name if t is not None and t.name in ggml else tid
            ident = id(ggml[key]) if key in ggml else id(ggml.get(tid))
            if ident in seen:
                continue
            seen.add(ident)
            outs.append(ggml[key] if key in ggml else ggml[tid])
    elif isinstance(ggml, (list, tuple)):
        outs = list(ggml)
    else:
        outs = [ggml]
    return np.asarray(outs[0], dtype=np.float32), runner


def _load_pair(repo: str):
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    from kev_trunk import KevCleanTrunk, pad_batch, rows_from_encoding

    ck = Checkpoint(repo)
    tok, model = ck.load("cpu", LoadOptions(dtype=torch.float32, merge=True))
    model.eval()
    trunk = KevCleanTrunk(model.lm, model.head).eval()
    return tok, model, trunk, pad_batch, rows_from_encoding


def _email_arrays(tok, model, pad_batch, rows_from_encoding, seq_len: int):
    rec, _meta = to_record(EMAIL)
    enc = model.encode(tok, rec)
    rows = rows_from_encoding(enc)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    live = max(len(r["ids"]) for r in rows)
    tgt = max(int(seq_len), live)
    arrays = pad_batch(rows, pad_id=int(pad_id), seq_len=tgt)
    k = [len(r["opts"]) for r in rows]
    return enc, arrays, k, rows


@pytest.mark.parametrize(
    "repo,gguf_name,hybrid",
    [
        ("jaredpalmer/kev-0.5b", "kev_0.5b_f16.gguf", False),
        ("jaredpalmer/kev-0.8b", "kev_0.8b_f16.gguf", True),
    ],
)
def test_kev_trunk_matches_official_rows(repo, gguf_name, hybrid):
    tok, model, trunk, pad_batch, rows_from_encoding = _load_pair(repo)
    assert bool(getattr(model, "hybrid", False)) is hybrid
    enc, arrays, ks, _rows = _email_arrays(tok, model, pad_batch, rows_from_encoding, 128)
    with torch.no_grad():
        official = model.forward(enc)
        tl = trunk(*arrays)
    for i, z in enumerate(official):
        k = ks[i]
        live = z.float().reshape(-1)[:k]
        pred = tl[i, :k].float()
        max_diff = (live - pred).abs().max().item()
        cos = F.cosine_similarity(live.flatten(), pred.flatten(), dim=0).item()
        assert max_diff < 5e-3, f"{repo} q{i} max_diff={max_diff}"
        assert cos > 0.999, f"{repo} q{i} cosine {cos}"


@pytest.mark.parametrize(
    "repo,gguf_name",
    [
        ("jaredpalmer/kev-0.5b", "kev_0.5b_f16.gguf"),
        ("jaredpalmer/kev-0.8b", "kev_0.8b_f16.gguf"),
    ],
)
def test_kev_gguf_f16_parity(repo, gguf_name):
    gguf = ROOT / "scratch" / gguf_name
    if not gguf.exists():
        pytest.skip(f"{gguf} not found")
    tok, model, trunk, pad_batch, rows_from_encoding = _load_pair(repo)
    _enc, arrays, ks, _rows = _email_arrays(tok, model, pad_batch, rows_from_encoding, 128)
    with torch.no_grad():
        tl = trunk(*arrays).float().numpy()
    gl, _ = _gguf_logits(gguf, arrays, device="cuda" if torch.cuda.is_available() else "cpu")
    gl = np.asarray(gl, dtype=np.float32)
    if gl.ndim == 1:
        gl = gl.reshape(tl.shape[0], -1)
    for i, k in enumerate(ks):
        a = tl[i, :k]
        b = gl[i, :k]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        max_diff = float(np.max(np.abs(a - b)))
        assert cos > 0.99, f"{gguf_name} q{i} cosine {cos} max_diff={max_diff}"
        assert max_diff < 0.15, f"{gguf_name} q{i} max_diff={max_diff}"
