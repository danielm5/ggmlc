"""Custom torch ops that map 1:1 onto Canonical IR / GGML fused kernels.

Keep these graphs opaque at ``torch.export`` time so Python loops inside the
reference implementation never leak into FX.
"""

from __future__ import annotations

import torch

_GDN_OP = "ggmlc::gated_delta_net"


def _ensure_gate_rank(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(-1) if t.ndim == 3 else t


def _gated_delta_net_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Recurrent gated delta rule (HF / FLA). ``q,k`` are already L2-normalised.

    Layouts (Canonical / PyTorch):
      q, k, v : [B, S, H, D]
      g, beta : [B, S, H, 1]  (scalar gate; KDA uses [B, S, H, D])
      state   : [B, H, D, D]  (s0)
    GGML applies ``q *= 1/sqrt(D)`` inside the kernel; this reference does too.
    """
    g = _ensure_gate_rank(g)
    beta = _ensure_gate_rank(beta)
    scale = q.shape[-1] ** -0.5
    qh = q.transpose(1, 2).to(dtype=torch.float32) * scale
    kh = k.transpose(1, 2).to(dtype=torch.float32)
    vh = v.transpose(1, 2).to(dtype=torch.float32)
    gh = g.transpose(1, 2).to(dtype=torch.float32)
    bh = beta.transpose(1, 2).to(dtype=torch.float32)
    rec = state.to(dtype=torch.float32)
    seq = vh.shape[2]
    out = vh.new_zeros(vh.shape)
    for i in range(seq):
        decay = gh[:, :, i, 0].exp().unsqueeze(-1).unsqueeze(-1)
        rec = rec * decay
        kt = kh[:, :, i]
        vt = vh[:, :, i]
        bt = bh[:, :, i]
        kv_mem = (rec * kt.unsqueeze(-1)).sum(dim=-2)
        delta = (vt - kv_mem) * bt
        rec = rec + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (rec * qh[:, :, i].unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).to(dtype=v.dtype)


@torch.library.custom_op(_GDN_OP, mutates_args=())
def gated_delta_net(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Fused Gated DeltaNet. Lowered to ``GGML_OP_GATED_DELTA_NET``."""
    return _gated_delta_net_ref(q, k, v, g, beta, state)


@gated_delta_net.register_fake
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(v)
