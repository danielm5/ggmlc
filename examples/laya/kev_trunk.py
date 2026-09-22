"""Exportable Kev DecisionModel trunk: merged Qwen2.5 + pointer head.

LoRA is merged into the base Linear weights *before* this module is built
(`PeftModel.merge_and_unload` in fp32). The GGUF therefore contains a single
dense backbone — no adapter tensors, no extra GEMMs at inference.

Row form (state + one question per batch row) is a standard causal LM plus
GET_ROWS pointer readout, which is what Qwen3.5 serving uses and what
attention-only Kev agrees with packed block-causal to ~1e-6.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from ggmlc.ops import gated_delta_net
from torch import nn

MAX_OPTS = 16
MAX_LEN = 2048  # kev packed cap; row form is state+branch <= 384+1024


def _inv_freq(lm: nn.Module, head_dim: int) -> torch.Tensor:
    rotary = getattr(lm, "rotary_emb", None)
    if rotary is not None and getattr(rotary, "inv_freq", None) is not None:
        return rotary.inv_freq.detach().to(dtype=torch.float32).cpu().clone()
    cfg = lm.config
    theta = float(getattr(cfg, "rope_theta", 1_000_000.0))
    dim = head_dim
    params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
    if isinstance(params, dict):
        theta = float(params.get("rope_theta", theta))
        pr = float(params.get("partial_rotary_factor", 1.0))
        dim = max(2, int(head_dim * pr))
        if dim % 2:
            dim -= 1
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # Stable softplus; do not use F.softplus (importer can constant-fold it to ones).
    return F.relu(x) + torch.log(1.0 + torch.exp(-torch.abs(x)))


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def count_lora_tensors(module: nn.Module) -> list[str]:
    """Parameter names that still look like a PEFT adapter (must be empty after merge)."""
    names: list[str] = []
    for name, _ in module.named_parameters():
        if "lora_" in name.lower():
            names.append(name)
    return names


class KevCleanTrunk(nn.Module):
    """Causal Qwen2/2.5 encoder + pointer head, no lm_head, no LoRA.

    Inputs (dynamic batch ``b`` and sequence ``s``, option axis static 16):
      input_ids      [B, S] int32
      attention_mask [B, S] float32   (1 = token, 0 = pad)
      opt_pos        [B, 16] int32    (index into flattened [B*S, H]; unused 0)
      decide_pos     [B]     int32    (index of <|fim_suffix|> / decide token)
      opt_mask       [B, 16] float32  (1 = live option, 0 = pad)

    Host code must bake batch offsets into the position tensors:
    ``opt_pos[b, k] = b * S + token_index`` (same for ``decide_pos``).

    Outputs:
      logits         [B, 16] float32  (padded slots ~ -1e4; temperature already baked)
    """

    def __init__(
        self, lm: nn.Module, head: nn.Module, max_opts: int = MAX_OPTS, max_len: int = MAX_LEN
    ):
        super().__init__()
        leftover = count_lora_tensors(lm)
        if leftover:
            preview = ", ".join(leftover[:8])
            raise ValueError(
                f"refusing to wrap an unmerged LoRA adapter ({len(leftover)} tensors; e.g. {preview})"
            )
        cfg = lm.config
        self.hidden = int(cfg.hidden_size)
        self.num_heads = int(cfg.num_attention_heads)
        self.num_kv_heads = int(getattr(cfg, "num_key_value_heads", self.num_heads))
        self.head_dim = int(getattr(cfg, "head_dim", None) or self.hidden // self.num_heads)
        self.max_opts = int(max_opts)
        self.max_len = int(max_len)
        self.buf_len = int(max_len) + 1
        self.enable_gqa = self.num_heads != self.num_kv_heads

        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.head_q = head.q
        self.head_k = head.k
        temperature = float(getattr(head, "temperature", 1.0) or 1.0)
        scale = float(getattr(head, "scale", 1.0 / math.sqrt(head.q.out_features))) / temperature
        self.register_buffer("ptr_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("inv_freq", _inv_freq(lm, self.head_dim))
        self.rotary_dim = int(self.inv_freq.numel() * 2)

        # Extra slot so slicing [:s] is never a no-op at s=max_len (torch.export guard).
        causal = torch.triu(torch.ones(self.buf_len, self.buf_len), diagonal=1) * -1.0e4
        self.register_buffer("causal_bias", causal.view(1, 1, self.buf_len, self.buf_len))

        types = getattr(cfg, "layer_types", None)
        self.layer_types: list[str] = (
            list(types) if types else ["full_attention"] * len(self.layers)
        )
        if len(self.layer_types) < len(self.layers):
            self.layer_types.extend(["full_attention"] * (len(self.layers) - len(self.layer_types)))
        self.hybrid = any(t == "linear_attention" for t in self.layer_types)

        attn0 = None
        for i, layer in enumerate(self.layers):
            if self.layer_types[i] != "linear_attention" and hasattr(layer, "self_attn"):
                attn0 = layer.self_attn
                break
        if attn0 is None:
            attn0 = getattr(self.layers[0], "self_attn", None)
        self.has_q_norm = bool(
            attn0 is not None
            and hasattr(attn0, "q_norm")
            and not isinstance(attn0.q_norm, nn.Identity)
        )
        self.has_k_norm = bool(
            attn0 is not None
            and hasattr(attn0, "k_norm")
            and not isinstance(attn0.k_norm, nn.Identity)
        )
        q_out = (
            int(attn0.q_proj.out_features) if attn0 is not None else self.num_heads * self.head_dim
        )
        self.q_gated = q_out == self.num_heads * self.head_dim * 2

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, dtype=torch.float32, device=q.device)
        freqs = pos.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(1)
        sin = emb.sin().unsqueeze(0).unsqueeze(1)
        rd = int(self.rotary_dim)
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        half = rd // 2

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)
        if q_pass.shape[-1] == 0:
            return q_emb, k_emb
        return torch.cat((q_emb, q_pass), dim=-1), torch.cat((k_emb, k_pass), dim=-1)

    def _full_attn(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        _b: int,
        s: int,
    ) -> torch.Tensor:
        if self.q_gated:
            qg = attn.q_proj(x).unflatten(-1, (self.num_heads, 2 * self.head_dim))
            q, gate = qg.split(self.head_dim, dim=-1)
            gate = torch.sigmoid(gate.flatten(-2, -1))
        else:
            q = attn.q_proj(x).unflatten(-1, (self.num_heads, self.head_dim))
            gate = None
        k = attn.k_proj(x).unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = attn.v_proj(x).unflatten(-1, (self.num_kv_heads, self.head_dim))
        if self.has_q_norm:
            q = attn.q_norm(q)
        if self.has_k_norm:
            k = attn.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self._apply_rope(q, k, s)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
            enable_gqa=self.enable_gqa,
        )
        out = out.transpose(1, 2).flatten(-2, -1)
        if gate is not None:
            out = out * gate
        return attn.o_proj(out)

    def _linear_attn(
        self, mixer: nn.Module, x: torch.Tensor, attention_mask: torch.Tensor, _b: int, s: int
    ) -> torch.Tensor:
        x = x * attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        mixed = mixer.in_proj_qkv(x).transpose(1, 2)
        conv = mixer.conv1d(mixed)[:, :, :s]
        mixed = F.silu(conv).transpose(1, 2)
        query, key, value = torch.split(
            mixed, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1
        )
        query = query.unflatten(-1, (mixer.num_k_heads, mixer.head_k_dim))
        key = key.unflatten(-1, (mixer.num_k_heads, mixer.head_k_dim))
        value = value.unflatten(-1, (mixer.num_v_heads, mixer.head_v_dim))
        n_rep = mixer.num_v_heads // mixer.num_k_heads
        if n_rep > 1:
            # Expand on a new axis so example batch (often 2) is not unified with n_rep.
            query = query.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).flatten(2, 3)
            key = key.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).flatten(2, 3)
        beta = torch.sigmoid(mixer.in_proj_b(x)).unsqueeze(-1)
        a = mixer.in_proj_a(x)
        g = (-mixer.A_log.float().exp() * _softplus(a.float() + mixer.dt_bias)).unsqueeze(-1)
        query = _l2norm(query)
        key = _l2norm(key)
        # Export-friendly zeros: ``new_zeros(b, ...)`` is constant-folded to the example batch.
        q0 = query[:, :1].transpose(1, 2)
        v0 = value[:, :1].transpose(1, 2)
        state = (q0.transpose(-1, -2) @ v0) * 0
        core = gated_delta_net(query, key, value, g.to(dtype=query.dtype), beta, state)
        z = mixer.in_proj_z(x).unflatten(-1, (mixer.num_v_heads, mixer.head_v_dim))
        core = mixer.norm(core, z)
        return mixer.out_proj(core.flatten(-2, -1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        opt_pos: torch.Tensor,
        decide_pos: torch.Tensor,
        opt_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, s = input_ids.shape
        h = self.embed_tokens(input_ids.to(dtype=torch.long))
        pad_bias = (attention_mask.to(dtype=h.dtype) - 1.0) * 1.0e4
        pad_bias = pad_bias[:, None, None, :]
        full_zeros = attention_mask[:, None, :, None] * (attention_mask[:, None, None, :] * 0.0)
        attn_bias = full_zeros.to(dtype=h.dtype) + pad_bias + self.causal_bias[:, :, :s, :s]

        for i, layer in enumerate(self.layers):
            residual = h
            x = layer.input_layernorm(h)
            if self.layer_types[i] == "linear_attention":
                h = residual + self._linear_attn(layer.linear_attn, x, attention_mask, b, s)
            else:
                h = residual + self._full_attn(layer.self_attn, x, attn_bias, b, s)

            residual = h
            x = layer.post_attention_layernorm(h)
            h = residual + layer.mlp.down_proj(
                F.silu(layer.mlp.gate_proj(x)) * layer.mlp.up_proj(x)
            )

        h = self.norm(h)
        flat = h.flatten(0, 1)
        # GET_ROWS via embedding — not aten.gather (importer lowers that to a prefix SLICE).
        h_opts = F.embedding(opt_pos.reshape(-1).to(dtype=torch.long), flat).unflatten(
            0, (input_ids.shape[0], self.max_opts)
        )
        h_decide = F.embedding(decide_pos.reshape(-1).to(dtype=torch.long), flat)
        logits = (self.head_k(h_opts) * self.head_q(h_decide).unsqueeze(1)).sum(
            dim=-1
        ) * self.ptr_scale
        return logits + (1.0 - opt_mask) * (-1.0e4)


def flatten_index(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Bake ``b * S`` into GET_ROWS indices. Host-side only (CUDA binbcast is F32/F16)."""
    if pos.ndim == 1:
        pos = pos.unsqueeze(-1)
        squeeze = True
    else:
        squeeze = False
    b = int(pos.shape[0])
    if b > 1:
        off = torch.arange(b, dtype=pos.dtype, device=pos.device).unsqueeze(1) * int(seq_len)
        pos = pos + off
    return pos.reshape(b) if squeeze else pos


def pad_batch(
    rows: list[dict[str, Any]],
    pad_id: int,
    seq_len: int,
    max_opts: int = MAX_OPTS,
) -> tuple[torch.Tensor, ...]:
    """Pad Kev row-form questions to ``[B, seq_len]`` / ``[B, 16]`` and flatten markers."""
    b = len(rows)
    ids = torch.full((b, seq_len), int(pad_id), dtype=torch.int32)
    att = torch.zeros((b, seq_len), dtype=torch.float32)
    opt_pos = torch.zeros((b, max_opts), dtype=torch.int32)
    opt_mask = torch.zeros((b, max_opts), dtype=torch.float32)
    decide_pos = torch.zeros((b,), dtype=torch.int32)
    for i, row in enumerate(rows):
        toks = row["ids"]
        n = len(toks)
        if n > seq_len:
            raise ValueError(f"row length {n} exceeds seq_len {seq_len}")
        ids[i, :n] = torch.tensor(toks, dtype=torch.int32)
        att[i, :n] = 1.0
        decide_pos[i] = int(row["decide"])
        opts = list(row["opts"])
        if len(opts) > max_opts:
            raise ValueError(f"{len(opts)} options exceeds max_opts={max_opts}")
        opt_pos[i, : len(opts)] = torch.tensor(opts, dtype=torch.int32)
        opt_mask[i, : len(opts)] = 1.0
    opt_pos = flatten_index(opt_pos, seq_len)
    decide_pos = flatten_index(decide_pos, seq_len)
    return ids, att, opt_pos, decide_pos, opt_mask


def rows_from_encoding(enc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a packed Kev encoding into independent causal rows (state + one question)."""
    from kev.model import rows_of

    state_ids, _state_pos, branches = rows_of(enc)
    rows = []
    for br in branches:
        rows.append(
            {
                "ids": list(state_ids) + list(br["ids"]),
                "decide": len(state_ids) + int(br["decide"]),
                "opts": [len(state_ids) + int(o) for o in br["opts"]],
            }
        )
    return rows
