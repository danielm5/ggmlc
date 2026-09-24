"""Compile fredzzp/plaidq-0.7b-16step to a GGUF the tab_completion binary can run.

The C++ engine calls a static canvas of shape [1, 256, 16] with inputs
``z``, ``gamma``, and ``x_selfcond``, reads one logits tensor, and looks up
a GGUF tensor named ``embedding_matrix`` plus ``plaidq.*`` metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import plaidq.hub
import plaidq.qwen3_trunk as qt
import torch
import torch.nn.functional as F
from ggmlc import compile as ggmlc_compile
from ggmlc.pipeline.tokenizer import BPETokenizer
from plaidq.sample import (
    _build_modules,
    _load_module_states,
    checkpoint_module_states,
    validate_checkpoint_metadata,
)

CHECKPOINT = "hf://fredzzp/plaidq-0.7b-16step"
TOKENIZER = "Qwen/Qwen3-0.6B"
CANVAS = 256
QUANT_CHOICES = ["f32", "f16", "q8_0", "q4_0", "q4_k_m", "ud_q4_k_m"]


def _patch_for_export() -> None:
    """Make the Qwen3 trunk traceable. Same patches as the mini differential test."""
    torch.amp.autocast = lambda *args, **kwargs: contextlib.nullcontext()

    def custom_rotary_forward(self, seq_len: int, device, dtype, position_ids=None):
        if position_ids is not None:
            t = position_ids.to(device=device, dtype=torch.float32)
            freqs = t.unsqueeze(-1) * self.inv_freq.to(device).unsqueeze(0).unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos()[:, :, None, :].to(dtype), emb.sin()[:, :, None, :].to(dtype)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t.unsqueeze(-1) * self.inv_freq.to(device).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :].to(dtype), emb.sin()[None, :, None, :].to(dtype)

    qt.Qwen3Rotary.forward = custom_rotary_forward

    def custom_forward(self, x, cos, sin, cu_seqlens=None, cond=None, is_causal=False):
        b, s = x.shape[0], x.shape[1]
        shift_a, scale_a, shift_m, scale_m = self._cond_chunks(x, cond)
        q, k, v = self._qkv(x, cos, sin, shift_a, scale_a)
        attn_t = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=is_causal,
            enable_gqa=True,
        )
        attn = attn_t.transpose(1, 2).reshape(b, s, self.n_heads * self.head_dim)
        return self._attn_out(x, attn, shift_m, scale_m)

    qt.Qwen3DiffusionBlock.forward = custom_forward


class Denoiser(torch.nn.Module):
    """Logits-only wrapper. Host-side DDIM rebuilds the latent from the codebook."""

    def __init__(self, model: torch.nn.Module, embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("embedding_matrix", embedding_matrix)

    def forward(
        self, z: torch.Tensor, gamma: torch.Tensor, x_selfcond: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = self.model(
            z=z,
            gamma=gamma,
            embedding_matrix=self.embedding_matrix,
            x_selfcond=x_selfcond,
            return_logits=True,
            return_reconst=False,
            clean_prefix_mask=None,
            clean_prefix_embeddings=None,
        )
        return logits


def load_denoiser() -> tuple[Denoiser, dict]:
    _patch_for_export()
    print(f"loading {CHECKPOINT} …")
    payload = plaidq.hub.load_checkpoint_payload(CHECKPOINT, map_location=torch.device("cpu"))
    # The sampler refuses a prefix-conditioned checkpoint without a prompt.
    # Prefix tokens are applied on the host by the C++ engine, not by this graph.
    metadata = validate_checkpoint_metadata(payload, requested_qwen3_size=None, prompt="")
    modules = _build_modules(metadata, torch.device("cpu"))
    _load_module_states(modules, checkpoint_module_states(payload))
    model = modules["model"].float().eval()
    with torch.no_grad():
        codebook = modules["embedding_matrix"]().detach().float().contiguous()
        gamma_0 = float(modules["gamma_bounds"].gamma_0.detach())
        gamma_1 = float(modules["gamma_bounds"].gamma_1.detach())
    schedule_t = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    with torch.no_grad():
        schedule_g = (
            modules["noise_schedule"](torch.tensor(schedule_t, dtype=torch.float64))
            .detach()
            .double()
            .cpu()
            .numpy()
        )
    print(
        "qwen3",
        metadata.qwen3_size,
        "embed",
        metadata.embed_dim,
        "codebook",
        tuple(codebook.shape),
        "gamma",
        gamma_0,
        gamma_1,
        "schedule",
        float(schedule_g[0]),
        float(schedule_g[len(schedule_g) // 2]),
        float(schedule_g[-1]),
    )
    wrapper = Denoiser(model, codebook).eval()
    meta = {
        "plaidq.vocab_size": int(codebook.shape[0]),
        "plaidq.embed_dim": int(codebook.shape[1]),
        "plaidq.gamma_0": gamma_0,
        "plaidq.gamma_1": gamma_1,
        "plaidq.canvas_len": CANVAS,
        "plaidq.checkpoint": "fredzzp/plaidq-0.7b-16step",
        "plaidq.schedule_t": [float(x) for x in schedule_t],
        "plaidq.schedule_g": [float(x) for x in schedule_g],
    }
    del payload, modules
    return wrapper, meta


def compile_one(quantize: str, output: Path | None) -> Path:
    wrapper, meta = load_denoiser()
    z = torch.zeros(1, CANVAS, meta["plaidq.embed_dim"], dtype=torch.float32)
    gamma = torch.zeros(1, dtype=torch.float32)
    x_selfcond = torch.zeros_like(z)
    with torch.no_grad():
        logits = wrapper(z, gamma, x_selfcond)
    print("example logits", tuple(logits.shape))

    if output is None:
        output = ROOT / "scratch" / f"plaidq_0.7b_16step_{quantize.replace('-', '_')}.gguf"
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer.from_huggingface(TOKENIZER, context_length=CANVAS)

    print(f"compiling -> {output} quantize={quantize}")
    quant_arg = None if quantize in ("f32", "none") else quantize
    out_path = ggmlc_compile(
        wrapper,
        (z, gamma, x_selfcond),
        output=output,
        model_name="plaidq-0.7b-16step",
        quantize=quant_arg,
        pipeline=tokenizer,
        tasks=["completion"],
        extra_metadata=meta,
    )
    print("wrote", out_path, "bytes", Path(out_path).stat().st_size)
    return Path(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile PlaidQ 0.7B 16-step to GGUF.")
    parser.add_argument("--quantize", default="q4_0", choices=QUANT_CHOICES)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    compile_one(args.quantize, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
