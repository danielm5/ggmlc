"""Compile Google TimesFM 3.0 (google/timesfm-3.0-pytorch) to GGUF."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import ggmlc
import torch
import torch.nn.functional as F
from ggmlc.transforms.fusion import FusionOptions
from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

CHECKPOINT = "google/timesfm-3.0-pytorch"
QUANT_CHOICES = ["f32", "f16", "q8_0", "q4_0", "q4_k_m", "ud_q4_k_m"]


class TimesFM3CleanTrunk(torch.nn.Module):
    """Exportable trunk: resblock, mixing stack, quantile head.

    Matches the C++ forecaster, which feeds patches of shape [B, 1, N, 192]
    and reads the last patch of a [B, 1, N, 576] quantile head.
    """

    def __init__(self, base_model: torch.nn.Module) -> None:
        super().__init__()
        self.pre_transformer_resblock = base_model.pre_transformer_resblock
        self.layers = base_model.transformer_stack.layers
        self.output_head = base_model.output_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_transformer_resblock(x)
        b, v, n, d = h.shape

        for layer in self.layers:
            residual = h
            h_seq = layer.pre_seq_attn_ln(h)
            h_seq_flat = h_seq.view(b * v, n, d)

            q = layer.seq_attn.query_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )
            k = layer.seq_attn.key_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )
            val = layer.seq_attn.value_proj(h_seq_flat).view(
                b * v, n, layer.seq_attn.num_heads, layer.seq_attn.head_dim
            )

            pos = torch.arange(n, device=x.device, dtype=torch.float32).unsqueeze(0)
            q = layer.seq_attn.rotary_position_embedding(q, pos)
            k = layer.seq_attn.rotary_position_embedding(k, pos)

            if layer.seq_attn.query_ln is not None:
                q = layer.seq_attn.query_ln(q)
            if layer.seq_attn.key_ln is not None:
                k = layer.seq_attn.key_ln(k)
            if layer.seq_attn.per_dim_scale is not None:
                q = layer.seq_attn.per_dim_scale(q)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            val = val.transpose(1, 2)

            scale = math.sqrt(layer.seq_attn.head_dim)
            attn_out = F.scaled_dot_product_attention(q, k, val, is_causal=True, scale=scale)
            attn_out = attn_out.transpose(1, 2).reshape(b * v, n, d)
            attn_out = layer.seq_attn.out_proj(attn_out).view(b, v, n, d)

            if layer.post_seq_attn_ln is not None:
                attn_out = layer.post_seq_attn_ln(attn_out)

            h = residual + attn_out

            if layer.var_attn is not None:
                residual = h
                h_var = layer.pre_var_attn_ln(h) if layer.pre_var_attn_ln is not None else h
                h_var_flat = h_var.permute(0, 2, 1, 3).contiguous().view(b * n, v, d)

                vq = layer.var_attn.query_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )
                vk = layer.var_attn.key_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )
                vval = layer.var_attn.value_proj(h_var_flat).view(
                    b * n, v, layer.var_attn.num_heads, layer.var_attn.head_dim
                )

                if layer.var_attn.query_ln is not None:
                    vq = layer.var_attn.query_ln(vq)
                if layer.var_attn.key_ln is not None:
                    vk = layer.var_attn.key_ln(vk)
                if layer.var_attn.per_dim_scale is not None:
                    vq = layer.var_attn.per_dim_scale(vq)

                vq = vq.transpose(1, 2)
                vk = vk.transpose(1, 2)
                vval = vval.transpose(1, 2)

                var_scale = math.sqrt(layer.var_attn.head_dim)
                var_out = F.scaled_dot_product_attention(
                    vq, vk, vval, is_causal=False, scale=var_scale
                )
                var_out = var_out.transpose(1, 2).reshape(b * n, v, d)
                var_out = (
                    layer.var_attn.out_proj(var_out)
                    .view(b, n, v, d)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

                if layer.post_var_attn_ln is not None:
                    var_out = layer.post_var_attn_ln(var_out)

                h = residual + var_out

            residual = h
            h_ff = layer.pre_ff_ln(h)
            mlp_out = layer.ff1(F.relu(layer.ff0(h_ff)))
            if layer.post_ff_ln is not None:
                mlp_out = layer.post_ff_ln(mlp_out)
            h = residual + mlp_out

        return self.output_head(h)


def compile_one(quantize: str, output: Path | None, max_batch: int, max_patches: int) -> Path:
    print(f"loading {CHECKPOINT} …")
    forecaster = TimesFM3Forecaster.from_pretrained(CHECKPOINT)
    model = forecaster.model.cpu().float().eval()
    trunk = TimesFM3CleanTrunk(model).eval()

    # Batch example must not be 1. RoPE broadcasts a length-1 position axis,
    # and torch.export then freezes the batch symbol to that constant.
    example = torch.randn(2, 1, 16, 192, dtype=torch.float32)
    with torch.no_grad():
        out = trunk(example)
    print("example", tuple(example.shape), "->", tuple(out.shape))

    if output is None:
        output = ROOT / "scratch" / f"timesfm3_{quantize.replace('-', '_')}.gguf"
    output.parent.mkdir(parents=True, exist_ok=True)

    dim_b = torch.export.Dim("b", min=1, max=max_batch)
    dim_n = torch.export.Dim("n", min=1, max=max_patches)
    # Custom rotary lives inside the trunk. It is not GGML_OP_ROPE.
    fusion = FusionOptions()
    fusion.enable_rope = False
    print(f"compiling -> {output} quantize={quantize} b=1..{max_batch} n=1..{max_patches}")
    ggmlc.compile(
        trunk,
        (example,),
        output=str(output),
        model_name="timesfm3",
        quantize=quantize,
        fusion_options=fusion,
        tasks=["forecast"],
        extra_metadata={
            "timesfm.checkpoint": CHECKPOINT,
            "timesfm.context_length": 15360,
            "timesfm.patch_length": 32,
            "timesfm.feature_dim": 192,
        },
        dynamic_shapes=({0: dim_b, 2: dim_n},),
    )
    print("wrote", output, "bytes", output.stat().st_size)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile TimesFM 3.0 to GGUF.")
    parser.add_argument("--quantize", default="f16", choices=QUANT_CHOICES)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=512)
    args = parser.parse_args()
    compile_one(
        args.quantize,
        Path(args.output) if args.output else None,
        args.max_batch,
        args.max_patches,
    )


if __name__ == "__main__":
    main()
