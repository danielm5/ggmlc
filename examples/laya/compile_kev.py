"""Compile Kev checkpoints to GGUF.

LoRA is always merged into the base weights in fp32 before export
(`kev.checkpoint.LoadOptions(merge=True)` → `PeftModel.merge_and_unload`).
Unmerged adapters are refused: they would add extra GEMMs at inference and
do not match Kev's own serving path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "laya"))

import ggmlc
import torch
from ggmlc.pipeline.decision import DecisionPipelineSpec
from ggmlc.pipeline.tokenizer import BPETokenizer
from ggmlc.transforms.fusion import FusionOptions
from kev.api import Choice, Noul, SystemOneRequest, to_record
from kev.checkpoint import Checkpoint, LoadOptions
from kev.model import SPECIAL
from kev_trunk import (
    MAX_LEN,
    MAX_OPTS,
    KevCleanTrunk,
    count_lora_tensors,
    pad_batch,
    rows_from_encoding,
)

CHECKPOINTS = {
    "0.5b": {
        "repo": "jaredpalmer/kev-0.5b",
        "stem": "kev_0.5b",
        "model_name": "kev-0.5b",
        "arch": "qwen2.5",
    },
    "0.8b": {
        "repo": "jaredpalmer/kev-0.8b",
        "stem": "kev_0.8b",
        "model_name": "kev-0.8b",
        "arch": "qwen3.5",
    },
    "4b": {
        "repo": "jaredpalmer/kev-4b",
        "stem": "kev_4b",
        "model_name": "kev-4b",
        "arch": "qwen3.5",
    },
    "9b": {
        "repo": "jaredpalmer/kev-9b",
        "stem": "kev_9b",
        "model_name": "kev-9b",
        "arch": "qwen3.5",
    },
}
ALIASES = {
    "kev": "0.5b",
    "kev-0.5b": "0.5b",
    "0.5": "0.5b",
    "kev-0.8b": "0.8b",
    "0.8": "0.8b",
    "kev-4b": "4b",
    "4": "4b",
    "kev-9b": "9b",
    "9": "9b",
}
QUANT_CHOICES = ["f32", "f16", "q8_0", "q4_0", "q4_k_m", "ud_q4_k_m"]


def _normalize_family(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in CHECKPOINTS:
        raise SystemExit(f"unknown family {name!r}; choose {list(CHECKPOINTS)}")
    return key


def _length_buckets(max_len: int) -> list[int]:
    return [b for b in (64, 128, 256, 512, 1024, 2048) if b <= max_len] or [max_len]


def _example_rows(tok, model, seq_len: int) -> tuple[torch.Tensor, ...]:
    req = SystemOneRequest(
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
    rec, _meta = to_record(req)
    enc = model.encode(tok, rec)
    rows = rows_from_encoding(enc)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    live = max(len(r["ids"]) for r in rows)
    tgt = max(int(seq_len), live)
    return pad_batch(rows, pad_id=int(pad_id), seq_len=tgt)


def _load_merged(repo: str, device: str = "cpu"):
    """Load checkpoint, fold LoRA into base weights in fp32, refuse leftovers."""
    print(f"loading {repo} (fp32, merge LoRA) …")
    ck = Checkpoint(repo)
    tok, model = ck.load(device, LoadOptions(dtype=torch.float32, merge=True))
    model.eval()
    leftover = count_lora_tensors(model)
    if leftover:
        preview = ", ".join(leftover[:8])
        raise SystemExit(
            f"refusing to compile unmerged LoRA ({len(leftover)} tensors; e.g. {preview}). "
            "Kev GGUFs must be dense fused weights."
        )
    if type(model.lm).__name__ == "PeftModel":
        raise SystemExit("refusing to compile a PeftModel; merge_and_unload did not run")
    n_params = sum(p.numel() for p in model.lm.parameters())
    print(
        f"merged backbone {type(model.lm).__name__}  params={n_params / 1e6:.1f}M  "
        f"hidden={model.lm.config.hidden_size}  layers={model.lm.config.num_hidden_layers}  "
        f"lora_rank={ck.meta.lora} -> 0  T={model.head.temperature}  "
        f"hybrid={getattr(model, 'hybrid', False)}"
    )
    return ck, tok, model


def compile_one(
    family: str, quantize: str, output: Path | None, max_batch: int, min_seq: int, device: str
) -> Path:
    spec = CHECKPOINTS[family]
    ck, tok, model = _load_merged(spec["repo"], device=device)

    max_len = min(MAX_LEN, int(getattr(model.lm.config, "max_position_embeddings", MAX_LEN)))
    min_seq = min(max(int(min_seq), 1), max_len)

    print(f"building merged trunk  max_len={max_len} max_opts={MAX_OPTS}")
    trunk = KevCleanTrunk(model.lm, model.head, max_opts=MAX_OPTS, max_len=max_len).eval()
    leftover = count_lora_tensors(trunk)
    if leftover:
        raise SystemExit(f"trunk still has LoRA tensors: {leftover[:8]}")

    example = _example_rows(tok, model, seq_len=min(128, max_len))
    print("example shapes", [tuple(t.shape) for t in example], [t.dtype for t in example])

    pipe_tok = BPETokenizer.from_huggingface(tok, context_length=max_len)
    special_ids = {name: int(tok.convert_tokens_to_ids(name)) for name in SPECIAL}
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    decision = DecisionPipelineSpec.kev(
        specials=special_ids,
        pad_id=pad_id,
        max_len=max_len,
        max_opts=MAX_OPTS,
        min_seq=int(min_seq),
        max_batch=int(max_batch),
        length_buckets=_length_buckets(max_len),
        temperature=float(model.head.temperature),
        temperature_baked=True,
        model_name=spec["model_name"],
        family=f"kev-{family}",
        checkpoint=spec["repo"],
    )
    extra = {
        "kev.max_len": max_len,
        "kev.max_opts": MAX_OPTS,
        "kev.max_batch": int(max_batch),
        "kev.min_seq": int(min_seq),
        "kev.length_buckets": json.dumps(_length_buckets(max_len)),
        "kev.temperature": float(model.head.temperature),
        "kev.model_name": spec["model_name"],
        "kev.family": f"kev-{family}",
        "kev.checkpoint": spec["repo"],
        "kev.base": ck.meta.base,
        "kev.arch": spec["arch"],
        "kev.lora_merged": True,
        "kev.lora_rank": int(ck.meta.lora),
        "kev.head_dim": int(ck.meta.head_dim),
        "kev.specials": json.dumps(special_ids),
        "kev.pad_token_id": pad_id,
        "laya.family": f"kev-{family}",
    }
    extra.update(decision.to_gguf_metadata())

    if output is None:
        suffix = quantize.replace("-", "_")
        output = ROOT / "scratch" / f"{spec['stem']}_{suffix}.gguf"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fusion = FusionOptions()
    fusion.enable_rope = False
    dim_b = torch.export.Dim("b", min=1, max=int(max_batch))
    dim_s = torch.export.Dim("s", min=int(min_seq), max=max_len)
    dynamic_shapes = (
        {0: dim_b, 1: dim_s},
        {0: dim_b, 1: dim_s},
        {0: dim_b},
        {0: dim_b},
        {0: dim_b},
    )
    print(
        f"compiling -> {output} family={family} quantize={quantize} "
        f"b=1..{max_batch} s={min_seq}..{max_len} (LoRA already fused)"
    )
    ggmlc.compile(
        trunk,
        example,
        output=str(output),
        model_name=spec["stem"],
        quantize=quantize,
        pipeline=pipe_tok,
        tasks=["classification"],
        extra_metadata=extra,
        fusion_options=fusion,
        dynamic_shapes=dynamic_shapes,
        release_module_storage=True,
    )
    print("wrote", output, "bytes", output.stat().st_size)
    del ck, model, trunk
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile Kev checkpoints to GGUF. LoRA is merged into the base weights "
            "in fp32 before export (required; adapters are not serialized)."
        )
    )
    parser.add_argument("--family", default="0.5b", help="0.5b | 0.8b | 4b | 9b")
    parser.add_argument(
        "--checkpoint", default=None, help="Alias for --family (HF repo or short name)"
    )
    parser.add_argument("--output", default=None, help="Output GGUF path")
    parser.add_argument("--quantize", default="f16", choices=QUANT_CHOICES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--min-seq", type=int, default=64)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Load, merge LoRA, print stats, and exit (no GGUF).",
    )
    args = parser.parse_args()

    family = _normalize_family(args.checkpoint or args.family)
    if args.merge_only:
        _load_merged(CHECKPOINTS[family]["repo"], device=args.device)
        return
    compile_one(
        family,
        args.quantize,
        Path(args.output) if args.output else None,
        args.max_batch,
        args.min_seq,
        args.device,
    )


if __name__ == "__main__":
    main()
