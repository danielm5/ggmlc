"""System-1 decision preprocessing baked into GGUF (chat-template analogue).

New architectures (Kev, …) must serialize ``ggmlc.decision`` at compile time.
The C++ runner interprets that program and does not switch on model family.

Distributed Laya GGUFs predate this key. If ``ggmlc.decision`` is missing, the
runner assumes the built-in Laya preprocessor (``laya.*`` metadata + defaults).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _tok(tid: int, mark: str | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"op": "tok", "id": int(tid)}
    if mark:
        step["mark"] = mark
    return step


def _user(
    src: str,
    *,
    sanitize: bool = False,
    mask_replace: bool = False,
    budget: str | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {"op": "user", "src": src}
    if sanitize:
        step["sanitize"] = True
    if mask_replace:
        step["mask_replace"] = True
    if budget:
        step["budget"] = budget
    return step


@dataclass
class DecisionPipelineSpec:
    """Compile-time recipe for System One encode / decode.

    Serializes to a single ``ggmlc.decision`` JSON string plus a human-readable
    ``ggmlc.decision.template`` (same role as ``tokenizer.chat_template``).
    """

    kind: str
    sequence: list[dict[str, Any]]
    template: str
    inputs: list[str]
    outputs: list[str]
    max_len: int = 512
    head_max_len: int = 192
    max_opts: int = 16
    min_seq: int = 64
    max_batch: int = 8
    length_buckets: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    pad_id: int = 0
    opt_token_cap: int = 0
    sanitize_user: bool = False
    option_format: str = "laya"
    noul_false: str = "false: no, the statement does not hold"
    noul_true: str = "true: yes, the statement holds"
    noul_false_name: str = "false"
    noul_true_name: str = "true"
    has_qtype_input: bool = True
    has_decide_input: bool = False
    has_act: bool = True
    flatten_markers: bool = True
    confidence: str = "shannon"
    noul_confidence: str = "max_p"
    score_confidence: str = "shannon"
    temperature_baked: bool = False
    temperature: Any = 1.0
    temperature_by_options: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    model_name: str = ""
    family: str = ""
    checkpoint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "sequence": self.sequence,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "max_len": self.max_len,
            "head_max_len": self.head_max_len,
            "max_opts": self.max_opts,
            "min_seq": self.min_seq,
            "max_batch": self.max_batch,
            "length_buckets": list(self.length_buckets),
            "pad_id": self.pad_id,
            "opt_token_cap": self.opt_token_cap,
            "sanitize_user": self.sanitize_user,
            "option_format": self.option_format,
            "noul_false": self.noul_false,
            "noul_true": self.noul_true,
            "noul_false_name": self.noul_false_name,
            "noul_true_name": self.noul_true_name,
            "has_qtype_input": self.has_qtype_input,
            "has_decide_input": self.has_decide_input,
            "has_act": self.has_act,
            "flatten_markers": self.flatten_markers,
            "confidence": self.confidence,
            "noul_confidence": self.noul_confidence,
            "score_confidence": self.score_confidence,
            "temperature_baked": self.temperature_baked,
            "temperature": self.temperature,
            "temperature_by_options": dict(self.temperature_by_options),
            "tokens": dict(self.tokens),
            "model_name": self.model_name,
            "family": self.family,
            "checkpoint": self.checkpoint,
        }

    def to_gguf_metadata(self) -> dict[str, Any]:
        d = self.to_dict()
        meta: dict[str, Any] = {
            "ggmlc.decision": json.dumps(d, separators=(",", ":")),
            "ggmlc.decision.kind": self.kind,
            "ggmlc.decision.template": self.template,
        }
        return meta

    @classmethod
    def laya(
        cls,
        *,
        cls_id: int,
        sep_id: int,
        pad_id: int,
        mask_id: int,
        max_len: int,
        head_max_len: int,
        max_opts: int,
        min_seq: int,
        max_batch: int,
        length_buckets: list[int],
        temperature: Any,
        temperature_by_options: dict[str, float],
        model_name: str,
        family: str,
        checkpoint: str,
    ) -> DecisionPipelineSpec:
        seq = [
            _tok(cls_id),
            {"op": "qtype", "budget": "head"},
            _user("instructions", mask_replace=True, budget="head"),
            _tok(sep_id),
            {
                "op": "opts",
                "body": [
                    _tok(mask_id, mark="option"),
                    {"op": "text", "s": " "},
                    _user("option", mask_replace=True),
                ],
            },
            _tok(sep_id),
            _user("state", mask_replace=True),
            _tok(sep_id),
        ]
        template = (
            "[CLS] {qtype} question: {instructions} [SEP] ([MASK] {option})* [SEP] {state} [SEP]"
        )
        return cls(
            kind="laya",
            sequence=seq,
            template=template,
            inputs=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            outputs=["logits", "act_logits"],
            max_len=max_len,
            head_max_len=head_max_len,
            max_opts=max_opts,
            min_seq=min_seq,
            max_batch=max_batch,
            length_buckets=length_buckets,
            pad_id=pad_id,
            opt_token_cap=48,
            option_format="laya",
            has_qtype_input=True,
            has_decide_input=False,
            has_act=True,
            confidence="shannon",
            noul_confidence="max_p",
            score_confidence="shannon",
            temperature=temperature,
            temperature_by_options=temperature_by_options,
            tokens={"cls": cls_id, "sep": sep_id, "pad": pad_id, "mask": mask_id},
            model_name=model_name,
            family=family,
            checkpoint=checkpoint,
        )

    @classmethod
    def kev(
        cls,
        *,
        specials: dict[str, int],
        pad_id: int,
        max_len: int,
        max_opts: int,
        min_seq: int,
        max_batch: int,
        length_buckets: list[int],
        temperature: float,
        temperature_baked: bool,
        model_name: str,
        family: str,
        checkpoint: str,
    ) -> DecisionPipelineSpec:
        pref = int(specials["<|fim_prefix|>"])
        qid = int(specials["<|fim_middle|>"])
        o_open = int(specials["<|box_start|>"])
        o_close = int(specials["<|box_end|>"])
        decide = int(specials["<|fim_suffix|>"])
        seq = [
            _tok(pref),
            _user("state", sanitize=True),
            _tok(qid),
            _user("instructions", sanitize=True),
            {
                "op": "opts",
                "body": [
                    _tok(o_open),
                    _user("option", sanitize=True),
                    _tok(o_close, mark="option"),
                ],
            },
            _tok(decide, mark="decide"),
        ]
        template = (
            "<|fim_prefix|>{state}<|fim_middle|>{instructions}"
            "(<|box_start|>{option}<|box_end|>)*<|fim_suffix|>"
        )
        return cls(
            kind="kev",
            sequence=seq,
            template=template,
            inputs=["input_ids", "attention_mask", "opt_pos", "decide_pos", "opt_mask"],
            outputs=["logits"],
            max_len=max_len,
            head_max_len=max_len,
            max_opts=max_opts,
            min_seq=min_seq,
            max_batch=max_batch,
            length_buckets=length_buckets,
            pad_id=pad_id,
            sanitize_user=True,
            option_format="kev",
            noul_false="no",
            noul_true="yes",
            noul_false_name="false",
            noul_true_name="true",
            has_qtype_input=False,
            has_decide_input=True,
            has_act=False,
            confidence="kev",
            noul_confidence="max_p",
            score_confidence="modal_distance",
            temperature_baked=temperature_baked,
            temperature=float(temperature),
            tokens={
                "fim_prefix": pref,
                "fim_middle": qid,
                "box_start": o_open,
                "box_end": o_close,
                "fim_suffix": decide,
                "pad": pad_id,
            },
            model_name=model_name,
            family=family,
            checkpoint=checkpoint,
        )
