from ggmlc.pipeline.decision import DecisionPipelineSpec


def test_laya_recipe_gguf_keys():
    spec = DecisionPipelineSpec.laya(
        cls_id=50281,
        sep_id=50282,
        pad_id=50283,
        mask_id=50284,
        max_len=512,
        head_max_len=192,
        max_opts=16,
        min_seq=64,
        max_batch=8,
        length_buckets=[64, 128, 256, 512],
        temperature=[1.6, 1.2, 1.9],
        temperature_by_options={"choice:3-5": 1.1},
        model_name="laya",
        family="english",
        checkpoint="convaiinnovations/laya",
    )
    meta = spec.to_gguf_metadata()
    assert "ggmlc.decision" in meta
    assert meta["ggmlc.decision.kind"] == "laya"
    assert "[CLS]" in meta["ggmlc.decision.template"]
    assert spec.inputs == ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


def test_kev_recipe_is_required_for_non_laya():
    spec = DecisionPipelineSpec.kev(
        specials={
            "<|fim_prefix|>": 151659,
            "<|fim_middle|>": 151660,
            "<|box_start|>": 151661,
            "<|box_end|>": 151662,
            "<|fim_suffix|>": 151663,
        },
        pad_id=151643,
        max_len=2048,
        max_opts=16,
        min_seq=64,
        max_batch=8,
        length_buckets=[64, 128, 256, 512, 1024, 2048],
        temperature=1.0,
        temperature_baked=True,
        model_name="kev-0.5b",
        family="kev-0.5b",
        checkpoint="jaredpalmer/kev-0.5b",
    )
    meta = spec.to_gguf_metadata()
    assert meta["ggmlc.decision.kind"] == "kev"
    assert spec.has_decide_input
    assert not spec.has_qtype_input
    assert spec.noul_false == "no"
    d = spec.to_dict()
    ops = [step["op"] for step in d["sequence"]]
    assert "opts" in ops
    assert any(step.get("mark") == "decide" for step in d["sequence"])
