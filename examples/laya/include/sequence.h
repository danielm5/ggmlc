#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "ggmlc/pipeline/tokenizer.h"
#include "questions.h"
#include "recipe.h"

namespace laya {

struct EncodedQuestion {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
    int32_t decide = -1;
    QType qtype = QType::Choice;
};

struct SequenceConfig {
    int max_len = 512;
    int head_max_len = 192;
    int max_opts = 16;
    int32_t cls_id = 50281;
    int32_t sep_id = 50282;
    int32_t pad_id = 50283;
    int32_t mask_id = 50284;

    static SequenceConfig from_recipe(const DecisionRecipe& rec);
};

// Laya kind (including GGUFs with no ggmlc.decision) uses the original encoder
// so already-distributed files stay bit-identical. Other kinds run the baked
// sequence program from the GGUF.
EncodedQuestion build_sequence(
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const std::string& state_text,
    const Question& q
);

void pad_encoded_batch(
    const std::vector<EncodedQuestion>& encs,
    const DecisionRecipe& rec,
    int batch,
    int seq_len,
    std::vector<int32_t>& input_ids,
    std::vector<float>& attention_mask,
    std::vector<int32_t>& marker_pos,
    std::vector<float>& marker_mask,
    std::vector<int32_t>& qtype,
    std::vector<int32_t>& decide_pos
);

}  // namespace laya
