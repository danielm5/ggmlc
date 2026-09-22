#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "json_util.h"
#include "questions.h"
#include "ggmlc/types.h"

namespace laya {

// Compile-time System One recipe stored as GGUF `ggmlc.decision` JSON
// (same role as `tokenizer.chat_template` for causal LLMs). The runner
// does not switch on model family; it interprets this program.
struct DecisionRecipe {
    std::string kind = "laya";
    std::string template_text;
    JsonValue sequence = JsonValue::array();
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    int max_len = 512;
    int head_max_len = 192;
    int max_opts = 16;
    int min_seq = 64;
    int max_batch = 8;
    std::vector<int> length_buckets = {64, 128, 256, 512};
    int32_t pad_id = 0;
    int opt_token_cap = 0;
    bool sanitize_user = false;
    std::string option_format = "laya";
    std::string noul_false = "false: no, the statement does not hold";
    std::string noul_true = "true: yes, the statement holds";
    std::string noul_false_name = "false";
    std::string noul_true_name = "true";
    bool has_qtype_input = true;
    bool has_decide_input = false;
    bool has_act = true;
    bool flatten_markers = true;
    std::string confidence = "shannon";
    std::string noul_confidence = "max_p";
    std::string score_confidence = "shannon";
    bool temperature_baked = false;
    std::vector<float> temperature = {1.0f, 1.0f, 1.0f};
    std::unordered_map<std::string, float> temperature_by_options;
    std::unordered_map<std::string, int32_t> tokens;
    std::string model_name;
    std::string family;
    std::string checkpoint;
    bool loaded = false;
    // True when `ggmlc.decision` was present. False = distributed Laya GGUF compat.
    bool baked = false;

    bool load_from_graph(const ggmlc::SerializedModelGraph& g, const std::string& path = "");
};

std::vector<std::string> render_options(const Question& q, const DecisionRecipe& rec);
float confidence_choice(const std::vector<float>& p, const std::string& kind);
float confidence_score(const std::vector<float>& p, const std::string& kind);
float confidence_noul(float p_true, const std::string& kind);

}  // namespace laya
