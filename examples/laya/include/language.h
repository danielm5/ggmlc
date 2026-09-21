#pragma once

#include <string>
#include "json_util.h"

namespace laya {

struct LangGuess {
    std::string script = "unknown";   // latin, han, arabic, ...
    bool is_english = true;
    int words = 0;
    int english_hits = 0;
    double english_ratio = 0.0;
    std::string reason;
};

// Flatten string leaves of a state (keys ignored — they are usually English schema).
std::string collect_state_text(const JsonValue& state, int max_chars = 4000);

// Script + English function-word routing, matching laya.Router's intent:
// non-Latin script → not English; otherwise count common English words.
LangGuess guess_language(const std::string& text);
LangGuess guess_language(const JsonValue& state);

std::string infer_family(const std::string& path,
                         const std::string& model_name = "",
                         const std::string& checkpoint = "");

// TypeSafe `model` field → Laya family. Empty `model` / "auto" → auto_route.
struct ModelRef {
    bool auto_route = true;
    bool unknown = false;
    std::string family;
};
ModelRef resolve_model_name(const std::string& model);

int quant_rank(const std::string& path);

}  // namespace laya
