#include "recipe.h"

#include <algorithm>
#include <cmath>
#include "language.h"

namespace laya {

static int64_t meta_int(const ggmlc::SerializedModelGraph& g, const std::string& key, int64_t def) {
    auto it = g.metadata_int.find(key);
    return it == g.metadata_int.end() ? def : it->second;
}

static std::string meta_str(const ggmlc::SerializedModelGraph& g, const std::string& key, const std::string& def = "") {
    auto it = g.metadata_str.find(key);
    return it == g.metadata_str.end() ? def : it->second;
}

static int j_int(const JsonValue* v, int def) {
    if (!v) return def;
    if (v->is_number()) return static_cast<int>(v->n);
    if (v->is_string()) {
        try { return std::stoi(v->s); } catch (...) { return def; }
    }
    return def;
}

static bool j_bool(const JsonValue* v, bool def) {
    if (!v) return def;
    if (v->is_bool()) return v->b;
    if (v->is_number()) return v->n != 0.0;
    if (v->is_string()) return v->s == "true" || v->s == "1";
    return def;
}

static std::string j_str(const JsonValue* v, const std::string& def = "") {
    if (!v) return def;
    if (v->is_string()) return v->s;
    if (v->is_number()) return std::to_string(static_cast<long long>(v->n));
    if (v->is_bool()) return v->b ? "true" : "false";
    return def;
}

static std::vector<int> j_int_list(const JsonValue* v) {
    std::vector<int> out;
    if (!v) return out;
    if (v->is_array()) {
        for (const auto& x : v->arr) {
            if (x.is_number()) out.push_back(static_cast<int>(x.n));
        }
    }
    return out;
}

static std::vector<std::string> j_str_list(const JsonValue* v) {
    std::vector<std::string> out;
    if (!v) return out;
    if (v->is_array()) {
        for (const auto& x : v->arr) {
            if (x.is_string()) out.push_back(x.s);
        }
    }
    return out;
}

static void fill_temperature(DecisionRecipe& r, const JsonValue* v) {
    r.temperature = {1.0f, 1.0f, 1.0f};
    if (!v) return;
    if (v->is_number()) {
        float t = static_cast<float>(v->n);
        r.temperature = {t, t, t};
        return;
    }
    if (v->is_array()) {
        for (size_t i = 0; i < v->arr.size() && i < r.temperature.size(); ++i) {
            if (v->arr[i].is_number()) r.temperature[i] = static_cast<float>(v->arr[i].n);
        }
        if (v->arr.size() == 1 && v->arr[0].is_number()) {
            float t = static_cast<float>(v->arr[0].n);
            r.temperature = {t, t, t};
        }
    }
}

static void apply_json(DecisionRecipe& r, const JsonValue& d) {
    if (!d.is_object()) return;
    r.kind = j_str(d.get("kind"), r.kind);
    r.template_text = j_str(d.get("template"), r.template_text);
    if (const JsonValue* seq = d.get("sequence"); seq && seq->is_array()) r.sequence = *seq;
    r.inputs = j_str_list(d.get("inputs"));
    r.outputs = j_str_list(d.get("outputs"));
    r.max_len = j_int(d.get("max_len"), r.max_len);
    r.head_max_len = j_int(d.get("head_max_len"), r.head_max_len);
    r.max_opts = j_int(d.get("max_opts"), r.max_opts);
    r.min_seq = j_int(d.get("min_seq"), r.min_seq);
    r.max_batch = j_int(d.get("max_batch"), r.max_batch);
    auto buckets = j_int_list(d.get("length_buckets"));
    if (!buckets.empty()) r.length_buckets = std::move(buckets);
    r.pad_id = static_cast<int32_t>(j_int(d.get("pad_id"), r.pad_id));
    r.opt_token_cap = j_int(d.get("opt_token_cap"), r.opt_token_cap);
    r.sanitize_user = j_bool(d.get("sanitize_user"), r.sanitize_user);
    r.option_format = j_str(d.get("option_format"), r.option_format);
    r.noul_false = j_str(d.get("noul_false"), r.noul_false);
    r.noul_true = j_str(d.get("noul_true"), r.noul_true);
    r.noul_false_name = j_str(d.get("noul_false_name"), r.noul_false_name);
    r.noul_true_name = j_str(d.get("noul_true_name"), r.noul_true_name);
    r.has_qtype_input = j_bool(d.get("has_qtype_input"), r.has_qtype_input);
    r.has_decide_input = j_bool(d.get("has_decide_input"), r.has_decide_input);
    r.has_act = j_bool(d.get("has_act"), r.has_act);
    r.flatten_markers = j_bool(d.get("flatten_markers"), r.flatten_markers);
    r.confidence = j_str(d.get("confidence"), r.confidence);
    r.noul_confidence = j_str(d.get("noul_confidence"), r.noul_confidence);
    r.score_confidence = j_str(d.get("score_confidence"), r.score_confidence);
    r.temperature_baked = j_bool(d.get("temperature_baked"), r.temperature_baked);
    fill_temperature(r, d.get("temperature"));
    if (const JsonValue* tb = d.get("temperature_by_options"); tb && tb->is_object()) {
        r.temperature_by_options.clear();
        for (const auto& kv : tb->obj) {
            if (kv.second.is_number()) r.temperature_by_options[kv.first] = static_cast<float>(kv.second.n);
        }
    }
    if (const JsonValue* tok = d.get("tokens"); tok && tok->is_object()) {
        r.tokens.clear();
        for (const auto& kv : tok->obj) {
            r.tokens[kv.first] = static_cast<int32_t>(j_int(&kv.second, 0));
        }
    }
    r.model_name = j_str(d.get("model_name"), r.model_name);
    r.family = j_str(d.get("family"), r.family);
    r.checkpoint = j_str(d.get("checkpoint"), r.checkpoint);
}

static JsonValue laya_sequence(int32_t cls, int32_t sep, int32_t mask) {
    JsonValue seq = JsonValue::array();
    auto tok = [](int32_t id, const char* mark = nullptr, const char* budget = nullptr) {
        JsonValue s = JsonValue::object();
        s.set("op", JsonValue::string("tok"));
        s.set("id", JsonValue::number(id));
        if (mark) s.set("mark", JsonValue::string(mark));
        if (budget) s.set("budget", JsonValue::string(budget));
        return s;
    };
    auto user = [](const char* src, bool mask_replace, const char* budget = nullptr) {
        JsonValue s = JsonValue::object();
        s.set("op", JsonValue::string("user"));
        s.set("src", JsonValue::string(src));
        if (mask_replace) s.set("mask_replace", JsonValue::boolean(true));
        if (budget) s.set("budget", JsonValue::string(budget));
        return s;
    };
    seq.arr.push_back(tok(cls));
    {
        JsonValue q = JsonValue::object();
        q.set("op", JsonValue::string("qtype"));
        q.set("budget", JsonValue::string("head"));
        seq.arr.push_back(q);
    }
    seq.arr.push_back(user("instructions", true, "head"));
    seq.arr.push_back(tok(sep));
    {
        JsonValue opts = JsonValue::object();
        opts.set("op", JsonValue::string("opts"));
        JsonValue body = JsonValue::array();
        body.arr.push_back(tok(mask, "option"));
        {
            JsonValue t = JsonValue::object();
            t.set("op", JsonValue::string("text"));
            t.set("s", JsonValue::string(" "));
            body.arr.push_back(t);
        }
        body.arr.push_back(user("option", true));
        opts.set("body", body);
        seq.arr.push_back(opts);
    }
    seq.arr.push_back(tok(sep));
    seq.arr.push_back(user("state", true));
    seq.arr.push_back(tok(sep));
    return seq;
}

static bool fallback_laya(DecisionRecipe& r, const ggmlc::SerializedModelGraph& g, const std::string& path) {
    r.kind = "laya";
    r.option_format = "laya";
    r.has_qtype_input = true;
    r.has_decide_input = false;
    r.has_act = true;
    r.confidence = "shannon";
    r.noul_confidence = "max_p";
    r.score_confidence = "shannon";
    r.inputs = {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
    r.outputs = {"logits", "act_logits"};
    r.max_len = static_cast<int>(meta_int(g, "laya.max_len", 512));
    r.head_max_len = static_cast<int>(meta_int(g, "laya.head_max_len", 192));
    r.max_opts = static_cast<int>(meta_int(g, "laya.max_opts", 16));
    r.min_seq = static_cast<int>(meta_int(g, "laya.min_seq", 64));
    r.max_batch = static_cast<int>(meta_int(g, "laya.max_batch", 8));
    r.pad_id = static_cast<int32_t>(meta_int(g, "laya.pad_token_id", 50283));
    r.opt_token_cap = 48;
    int32_t cls = static_cast<int32_t>(meta_int(g, "laya.cls_token_id", 50281));
    int32_t sep = static_cast<int32_t>(meta_int(g, "laya.sep_token_id", 50282));
    int32_t mask = static_cast<int32_t>(meta_int(g, "laya.mask_token_id", 50284));
    r.tokens["cls"] = cls;
    r.tokens["sep"] = sep;
    r.tokens["pad"] = r.pad_id;
    r.tokens["mask"] = mask;
    r.model_name = meta_str(g, "laya.model_name", "laya");
    r.family = meta_str(g, "laya.family");
    r.checkpoint = meta_str(g, "laya.checkpoint");
    if (r.family.empty()) r.family = infer_family(path, r.model_name, r.checkpoint);
    std::string bj = meta_str(g, "laya.length_buckets");
    if (!bj.empty()) {
        try {
            JsonValue arr = JsonParser::parse_string(bj);
            auto b = j_int_list(&arr);
            if (!b.empty()) r.length_buckets = std::move(b);
        } catch (...) {
        }
    }
    std::string tjson = meta_str(g, "laya.temperature");
    if (!tjson.empty()) {
        try {
            JsonValue t = JsonParser::parse_string(tjson);
            fill_temperature(r, &t);
        } catch (...) {
        }
    } else {
        r.temperature = {1.6369f, 1.25143f, 1.9834f};
    }
    std::string bjson = meta_str(g, "laya.temperature_by_options");
    if (!bjson.empty()) {
        try {
            JsonValue b = JsonParser::parse_string(bjson);
            if (b.is_object()) {
                r.temperature_by_options.clear();
                for (const auto& kv : b.obj) {
                    if (kv.second.is_number()) r.temperature_by_options[kv.first] = static_cast<float>(kv.second.n);
                }
            }
        } catch (...) {
        }
    }
    r.sequence = laya_sequence(cls, sep, mask);
    r.template_text = "[CLS] {qtype} question: {instructions} [SEP] ([MASK] {option})* [SEP] {state} [SEP]";
    r.loaded = true;
    r.baked = false;
    return true;
}

bool DecisionRecipe::load_from_graph(const ggmlc::SerializedModelGraph& g, const std::string& path) {
    *this = DecisionRecipe{};
    std::string json = meta_str(g, "ggmlc.decision");
    if (!json.empty()) {
        try {
            JsonValue d = JsonParser::parse_string(json);
            apply_json(*this, d);
            if (template_text.empty()) template_text = meta_str(g, "ggmlc.decision.template");
            if (kind.empty()) kind = meta_str(g, "ggmlc.decision.kind", "laya");
            if (family.empty()) {
                family = meta_str(g, "laya.family");
                if (family.empty()) family = meta_str(g, "kev.family");
            }
            if (family.empty()) family = infer_family(path, model_name, checkpoint);
            loaded = true;
            baked = true;
            return true;
        } catch (...) {
            // fall through: missing/corrupt recipe is treated as Laya
        }
    }
    // Distributed Laya GGUFs (english / multilingual / typed-decisions) predate
    // ggmlc.decision. Any file without a baked recipe is assumed to be Laya.
    return fallback_laya(*this, g, path);
}

std::vector<std::string> render_options(const Question& q, const DecisionRecipe& rec) {
    std::vector<std::string> opts;
    const bool kev = rec.option_format == "kev";
    if (q.type == QType::Choice) {
        for (const auto& kv : q.criteria) {
            if (kv.second.empty()) opts.push_back(kv.first);
            else opts.push_back(kv.first + ": " + kv.second);
        }
        return opts;
    }
    if (q.type == QType::Score) {
        for (size_t i = 0; i < q.criteria.size(); ++i) {
            if (kev) opts.push_back(q.criteria[i].second);
            else opts.push_back("level " + std::to_string(i) + ": " + q.criteria[i].second);
        }
        return opts;
    }
    std::string false_crit, true_crit;
    for (const auto& kv : q.criteria) {
        if (kv.first == "false" || kv.first == rec.noul_false_name) false_crit = kv.second;
        if (kv.first == "true" || kv.first == rec.noul_true_name) true_crit = kv.second;
    }
    if (kev) {
        opts.push_back(false_crit.empty() ? rec.noul_false : false_crit);
        opts.push_back(true_crit.empty() ? rec.noul_true : true_crit);
        return opts;
    }
    opts.push_back(
        "false: " + (false_crit.empty() ? std::string("no, the statement does not hold") : false_crit)
    );
    opts.push_back(
        "true: " + (true_crit.empty() ? std::string("yes, the statement holds") : true_crit)
    );
    return opts;
}

float confidence_choice(const std::vector<float>& p, const std::string& kind) {
    if (kind == "kev" || kind == "modal_gap") {
        const int k = static_cast<int>(p.size());
        if (k <= 1) return 1.0f;
        float mx = 0.0f;
        for (float v : p) mx = std::max(mx, v);
        const float uniform = 1.0f / static_cast<float>(k);
        return (mx - uniform) / (1.0f - uniform);
    }
    return confidence_from_probs(p);
}

float confidence_score(const std::vector<float>& p, const std::string& kind) {
    if (kind == "modal_distance" || kind == "kev") {
        const int L = static_cast<int>(p.size());
        if (L <= 1) return 1.0f;
        int mode = 0;
        for (int i = 1; i < L; ++i) if (p[i] > p[mode]) mode = i;
        double acc = 0.0;
        for (int i = 0; i < L; ++i) acc += static_cast<double>(p[i]) * std::abs(i - mode);
        float c = static_cast<float>(1.0 - acc / (L - 1));
        if (c < 0.0f) c = 0.0f;
        if (c > 1.0f) c = 1.0f;
        return c;
    }
    return confidence_from_probs(p);
}

float confidence_noul(float p_true, const std::string& kind) {
    (void)kind;
    return std::max(p_true, 1.0f - p_true);
}

}  // namespace laya
