#include "sequence.h"

#include <algorithm>
#include <cctype>

namespace laya {

SequenceConfig SequenceConfig::from_recipe(const DecisionRecipe& rec) {
    SequenceConfig cfg;
    cfg.max_len = rec.max_len;
    cfg.head_max_len = rec.head_max_len;
    cfg.max_opts = rec.max_opts;
    cfg.pad_id = rec.pad_id;
    auto tok = [&](const char* k, int32_t def) -> int32_t {
        auto it = rec.tokens.find(k);
        return it == rec.tokens.end() ? def : it->second;
    };
    cfg.cls_id = tok("cls", 50281);
    cfg.sep_id = tok("sep", 50282);
    cfg.mask_id = tok("mask", 50284);
    if (rec.tokens.count("pad")) cfg.pad_id = rec.tokens.at("pad");
    return cfg;
}

static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static std::string mask_token_str(const ggmlc::pipeline::BPETokenizer& tok, int32_t mask_id) {
    std::string t = tok.decode({mask_id}, false);
    if (t.empty()) t = "[MASK]";
    return t;
}

// Kev: `<|name|>` → `<¦name¦>` so user text cannot mint delimiter tokens.
static std::string sanitize_user(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            size_t j = i + 2;
            while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) || text[j] == '_')) {
                ++j;
            }
            if (j + 1 < text.size() && text[j] == '|' && text[j + 1] == '>') {
                out += "<\xC2\xA6";
                out.append(text, i + 2, j - (i + 2));
                out += "\xC2\xA6>";
                i = j + 2;
                continue;
            }
        }
        out.push_back(text[i++]);
    }
    return out;
}

static EncodedQuestion build_sequence_laya(
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const std::string& state_text,
    const Question& q
) {
    const SequenceConfig cfg = SequenceConfig::from_recipe(rec);
    EncodedQuestion enc;
    enc.qtype = q.type;
    const std::string mask_tok = mask_token_str(tok, cfg.mask_id);
    const std::vector<std::string> opts = render_options(q, rec);

    std::string ins = replace_all(q.instructions, mask_tok, " ");
    const std::string head_text = std::string(qtype_name(q.type)) + " question: " + ins;
    std::vector<int32_t> head_ids = tok.encode(head_text, 0, false, false);

    std::vector<std::vector<int32_t>> opt_ids;
    opt_ids.reserve(opts.size());
    const int cap = rec.opt_token_cap > 0 ? rec.opt_token_cap : 48;
    for (const auto& opt : opts) {
        std::string ot = " " + replace_all(opt, mask_tok, " ");
        std::vector<int32_t> o = tok.encode(ot, 0, false, false);
        if (static_cast<int>(o.size()) > cap) o.resize(static_cast<size_t>(cap));
        std::vector<int32_t> full;
        full.push_back(cfg.mask_id);
        full.insert(full.end(), o.begin(), o.end());
        opt_ids.push_back(std::move(full));
    }

    int opt_sum = 0;
    for (const auto& o : opt_ids) opt_sum += static_cast<int>(o.size());
    int opt_budget = cfg.head_max_len - opt_sum;
    if (opt_budget < 16) {
        int n = std::max(1, static_cast<int>(opt_ids.size()));
        int per = std::max(4, (cfg.head_max_len - 16) / n);
        opt_sum = 0;
        for (auto& o : opt_ids) {
            if (static_cast<int>(o.size()) > per) o.resize(per);
            opt_sum += static_cast<int>(o.size());
        }
        opt_budget = cfg.head_max_len - opt_sum;
    }
    int head_keep = std::max(8, opt_budget);
    if (static_cast<int>(head_ids.size()) > head_keep) head_ids.resize(head_keep);

    std::vector<int32_t> ids;
    ids.push_back(cfg.cls_id);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(cfg.sep_id);

    std::vector<int32_t> markers;
    for (const auto& o : opt_ids) {
        markers.push_back(static_cast<int32_t>(ids.size()));
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(cfg.sep_id);

    int room = std::max(0, cfg.max_len - static_cast<int>(ids.size()) - 1);
    std::string st_text = replace_all(state_text, mask_tok, " ");
    std::vector<int32_t> st = tok.encode(st_text, 0, false, false);
    if (static_cast<int>(st.size()) > room) st.resize(room);
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(cfg.sep_id);

    if (static_cast<int>(ids.size()) > cfg.max_len) ids.resize(cfg.max_len);
    for (int32_t m : markers) {
        if (m < cfg.max_len) enc.markers.push_back(m);
    }
    enc.ids = std::move(ids);
    return enc;
}

static std::string jstr(const JsonValue& s, const char* k, const std::string& def = "") {
    const JsonValue* v = s.get(k);
    if (v && v->is_string()) return v->s;
    return def;
}

static bool jbool(const JsonValue& s, const char* k) {
    const JsonValue* v = s.get(k);
    if (!v) return false;
    if (v->is_bool()) return v->b;
    if (v->is_number()) return v->n != 0.0;
    return v->is_string() && (v->s == "true" || v->s == "1");
}

static int32_t jid(const JsonValue& s) {
    const JsonValue* v = s.get("id");
    if (v && v->is_number()) return static_cast<int32_t>(v->n);
    return 0;
}

struct Piece {
    std::vector<int32_t> ids;
    std::string mark;
    bool is_state = false;
    std::string state_text;
};

static std::vector<int32_t> encode_user(
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const JsonValue& step,
    const std::string& raw,
    const std::string& mask_tok
) {
    std::string text = raw;
    if (jbool(step, "sanitize") || rec.sanitize_user) text = sanitize_user(text);
    if (jbool(step, "mask_replace") && !mask_tok.empty()) text = replace_all(text, mask_tok, " ");
    std::vector<int32_t> ids = tok.encode(text, 0, false, false);
    int cap = 0;
    const JsonValue* cv = step.get("cap");
    if (cv && cv->is_number()) cap = static_cast<int>(cv->n);
    if (jstr(step, "src") == "option" && rec.opt_token_cap > 0 && cap <= 0) cap = rec.opt_token_cap;
    if (cap > 0 && static_cast<int>(ids.size()) > cap) ids.resize(static_cast<size_t>(cap));
    return ids;
}

static void emit_steps(
    const JsonValue& steps,
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const std::string& state_text,
    const Question& q,
    const std::vector<std::string>& opts,
    const std::string& option,
    const std::string& mask_tok,
    std::vector<Piece>& out
) {
    if (!steps.is_array()) return;
    for (const auto& step : steps.arr) {
        if (!step.is_object()) continue;
        const std::string op = jstr(step, "op");
        Piece p;
        p.mark = jstr(step, "mark");
        if (op == "tok") {
            p.ids.push_back(jid(step));
            out.push_back(std::move(p));
        } else if (op == "text") {
            p.ids = tok.encode(jstr(step, "s"), 0, false, false);
            out.push_back(std::move(p));
        } else if (op == "qtype") {
            p.ids = tok.encode(std::string(qtype_name(q.type)) + " question: ", 0, false, false);
            out.push_back(std::move(p));
        } else if (op == "user") {
            const std::string src = jstr(step, "src");
            if (src == "state") {
                p.is_state = true;
                p.state_text = state_text;
                out.push_back(std::move(p));
            } else if (src == "option") {
                p.ids = encode_user(tok, rec, step, option, mask_tok);
                out.push_back(std::move(p));
            } else {
                p.ids = encode_user(tok, rec, step, q.instructions, mask_tok);
                out.push_back(std::move(p));
            }
        } else if (op == "opts") {
            const JsonValue* body = step.get("body");
            if (!body) continue;
            for (const auto& opt : opts) {
                emit_steps(*body, tok, rec, state_text, q, opts, opt, mask_tok, out);
            }
        }
    }
}

static EncodedQuestion build_sequence_program(
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const std::string& state_text,
    const Question& q
) {
    EncodedQuestion enc;
    enc.qtype = q.type;
    const std::vector<std::string> opts = render_options(q, rec);
    int32_t mask_id = 0;
    auto mit = rec.tokens.find("mask");
    if (mit != rec.tokens.end()) mask_id = mit->second;
    const std::string mask_tok = mask_id ? mask_token_str(tok, mask_id) : std::string();

    std::vector<Piece> pieces;
    emit_steps(rec.sequence, tok, rec, state_text, q, opts, "", mask_tok, pieces);

    int used = 0;
    int n_state = 0;
    for (const auto& p : pieces) {
        if (p.is_state) ++n_state;
        else used += static_cast<int>(p.ids.size());
    }
    int room = std::max(0, rec.max_len - used);
    int per = n_state > 0 ? room / n_state : 0;
    for (auto& p : pieces) {
        if (!p.is_state) continue;
        std::string text = p.state_text;
        if (rec.sanitize_user) text = sanitize_user(text);
        if (!mask_tok.empty()) text = replace_all(text, mask_tok, " ");
        p.ids = tok.encode(text, 0, false, false);
        if (per >= 0 && static_cast<int>(p.ids.size()) > per) p.ids.resize(static_cast<size_t>(per));
    }

    std::vector<int32_t> ids;
    ids.reserve(static_cast<size_t>(rec.max_len));
    for (const auto& p : pieces) {
        if (p.mark == "option") enc.markers.push_back(static_cast<int32_t>(ids.size()));
        if (p.mark == "decide") enc.decide = static_cast<int32_t>(ids.size());
        ids.insert(ids.end(), p.ids.begin(), p.ids.end());
    }
    if (static_cast<int>(ids.size()) > rec.max_len) ids.resize(static_cast<size_t>(rec.max_len));
    std::vector<int32_t> keep;
    for (int32_t m : enc.markers) {
        if (m >= 0 && m < static_cast<int32_t>(ids.size())) keep.push_back(m);
    }
    enc.markers = std::move(keep);
    if (enc.decide >= static_cast<int32_t>(ids.size())) enc.decide = -1;
    enc.ids = std::move(ids);
    return enc;
}

EncodedQuestion build_sequence(
    const ggmlc::pipeline::BPETokenizer& tok,
    const DecisionRecipe& rec,
    const std::string& state_text,
    const Question& q
) {
    if (rec.kind == "laya" || rec.kind.empty()) {
        return build_sequence_laya(tok, rec, state_text, q);
    }
    return build_sequence_program(tok, rec, state_text, q);
}

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
) {
    if (batch < 1) batch = 1;
    if (seq_len <= 0) seq_len = rec.max_len;
    const int opts = rec.max_opts;
    input_ids.assign(static_cast<size_t>(batch) * seq_len, rec.pad_id);
    attention_mask.assign(static_cast<size_t>(batch) * seq_len, 0.0f);
    marker_pos.assign(static_cast<size_t>(batch) * opts, 0);
    marker_mask.assign(static_cast<size_t>(batch) * opts, 0.0f);
    qtype.assign(static_cast<size_t>(batch), 0);
    decide_pos.assign(static_cast<size_t>(batch), 0);
    for (int b = 0; b < batch; ++b) {
        if (b >= static_cast<int>(encs.size())) continue;
        const EncodedQuestion& enc = encs[b];
        const int n = std::min(static_cast<int>(enc.ids.size()), seq_len);
        for (int i = 0; i < n; ++i) {
            input_ids[b * seq_len + i] = enc.ids[i];
            attention_mask[b * seq_len + i] = 1.0f;
        }
        const int k = std::min(static_cast<int>(enc.markers.size()), opts);
        for (int i = 0; i < k; ++i) {
            int32_t pos = enc.markers[i];
            if (rec.flatten_markers) pos = b * seq_len + pos;
            marker_pos[b * opts + i] = pos;
            marker_mask[b * opts + i] = 1.0f;
        }
        qtype[b] = static_cast<int32_t>(enc.qtype);
        int32_t d = enc.decide >= 0 ? enc.decide : 0;
        if (rec.flatten_markers) d = b * seq_len + d;
        decide_pos[b] = d;
    }
}

}  // namespace laya
