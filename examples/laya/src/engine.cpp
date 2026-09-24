#include "engine.h"
#include "language.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace laya {

static std::string meta_str(const ggmlc::SerializedModelGraph& g, const std::string& key, const std::string& def = "") {
    auto it = g.metadata_str.find(key);
    return it == g.metadata_str.end() ? def : it->second;
}

static uint32_t find_input(const ggmlc::SerializedModelGraph& g, const std::vector<std::string>& needles, uint32_t fallback) {
    for (uint32_t id : g.inputs) {
        auto it = g.tensors.find(id);
        if (it == g.tensors.end()) continue;
        const std::string& name = it->second.name;
        for (const auto& n : needles) {
            if (name.find(n) != std::string::npos) return id;
        }
    }
    return fallback;
}

bool DecisionEngine::load_model(const std::string& gguf_path, const EngineOptions& opt) {
    model_path_ = gguf_path;
    device_ = opt.device;
    n_threads_ = opt.n_threads;
    cuda_graph_ = opt.cuda_graph;
    loaded_ = false;

    std::cerr << "[laya] loading " << gguf_path << " device=" << device_ << std::endl;
    try {
        graph_ = ggmlc::ModelLoader::load_from_file(gguf_path);
        try {
            executor_ = std::make_unique<ggmlc::ModelExecutor>(graph_, device_);
        } catch (const std::exception& e) {
            if (device_ == "auto") {
                std::cerr << "[laya] auto device failed (" << e.what() << "), falling back to cpu\n";
                executor_ = std::make_unique<ggmlc::ModelExecutor>(graph_, "cpu");
            } else {
                throw;
            }
        }
        device_ = executor_->device();
    } catch (const std::exception& e) {
        std::cerr << "[laya] load failed: " << e.what() << std::endl;
        return false;
    }

    recipe_.load_from_graph(graph_, gguf_path);
    if (!recipe_.baked) {
        std::cerr << "[laya] no ggmlc.decision in GGUF; using built-in Laya preprocessor "
                     "(compat with already-distributed english/multilingual/typed GGUFs)\n";
    }

    if (graph_.inputs.size() < 3 || graph_.outputs.empty()) {
        std::cerr << "[laya] unexpected graph arity: inputs=" << graph_.inputs.size()
                  << " outputs=" << graph_.outputs.size() << std::endl;
        return false;
    }

    in_ids_ = find_input(graph_, {"input_ids", "ids"}, graph_.inputs[0]);
    in_att_ = find_input(graph_, {"attention_mask", "attn"}, graph_.inputs.size() > 1 ? graph_.inputs[1] : graph_.inputs[0]);
    in_mpos_ = find_input(graph_, {"marker_pos", "opt_pos"}, graph_.inputs.size() > 2 ? graph_.inputs[2] : graph_.inputs[0]);
    in_mmask_ = find_input(graph_, {"marker_mask", "opt_mask"}, graph_.inputs.size() > 3 ? graph_.inputs[3] : graph_.inputs[0]);
    has_qtype_ = recipe_.has_qtype_input;
    has_decide_ = recipe_.has_decide_input;
    has_act_ = recipe_.has_act;
    in_qtype_ = has_qtype_
        ? find_input(graph_, {"qtype"}, graph_.inputs.size() > 4 ? graph_.inputs[4] : graph_.inputs[0])
        : 0;
    in_decide_ = has_decide_
        ? find_input(graph_, {"decide_pos", "decide"}, graph_.inputs.size() > 3 ? graph_.inputs[3] : graph_.inputs[0])
        : 0;
    out_logits_ = graph_.outputs[0];
    out_act_ = (has_act_ && graph_.outputs.size() > 1) ? graph_.outputs[1] : graph_.outputs[0];

    seq_ = SequenceConfig::from_recipe(recipe_);
    max_batch_ = recipe_.max_batch;
    if (opt.max_batch > 0) max_batch_ = opt.max_batch;
    if (max_batch_ < 1) max_batch_ = 1;
    min_seq_ = recipe_.min_seq;
    if (min_seq_ < 1) min_seq_ = 1;
    if (min_seq_ > seq_.max_len) min_seq_ = seq_.max_len;

    dynamic_ = !graph_.symbol_table.empty();
    length_buckets_ = recipe_.length_buckets;
    if (length_buckets_.empty()) length_buckets_ = {64, 128, 256, 512};
    if (!dynamic_) {
        length_buckets_ = {seq_.max_len};
        min_seq_ = seq_.max_len;
        max_batch_ = 1;
    }

    if (!tokenizer_.init_from_gguf_file(gguf_path)) {
        std::cerr << "[laya] warning: GGUF has no tokenizer metadata; sequence encoding will fail." << std::endl;
    }

    model_name_ = recipe_.model_name.empty() ? meta_str(graph_, "laya.model_name", "laya") : recipe_.model_name;
    family_ = recipe_.family;
    if (family_.empty()) {
        family_ = infer_family(gguf_path, model_name_, recipe_.checkpoint);
    }

    if (cuda_graph_) executor_->set_enable_cuda_graph(true);

    loaded_ = true;
    std::cerr << "[laya] ready  kind=" << recipe_.kind
              << (recipe_.baked ? "  recipe=gguf" : "  recipe=laya-compat")
              << "  max_len=" << seq_.max_len
              << " max_opts=" << seq_.max_opts
              << " max_batch=" << max_batch_
              << " dynamic=" << (dynamic_ ? "b,s" : "static")
              << " vocab=" << tokenizer_.vocab_size() << std::endl;
    return true;
}

int DecisionEngine::clamp_seq(int n) const {
    int s = std::max(n, min_seq_);
    if (s > seq_.max_len) s = seq_.max_len;
    return s;
}

int DecisionEngine::batch_cap_for_seq(int seq_len) const {
    if (!dynamic_ || seq_len <= 0) return 1;
    // Token budget before the OOM-halve fallback. 1024 forced Kev's 7-row
    // email (S≈316) into three weight reads; one padded batch fits the 6 GB
    // laptop and matches Python collate. Laya's published email shape
    // (S=124, B=7) stays under max_batch either way.
    const int budget = (device_ == "cpu") ? 8192 : 8192;
    int cap = std::max(1, budget / seq_len);
    return std::min(cap, max_batch_);
}

static void bind_dim_value(
    const std::shared_ptr<ggmlc::DimExpr>& d,
    int64_t val,
    const std::vector<std::string>& table,
    std::unordered_map<std::string, int64_t>& env
) {
    if (!d) return;
    if (d->type == ggmlc::DimType::SYMBOL) {
        if (d->val >= 0 && d->val < static_cast<int64_t>(table.size())) {
            env[table[static_cast<size_t>(d->val)]] = val;
        }
        return;
    }
    bind_dim_value(d->left, val, table, env);
    bind_dim_value(d->right, val, table, env);
}

void DecisionEngine::fill_symbol_env(
    int batch, int seq_len, std::unordered_map<std::string, int64_t>& env
) const {
    env.clear();
    env["b"] = batch;
    env["s"] = seq_len;
    auto it = graph_.tensors.find(in_ids_);
    if (it != graph_.tensors.end()) {
        // PyTorch [B, S] -> GGML ne[0]=S, ne[1]=B
        bind_dim_value(it->second.ne[0], seq_len, graph_.symbol_table, env);
        bind_dim_value(it->second.ne[1], batch, graph_.symbol_table, env);
    }
    for (const auto& sym : graph_.symbol_table) {
        if (env.find(sym) == env.end()) env[sym] = seq_len;
    }
}

bool DecisionEngine::prepare_shape(int batch, int seq_len) {
    std::unordered_map<std::string, int64_t> env;
    if (dynamic_) fill_symbol_env(batch, seq_len, env);
    try {
        executor_->prepare(env, true);
        if (cuda_graph_) executor_->set_enable_cuda_graph(true);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[laya] prepare failed B=" << batch << " S=" << seq_len
                  << " : " << e.what() << std::endl;
        return false;
    }
}

void DecisionEngine::bind_and_run_batch(
    int batch,
    int seq_len,
    const std::vector<int32_t>& ids,
    const std::vector<float>& att,
    const std::vector<int32_t>& mpos,
    const std::vector<float>& mmask,
    const std::vector<int32_t>& qtype,
    const std::vector<int32_t>& decide,
    std::vector<float>& logits,
    std::vector<float>& act
) {
    if (!prepare_shape(batch, seq_len)) {
        throw std::runtime_error("laya prepare_shape failed");
    }
    executor_->set_input(in_ids_, ids.data(), ids.size() * sizeof(int32_t));
    executor_->set_input(in_att_, att.data(), att.size() * sizeof(float));
    executor_->set_input(in_mpos_, mpos.data(), mpos.size() * sizeof(int32_t));
    executor_->set_input(in_mmask_, mmask.data(), mmask.size() * sizeof(float));
    if (has_qtype_) {
        executor_->set_input(in_qtype_, qtype.data(), qtype.size() * sizeof(int32_t));
    }
    if (has_decide_) {
        executor_->set_input(in_decide_, decide.data(), decide.size() * sizeof(int32_t));
    }
    executor_->run(n_threads_);

    const int opts = seq_.max_opts;
    logits.assign(static_cast<size_t>(batch) * opts, -1.0e4f);
    const float* lp = static_cast<const float*>(executor_->get_output_data(out_logits_));
    if (lp) {
        size_t n = executor_->get_tensor_size_bytes(out_logits_) / sizeof(float);
        size_t want = static_cast<size_t>(batch) * opts;
        if (n > want) n = want;
        std::memcpy(logits.data(), lp, n * sizeof(float));
    }
    act.assign(static_cast<size_t>(batch) * 2, 0.0f);
    if (has_act_ && out_act_ != out_logits_) {
        const float* ap = static_cast<const float*>(executor_->get_output_data(out_act_));
        if (ap) {
            size_t n = executor_->get_tensor_size_bytes(out_act_) / sizeof(float);
            size_t want = static_cast<size_t>(batch) * 2;
            if (n > want) n = want;
            std::memcpy(act.data(), ap, n * sizeof(float));
        }
    }
}

Answer DecisionEngine::decode_answer(
    const Question& q, const float* logits, int k, const float* act
) const {
    std::vector<float> z(logits, logits + std::max(0, k));
    float tscale = 1.0f;
    if (!recipe_.temperature_baked) {
        tscale = recipe_.temperature[static_cast<int>(q.type) % 3];
        std::string bucket = temp_bucket(q.type, k);
        auto it = recipe_.temperature_by_options.find(bucket);
        if (it != recipe_.temperature_by_options.end()) tscale = it->second;
    }
    std::vector<float> p = softmax_temp(z, tscale);

    float act_p = 0.0f;
    if (has_act_ && act) {
        float m = std::max(act[0], act[1]);
        float e0 = std::exp(act[0] - m);
        float e1 = std::exp(act[1] - m);
        act_p = e0 / (e0 + e1);
    }

    Answer a;
    a.id = q.id;
    a.type = q.type;
    a.act_probability = act_p;
    const auto keys = q.criteria;
    if (q.type == QType::Choice) {
        int arg = 0;
        for (int i = 1; i < k; ++i) if (p[i] > p[arg]) arg = i;
        if (arg < static_cast<int>(keys.size())) a.choice = keys[arg].first;
        for (int i = 0; i < k && i < static_cast<int>(keys.size()); ++i) {
            a.probabilities.emplace_back(keys[i].first, p[i]);
        }
        a.confidence = confidence_choice(p, recipe_.confidence);
    } else if (q.type == QType::Score) {
        float expv = 0.0f;
        for (int i = 0; i < k; ++i) {
            expv += static_cast<float>(i) * p[i];
            a.probabilities.emplace_back(std::to_string(i), p[i]);
            if (i < static_cast<int>(keys.size())) a.legend.push_back(keys[i].second);
        }
        a.score = expv;
        a.confidence = confidence_score(p, recipe_.score_confidence);
    } else {
        float pt = (k >= 2) ? p[1] : 0.0f;
        a.noul = pt;
        a.confidence = confidence_noul(pt, recipe_.noul_confidence);
        if (k >= 2) {
            a.probabilities.emplace_back(recipe_.noul_false_name, p[0]);
            a.probabilities.emplace_back(recipe_.noul_true_name, p[1]);
        }
    }
    return a;
}

DecideResult DecisionEngine::decide_one(const JsonValue& state, const Question& q) {
    return decide(state, std::vector<Question>{q});
}

DecideResult DecisionEngine::decide(const JsonValue& state, const std::vector<Question>& questions) {
    DecideResult result;
    result.model = model_name_;
    if (!loaded_ || !executor_) return result;

    const std::string state_text = serialize_state(state);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<EncodedQuestion> encs;
    encs.reserve(questions.size());
    int tokens = 0;
    int max_live = 0;
    for (const auto& q : questions) {
        EncodedQuestion enc = build_sequence(tokenizer_, recipe_, state_text, q);
        // The compiled graph has exactly max_opts marker slots. pad_encoded_batch silently
        // truncated a longer option list to that many, while the readback below still
        // trusted markers.size() and walked that many floats out of a max_opts-wide row --
        // reading into the next row and past the end of the logits buffer. That produced
        // garbage (often NaN) probabilities indistinguishable from real answers, plus a
        // strong bias toward option 1. Refuse the request rather than guess.
        if (static_cast<int>(enc.markers.size()) > seq_.max_opts) {
            throw std::runtime_error(
                "question '" + q.id + "' has " + std::to_string(enc.markers.size()) +
                " options but this model supports at most " + std::to_string(seq_.max_opts) +
                " (laya.max_opts); split the question or recompile the GGUF with a"
                " larger max_opts");
        }
        tokens += static_cast<int>(enc.ids.size());
        max_live = std::max(max_live, static_cast<int>(enc.ids.size()));
        encs.push_back(std::move(enc));
    }

    // One rectangular batch padded to max(len_i). Attention stays B×S
    // (not concat-packed). Chunk only when B*S exceeds batch_cap_for_seq.
    std::vector<int> idxs(encs.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    result.answers.assign(questions.size(), Answer{});
    result.input_tokens = tokens;
    result.seq_bucket = max_live > 0 ? clamp_seq(max_live) : 0;

    {
        int S = min_seq_;
        for (int qi : idxs) {
            S = std::max(S, clamp_seq(static_cast<int>(encs[qi].ids.size())));
        }
        const int cap = batch_cap_for_seq(S);
        int graph_B = 0;
        size_t cursor = 0;
        while (cursor < idxs.size()) {
            const int remaining = static_cast<int>(idxs.size() - cursor);
            int live = std::min(remaining, cap);
            int batch = live;
            // Hold one (B,S) for the group. Dummy-pad later chunks so prepare
            // and CUDA graphs stay warm across the preset's forwards.
            if (graph_B < 1) graph_B = live;
            batch = graph_B;
            live = std::min(live, remaining);
            if (live < 1) live = 1;
            if (batch < live) batch = live;

            bool ok = false;
            for (int try_b = batch; try_b >= 1; try_b = (try_b > 1) ? (try_b / 2) : 0) {
                const int take = std::min(try_b, remaining);
                std::vector<EncodedQuestion> slice;
                slice.reserve(static_cast<size_t>(take));
                for (int j = 0; j < take; ++j) slice.push_back(encs[idxs[cursor + j]]);
                std::vector<int32_t> ids, mpos, qt, decide;
                std::vector<float> att, mmask;
                pad_encoded_batch(slice, recipe_, try_b, S, ids, att, mpos, mmask, qt, decide);
                try {
                    std::vector<float> logits, act;
                    bind_and_run_batch(try_b, S, ids, att, mpos, mmask, qt, decide, logits, act);
                    const int opts = seq_.max_opts;
                    for (int j = 0; j < take; ++j) {
                        const int qi = idxs[cursor + j];
                        // Never read more than the row actually holds: the logits row is
                        // opts wide, so clamping here keeps a stale or oversized option
                        // list from walking into the next row / off the end of the buffer.
                        const int k = std::min(static_cast<int>(encs[qi].markers.size()), opts);
                        const float* lp = logits.data() + static_cast<size_t>(j) * opts;
                        const float* ap = (act.size() >= static_cast<size_t>(j + 1) * 2)
                            ? act.data() + static_cast<size_t>(j) * 2
                            : nullptr;
                        result.answers[qi] = decode_answer(questions[qi], lp, k, ap);
                    }
                    result.n_forwards += 1;
                    result.seq_bucket = std::max(result.seq_bucket, S);
                    result.batch_bucket = std::max(result.batch_bucket, try_b);
                    cursor += static_cast<size_t>(take);
                    graph_B = try_b;
                    ok = true;
                    break;
                } catch (const std::exception& e) {
                    std::cerr << "[laya] forward failed B=" << try_b << " S=" << S
                              << " : " << e.what() << std::endl;
                }
            }
            if (!ok) throw std::runtime_error("laya forward failed");
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

void DecisionEngine::print_info() const {
    std::cout << "model: " << meta_str(graph_, "general.name", graph_.name) << "\n"
              << "family: " << family_ << "\n"
              << "kind: " << recipe_.kind
              << (recipe_.baked ? "  (ggmlc.decision)" : "  (assumed Laya; no ggmlc.decision)") << "\n"
              << "checkpoint: " << recipe_.checkpoint << "\n"
              << "device: " << device_ << "\n"
              << "max_len: " << seq_.max_len << "  min_seq: " << min_seq_
              << "  head_max_len: " << seq_.head_max_len
              << "  max_opts: " << seq_.max_opts << "  max_batch: " << max_batch_ << "\n"
              << "dynamic: " << (dynamic_ ? "b,s" : "static")
              << "  pad: max-in-batch  metadata buckets: [";
    for (size_t i = 0; i < length_buckets_.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << length_buckets_[i];
    }
    std::cout << "]\n"
              << "symbols:";
    for (const auto& s : graph_.symbol_table) std::cout << " " << s;
    std::cout << "\n";
    if (!recipe_.template_text.empty()) {
        std::cout << "template: " << recipe_.template_text << "\n";
    }
    std::cout << "special tokens";
    if (!recipe_.tokens.empty()) {
        for (const auto& kv : recipe_.tokens) {
            std::cout << "  " << kv.first << "=" << kv.second;
        }
    } else {
        std::cout << "  cls=" << seq_.cls_id << " sep=" << seq_.sep_id
                  << " pad=" << seq_.pad_id << " mask=" << seq_.mask_id;
    }
    std::cout << "\n"
              << "vocab: " << tokenizer_.vocab_size() << "\n"
              << "inputs: " << graph_.inputs.size() << "  outputs: " << graph_.outputs.size()
              << "  ops: " << graph_.ops.size() << "\n"
              << "temperature: [";
    for (size_t i = 0; i < recipe_.temperature.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << recipe_.temperature[i];
    }
    std::cout << "]";
    if (recipe_.temperature_baked) std::cout << "  (baked into weights)";
    std::cout << "\n";
    if (!recipe_.temperature_by_options.empty()) {
        std::cout << "temperature_by_options:\n";
        for (const auto& kv : recipe_.temperature_by_options) {
            std::cout << "  " << kv.first << " = " << kv.second << "\n";
        }
    }
}

void DecisionEngine::benchmark(const JsonValue& state, const std::vector<Question>& questions, int runs, int warmup) {
    const int nq = static_cast<int>(questions.size());
    std::cout << "\n=======================================================\n"
              << " Laya System 1 benchmark\n"
              << " device=" << device_
              << "  cuda_graph=" << (cuda_graph_ ? "on" : "off")
              << "  threads=" << n_threads_ << "\n"
              << " questions=" << nq
              << "  dynamic=" << (dynamic_ ? "b,s" : "static")
              << "  max_batch=" << max_batch_ << "\n"
              << " warmup=" << warmup << "  runs=" << runs << "\n"
              << "=======================================================\n";

    for (int i = 0; i < warmup; ++i) decide(state, questions);

    auto percentile = [](std::vector<double> v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        double idx = p * static_cast<double>(v.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, v.size() - 1);
        double t = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - t) + v[hi] * t;
    };

    std::vector<double> wall;
    int tokens = 0, seq_b = 0, batch_b = 0, nfwd = 0;
    for (int i = 0; i < runs; ++i) {
        auto r = decide(state, questions);
        wall.push_back(r.latency_ms);
        tokens = r.input_tokens;
        seq_b = r.seq_bucket;
        batch_b = r.batch_bucket;
        nfwd = r.n_forwards;
        std::cout << "  run " << (i + 1) << "/" << runs
                  << ": wall " << std::fixed << std::setprecision(1) << r.latency_ms << " ms"
                  << "  (" << nq << " q, " << tokens << " tok"
                  << "  S=" << seq_b << " B=" << batch_b
                  << "  forwards=" << nfwd << ")\n";
    }

    const double mean_wall = std::accumulate(wall.begin(), wall.end(), 0.0) / std::max(1, runs);
    const double per_q = nq ? mean_wall / nq : 0.0;
    const double qps = (mean_wall > 0.0) ? (1000.0 * nq / mean_wall) : 0.0;
    std::cout << "-------------------------------------------------------\n"
              << std::fixed << std::setprecision(1)
              << "  Wall clock  mean " << mean_wall
              << "  p50 " << percentile(wall, 0.50)
              << "  best " << *std::min_element(wall.begin(), wall.end()) << " ms\n"
              << "  Equivalent  " << per_q << " ms/q\n"
              << std::setprecision(2)
              << "  Throughput  " << qps << " q/s  (wall, " << nq << "-question preset)\n"
              << "=======================================================\n";
}

}  // namespace laya
