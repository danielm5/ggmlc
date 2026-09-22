#include "server.h"
#include "web_assets.h"
#include "presets.h"
#include "questions.h"
#include "router.h"
#include "language.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#define SHUT_SEND SD_SEND
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#define closesocket close
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define SHUT_SEND SHUT_WR
#endif

namespace laya {

static constexpr int kMaxBatchStates = 256;

static bool send_all(SOCKET fd, const std::string& data) {
    size_t total = 0;
    while (total < data.size()) {
        int n = static_cast<int>(std::min<size_t>(data.size() - total, 65536));
        int sent = send(fd, data.c_str() + total, n, 0);
        if (sent <= 0) return false;
        total += static_cast<size_t>(sent);
    }
    return true;
}

static const char* status_phrase(int code) {
    switch (code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 422: return "Unprocessable Entity";
        case 500: return "Internal Server Error";
        default: return "OK";
    }
}

static std::string http_json(int code, const std::string& body,
                             const std::vector<std::pair<std::string, std::string>>& extra = {}) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << code << " " << status_phrase(code) << "\r\n"
        << "Content-Type: application/json; charset=utf-8\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n";
    for (const auto& kv : extra) oss << kv.first << ": " << kv.second << "\r\n";
    oss << "Connection: close\r\n\r\n" << body;
    return oss.str();
}

static std::string http_html(const std::string& body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 200 OK\r\n"
        << "Content-Type: text/html; charset=utf-8\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Connection: close\r\n\r\n"
        << body;
    return oss.str();
}

static std::string error_body(const std::string& code, const std::string& message,
                              const std::string& param = "") {
    JsonValue err = JsonValue::object();
    JsonValue e = JsonValue::object();
    e.set("code", JsonValue::string(code));
    e.set("message", JsonValue::string(message));
    if (!param.empty()) e.set("param", JsonValue::string(param));
    err.set("error", e);
    return json_dumps(err) + "\n";
}

static std::string http_error(int code, const std::string& err_code, const std::string& message,
                              const std::string& param = "") {
    return http_json(code, error_body(err_code, message, param));
}

static std::string trim_copy(std::string s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static std::string http_header(const std::string& headers, const char* name) {
    std::string key = std::string(name) + ":";
    std::string lower_key = key;
    std::string lower = headers;
    std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    size_t pos = 0;
    while (pos < lower.size()) {
        size_t line_end = lower.find("\r\n", pos);
        if (line_end == std::string::npos) line_end = lower.size();
        if (lower.compare(pos, lower_key.size(), lower_key) == 0) {
            return trim_copy(headers.substr(pos + key.size(), line_end - (pos + key.size())));
        }
        if (line_end == lower.size()) break;
        pos = line_end + 2;
    }
    return "";
}

static bool bearer_matches(const std::string& authorization, const std::string& expected) {
    if (expected.empty()) return true;
    std::string auth = trim_copy(authorization);
    if (auth.size() < 7) return false;
    std::string scheme = auth.substr(0, 6);
    std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (scheme != "bearer") return false;
    return trim_copy(auth.substr(6)) == expected;
}

static std::string env_api_key() {
    const char* k = std::getenv("LAYA_API_KEY");
    if (k && *k) return k;
    k = std::getenv("TYPESAFE_API_KEY");
    if (k && *k) return k;
    return "";
}

static std::string family_description(const std::string& family) {
    if (family == "english") {
        return "English ModernBERT-large DecisionModel (convaiinnovations/laya) compiled by ggmlc.";
    }
    if (family == "multilingual") {
        return "Multilingual mmBERT DecisionModel (convaiinnovations/laya-multilingual) compiled by ggmlc.";
    }
    if (family == "typed-decisions") {
        return "English typed-decisions specialist (convaiinnovations/laya-typed-decisions) compiled by ggmlc.";
    }
    if (family == "kev-0.5b") {
        return "Kev 0.5B (Qwen2.5 + pointer head, jaredpalmer/kev-0.5b) compiled by ggmlc.";
    }
    if (family == "kev-0.8b") {
        return "Kev 0.8B (Qwen3.5 Gated DeltaNet + pointer head, jaredpalmer/kev-0.8b) compiled by ggmlc.";
    }
    if (family == "kev-4b") {
        return "Kev 4B (Qwen3.5 Gated DeltaNet + pointer head, jaredpalmer/kev-4b) compiled by ggmlc.";
    }
    return "System 1 DecisionModel compiled by ggmlc.";
}

Server::Server(DecisionEngine& engine, int port)
    : engine_(&engine), port_(port), running_(false), api_key_(env_api_key()) {}
Server::Server(DecisionRouter& router, int port)
    : router_(&router), port_(port), running_(false), api_key_(env_api_key()) {}
Server::~Server() { stop(); }
void Server::stop() { running_ = false; }

std::string Server::device() const {
    if (router_) return router_->device();
    if (engine_) return engine_->device();
    return "";
}

DecideResult Server::decide(const JsonValue& state, const std::vector<Question>& qs,
                            const std::string& model) {
    if (router_) return router_->decide(state, qs, model);
    if (engine_) {
        if (!model.empty()) {
            ModelRef ref = resolve_model_name(model);
            if (ref.unknown) throw std::runtime_error("unknown model '" + model + "'");
            if (!ref.auto_route && ref.family != engine_->family()) {
                throw std::runtime_error("unknown model '" + model + "'");
            }
        }
        return engine_->decide(state, qs);
    }
    throw std::runtime_error("no decision engine");
}

JsonValue Server::models_json() const {
    JsonValue root = JsonValue::object();
    JsonValue models = JsonValue::array();
    auto add = [&](const std::string& name, const std::string& desc) {
        JsonValue m = JsonValue::object();
        m.set("name", JsonValue::string(name));
        m.set("description", JsonValue::string(desc));
        m.set("release_date", JsonValue::string("2026-09-20"));
        models.arr.push_back(std::move(m));
    };
    std::vector<std::string> families;
    if (router_) families = router_->discovered_families();
    else if (engine_) families.push_back(engine_->family());
    for (const auto& fam : families) add(fam, family_description(fam));
    auto has = [&](const std::string& f) {
        return std::find(families.begin(), families.end(), f) != families.end();
    };
    if (has("english")) {
        add("laya", "Alias for the English Laya family.");
        add("jev-latest", "TypeSafe SDK default model alias; served locally by Laya English.");
    }
    if (has("multilingual")) {
        add("laya-multilingual", "Alias for the multilingual Laya family.");
    }
    if (has("typed-decisions")) {
        add("laya-typed-decisions", "Alias for the typed-decisions Laya family.");
    }
    if (has("kev-0.5b")) {
        add("kev", "Alias for kev-0.5b.");
        add("jaredpalmer/kev-0.5b", "Hugging Face id for kev-0.5b.");
    }
    if (has("kev-0.8b")) {
        add("jaredpalmer/kev-0.8b", "Hugging Face id for kev-0.8b.");
    }
    if (has("kev-4b")) {
        add("jaredpalmer/kev-4b", "Hugging Face id for kev-4b.");
    }
    root.set("models", std::move(models));
    return root;
}

static bool parse_system_one(const JsonValue& req, JsonValue& state, std::vector<Question>& qs,
                             std::string& model, std::string& err, std::string& param) {
    if (!req.is_object()) {
        err = "request body must be a JSON object";
        param = "";
        return false;
    }
    const JsonValue* st = req.get("state");
    if (!st) {
        err = "state is required";
        param = "state";
        return false;
    }
    if (!(st->is_string() || st->is_object() || st->is_array())) {
        err = "state must be a string, object, or array";
        param = "state";
        return false;
    }
    state = *st;
    const JsonValue* q = req.get("questions");
    if (!q) {
        err = "questions is required";
        param = "questions";
        return false;
    }
    std::string v = validate_questions_json(*q, &param);
    if (!v.empty()) {
        err = v;
        return false;
    }
    qs = questions_from_json(*q);
    if (const JsonValue* m = req.get("model")) {
        if (!m->is_string() || m->s.empty()) {
            err = "model must be a nonempty string";
            param = "model";
            return false;
        }
        model = m->s;
        ModelRef ref = resolve_model_name(model);
        if (ref.unknown) {
            err = "unknown model '" + model + "'";
            param = "model";
            return false;
        }
    }
    return true;
}

std::string Server::handle_request(const std::string& method, const std::string& path,
                                   const std::string& body, const std::string& authorization) {
    std::string p = path;
    auto qpos = p.find('?');
    if (qpos != std::string::npos) p = p.substr(0, qpos);

    if (p == "/favicon.ico") {
        return "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    }
    if (method == "OPTIONS") {
        return "HTTP/1.1 204 No Content\r\n"
               "Access-Control-Allow-Origin: *\r\n"
               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
               "Access-Control-Allow-Headers: Content-Type, Authorization, Accept, "
               "X-TypeSafe-SDK, X-TypeSafe-Runtime, X-TypeSafe-Retry-Count\r\n"
               "Content-Length: 0\r\nConnection: close\r\n\r\n";
    }
    if (method == "GET" && (p == "/" || p == "/index.html")) {
        return http_html(get_index_html());
    }

    const bool public_get = method == "GET" &&
                            (p == "/health" || p == "/" || p == "/index.html" || p == "/v1/presets");
    const bool needs_auth = !api_key_.empty() && !public_get;
    if (needs_auth && !bearer_matches(authorization, api_key_)) {
        return http_error(401, "authentication_error",
                          "Invalid API key. Pass Authorization: Bearer <key> or unset LAYA_API_KEY / TYPESAFE_API_KEY.");
    }

    if (method == "GET" && p == "/health") {
        if (router_) return http_json(200, json_dumps(router_->health_json()) + "\n");
        JsonValue o = JsonValue::object();
        o.set("status", JsonValue::string("ok"));
        o.set("model", JsonValue::string(engine_ ? engine_->model_name() : "laya"));
        o.set("device", JsonValue::string(device()));
        if (engine_) o.set("family", JsonValue::string(engine_->family()));
        return http_json(200, json_dumps(o) + "\n");
    }
    if (method == "GET" && p == "/v1/models") {
        return http_json(200, json_dumps_pretty(models_json()) + "\n");
    }
    if (method == "GET" && p == "/v1/presets") {
        JsonValue arr = JsonValue::array();
        for (const auto& pr : all_presets()) {
            JsonValue o = JsonValue::object();
            o.set("name", JsonValue::string(pr.name));
            o.set("title", JsonValue::string(pr.title));
            o.set("blurb", JsonValue::string(pr.blurb));
            o.set("state_key", JsonValue::string(pr.state_key));
            o.set("state", pr.default_state);
            o.set("questions", questions_to_json(pr.questions));
            arr.arr.push_back(std::move(o));
        }
        return http_json(200, json_dumps_pretty(arr) + "\n");
    }

    const bool system_one = method == "POST" && (p == "/v1/systemone" || p == "/v1/decide");
    const bool batch = method == "POST" && p == "/v1/decide/batch";
    if (system_one || batch) {
        JsonValue req;
        try {
            req = JsonParser::parse_string(body.empty() ? "{}" : body);
        } catch (const std::exception& e) {
            return http_error(400, "invalid_request_error", e.what());
        }

        auto run_one = [&](const JsonValue& one) -> std::string {
            JsonValue state;
            std::vector<Question> qs;
            std::string model;
            std::string err, param;
            if (!parse_system_one(one, state, qs, model, err, param)) {
                return http_error(422, "invalid_request_error", err, param);
            }
            auto t0 = std::chrono::steady_clock::now();
            DecideResult r;
            try {
                r = decide(state, qs, model);
            } catch (const std::exception& e) {
                std::string msg = e.what();
                if (msg.find("unknown model") != std::string::npos ||
                    msg.find("no GGUF catalogued") != std::string::npos) {
                    return http_error(400, "invalid_request_error", msg, "model");
                }
                return http_error(500, "internal_error", msg);
            }
            auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (r.latency_ms <= 0.0) r.latency_ms = ms;
            std::ostringstream hdr;
            hdr << static_cast<int>(r.latency_ms + 0.5);
            return http_json(200, format_answer_json(r, true) + "\n",
                             {{"X-Response-Time-Ms", hdr.str()}});
        };

        if (system_one) return run_one(req);

        if (!req.is_object()) return http_error(400, "invalid_request_error", "request body must be a JSON object");
        const JsonValue* states = req.get("states");
        const JsonValue* questions = req.get("questions");
        if (!states || !states->is_array() || states->arr.empty()) {
            return http_error(422, "invalid_request_error", "states must be a nonempty array", "states");
        }
        if (static_cast<int>(states->arr.size()) > kMaxBatchStates) {
            return http_error(400, "invalid_request_error",
                              "batch is limited to " + std::to_string(kMaxBatchStates) + " states", "states");
        }
        if (!questions) {
            return http_error(422, "invalid_request_error", "questions is required", "questions");
        }
        std::string param;
        std::string v = validate_questions_json(*questions, &param);
        if (!v.empty()) return http_error(422, "invalid_request_error", v, param);
        std::string model;
        if (const JsonValue* m = req.get("model")) {
            if (!m->is_string() || m->s.empty()) {
                return http_error(422, "invalid_request_error", "model must be a nonempty string", "model");
            }
            model = m->s;
            ModelRef ref = resolve_model_name(model);
            if (ref.unknown) return http_error(400, "invalid_request_error", "unknown model '" + model + "'", "model");
        }
        JsonValue results = JsonValue::array();
        auto t0 = std::chrono::steady_clock::now();
        try {
            for (const auto& st : states->arr) {
                JsonValue one = JsonValue::object();
                one.set("state", st);
                one.set("questions", *questions);
                if (!model.empty()) one.set("model", JsonValue::string(model));
                JsonValue state;
                std::vector<Question> qs;
                std::string mdl, err, pth;
                if (!parse_system_one(one, state, qs, mdl, err, pth)) {
                    return http_error(422, "invalid_request_error", err, pth);
                }
                DecideResult r = decide(state, qs, mdl);
                results.arr.push_back(JsonParser::parse_string(format_answer_json(r, false)));
            }
        } catch (const std::exception& e) {
            std::string msg = e.what();
            if (msg.find("unknown model") != std::string::npos ||
                msg.find("no GGUF catalogued") != std::string::npos) {
                return http_error(400, "invalid_request_error", msg, "model");
            }
            return http_error(500, "internal_error", msg);
        }
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        JsonValue root = JsonValue::object();
        root.set("results", std::move(results));
        std::ostringstream hdr;
        hdr << static_cast<int>(ms + 0.5);
        return http_json(200, json_dumps_pretty(root) + "\n", {{"X-Response-Time-Ms", hdr.str()}});
    }

    return http_error(404, "not_found_error", "Endpoint not found");
}

bool Server::start() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::cerr << "WSAStartup failed.\n";
        return false;
    }
#endif
    SOCKET server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        std::cerr << "Socket creation failed.\n";
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == SOCKET_ERROR) {
        std::cerr << "Bind failed on port " << port_ << "\n";
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    if (listen(server_fd, 10) == SOCKET_ERROR) {
        std::cerr << "Listen failed on port " << port_ << "\n";
        closesocket(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return false;
    }
    running_ = true;
    std::cout << "\n======================================================================\n"
              << " Laya System 1 Decision Studio  (TypeSafe-compatible)\n"
              << " -> Web UI     : http://localhost:" << port_ << "/\n"
              << " -> System One : POST http://localhost:" << port_ << "/v1/systemone\n"
              << " -> Models     : GET  http://localhost:" << port_ << "/v1/models\n"
              << " -> Health     : GET  http://localhost:" << port_ << "/health\n";
    if (!api_key_.empty()) {
        std::cout << " -> Auth       : Bearer token required (LAYA_API_KEY / TYPESAFE_API_KEY)\n";
    } else {
        std::cout << " -> Auth       : open (set LAYA_API_KEY or TYPESAFE_API_KEY to require Bearer)\n";
    }
    std::cout << "======================================================================\n\n" << std::flush;

    while (running_) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        SOCKET client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd == INVALID_SOCKET) {
            if (!running_) break;
            continue;
        }
#ifdef _WIN32
        DWORD timeout = 8000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = 8;
        tv.tv_usec = 0;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
#endif
        std::string req;
        char buffer[16384];
        size_t content_length = 0;
        bool headers_complete = false;
        while (running_) {
            int bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            if (bytes_read <= 0) break;
            req.append(buffer, bytes_read);
            if (!headers_complete) {
                size_t header_end = req.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    headers_complete = true;
                    std::string req_headers = req.substr(0, header_end);
                    std::string req_lower = req_headers;
                    std::transform(req_lower.begin(), req_lower.end(), req_lower.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    size_t cl_pos = req_lower.find("content-length:");
                    if (cl_pos != std::string::npos) {
                        size_t val_start = cl_pos + 15;
                        while (val_start < req_lower.size() &&
                               (req_lower[val_start] == ' ' || req_lower[val_start] == '\t'))
                            val_start++;
                        size_t val_end = val_start;
                        while (val_end < req_lower.size() &&
                               std::isdigit(static_cast<unsigned char>(req_lower[val_end])))
                            val_end++;
                        try {
                            content_length = std::stoull(req_lower.substr(val_start, val_end - val_start));
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }
            if (headers_complete) {
                size_t body_start = req.find("\r\n\r\n") + 4;
                if (req.size() - body_start >= content_length) break;
            }
        }
        if (!req.empty()) {
            std::stringstream ss(req);
            std::string method, path, version;
            ss >> method >> path >> version;
            std::string headers;
            std::string body;
            size_t body_pos = req.find("\r\n\r\n");
            if (body_pos != std::string::npos) {
                headers = req.substr(0, body_pos);
                body = req.substr(body_pos + 4);
            }
            const std::string authorization = http_header(headers, "Authorization");
            std::cout << "[laya] " << method << " " << path;
            if (!body.empty()) std::cout << " (" << body.size() << " bytes)";
            std::cout << std::endl;
            std::string response = handle_request(method, path, body, authorization);
            send_all(client_fd, response);
            shutdown(client_fd, SHUT_SEND);
        }
        closesocket(client_fd);
    }
    closesocket(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return true;
}

}  // namespace laya
