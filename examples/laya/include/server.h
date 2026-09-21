#pragma once

#include "engine.h"
#include "questions.h"

#include <string>

namespace laya {

class DecisionRouter;

class Server {
public:
    Server(DecisionEngine& engine, int port = 8080);
    Server(DecisionRouter& router, int port = 8080);
    ~Server();
    bool start();
    void stop();

private:
    DecisionEngine* engine_ = nullptr;
    DecisionRouter* router_ = nullptr;
    int port_ = 8080;
    bool running_ = false;
    std::string api_key_;
    std::string handle_request(const std::string& method, const std::string& path,
                               const std::string& body, const std::string& authorization);
    std::string device() const;
    DecideResult decide(const JsonValue& state, const std::vector<Question>& qs,
                        const std::string& model);
    JsonValue models_json() const;
};

}  // namespace laya
