#pragma once

#include <memory>
#include <string>
#include <thread>

#include "transferqueue/Message.h"

namespace tq {
class RpcServerBase {
public:
    explicit RpcServerBase(std::string server_id);
    virtual ~RpcServerBase();

    RpcServerBase(const RpcServerBase&) = delete;
    RpcServerBase& operator=(const RpcServerBase&) = delete;

    std::string start(const std::string& address);
    void stop();

protected:
    const std::string& server_id() const { return server_id_; }

private:
    virtual Message handle_request(const Message& request) = 0;

    std::string server_id_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::thread thread_;
};

}
