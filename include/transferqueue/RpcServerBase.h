#pragma once

#include <memory>
#include <string>
#include <thread>

#include "transferqueue/Message.h"

namespace tq {

// Shared ZMQ ROUTER request-loop machinery: one I/O thread (recv/send
// only -- ZMQ sockets aren't safe to use from multiple threads), request
// *processing* handed to a thread pool so one slow/blocked request doesn't
// stall the I/O thread from serving independent ones. Always exactly one
// reply per request (handler exception -> REQUEST_ERROR), matching
// upstream's actual contract. Extracted from the original monolithic
// TransferQueueServer (Phase 3/4) once ControllerServer/StorageServer
// (Phase 5) needed the identical plumbing with a different handle_request.
//
// zmq::context_t/socket_t are hidden behind Impl so callers don't need
// <zmq.hpp> in their include path.
class RpcServerBase {
public:
    explicit RpcServerBase(std::string server_id);
    virtual ~RpcServerBase();

    RpcServerBase(const RpcServerBase&) = delete;
    RpcServerBase& operator=(const RpcServerBase&) = delete;

    // Binds to `address` (e.g. "tcp://127.0.0.1:0" for an ephemeral port)
    // and starts the background request-loop thread. Returns the actually-
    // bound endpoint (resolves ":0" to the real port) once bind succeeds.
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
