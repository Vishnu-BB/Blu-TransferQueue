#include "transferqueue/RpcServerBase.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <utility>

#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "utils/ThreadPool.h"

namespace tq {

namespace {
// Fixed, small, documented-as-tunable -- not derived from
// hardware_concurrency() to avoid an oversized pool on a big shared
// machine for a workload that's not CPU-bound (most of the work here is
// waiting on locks/condition variables, not computing).
constexpr std::size_t kWorkerThreads = 4;
} // namespace

struct RpcServerBase::Impl {
    zmq::context_t ctx;
    zmq::socket_t router{ctx, zmq::socket_type::router};
    std::atomic<bool> running{false};

    OwnTensor::utils::ThreadPool pool{kWorkerThreads};

    // Plain bytes, not zmq::message_t: message_t is move-only, which would
    // make the enqueue_detach lambda below move-only too -- incompatible
    // with std::function's copy-constructible requirement.
    std::mutex results_mutex;
    std::deque<std::pair<std::vector<std::uint8_t>, std::vector<std::uint8_t>>> results; // (identity, response)
};

RpcServerBase::RpcServerBase(std::string server_id) : server_id_(std::move(server_id)), impl_(std::make_unique<Impl>()) {}

RpcServerBase::~RpcServerBase() { stop(); }

std::string RpcServerBase::start(const std::string& address) {
    impl_->router.bind(address);
    std::string bound_address = impl_->router.get(zmq::sockopt::last_endpoint);

    // Poll with a timeout rather than relying on another thread forcibly
    // interrupting a blocking recv -- simpler and avoids any socket
    // thread-safety pitfalls around closing from a different thread. Short
    // (20ms): also the upper bound on how long a completed-but-not-yet-sent
    // reply can sit in results_ waiting for the loop to come back around.
    impl_->router.set(zmq::sockopt::rcvtimeo, 20);
    impl_->running = true;

    thread_ = std::thread([this]() {
        while (impl_->running.load()) {
            zmq::multipart_t request_frames;
            if (request_frames.recv(impl_->router) && request_frames.size() == 2) {
                zmq::message_t identity_msg = request_frames.pop();
                zmq::message_t payload = request_frames.pop();
                std::vector<std::uint8_t> identity(static_cast<const std::uint8_t*>(identity_msg.data()),
                                                    static_cast<const std::uint8_t*>(identity_msg.data()) +
                                                        identity_msg.size());
                std::vector<std::uint8_t> bytes(static_cast<const std::uint8_t*>(payload.data()),
                                                 static_cast<const std::uint8_t*>(payload.data()) + payload.size());

                impl_->pool.enqueue_detach([this, identity = std::move(identity), bytes = std::move(bytes)]() mutable {
                    Message response;
                    try {
                        Message request = Message::deserialize(bytes);
                        response = handle_request(request);
                    } catch (const std::exception& e) {
                        MessageBody error_body;
                        error_body.success = false;
                        error_body.error_message = e.what();
                        response = Message::create(RequestType::REQUEST_ERROR, server_id_, error_body);
                    }

                    auto response_bytes = response.serialize();
                    std::lock_guard<std::mutex> lock(impl_->results_mutex);
                    impl_->results.emplace_back(std::move(identity), std::move(response_bytes));
                });
            }
            // else: timed out (re-check running_) or a malformed envelope
            // (not [identity, payload]) was dropped -- either way, fall
            // through to draining whatever the pool has already finished.

            std::lock_guard<std::mutex> lock(impl_->results_mutex);
            while (!impl_->results.empty()) {
                auto [identity, response_bytes] = std::move(impl_->results.front());
                impl_->results.pop_front();

                zmq::multipart_t reply_frames;
                reply_frames.addmem(identity.data(), identity.size());
                reply_frames.addmem(response_bytes.data(), response_bytes.size());
                reply_frames.send(impl_->router);
            }
        }
    });

    return bound_address;
}

void RpcServerBase::stop() {
    if (!impl_->running.exchange(false)) {
        return; // already stopped (or never started)
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

}
