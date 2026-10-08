#include "transferqueue/RpcServerBase.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <utility>

#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "utils/ThreadPool.h"

namespace tq {

namespace {
constexpr std::size_t kWorkerThreads = 4;

// Same technique and threshold as RpcClient.cpp's send_owned_buffer() --
// zmq::multipart_t::addmem() always deep-copies into a new ZMQ message,
// which is fine for the small identity/header frames but wasteful for a
// reply payload that can be hundreds of MB (e.g. a GET_META_RESPONSE's
// batch data). zmq_msg_init_data lets ZMQ read directly out of `buffer`'s
// own memory instead.
void add_owned_buffer(zmq::multipart_t& frames, std::vector<std::uint8_t> buffer) {
    if (buffer.size() < 4096) {
        frames.addmem(buffer.data(), buffer.size());
        return;
    }
    auto* owner = new std::vector<std::uint8_t>(std::move(buffer));
    frames.add(zmq::message_t(
        owner->data(), owner->size(),
        [](void*, void* hint) { delete static_cast<std::vector<std::uint8_t>*>(hint); }, owner));
}
}

struct RpcServerBase::Impl {
    zmq::context_t ctx;
    zmq::socket_t router{ctx, zmq::socket_type::router};
    std::atomic<bool> running{false};

    OwnTensor::utils::ThreadPool pool{kWorkerThreads};

    struct PendingReply {
        std::vector<std::uint8_t> identity;
        std::vector<std::uint8_t> header;
        std::vector<std::uint8_t> payload;
    };

    std::mutex results_mutex;
    std::deque<PendingReply> results;

    // Self-pipe trick: a worker thread that just finished pushes one byte
    // here so the reactor's poll() below returns immediately instead of
    // only learning about the finished result once its router-side timeout
    // expires. wake_push is shared across the thread pool's worker threads,
    // which (like any ZMQ socket) is only safe under external
    // synchronization -- wake_mutex provides that; it's never held for more
    // than a 1-byte non-blocking send, so it's not a contention concern.
    zmq::socket_t wake_pull{ctx, zmq::socket_type::pull};
    zmq::socket_t wake_push{ctx, zmq::socket_type::push};
    std::mutex wake_mutex;
};

RpcServerBase::RpcServerBase(std::string server_id) : server_id_(std::move(server_id)), impl_(std::make_unique<Impl>()) {}

RpcServerBase::~RpcServerBase() { stop(); }

std::string RpcServerBase::start(const std::string& address) {
    impl_->router.bind(address);
    std::string bound_address = impl_->router.get(zmq::sockopt::last_endpoint);

    // inproc address is scoped to this instance's own zmq::context_t, so a
    // fixed name can't collide with another RpcServerBase's wake pipe.
    impl_->wake_pull.bind("inproc://rpc-server-wake");
    impl_->wake_push.connect("inproc://rpc-server-wake");

    impl_->running = true;

    thread_ = std::thread([this]() {
        zmq_pollitem_t items[2] = {
            {impl_->router.handle(), 0, ZMQ_POLLIN, 0},
            {impl_->wake_pull.handle(), 0, ZMQ_POLLIN, 0},
        };

        while (impl_->running.load()) {
            // 20ms is now just a safety-net fallback (covers the case where
            // results land between the previous drain and this poll, with
            // no new wake byte in flight) -- the common case returns as
            // soon as either a real request or a worker's wake byte is
            // ready, not after waiting out the full timeout.
            zmq::poll(items, 2, std::chrono::milliseconds(20));

            if (items[0].revents & ZMQ_POLLIN) {
                // 3 frames: ROUTER's own identity envelope, then the
                // header/payload pair TransferQueueRpcClient sends (see
                // RpcClient.cpp's call()) -- keeping payload as its own
                // frame, never merged with the header into one buffer, is
                // what avoids the extra full-payload copy on both ends. See
                // docs/TransferQueue-Benchmark.md's Section 8/9.
                zmq::multipart_t request_frames;
                if (request_frames.recv(impl_->router, ZMQ_DONTWAIT) && request_frames.size() == 3) {
                    zmq::message_t identity_msg = request_frames.pop();
                    zmq::message_t header_msg = request_frames.pop();
                    zmq::message_t payload_msg = request_frames.pop();
                    std::vector<std::uint8_t> identity(static_cast<const std::uint8_t*>(identity_msg.data()),
                                                        static_cast<const std::uint8_t*>(identity_msg.data()) +
                                                            identity_msg.size());
                    std::vector<std::uint8_t> header_bytes(static_cast<const std::uint8_t*>(header_msg.data()),
                                                            static_cast<const std::uint8_t*>(header_msg.data()) +
                                                                header_msg.size());
                    std::vector<std::uint8_t> payload_bytes(static_cast<const std::uint8_t*>(payload_msg.data()),
                                                             static_cast<const std::uint8_t*>(payload_msg.data()) +
                                                                 payload_msg.size());

                    impl_->pool.enqueue_detach([this, identity = std::move(identity), header_bytes = std::move(header_bytes),
                                                 payload_bytes = std::move(payload_bytes)]() mutable {
                        Message response;
                        try {
                            Message request = Message::deserialize_split(header_bytes, std::move(payload_bytes));
                            response = handle_request(request);
                        } catch (const std::exception& e) {
                            MessageBody error_body;
                            error_body.success = false;
                            error_body.error_message = e.what();
                            response = Message::create(RequestType::REQUEST_ERROR, server_id_, error_body);
                        }

                        auto response_header = response.serialize_header();
                        auto response_payload = std::move(response.body.payload);
                        {
                            std::lock_guard<std::mutex> lock(impl_->results_mutex);
                            impl_->results.push_back(Impl::PendingReply{std::move(identity), std::move(response_header),
                                                                         std::move(response_payload)});
                        }
                        std::lock_guard<std::mutex> wake_lock(impl_->wake_mutex);
                        static const std::uint8_t kWakeByte = 0;
                        impl_->wake_push.send(zmq::buffer(&kWakeByte, 1), zmq::send_flags::dontwait);
                    });
                }
            }

            if (items[1].revents & ZMQ_POLLIN) {
                // Drain every pending wake byte -- how many arrived doesn't
                // matter, only that at least one did; the loop below always
                // drains the whole results queue regardless.
                zmq::message_t ping;
                while (impl_->wake_pull.recv(ping, zmq::recv_flags::dontwait)) {
                }
            }

            std::lock_guard<std::mutex> lock(impl_->results_mutex);
            while (!impl_->results.empty()) {
                auto reply = std::move(impl_->results.front());
                impl_->results.pop_front();

                zmq::multipart_t reply_frames;
                reply_frames.addmem(reply.identity.data(), reply.identity.size());
                reply_frames.addmem(reply.header.data(), reply.header.size());
                add_owned_buffer(reply_frames, std::move(reply.payload));
                reply_frames.send(impl_->router);
            }
        }
    });

    return bound_address;
}

void RpcServerBase::stop() {
    if (!impl_->running.exchange(false)) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
}

}
