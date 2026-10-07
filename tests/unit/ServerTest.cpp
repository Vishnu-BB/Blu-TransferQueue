// Unit tests for the colocated TransferQueueServer (include/transferqueue/
// Server.h, src/Server.cpp) and, through it, the shared RpcServerBase
// request loop (include/transferqueue/RpcServerBase.h,
// src/RpcServerBase.cpp -- it has no public surface beyond start()/stop(),
// already exercised here). Real ZMQ ROUTER socket over real loopback TCP,
// no mocks. Build: `make unit_rpc_ServerTest` (needs Tensor-Implementations
// + cppzmq, same as `make test_rpc`).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <zmq.hpp>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"

#include "TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

Tensor make_scalar_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

// Sends a hand-built Message over a raw DEALER socket and returns the
// decoded reply -- for request shapes TransferQueueRpcClient has no method
// for (an unsupported request type) or wouldn't build (a corrupted payload).
tq::Message raw_roundtrip(const std::string& address, const tq::Message& request, int timeout_ms = 5000) {
    zmq::context_t ctx;
    zmq::socket_t dealer(ctx, zmq::socket_type::dealer);
    dealer.set(zmq::sockopt::rcvtimeo, timeout_ms);
    dealer.connect(address);

    auto bytes = request.serialize();
    dealer.send(zmq::buffer(bytes), zmq::send_flags::none);

    zmq::message_t reply;
    auto result = dealer.recv(reply, zmq::recv_flags::none);
    CHECK(result.has_value()); // must not hang/drop -- exactly one reply, always
    if (!result.has_value()) {
        return tq::Message{}; // caller's subsequent CHECKs will just fail informatively
    }
    std::vector<std::uint8_t> reply_bytes(static_cast<const std::uint8_t*>(reply.data()),
                                           static_cast<const std::uint8_t*>(reply.data()) + reply.size());
    return tq::Message::deserialize(reply_bytes);
}

struct Harness {
    std::shared_ptr<tq::TransferQueueController> controller = std::make_shared<tq::TransferQueueController>();
    std::shared_ptr<tq::SimpleStorageManager> storage = std::make_shared<tq::SimpleStorageManager>();
    std::shared_ptr<tq::FifoSampler> sampler = std::make_shared<tq::FifoSampler>();
    tq::TransferQueueServer server{"srv", controller, storage, sampler};
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client{"cli", address};

    ~Harness() { server.stop(); }
};

} // namespace

int main() {
    // ---- Success path for every RequestType the switch actually handles ----
    {
        Harness h;

        CHECK_NOTHROW(h.client.handshake());

        CHECK_NOTHROW(h.client.declare_schema("p@schema", {{"reward", tq::FieldDtype::Float32}}));

        tq::Record r;
        r["reward"] = make_scalar_f32(3.0f);
        CHECK_NOTHROW(h.client.put("p@schema", {"reward"}, {{1, r}}));

        auto got = h.client.get("p@schema", {"reward"}, "trainer", 10);
        CHECK(got.size() == 1);
        CHECK(got.count(1) == 1);

        CHECK_NOTHROW(h.client.clear_data("p@schema", {1}));
        CHECK(h.storage->current_bytes() == 0);

        tq::Record r2;
        r2["reward"] = make_scalar_f32(4.0f);
        h.client.put("p@clear_partition", {"reward"}, {{1, r2}});
        CHECK_NOTHROW(h.client.clear_partition("p@clear_partition"));

        CHECK_NOTHROW(h.client.reset_consumption("p@schema"));

        auto v1 = h.client.advance_version();
        auto v2 = h.client.advance_version();
        CHECK(v2 == v1 + 1);
    }

    // ---- Unhandled request types: exactly one well-formed REQUEST_ERROR
    //      reply, never a hang, never silence. GET_DATA and
    //      NOTIFY_DATA_UPDATE are both StorageServer/ControllerServer
    //      (Phase 5) concerns, not implemented on this colocated server. ----
    {
        Harness h;

        tq::MessageBody body;
        body.partition_id = "whatever";
        body.sample_ids = {1};
        body.fields = {"reward"};
        auto reply = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::GET_DATA, "raw", body));
        CHECK(reply.request_type == tq::RequestType::REQUEST_ERROR);
        CHECK(!reply.body.success);
        CHECK(!reply.body.error_message.empty());

        tq::MessageBody body2;
        body2.partition_id = "whatever";
        auto reply2 = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::NOTIFY_DATA_UPDATE, "raw", body2));
        CHECK(reply2.request_type == tq::RequestType::REQUEST_ERROR);
        CHECK(!reply2.body.success);
    }

    // ---- Malformed PUT_DATA payload: too short to deserialize as a real
    //      batch -- must still come back as a clean REQUEST_ERROR, not a
    //      crash or a hang. ----
    {
        Harness h;

        tq::MessageBody body;
        body.partition_id = "p@malformed";
        body.payload = {0x01, 0x02}; // far too short for serialize_batch's real encoding
        auto reply = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::PUT_DATA, "raw", body));
        CHECK(reply.request_type == tq::RequestType::REQUEST_ERROR);
        CHECK(!reply.body.success);

        // Also try a truncated-but-nonempty payload and a payload that's
        // valid-length noise -- same contract either way.
        tq::MessageBody body2;
        body2.partition_id = "p@malformed";
        body2.payload = {}; // empty
        auto reply2 = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::PUT_DATA, "raw", body2));
        CHECK(reply2.request_type == tq::RequestType::REQUEST_ERROR);

        tq::MessageBody body3;
        body3.partition_id = "p@malformed";
        body3.payload = std::vector<std::uint8_t>(37, 0xFF); // plausible-length garbage
        auto reply3 = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::PUT_DATA, "raw", body3));
        CHECK(reply3.request_type == tq::RequestType::REQUEST_ERROR);
    }

    // ---- HANDSHAKE on the COLOCATED server: unconditional ACK regardless
    //      of shard_index/address in the request body, and -- unlike
    //      ControllerServer's HANDSHAKE handler -- no shard registration
    //      side effect, since this server's Controller's shard registry
    //      isn't part of this deployment shape at all. ----
    {
        Harness h;

        tq::MessageBody body;
        body.shard_index = 7;
        body.address = "tcp://127.0.0.1:9999";
        auto reply = raw_roundtrip(h.address, tq::Message::create(tq::RequestType::HANDSHAKE, "raw", body));
        CHECK(reply.request_type == tq::RequestType::HANDSHAKE_ACK);

        // The colocated Server's HANDSHAKE handler ignores shard_index/
        // address entirely -- confirm nothing was registered.
        CHECK(!h.controller->shard_address(7).has_value());

        // A plain handshake (no shard fields set) behaves identically --
        // still just an ACK, still no registration.
        CHECK_NOTHROW(h.client.handshake());
        CHECK(!h.controller->shard_address(-1).has_value());
    }

    // ---- GET_META response bundling: sample_versions/current_version are
    //      correct in the raw MessageBody, not just in whatever get()
    //      extracts from the payload. Multiple samples written at different
    //      global versions within the same partition. ----
    {
        Harness h;

        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        h.client.put("p@versions", {"reward"}, {{1, r1}}); // version 0

        h.client.advance_version(); // -> 1
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        h.client.put("p@versions", {"reward"}, {{2, r2}}); // version 1

        h.client.advance_version(); // -> 2
        h.client.advance_version(); // -> 3
        tq::Record r3;
        r3["reward"] = make_scalar_f32(3.0f);
        h.client.put("p@versions", {"reward"}, {{3, r3}}); // version 3

        auto meta = h.client.get_meta("p@versions", {"reward"}, "trainer", 10);
        CHECK(meta.sample_ids.size() == 3);
        CHECK(meta.current_version == 3);

        std::unordered_map<tq::SampleId, std::int64_t> version_by_id;
        for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
            version_by_id[meta.sample_ids[i]] = meta.sample_versions[i];
        }
        CHECK(version_by_id.at(1) == 0);
        CHECK(version_by_id.at(2) == 1);
        CHECK(version_by_id.at(3) == 3);
    }

    // ---- Concurrency: mixed request types (declare_schema, put, get,
    //      advance_version) across many threads, each owning its own
    //      partition (so correctness doesn't need cross-thread
    //      coordination) except for advance_version, which is genuinely
    //      shared global state -- collecting every returned value across
    //      all threads must be exactly the set {1..total_advances} with no
    //      duplicates and no gaps, which only holds if Controller's
    //      lock-protected ++current_version_ is actually atomic under
    //      concurrent RPC load, not just single-threaded. ----
    {
        Harness h;
        constexpr int kThreads = 10;
        constexpr int kAdvancesPerThread = 5;
        std::vector<std::thread> threads;
        std::vector<char> thread_ok(kThreads, 0);
        std::mutex versions_mutex;
        std::vector<std::int64_t> all_versions;

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                tq::TransferQueueRpcClient tc("thread-" + std::to_string(t), h.address);
                std::string partition = "p@concurrency_" + std::to_string(t);
                bool ok = true;

                try {
                    tc.declare_schema(partition, {{"reward", tq::FieldDtype::Float32}});
                    for (int i = 0; i < 15; ++i) {
                        tq::Record r;
                        r["reward"] = make_scalar_f32(static_cast<float>(t * 100 + i));
                        tc.put(partition, {"reward"}, {{static_cast<tq::SampleId>(i), r}});
                    }
                    auto got = tc.get(partition, {"reward"}, "trainer", 15);
                    if (got.size() != 15) ok = false;
                    for (auto& [id, record] : got) {
                        float expected = static_cast<float>(t * 100 + static_cast<int>(id));
                        float actual = static_cast<float*>(record.at("reward").data())[0];
                        if (actual != expected) ok = false;
                    }

                    std::vector<std::int64_t> local_versions;
                    for (int i = 0; i < kAdvancesPerThread; ++i) {
                        local_versions.push_back(tc.advance_version());
                    }
                    std::lock_guard<std::mutex> lock(versions_mutex);
                    all_versions.insert(all_versions.end(), local_versions.begin(), local_versions.end());
                } catch (const std::exception&) {
                    ok = false;
                }
                thread_ok[t] = ok ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) CHECK(thread_ok[t]);

        CHECK(all_versions.size() == static_cast<std::size_t>(kThreads * kAdvancesPerThread));
        std::sort(all_versions.begin(), all_versions.end());
        bool no_dup_no_gap = true;
        for (std::size_t i = 0; i < all_versions.size(); ++i) {
            if (all_versions[i] != static_cast<std::int64_t>(i + 1)) {
                no_dup_no_gap = false;
                break;
            }
        }
        CHECK(no_dup_no_gap);
    }

    // Regression for a fixed bug (see docs/UNIT_TEST_FINDINGS.md): PUT_DATA
    // used to mark every field in body.fields as produced for a sample
    // regardless of whether that sample's record actually contained it. A
    // sample whose record is missing a "claimed" field must now correctly
    // NOT be reported ready for it.
    {
        Harness h;
        h.client.declare_schema("gap", {{"a", tq::FieldDtype::Float32}, {"b", tq::FieldDtype::Float32}});
        tq::Record partial;
        partial["a"] = make_scalar_f32(1.0f); // "b" deliberately omitted
        h.client.put("gap", {"a", "b"}, {{1, partial}});

        auto not_ready = h.client.get("gap", {"a", "b"}, "trainer", 10);
        CHECK(not_ready.empty()); // "b" was never produced -> not ready for both

        auto ready_a_only = h.client.get("gap", {"a"}, "trainer", 10);
        CHECK(ready_a_only.size() == 1);
        CHECK(ready_a_only.at(1).count("a") == 1);
    }

    // ---- Abrupt client disconnect mid-transfer: a client sends a request
    // then closes its socket (and context) before ever reading the reply.
    // The server's ROUTER socket writes the reply keyed by the DEALER's
    // identity frame; with no ZMQ_ROUTER_MANDATORY set, a send to a
    // vanished peer is simply dropped, not an error the request-loop thread
    // has to handle -- confirm that holds in practice: the server must not
    // crash or hang, and must still serve a brand new client afterward. ----
    {
        Harness h;
        h.client.declare_schema("p@disconnect", {{"reward", tq::FieldDtype::Float32}});

        {
            zmq::context_t ctx;
            zmq::socket_t dealer(ctx, zmq::socket_type::dealer);
            dealer.connect(h.address);

            tq::MessageBody body;
            body.partition_id = "p@disconnect";
            tq::Record r;
            r["reward"] = make_scalar_f32(1.0f);
            body.payload = tq::serialize_batch({{1, r}});
            body.sample_ids = {1};
            body.fields = {"reward"};
            auto request = tq::Message::create(tq::RequestType::PUT_DATA, "vanishing", body);
            auto bytes = request.serialize();
            dealer.send(zmq::buffer(bytes), zmq::send_flags::none);
            // Socket and context destroyed here, with no recv() -- the
            // reply the server is about to send has nowhere to land.
        }

        // Give the request loop a moment to actually process the request
        // and attempt (and drop) its reply before checking the server is
        // still alive.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Server must still be fully responsive to a fresh client.
        CHECK_NOTHROW(h.client.handshake());
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        CHECK_NOTHROW(h.client.put("p@disconnect", {"reward"}, {{2, r2}}));
        auto got = h.client.get("p@disconnect", {"reward"}, "trainer", 10);
        // The vanished client's own write (sample 1) still completed
        // server-side -- only its reply was undeliverable, the request
        // itself was fully processed before that -- so both samples show
        // up here, which is itself confirmation the server kept working
        // normally through the disconnect rather than wedging on it.
        CHECK(got.size() == 2);
        CHECK(got.count(1) == 1);
        CHECK(got.count(2) == 1);
    }

    return tq::test::summary("ServerTest");
}
