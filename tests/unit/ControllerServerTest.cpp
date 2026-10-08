// Unit tests for ControllerServer (include/transferqueue/ControllerServer.h,
// src/ControllerServer.cpp) -- Phase 5's metadata-only server: no
// StorageManager, holds a TransferQueueController. Real ZMQ sockets, real
// TransferQueueRpcClients -- no mocks. Build: `make unit_rpc_ControllerServerTest`.

#include "transferqueue/ControllerServer.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <zmq.hpp>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/Message.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

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
} // namespace

int main() {
    // ---- HANDSHAKE: only a shard_index+address announcement registers
    // anything; a plain connectivity check must not. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        CHECK_NOTHROW(client.handshake()); // plain connectivity check, no shard_index/address set
        CHECK(!client.get_shard_address(0).has_value()); // must NOT have registered shard 0 as a side effect

        client.announce_shard(0, "tcp://127.0.0.1:7000");
        auto resolved = client.get_shard_address(0);
        CHECK(resolved.has_value());
        CHECK(resolved.value() == "tcp://127.0.0.1:7000");

        // GET_SHARD_ADDRESS for a shard that was never announced.
        CHECK(!client.get_shard_address(999).has_value());

        // Re-announcing the same shard_index with a new address overwrites
        // (a shard can restart and rebind -- current behavior, no
        // versioning/rejection of a stale re-announcement).
        client.announce_shard(0, "tcp://127.0.0.1:7001");
        CHECK(client.get_shard_address(0).value() == "tcp://127.0.0.1:7001");

        server.stop();
    }

    // ---- DECLARE_SCHEMA + NOTIFY_DATA_UPDATE: success path updates the
    // real Controller's production status and shard attribution. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@ok";
        CHECK_NOTHROW(client.declare_schema(partition, {{"reward", tq::FieldDtype::Float32}}));
        CHECK_NOTHROW(client.notify_data_update(partition, {1, 2}, {"reward"}, {tq::FieldDtype::Float32},
                                                 /*shard_index=*/3));

        // Verify directly against the real underlying Controller object --
        // not just "the RPC call didn't throw".
        CHECK(controller->shard_for_sample(partition, 1) == 3);
        CHECK(controller->shard_for_sample(partition, 2) == 3);
        auto ready = controller->ready_indexes(partition, {"reward"}, "trainer");
        CHECK(ready.size() == 2);

        // shard_index omitted (left at the -1 default) must not record a
        // shard for the sample.
        CHECK_NOTHROW(client.notify_data_update(partition, {5}, {"reward"}, {tq::FieldDtype::Float32},
                                                 /*shard_index=*/-1));
        CHECK(controller->shard_for_sample(partition, 5) == -1);

        server.stop();
    }

    // ---- NOTIFY_DATA_UPDATE: schema-conflict failure path. No
    // two-phase-commit -- the (hypothetical) data already landed on a
    // shard, but the Controller must still refuse to mark it produced. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@conflict";
        client.declare_schema(partition, {{"reward", tq::FieldDtype::Float32}});

        // Conflicting dtype (declared Float32, claiming Int64) must throw,
        // and must NOT mark the sample produced.
        CHECK_THROWS(client.notify_data_update(partition, {7}, {"reward"}, {tq::FieldDtype::Int64},
                                                /*shard_index=*/0),
                     std::runtime_error);
        auto ready = controller->ready_indexes(partition, {"reward"}, "trainer");
        CHECK(ready.empty()); // sample 7 must not be ready despite the "write" having been attempted
        CHECK(controller->shard_for_sample(partition, 7) == -1); // shard attribution must not have been recorded either

        server.stop();
    }

    // ---- GET_META bundling: no StorageManager at all, so the response
    // must carry zero payload bytes, but must still correctly resolve
    // every distinct shard among the selected samples, plus Phase 6's
    // sample_versions/current_version. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@meta";
        client.announce_shard(10, "tcp://127.0.0.1:8010");
        client.announce_shard(20, "tcp://127.0.0.1:8020");

        client.notify_data_update(partition, {1}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/10);
        auto bumped = client.advance_version();
        CHECK(bumped == 1);
        client.notify_data_update(partition, {2}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/20);

        auto meta = client.get_meta(partition, {"reward"}, "trainer", /*batch_size=*/2);
        CHECK(meta.sample_ids.size() == 2);
        CHECK(meta.shard_addresses.size() == 2);
        CHECK(meta.shard_addresses.at(10) == "tcp://127.0.0.1:8010");
        CHECK(meta.shard_addresses.at(20) == "tcp://127.0.0.1:8020");

        std::unordered_map<tq::SampleId, std::int32_t> shard_by_id;
        std::unordered_map<tq::SampleId, std::int64_t> version_by_id;
        for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
            shard_by_id[meta.sample_ids[i]] = meta.sample_shard_indices[i];
            version_by_id[meta.sample_ids[i]] = meta.sample_versions[i];
        }
        CHECK(shard_by_id.at(1) == 10);
        CHECK(shard_by_id.at(2) == 20);
        CHECK(version_by_id.at(1) == 0); // produced before advance_version()
        CHECK(version_by_id.at(2) == 1); // produced after
        CHECK(meta.current_version == 1);

        // Wire-level check: a ControllerServer's GET_META_RESPONSE must
        // carry no tensor payload at all (it has no StorageManager to have
        // fetched one from) -- unlike the colocated Server's, which bundles
        // real data. Go around RpcClient (its MetaResult doesn't even
        // expose a payload field) with a raw DEALER socket.
        {
            zmq::context_t ctx;
            zmq::socket_t dealer(ctx, zmq::socket_type::dealer);
            dealer.set(zmq::sockopt::rcvtimeo, 5000);
            dealer.connect(addr);

            tq::MessageBody body;
            body.partition_id = partition;
            body.fields = {"reward"};
            body.task_name = "raw-check";
            body.batch_size = 2;
            auto request = tq::Message::create(tq::RequestType::GET_META, "raw-client", body);
            // 2 frames (header, payload) -- matches TransferQueueRpcClient's
            // real wire format (see RpcClient.cpp's call()).
            auto header = request.serialize_header();
            dealer.send(zmq::buffer(header), zmq::send_flags::sndmore);
            dealer.send(zmq::buffer(request.body.payload), zmq::send_flags::none);

            zmq::message_t reply_header;
            auto result = dealer.recv(reply_header, zmq::recv_flags::none);
            CHECK(result.has_value());
            zmq::message_t reply_payload;
            auto result2 = dealer.recv(reply_payload, zmq::recv_flags::none);
            CHECK(result2.has_value());
            std::vector<std::uint8_t> header_bytes(static_cast<const std::uint8_t*>(reply_header.data()),
                                                    static_cast<const std::uint8_t*>(reply_header.data()) +
                                                        reply_header.size());
            std::vector<std::uint8_t> payload_bytes(static_cast<const std::uint8_t*>(reply_payload.data()),
                                                     static_cast<const std::uint8_t*>(reply_payload.data()) +
                                                         reply_payload.size());
            auto decoded = tq::Message::deserialize_split(header_bytes, std::move(payload_bytes));
            CHECK(decoded.request_type == tq::RequestType::GET_META_RESPONSE);
            CHECK(decoded.body.payload.empty());
        }

        server.stop();
    }

    // ---- GET_META with a distinct task_name sees nothing consumed by a
    // different task, and RESET_CONSUMPTION makes a task's own consumed
    // samples ready again, same contract as the colocated server. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@reset";
        client.notify_data_update(partition, {1}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/-1);

        auto meta1 = client.get_meta(partition, {"reward"}, "trainer", 1);
        CHECK(meta1.sample_ids.size() == 1);
        auto meta2 = client.get_meta(partition, {"reward"}, "trainer", 1);
        CHECK(meta2.sample_ids.empty()); // already consumed by "trainer"
        auto meta_other_task = client.get_meta(partition, {"reward"}, "other_task", 1);
        CHECK(meta_other_task.sample_ids.size() == 1); // a different task hasn't consumed it

        CHECK_NOTHROW(client.reset_consumption(partition));
        auto meta3 = client.get_meta(partition, {"reward"}, "trainer", 1);
        CHECK(meta3.sample_ids.size() == 1); // ready again for "trainer" after reset

        server.stop();
    }

    // ---- CLEAR_PARTITION fan-out: two REAL StorageServers, both must
    // actually free bytes. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string ctrl_addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", ctrl_addr);

        auto shard0_storage = std::make_shared<tq::SimpleStorageManager>();
        auto shard1_storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer shard0("shard0", 0, shard0_storage);
        tq::StorageServer shard1("shard1", 1, shard1_storage);
        std::string shard0_addr = shard0.start("tcp://127.0.0.1:0");
        std::string shard1_addr = shard1.start("tcp://127.0.0.1:0");
        client.announce_shard(0, shard0_addr);
        client.announce_shard(1, shard1_addr);

        tq::TransferQueueRpcClient shard0_writer("w0", shard0_addr);
        tq::TransferQueueRpcClient shard1_writer("w1", shard1_addr);

        const std::string partition = "rollout@clear";
        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        shard0_writer.put(partition, {"reward"}, {{1, r1}});
        shard1_writer.put(partition, {"reward"}, {{2, r2}});
        client.notify_data_update(partition, {1}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/0);
        client.notify_data_update(partition, {2}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/1);

        CHECK(shard0_storage->current_bytes() > 0);
        CHECK(shard1_storage->current_bytes() > 0);

        CHECK_NOTHROW(client.clear_partition(partition));
        CHECK(shard0_storage->current_bytes() == 0);
        CHECK(shard1_storage->current_bytes() == 0);

        shard0.stop();
        shard1.stop();
        server.stop();
    }

    // ---- FIXED (was a real correctness/observability gap -- see
    // docs/UNIT_TEST_FINDINGS.md): a sample attributed to a shard_index
    // that was NEVER register_shard'd (shard_address() -> nullopt) used to
    // be silently skipped by the CLEAR_PARTITION fan-out -- no crash, no
    // throw, success=true regardless, even though that shard's bytes were
    // never actually cleared. Now the caller gets a loud signal: the
    // response comes back success=false with the unreachable shard index
    // named in error_message, which RpcClient::clear_partition() surfaces
    // as a thrown exception (the same throw_if_error() path every other
    // RpcClient method already uses for a failure response). ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@orphan";
        // shard_index 42 is attributed via NOTIFY_DATA_UPDATE but never
        // announced/registered -- shard_address(42) will resolve to nullopt
        // inside the server's CLEAR_PARTITION handler.
        client.notify_data_update(partition, {9}, {"reward"}, {tq::FieldDtype::Float32}, /*shard_index=*/42);
        CHECK(!client.get_shard_address(42).has_value());

        bool threw = false;
        std::string message;
        try {
            client.clear_partition(partition);
        } catch (const std::runtime_error& e) {
            threw = true;
            message = e.what();
        }
        CHECK(threw); // now a loud failure instead of a silent success
        CHECK(message.find("42") != std::string::npos); // names the unreachable shard

        // The Controller's own metadata is still cleared regardless of
        // whether the (unreachable) shard's bytes were -- confirm that
        // much at least actually happened (Controller can't "undo" its own
        // clear just because a remote shard was unreachable).
        auto ready = controller->ready_indexes(partition, {"reward"}, "trainer");
        CHECK(ready.empty());

        server.stop();
    }

    // (The all-shards-reachable success path -- clear_partition() must
    // still succeed cleanly, no false positives from the new
    // failure-reporting path -- is already covered by the "two REAL
    // StorageServers, both must actually free bytes" test above, which
    // asserts CHECK_NOTHROW on exactly this call shape.)

    // ---- Request types ControllerServer does not implement (storage-only:
    // PUT_DATA/GET_DATA/CLEAR_DATA belong to StorageServer) must come back
    // as a clean REQUEST_ERROR via the existing RpcClient methods -- never
    // a hang, and the client-side call must throw. ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer server("ctrl", controller, sampler);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        CHECK_THROWS(client.put("rollout@x", {"reward"}, {}), std::runtime_error);
        CHECK_THROWS(client.get_data("rollout@x", {1}, {"reward"}), std::runtime_error);
        CHECK_THROWS(client.clear_data("rollout@x", {1}), std::runtime_error);

        server.stop();
    }

    return tq::test::summary("ControllerServerTest");
}
