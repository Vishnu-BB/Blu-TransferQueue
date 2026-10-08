// Unit tests for StorageServer (include/transferqueue/StorageServer.h,
// src/StorageServer.cpp) -- Phase 5's data-only server: no
// TransferQueueController, holds a StorageManager. Real ZMQ sockets, real
// TransferQueueRpcClients -- no mocks. Build: `make unit_rpc_StorageServerTest`.

#include "transferqueue/StorageServer.h"

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>

#include "core/Tensor.h"
#include "transferqueue/Message.h"
#include "transferqueue/RpcClient.h"
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
Tensor make_token_ids(const std::vector<std::int64_t>& ids) {
    Tensor t(Shape{{static_cast<std::int64_t>(ids.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < ids.size(); ++i) data[i] = ids[i];
    return t;
}
} // namespace

int main() {
    // ---- HANDSHAKE always just ACKs -- no registration logic lives here
    // at all (that's ControllerServer's job), including when shard_index/
    // address happen to be set (as if someone mistakenly pointed
    // announce_shard() at a StorageServer instead of a ControllerServer). ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        CHECK_NOTHROW(client.handshake());
        CHECK_NOTHROW(client.announce_shard(5, "tcp://127.0.0.1:1234")); // harmless no-op here, must not throw

        server.stop();
    }

    // ---- shard_index() accessor: matches construction, unaffected by
    // traffic served afterward. ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 7, storage);
        CHECK(server.shard_index() == 7);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);
        tq::Record r;
        r["reward"] = make_scalar_f32(1.0f);
        client.put("p", {"reward"}, {{1, r}});
        CHECK(server.shard_index() == 7); // unchanged by serving a request
        server.stop();

        // Negative shard_index (unassigned placeholder, or just an unusual
        // value) must round-trip exactly, not clamp/wrap.
        tq::StorageServer unassigned("shard-neg", -1, storage);
        CHECK(unassigned.shard_index() == -1);
    }

    // ---- PUT_DATA/GET_DATA/CLEAR_DATA round trip: multiple samples,
    // multiple fields, multiple dtypes, byte-for-byte correctness. ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@0";
        tq::Record r1;
        r1["prompt_ids"] = make_token_ids({11, 12, 13});
        r1["reward"] = make_scalar_f32(0.5f);
        tq::Record r2;
        r2["prompt_ids"] = make_token_ids({21, 22});
        r2["reward"] = make_scalar_f32(-2.25f);
        client.put(partition, {"prompt_ids", "reward"}, {{1, r1}, {2, r2}});

        CHECK(storage->current_bytes() > 0);

        auto got = client.get_data(partition, {1, 2}, {"prompt_ids", "reward"});
        CHECK(got.size() == 2);
        CHECK(got.count(1) == 1 && got.count(2) == 1);
        CHECK(got.at(1).at("reward").dtype() == Dtype::Float32);
        CHECK(static_cast<float*>(got.at(1).at("reward").data())[0] == 0.5f);
        CHECK(static_cast<float*>(got.at(2).at("reward").data())[0] == -2.25f);
        auto& p1 = got.at(1).at("prompt_ids");
        CHECK(p1.dtype() == Dtype::Int64);
        CHECK(p1.numel() == 3);
        auto* p1data = static_cast<std::int64_t*>(p1.data());
        CHECK(p1data[0] == 11 && p1data[1] == 12 && p1data[2] == 13);

        // GET_DATA for an id that was never written: must not throw, must
        // simply not appear in the result (not an empty Record either).
        auto partial = client.get_data(partition, {1, 999}, {"reward"});
        CHECK(partial.size() == 1);
        CHECK(partial.count(1) == 1);
        CHECK(partial.count(999) == 0);

        // GET_DATA requesting a field that was never written for a sample
        // that DOES exist: the sample comes back, just without that field
        // -- and, now that get_data() filters by the requested fields (see
        // below), also without any OTHER field that wasn't asked for.
        auto missing_field = client.get_data(partition, {1}, {"reward", "nonexistent_field"});
        CHECK(missing_field.count(1) == 1);
        CHECK(missing_field.at(1).count("reward") == 1);
        CHECK(missing_field.at(1).count("nonexistent_field") == 0);
        CHECK(missing_field.at(1).count("prompt_ids") == 0); // not requested -> not returned

        // FIXED (was a real asymmetry bug -- see docs/UNIT_TEST_FINDINGS.md):
        // SimpleStorageManager::get_data() now filters by the requested
        // `fields`, matching put_data()'s own filtering. Requesting only
        // "reward" for sample 1 (which also has "prompt_ids" stored) no
        // longer leaks "prompt_ids" back.
        auto over_fetch = client.get_data(partition, {1}, {"reward"}); // asked for ONLY "reward"
        CHECK(over_fetch.at(1).count("reward") == 1);
        CHECK(over_fetch.at(1).count("prompt_ids") == 0); // correctly filtered out now

        std::size_t before_clear = storage->current_bytes();
        CHECK_NOTHROW(client.clear_data(partition, {1}));
        CHECK(storage->current_bytes() < before_clear); // sample 1's bytes freed
        auto after_clear = client.get_data(partition, {1, 2}, {"prompt_ids", "reward"});
        CHECK(after_clear.count(1) == 0); // gone
        CHECK(after_clear.count(2) == 1); // sample 2 untouched by clearing sample 1

        server.stop();
    }

    // ---- No schema validation at all: there's no TransferQueueController
    // here to validate against, so writing the SAME field name with
    // different dtypes across different samples (or even overwriting the
    // same sample/field with a different dtype) must succeed unconditionally
    // -- unlike the colocated Server/ControllerServer, which would refuse
    // this once a schema is declared. Confirms the documented "fail-fast
    // only happens later via NOTIFY_DATA_UPDATE" limitation is actually
    // true of this code, not just claimed in a doc comment. ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        const std::string partition = "rollout@noschema";
        tq::Record float_record;
        float_record["reward"] = make_scalar_f32(1.0f);
        CHECK_NOTHROW(client.put(partition, {"reward"}, {{1, float_record}}));

        tq::Record int_record;
        int_record["reward"] = make_token_ids({42}); // same field name "reward", now Int64 -- no validation to stop it
        CHECK_NOTHROW(client.put(partition, {"reward"}, {{2, int_record}}));

        auto got = client.get_data(partition, {1, 2}, {"reward"});
        CHECK(got.at(1).at("reward").dtype() == Dtype::Float32);
        CHECK(got.at(2).at("reward").dtype() == Dtype::Int64); // both dtypes coexist, unvalidated

        // Overwriting sample 1's "reward" with a different dtype in a later
        // PUT_DATA must also succeed unconditionally.
        tq::Record overwrite;
        overwrite["reward"] = make_token_ids({7, 8});
        CHECK_NOTHROW(client.put(partition, {"reward"}, {{1, overwrite}}));
        auto got2 = client.get_data(partition, {1}, {"reward"});
        CHECK(got2.at(1).at("reward").dtype() == Dtype::Int64);

        server.stop();
    }

    // ---- Request types StorageServer does not implement (control-plane
    // only: DECLARE_SCHEMA/GET_META/CLEAR_PARTITION/RESET_CONSUMPTION/
    // GET_SHARD_ADDRESS/NOTIFY_DATA_UPDATE/ADVANCE_VERSION all belong to
    // ControllerServer) must come back as a clean REQUEST_ERROR -- never a
    // hang -- and the client-side call must throw. ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        CHECK_THROWS(client.declare_schema("p", {{"reward", tq::FieldDtype::Float32}}), std::runtime_error);
        CHECK_THROWS(client.get_meta("p", {"reward"}, "trainer", 1), std::runtime_error);
        CHECK_THROWS(client.clear_partition("p"), std::runtime_error);
        CHECK_THROWS(client.reset_consumption("p"), std::runtime_error);
        CHECK(!client.get_shard_address(0).has_value()); // success=false -> nullopt, not a throw (matches get_shard_address's own contract)
        CHECK_THROWS(client.notify_data_update("p", {1}, {"reward"}, {tq::FieldDtype::Float32}, 0),
                     std::runtime_error);
        CHECK_THROWS(client.advance_version(), std::runtime_error);

        server.stop();
    }

    // ---- Exception safety: a PUT_DATA whose payload fails to deserialize
    // must still get exactly one well-formed REQUEST_ERROR reply, not a
    // hang or a crash. Goes around RpcClient with a raw DEALER socket,
    // same pattern as tests/test_rpc_smoke.cpp's equivalent check for the
    // colocated Server. ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");

        zmq::context_t ctx;
        zmq::socket_t dealer(ctx, zmq::socket_type::dealer);
        dealer.set(zmq::sockopt::rcvtimeo, 5000);
        dealer.connect(addr);

        tq::MessageBody body;
        body.partition_id = "p";
        body.payload = {0x01, 0x02}; // too short to be a valid serialize_batch encoding
        auto bad_request = tq::Message::create(tq::RequestType::PUT_DATA, "bad-client", body);
        // 2 frames (header, payload) -- matches TransferQueueRpcClient's
        // real wire format (see RpcClient.cpp's call()).
        auto header = bad_request.serialize_header();
        dealer.send(zmq::buffer(header), zmq::send_flags::sndmore);
        dealer.send(zmq::buffer(bad_request.body.payload), zmq::send_flags::none);

        zmq::message_t reply_header;
        auto result = dealer.recv(reply_header, zmq::recv_flags::none);
        CHECK(result.has_value()); // didn't hang
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
        CHECK(decoded.request_type == tq::RequestType::REQUEST_ERROR);
        CHECK(!decoded.body.success);

        // The server must still be alive and serving correctly afterward --
        // a malformed request must not have corrupted or wedged its state.
        tq::TransferQueueRpcClient client("client", addr);
        tq::Record r;
        r["reward"] = make_scalar_f32(3.0f);
        CHECK_NOTHROW(client.put("p", {"reward"}, {{1, r}}));
        auto got = client.get_data("p", {1}, {"reward"});
        CHECK(got.count(1) == 1);

        server.stop();
    }

    // ---- Multiple independent StorageServer instances (different
    // shard_index, different SimpleStorageManager) running concurrently:
    // no cross-talk between them even when written with the same
    // partition_id/sample_id/field combination. ----
    {
        auto storage_a = std::make_shared<tq::SimpleStorageManager>();
        auto storage_b = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer server_a("shard-a", 0, storage_a);
        tq::StorageServer server_b("shard-b", 1, storage_b);
        std::string addr_a = server_a.start("tcp://127.0.0.1:0");
        std::string addr_b = server_b.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client_a("client-a", addr_a);
        tq::TransferQueueRpcClient client_b("client-b", addr_b);

        // Same partition_id, same sample_id, same field name, different
        // values -- must land in genuinely separate storage, not alias.
        tq::Record ra;
        ra["reward"] = make_scalar_f32(100.0f);
        tq::Record rb;
        rb["reward"] = make_scalar_f32(200.0f);
        client_a.put("rollout@shared_name", {"reward"}, {{1, ra}});
        client_b.put("rollout@shared_name", {"reward"}, {{1, rb}});

        auto got_a = client_a.get_data("rollout@shared_name", {1}, {"reward"});
        auto got_b = client_b.get_data("rollout@shared_name", {1}, {"reward"});
        CHECK(static_cast<float*>(got_a.at(1).at("reward").data())[0] == 100.0f);
        CHECK(static_cast<float*>(got_b.at(1).at("reward").data())[0] == 200.0f);

        // Clearing shard A's copy must not affect shard B's.
        client_a.clear_data("rollout@shared_name", {1});
        CHECK(client_a.get_data("rollout@shared_name", {1}, {"reward"}).empty());
        CHECK(client_b.get_data("rollout@shared_name", {1}, {"reward"}).count(1) == 1);

        CHECK(server_a.shard_index() == 0);
        CHECK(server_b.shard_index() == 1);

        server_a.stop();
        server_b.stop();
    }

    // ---- The oversized-write deadlock fix (see
    // tests/unit/StorageManagerTest.cpp) applies over the real wire path
    // too: a single PUT_DATA batch exceeding the shard's capacity_bytes
    // must get a clean REQUEST_ERROR reply, never hang the server. Bounded
    // wait on the client call itself is the real safety net here -- a
    // regression would otherwise hang this whole test binary waiting on
    // recv(). ----
    {
        auto storage = std::make_shared<tq::SimpleStorageManager>(/*capacity_bytes=*/8);
        tq::StorageServer server("shard", 0, storage);
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("client", addr);

        tq::Record oversized;
        oversized["reward"] = make_token_ids({1, 2}); // 16 bytes -- exceeds the 8-byte capacity on its own

        std::promise<bool> threw;
        auto threw_future = threw.get_future();
        std::thread caller([&client, oversized, p = std::move(threw)]() mutable {
            bool did_throw = false;
            try {
                client.put("rollout@oversized", {"reward"}, {{1, oversized}});
            } catch (const std::exception&) {
                did_throw = true;
            }
            p.set_value(did_throw);
        });
        bool returned_promptly = threw_future.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
        CHECK(returned_promptly); // must come back as a clean error, never hang
        if (returned_promptly) {
            CHECK(threw_future.get()); // RpcClient surfaces the server's REQUEST_ERROR as an exception
            caller.join();
        } else {
            caller.detach(); // avoid hanging the test binary on a real regression
        }
        CHECK(storage->current_bytes() == 0); // the oversized write must not have partially landed

        // The server must still be alive and correctly serving requests
        // afterward -- a thrown handler exception must not have taken the
        // request loop down.
        tq::Record fits;
        fits["reward"] = make_token_ids({1}); // 8 bytes -- exactly fits
        CHECK_NOTHROW(client.put("rollout@oversized", {"reward"}, {{2, fits}}));
        CHECK(storage->current_bytes() == 8);

        server.stop();
    }

    return tq::test::summary("StorageServerTest");
}
