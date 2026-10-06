// Unit tests for TransferQueueRpcClient (include/transferqueue/RpcClient.h,
// src/RpcClient.cpp) -- every public method, against a real colocated
// TransferQueueServer over real loopback TCP. Build:
// `make unit_rpc_RpcClientTest` (needs Tensor-Implementations + cppzmq,
// same as `make test_rpc`).

#include <string>
#include <thread>
#include <vector>

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

struct Harness {
    std::shared_ptr<tq::TransferQueueController> controller = std::make_shared<tq::TransferQueueController>();
    std::shared_ptr<tq::SimpleStorageManager> storage = std::make_shared<tq::SimpleStorageManager>();
    std::shared_ptr<tq::FifoSampler> sampler = std::make_shared<tq::FifoSampler>();
    tq::TransferQueueServer server{"srv", controller, storage, sampler};
    std::string address = server.start("tcp://127.0.0.1:0");

    ~Harness() { server.stop(); }
};

} // namespace

int main() {
    // ---- handshake() ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        CHECK_NOTHROW(client.handshake());
        CHECK_NOTHROW(client.handshake()); // idempotent, callable repeatedly
    }

    // ---- declare_schema() + put()/get() round trip, including batch_size
    //      edge values (0, exactly the ready count, more than available). ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);

        CHECK_NOTHROW(client.declare_schema("p@basic", {{"reward", tq::FieldDtype::Float32}}));

        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        client.put("p@basic", {"reward"}, {{1, r1}, {2, r2}});

        // batch_size=0 -> nothing selected, not an error.
        auto none = client.get("p@basic", {"reward"}, "trainer", 0);
        CHECK(none.empty());

        // batch_size larger than what's ready -> returns everything ready,
        // not an error/short-count failure.
        auto all = client.get("p@basic", {"reward"}, "trainer", 1000);
        CHECK(all.size() == 2);
        CHECK(static_cast<float*>(all.at(1).at("reward").data())[0] == 1.0f);
        CHECK(static_cast<float*>(all.at(2).at("reward").data())[0] == 2.0f);

        // Already consumed by "trainer" -- a repeat get for the same task
        // sees nothing.
        auto again = client.get("p@basic", {"reward"}, "trainer", 1000);
        CHECK(again.empty());

        // A different task_name hasn't consumed anything yet.
        auto other_task = client.get("p@basic", {"reward"}, "other_task", 1000);
        CHECK(other_task.size() == 2);
    }

    // ---- get_meta(): sample_ids, sample_shard_indices (all -1 on the
    //      colocated server -- no shard registry participation at all),
    //      shard_addresses (empty -- nothing was ever registered),
    //      sample_versions, current_version. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);

        tq::Record r;
        r["reward"] = make_scalar_f32(5.0f);
        client.put("p@meta", {"reward"}, {{1, r}});

        auto meta = client.get_meta("p@meta", {"reward"}, "trainer", 10);
        CHECK(meta.sample_ids.size() == 1 && meta.sample_ids[0] == 1);
        // The colocated Server's GET_META handler never populates
        // sample_shard_indices/shard_registry_* at all (that bundling is
        // ControllerServer-only, Phase 5) -- it stays empty here, not a
        // vector of -1 sentinels.
        CHECK(meta.sample_shard_indices.empty());
        CHECK(meta.shard_addresses.empty());
        CHECK(meta.sample_versions.size() == 1 && meta.sample_versions[0] == 0);
        CHECK(meta.current_version == 0);
    }

    // ---- clear_data(): a partial clear by explicit sample id, leaving
    //      other samples in the same partition untouched. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);

        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        client.put("p@clear_data", {"reward"}, {{1, r1}, {2, r2}});
        std::size_t before = h.storage->current_bytes();
        CHECK(before == 8); // two float32 scalars

        CHECK_NOTHROW(client.clear_data("p@clear_data", {1}));
        CHECK(h.storage->current_bytes() == before - 4); // only sample 1's 4 bytes freed

        auto remaining = client.get("p@clear_data", {"reward"}, "trainer", 10);
        CHECK(remaining.size() == 1);
        CHECK(remaining.count(2) == 1);
        CHECK(remaining.count(1) == 0);
    }

    // ---- clear_partition() + reset_consumption() interaction: clearing a
    //      partition wipes both production and consumption state, so
    //      re-writing the same sample id afterward makes it immediately
    //      ready again even for a task that had already consumed it. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);

        tq::Record r;
        r["reward"] = make_scalar_f32(1.0f);
        client.put("p@clear_partition", {"reward"}, {{1, r}});
        client.get("p@clear_partition", {"reward"}, "trainer", 10); // consume it

        CHECK_NOTHROW(client.clear_partition("p@clear_partition"));
        CHECK(h.storage->current_bytes() == 0);

        tq::Record r2;
        r2["reward"] = make_scalar_f32(9.0f);
        client.put("p@clear_partition", {"reward"}, {{1, r2}}); // same id, fresh partition
        auto got = client.get("p@clear_partition", {"reward"}, "trainer", 10);
        CHECK(got.size() == 1); // not blocked by the old (now-cleared) consumption record

        // reset_consumption() alone (no clear) makes already-consumed
        // samples visible again without touching storage/production.
        client.put("p@reset", {"reward"}, {{1, r}});
        client.get("p@reset", {"reward"}, "trainer", 10);
        CHECK(client.get("p@reset", {"reward"}, "trainer", 10).empty());
        CHECK_NOTHROW(client.reset_consumption("p@reset"));
        CHECK(client.get("p@reset", {"reward"}, "trainer", 10).size() == 1);
    }

    // ---- advance_version(): strictly increasing from a fresh controller,
    //      starting at 1. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        CHECK(client.advance_version() == 1);
        CHECK(client.advance_version() == 2);
        CHECK(client.advance_version() == 3);
    }

    // ---- announce_shard() against the colocated server: the call itself
    //      succeeds (HANDSHAKE always ACKs here), but has no registration
    //      side effect -- this server's shard registry is simply never
    //      consulted in this deployment shape. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        CHECK_NOTHROW(client.announce_shard(3, "tcp://127.0.0.1:1234"));
        CHECK(!h.controller->shard_address(3).has_value());
    }

    // ---- get_data() against the colocated server: no GET_DATA handler
    //      exists here (that's StorageServer's job, Phase 5) -- must throw,
    //      and quickly (the server replies with REQUEST_ERROR, this is not
    //      the 5s timeout path). ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        CHECK_THROWS(client.get_data("p@whatever", {1}, {"reward"}), std::runtime_error);
    }

    // ---- get_shard_address() against the colocated server: a real,
    //      slightly surprising behavioral quirk worth locking in --
    //      GET_SHARD_ADDRESS is also unhandled here, but get_shard_address()
    //      treats ANY unsuccessful response as "not found" rather than
    //      propagating the error, so it returns nullopt rather than
    //      throwing (unlike get_data(), which does throw on the same kind
    //      of unhandled-request-type failure).
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        std::optional<std::string> result;
        CHECK_NOTHROW(result = client.get_shard_address(0));
        CHECK(!result.has_value());
    }

    // ---- notify_data_update() against the colocated server: also
    //      unhandled here (ControllerServer-only) -- must throw, with a
    //      non-empty message. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);
        CHECK_THROWS(client.notify_data_update("p@whatever", {1}, {"reward"}, {tq::FieldDtype::Float32}, 0),
                     std::runtime_error);
    }

    // ---- Error propagation: a real server-side failure must surface as a
    //      thrown exception with a non-empty, meaningful message, not a
    //      silent no-op or a generic/empty one. Two distinct failure
    //      sources: (a) put() after declare_schema() with a conflicting
    //      dtype, (b) declare_schema() called twice with conflicting
    //      dtypes for the same field. ----
    {
        Harness h;
        tq::TransferQueueRpcClient client("cli", h.address);

        client.declare_schema("p@errors", {{"reward", tq::FieldDtype::Float32}});
        tq::Record bad;
        bad["reward"] = Tensor(Shape{{1}}, Dtype::Int64); // declared Float32, this is Int64
        bool threw_a = false;
        std::string message_a;
        try {
            client.put("p@errors", {"reward"}, {{1, bad}});
        } catch (const std::runtime_error& e) {
            threw_a = true;
            message_a = e.what();
        }
        CHECK(threw_a);
        CHECK(!message_a.empty());

        bool threw_b = false;
        std::string message_b;
        try {
            client.declare_schema("p@errors", {{"reward", tq::FieldDtype::Int64}}); // conflicts with Float32 above
        } catch (const std::runtime_error& e) {
            threw_b = true;
            message_b = e.what();
        }
        CHECK(threw_b);
        CHECK(!message_b.empty());
    }

    // ---- Nothing listening: a client constructed against a dead address
    //      must time out and throw, not hang forever. Exactly one such
    //      case (the dealer socket's rcvtimeo is 5000ms, so this is
    //      inherently a slow test -- keep it singular).
    //
    //      FIXED (was a real bug -- see docs/UNIT_TEST_FINDINGS.md):
    //      TransferQueueRpcClient now sets ZMQ_LINGER=0 on its DEALER
    //      socket. Before this fix, handshake()'s undelivered HANDSHAKE
    //      message (no peer ever accepts it) made destroying the client
    //      afterward block indefinitely on ZMQ's default LINGER=-1 -- this
    //      test used to have to deliberately leak the client object to
    //      avoid hanging the whole suite on exit. Now the client is
    //      destroyed normally (real RAII, no leak) and this test completing
    //      at all is itself proof the fix works -- a regression would hang
    //      this entire binary, not just fail a single CHECK.
    {
        std::string dead_address;
        {
            Harness h;
            dead_address = h.address; // capture the real bound port
        } // h's destructor stops the server -- nothing is listening now

        tq::TransferQueueRpcClient client("cli", dead_address);
        CHECK_THROWS(client.handshake(), std::runtime_error);
        // client's destructor runs here, at end of scope -- must return
        // promptly (LINGER=0), not hang.
    }

    // ---- Multiple independent client objects (not threads sharing one
    //      client -- genuinely separate TransferQueueRpcClient instances,
    //      each its own DEALER socket) talking to the same server
    //      concurrently from their own threads. ----
    {
        Harness h;
        constexpr int kClients = 6;
        std::vector<std::thread> threads;
        std::vector<char> ok(kClients, 0);

        for (int i = 0; i < kClients; ++i) {
            threads.emplace_back([&, i]() {
                tq::TransferQueueRpcClient own_client("independent-" + std::to_string(i), h.address);
                std::string partition = "p@independent_" + std::to_string(i);
                tq::Record r;
                r["reward"] = make_scalar_f32(static_cast<float>(i));
                own_client.put(partition, {"reward"}, {{1, r}});
                auto got = own_client.get(partition, {"reward"}, "trainer", 10);
                bool good = got.size() == 1 && got.count(1) == 1 &&
                            static_cast<float*>(got.at(1).at("reward").data())[0] == static_cast<float>(i);
                ok[i] = good ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();
        for (int i = 0; i < kClients; ++i) CHECK(ok[i]);
    }

    return tq::test::summary("RpcClientTest");
}
