// Needs Tensor-Implementations + cppzmq -- build with `make test_rpc`, no
// mpirun needed. Runs a real TransferQueueServer (ROUTER) on a background
// thread and a real TransferQueueRpcClient (DEALER) in main(), talking over
// an actual loopback TCP socket -- not an in-process shortcut.
//
// Two threads in one process rather than two OS processes (fork()): a ZMQ
// context isn't fork-safe once sockets exist on it, and threads already
// exercise the real ROUTER/DEALER wire path (TCP loopback) the walkthrough
// cared about proving.
//
// Run with: ./test_rpc_smoke

#include <cassert>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/Message.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

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
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();

    tq::TransferQueueServer server("server-1", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0"); // ephemeral port

    tq::TransferQueueRpcClient client("client-1", address);

    // Handshake round trip over the real socket.
    client.handshake();

    // Schema validation (Option B) over the real socket: declare_schema(),
    // then a put() with a conflicting dtype throws (the server's PUT_DATA
    // handler validates before writing), a matching dtype succeeds.
    {
        client.declare_schema("rollout@schema", {{"reward", tq::FieldDtype::Float32}});

        tq::Record good;
        good["reward"] = make_scalar_f32(2.5f);
        client.put("rollout@schema", {"reward"}, {{1, good}}); // matches -- no throw

        tq::Record bad;
        bad["reward"] = Tensor(Shape{{1}}, Dtype::Int64);
        bool threw = false;
        try {
            client.put("rollout@schema", {"reward"}, {{2, bad}});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    // Full put -> get round trip, same invariants proven in-process back in
    // Phase 0/2, now over a real ROUTER/DEALER socket boundary.
    // current_bytes() is global across the whole StorageManager (shared
    // with the schema-validation test above), so capture a baseline rather
    // than asserting an absolute value -- isolates this section's own
    // contribution regardless of what else has been written.
    std::size_t bytes_baseline = storage->current_bytes();
    tq::Record record;
    record["reward"] = make_scalar_f32(4.5f);
    client.put("rollout@0", {"reward"}, {{3, record}});

    auto got = client.get("rollout@0", {"reward"}, "trainer", /*batch_size=*/10);
    assert(got.size() == 1);
    assert(static_cast<float*>(got.at(3).at("reward").data())[0] == 4.5f);

    // Consumed by "trainer" -- a second get for the same task over the wire
    // sees nothing left, same as the in-process Client.
    auto got_again = client.get("rollout@0", {"reward"}, "trainer", /*batch_size=*/10);
    assert(got_again.empty());

    // CLEAR_PARTITION round trip -- and confirm it actually frees storage
    // bytes now, not just Controller metadata. Consuming via get() above
    // didn't remove anything from storage (only an explicit clear does), so
    // the 4 bytes from the float32 "reward" scalar are still sitting there
    // until this call.
    assert(storage->current_bytes() == bytes_baseline + 4);
    client.clear_partition("rollout@0", /*clear_consumption=*/true);
    assert(storage->current_bytes() == bytes_baseline);

    // Unmatched/erroring requests still get exactly one well-formed reply,
    // never a hang -- reset_consumption on a partition that no longer
    // exists (just cleared) is a no-op server-side, not an error, but
    // proves the round trip completes either way.
    client.reset_consumption("rollout@0");

    // Phase 6: ADVANCE_VERSION + GET_META bundling over the real socket --
    // the colocated TransferQueueServer tracks staleness exactly like the
    // split ControllerServer does (see the Phase 5 block below).
    {
        assert(client.advance_version() == 1); // fresh counter for this shared controller, first bump
        tq::Record r;
        r["reward"] = make_scalar_f32(9.5f);
        client.put("rollout@version", {"reward"}, {{1, r}});
        assert(client.advance_version() == 2);

        auto meta = client.get_meta("rollout@version", {"reward"}, "trainer", /*batch_size=*/10);
        assert(meta.sample_ids.size() == 1 && meta.sample_ids[0] == 1);
        assert((meta.sample_versions == std::vector<std::int64_t>{1})); // stamped right after the first advance
        assert(meta.current_version == 2);
    }

    // Server's exception-safety contract: a PUT_DATA whose payload fails to
    // deserialize must still get exactly one well-formed REQUEST_ERROR
    // reply, not a hang or a crash. RpcClient only ever builds well-formed
    // requests, so this goes around it with a raw DEALER socket.
    {
        zmq::context_t ctx;
        zmq::socket_t dealer(ctx, zmq::socket_type::dealer);
        dealer.set(zmq::sockopt::rcvtimeo, 5000);
        dealer.connect(address);

        tq::MessageBody body;
        body.partition_id = "rollout@0";
        body.payload = {0x01, 0x02}; // too short to be a valid serialize_batch encoding
        auto bad_request = tq::Message::create(tq::RequestType::PUT_DATA, "bad-client", body);
        // 2 frames (header, payload) -- matches TransferQueueRpcClient's
        // real wire format (see RpcClient.cpp's call()).
        auto header = bad_request.serialize_header();
        dealer.send(zmq::buffer(header), zmq::send_flags::sndmore);
        dealer.send(zmq::buffer(bad_request.body.payload), zmq::send_flags::none);

        zmq::message_t reply_header;
        auto result = dealer.recv(reply_header, zmq::recv_flags::none);
        assert(result.has_value()); // didn't hang -- a reply actually came back
        zmq::message_t reply_payload;
        auto result2 = dealer.recv(reply_payload, zmq::recv_flags::none);
        assert(result2.has_value());

        std::vector<std::uint8_t> header_bytes(static_cast<const std::uint8_t*>(reply_header.data()),
                                                static_cast<const std::uint8_t*>(reply_header.data()) +
                                                    reply_header.size());
        std::vector<std::uint8_t> payload_bytes(static_cast<const std::uint8_t*>(reply_payload.data()),
                                                 static_cast<const std::uint8_t*>(reply_payload.data()) +
                                                     reply_payload.size());
        auto decoded = tq::Message::deserialize_split(header_bytes, std::move(payload_bytes));
        assert(decoded.request_type == tq::RequestType::REQUEST_ERROR);
        assert(!decoded.body.success);
    }

    // GPU tensor over the real wire: PUT_DATA/GET_META go through
    // serialize_batch/deserialize_batch (Phase 2's Message::payload
    // bridge), which load_tensor always reconstructs on CPU regardless of
    // the original device (confirmed directly in test_tensor_smoke.cpp's
    // unit-level test of that bridge). Confirming the same holds true
    // end-to-end over the actual socket path, not just at the bridge
    // function in isolation.
    {
        tq::Record gpu_record;
        gpu_record["reward"] = make_scalar_f32(6.5f).to_cuda(0);
        client.put("rollout@gpu", {"reward"}, {{1, gpu_record}});

        auto got_gpu = client.get("rollout@gpu", {"reward"}, "trainer", /*batch_size=*/10);
        assert(got_gpu.size() == 1);
        Tensor& result = got_gpu.at(1).at("reward");
        assert(!result.is_cuda()); // came back on CPU -- the wire path always does this
        assert(static_cast<float*>(result.data())[0] == 6.5f); // value still correct
    }

    // Concurrency: N client threads, each its own TransferQueueRpcClient
    // (one ZMQ socket per thread -- sockets aren't shareable across
    // threads) hammering the same server concurrently, each with its own
    // partition so correctness-checking doesn't need any cross-thread
    // coordination. Proves the thread-pool-backed request loop (Server.cpp)
    // and the locking added to Controller/DataPartitionStatus/
    // PartitionIndexManager/StorageManager actually hold up under real
    // concurrent access, not just "compiles and the single-client tests
    // still pass."
    {
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 20;
        std::vector<std::thread> threads;
        // char, not vector<bool>: vector<bool> is bit-packed, so concurrent
        // writes to different "elements" from different threads is itself
        // a data race on the underlying shared word.
        std::vector<char> thread_ok(kThreads, 0);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                tq::TransferQueueRpcClient thread_client("stress-" + std::to_string(t), address);
                std::string partition = "stress@" + std::to_string(t);
                bool ok = true;
                for (int i = 0; i < kOpsPerThread; ++i) {
                    float expected = static_cast<float>(t * 1000 + i);
                    tq::Record r;
                    r["reward"] = make_scalar_f32(expected);
                    thread_client.put(partition, {"reward"}, {{static_cast<tq::SampleId>(i), r}});

                    auto result = thread_client.get(partition, {"reward"}, "trainer", /*batch_size=*/1);
                    if (result.size() != 1 || result.count(static_cast<tq::SampleId>(i)) != 1) {
                        std::cerr << "thread " << t << " op " << i << ": expected 1 result for sample " << i
                                  << ", got " << result.size() << " results\n";
                        ok = false;
                        continue;
                    }
                    float actual = static_cast<float*>(result.at(static_cast<tq::SampleId>(i)).at("reward").data())[0];
                    if (actual != expected) {
                        std::cerr << "thread " << t << " op " << i << ": expected " << expected << " got " << actual
                                  << "\n";
                        ok = false;
                    }
                }
                thread_ok[t] = ok ? 1 : 0;
            });
        }
        for (auto& th : threads) {
            th.join();
        }
        for (int t = 0; t < kThreads; ++t) {
            assert(thread_ok[t]);
        }
    }

    server.stop();

    // GRPOGroupNSampler wired into a real colocated server via GET_META: a
    // dangling, incomplete group must never be handed to a reader, even
    // when it's the only thing ready -- only a full, consecutive
    // n_samples_per_prompt run is ever returned.
    {
        auto grpo_controller = std::make_shared<tq::TransferQueueController>();
        auto grpo_storage = std::make_shared<tq::SimpleStorageManager>();
        auto grpo_sampler = std::make_shared<tq::GRPOGroupNSampler>(/*n_samples_per_prompt=*/2);
        tq::TransferQueueServer grpo_server("grpo-server", grpo_controller, grpo_storage, grpo_sampler);
        std::string grpo_address = grpo_server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient grpo_client("grpo-client", grpo_address);

        // Group "group-A" ({0, 1}) is complete; "group-B" has only sample 2
        // so far (its partner, 3, hasn't been written yet) -- not a
        // complete group of 2. Two ids belonging to different groups can't
        // share one put() call (one group_id per call), so two calls.
        tq::Record r0;
        r0["reward"] = make_scalar_f32(0.0f);
        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        grpo_client.put("rollout@grpo", {"reward"}, {{0, r0}, {1, r1}}, "group-A");
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        grpo_client.put("rollout@grpo", {"reward"}, {{2, r2}}, "group-B");

        auto first = grpo_client.get("rollout@grpo", {"reward"}, "trainer", /*batch_size=*/2);
        assert(first.size() == 2);
        assert(first.count(0) == 1 && first.count(1) == 1);
        assert(first.count(2) == 0); // the dangling single must not leak into a batch

        // Only sample 2 is left ready, still not a complete group -- a
        // second GET_META must come back empty, not a partial/short batch.
        auto second = grpo_client.get("rollout@grpo", {"reward"}, "trainer", /*batch_size=*/2);
        assert(second.empty());

        // Writing sample 3 completes "group-B" ({2, 3}); now it's selectable.
        tq::Record r3;
        r3["reward"] = make_scalar_f32(3.0f);
        grpo_client.put("rollout@grpo", {"reward"}, {{3, r3}}, "group-B");
        auto third = grpo_client.get("rollout@grpo", {"reward"}, "trainer", /*batch_size=*/2);
        assert(third.size() == 2);
        assert(third.count(2) == 1 && third.count(3) == 1);

        grpo_server.stop();
    }

    // FIND_STRANDED_GROUPS over a real socket: track-and-expose only --
    // confirms the report is correct AND confirms nothing is actually
    // consumed/cleared by it (a subsequent real GET_META still sees
    // everything exactly as before).
    {
        auto stranded_controller = std::make_shared<tq::TransferQueueController>();
        auto stranded_storage = std::make_shared<tq::SimpleStorageManager>();
        auto stranded_sampler = std::make_shared<tq::GRPOGroupNSampler>(/*n_samples_per_prompt=*/2);
        tq::TransferQueueServer stranded_server("stranded-server", stranded_controller, stranded_storage,
                                                 stranded_sampler);
        std::string stranded_address = stranded_server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient stranded_client("stranded-client", stranded_address);

        // "group-A" ({0,1}) complete -- must never be reported, at any
        // age. "group-B" ({2}) incomplete. Sample 3 has no group_id at
        // all -- always eligible.
        tq::Record r0;
        r0["reward"] = make_scalar_f32(0.0f);
        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.0f);
        stranded_client.put("rollout@stranded", {"reward"}, {{0, r0}, {1, r1}}, "group-A");
        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.0f);
        stranded_client.put("rollout@stranded", {"reward"}, {{2, r2}}, "group-B");
        tq::Record r3;
        r3["reward"] = make_scalar_f32(3.0f);
        stranded_client.put("rollout@stranded", {"reward"}, {{3, r3}}); // no group_id

        auto stranded = stranded_client.find_stranded_groups("rollout@stranded", {"reward"}, "trainer",
                                                               /*max_age_ms=*/0);
        assert(stranded.size() == 2); // "group-B" and "" -- never "group-A"
        bool found_b = false, found_ungrouped = false;
        for (const auto& group : stranded) {
            assert(group.group_id != "group-A");
            assert(group.oldest_age_ms >= 0);
            if (group.group_id == "group-B") {
                found_b = true;
                assert((group.sample_ids == std::vector<tq::SampleId>{2}));
            } else if (group.group_id.empty()) {
                found_ungrouped = true;
                assert((group.sample_ids == std::vector<tq::SampleId>{3}));
            }
        }
        assert(found_b && found_ungrouped);

        // A threshold nothing meets -> empty report, no error.
        auto stranded_far_future =
            stranded_client.find_stranded_groups("rollout@stranded", {"reward"}, "trainer", /*max_age_ms=*/3600000);
        assert(stranded_far_future.empty());

        // Nothing was consumed/cleared by any of the above: "group-A" is
        // still there, ready, and fetchable exactly as if find_stranded_groups
        // had never been called.
        auto got = stranded_client.get("rollout@stranded", {"reward"}, "trainer", /*batch_size=*/2);
        assert(got.size() == 2);
        assert(got.count(0) == 1 && got.count(1) == 1);

        stranded_server.stop();
    }

    // Phase 5: ControllerServer (metadata only) + two StorageServers (data
    // only), wired together exactly like real processes would be -- each
    // shard announces itself via HANDSHAKE, a writer writes straight to a
    // shard then NOTIFY_DATA_UPDATEs the controller, and a reader does the
    // real two-hop path: GET_META against the controller, then GET_DATA
    // against whichever shard(s) it was told to use.
    {
        auto p5_controller = std::make_shared<tq::TransferQueueController>();
        auto p5_sampler = std::make_shared<tq::FifoSampler>();
        tq::ControllerServer controller_server("controller-1", p5_controller, p5_sampler);
        std::string controller_address = controller_server.start("tcp://127.0.0.1:0");

        auto shard0_storage = std::make_shared<tq::SimpleStorageManager>();
        auto shard1_storage = std::make_shared<tq::SimpleStorageManager>();
        tq::StorageServer shard0("shard-0", /*shard_index=*/0, shard0_storage);
        tq::StorageServer shard1("shard-1", /*shard_index=*/1, shard1_storage);
        std::string shard0_address = shard0.start("tcp://127.0.0.1:0");
        std::string shard1_address = shard1.start("tcp://127.0.0.1:0");

        tq::TransferQueueRpcClient shard0_announcer("shard-0", controller_address);
        shard0_announcer.announce_shard(0, shard0_address);
        tq::TransferQueueRpcClient shard1_announcer("shard-1", controller_address);
        shard1_announcer.announce_shard(1, shard1_address);

        // A GET_SHARD_ADDRESS lookup resolves what HANDSHAKE just
        // registered -- and nullopt for a shard that was never announced.
        tq::TransferQueueRpcClient controller_client("writer-1", controller_address);
        auto resolved_shard0 = controller_client.get_shard_address(0);
        assert(resolved_shard0.has_value() && *resolved_shard0 == shard0_address);
        assert(!controller_client.get_shard_address(42).has_value());

        // Write sample 1 to shard 0, sample 2 to shard 1 -- a stand-in for
        // GRPO group-affinity routing actually picking the shard. Each
        // write goes straight to its shard, then NOTIFY_DATA_UPDATE tells
        // the controller where it landed.
        tq::TransferQueueRpcClient shard0_writer("writer-1", shard0_address);
        tq::TransferQueueRpcClient shard1_writer("writer-1", shard1_address);

        tq::Record r1;
        r1["reward"] = make_scalar_f32(1.5f);
        shard0_writer.put("rollout@p5", {"reward"}, {{1, r1}});
        controller_client.notify_data_update("rollout@p5", {1}, {"reward"}, {tq::FieldDtype::Float32},
                                              /*shard_index=*/0);

        // Phase 6: a weight sync happens between the two writes, so the two
        // samples land with different policy versions despite being in the
        // same partition -- exactly the gap staleness tracking exists for.
        assert(controller_client.advance_version() == 1);

        tq::Record r2;
        r2["reward"] = make_scalar_f32(2.5f);
        shard1_writer.put("rollout@p5", {"reward"}, {{2, r2}});
        controller_client.notify_data_update("rollout@p5", {2}, {"reward"}, {tq::FieldDtype::Float32},
                                              /*shard_index=*/1);

        // Reader: GET_META against the controller resolves both samples
        // across both shards in one round trip (bundled shard_registry),
        // then GET_DATA against each shard fetches the actual tensors.
        auto meta = controller_client.get_meta("rollout@p5", {"reward"}, "trainer", /*batch_size=*/2);
        assert(meta.sample_ids.size() == 2);
        assert(meta.shard_addresses.size() == 2);
        assert(meta.shard_addresses.at(0) == shard0_address);
        assert(meta.shard_addresses.at(1) == shard1_address);

        // Phase 6: sample 1 was produced before the weight sync (version 0),
        // sample 2 after (version 1); current_version reflects the one
        // advance_version() call made above.
        std::unordered_map<tq::SampleId, std::int64_t> version_by_id;
        for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
            version_by_id[meta.sample_ids[i]] = meta.sample_versions[i];
        }
        assert(version_by_id.at(1) == 0);
        assert(version_by_id.at(2) == 1);
        assert(meta.current_version == 1);

        std::unordered_map<tq::SampleId, float> fetched;
        for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
            tq::SampleId id = meta.sample_ids[i];
            std::int32_t shard = meta.sample_shard_indices[i];
            tq::TransferQueueRpcClient reader("reader-1", meta.shard_addresses.at(shard));

            auto data = reader.get_data("rollout@p5", {id}, {"reward"});
            assert(data.size() == 1 && data.count(id) == 1);
            fetched[id] = static_cast<float*>(data.at(id).at("reward").data())[0];
        }
        assert(fetched.at(1) == 1.5f);
        assert(fetched.at(2) == 2.5f);

        assert(p5_controller->shard_for_sample("rollout@p5", 1) == 0);
        assert(p5_controller->shard_for_sample("rollout@p5", 2) == 1);

        // FIND_STRANDED_GROUPS against the real split ControllerServer,
        // with a plain FifoSampler (no grouping concept at all) --
        // confirms the generic virtual-dispatch wiring works even for a
        // sampler that always reports nothing (BaseSampler's default),
        // using "other_task" specifically so there ARE real ready samples
        // to scan (1 and 2 were already consumed for "trainer" above, but
        // never for "other_task") -- this isn't just a trivial
        // empty-ready-set pass.
        auto p5_ready_other_task = p5_controller->ready_indexes("rollout@p5", {"reward"}, "other_task");
        assert(p5_ready_other_task.size() == 2);
        auto p5_stranded =
            controller_client.find_stranded_groups("rollout@p5", {"reward"}, "other_task", /*max_age_ms=*/0);
        assert(p5_stranded.empty());

        // CLEAR_PARTITION on the controller fans out CLEAR_DATA to both
        // shards -- verify both StorageManagers actually freed their bytes.
        auto baseline0 = shard0_storage->current_bytes();
        auto baseline1 = shard1_storage->current_bytes();
        assert(baseline0 > 0);
        assert(baseline1 > 0);
        controller_client.clear_partition("rollout@p5");
        assert(shard0_storage->current_bytes() < baseline0);
        assert(shard1_storage->current_bytes() < baseline1);

        shard0.stop();
        shard1.stop();
        controller_server.stop();
    }

    std::cout << "test_rpc_smoke: PASS\n";
    return 0;
}
