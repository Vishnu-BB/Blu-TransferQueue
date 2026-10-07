#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/GroupRouter.h"
#include "transferqueue/Message.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"
#include "unit/TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using tq::FieldDtype;
using tq::Record;
using tq::SampleId;

namespace {

Tensor make_i64_tensor(const std::vector<std::int64_t>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

Tensor make_f32_tensor(const std::vector<float>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Float32);
    auto* data = static_cast<float*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

bool check_i64_tensor(const Tensor& t, const std::vector<std::int64_t>& expected) {
    if (t.dtype() != Dtype::Int64 || t.numel() != static_cast<std::int64_t>(expected.size())) return false;
    const auto* data = static_cast<const std::int64_t*>(t.data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) return false;
    }
    return true;
}

bool check_f32_tensor(const Tensor& t, const std::vector<float>& expected) {
    if (t.dtype() != Dtype::Float32 || t.numel() != static_cast<std::int64_t>(expected.size())) return false;
    const auto* data = static_cast<const float*>(t.data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(data[i] - expected[i]) > 1e-5f) return false;
    }
    return true;
}

// ============================================================================
// PHASE 1: Multi-Stage Streaming Pipeline & Wire Tensor Integrity
// ============================================================================

void test_schema_enforcement_and_validation() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();

    tq::TransferQueueServer server("server-schema", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-schema", address);

    const std::string partition = "rollout@schema_test";

    // 1. Declare valid schema
    client.declare_schema(partition, {{"prompt_ids", FieldDtype::Int64},
                                      {"reward", FieldDtype::Float32}});

    // 2. Conflicting re-declaration must throw
    CHECK_THROWS(client.declare_schema(partition, {{"reward", FieldDtype::Int64}}), std::runtime_error);

    // 3. Writing matching dtypes succeeds
    Record valid_r;
    valid_r["prompt_ids"] = make_i64_tensor({1, 2, 3});
    valid_r["reward"] = make_f32_tensor({1.0f});
    CHECK_NOTHROW(client.put(partition, {"prompt_ids", "reward"}, {{10, valid_r}}));

    // 4. Writing mismatched dtype throws before writing data
    Record invalid_r;
    invalid_r["prompt_ids"] = make_f32_tensor({1.0f, 2.0f}); // expected Int64
    invalid_r["reward"] = make_f32_tensor({1.0f});
    CHECK_THROWS(client.put(partition, {"prompt_ids", "reward"}, {{11, invalid_r}}), std::runtime_error);

    server.stop();
}

void test_multistage_streaming_overlap() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();

    tq::TransferQueueServer server("server-streaming", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-streaming", address);

    const std::string partition = "rollout@streaming";
    const std::string task = "trainer";

    client.declare_schema(partition, {{"prompt_ids", FieldDtype::Int64},
                                      {"response_ids", FieldDtype::Int64},
                                      {"log_probs", FieldDtype::Float32},
                                      {"reward", FieldDtype::Float32}});

    // STAGE A: Rollout Engine writes prompt_ids and response_ids for sample 100
    Record r_stageA;
    r_stageA["prompt_ids"] = make_i64_tensor({10, 11, 12});
    r_stageA["response_ids"] = make_i64_tensor({100, 101});
    client.put(partition, {"prompt_ids", "response_ids"}, {{100, r_stageA}});

    // Trainer queries for ALL 4 training fields -> MUST BE EMPTY because log_probs & reward missing
    auto ready_full = client.get(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task, 1);
    CHECK(ready_full.empty());

    // Inference worker queries for {"prompt_ids", "response_ids"} -> MUST BE READY!
    auto ready_for_inf = client.get(partition, {"prompt_ids", "response_ids"}, "inference_worker", 1);
    CHECK(ready_for_inf.size() == 1);
    CHECK(ready_for_inf.count(100) == 1);

    // STAGE B: Inference Worker writes log_probs
    Record r_stageB;
    r_stageB["log_probs"] = make_f32_tensor({-0.12f, -0.45f});
    client.put(partition, {"log_probs"}, {{100, r_stageB}});

    // Trainer queries again -> STILL EMPTY because reward is missing
    ready_full = client.get(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task, 1);
    CHECK(ready_full.empty());

    // STAGE C: Reward Model writes reward
    Record r_stageC;
    r_stageC["reward"] = make_f32_tensor({0.88f});
    client.put(partition, {"reward"}, {{100, r_stageC}});

    // STAGE D: Trainer queries now -> ALL 4 FIELDS ARE READY!
    ready_full = client.get(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task, 1);
    CHECK(ready_full.size() == 1);
    CHECK(ready_full.count(100) == 1);

    // Verify bit-exact tensor data over the wire
    const auto& consumed_rec = ready_full.at(100);
    CHECK(check_i64_tensor(consumed_rec.at("prompt_ids"), {10, 11, 12}));
    CHECK(check_i64_tensor(consumed_rec.at("response_ids"), {100, 101}));
    CHECK(check_f32_tensor(consumed_rec.at("log_probs"), {-0.12f, -0.45f}));
    CHECK(check_f32_tensor(consumed_rec.at("reward"), {0.88f}));

    server.stop();
}

// ============================================================================
// PHASE 2: GRPO Group-N Sampling & Policy Lag Versioning
// ============================================================================

void test_grpo_group_sampling_and_all_or_nothing() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    // Group size = 4
    auto sampler = std::make_shared<tq::GRPOGroupNSampler>(4);

    tq::TransferQueueServer server("server-grpo", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-grpo", address);

    const std::string partition = "rollout@grpo";
    const std::string task = "trainer";

    client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Group A: 4 completions (complete)
    for (SampleId id : {0, 1, 2, 3}) {
        Record r;
        r["reward"] = make_f32_tensor({static_cast<float>(id) * 0.1f});
        client.put(partition, {"reward"}, {{id, r}}, "prompt_A");
    }

    // Group B: only 3 completions (incomplete, sample 7 in-flight)
    for (SampleId id : {4, 5, 6}) {
        Record r;
        r["reward"] = make_f32_tensor({static_cast<float>(id) * 0.1f});
        client.put(partition, {"reward"}, {{id, r}}, "prompt_B");
    }

    // Request batch_size = 4 -> MUST select complete Group A {0, 1, 2, 3}
    auto meta1 = client.get_meta(partition, {"reward"}, task, 4);
    CHECK(meta1.sample_ids.size() == 4);
    std::vector<SampleId> expected_A = {0, 1, 2, 3};
    CHECK(meta1.sample_ids == expected_A);

    // Request batch_size = 4 again -> Group B is incomplete (only 3 ready), MUST RETURN EMPTY
    auto meta2 = client.get_meta(partition, {"reward"}, task, 4);
    CHECK(meta2.sample_ids.empty());

    // Test stranded group detection: prompt_B should be flagged if max_age_ms is 0
    auto stranded = client.find_stranded_groups(partition, {"reward"}, task, /*max_age_ms=*/0);
    CHECK(stranded.size() == 1);
    CHECK(stranded[0].group_id == "prompt_B");
    CHECK(stranded[0].sample_ids.size() == 3);

    // Write missing completion 7 for Group B
    Record r7;
    r7["reward"] = make_f32_tensor({0.7f});
    client.put(partition, {"reward"}, {{7, r7}}, "prompt_B");

    // Request batch_size = 4 now -> Group B is complete and selected!
    auto meta3 = client.get_meta(partition, {"reward"}, task, 4);
    CHECK(meta3.sample_ids.size() == 4);
    std::vector<SampleId> expected_B = {4, 5, 6, 7};
    CHECK(meta3.sample_ids == expected_B);

    server.stop();
}

void test_cross_group_interleaving_isolation() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::GRPOGroupNSampler>(2);

    tq::TransferQueueServer server("server-iso", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-iso", address);

    const std::string partition = "rollout@iso";
    client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Interleave arrival of prompt_X and prompt_Y:
    // X gets ID 100, Y gets ID 101, X gets ID 102, Y gets ID 103
    Record r;
    r["reward"] = make_f32_tensor({1.0f});
    client.put(partition, {"reward"}, {{100, r}}, "prompt_X");
    client.put(partition, {"reward"}, {{101, r}}, "prompt_Y");
    client.put(partition, {"reward"}, {{102, r}}, "prompt_X");
    client.put(partition, {"reward"}, {{103, r}}, "prompt_Y");

    // Request batch_size = 2 -> GRPOGroupNSampler sorts buckets by group_id ("prompt_X" < "prompt_Y")
    // Group X should be selected with IDs {100, 102} (NOT {100, 101}!)
    auto meta1 = client.get_meta(partition, {"reward"}, "trainer", 2);
    CHECK(meta1.sample_ids.size() == 2);
    std::vector<SampleId> expected_X = {100, 102};
    CHECK(meta1.sample_ids == expected_X);

    // Next request -> Group Y with IDs {101, 103}
    auto meta2 = client.get_meta(partition, {"reward"}, "trainer", 2);
    CHECK(meta2.sample_ids.size() == 2);
    std::vector<SampleId> expected_Y = {101, 103};
    CHECK(meta2.sample_ids == expected_Y);

    server.stop();
}

void test_policy_lag_and_heterogeneous_versions() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::GRPOGroupNSampler>(4);

    tq::TransferQueueServer server("server-staleness", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-staleness", address);

    const std::string partition = "rollout@staleness";
    client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Check version starts at 0
    auto meta0 = client.get_meta(partition, {"reward"}, "trainer", 4);
    CHECK(meta0.current_version == 0);

    // Group 1 produced at version 0
    Record r;
    r["reward"] = make_f32_tensor({1.0f});
    for (SampleId id : {0, 1, 2, 3}) {
        client.put(partition, {"reward"}, {{id, r}}, "grp_1");
    }

    // Weight sync occurs: bump version
    auto bumped1 = client.advance_version();
    CHECK(bumped1 == 1);

    // Group 2: first 2 samples produced at version 1
    for (SampleId id : {4, 5}) {
        client.put(partition, {"reward"}, {{id, r}}, "grp_2");
    }

    // Another weight sync occurs: bump version to 2
    auto bumped2 = client.advance_version();
    CHECK(bumped2 == 2);

    // Group 2: remaining 2 samples produced at version 2
    for (SampleId id : {6, 7}) {
        client.put(partition, {"reward"}, {{id, r}}, "grp_2");
    }

    // Consume Group 1:
    // Produced at version 0, consumed when current_version is 2 -> Lag tau = 2 - 0 = 2
    auto meta_g1 = client.get_meta(partition, {"reward"}, "trainer", 4);
    CHECK(meta_g1.sample_ids.size() == 4);
    CHECK(meta_g1.current_version == 2);
    std::vector<std::int64_t> expected_g1_versions = {0, 0, 0, 0};
    CHECK(meta_g1.sample_versions == expected_g1_versions);

    // Consume Group 2:
    // Samples 4, 5 stamped version 1; samples 6, 7 stamped version 2.
    // Heterogeneous versions inside the same group!
    auto meta_g2 = client.get_meta(partition, {"reward"}, "trainer", 4);
    CHECK(meta_g2.sample_ids.size() == 4);
    CHECK(meta_g2.current_version == 2);
    std::vector<std::int64_t> expected_g2_versions = {1, 1, 2, 2};
    CHECK(meta_g2.sample_versions == expected_g2_versions);

    server.stop();
}

// ============================================================================
// PHASE 3: Disaggregated Multi-Node Sharding & Routing
// ============================================================================

void test_disaggregated_sharding_and_routing() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto sampler = std::make_shared<tq::GRPOGroupNSampler>(2);
    tq::ControllerServer controller_server("ctrl-server", controller, sampler);
    std::string ctrl_addr = controller_server.start("tcp://127.0.0.1:0");

    // Shard 0 (StorageServer)
    auto storage0 = std::make_shared<tq::SimpleStorageManager>();
    tq::StorageServer shard0("shard-0", 0, storage0);
    std::string shard0_addr = shard0.start("tcp://127.0.0.1:0");

    // Shard 1 (StorageServer)
    auto storage1 = std::make_shared<tq::SimpleStorageManager>();
    tq::StorageServer shard1("shard-1", 1, storage1);
    std::string shard1_addr = shard1.start("tcp://127.0.0.1:0");

    // Announce shards to ControllerServer
    tq::TransferQueueRpcClient announcer0("announcer-0", ctrl_addr);
    announcer0.announce_shard(0, shard0_addr);

    tq::TransferQueueRpcClient announcer1("announcer-1", ctrl_addr);
    announcer1.announce_shard(1, shard1_addr);

    const std::string partition = "rollout@sharded";
    tq::TransferQueueRpcClient ctrl_client("ctrl-client", ctrl_addr);
    ctrl_client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Use GroupRouter to route group "prompt_alpha" and group "prompt_beta"
    tq::GroupRouter router(2);
    std::string grp_a = "prompt_alpha";
    std::string grp_b = "prompt_beta";
    int rank_a = router.target_rank(grp_a);
    int rank_b = router.target_rank(grp_b);

    // If both hash to the same rank, tweak grp_b so they test multi-shard routing
    int counter = 0;
    while (rank_b == rank_a) {
        grp_b = "prompt_beta_" + std::to_string(++counter);
        rank_b = router.target_rank(grp_b);
    }
    CHECK(rank_a != rank_b);

    // Prepare writers
    tq::TransferQueueRpcClient writer_shard0("writer-s0", shard0_addr);
    tq::TransferQueueRpcClient writer_shard1("writer-s1", shard1_addr);

    // Write grp_a samples {10, 11} to rank_a shard
    Record ra;
    ra["reward"] = make_f32_tensor({10.5f});
    if (rank_a == 0) {
        writer_shard0.put(partition, {"reward"}, {{10, ra}, {11, ra}});
    } else {
        writer_shard1.put(partition, {"reward"}, {{10, ra}, {11, ra}});
    }
    ctrl_client.notify_data_update(partition, {10, 11}, {"reward"}, {FieldDtype::Float32}, rank_a, grp_a);

    // Write grp_b samples {20, 21} to rank_b shard
    Record rb;
    rb["reward"] = make_f32_tensor({20.5f});
    if (rank_b == 0) {
        writer_shard0.put(partition, {"reward"}, {{20, rb}, {21, rb}});
    } else {
        writer_shard1.put(partition, {"reward"}, {{20, rb}, {21, rb}});
    }
    ctrl_client.notify_data_update(partition, {20, 21}, {"reward"}, {FieldDtype::Float32}, rank_b, grp_b);

    // Reader: call get_meta on ControllerServer for batch_size = 4
    auto meta = ctrl_client.get_meta(partition, {"reward"}, "trainer", 4);
    CHECK(meta.sample_ids.size() == 4);
    CHECK(meta.shard_addresses.size() == 2);
    CHECK(meta.shard_addresses.at(0) == shard0_addr);
    CHECK(meta.shard_addresses.at(1) == shard1_addr);

    // Verify two-hop direct GET_DATA to each respective shard fetches real tensors
    for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
        SampleId id = meta.sample_ids[i];
        std::int32_t shard = meta.sample_shard_indices[i];
        tq::TransferQueueRpcClient shard_reader("reader", meta.shard_addresses.at(shard));
        auto fetched = shard_reader.get_data(partition, {id}, {"reward"});
        CHECK(fetched.count(id) == 1);
        float expected_val = (id == 10 || id == 11) ? 10.5f : 20.5f;
        CHECK(check_f32_tensor(fetched.at(id).at("reward"), {expected_val}));
    }

    // Test clear_partition fan-out: clears controller metadata AND fans out to both shards
    CHECK(storage0->current_bytes() > 0);
    CHECK(storage1->current_bytes() > 0);
    ctrl_client.clear_partition(partition);
    CHECK(storage0->current_bytes() == 0);
    CHECK(storage1->current_bytes() == 0);

    shard0.stop();
    shard1.stop();
    controller_server.stop();
}

// ============================================================================
// PHASE 4: Concurrency & Storage Capacity Backpressure
// ============================================================================

void test_concurrent_readers_no_duplicate_consumption() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();

    tq::TransferQueueServer server("server-conc", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-init", address);

    const std::string partition = "rollout@conc";
    client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Populate 40 distinct samples
    std::unordered_map<SampleId, Record> batch;
    for (SampleId id = 0; id < 40; ++id) {
        Record r;
        r["reward"] = make_f32_tensor({static_cast<float>(id)});
        batch[id] = r;
    }
    client.put(partition, {"reward"}, batch);

    // 4 concurrent readers requesting batch_size = 5 on the SAME task_name
    constexpr int kNumThreads = 4;
    std::vector<std::thread> threads;
    std::mutex collected_mutex;
    std::vector<SampleId> all_consumed;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&, t]() {
            tq::TransferQueueRpcClient reader("reader-" + std::to_string(t), address);
            for (int step = 0; step < 2; ++step) {
                auto data = reader.get(partition, {"reward"}, "shared_trainer", 5);
                std::lock_guard<std::mutex> lock(collected_mutex);
                for (const auto& [id, _] : data) {
                    all_consumed.push_back(id);
                }
            }
        });
    }

    for (auto& th : threads) th.join();

    // 4 threads * 2 steps * 5 batch_size = 40 samples consumed total
    CHECK(all_consumed.size() == 40);

    // Verify ZERO duplicates across threads!
    std::sort(all_consumed.begin(), all_consumed.end());
    for (std::size_t i = 1; i < all_consumed.size(); ++i) {
        CHECK(all_consumed[i] != all_consumed[i - 1]);
    }

    server.stop();
}

void test_storage_capacity_backpressure() {
    auto controller = std::make_shared<tq::TransferQueueController>();
    // Set finite capacity: space for 80 bytes
    constexpr std::size_t kCapacity = 80;
    auto storage = std::make_shared<tq::SimpleStorageManager>(kCapacity);
    auto sampler = std::make_shared<tq::FifoSampler>();

    tq::TransferQueueServer server("server-bp", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("client-bp", address);

    const std::string partition = "rollout@backpressure";
    client.declare_schema(partition, {{"reward", FieldDtype::Float32}});

    // Write sample 1 (10 floats = 40 bytes)
    Record r1;
    r1["reward"] = make_f32_tensor(std::vector<float>(10, 1.0f));
    client.put(partition, {"reward"}, {{1, r1}});

    // Write sample 2 (8 floats = 32 bytes) -> Total 72 bytes <= 80 bytes
    Record r2;
    r2["reward"] = make_f32_tensor(std::vector<float>(8, 2.0f));
    client.put(partition, {"reward"}, {{2, r2}});

    std::atomic<bool> blocked_write_finished{false};
    std::thread background_writer([&]() {
        tq::TransferQueueRpcClient writer_th("writer-bg", address);
        Record r3;
        // 10 floats = 40 bytes. 72 + 40 = 112 bytes > 80 capacity -> MUST BLOCK
        r3["reward"] = make_f32_tensor(std::vector<float>(10, 3.0f));
        writer_th.put(partition, {"reward"}, {{3, r3}});
        blocked_write_finished = true;
    });

    // Wait 100ms and verify background writer is indeed blocked
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK(!blocked_write_finished.load());

    // Free space by clearing sample 1 (frees 40 bytes)
    client.clear_data(partition, {1});

    // Background writer must unblock!
    background_writer.join();
    CHECK(blocked_write_finished.load());

    server.stop();
}

} // namespace

int main() {
    std::cout << "Running Blu-TransferQueue End-to-End Pipeline Correctness Tests...\n";

    // Phase 1
    test_schema_enforcement_and_validation();
    test_multistage_streaming_overlap();

    // Phase 2
    test_grpo_group_sampling_and_all_or_nothing();
    test_cross_group_interleaving_isolation();
    test_policy_lag_and_heterogeneous_versions();

    // Phase 3
    test_disaggregated_sharding_and_routing();

    // Phase 4
    test_concurrent_readers_no_duplicate_consumption();
    test_storage_capacity_backpressure();

    return ::tq::test::summary("TransferQueue Pipeline Correctness Test Suite");
}
