// Control-plane microbenchmarks -- pure in-process timing, no tensors, no
// sockets. Covers two of the audit's benchmark categories (see
// docs/TransferQueue-Benchmark.md Section 13):
//   #6 Control-plane microbenchmarks: PartitionIndexManager,
//      DataPartitionStatus, Controller::advance_version, all at zero
//      payload cost (metadata only).
//   #3 Sampling policy performance: GRPOGroupNSampler's own grouping/
//      bucketing cost in isolation, plus Controller::select_and_consume/
//      find_stranded_groups at scale, with a mix of complete and
//      incomplete (stranded) groups.
//
// Not a correctness test -- see tests/unit/ for those. CSV output:
// name,n,seconds,ops_per_sec
//
// Build: make control_plane_benchmark (core tier only -- no
// Tensor-Implementations or ZMQ needed, since none of these components
// touch tensors or sockets).

#include <chrono>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "transferqueue/Controller.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/PartitionIndexManager.h"
#include "transferqueue/Sampler.h"

using Clock = std::chrono::steady_clock;

namespace {

double elapsed_seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

void report(const std::string& name, std::size_t n, double seconds) {
    double ops_per_sec = seconds > 0.0 ? static_cast<double>(n) / seconds : 0.0;
    std::cout << name << "," << n << "," << seconds << "," << ops_per_sec << "\n";
}

// ============================================================
// #6: Control-plane microbenchmarks (zero payload)
// ============================================================

void bench_partition_index_manager(std::size_t n) {
    {
        tq::PartitionIndexManager mgr;
        auto t0 = Clock::now();
        auto ids = mgr.allocate_indexes("p", n);
        report("PartitionIndexManager.allocate_indexes_bulk", n, elapsed_seconds(t0));

        auto t1 = Clock::now();
        mgr.release_indexes("p", ids);
        report("PartitionIndexManager.release_indexes_bulk", n, elapsed_seconds(t1));
    }
    {
        // One call per id (count=1 each) -- exercises the mutex-per-call
        // cost a one-sample-at-a-time caller pays, not just the bulk call's
        // internal loop throughput.
        tq::PartitionIndexManager mgr;
        std::vector<tq::SampleId> ids;
        ids.reserve(n);
        auto t0 = Clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            ids.push_back(mgr.allocate_indexes("p", 1)[0]);
        }
        report("PartitionIndexManager.allocate_indexes_per_call", n, elapsed_seconds(t0));

        auto t1 = Clock::now();
        for (tq::SampleId id : ids) {
            mgr.release_indexes("p", {id});
        }
        report("PartitionIndexManager.release_indexes_per_call", n, elapsed_seconds(t1));
    }
}

void bench_data_partition_status(std::size_t n) {
    std::vector<tq::SampleId> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<tq::SampleId>(i);

    tq::DataPartitionStatus status("p");
    auto t0 = Clock::now();
    status.update_production_status(ids, {"reward"});
    report("DataPartitionStatus.update_production_status_bulk", n, elapsed_seconds(t0));

    auto t1 = Clock::now();
    auto ready = status.scan_data_status({"reward"}, "trainer");
    report("DataPartitionStatus.scan_data_status", n, elapsed_seconds(t1));
    if (ready.size() != n) {
        std::cerr << "warning: scan_data_status returned " << ready.size() << ", expected " << n << "\n";
    }

    // Lock contention: kThreads concurrently calling update_production_status
    // for disjoint id ranges on the SAME partition -- simulates concurrent
    // writers hammering one partition's single mutex (see
    // DataPartitionStatus.h's own doc comment on why it needs one at all:
    // TransferQueueServer dispatches across a thread pool).
    constexpr int kThreads = 8;
    tq::DataPartitionStatus contended("p2");
    std::size_t per_thread = n / kThreads;
    std::vector<std::thread> threads;
    auto t2 = Clock::now();
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<tq::SampleId> chunk(per_thread);
            for (std::size_t i = 0; i < per_thread; ++i) {
                chunk[i] = static_cast<tq::SampleId>(t * per_thread + i);
            }
            contended.update_production_status(chunk, {"reward"});
        });
    }
    for (auto& th : threads) th.join();
    report("DataPartitionStatus.update_production_status_contended_8t", n, elapsed_seconds(t2));
}

void bench_controller_advance_version(std::size_t n) {
    tq::TransferQueueController controller;
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < n; ++i) {
        controller.advance_version();
    }
    report("Controller.advance_version", n, elapsed_seconds(t0));

    controller.create_partition("p");
    std::vector<tq::SampleId> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<tq::SampleId>(i);
    controller.update_production_status("p", ids, {"reward"});

    auto t1 = Clock::now();
    for (tq::SampleId id : ids) {
        (void)controller.version_for_sample("p", id);
    }
    report("Controller.version_for_sample_lookup", n, elapsed_seconds(t1));
}

// ============================================================
// #3: Sampling policy performance (GRPOGroupNSampler vs. FifoSampler shape)
// ============================================================

// Pure sampler cost, no Controller/DataPartitionStatus involved -- isolates
// GRPOGroupNSampler::sample()'s own sort-then-bucket cost from everything
// around it. Every sample belongs to a complete group (uniform group_size),
// so the whole batch is selectable in one call.
void bench_grpo_sample_pure(std::size_t n, std::size_t group_size) {
    tq::GRPOGroupNSampler sampler(group_size);
    std::size_t num_groups = n / group_size;
    n = num_groups * group_size; // round down to a whole number of groups

    std::vector<tq::SampleId> ready_ids(n);
    std::vector<std::string> group_ids(n);
    for (std::size_t i = 0; i < n; ++i) {
        ready_ids[i] = static_cast<tq::SampleId>(i);
        group_ids[i] = "group-" + std::to_string(i % num_groups);
    }

    auto t0 = Clock::now();
    auto [selected, remaining] = sampler.sample(ready_ids, group_ids, n);
    double secs = elapsed_seconds(t0);
    report("GRPOGroupNSampler.sample_pure_all_complete", n, secs);
    if (selected.size() != n) {
        std::cerr << "warning: expected " << n << " selected, got " << selected.size() << "\n";
    }
    (void)remaining;
}

// Pure find_stranded() cost -- every group here is deliberately incomplete
// (one short of group_size), so every bucket shows up in the result.
void bench_grpo_find_stranded_pure(std::size_t n, std::size_t group_size) {
    tq::GRPOGroupNSampler sampler(group_size);
    std::size_t incomplete_size = group_size > 1 ? group_size - 1 : 1;
    std::size_t num_groups = n / incomplete_size;
    n = num_groups * incomplete_size;

    std::vector<tq::SampleId> ready_ids(n);
    std::vector<std::string> group_ids(n);
    std::vector<std::int64_t> produced_at_ms(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        ready_ids[i] = static_cast<tq::SampleId>(i);
        group_ids[i] = "group-" + std::to_string(i / incomplete_size);
    }

    auto t0 = Clock::now();
    auto stranded = sampler.find_stranded(ready_ids, group_ids, produced_at_ms, /*now_ms=*/1000, /*max_age_ms=*/0);
    double secs = elapsed_seconds(t0);
    report("GRPOGroupNSampler.find_stranded_pure_all_incomplete", n, secs);
    if (stranded.size() != num_groups) {
        std::cerr << "warning: expected " << num_groups << " stranded groups, got " << stranded.size() << "\n";
    }
}

// Populates `controller`'s partition with `complete_total` samples in full
// (group_size-member) groups followed by `incomplete_total` samples in
// groups missing exactly one member each -- the "waiting on the last
// straggler rollout worker" shape named in the ask.
void setup_partial_groups(tq::TransferQueueController& controller, const std::string& partition,
                           std::size_t complete_total, std::size_t incomplete_total, std::size_t group_size) {
    controller.create_partition(partition);
    tq::SampleId next_id = 0;
    std::size_t group_idx = 0;

    for (std::size_t written = 0; written < complete_total; written += group_size) {
        std::string group = "complete-" + std::to_string(group_idx++);
        std::vector<tq::SampleId> ids;
        for (std::size_t k = 0; k < group_size; ++k) ids.push_back(next_id++);
        controller.update_production_status(partition, ids, {"reward"}, std::nullopt, group);
    }

    std::size_t incomplete_size = group_size > 1 ? group_size - 1 : 1;
    for (std::size_t written = 0; written < incomplete_total; written += incomplete_size) {
        std::string group = "incomplete-" + std::to_string(group_idx++);
        std::vector<tq::SampleId> ids;
        for (std::size_t k = 0; k < incomplete_size; ++k) ids.push_back(next_id++);
        controller.update_production_status(partition, ids, {"reward"}, std::nullopt, group);
    }
}

void bench_select_and_consume(std::size_t total_samples, std::size_t group_size, double incomplete_fraction) {
    std::size_t incomplete_total = static_cast<std::size_t>(static_cast<double>(total_samples) * incomplete_fraction);
    std::size_t complete_total = total_samples - incomplete_total;
    std::size_t complete_groups = complete_total / group_size;
    complete_total = complete_groups * group_size; // exact multiple, matches batch_size's own requirement

    tq::TransferQueueController controller;
    setup_partial_groups(controller, "p", complete_total, incomplete_total, group_size);

    tq::GRPOGroupNSampler sampler(group_size);
    std::size_t batch_size = complete_groups * group_size;

    auto t0 = Clock::now();
    auto selected = controller.select_and_consume("p", {"reward"}, "trainer", sampler, batch_size);
    double secs = elapsed_seconds(t0);
    int pct = static_cast<int>(incomplete_fraction * 100.0);
    report("Controller.select_and_consume_" + std::to_string(pct) + "pct_incomplete",
           complete_total + incomplete_total, secs);
    if (selected.size() != batch_size) {
        std::cerr << "warning: expected " << batch_size << " selected, got " << selected.size() << "\n";
    }
}

void bench_find_stranded_groups(std::size_t total_samples, std::size_t group_size, double incomplete_fraction) {
    std::size_t incomplete_total = static_cast<std::size_t>(static_cast<double>(total_samples) * incomplete_fraction);
    std::size_t complete_total = total_samples - incomplete_total;

    tq::TransferQueueController controller;
    setup_partial_groups(controller, "p", complete_total, incomplete_total, group_size);

    tq::GRPOGroupNSampler sampler(group_size);
    auto t0 = Clock::now();
    auto stranded = controller.find_stranded_groups("p", {"reward"}, "trainer", sampler, /*max_age_ms=*/0);
    double secs = elapsed_seconds(t0);
    int pct = static_cast<int>(incomplete_fraction * 100.0);
    report("Controller.find_stranded_groups_" + std::to_string(pct) + "pct_incomplete",
           complete_total + incomplete_total, secs);
    if (stranded.empty() && incomplete_total > 0) {
        std::cerr << "warning: expected stranded groups, got none\n";
    }
}

} // namespace

int main() {
    std::cout << "name,n,seconds,ops_per_sec\n";

    // #6: control-plane microbenchmarks.
    bench_partition_index_manager(1'000'000);
    for (std::size_t n : {50'000UL, 100'000UL}) {
        bench_data_partition_status(n);
    }
    bench_controller_advance_version(100'000);

    // #3: sampling policy performance, at the three scales named in the ask.
    constexpr std::size_t kGroupSize = 8; // GRPO's G (completions per prompt)
    for (std::size_t n : {10'000UL, 50'000UL, 100'000UL}) {
        bench_grpo_sample_pure(n, kGroupSize);
        bench_grpo_find_stranded_pure(n, kGroupSize);
        bench_select_and_consume(n, kGroupSize, /*incomplete_fraction=*/0.0);
        bench_select_and_consume(n, kGroupSize, /*incomplete_fraction=*/0.3);
        bench_find_stranded_groups(n, kGroupSize, /*incomplete_fraction=*/0.3);
    }

    return 0;
}
