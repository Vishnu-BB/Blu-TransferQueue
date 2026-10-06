// Unit tests for TransferQueueController (include/transferqueue/Controller.h,
// src/Controller.cpp). Zero dependencies -- build with `make unit_tests` or:
// g++ -std=c++2a -Iinclude src/BatchMeta.cpp src/Message.cpp \
//   src/PartitionIndexManager.cpp src/DataPartitionStatus.cpp src/Controller.cpp \
//   src/Sampler.cpp src/GRPOGroupNSampler.cpp src/GroupRouter.cpp \
//   tests/unit/ControllerTest.cpp -o /tmp/t && /tmp/t
//
// Scope: TransferQueueController's own contract -- partition lifecycle,
// routing, the global shard registry, the global version counter, and
// clear_partition's returned struct. DataPartitionStatus/PartitionIndexManager
// are real collaborators here (never mocked, matching this project's
// convention) but their own internal edge cases are covered by their
// dedicated test files, not re-proven here.

#include "transferqueue/Controller.h"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "TestUtils.h"

namespace {

bool contains(const std::vector<tq::SampleId>& v, tq::SampleId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

} // namespace

int main() {
    // ---- create_partition: lifecycle, idempotency-reporting ----
    {
        tq::TransferQueueController controller;
        CHECK(controller.create_partition("p1"));  // first call: created
        CHECK(!controller.create_partition("p1")); // second call: already exists
        CHECK(controller.create_partition("p2"));  // independent partition

        // Exists but empty -- ready_indexes() on a brand new partition is an
        // empty result, not an error/throw.
        CHECK(controller.ready_indexes("p1", {"reward"}, "trainer").empty());
    }

    // ---- declare_schema: implicit creation, no deadlock, conflict detection ----
    {
        tq::TransferQueueController controller;

        // Declaring on a partition that doesn't exist yet implicitly
        // creates it -- proven by a subsequent update_production_status()
        // succeeding (returns true) with no explicit create_partition()
        // call in between.
        CHECK_NOTHROW(controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}}));
        CHECK(controller.update_production_status("p1", {1}, {"reward"}));

        // Re-declaring the same field with the SAME dtype is idempotent, no
        // throw.
        CHECK_NOTHROW(controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}}));

        // Conflicting dtype for an already-declared field throws.
        CHECK_THROWS(controller.declare_schema("p1", {{"reward", tq::FieldDtype::Int64}}), std::invalid_argument);

        // Calling declare_schema() on an *already-existing* partition must
        // not deadlock (ensure_partition_locked is lock-free internally,
        // called while already holding mutex_ -- a bug here would hang this
        // whole process forever, not just fail an assertion). Interleave
        // repeatedly with create_partition() on the same id to exercise
        // both "create if missing" paths back-to-back.
        for (int i = 0; i < 50; ++i) {
            CHECK_NOTHROW(controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}}));
            CHECK(!controller.create_partition("p1")); // already exists every time
        }
    }

    // ---- validate_schema: no-op on unknown partition, gradual adoption ----
    {
        tq::TransferQueueController controller;

        // Unknown partition: silent no-op, never throws -- matches every
        // other method's "unknown partition" convention here.
        CHECK_NOTHROW(controller.validate_schema("missing", {{"reward", tq::FieldDtype::Int64}}));

        controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}});

        // A field never declared anywhere is never validated (gradual
        // adoption) -- any dtype is accepted silently.
        CHECK_NOTHROW(controller.validate_schema("p1", {{"undeclared_field", tq::FieldDtype::Int64}}));

        // Matching declared dtype: silent.
        CHECK_NOTHROW(controller.validate_schema("p1", {{"reward", tq::FieldDtype::Float32}}));

        // Conflicting declared dtype: throws.
        CHECK_THROWS(controller.validate_schema("p1", {{"reward", tq::FieldDtype::Int64}}), std::invalid_argument);
    }

    // ---- update_production_status: return value, shard_index, version stamping ----
    {
        tq::TransferQueueController controller;

        // Unknown partition -> false, no side effects.
        CHECK(!controller.update_production_status("missing", {1}, {"reward"}));

        controller.create_partition("p1");

        // Known partition -> true.
        CHECK(controller.update_production_status("p1", {1}, {"reward"}));

        // shard_index omitted -> shard_for_sample is -1 (not 0, not garbage).
        CHECK(controller.shard_for_sample("p1", 1) == -1);

        // shard_index supplied -> retrievable exactly.
        CHECK(controller.update_production_status("p1", {2}, {"reward"}, /*shard_index=*/7));
        CHECK(controller.shard_for_sample("p1", 2) == 7);

        // Version stamping: every update_production_status() call stamps
        // whatever current_version() is AT THAT MOMENT, regardless of
        // whether the caller passed anything -- this is the Controller's
        // own responsibility, not a caller-supplied parameter. Verify
        // across repeated advance/write cycles.
        CHECK(controller.current_version() == 0);
        CHECK(controller.version_for_sample("p1", 1) == 0); // stamped at version 0

        CHECK(controller.advance_version() == 1);
        CHECK(controller.advance_version() == 2);
        controller.update_production_status("p1", {3}, {"reward"});
        CHECK(controller.version_for_sample("p1", 3) == 2);

        CHECK(controller.advance_version() == 3);
        controller.update_production_status("p1", {4}, {"reward"});
        CHECK(controller.version_for_sample("p1", 4) == 3);

        // Earlier samples are unaffected by later advances -- the stamp is
        // permanent, not a live reference to current_version_.
        CHECK(controller.version_for_sample("p1", 1) == 0);
        CHECK(controller.version_for_sample("p1", 3) == 2);
    }

    // ---- shard_for_sample / version_for_sample: unknown cases ----
    {
        tq::TransferQueueController controller;
        CHECK(controller.shard_for_sample("missing", 1) == -1);
        CHECK(controller.version_for_sample("missing", 1) == -1);

        controller.create_partition("p1");
        CHECK(controller.shard_for_sample("p1", 999) == -1);   // known partition, never-written sample
        CHECK(controller.version_for_sample("p1", 999) == -1); // known partition, never-written sample
    }

    // ---- register_shard / shard_address: global registry ----
    {
        tq::TransferQueueController controller;

        CHECK(!controller.shard_address(5).has_value()); // never registered

        controller.register_shard(5, "tcp://host-a:1000");
        CHECK(controller.shard_address(5).has_value());
        CHECK(controller.shard_address(5).value() == "tcp://host-a:1000");

        // Re-registering the same shard_index with a different address: the
        // real implementation does `shard_registry_[shard_index] = address`
        // -- last write wins, not sticky to the first registration. Verify
        // that's actually what happens (not assumed).
        controller.register_shard(5, "tcp://host-b:2000");
        CHECK(controller.shard_address(5).value() == "tcp://host-b:2000");

        // Global, not per-partition: the registry has no partition_id
        // parameter at all, and survives partition creation/clearing
        // entirely untouched (clear_partition never touches shard_registry_).
        CHECK(controller.create_partition("A"));
        controller.register_shard(9, "tcp://host-c:3000");
        CHECK(controller.shard_address(9).value() == "tcp://host-c:3000");
        controller.clear_partition("A"); // partition gone...
        CHECK(controller.shard_address(9).value() == "tcp://host-c:3000"); // ...registry unaffected
        CHECK(controller.shard_address(5).value() == "tcp://host-b:2000"); // unrelated entry also unaffected
    }

    // ---- advance_version / current_version ----
    {
        tq::TransferQueueController controller;
        CHECK(controller.current_version() == 0); // starts at 0

        CHECK(controller.advance_version() == 1); // returns the NEW value, not the old one
        CHECK(controller.advance_version() == 2);
        CHECK(controller.advance_version() == 3);

        // current_version() is a pure read -- stable across repeated calls,
        // never auto-increments on its own.
        CHECK(controller.current_version() == 3);
        CHECK(controller.current_version() == 3);
        CHECK(controller.current_version() == 3);

        CHECK(controller.advance_version() == 4);
        CHECK(controller.current_version() == 4);
    }

    // ---- mark_consumed / ready_indexes / reset_consumption: routing + no-ops ----
    {
        tq::TransferQueueController controller;

        // Unknown partition: silent no-ops, empty results, never throws.
        CHECK_NOTHROW(controller.mark_consumed("missing", "trainer", {1}));
        CHECK(controller.ready_indexes("missing", {"reward"}, "trainer").empty());
        CHECK_NOTHROW(controller.reset_consumption("missing"));
        CHECK_NOTHROW(controller.reset_consumption("missing", std::optional<std::string>("trainer")));

        controller.create_partition("p1");
        controller.update_production_status("p1", {1, 2}, {"reward"});

        auto ready = controller.ready_indexes("p1", {"reward"}, "trainer");
        CHECK(contains(ready, 1) && contains(ready, 2));

        controller.mark_consumed("p1", "trainer", {1});
        auto ready_after_consume = controller.ready_indexes("p1", {"reward"}, "trainer");
        CHECK(!contains(ready_after_consume, 1));
        CHECK(contains(ready_after_consume, 2));

        // A different task hasn't consumed sample 1 -- still ready for it.
        CHECK(contains(controller.ready_indexes("p1", {"reward"}, "other_task"), 1));

        controller.reset_consumption("p1", std::optional<std::string>("trainer"));
        auto ready_after_reset = controller.ready_indexes("p1", {"reward"}, "trainer");
        CHECK(contains(ready_after_reset, 1) && contains(ready_after_reset, 2));

        // Routing correctness: a second, independent partition's state must
        // never leak into/out of "p1"'s.
        controller.create_partition("p2");
        controller.update_production_status("p2", {100}, {"reward"});
        controller.mark_consumed("p1", "trainer", {1, 2});
        CHECK(contains(controller.ready_indexes("p2", {"reward"}, "trainer"), 100)); // untouched by p1's consume
        CHECK(controller.ready_indexes("p1", {"reward"}, "trainer").empty());        // p1 fully consumed
    }

    // ---- clear_partition: unknown partition ----
    {
        tq::TransferQueueController controller;
        auto cleared = controller.clear_partition("missing");
        CHECK(cleared.sample_ids.empty());
        CHECK(cleared.shard_indices.empty());
    }

    // ---- clear_partition: parallel arrays, placeholder id included, shard attribution ----
    {
        tq::TransferQueueController controller;
        CHECK(controller.create_partition("p1")); // mints the pre-allocated placeholder id (0, fresh controller)

        controller.update_production_status("p1", {5}, {"reward"}, /*shard_index=*/3);
        controller.update_production_status("p1", {6}, {"reward"}); // no shard -> -1

        auto cleared = controller.clear_partition("p1");

        // sample_ids includes the pre-allocated placeholder (0) alongside
        // the two real written ids -- clear_partition's source of truth is
        // DataPartitionStatus::all_sample_ids(), which unions both sets.
        CHECK(cleared.sample_ids.size() == 3);
        CHECK(cleared.shard_indices.size() == cleared.sample_ids.size()); // parallel arrays, same length

        std::unordered_map<tq::SampleId, std::int32_t> shard_by_id;
        for (std::size_t i = 0; i < cleared.sample_ids.size(); ++i) {
            shard_by_id[cleared.sample_ids[i]] = cleared.shard_indices[i];
        }
        CHECK(shard_by_id.count(0) == 1 && shard_by_id.at(0) == -1); // placeholder, never shard-tagged
        CHECK(shard_by_id.count(5) == 1 && shard_by_id.at(5) == 3);  // explicit shard
        CHECK(shard_by_id.count(6) == 1 && shard_by_id.at(6) == -1); // no shard given

        // Clearing releases the partition entirely: it goes back to
        // "unknown partition" behavior, not a lingering empty-but-known one.
        CHECK(controller.ready_indexes("p1", {"reward"}, "trainer").empty());
        CHECK(controller.create_partition("p1")); // true again, not false -- it's genuinely gone
    }

    // ---- clear_partition: clear_consumption flag doesn't crash either way ----
    // (The partition object is erased immediately after clear_data() runs, so
    // this flag's effect on consumption_by_task_ has no way to be observed
    // through Controller's own API afterward -- see the "Optimization
    // findings" note in the final report. This just proves both values are
    // safe to pass and produce the same externally-visible result shape.)
    {
        for (bool clear_consumption : {true, false}) {
            tq::TransferQueueController controller;
            controller.create_partition("p1");
            controller.update_production_status("p1", {1}, {"reward"});
            controller.mark_consumed("p1", "trainer", {1});
            auto cleared = controller.clear_partition("p1", clear_consumption);
            CHECK(cleared.sample_ids.size() == 2); // placeholder + sample 1
            CHECK(controller.ready_indexes("p1", {"reward"}, "trainer").empty());
        }
    }

    // ---- Thread safety: concurrent, independent partitions ----
    {
        tq::TransferQueueController controller;
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 200;
        std::vector<std::thread> threads;
        std::vector<char> thread_ok(kThreads, 0); // vector<bool> is bit-packed and not safe to write
                                                   // concurrently across different "elements" -- known
                                                   // pitfall already documented elsewhere in this repo.

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::string partition = "thread-partition-" + std::to_string(t);
                bool ok = controller.create_partition(partition);
                for (int i = 0; i < kOpsPerThread; ++i) {
                    tq::SampleId id = static_cast<tq::SampleId>(i);
                    ok = ok && controller.update_production_status(partition, {id}, {"reward"}, /*shard_index=*/t);
                    ok = ok && (controller.shard_for_sample(partition, id) == t);
                }
                auto ready = controller.ready_indexes(partition, {"reward"}, "trainer");
                ok = ok && (ready.size() == static_cast<std::size_t>(kOpsPerThread));
                controller.mark_consumed(partition, "trainer", ready);
                ok = ok && controller.ready_indexes(partition, {"reward"}, "trainer").empty();
                thread_ok[t] = ok ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();

        for (int t = 0; t < kThreads; ++t) {
            CHECK(thread_ok[t] == 1);
        }

        // No cross-partition contamination: each thread's shard attribution
        // must still read back correctly after every thread has finished.
        for (int t = 0; t < kThreads; ++t) {
            std::string partition = "thread-partition-" + std::to_string(t);
            for (int i = 0; i < kOpsPerThread; ++i) {
                CHECK(controller.shard_for_sample(partition, static_cast<tq::SampleId>(i)) == t);
            }
        }
    }

    return tq::test::summary("ControllerTest");
}
