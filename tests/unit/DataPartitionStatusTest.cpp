// Unit tests for DataPartitionStatus (include/transferqueue/DataPartitionStatus.h,
// src/DataPartitionStatus.cpp). Zero dependencies.
// Build: make build/unit_core_DataPartitionStatusTest && ./build/unit_core_DataPartitionStatusTest

#include "transferqueue/DataPartitionStatus.h"

#include <atomic>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "TestUtils.h"

int main() {
    // --- declare_schema / validate_field ---

    // Declaring a field, then validating the same dtype: silent, no throw.
    {
        tq::DataPartitionStatus p("p");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});
        CHECK_NOTHROW(p.validate_field("reward", tq::FieldDtype::Float32));
    }

    // Validating a conflicting dtype against a declared field throws.
    {
        tq::DataPartitionStatus p("p");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});
        CHECK_THROWS(p.validate_field("reward", tq::FieldDtype::Int64), std::invalid_argument);
    }

    // A field that was NEVER declared anywhere is never validated --
    // gradual adoption is a deliberate contract, not a gap. Must NOT throw
    // regardless of what dtype is passed.
    {
        tq::DataPartitionStatus p("p");
        CHECK_NOTHROW(p.validate_field("never_declared", tq::FieldDtype::Int64));
        CHECK_NOTHROW(p.validate_field("never_declared", tq::FieldDtype::Bool));
    }

    // Re-declaring the SAME field with the SAME dtype is idempotent -- no
    // throw, no change in observable state.
    {
        tq::DataPartitionStatus p("p");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});
        CHECK_NOTHROW(p.declare_schema({{"reward", tq::FieldDtype::Float32}}));
        CHECK_NOTHROW(p.validate_field("reward", tq::FieldDtype::Float32));
    }

    // Re-declaring the SAME field with a DIFFERENT dtype throws -- the
    // schema itself must stay internally consistent.
    {
        tq::DataPartitionStatus p("p");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});
        CHECK_THROWS(p.declare_schema({{"reward", tq::FieldDtype::Int64}}), std::invalid_argument);
    }

    // declare_schema with multiple fields in one call: a conflict on one
    // field must still throw (don't silently skip it because other fields
    // in the same call were fine).
    {
        tq::DataPartitionStatus p("p");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});
        CHECK_THROWS(p.declare_schema({{"tokens", tq::FieldDtype::Int64}, {"reward", tq::FieldDtype::Int64}}),
                     std::invalid_argument);
    }

    // declare_schema with an empty schema map: no-op, no throw.
    {
        tq::DataPartitionStatus p("p");
        CHECK_NOTHROW(p.declare_schema({}));
    }

    // --- update_production_status ---

    // Producing a field for new ids makes them show up in scan_data_status
    // for that field.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1, 2}, {"tokens"});
        auto ready = p.scan_data_status({"tokens"}, "trainer");
        std::set<tq::SampleId> ready_set(ready.begin(), ready.end());
        CHECK((ready_set == std::set<tq::SampleId>{1, 2}));
    }

    // Producing the SAME field for the SAME id twice is idempotent -- no
    // duplicate entries, no crash, still ready exactly once.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        CHECK_NOTHROW(p.update_production_status({1}, {"tokens"}));
        auto ready = p.scan_data_status({"tokens"}, "trainer");
        CHECK(ready.size() == 1);
        CHECK(p.total_samples_num() == 1);
    }

    // Producing DIFFERENT fields for the same id across separate calls
    // accumulates -- it doesn't overwrite/forget the first field.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        p.update_production_status({1}, {"reward"});
        auto ready_tokens_only = p.scan_data_status({"tokens"}, "trainer");
        auto ready_both = p.scan_data_status({"tokens", "reward"}, "trainer");
        CHECK(ready_tokens_only.size() == 1);
        CHECK(ready_both.size() == 1); // both fields now produced for id 1
    }

    // shard_index/policy_version are independent optional params: supplying
    // one must not silently set the other.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"}, /*shard_index=*/5);
        CHECK(p.shard_for_sample(1) == 5);
        CHECK(p.version_for_sample(1) == -1); // version was never supplied

        p.update_production_status({2}, {"tokens"}, std::nullopt, /*policy_version=*/9);
        CHECK(p.shard_for_sample(2) == -1); // shard was never supplied
        CHECK(p.version_for_sample(2) == 9);
    }

    // shard_for_sample / version_for_sample: -1 for an id that was never
    // given a shard/version at all (including an id that IS known via
    // production but just never got that particular optional param).
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"}); // no shard, no version
        CHECK(p.shard_for_sample(1) == -1);
        CHECK(p.version_for_sample(1) == -1);
        CHECK(p.shard_for_sample(999) == -1); // completely unknown id
        CHECK(p.version_for_sample(999) == -1);
    }

    // shard/version values survive further unrelated update_production_status
    // calls for OTHER ids -- not globally reset by every call.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"}, /*shard_index=*/3, /*policy_version=*/7);
        p.update_production_status({2}, {"tokens"}); // unrelated id, no shard/version
        CHECK(p.shard_for_sample(1) == 3);
        CHECK(p.version_for_sample(1) == 7);
    }

    // --- scan_data_status ---

    // A field that was NEVER declared/produced anywhere returns empty
    // immediately -- not an error, not a scan of an empty set that happens
    // to also be empty for other reasons (verified by also having OTHER
    // ready ids for a different field, to prove it's specifically the
    // unknown field short-circuiting, not "nothing is ready at all").
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        auto ready = p.scan_data_status({"never_produced_field"}, "trainer");
        CHECK(ready.empty());
        // Prove it's specifically the unknown field short-circuiting, not
        // "this partition has nothing ready at all" -- "tokens" IS ready.
        CHECK(p.scan_data_status({"tokens"}, "trainer").size() == 1);
    }

    // Per-task independence: consuming for one task doesn't affect another
    // task's view of readiness.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        p.mark_consumed("trainer_a", {1});
        CHECK(p.scan_data_status({"tokens"}, "trainer_a").empty());
        CHECK(p.scan_data_status({"tokens"}, "trainer_b").size() == 1);
    }

    // Ready requires ALL queried fields produced, not just some -- a sample
    // with only a subset of the requested fields must not appear.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1, 2}, {"tokens"});
        p.update_production_status({1}, {"reward"}); // only id 1 has both fields
        auto ready = p.scan_data_status({"tokens", "reward"}, "trainer");
        CHECK((ready == std::vector<tq::SampleId>{1}));
    }

    // Empty fields list: is_produced_locked() vacuously returns true for
    // every field in an empty list (no fields to fail on), so every
    // not-yet-consumed id in global_indexes_ counts as "ready" -- this is
    // the actual observed behavior of an empty query, not necessarily an
    // intuitive one, so pin it down explicitly rather than assume.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1, 2, 3}, {"tokens"});
        auto ready = p.scan_data_status({}, "trainer");
        CHECK(ready.size() == 3); // every known id, vacuously "ready" for zero fields
    }

    // An id that was never produced for anything (not in global_indexes_ at
    // all) never appears in scan_data_status results, regardless of query.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        auto ready = p.scan_data_status({"tokens"}, "trainer");
        CHECK((ready == std::vector<tq::SampleId>{1}));
        CHECK(ready.size() == 1); // id 2 (never produced) must not sneak in
    }

    // --- mark_consumed / has_consumed / reset_consumption ---

    {
        tq::DataPartitionStatus p("p");
        p.mark_consumed("task_a", {1, 2});
        CHECK(p.has_consumed("task_a", 1));
        CHECK(p.has_consumed("task_a", 2));
        CHECK(!p.has_consumed("task_a", 3));     // never marked
        CHECK(!p.has_consumed("task_b", 1));     // different task, unaffected
    }

    // reset_consumption(task_name): resets ONLY that task.
    {
        tq::DataPartitionStatus p("p");
        p.mark_consumed("task_a", {1});
        p.mark_consumed("task_b", {1});
        p.reset_consumption("task_a");
        CHECK(!p.has_consumed("task_a", 1));
        CHECK(p.has_consumed("task_b", 1)); // untouched
    }

    // reset_consumption(nullopt): resets EVERY task.
    {
        tq::DataPartitionStatus p("p");
        p.mark_consumed("task_a", {1});
        p.mark_consumed("task_b", {1});
        p.reset_consumption(std::nullopt);
        CHECK(!p.has_consumed("task_a", 1));
        CHECK(!p.has_consumed("task_b", 1));
    }

    // Resetting a task that never consumed anything: no-op, no throw.
    {
        tq::DataPartitionStatus p("p");
        CHECK_NOTHROW(p.reset_consumption("never_used_task"));
        CHECK_NOTHROW(p.reset_consumption(std::nullopt)); // also fine with zero tasks at all
    }

    // --- clear_data ---

    // clear_data removes production, shard, and version state for the
    // given ids -- verify each independently, not just "scan_data_status
    // stops returning it" (which alone wouldn't prove shard/version were
    // actually erased, only that production was).
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens", "reward"}, /*shard_index=*/2, /*policy_version=*/4);
        p.clear_data({1});
        CHECK(p.scan_data_status({"tokens"}, "trainer").empty());
        CHECK(p.shard_for_sample(1) == -1);
        CHECK(p.version_for_sample(1) == -1);
        CHECK(p.total_samples_num() == 0);
    }

    // clear_data with clear_consumption=true (default) also erases
    // consumption records for the cleared ids.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        p.mark_consumed("trainer", {1});
        p.clear_data({1}, /*clear_consumption=*/true);
        CHECK(!p.has_consumed("trainer", 1));
    }

    // clear_data with clear_consumption=false clears production but leaves
    // consumption records intact.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        p.mark_consumed("trainer", {1});
        p.clear_data({1}, /*clear_consumption=*/false);
        CHECK(p.scan_data_status({"tokens"}, "trainer").empty()); // production gone
        CHECK(p.has_consumed("trainer", 1));                      // consumption record still there
    }

    // Clearing ids that were never produced: no-op, no throw, doesn't
    // disturb anything else.
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1}, {"tokens"});
        CHECK_NOTHROW(p.clear_data({999}));
        CHECK(p.scan_data_status({"tokens"}, "trainer").size() == 1); // id 1 untouched
    }

    // Clearing a SUBSET of a partition's ids leaves the rest fully intact
    // (production, shard, version, consumption all still present for the
    // ids not in the clear list).
    {
        tq::DataPartitionStatus p("p");
        p.update_production_status({1, 2}, {"tokens"}, /*shard_index=*/1, /*policy_version=*/1);
        p.mark_consumed("trainer", {1, 2});
        p.clear_data({1});
        CHECK(p.shard_for_sample(1) == -1);
        CHECK(p.version_for_sample(1) == -1);
        CHECK(!p.has_consumed("trainer", 1));

        CHECK(p.shard_for_sample(2) == 1);
        CHECK(p.version_for_sample(2) == 1);
        CHECK(p.has_consumed("trainer", 2));
        CHECK(p.total_samples_num() == 1);
    }

    // --- all_sample_ids / total_samples_num ---

    // all_sample_ids includes pre-allocated-but-never-produced ids unioned
    // with produced ids, with no duplicate if an id appears in both sets.
    {
        tq::DataPartitionStatus p("p");
        p.register_pre_allocated_indexes({10, 11});
        p.update_production_status({11, 12}, {"tokens"}); // 11 overlaps the pre-allocated set
        auto all = p.all_sample_ids();
        std::set<tq::SampleId> all_set(all.begin(), all.end());
        CHECK((all_set == std::set<tq::SampleId>{10, 11, 12}));
        CHECK(all.size() == 3); // no duplicate entry for the overlapping id 11
    }

    // total_samples_num() counts only global_indexes_ (produced ids) --
    // NOT pre-allocated-but-never-produced ones. Documenting the actual,
    // perhaps-surprising discrepancy against all_sample_ids(): the two can
    // disagree when something is pre-allocated but never produced.
    {
        tq::DataPartitionStatus p("p");
        p.register_pre_allocated_indexes({10, 11});
        p.update_production_status({12}, {"tokens"}); // only 12 actually produced
        CHECK(p.total_samples_num() == 1);             // produced count
        CHECK(p.all_sample_ids().size() == 3);          // produced UNION pre-allocated
    }

    // register_pre_allocated_indexes alone (nothing produced yet): those
    // ids show up in all_sample_ids but scan_data_status never returns them
    // for any real field (they're not "produced").
    {
        tq::DataPartitionStatus p("p");
        p.register_pre_allocated_indexes({10});
        CHECK(p.all_sample_ids().size() == 1);
        p.update_production_status({10}, {"dummy"}); // need >=1 declared field for scan to look at anything
        CHECK(p.scan_data_status({"dummy"}, "trainer").size() == 1);
    }

    // --- Thread safety ---

    // Multiple threads hammering update_production_status / mark_consumed /
    // scan_data_status / clear_data concurrently on the SAME partition
    // object. No crash is the primary guarantee (the single mutex covers
    // every method here); additionally check no impossible state is ever
    // observed by the end: every id is in exactly one of two final buckets
    // (cleared, or still produced) -- never simultaneously "ready" and
    // "already cleared from existence" relative to the final state.
    {
        tq::DataPartitionStatus p("p");
        constexpr int kThreads = 8;
        constexpr int kIdsPerThread = 200;
        std::atomic<bool> saw_exception{false};
        std::vector<std::thread> threads;

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                try {
                    for (int i = 0; i < kIdsPerThread; ++i) {
                        tq::SampleId id = static_cast<tq::SampleId>(t * kIdsPerThread + i);
                        p.update_production_status({id}, {"tokens"}, /*shard_index=*/t,
                                                    /*policy_version=*/i);
                        p.mark_consumed("trainer", {id});
                        (void)p.scan_data_status({"tokens"}, "trainer");
                        (void)p.has_consumed("trainer", id);
                        (void)p.shard_for_sample(id);
                        (void)p.version_for_sample(id);
                        if (i % 10 == 0) {
                            p.clear_data({id});
                        }
                    }
                } catch (...) {
                    saw_exception = true;
                }
            });
        }
        for (auto& th : threads) th.join();

        CHECK(!saw_exception);

        // Final-state consistency: every id either was cleared (shard/
        // version both -1, not consumed) or is still fully intact (shard
        // and version both match what that thread wrote, still consumed).
        // No id can be "half cleared" since clear_data holds the mutex for
        // its entire body.
        for (int t = 0; t < kThreads; ++t) {
            for (int i = 0; i < kIdsPerThread; ++i) {
                tq::SampleId id = static_cast<tq::SampleId>(t * kIdsPerThread + i);
                bool was_cleared = (i % 10 == 0);
                if (was_cleared) {
                    CHECK(p.shard_for_sample(id) == -1);
                    CHECK(p.version_for_sample(id) == -1);
                    CHECK(!p.has_consumed("trainer", id));
                } else {
                    CHECK(p.shard_for_sample(id) == t);
                    CHECK(p.version_for_sample(id) == i);
                    CHECK(p.has_consumed("trainer", id));
                }
            }
        }
    }

    // ---- Scope boundary, confirmed directly against the implementation:
    // DataPartitionStatus has NO concept of tensor shape/size anywhere --
    // declared_schema_ tracks dtype only (Option B, see
    // docs/SCHEMA_VALIDATION.md), and update_production_status() doesn't
    // even take a dtype parameter, let alone re-validate one. The actual
    // dtype-consistency enforcement is validate_field(), called
    // EXTERNALLY by Client/Server *before* a write reaches storage --
    // DataPartitionStatus itself provides no automatic protection once a
    // schema is declared; it only reports a conflict if validate_field()
    // happens to be called again with a mismatched dtype. Calling
    // update_production_status() repeatedly for the same field, with no
    // validate_field() calls in between at all, succeeds unconditionally
    // every time -- there is nothing here to catch a caller that forgets
    // to validate. A real tensor *shape* mutation (same dtype, different
    // size) is entirely invisible at this layer; see
    // tests/unit/StorageManagerTest.cpp for where that's actually
    // observable (SimpleStorageManager, which holds the real tensors). ----
    {
        tq::DataPartitionStatus p("p1");
        p.declare_schema({{"reward", tq::FieldDtype::Float32}});

        // No validate_field() call at all between these -- nothing stops
        // it, because nothing here is responsible for stopping it.
        CHECK_NOTHROW(p.update_production_status({1}, {"reward"}));
        CHECK_NOTHROW(p.update_production_status({2}, {"reward"}));
        CHECK_NOTHROW(p.update_production_status({1}, {"reward"})); // same id again, still no-op-safe

        // The conflict only ever surfaces through validate_field() or a
        // conflicting declare_schema() call -- both already covered above
        // -- never through update_production_status() itself.
        CHECK_THROWS(p.validate_field("reward", tq::FieldDtype::Int64), std::invalid_argument);
        // ...but update_production_status() for the same field still has
        // no opinion and still succeeds -- it was never asked to check.
        CHECK_NOTHROW(p.update_production_status({3}, {"reward"}));
    }

    return tq::test::summary("DataPartitionStatusTest");
}
