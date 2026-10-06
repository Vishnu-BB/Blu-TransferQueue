// Single-process smoke test -- NO BluTrain headers at all, plain g++. Only
// the genuinely dependency-free pieces: BatchMeta, FifoSampler,
// PartitionIndexManager, DataPartitionStatus, TransferQueueController,
// Message. StorageManager/Client now hold a real OwnTensor::Tensor and live
// in test_tensor_smoke.cpp (needs Tensor-Implementations, see `make
// test_tensor`). RankAwareSampler/Transport need the rest of BluTrain + a
// real process group (mpirun, see `make lib`).
//
// Run with: ./test_scaffold_smoke

#include <cassert>
#include <iostream>

#include "transferqueue/BatchMeta.h"
#include "transferqueue/Controller.h"
#include "transferqueue/DataPartitionStatus.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/GroupRouter.h"
#include "transferqueue/Message.h"
#include "transferqueue/PartitionIndexManager.h"
#include "transferqueue/Sampler.h"

int main() {
    // BatchMeta: is_ready() is false until every sample is produced.
    tq::BatchMeta not_ready({1, 2}, {"train", "train"}, {}, {true, false});
    assert(!not_ready.is_ready());
    tq::BatchMeta ready({1, 2}, {"train", "train"}, {}, {true, true});
    assert(ready.is_ready());
    tq::BatchMeta empty_batch;
    assert(!empty_batch.is_ready());

    // Sampler: FIFO splits ready ids into selected/remaining. group_ids is
    // ignored (plain FIFO, no grouping) -- pass an empty vector, same
    // length convention as a real caller would use when it doesn't track
    // groups at all.
    tq::FifoSampler sampler;
    auto [selected, remaining] = sampler.sample({10, 11, 12}, {"", "", ""}, 2);
    assert((selected == std::vector<tq::SampleId>{10, 11}));
    assert((remaining == std::vector<tq::SampleId>{12}));

    // GRPOGroupNSampler: groups by the real group_id parallel array, not by
    // numeric id adjacency (changed after an external review correctly
    // identified a silent cross-group contamination risk in the
    // adjacency-based approach -- see docs/GRPO_GROUP_N_SAMPLER.md).
    {
        tq::GRPOGroupNSampler sampler3(/*n_samples_per_prompt=*/3);

        // Group "A" only has 2 members (incomplete) -- nothing is
        // selected, everything stays ready, regardless of what the ids
        // themselves look like numerically.
        auto [sel1, rem1] = sampler3.sample({0, 1, 3, 4, 6, 7}, {"A", "A", "", "", "", ""}, /*batch_size=*/3);
        assert(sel1.empty());
        assert((rem1 == std::vector<tq::SampleId>{0, 1, 3, 4, 6, 7})); // sorted

        // Two complete groups "A" (ids 3,4,5) and "B" (ids 9,10,11); 0,1,6,7
        // have no group_id at all and can never be selected.
        auto [sel2, rem2] =
            sampler3.sample({0, 1, 3, 4, 5, 6, 7, 9, 10, 11}, {"", "", "A", "A", "A", "", "", "B", "B", "B"},
                             /*batch_size=*/6);
        assert((sel2 == std::vector<tq::SampleId>{3, 4, 5, 9, 10, 11}));
        assert((rem2 == std::vector<tq::SampleId>{0, 1, 6, 7}));

        // THE FIX, verified directly: the exact silent-contamination
        // scenario from the review. Worker A (prompt "p1") produces ids
        // {100,102}; worker B (prompt "p2") produces ids {101,103} --
        // numerically interleaved as [100,101,102,103]. The old
        // adjacency-based algorithm would have silently fused these into
        // one fake group of 4. Grouping by real group_id instead keeps
        // them correctly separated into two groups of 2.
        tq::GRPOGroupNSampler sampler2(/*n_samples_per_prompt=*/2);
        auto [sel_contam, rem_contam] =
            sampler2.sample({100, 102, 101, 103}, {"p1", "p1", "p2", "p2"}, /*batch_size=*/4);
        assert((sel_contam == std::vector<tq::SampleId>{100, 101, 102, 103})); // both real groups, never fused
        assert(rem_contam.empty());

        // batch_size must be a multiple of n_samples_per_prompt.
        bool threw = false;
        try {
            sampler3.sample({0, 1, 2}, {"A", "A", "A"}, /*batch_size=*/4);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        // ready_ids and group_ids must be the same length.
        bool length_mismatch_threw = false;
        try {
            sampler3.sample({0, 1, 2}, {"A", "A"}, /*batch_size=*/3);
        } catch (const std::invalid_argument&) {
            length_mismatch_threw = true;
        }
        assert(length_mismatch_threw);

        // n_samples_per_prompt must be positive.
        bool ctor_threw = false;
        try {
            tq::GRPOGroupNSampler bad_sampler(0);
        } catch (const std::invalid_argument&) {
            ctor_threw = true;
        }
        assert(ctor_threw);

        // n_samples_per_prompt=1: every id with a (non-empty) group_id is
        // trivially its own complete "group" -- an id with no group_id
        // still can't be selected even at n=1.
        tq::GRPOGroupNSampler sampler1(1);
        auto [sel3, rem3] = sampler1.sample({5, 2, 8}, {"g5", "g2", ""}, /*batch_size=*/2);
        assert((sel3 == std::vector<tq::SampleId>{2, 5})); // sorted, group_id order doesn't matter here
        assert((rem3 == std::vector<tq::SampleId>{8}));    // "8" has no group_id -- stranded regardless of batch_size
    }

    // PartitionIndexManager: reuse pool (FIFO) checked before minting new ids.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p1", 3); // {0, 1, 2}
        assert(a.size() == 3);
        mgr.release_indexes("p1", {a[1]});
        auto b = mgr.allocate_indexes("p1", 1);
        assert(b[0] == a[1]); // reused, not a fresh id
        auto released = mgr.release_partition("p1");
        assert(released.size() == 3);
        assert(mgr.get_indexes_for_partition("p1").empty());
    }

    // DataPartitionStatus: scan_data_status is per-task -- ready for one
    // task doesn't mean ready for another, and consuming removes it only
    // for the consuming task.
    {
        tq::DataPartitionStatus partition("p1");
        partition.update_production_status({1, 2}, {"tokens"});
        partition.update_production_status({1}, {"reward"}); // sample 2 missing "reward"

        auto ready_for_actor = partition.scan_data_status({"tokens", "reward"}, "trainer");
        assert((ready_for_actor == std::vector<tq::SampleId>{1}));

        partition.mark_consumed("trainer", {1});
        assert(partition.scan_data_status({"tokens", "reward"}, "trainer").empty());
        // A different task hasn't consumed it -- still ready for that task.
        assert((partition.scan_data_status({"tokens", "reward"}, "other_task") == std::vector<tq::SampleId>{1}));

        // Phase 5: shard_index is -1 (unknown) until recorded via a
        // sharded update_production_status call; clear_data removes it.
        assert(partition.shard_for_sample(1) == -1);
        partition.update_production_status({1}, {"tokens"}, /*shard_index=*/2);
        assert(partition.shard_for_sample(1) == 2);
        partition.clear_data({1});
        assert(partition.shard_for_sample(1) == -1);

        // Phase 6: policy_version is -1 (unknown) until recorded via an
        // update_production_status call that supplies one; clear_data
        // removes it too.
        assert(partition.version_for_sample(2) == -1);
        partition.update_production_status({2}, {"tokens"}, std::nullopt, /*policy_version=*/7);
        assert(partition.version_for_sample(2) == 7);
        partition.clear_data({2});
        assert(partition.version_for_sample(2) == -1);

        // group_id (GRPOGroupNSampler's grouping key): "" (unknown) until
        // recorded; clear_data removes it too.
        assert(partition.group_id_for_sample(3) == "");
        partition.update_production_status({3}, {"tokens"}, std::nullopt, std::nullopt, /*group_id=*/"prompt-9");
        assert(partition.group_id_for_sample(3) == "prompt-9");
        partition.clear_data({3});
        assert(partition.group_id_for_sample(3) == "");

        // produced_at: -1 (unknown) until recorded; ALWAYS auto-stamped by
        // update_production_status (no optional param, unlike shard/
        // version/group_id -- there's nothing for a caller to supply, it's
        // just "when did this call happen"); first-write-wins (a later
        // update for the same id doesn't move it); clear_data removes it.
        assert(partition.produced_at(4) == -1);
        partition.update_production_status({4}, {"tokens"});
        std::int64_t first_stamp = partition.produced_at(4);
        assert(first_stamp >= 0);
        partition.update_production_status({4}, {"reward"}); // same id, different field, later call
        assert(partition.produced_at(4) == first_stamp); // unchanged -- first-write-wins
        partition.clear_data({4});
        assert(partition.produced_at(4) == -1);
    }

    // TransferQueueController: routes to the right partition; unknown
    // partition/field returns empty rather than throwing.
    {
        tq::TransferQueueController controller;
        assert(controller.create_partition("p1"));
        assert(!controller.create_partition("p1")); // already exists
        assert(controller.update_production_status("p1", {5}, {"tokens"}));
        assert(!controller.update_production_status("missing", {5}, {"tokens"}));
        assert((controller.ready_indexes("p1", {"tokens"}, "trainer") == std::vector<tq::SampleId>{5}));
        assert(controller.ready_indexes("missing", {"tokens"}, "trainer").empty());

        // Phase 5: shard_index threads through update_production_status,
        // and the global shard registry (independent of any partition)
        // resolves addresses.
        assert(controller.update_production_status("p1", {6}, {"tokens"}, /*shard_index=*/3));
        assert(controller.shard_for_sample("p1", 6) == 3);
        assert(controller.shard_for_sample("p1", 5) == -1); // recorded without a shard above
        assert(controller.shard_for_sample("missing", 6) == -1);

        assert(!controller.shard_address(7).has_value());
        controller.register_shard(7, "tcp://127.0.0.1:9000");
        assert(controller.shard_address(7).value() == "tcp://127.0.0.1:9000");

        // group_id (GRPOGroupNSampler's grouping key) threads through the
        // same way, as a trailing optional param.
        assert(controller.group_id_for_sample("p1", 6) == "");
        assert(controller.update_production_status("p1", {6}, {"tokens"}, /*shard_index=*/3,
                                                     /*group_id=*/"prompt-A"));
        assert(controller.group_id_for_sample("p1", 6) == "prompt-A");
        assert(controller.group_id_for_sample("p1", 5) == ""); // recorded without a group_id above
        assert(controller.group_id_for_sample("missing", 6) == "");

        // produced_at threads through too, auto-stamped (no caller param).
        assert(controller.produced_at("p1", 6) >= 0);
        assert(controller.produced_at("p1", 999) == -1); // unknown sample
        assert(controller.produced_at("missing", 6) == -1); // unknown partition

        // clear_partition now also reports each cleared id's shard.
        // sample_ids also includes the pre-allocated placeholder id (0,
        // minted by create_partition) alongside the real 5 and 6 -- see
        // docs/CLEAR_PARTITION_FIX.md.
        auto cleared = controller.clear_partition("p1");
        assert(cleared.sample_ids.size() == 3);
        for (std::size_t i = 0; i < cleared.sample_ids.size(); ++i) {
            std::int32_t expected = cleared.sample_ids[i] == 6 ? 3 : -1;
            assert(cleared.shard_indices[i] == expected);
        }
    }

    // Controller::find_stranded_groups: track-and-expose age-based
    // reporting end to end through Controller + a real GRPOGroupNSampler --
    // never consumes/clears anything (confirmed decision, see
    // docs/GRPO_GROUP_N_SAMPLER.md). max_age_ms=0 means "report anything
    // already produced" (avoids a flaky sleep()); a huge max_age_ms means
    // "report nothing yet," proving the age check is real, not a no-op.
    {
        tq::TransferQueueController controller;
        controller.create_partition("p1");
        tq::GRPOGroupNSampler sampler(/*n_samples_per_prompt=*/3);

        controller.update_production_status("p1", {1, 2}, {"tokens"}, std::nullopt, "A"); // incomplete (2 < 3)
        controller.update_production_status("p1", {3, 4, 5}, {"tokens"}, std::nullopt, "B"); // complete (3 == 3)
        controller.update_production_status("p1", {6}, {"tokens"}); // no group_id at all

        auto stranded_now = controller.find_stranded_groups("p1", {"tokens"}, "trainer", sampler, /*max_age_ms=*/0);
        assert(stranded_now.size() == 2); // "A" (incomplete) and "" (ungrouped) -- never "B" (complete)
        bool found_a = false;
        bool found_ungrouped = false;
        for (const auto& group : stranded_now) {
            assert(group.oldest_age_ms >= 0);
            if (group.group_id == "A") {
                found_a = true;
                assert((group.sample_ids == std::vector<tq::SampleId>{1, 2}));
            } else if (group.group_id.empty()) {
                found_ungrouped = true;
                assert((group.sample_ids == std::vector<tq::SampleId>{6}));
            } else {
                assert(false && "unexpected stranded group_id"); // "B" must never appear here
            }
        }
        assert(found_a && found_ungrouped);

        // Nothing is old enough yet with a huge threshold.
        auto stranded_far_future =
            controller.find_stranded_groups("p1", {"tokens"}, "trainer", sampler, /*max_age_ms=*/3600000);
        assert(stranded_far_future.empty());

        // Unknown partition -> empty, not a throw.
        assert(controller.find_stranded_groups("missing", {"tokens"}, "trainer", sampler, 0).empty());

        // Confirm nothing was actually consumed/cleared by any of the
        // calls above -- ready_indexes still sees every sample untouched.
        auto still_ready = controller.ready_indexes("p1", {"tokens"}, "trainer");
        assert(still_ready.size() == 6);
    }

    // Phase 6: current_version() starts at 0 and only moves via
    // advance_version(); update_production_status() auto-stamps every
    // sample with whatever that was at write time -- the caller never
    // supplies a version itself, since the Controller is the single source
    // of truth for "what version is it right now."
    {
        tq::TransferQueueController controller;
        assert(controller.create_partition("p1"));
        assert(controller.current_version() == 0);

        assert(controller.update_production_status("p1", {1}, {"tokens"}));
        assert(controller.version_for_sample("p1", 1) == 0); // stamped at version 0

        assert(controller.advance_version() == 1);
        assert(controller.advance_version() == 2);
        assert(controller.current_version() == 2);

        assert(controller.update_production_status("p1", {2}, {"tokens"}));
        assert(controller.version_for_sample("p1", 2) == 2); // stamped at the now-current version 2
        assert(controller.version_for_sample("p1", 1) == 0); // unaffected by later advances

        assert(controller.version_for_sample("p1", 999) == -1);   // unknown sample
        assert(controller.version_for_sample("missing", 1) == -1); // unknown partition
    }

    // Schema validation (Option B): declare_schema() is upfront
    // registration -- creates the partition if needed, re-declaring the
    // same field with a different dtype throws, matching dtypes validate
    // silently, mismatches throw, and a field never declared is never
    // checked at all (gradual adoption).
    {
        tq::TransferQueueController controller;
        controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}});

        controller.validate_schema("p1", {{"reward", tq::FieldDtype::Float32}}); // matches, no throw
        controller.validate_schema("p1", {{"undeclared_field", tq::FieldDtype::Int64}}); // never declared, no throw
        controller.validate_schema("missing", {{"reward", tq::FieldDtype::Int64}}); // unknown partition, no throw

        bool threw = false;
        try {
            controller.validate_schema("p1", {{"reward", tq::FieldDtype::Float64}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        bool redeclare_threw = false;
        try {
            controller.declare_schema("p1", {{"reward", tq::FieldDtype::Int64}}); // conflicting re-declaration
        } catch (const std::invalid_argument&) {
            redeclare_threw = true;
        }
        assert(redeclare_threw);

        // Re-declaring with the SAME dtype is fine (idempotent).
        controller.declare_schema("p1", {{"reward", tq::FieldDtype::Float32}});
    }

    // Message: round-trips the control-plane envelope -- GET_META-shaped
    // request (partition/fields/task_name), a CLEAR_PARTITION-shaped request
    // using `flag` for clear_consumption, and a PUT_DATA-shaped request
    // carrying an opaque payload blob (real tensor (de)serialization is
    // test_tensor_smoke.cpp's job now).
    {
        tq::MessageBody body;
        body.partition_id = "rollout@0";
        body.sample_ids = {1, 2, 3};
        body.fields = {"tokens", "reward"};
        body.task_name = "trainer";
        body.batch_size = 8;
        body.field_dtypes = {tq::FieldDtype::Float32, tq::FieldDtype::Int64};
        body.payload = {0xDE, 0xAD, 0xBE, 0xEF};
        body.shard_index = -1; // "no shard" sentinel -- must survive as negative, not wrap to a huge positive
        body.address = "tcp://127.0.0.1:5555";
        body.sample_shard_indices = {0, 1, -1};
        body.shard_registry_indices = {0, 1};
        body.shard_registry_addresses = {"tcp://127.0.0.1:6000", "tcp://127.0.0.1:6001"};
        body.sample_versions = {0, 2, -1}; // "no version" sentinel must survive as negative, not wrap
        body.current_version = 2;
        body.group_id = "prompt-42";

        auto msg = tq::Message::create(tq::RequestType::GET_META, "client-1", body, "controller-1");
        auto bytes = msg.serialize();
        auto decoded = tq::Message::deserialize(bytes);

        assert(decoded.request_type == tq::RequestType::GET_META);
        assert(decoded.sender_id == "client-1");
        assert(decoded.receiver_id.has_value() && *decoded.receiver_id == "controller-1");
        assert(decoded.request_id == msg.request_id);
        assert(decoded.body.partition_id == "rollout@0");
        assert((decoded.body.sample_ids == std::vector<tq::SampleId>{1, 2, 3}));
        assert((decoded.body.fields == std::vector<std::string>{"tokens", "reward"}));
        assert(decoded.body.task_name == "trainer");
        assert(decoded.body.batch_size == 8);
        assert((decoded.body.field_dtypes == std::vector<tq::FieldDtype>{tq::FieldDtype::Float32, tq::FieldDtype::Int64}));
        assert((decoded.body.payload == std::vector<std::uint8_t>{0xDE, 0xAD, 0xBE, 0xEF}));
        assert(decoded.body.shard_index == -1);
        assert(decoded.body.address == "tcp://127.0.0.1:5555");
        assert((decoded.body.sample_shard_indices == std::vector<std::int32_t>{0, 1, -1}));
        assert((decoded.body.shard_registry_indices == std::vector<std::int32_t>{0, 1}));
        assert((decoded.body.shard_registry_addresses ==
                std::vector<std::string>{"tcp://127.0.0.1:6000", "tcp://127.0.0.1:6001"}));
        assert((decoded.body.sample_versions == std::vector<std::int64_t>{0, 2, -1}));
        assert(decoded.body.current_version == 2);
        assert(decoded.body.group_id == "prompt-42");

        // No receiver_id -> round-trips as nullopt, not an empty string.
        auto broadcast = tq::Message::create(tq::RequestType::HANDSHAKE, "storage-1", {});
        assert(!tq::Message::deserialize(broadcast.serialize()).receiver_id.has_value());

        // `flag` (request-side bool) and `success`/`error_message`
        // (response-side) round-trip independently.
        tq::MessageBody clear_body;
        clear_body.partition_id = "rollout@0";
        clear_body.flag = true; // clear_consumption
        auto clear_msg = tq::Message::create(tq::RequestType::CLEAR_PARTITION, "client-1", clear_body);
        assert(tq::Message::deserialize(clear_msg.serialize()).body.flag);

        tq::MessageBody error_body;
        error_body.success = false;
        error_body.error_message = "partition not found";
        auto error_msg = tq::Message::create(tq::RequestType::REQUEST_ERROR, "controller-1", error_body);
        auto decoded_error = tq::Message::deserialize(error_msg.serialize());
        assert(!decoded_error.body.success);
        assert(decoded_error.body.error_message == "partition not found");

        // Truncated/empty buffers must fail loudly, not silently misparse --
        // never read past the buffer, even for a single-byte field.
        auto expect_throw = [](const std::vector<std::uint8_t>& bytes) {
            bool threw = false;
            try {
                tq::Message::deserialize(bytes);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            assert(threw);
        };
        expect_throw({});
        expect_throw({0x00});             // request_type only, nothing after
        expect_throw({0x00, 0x00, 0x00}); // mid-length-prefix of sender_id
    }

    // GroupRouter: deterministic (same group_id -> same rank, every call),
    // every answer within [0, world_size), and different world_size values
    // don't crash on a world_size of 1 (the degenerate single-rank case).
    {
        tq::GroupRouter router(4);
        int first = router.target_rank("grpo-group-7");
        for (int i = 0; i < 5; ++i) {
            assert(router.target_rank("grpo-group-7") == first);
        }
        assert(first >= 0 && first < 4);

        tq::GroupRouter single_rank(1);
        assert(single_rank.target_rank("grpo-group-7") == 0);

        bool threw = false;
        try {
            tq::GroupRouter bad(0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    std::cout << "test_scaffold_smoke: PASS\n";
    return 0;
}
