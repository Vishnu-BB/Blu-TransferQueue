// Unit tests for the in-process TransferQueueClient
// (include/transferqueue/Client.h, src/Client.cpp), wired to a real
// TransferQueueController + SimpleStorageManager + sampler (FifoSampler and
// GRPOGroupNSampler both exercised). Needs Tensor-Implementations -- build
// with `make unit_tensor_ClientTest`.

#include "transferqueue/Client.h"

#include <memory>
#include <stdexcept>
#include <thread>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"
#include "TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

Tensor make_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

Tensor make_i64(const std::vector<std::int64_t>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

std::unique_ptr<tq::TransferQueueClient> make_client(std::shared_ptr<tq::BaseSampler> sampler = nullptr) {
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    if (!sampler) sampler = std::make_shared<tq::FifoSampler>();
    return std::make_unique<tq::TransferQueueClient>(controller, storage, sampler);
}

} // namespace

int main() {
    // ---- declare_schema + put: fail-fast, whole-call atomicity ----
    {
        auto client = make_client();
        client->declare_schema("p1", {{"reward", tq::FieldDtype::Float32}});

        // Matching dtype succeeds.
        tq::Record good;
        good["reward"] = make_f32(1.0f);
        CHECK_NOTHROW(client->put("p1", {"reward"}, {{1, good}}));

        // Conflicting dtype throws, and crucially does so BEFORE any write
        // for that call -- including fields in the SAME record that didn't
        // conflict. One record with a valid "tokens" field and a
        // conflicting "reward" field: neither must be written.
        client->declare_schema("p2", {{"reward", tq::FieldDtype::Float32}});
        tq::Record mixed;
        mixed["tokens"] = make_i64({1, 2, 3}); // fine, never declared
        mixed["reward"] = make_i64({1});       // conflicts: declared Float32, this is Int64
        CHECK_THROWS(client->put("p2", {"tokens", "reward"}, {{5, mixed}}), std::invalid_argument);

        // Neither field of sample 5 was actually written -- get() for
        // "tokens" alone (never declared, always valid) must still see
        // nothing, proving the whole call was rejected before storage_->put_data.
        auto got = client->get("p2", {"tokens"}, "trainer", 10);
        CHECK(got.empty());

        // A field never declared is never validated -- put() for a
        // genuinely undeclared field in a fresh partition must succeed.
        auto client2 = make_client();
        tq::Record undeclared;
        undeclared["anything"] = make_f32(42.0f);
        CHECK_NOTHROW(client2->put("p3", {"anything"}, {{1, undeclared}}));
    }

    // ---- put/get round trip + per-task consumption scoping ----
    {
        auto client = make_client();
        tq::Record r;
        r["reward"] = make_f32(7.0f);
        client->put("p1", {"reward"}, {{1, r}});

        auto got_trainer = client->get("p1", {"reward"}, "trainer", 10);
        CHECK(got_trainer.size() == 1);
        CHECK(static_cast<float*>(got_trainer.at(1).at("reward").data())[0] == 7.0f);

        // Same task, same data: nothing left -- consumption is real, not a
        // read-only peek.
        auto got_trainer_again = client->get("p1", {"reward"}, "trainer", 10);
        CHECK(got_trainer_again.empty());

        // A DIFFERENT task never consumed it -- still ready for "other_task".
        auto got_other = client->get("p1", {"reward"}, "other_task", 10);
        CHECK(got_other.size() == 1);
    }

    // ---- get() respects batch_size: partial availability and zero-ready ----
    {
        auto client = make_client();
        for (tq::SampleId id = 0; id < 3; ++id) {
            tq::Record r;
            r["reward"] = make_f32(static_cast<float>(id));
            client->put("p1", {"reward"}, {{id, r}});
        }
        // Only 3 ready, batch_size asks for 10 -- get exactly 3, not an
        // error, not padded with anything.
        auto got = client->get("p1", {"reward"}, "trainer", 10);
        CHECK(got.size() == 3);

        // Nothing left ready now -- batch_size > 0 with zero ready ids
        // returns empty, not an error.
        auto got_empty = client->get("p1", {"reward"}, "trainer", 10);
        CHECK(got_empty.empty());

        // batch_size=0 against ready data returns nothing (asked for
        // nothing, got nothing) without throwing.
        tq::Record r;
        r["reward"] = make_f32(99.0f);
        client->put("p1", {"reward"}, {{100, r}});
        auto got_zero = client->get("p1", {"reward"}, "trainer2", 0);
        CHECK(got_zero.empty());
    }

    // ---- Client genuinely delegates sampling policy: wired to a real
    //      GRPOGroupNSampler, not just FifoSampler, it must withhold a
    //      partial group and return the full group once complete. ----
    {
        auto grpo_sampler = std::make_shared<tq::GRPOGroupNSampler>(/*n_samples_per_prompt=*/2);
        auto client = make_client(grpo_sampler);

        tq::Record r0;
        r0["reward"] = make_f32(0.0f);
        client->put("grp", {"reward"}, {{0, r0}}, "group-x"); // only half of the group {0,1}

        auto got_partial = client->get("grp", {"reward"}, "trainer", 2);
        CHECK(got_partial.empty()); // GRPOGroupNSampler refuses a partial group

        tq::Record r1;
        r1["reward"] = make_f32(1.0f);
        client->put("grp", {"reward"}, {{1, r1}}, "group-x"); // completes {0,1}

        auto got_full = client->get("grp", {"reward"}, "trainer", 2);
        CHECK(got_full.size() == 2);
        CHECK(got_full.count(0) == 1 && got_full.count(1) == 1);
    }

    // ---- Multiple partitions through one Client stay independent, even
    //      with overlapping sample ids (same spirit as
    //      StorageManagerTest's partition-scoping test, exercised through
    //      the full Client API: Controller + StorageManager together). ----
    {
        auto client = make_client();
        tq::Record ra;
        ra["reward"] = make_f32(1.0f);
        tq::Record rb;
        rb["reward"] = make_f32(2.0f);
        client->put("partA", {"reward"}, {{5, ra}});
        client->put("partB", {"reward"}, {{5, rb}});

        auto got_a = client->get("partA", {"reward"}, "trainer", 10);
        auto got_b = client->get("partB", {"reward"}, "trainer", 10);
        CHECK(got_a.size() == 1 && static_cast<float*>(got_a.at(5).at("reward").data())[0] == 1.0f);
        CHECK(got_b.size() == 1 && static_cast<float*>(got_b.at(5).at("reward").data())[0] == 2.0f);
    }

    // ---- FIXED (was a real correctness bug -- see docs/UNIT_TEST_FINDINGS.md):
    //      Client::put() used to mark every field in `fields` as produced
    //      for `id` regardless of whether `data[id]` actually contained
    //      every one of those fields. It now only marks a field produced
    //      for a sample if that sample's own record actually has it, so a
    //      sample missing a "claimed" field is correctly NOT reported
    //      ready for it (and therefore not ready at all, since ready
    //      requires every queried field). ----
    {
        auto client = make_client();
        tq::Record partial;
        partial["a"] = make_f32(1.0f); // "b" deliberately omitted
        client->put("gap", {"a", "b"}, {{1, partial}});

        // Not ready for {"a","b"} -- "b" was never actually produced.
        auto got = client->get("gap", {"a", "b"}, "trainer", 10);
        CHECK(got.empty());

        // Still correctly ready for just {"a"}, which really was produced.
        auto got_a_only = client->get("gap", {"a"}, "trainer", 10);
        CHECK(got_a_only.size() == 1);
        CHECK(got_a_only.at(1).count("a") == 1);
    }

    // ---- Thread safety: multiple threads sharing ONE TransferQueueClient
    // instance. Client itself holds no state of its own beyond the three
    // shared_ptrs (Controller, StorageManager, sampler) -- Controller
    // serializes every call behind its own single mutex (see Controller.h),
    // and SimpleStorageManager has its own mutex too, so a single Client
    // shared across threads should be exactly as safe as each thread
    // holding its own Client pointed at the same collaborators. Each thread
    // uses its own partition to avoid legitimate (non-bug) contention over
    // which thread's put "wins" a shared batch. ----
    {
        auto client = make_client();
        constexpr int kThreads = 8;
        constexpr int kSamplesPerThread = 200;
        std::vector<std::thread> threads;
        std::vector<char> thread_ok(kThreads, 0);

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::string partition = "thread-client-" + std::to_string(t);
                bool ok = true;
                for (int i = 0; i < kSamplesPerThread; ++i) {
                    tq::Record r;
                    r["reward"] = make_f32(static_cast<float>(t));
                    try {
                        client->put(partition, {"reward"}, {{static_cast<tq::SampleId>(i), r}});
                    } catch (...) {
                        ok = false;
                    }
                }
                auto got = client->get(partition, {"reward"}, "trainer", kSamplesPerThread);
                ok = ok && (got.size() == static_cast<std::size_t>(kSamplesPerThread));
                for (const auto& [id, record] : got) {
                    (void)id;
                    ok = ok && (static_cast<const float*>(record.at("reward").data())[0] == static_cast<float>(t));
                }
                thread_ok[t] = ok ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) CHECK(thread_ok[t] == 1);
    }

    // ---- put() now groups samples by their exact produced-fields
    // signature before calling Controller::update_production_status, to
    // collapse N per-sample lock acquisitions into one call per group (see
    // the comment at its call site). The old per-sample loop never had to
    // get more than a single two-way split right (one sample missing one
    // field, covered above) -- this specifically exercises THREE distinct
    // signatures landing in the same put() call, to prove the new grouping
    // logic itself (not just the already-covered single-gap case) assigns
    // every sample to the right group and produces the exact same end state
    // the old unbatched per-sample calls would have. ----
    {
        auto client = make_client();
        tq::Record full;        // produces both "a" and "b"
        full["a"] = make_f32(1.0f);
        full["b"] = make_f32(2.0f);
        tq::Record a_only;      // produces only "a"
        a_only["a"] = make_f32(3.0f);
        tq::Record b_only;      // produces only "b"
        b_only["b"] = make_f32(4.0f);
        tq::Record neither;     // produces neither requested field
        neither["c"] = make_f32(5.0f);

        client->put("mixed_groups", {"a", "b"},
                    {{1, full}, {2, a_only}, {3, b_only}, {4, neither}});

        auto ready_both = client->get("mixed_groups", {"a", "b"}, "trainer", 10);
        CHECK(ready_both.size() == 1);
        CHECK(ready_both.count(1) == 1); // only sample 1 produced BOTH fields

        auto ready_a = client->get("mixed_groups", {"a"}, "trainer", 10);
        CHECK(ready_a.size() == 1);
        CHECK(ready_a.count(2) == 1); // sample 2 produced "a" (consumed above already excludes 1)

        auto ready_b = client->get("mixed_groups", {"b"}, "trainer", 10);
        CHECK(ready_b.size() == 1);
        CHECK(ready_b.count(3) == 1); // sample 3 produced "b"

        // Sample 4 (neither field) must never show up as ready for "a" or
        // "b" no matter which field is queried.
        auto ready_a_again = client->get("mixed_groups", {"a"}, "trainer", 10);
        CHECK(ready_a_again.count(4) == 0);
    }

    return tq::test::summary("ClientTest");
}
