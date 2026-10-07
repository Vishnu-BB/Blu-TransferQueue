// Unit tests for StorageManager/SimpleStorageManager, to_field_dtype,
// serialize_batch/deserialize_batch, and make_batch_message/extract_batch
// (include/transferqueue/StorageManager.h, src/StorageManager.cpp).
// Needs Tensor-Implementations -- build with `make unit_tensor_StorageManagerTest`.

#include "transferqueue/StorageManager.h"

#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Message.h"
#include "TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

Tensor make_i64(const std::vector<std::int64_t>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

Tensor make_f32(const std::vector<float>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Float32);
    auto* data = static_cast<float*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

bool i64_eq(const Tensor& t, const std::vector<std::int64_t>& expected) {
    if (static_cast<std::size_t>(t.numel()) != expected.size()) return false;
    const auto* data = static_cast<const std::int64_t*>(t.data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) return false;
    }
    return true;
}

bool f32_eq(const Tensor& t, const std::vector<float>& expected) {
    if (static_cast<std::size_t>(t.numel()) != expected.size()) return false;
    const auto* data = static_cast<const float*>(t.data());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (data[i] != expected[i]) return false;
    }
    return true;
}

} // namespace

int main() {
    // ---- to_field_dtype: exhaustive, total, lossless, injective mapping ----
    {
        const OwnTensor::Dtype all_dtypes[] = {
            Dtype::Int8,       Dtype::Int16,  Dtype::Int32,      Dtype::Int64,       Dtype::UInt8,
            Dtype::UInt16,     Dtype::UInt32, Dtype::UInt64,     Dtype::Bfloat16,    Dtype::Float16,
            Dtype::Float32,    Dtype::Float64, Dtype::Bool,      Dtype::Complex32,   Dtype::Complex64,
            Dtype::Complex128, Dtype::Float4_e2m1, Dtype::Float4_e2m1_2x,
        };
        const tq::FieldDtype expected[] = {
            tq::FieldDtype::Int8,       tq::FieldDtype::Int16,  tq::FieldDtype::Int32,      tq::FieldDtype::Int64,
            tq::FieldDtype::UInt8,      tq::FieldDtype::UInt16, tq::FieldDtype::UInt32,     tq::FieldDtype::UInt64,
            tq::FieldDtype::Bfloat16,   tq::FieldDtype::Float16, tq::FieldDtype::Float32,   tq::FieldDtype::Float64,
            tq::FieldDtype::Bool,       tq::FieldDtype::Complex32, tq::FieldDtype::Complex64,
            tq::FieldDtype::Complex128, tq::FieldDtype::Float4_e2m1, tq::FieldDtype::Float4_e2m1_2x,
        };
        constexpr std::size_t kCount = sizeof(all_dtypes) / sizeof(all_dtypes[0]);
        static_assert(kCount == 18, "update this test if OwnTensor::Dtype gains/loses values");
        std::unordered_set<std::uint8_t> seen_targets;
        for (std::size_t i = 0; i < kCount; ++i) {
            CHECK(tq::to_field_dtype(all_dtypes[i]) == expected[i]);
            seen_targets.insert(static_cast<std::uint8_t>(expected[i]));
        }
        // Injective: 18 distinct inputs must produce 18 distinct outputs --
        // no two different OwnTensor::Dtype values silently collapse to the
        // same FieldDtype (would corrupt schema validation).
        CHECK(seen_targets.size() == kCount);
    }

    // ---- SimpleStorageManager: basic put/get round trip, exact values ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta meta({1}, {"p1"}, {"tokens"});
        tq::Record r;
        r["tokens"] = make_i64({10, 20, 30});
        mgr.put_data(meta, {{1, r}});

        auto got = mgr.get_data(meta);
        CHECK(got.size() == 1);
        CHECK(got.count(1) == 1);
        CHECK(i64_eq(got.at(1).at("tokens"), {10, 20, 30}));
    }

    // ---- Partition-scoping: same sample id, different partitions, must
    //      not collide (the real Phase-3 bug this field exists to fix) ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta meta_a({5}, {"a"}, {"x"});
        tq::BatchMeta meta_b({5}, {"b"}, {"x"});
        tq::Record ra;
        ra["x"] = make_f32({1.0f});
        tq::Record rb;
        rb["x"] = make_f32({2.0f});
        mgr.put_data(meta_a, {{5, ra}});
        mgr.put_data(meta_b, {{5, rb}});

        auto got_a = mgr.get_data(meta_a);
        auto got_b = mgr.get_data(meta_b);
        CHECK(got_a.size() == 1 && f32_eq(got_a.at(5).at("x"), {1.0f}));
        CHECK(got_b.size() == 1 && f32_eq(got_b.at(5).at("x"), {2.0f}));

        // Clearing partition "a"'s sample 5 must not touch partition "b"'s.
        mgr.clear_data(meta_a);
        CHECK(mgr.get_data(meta_a).empty());
        CHECK(mgr.get_data(meta_b).size() == 1);
    }

    // ---- BatchMeta.sample_ids/.partition_ids length mismatch must throw,
    //      independently for each of put_data/get_data/clear_data (each
    //      calls partition_lookup() itself) ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta bad;
        bad.sample_ids = {1, 2};
        bad.partition_ids = {"p"}; // mismatched on purpose, bypassing BatchMeta's own constructor check
        bad.fields = {"x"};
        tq::Record r;
        r["x"] = make_f32({1.0f});
        CHECK_THROWS(mgr.put_data(bad, {{1, r}}), std::invalid_argument);
        CHECK_THROWS(mgr.get_data(bad), std::invalid_argument);
        CHECK_THROWS(mgr.clear_data(bad), std::invalid_argument);
    }

    // ---- Overwrite accounting: current_bytes() reflects the byte DELTA,
    //      not a double-count or a leak, when a field is overwritten with a
    //      different-sized tensor ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta meta({1}, {"p"}, {"x"});
        tq::Record small;
        small["x"] = make_i64({1, 2}); // 16 bytes
        mgr.put_data(meta, {{1, small}});
        std::size_t after_small = mgr.current_bytes();
        CHECK(after_small == 16);

        tq::Record big;
        big["x"] = make_i64({1, 2, 3, 4, 5}); // 40 bytes
        mgr.put_data(meta, {{1, big}});       // overwrites the same (partition, id, field)
        CHECK(mgr.current_bytes() == 40);      // not 56 (double-counted), not 16 (old size leaked)

        tq::Record tiny;
        tiny["x"] = make_i64({9}); // 8 bytes
        mgr.put_data(meta, {{1, tiny}});
        CHECK(mgr.current_bytes() == 8);

        mgr.clear_data(meta);
        CHECK(mgr.current_bytes() == 0);
    }

    // ---- Runtime mutation of an existing field with a DIFFERENT DTYPE
    // (not just a different size/shape, which the test above already
    // covers): SimpleStorageManager::put_data() has no shape/dtype
    // consistency check at all -- `row[field] = value;` unconditionally
    // overwrites, regardless of what was there before. Confirmed real,
    // documented here rather than assumed. (The dtype-consistency
    // enforcement this project actually has -- Option B schema validation,
    // see docs/SCHEMA_VALIDATION.md -- lives one layer up, at
    // Controller::validate_schema()/DataPartitionStatus::validate_field(),
    // called by Client/Server *before* reaching this method; it is not
    // SimpleStorageManager's own job to re-check it, and nothing here
    // does.) ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta meta({1}, {"p"}, {"x"});

        tq::Record as_int;
        as_int["x"] = make_i64({42}); // Int64, 8 bytes
        mgr.put_data(meta, {{1, as_int}});
        CHECK(mgr.get_data(meta).at(1).at("x").dtype() == Dtype::Int64);

        tq::Record as_float;
        as_float["x"] = make_f32({1.0f, 2.0f, 3.0f}); // Float32, 12 bytes -- different dtype AND shape
        CHECK_NOTHROW(mgr.put_data(meta, {{1, as_float}})); // succeeds unconditionally, no validation
        auto& overwritten = mgr.get_data(meta).at(1).at("x");
        CHECK(overwritten.dtype() == Dtype::Float32); // the new dtype fully replaced the old
        CHECK(overwritten.numel() == 3);
        CHECK(mgr.current_bytes() == 12); // byte accounting still correct across the dtype change
    }

    // ---- get_data/clear_data for an id that was never written: must not
    //      crash, and must not fabricate an entry ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta meta({42}, {"p"}, {"x"});
        auto got = mgr.get_data(meta);
        CHECK(got.empty()); // silently absent, not an exception
        CHECK_NOTHROW(mgr.clear_data(meta)); // no-op, not an error

        // Same, but the partition itself has never been touched at all
        // (not just the id within an existing partition).
        tq::BatchMeta meta_unknown_partition({1}, {"never-seen"}, {"x"});
        CHECK(mgr.get_data(meta_unknown_partition).empty());
        CHECK_NOTHROW(mgr.clear_data(meta_unknown_partition));
    }

    // ---- FIXED (was a real asymmetry bug -- see docs/UNIT_TEST_FINDINGS.md):
    //      get_data() now filters by meta.fields, matching put_data's own
    //      filtering. A caller that only lists one field in `fields` gets
    //      back only that field, not every field ever written for that
    //      sample. An empty meta.fields (no filter requested) still
    //      returns the whole row -- verified separately below. ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta write_meta({1}, {"p"}, {"a", "b"});
        tq::Record r;
        r["a"] = make_f32({1.0f});
        r["b"] = make_f32({2.0f});
        mgr.put_data(write_meta, {{1, r}});

        tq::BatchMeta read_meta_a_only({1}, {"p"}, {"a"}); // only asks for "a"
        auto got = mgr.get_data(read_meta_a_only);
        CHECK(got.size() == 1);
        CHECK(got.at(1).count("a") == 1);
        CHECK(got.at(1).count("b") == 0); // "b" correctly filtered out now

        // Empty fields list -- no filter requested -- still returns the
        // whole row (the pre-fix default, preserved for this one case).
        tq::BatchMeta read_meta_no_filter({1}, {"p"}, {});
        auto got_unfiltered = mgr.get_data(read_meta_no_filter);
        CHECK(got_unfiltered.at(1).count("a") == 1);
        CHECK(got_unfiltered.at(1).count("b") == 1);
    }

    // ---- clear_data clears the WHOLE row regardless of meta.fields -- NOT
    //      a bug: every real caller (Server.cpp's CLEAR_DATA/CLEAR_PARTITION
    //      handlers, StorageServer.cpp's CLEAR_DATA handler,
    //      RpcClient::clear_data()'s very signature) never populates
    //      meta.fields for a clear call at all -- "delete this sample
    //      entirely" is the only semantics clear_data has ever been asked
    //      for. Confirmed deliberately left as-is. ----
    {
        tq::SimpleStorageManager mgr;
        tq::BatchMeta write_meta({1}, {"p"}, {"a", "b"});
        tq::Record r;
        r["a"] = make_f32({1.0f});
        r["b"] = make_f32({2.0f});
        mgr.put_data(write_meta, {{1, r}});
        CHECK(mgr.current_bytes() == 8);

        tq::BatchMeta clear_meta_a_only({1}, {"p"}, {"a"}); // only names "a"
        mgr.clear_data(clear_meta_a_only);
        CHECK(mgr.current_bytes() == 0); // "b" was cleared too, not left behind
        CHECK(mgr.get_data(write_meta).empty());
    }

    // ---- Backpressure: put_data blocks while the write wouldn't fit,
    //      wakes up once clear_data frees enough space. Uses a
    //      std::promise/future with a bounded wait_for so a real deadlock
    //      regression fails the test instead of hanging the whole binary;
    //      the manager/record are captured by value (shared_ptr/copy) into
    //      the thread so a timeout-path detach never leaves a dangling
    //      reference. ----
    {
        auto mgr = std::make_shared<tq::SimpleStorageManager>(/*capacity_bytes=*/8);
        tq::BatchMeta meta1({100}, {"p"}, {"x"});
        tq::Record r1;
        r1["x"] = make_i64({42}); // exactly 8 bytes -- fits, should not block

        std::promise<void> p1;
        auto f1 = p1.get_future();
        std::thread writer1([mgr, meta1, r1, p = std::move(p1)]() mutable {
            mgr->put_data(meta1, {{100, r1}});
            p.set_value();
        });
        CHECK(f1.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
        if (f1.wait_for(std::chrono::seconds(0)) == std::future_status::ready) writer1.join();
        else writer1.detach();
        CHECK(mgr->current_bytes() == 8);

        // Capacity is now full (8/8) -- a second 8-byte write must block.
        tq::BatchMeta meta2({101}, {"p"}, {"x"});
        tq::Record r2;
        r2["x"] = make_i64({99});
        std::promise<void> p2;
        auto f2 = p2.get_future();
        std::thread writer2([mgr, meta2, r2, p = std::move(p2)]() mutable {
            mgr->put_data(meta2, {{101, r2}});
            p.set_value();
        });
        CHECK(f2.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout); // still blocked

        // Free sample 100's 8 bytes -- should wake the blocked writer.
        mgr->clear_data(meta1);
        bool unblocked = f2.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        CHECK(unblocked);
        if (unblocked) {
            writer2.join();
            CHECK(mgr->current_bytes() == 8); // sample 101's bytes only
        } else {
            writer2.detach(); // avoid hanging the test binary on a real regression
        }
    }

    // ---- FIXED (was a real deadlock -- confirmed via external review,
    //      not assumed, see docs/UNIT_TEST_FINDINGS.md): a single batch
    //      whose own byte size exceeds total capacity_bytes used to block
    //      forever in put_data's wait() -- the predicate
    //      (current_bytes_ + incoming_bytes <= capacity_bytes_) can never
    //      be true, even at current_bytes_ == 0, so no clear_data() could
    //      ever wake it. Now rejected immediately instead. Bounded wait
    //      here too, defense in depth: a real regression fails this CHECK
    //      instead of hanging the whole binary. ----
    {
        auto mgr = std::make_shared<tq::SimpleStorageManager>(/*capacity_bytes=*/8);
        tq::BatchMeta meta({200}, {"p"}, {"x"});
        tq::Record r;
        r["x"] = make_i64({1, 2}); // 16 bytes -- exceeds the 8-byte total capacity on its own

        std::promise<void> p;
        auto f = p.get_future();
        std::thread writer([mgr, meta, r, p = std::move(p)]() mutable {
            try {
                mgr->put_data(meta, {{200, r}});
            } catch (...) {
                // swallow -- the test thread checks via the exception
                // re-thrown below, this thread only needs to prove it
                // returned promptly rather than hanging.
            }
            p.set_value();
        });
        bool returned_promptly = f.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        CHECK(returned_promptly); // must reject immediately, never hang
        if (returned_promptly) {
            writer.join();
        } else {
            writer.detach(); // avoid hanging the test binary on a real regression
        }
        CHECK(mgr->current_bytes() == 0); // the oversized write must not have partially landed

        // Confirmed on the calling thread too (no concurrency involved):
        // throws synchronously, not just "eventually returns" via some
        // other path.
        CHECK_THROWS(mgr->put_data(meta, {{200, r}}), std::invalid_argument);

        // A batch that exactly equals capacity must still succeed (the
        // fix's boundary: > capacity_bytes_ rejects, == does not).
        tq::BatchMeta meta_exact({201}, {"p"}, {"x"});
        tq::Record r_exact;
        r_exact["x"] = make_i64({1}); // 8 bytes -- exactly the capacity
        CHECK_NOTHROW(mgr->put_data(meta_exact, {{201, r_exact}}));
        CHECK(mgr->current_bytes() == 8);
    }

    // ---- capacity_bytes=0 is genuinely unbounded: a synchronous put of a
    //      nontrivial amount of data must never block (capacity_bytes_ > 0
    //      is false, so the wait predicate is never even evaluated). ----
    {
        tq::SimpleStorageManager mgr(/*capacity_bytes=*/0);
        tq::BatchMeta meta({1}, {"p"}, {"x"});
        std::vector<std::int64_t> big(100000, 7); // ~800KB, well past any small "capacity" that would matter
        tq::Record r;
        r["x"] = make_i64(big);
        CHECK_NOTHROW(mgr.put_data(meta, {{1, r}})); // must return promptly, not hang
        CHECK(mgr.current_bytes() == big.size() * sizeof(std::int64_t));
    }

    // ---- Concurrency stress: many threads, each its own partition,
    //      concurrent put/get/clear -- no crash, no cross-partition
    //      corruption, byte accounting self-consistent at the end. ----
    {
        tq::SimpleStorageManager mgr;
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 200;
        std::vector<char> thread_ok(kThreads, 0);
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::string partition = "stress@" + std::to_string(t);
                bool ok = true;
                for (int i = 0; i < kOpsPerThread; ++i) {
                    tq::SampleId id = static_cast<tq::SampleId>(i % 20); // deliberate id reuse within partition
                    tq::BatchMeta meta({id}, {partition}, {"v"});
                    tq::Record r;
                    float expected = static_cast<float>(t * 1000 + i);
                    r["v"] = make_f32({expected});
                    mgr.put_data(meta, {{id, r}});

                    auto got = mgr.get_data(meta);
                    if (got.size() != 1 || !f32_eq(got.at(id).at("v"), {expected})) {
                        ok = false;
                    }
                    if (i % 7 == 0) {
                        mgr.clear_data(meta);
                    }
                }
                thread_ok[t] = ok ? 1 : 0;
            });
        }
        for (auto& th : threads) th.join();
        for (int t = 0; t < kThreads; ++t) CHECK(thread_ok[t] == 1);
        // current_bytes() must be a sane non-negative-overflow value -- a
        // corrupted accounting bug would typically show up as a huge
        // (wrapped-around size_t) number here.
        CHECK(mgr.current_bytes() < 1000000);
    }

    // ---- serialize_batch/deserialize_batch: round trip survives mixed
    //      dtypes within one record, multiple samples each with multiple
    //      fields, and an empty batch. Self-describing encoding -- must not
    //      depend on map iteration order matching anything external. ----
    {
        tq::Record r1;
        r1["tokens"] = make_i64({1, 2, 3});
        r1["reward"] = make_f32({0.5f});
        tq::Record r2;
        r2["tokens"] = make_i64({4, 5});
        r2["reward"] = make_f32({-1.25f});
        std::unordered_map<tq::SampleId, tq::Record> data = {{7, r1}, {3, r2}}; // ids deliberately out of order

        auto bytes = tq::serialize_batch(data);
        auto decoded = tq::deserialize_batch(bytes);
        CHECK(decoded.size() == 2);
        CHECK(i64_eq(decoded.at(7).at("tokens"), {1, 2, 3}));
        CHECK(f32_eq(decoded.at(7).at("reward"), {0.5f}));
        CHECK(i64_eq(decoded.at(3).at("tokens"), {4, 5}));
        CHECK(f32_eq(decoded.at(3).at("reward"), {-1.25f}));
    }
    {
        // Empty batch round-trips to an empty map, not a crash/throw.
        std::unordered_map<tq::SampleId, tq::Record> empty_data;
        auto bytes = tq::serialize_batch(empty_data);
        auto decoded = tq::deserialize_batch(bytes);
        CHECK(decoded.empty());
    }
    {
        // Truncated buffer must throw, not read out of bounds / crash.
        tq::Record r;
        r["x"] = make_f32({1.0f});
        auto bytes = tq::serialize_batch({{1, r}});
        CHECK(bytes.size() > 4);
        std::vector<std::uint8_t> truncated(bytes.begin(), bytes.begin() + 4);
        CHECK_THROWS(tq::deserialize_batch(truncated), std::invalid_argument);
        CHECK_THROWS(tq::deserialize_batch({}), std::invalid_argument);
    }

    // ---- make_batch_message/extract_batch: survives an actual wire
    //      round trip (Message::serialize()/deserialize()), not just an
    //      in-memory construct-then-extract. ----
    {
        tq::Record r;
        r["tokens"] = make_i64({11, 12, 13});
        std::unordered_map<tq::SampleId, tq::Record> data = {{9, r}};

        tq::MessageBody body;
        body.partition_id = "rollout@wire";
        auto msg = tq::make_batch_message(tq::RequestType::GET_META_RESPONSE, "server-1", body, data);
        auto wire_bytes = msg.serialize();
        auto decoded_msg = tq::Message::deserialize(wire_bytes);
        CHECK(decoded_msg.request_type == tq::RequestType::GET_META_RESPONSE);
        CHECK(decoded_msg.body.partition_id == "rollout@wire");

        auto extracted = tq::extract_batch(decoded_msg);
        CHECK(extracted.size() == 1);
        CHECK(i64_eq(extracted.at(9).at("tokens"), {11, 12, 13}));
    }

    return tq::test::summary("StorageManagerTest");
}
