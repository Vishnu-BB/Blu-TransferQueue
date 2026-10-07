// Unit tests for BatchMeta (include/transferqueue/BatchMeta.h,
// src/BatchMeta.cpp). Zero dependencies -- build with `make unit_tests` or
// directly: g++ -std=c++2a -Iinclude src/BatchMeta.cpp tests/unit/BatchMetaTest.cpp -o /tmp/t && /tmp/t

#include "transferqueue/BatchMeta.h"

#include <stdexcept>
#include <vector>

#include "TestUtils.h"

int main() {
    // Default construction: empty, never ready.
    {
        tq::BatchMeta m;
        CHECK(m.sample_ids.empty());
        CHECK(m.partition_ids.empty());
        CHECK(m.fields.empty());
        CHECK(m.production_status.empty());
        CHECK(!m.is_ready());
    }

    // production_status omitted entirely -> defaults to all-false, matching
    // upstream's "default zeros" behavior, not left empty/mismatched.
    {
        tq::BatchMeta m({1, 2, 3}, {"p", "p", "p"});
        CHECK(m.production_status.size() == 3);
        for (bool produced : m.production_status) CHECK(!produced);
        CHECK(!m.is_ready());
    }

    // All produced -> ready.
    {
        tq::BatchMeta m({1, 2, 3}, {"p", "p", "p"}, {}, {true, true, true});
        CHECK(m.is_ready());
    }

    // Any single not-produced sample makes the whole batch not ready,
    // regardless of position (first, middle, last).
    {
        CHECK(!tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {}, {false, true, true}).is_ready());
        CHECK(!tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {}, {true, false, true}).is_ready());
        CHECK(!tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {}, {true, true, false}).is_ready());
    }

    // Single-sample batch, both states.
    {
        CHECK(tq::BatchMeta({1}, {"p"}, {}, {true}).is_ready());
        CHECK(!tq::BatchMeta({1}, {"p"}, {}, {false}).is_ready());
    }

    // Empty sample_ids is never ready even with a (vacuously matching) empty
    // production_status -- is_ready() explicitly special-cases this rather
    // than falling through an empty for-loop to `true`.
    {
        tq::BatchMeta m({}, {}, {}, {});
        CHECK(!m.is_ready());
    }

    // A batch can span partitions -- one partition_id per sample, not one
    // for the whole batch -- and is_ready() doesn't care about partition
    // boundaries at all.
    {
        tq::BatchMeta m({1, 2}, {"rollout@0", "rollout@1"}, {}, {true, true});
        CHECK(m.is_ready());
        CHECK(m.partition_ids[0] == "rollout@0");
        CHECK(m.partition_ids[1] == "rollout@1");
    }

    // fields has no length relationship to sample_ids -- it's a separate
    // "which fields does this view cover" list, not per-sample. Must not
    // throw regardless of its size relative to sample_ids.
    {
        CHECK_NOTHROW(tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {"tokens", "reward"}, {}));
        CHECK_NOTHROW(tq::BatchMeta({1}, {"p"}, {}, {}));
        CHECK_NOTHROW(tq::BatchMeta({1}, {"p"}, {"a", "b", "c", "d"}, {}));
    }

    // partition_ids length mismatch (too short, too long, and empty when
    // sample_ids is not) must throw -- the constructor's one hard
    // invariant.
    {
        CHECK_THROWS(tq::BatchMeta({1, 2, 3}, {"p", "p"}), std::invalid_argument);       // too short
        CHECK_THROWS(tq::BatchMeta({1, 2}, {"p", "p", "p"}), std::invalid_argument);     // too long
        CHECK_THROWS(tq::BatchMeta({1, 2, 3}, {}), std::invalid_argument);               // empty vs non-empty
        CHECK_THROWS(tq::BatchMeta({}, {"p"}), std::invalid_argument);                   // non-empty vs empty
    }

    // production_status, when explicitly given (non-empty), must match
    // sample_ids length exactly -- an empty one is the "omitted, default to
    // all-false" case (already covered above), not a mismatch.
    {
        CHECK_THROWS(tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {}, {true, false}), std::invalid_argument);
        CHECK_THROWS(tq::BatchMeta({1, 2, 3}, {"p", "p", "p"}, {}, {true, false, true, false}),
                     std::invalid_argument);
    }

    // A failed partition_ids check must throw before production_status is
    // ever validated -- both mismatched at once should still throw exactly
    // once (not, say, a different/wrong exception from a later check), and
    // the object must not be left partially constructed in a way a caller
    // could observe (construction either fully succeeds or throws, there's
    // no partial BatchMeta to inspect after a thrown exception).
    {
        CHECK_THROWS(tq::BatchMeta({1, 2, 3}, {"p"}, {}, {true}), std::invalid_argument);
    }

    // Large batch: no off-by-one at a size that would expose an
    // index/iterator bug (e.g. production_status.size() - 1 underflowing,
    // or a loop bound mismatch) -- exercise is_ready() scanning all of a
    // non-trivial vector, with the single unproduced sample at the very
    // last index specifically.
    {
        constexpr std::size_t kN = 10000;
        std::vector<tq::SampleId> ids(kN);
        std::vector<std::string> partitions(kN, "p");
        std::vector<bool> status(kN, true);
        for (std::size_t i = 0; i < kN; ++i) ids[i] = i;
        status[kN - 1] = false;
        tq::BatchMeta m(ids, partitions, {}, status);
        CHECK(!m.is_ready());
        status[kN - 1] = true;
        tq::BatchMeta m2(ids, partitions, {}, status);
        CHECK(m2.is_ready());
    }

    // ---- Duplicate sample_ids within a single batch: NOT rejected by the
    // constructor (it only checks array LENGTHS match, never uniqueness of
    // VALUES within sample_ids). Documents the actual, current behavior
    // rather than assuming a uniqueness invariant that doesn't exist here
    // -- BatchMeta is a plain addressing struct, and nothing downstream
    // that was checked (StorageManager's map-keyed storage) is corrupted
    // by a duplicate, just redundant. Production_status can legitimately
    // disagree between the two listed occurrences of the same id (array
    // position, not id identity, is what production_status is indexed
    // by) -- that asymmetry is a real, confirmed consequence worth
    // pinning down explicitly. ----
    {
        // Same id (7) listed twice, with DIFFERENT production_status
        // entries at each position -- constructor accepts this silently.
        CHECK_NOTHROW(tq::BatchMeta({7, 7}, {"p", "p"}, {}, {true, false}));
        tq::BatchMeta dup({7, 7}, {"p", "p"}, {}, {true, false});
        CHECK(dup.sample_ids.size() == 2);
        CHECK(dup.sample_ids[0] == 7 && dup.sample_ids[1] == 7);
        // is_ready() scans production_status positionally, not by unique
        // id -- the second (false) entry makes the whole batch not ready,
        // even though "sample 7" also appears as "produced" at position 0.
        CHECK(!dup.is_ready());

        // All duplicates, all produced -> ready (positional scan, every
        // position is true regardless of id repetition).
        tq::BatchMeta dup_ready({3, 3, 3}, {"p", "p", "p"}, {}, {true, true, true});
        CHECK(dup_ready.is_ready());
    }

    // ---- Empty field name strings in `fields`: NOT rejected. `fields` has
    // no length relationship to sample_ids (confirmed by an earlier test in
    // this file) and no per-element validation at all -- an empty string is
    // just another string as far as the constructor is concerned. ----
    {
        CHECK_NOTHROW(tq::BatchMeta({1}, {"p"}, {""}, {}));
        tq::BatchMeta m({1}, {"p"}, {"", "reward", ""}, {});
        CHECK(m.fields.size() == 3);
        CHECK(m.fields[0].empty());
        CHECK(m.fields[1] == "reward");
        CHECK(m.fields[2].empty());
        // Doesn't affect is_ready() at all -- `fields` plays no role in
        // readiness, only `production_status` does.
        tq::BatchMeta ready_with_empty_field({1}, {"p"}, {""}, {true});
        CHECK(ready_with_empty_field.is_ready());
    }

    return tq::test::summary("BatchMetaTest");
}
