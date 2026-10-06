// Unit tests for PartitionIndexManager (include/transferqueue/PartitionIndexManager.h,
// src/PartitionIndexManager.cpp). Zero dependencies.
// Build: make build/unit_core_PartitionIndexManagerTest && ./build/unit_core_PartitionIndexManagerTest

#include "transferqueue/PartitionIndexManager.h"

#include <algorithm>
#include <atomic>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "TestUtils.h"

int main() {
    // allocate_indexes: count must be > 0.
    {
        tq::PartitionIndexManager mgr;
        CHECK_THROWS(mgr.allocate_indexes("p", 0), std::invalid_argument);
    }

    // count=1 and count>1 mint fresh, globally-increasing ids when the reuse
    // pool is empty.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 1);
        CHECK(a.size() == 1);
        auto b = mgr.allocate_indexes("p", 3);
        CHECK(b.size() == 3);
        // Fresh ids: no overlap with a, and all distinct from each other.
        CHECK(std::find(b.begin(), b.end(), a[0]) == b.end());
        CHECK(b[0] != b[1] && b[1] != b[2] && b[0] != b[2]);
    }

    // Reuse pool is preferred over minting: release then allocate the same
    // count back -> must get exactly the released ids back (not fresh ones).
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 3); // e.g. {0,1,2}
        mgr.release_indexes("p", {a[0], a[1]});
        auto reused = mgr.allocate_indexes("other", 2);
        std::set<tq::SampleId> reused_set(reused.begin(), reused.end());
        std::set<tq::SampleId> released_set{a[0], a[1]};
        CHECK(reused_set == released_set);
    }

    // FIFO order of the reuse pool specifically: release in a known order,
    // allocate one at a time, confirm they come back in release order (not
    // reverse/LIFO, not arbitrary).
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 4); // {0,1,2,3}
        mgr.release_indexes("p", {a[2]});      // release 2 first
        mgr.release_indexes("p", {a[0]});      // then release 0
        mgr.release_indexes("p", {a[3]});      // then release 3
        // Reuse pool is now [2, 0, 3] in release order -- FIFO means they
        // come back in exactly that order, one at a time.
        auto first = mgr.allocate_indexes("q", 1);
        CHECK(first[0] == a[2]);
        auto second = mgr.allocate_indexes("q", 1);
        CHECK(second[0] == a[0]);
        auto third = mgr.allocate_indexes("q", 1);
        CHECK(third[0] == a[3]);
    }

    // A single allocate_indexes() call can straddle the reuse pool and fresh
    // minting: ask for more than what's in the pool, verify the front of the
    // result is the reused ids (in FIFO order) and the rest are freshly
    // minted (not colliding with anything already owned).
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 2); // {0,1}
        mgr.release_indexes("p", {a[0], a[1]});
        auto mixed = mgr.allocate_indexes("q", 5); // 2 reused + 3 fresh
        CHECK(mixed.size() == 5);
        CHECK(mixed[0] == a[0]);
        CHECK(mixed[1] == a[1]);
        std::set<tq::SampleId> all(mixed.begin(), mixed.end());
        CHECK(all.size() == 5); // no duplicates among the 5
    }

    // Ids stay globally unique across interleaved allocations for different
    // partition_ids -- partitions don't share a separate id space.
    {
        tq::PartitionIndexManager mgr;
        auto p1 = mgr.allocate_indexes("p1", 3);
        auto p2 = mgr.allocate_indexes("p2", 3);
        auto p3 = mgr.allocate_indexes("p1", 2); // more for p1, interleaved
        std::set<tq::SampleId> all;
        for (auto v : {p1, p2, p3}) {
            for (auto id : v) {
                CHECK(all.insert(id).second); // true iff newly inserted -> no duplicate
            }
        }
        CHECK(all.size() == 8);
    }

    // release_indexes on an unknown partition_id: silently no-ops, does not
    // throw (verified actual behavior -- unlike an unowned id on a *known*
    // partition, which does throw; see below).
    {
        tq::PartitionIndexManager mgr;
        CHECK_NOTHROW(mgr.release_indexes("never-allocated", {999}));
    }

    // release_indexes with an id not owned by a *known* partition throws,
    // and does so without partially releasing the ids that *were* valid in
    // the same call (two-pass validate-then-erase: a thrown call must leave
    // every originally-owned id still owned).
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 2); // {0,1}
        tq::SampleId foreign = a[1] + 1000;     // definitely not owned by "p"
        CHECK_THROWS(mgr.release_indexes("p", {a[0], foreign}), std::invalid_argument);
        // a[0] must NOT have been released despite appearing before the bad
        // id in the list -- the partition still owns both of its real ids.
        auto owned = mgr.get_indexes_for_partition("p");
        std::set<tq::SampleId> owned_set(owned.begin(), owned.end());
        CHECK(owned_set.count(a[0]) == 1);
        CHECK(owned_set.count(a[1]) == 1);
    }

    // release_indexes releasing a *subset* of a partition's ids leaves the
    // rest owned and queryable.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 3); // {0,1,2}
        mgr.release_indexes("p", {a[1]});
        auto owned = mgr.get_indexes_for_partition("p");
        std::set<tq::SampleId> owned_set(owned.begin(), owned.end());
        CHECK(owned_set.size() == 2);
        CHECK(owned_set.count(a[0]) == 1);
        CHECK(owned_set.count(a[2]) == 1);
        CHECK(owned_set.count(a[1]) == 0);
    }

    // Releasing *every* id a partition owns via release_indexes (not
    // release_partition) removes the partition entirely -- a subsequent
    // get_indexes_for_partition must return empty, same as if it never
    // existed.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 2);
        mgr.release_indexes("p", {a[0], a[1]});
        CHECK(mgr.get_indexes_for_partition("p").empty());
    }

    // Double-release across two *separate* calls, where the partition still
    // owns OTHER ids after the first release: the second call throws, since
    // the partition map entry still exists but no longer owns that id.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 2); // {0,1} -- keep a[1] owned
        mgr.release_indexes("p", {a[0]});
        CHECK_THROWS(mgr.release_indexes("p", {a[0]}), std::invalid_argument);
    }

    // Double-release where the first release empties the partition
    // entirely: releasing its *last* id removes the partition_to_indexes_
    // entry altogether (verified above: get_indexes_for_partition then
    // reports empty, same as an unknown partition). So the second release
    // call hits the "unknown partition" branch -- a silent no-op, NOT a
    // throw. This differs from the case above only in whether any ids are
    // left owned after the first release; worth pinning down explicitly
    // since it's easy to assume release-of-a-stale-id always throws.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 1); // {0} -- only id "p" owns
        mgr.release_indexes("p", {a[0]});
        CHECK_NOTHROW(mgr.release_indexes("p", {a[0]}));
    }

    // FIXED (was a real double-allocation bug -- see docs/UNIT_TEST_FINDINGS.md):
    // release_indexes() now deduplicates indexes_to_release before
    // extending the reuse pool. A duplicate *within a single call* passes
    // the ownership pre-check (both copies reference the same still-owned
    // id) but must only be pushed into reusable_indexes_ once, so the same
    // global id can never be handed out twice by two different
    // allocate_indexes() calls.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 1); // {0}
        CHECK_NOTHROW(mgr.release_indexes("p", {a[0], a[0]})); // duplicate in one call
        auto reused = mgr.allocate_indexes("q", 2);
        // a[0] comes back exactly once; the second slot mints a genuinely
        // new id instead of re-handing out a[0] -- no double-allocation.
        CHECK(reused[0] == a[0]);
        CHECK(reused[1] != a[0]);
        CHECK(reused[1] >= 1); // fresh id, not a reuse-pool phantom duplicate
    }

    // release_indexes with an empty indexes_to_release on a known,
    // non-empty partition: no-op, no throw, ownership unaffected.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 2);
        CHECK_NOTHROW(mgr.release_indexes("p", {}));
        CHECK(mgr.get_indexes_for_partition("p").size() == 2);
        (void)a;
    }

    // release_partition: releases everything atomically, returns exactly
    // what was released, and the partition is gone afterward.
    {
        tq::PartitionIndexManager mgr;
        auto a = mgr.allocate_indexes("p", 3);
        auto released = mgr.release_partition("p");
        std::set<tq::SampleId> released_set(released.begin(), released.end());
        std::set<tq::SampleId> a_set(a.begin(), a.end());
        CHECK(released_set == a_set);
        CHECK(mgr.get_indexes_for_partition("p").empty());
    }

    // release_partition on a partition with zero allocated ids: can't
    // directly construct this (allocate always adds to the map), but an
    // already-released partition behaves identically -- releasing it again
    // returns empty, no throw.
    {
        tq::PartitionIndexManager mgr;
        mgr.allocate_indexes("p", 1);
        mgr.release_partition("p");
        std::vector<tq::SampleId> second;
        CHECK_NOTHROW(second = mgr.release_partition("p"));
        CHECK(second.empty());
    }

    // release_partition on a partition that was never allocated at all:
    // empty, no throw.
    {
        tq::PartitionIndexManager mgr;
        std::vector<tq::SampleId> r;
        CHECK_NOTHROW(r = mgr.release_partition("never-existed"));
        CHECK(r.empty());
    }

    // get_indexes_for_partition on an unknown partition_id: empty, not a
    // throw.
    {
        tq::PartitionIndexManager mgr;
        std::vector<tq::SampleId> r;
        CHECK_NOTHROW(r = mgr.get_indexes_for_partition("nope"));
        CHECK(r.empty());
    }

    // Reused ids returned by get_indexes_for_partition after allocating
    // more for the same partition: accumulates correctly (set union, no
    // duplicates even if the same id were somehow allocated twice to the
    // same partition across calls -- though that can't happen validly here).
    {
        tq::PartitionIndexManager mgr;
        mgr.allocate_indexes("p", 2);
        mgr.allocate_indexes("p", 3);
        CHECK(mgr.get_indexes_for_partition("p").size() == 5);
    }

    // Large-N: thousands of allocate/release cycles through the deque-backed
    // reuse pool -- the real motivation for vector->deque was avoiding
    // O(pool size) shifts on every front-erase; this at least proves
    // correctness holds at a size where a vector-based bug (e.g. an
    // off-by-one on a large front-range erase) would be more likely to
    // surface than at trivial sizes.
    {
        tq::PartitionIndexManager mgr;
        constexpr std::size_t kN = 20000;
        auto all = mgr.allocate_indexes("p", kN);
        CHECK(all.size() == kN);
        std::set<tq::SampleId> all_set(all.begin(), all.end());
        CHECK(all_set.size() == kN); // every id unique

        // Release the first half, reallocate that many -- must get exactly
        // the released half back (FIFO), not fresh ids.
        std::vector<tq::SampleId> first_half(all.begin(), all.begin() + kN / 2);
        mgr.release_indexes("p", first_half);
        auto reused = mgr.allocate_indexes("q", kN / 2);
        std::set<tq::SampleId> reused_set(reused.begin(), reused.end());
        std::set<tq::SampleId> first_half_set(first_half.begin(), first_half.end());
        CHECK(reused_set == first_half_set);

        // A further allocation beyond pool capacity mints fresh, still-
        // unique ids continuing past next_index_'s high-water mark.
        auto fresh = mgr.allocate_indexes("r", 10);
        for (auto id : fresh) {
            CHECK(all_set.count(id) == 0);
        }
    }

    // Thread safety: many threads concurrently allocate/release against
    // their own partition_id. No crash, no duplicate id ever observed live
    // across the whole run's union of final per-partition ownership once
    // threads stop releasing (allocated-but-not-released ids per thread
    // must be globally unique and must sum to the expected count).
    {
        tq::PartitionIndexManager mgr;
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 500;
        std::vector<std::thread> threads;
        std::vector<std::vector<tq::SampleId>> held(kThreads); // ids each thread ends up still holding
        std::atomic<bool> saw_problem{false};

        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t]() {
                std::string partition = "stress@" + std::to_string(t);
                std::vector<tq::SampleId> currently_held;
                for (int i = 0; i < kOpsPerThread; ++i) {
                    auto ids = mgr.allocate_indexes(partition, 1 + (i % 3));
                    currently_held.insert(currently_held.end(), ids.begin(), ids.end());
                    // Occasionally release some of what this thread holds --
                    // each thread only ever touches ids it allocated itself
                    // under its own partition_id, so this never races with
                    // another thread's ownership of the *same* id.
                    if (i % 4 == 0 && currently_held.size() > 2) {
                        tq::SampleId to_release = currently_held.back();
                        currently_held.pop_back();
                        try {
                            mgr.release_indexes(partition, {to_release});
                        } catch (...) {
                            saw_problem = true;
                        }
                    }
                }
                held[t] = currently_held;
            });
        }
        for (auto& th : threads) th.join();

        CHECK(!saw_problem);

        // Every id each thread believes it still holds must actually be
        // reported as owned by that thread's partition (no corruption from
        // concurrent access to the shared maps/deque).
        for (int t = 0; t < kThreads; ++t) {
            std::string partition = "stress@" + std::to_string(t);
            auto owned = mgr.get_indexes_for_partition(partition);
            std::set<tq::SampleId> owned_set(owned.begin(), owned.end());
            std::set<tq::SampleId> held_set(held[t].begin(), held[t].end());
            CHECK(owned_set == held_set);
        }

        // Global uniqueness: no two threads ever ended up both "holding"
        // the same live id (would indicate a race handing out a duplicate).
        std::set<tq::SampleId> global;
        bool all_unique = true;
        for (int t = 0; t < kThreads; ++t) {
            for (auto id : held[t]) {
                if (!global.insert(id).second) all_unique = false;
            }
        }
        CHECK(all_unique);
    }

    return tq::test::summary("PartitionIndexManagerTest");
}
