// Unit tests for GroupRouter (include/transferqueue/GroupRouter.h,
// src/GroupRouter.cpp). Zero dependencies.
//
// GroupRouter is a pure function of (group_id, world_size): target_rank()
// is `std::hash<std::string>{}(group_id) % world_size`, with no seed, no
// mutable state, no I/O. Every test below is written against that actual
// implementation, not an assumed one.

#include "transferqueue/GroupRouter.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "TestUtils.h"

int main() {
    // Constructor validation: world_size must be > 0. The parameter type is
    // `int`, so both 0 and a genuinely negative value are reachable and
    // must both throw (not just "zero is special-cased").
    {
        CHECK_THROWS(tq::GroupRouter(0), std::invalid_argument);
        CHECK_THROWS(tq::GroupRouter(-1), std::invalid_argument);
        CHECK_THROWS(tq::GroupRouter(-1000), std::invalid_argument);
        CHECK_NOTHROW(tq::GroupRouter(1));
        CHECK_NOTHROW(tq::GroupRouter(1000));
    }

    // world_size = 1: every possible group_id must route to rank 0,
    // including the empty string (no special-casing of "no group" anywhere
    // in the implementation -- it's just another string to hash).
    {
        tq::GroupRouter router(1);
        CHECK(router.target_rank("") == 0);
        CHECK(router.target_rank("grpo-group-0") == 0);
        CHECK(router.target_rank("anything at all, really") == 0);
        CHECK(router.target_rank(std::string(500, 'x')) == 0);
    }

    // Determinism: the SAME instance called repeatedly with the SAME
    // group_id must always return the SAME rank -- checked many times, not
    // just twice, to rule out any accidental non-determinism (e.g. an
    // uninitialized read that happens to agree the first couple of calls).
    {
        tq::GroupRouter router(16);
        int first = router.target_rank("grpo-group-42");
        for (int i = 0; i < 200; ++i) {
            CHECK(router.target_rank("grpo-group-42") == first);
        }
    }

    // Cross-instance consistency: target_rank() is a pure function of
    // (group_id, world_size) -- std::hash<std::string> carries no
    // per-instance seed in any standard implementation, and GroupRouter
    // itself stores nothing but world_size_. So two SEPARATE instances
    // constructed with the same world_size must agree on every group_id,
    // within a single process run (hash stability ACROSS separate process
    // runs / builds is not guaranteed by the standard and isn't claimed or
    // relied on anywhere in this codebase -- routing only needs every rank
    // within the SAME running job to agree).
    {
        tq::GroupRouter router_a(8);
        tq::GroupRouter router_b(8);
        const std::vector<std::string> group_ids = {"", "a", "grpo-group-7", "prompt-12345", "x-y-z"};
        for (const auto& id : group_ids) {
            CHECK(router_a.target_rank(id) == router_b.target_rank(id));
        }
    }

    // Range check: every returned rank must land in [0, world_size), across
    // several world_size values and a large, varied set of group_id
    // strings (short, long, numeric-looking, special characters, empty).
    {
        std::vector<std::string> group_ids;
        group_ids.push_back("");
        for (int i = 0; i < 300; ++i) {
            group_ids.push_back("grpo-group-" + std::to_string(i));
        }
        group_ids.push_back(std::string(1000, 'a'));          // very long
        group_ids.push_back("123456789");                      // numeric-looking
        group_ids.push_back("!@#$%^&*()_+-=[]{}|;:,.<>?/~`");   // special characters
        group_ids.push_back(" ");                               // single space
        group_ids.push_back("\t\n");                            // whitespace/control chars

        for (int world_size : {1, 2, 7, 100}) {
            tq::GroupRouter router(world_size);
            for (const auto& id : group_ids) {
                int rank = router.target_rank(id);
                CHECK(rank >= 0);
                CHECK(rank < world_size);
            }
        }
    }

    // Distribution sanity (not a strict uniformity proof): with a
    // world_size > 1 and many distinct group_ids, more than one distinct
    // rank must actually be used -- catches a degenerate hash/modulo setup
    // that always collapses to rank 0 regardless of input.
    {
        tq::GroupRouter router(97); // prime world_size, incidental choice
        std::unordered_set<int> ranks_seen;
        for (int i = 0; i < 500; ++i) {
            ranks_seen.insert(router.target_rank("prompt-" + std::to_string(i)));
        }
        CHECK(ranks_seen.size() > 1);
        // With 500 distinct inputs over 97 buckets, a healthy hash should
        // realistically spread across a large fraction of the buckets, not
        // just a handful -- a looser sanity bound than "more than one", but
        // still not a strict uniformity requirement.
        CHECK(ranks_seen.size() > 10);
    }

    // Similar strings (one differs by a single character, or one is a
    // prefix of the other) are not required to land on different ranks,
    // but a naive/broken hash that makes ALL such pairs collide would be a
    // red flag. Checked as an aggregate "most pairs differ" sanity bound
    // rather than asserting any single pair individually (a real good hash
    // can still legitimately collide on any one specific pair by chance).
    {
        tq::GroupRouter router(1000); // large world_size to make incidental collisions rare
        int same_rank_count = 0;
        int total_pairs = 0;
        for (int i = 0; i < 100; ++i) {
            std::string base = "grpo-group-" + std::to_string(i);
            std::string one_char_diff = base + "x";          // base is a prefix of this one
            std::string also_diff = "x" + base;               // differs at the front instead
            ++total_pairs;
            if (router.target_rank(base) == router.target_rank(one_char_diff)) ++same_rank_count;
            ++total_pairs;
            if (router.target_rank(base) == router.target_rank(also_diff)) ++same_rank_count;
        }
        // With world_size=1000, a non-degenerate hash should agree on only
        // a small fraction of these 200 similar-but-distinct pairs purely
        // by chance (expected ~0.1%), so a generous upper bound here still
        // catches a hash that trivially ties rank to a shared prefix.
        CHECK(same_rank_count < total_pairs / 2);
    }

    return tq::test::summary("GroupRouterTest");
}
