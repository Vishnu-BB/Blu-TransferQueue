// Unit tests for GRPOGroupNSampler (include/transferqueue/GRPOGroupNSampler.h,
// src/GRPOGroupNSampler.cpp). Zero dependencies.
//
// Rewritten after the sampler's algorithm changed from numeric-id-adjacency
// grouping to real group_id-based grouping (see docs/GRPO_GROUP_N_SAMPLER.md
// -- an external review correctly identified a silent cross-group
// contamination risk in the adjacency-based approach). Every test here
// exercises the (ready_ids, group_ids, batch_size) signature.

#include "transferqueue/GRPOGroupNSampler.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "TestUtils.h"

int main() {
    // Constructor validation: 0 throws, 1 and large positive values succeed.
    {
        CHECK_THROWS(tq::GRPOGroupNSampler(0), std::invalid_argument);
        CHECK_NOTHROW(tq::GRPOGroupNSampler(1));
        CHECK_NOTHROW(tq::GRPOGroupNSampler(1000));
    }

    // batch_size not a multiple of n_samples_per_prompt throws, checked
    // against several different non-multiple values, not just one.
    {
        tq::GRPOGroupNSampler sampler(4);
        std::vector<std::string> g = {"A", "A", "A", "A"};
        CHECK_THROWS(sampler.sample({0, 1, 2, 3}, g, 1), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1, 2, 3}, g, 2), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1, 2, 3}, g, 3), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1, 2, 3}, g, 5), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1, 2, 3}, g, 7), std::invalid_argument);
        CHECK_NOTHROW(sampler.sample({0, 1, 2, 3}, g, 0));
        CHECK_NOTHROW(sampler.sample({0, 1, 2, 3}, g, 4));
        CHECK_NOTHROW(sampler.sample({0, 1, 2, 3}, g, 8));
    }

    // ready_ids and group_ids must be the same length.
    {
        tq::GRPOGroupNSampler sampler(2);
        CHECK_THROWS(sampler.sample({0, 1, 2}, {"A", "A"}, 2), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1}, {"A", "A", "B"}, 2), std::invalid_argument);
        CHECK_THROWS(sampler.sample({0, 1}, {}, 2), std::invalid_argument);
    }

    // batch_size = 0 -> required_groups = 0, trivially satisfied: must
    // return promptly with nothing selected and every ready id (sorted)
    // intact as still_ready, regardless of grouping.
    {
        tq::GRPOGroupNSampler sampler(4);
        auto [selected, still_ready] =
            sampler.sample({0, 1, 2, 3, 7, 8}, {"A", "A", "A", "A", "B", "B"}, 0);
        CHECK(selected.empty());
        CHECK((still_ready == std::vector<tq::SampleId>{0, 1, 2, 3, 7, 8}));
    }

    // Empty ready_ids with a non-zero batch_size -> insufficient groups ->
    // ({}, {}).
    {
        tq::GRPOGroupNSampler sampler(4);
        auto [selected, still_ready] = sampler.sample({}, {}, 4);
        CHECK(selected.empty());
        CHECK(still_ready.empty());
    }

    // THE FIX, verified directly: real group_id grouping survives
    // numerically interleaved ids that an adjacency-based scan would have
    // fused into one fake group. Worker A (prompt "p1") produces {100,102};
    // worker B (prompt "p2") produces {101,103}.
    {
        tq::GRPOGroupNSampler sampler(2);
        auto [selected, still_ready] =
            sampler.sample({100, 102, 101, 103}, {"p1", "p1", "p2", "p2"}, 4);
        // Both real groups selected, correctly kept separate -- not fused
        // into one group of 4 the way id-adjacency would have.
        CHECK((selected == std::vector<tq::SampleId>{100, 101, 102, 103}));
        CHECK(still_ready.empty());

        // Asking for only 1 group: groups are visited in sorted-group_id
        // order ("p1" < "p2"), so {100,102} ("p1") is the one taken, not
        // {101,103}.
        tq::GRPOGroupNSampler sampler1(2);
        auto [sel1, rem1] = sampler1.sample({100, 102, 101, 103}, {"p1", "p1", "p2", "p2"}, 2);
        CHECK((sel1 == std::vector<tq::SampleId>{100, 102}));
        CHECK((rem1 == std::vector<tq::SampleId>{101, 103}));
    }

    // A sample with NO recorded group_id ("") can never be selected, even
    // when it sits numerically right next to a real, complete group --
    // this is the "fail-safe, not fail-dangerous" guarantee the fix is
    // built on: refusing to guess beats guessing wrong.
    {
        tq::GRPOGroupNSampler sampler(2);
        auto [selected, still_ready] = sampler.sample({0, 1, 2}, {"A", "A", ""}, 2);
        CHECK((selected == std::vector<tq::SampleId>{0, 1}));
        CHECK((still_ready == std::vector<tq::SampleId>{2})); // id 2 stranded regardless of adjacency to {0,1}
    }

    // More complete groups available than required_groups asks for: must
    // stop as soon as enough are found (in sorted-group_id order), not
    // return every complete group it could find.
    {
        tq::GRPOGroupNSampler sampler(2);
        auto [selected, still_ready] =
            sampler.sample({0, 1, 2, 3, 4, 5}, {"A", "A", "B", "B", "C", "C"}, 2);
        CHECK((selected == std::vector<tq::SampleId>{0, 1})); // "A" visited first alphabetically
        CHECK((still_ready == std::vector<tq::SampleId>{2, 3, 4, 5}));
    }

    // An incomplete group (fewer than n_samples_per_prompt members) is
    // skipped entirely, even if a later, complete group exists -- the
    // incomplete one's members stay in still_ready, untouched.
    {
        tq::GRPOGroupNSampler sampler(3);
        auto [selected, still_ready] =
            sampler.sample({10, 11, 20, 21, 22}, {"A", "A", "B", "B", "B"}, 3);
        CHECK((selected == std::vector<tq::SampleId>{20, 21, 22})); // "A" (size 2) skipped, "B" (size 3) taken
        CHECK((still_ready == std::vector<tq::SampleId>{10, 11}));
    }

    // A group with MORE than n_samples_per_prompt ready members (a
    // real-world possibility if a caller over-produces for one group):
    // selects exactly n of them (the smallest ids), leaves the rest ready.
    {
        tq::GRPOGroupNSampler sampler(2);
        auto [selected, still_ready] = sampler.sample({5, 6, 7}, {"A", "A", "A"}, 2);
        CHECK((selected == std::vector<tq::SampleId>{5, 6}));
        CHECK((still_ready == std::vector<tq::SampleId>{7}));
    }

    // n_samples_per_prompt = 1 degenerate case: every id with a real
    // group_id is trivially its own complete "group". Input given unsorted
    // on purpose; an id with no group_id is never selectable even at n=1.
    {
        tq::GRPOGroupNSampler sampler(1);
        auto [selected, still_ready] =
            sampler.sample({50, 7, 99, 1, 23}, {"g50", "g7", "", "g1", "g23"}, 3);
        // Groups visited in sorted-group_id order: "g1"(id1), "g23"(id23),
        // "g50"(id50) -- the first 3 alphabetically, then the whole
        // selected vector is sorted by id per the sampler's own contract.
        CHECK((selected == std::vector<tq::SampleId>{1, 23, 50}));
        // 7 (group "g7", not reached) and 99 (no group_id at all, never
        // selectable) both remain, sorted.
        CHECK((still_ready == std::vector<tq::SampleId>{7, 99}));
    }

    // n_samples_per_prompt = 1 with batch_size = 0: zero groups required,
    // nothing selected regardless of how much is ready.
    {
        tq::GRPOGroupNSampler sampler(1);
        auto [selected, still_ready] = sampler.sample({5, 2, 8}, {"g5", "g2", "g8"}, 0);
        CHECK(selected.empty());
        CHECK((still_ready == std::vector<tq::SampleId>{2, 5, 8})); // sorted
    }

    // Large n_samples_per_prompt (50) with exactly one real complete group
    // (all sharing group_id "big") plus plenty of ungrouped noise ids that
    // happen to sit numerically close to it but can never be swept in.
    {
        constexpr std::size_t kN = 50;
        std::vector<tq::SampleId> ready;
        std::vector<std::string> groups;
        for (tq::SampleId id : {1000, 1002, 1005, 1009, 1020}) {
            ready.push_back(id);
            groups.push_back(""); // no group_id -- noise, never selectable
        }
        for (tq::SampleId id = 100; id < 100 + kN; ++id) {
            ready.push_back(id);
            groups.push_back("big"); // the one real group
        }
        for (tq::SampleId id : {5000, 5001, 5003}) {
            ready.push_back(id);
            groups.push_back("");
        }

        tq::GRPOGroupNSampler sampler(kN);
        auto [selected, still_ready] = sampler.sample(ready, groups, kN);
        CHECK(selected.size() == kN);
        for (std::size_t k = 0; k < kN; ++k) {
            CHECK(selected[k] == 100 + static_cast<tq::SampleId>(k));
        }
        CHECK(still_ready.size() == ready.size() - kN);
        CHECK(std::find(selected.begin(), selected.end(), 1000) == selected.end());
        CHECK(std::find(selected.begin(), selected.end(), 5000) == selected.end());

        // Asking for a second group of 50 that doesn't exist -> insufficient,
        // returns nothing and leaves every id present, sorted.
        tq::GRPOGroupNSampler sampler2(kN);
        auto [sel2, rem2] = sampler2.sample(ready, groups, kN * 2);
        CHECK(sel2.empty());
        std::vector<tq::SampleId> sorted_ready = ready;
        std::sort(sorted_ready.begin(), sorted_ready.end());
        CHECK(rem2 == sorted_ready);
    }

    // Duplicate SampleId within ready_ids, same group_id both times (a
    // real caller-side bug -- sample ids should be unique -- not something
    // the sampler is contracted to handle meaningfully). Document the
    // actual behavior rather than assert a "correct" semantics that
    // doesn't exist for malformed input: both copies land in the same
    // bucket, so a bucket can appear to have more members than truly
    // distinct ids exist.
    {
        tq::GRPOGroupNSampler sampler(2);
        auto [selected, still_ready] = sampler.sample({0, 0, 1}, {"A", "A", "A"}, 2);
        // Bucket "A" = [0,0,1] (3 entries, only 2 distinct values) -- the
        // sampler selects the first 2 by its internal order, which here is
        // the two "0" entries (both sort to the front).
        CHECK(selected.size() == 2);
        CHECK(selected[0] == 0);
        CHECK(selected[1] == 0);
        CHECK((still_ready == std::vector<tq::SampleId>{1}));
    }

    // Unsorted input generally: the algorithm must sort internally rather
    // than assume the caller already did, proven with a non-trivial
    // shuffle of a real complete group plus ungrouped noise.
    {
        tq::GRPOGroupNSampler sampler(3);
        auto [selected, still_ready] =
            sampler.sample({9, 2, 0, 1, 8}, {"", "X", "X", "X", ""}, 3);
        CHECK((selected == std::vector<tq::SampleId>{0, 1, 2}));
        CHECK((still_ready == std::vector<tq::SampleId>{8, 9})); // both ungrouped, sorted
    }

    // ---- find_stranded(): track-and-expose age-based reporting. Fixed
    // now_ms/produced_at_ms values throughout -- no wall-clock dependency,
    // no sleep(), fully deterministic. ----

    // Length mismatches (any of the three arrays) must throw, same
    // discipline as sample().
    {
        tq::GRPOGroupNSampler sampler(2);
        CHECK_THROWS(sampler.find_stranded({0, 1}, {"A"}, {0, 0}, 100, 0), std::invalid_argument);
        CHECK_THROWS(sampler.find_stranded({0, 1}, {"A", "A"}, {0}, 100, 0), std::invalid_argument);
    }

    // Empty input -> empty report, no crash.
    {
        tq::GRPOGroupNSampler sampler(2);
        CHECK(sampler.find_stranded({}, {}, {}, 1000, 0).empty());
    }

    // Core scenario: an incomplete group old enough -> reported; a
    // COMPLETE group, even though it's even older, is NEVER reported
    // (completeness always trumps age); a young incomplete group isn't
    // reported yet; the "" (ungrouped) bucket is always eligible
    // regardless of its member count.
    {
        tq::GRPOGroupNSampler sampler(3); // n_samples_per_prompt = 3
        std::vector<tq::SampleId> ready_ids = {10, 11, 20, 21, 22, 30, 40, 41};
        std::vector<std::string> group_ids = {"A", "A", "B", "B", "B", "", "C", "C"};
        std::vector<std::int64_t> produced_at_ms = {100, 200, 500, 600, 700, 50, 990, 995};
        std::int64_t now_ms = 1000;

        // "A": incomplete (2 < 3), oldest=100, age=900.
        // "B": COMPLETE (3 == 3), oldest=500, age=500 -- must never appear.
        // "" : always eligible, oldest=50, age=950.
        // "C": incomplete (2 < 3), oldest=990, age=10 -- too young for max_age_ms=500.
        auto stranded = sampler.find_stranded(ready_ids, group_ids, produced_at_ms, now_ms, /*max_age_ms=*/500);
        CHECK(stranded.size() == 2);
        bool found_a = false, found_ungrouped = false;
        for (const auto& group : stranded) {
            CHECK(group.group_id != "B"); // complete group must never be reported, at any age
            CHECK(group.group_id != "C"); // too young
            if (group.group_id == "A") {
                found_a = true;
                CHECK((group.sample_ids == std::vector<tq::SampleId>{10, 11}));
                CHECK(group.oldest_age_ms == 900);
            } else if (group.group_id.empty()) {
                found_ungrouped = true;
                CHECK((group.sample_ids == std::vector<tq::SampleId>{30}));
                CHECK(group.oldest_age_ms == 950);
            }
        }
        CHECK(found_a && found_ungrouped);

        // Lower the threshold so "C" also qualifies now (age 10 >= 5).
        auto stranded_lower_bar =
            sampler.find_stranded(ready_ids, group_ids, produced_at_ms, now_ms, /*max_age_ms=*/5);
        CHECK(stranded_lower_bar.size() == 3); // A, "", and now C too -- never B
        bool found_c = false;
        for (const auto& group : stranded_lower_bar) {
            CHECK(group.group_id != "B");
            if (group.group_id == "C") {
                found_c = true;
                CHECK((group.sample_ids == std::vector<tq::SampleId>{40, 41}));
                CHECK(group.oldest_age_ms == 10);
            }
        }
        CHECK(found_c);

        // A threshold nothing clears -> empty report.
        CHECK(sampler.find_stranded(ready_ids, group_ids, produced_at_ms, now_ms, /*max_age_ms=*/100000).empty());
    }

    // Unknown produced_at (-1 sentinel): excluded from the oldest-member
    // search rather than treated as infinitely old; a bucket where EVERY
    // member has -1 can't be judged and is never reported. n=3 with a
    // 2-member bucket so it's genuinely incomplete (a 2-member bucket at
    // n=2 would be COMPLETE and correctly skipped entirely, regardless of
    // timestamps -- not what this case is testing).
    {
        tq::GRPOGroupNSampler sampler(3);
        // Bucket "A": one real timestamp (200) and one unknown (-1) --
        // oldest must be 200 (the -1 is ignored), not -1 itself.
        auto stranded = sampler.find_stranded({0, 1}, {"A", "A"}, {200, -1}, /*now_ms=*/1000, /*max_age_ms=*/0);
        CHECK(stranded.size() == 1);
        if (stranded.size() == 1) {
            CHECK(stranded[0].oldest_age_ms == 800); // 1000 - 200
        }

        // Bucket "B": both unknown -- can't judge age, never reported even
        // with max_age_ms=0.
        auto stranded_all_unknown =
            sampler.find_stranded({2, 3}, {"B", "B"}, {-1, -1}, /*now_ms=*/1000, /*max_age_ms=*/0);
        CHECK(stranded_all_unknown.empty());
    }

    return tq::test::summary("GRPOGroupNSamplerTest");
}
