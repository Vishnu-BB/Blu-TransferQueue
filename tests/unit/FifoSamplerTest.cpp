// Unit tests for BaseSampler/FifoSampler (include/transferqueue/Sampler.h,
// src/Sampler.cpp). Zero dependencies.

#include "transferqueue/Sampler.h"

#include <vector>

#include "TestUtils.h"

namespace {
// FifoSampler ignores group_ids entirely (no grouping) -- these tests only
// need a parallel vector of the right length to satisfy the interface.
std::vector<std::string> no_groups(std::size_t n) { return std::vector<std::string>(n, ""); }
} // namespace

int main() {
    // batch_size = 0 -> nothing selected, everything stays ready, in order.
    {
        tq::FifoSampler sampler;
        auto [selected, remaining] = sampler.sample({10, 11, 12}, no_groups(3), 0);
        CHECK(selected.empty());
        CHECK((remaining == std::vector<tq::SampleId>{10, 11, 12}));
    }

    // batch_size larger than ready_ids.size() -> selects everything
    // available, remaining empty, no out-of-bounds read (std::min clamps).
    {
        tq::FifoSampler sampler;
        auto [selected, remaining] = sampler.sample({10, 11, 12}, no_groups(3), 100);
        CHECK((selected == std::vector<tq::SampleId>{10, 11, 12}));
        CHECK(remaining.empty());
    }

    // batch_size exactly equal to ready_ids.size() -> selects everything,
    // remaining empty (the boundary between the two cases above).
    {
        tq::FifoSampler sampler;
        auto [selected, remaining] = sampler.sample({10, 11, 12}, no_groups(3), 3);
        CHECK((selected == std::vector<tq::SampleId>{10, 11, 12}));
        CHECK(remaining.empty());
    }

    // Empty ready_ids with a non-zero batch_size -> both empty, no crash
    // (begin()+0 on an empty vector must stay a valid, dereferenceable-free
    // iterator range).
    {
        tq::FifoSampler sampler;
        auto [selected, remaining] = sampler.sample({}, {}, 5);
        CHECK(selected.empty());
        CHECK(remaining.empty());
    }

    // Empty ready_ids with batch_size = 0 too -- the doubly-degenerate case.
    {
        tq::FifoSampler sampler;
        auto [selected, remaining] = sampler.sample({}, {}, 0);
        CHECK(selected.empty());
        CHECK(remaining.empty());
    }

    // Single-element input, both sides of the batch_size boundary.
    {
        tq::FifoSampler sampler;
        auto [sel0, rem0] = sampler.sample({42}, no_groups(1), 0);
        CHECK(sel0.empty());
        CHECK((rem0 == std::vector<tq::SampleId>{42}));

        auto [sel1, rem1] = sampler.sample({42}, no_groups(1), 1);
        CHECK((sel1 == std::vector<tq::SampleId>{42}));
        CHECK(rem1.empty());
    }

    // Order preservation: FIFO means the first batch_size elements *in the
    // order given*, not sorted -- ready_ids is intentionally not in
    // ascending order here to prove FifoSampler never reorders anything.
    {
        tq::FifoSampler sampler;
        std::vector<tq::SampleId> ready = {50, 7, 99, 1, 23};
        auto [selected, remaining] = sampler.sample(ready, no_groups(ready.size()), 2);
        CHECK((selected == std::vector<tq::SampleId>{50, 7}));       // first 2, as given, not sorted
        CHECK((remaining == std::vector<tq::SampleId>{99, 1, 23}));  // the rest, same relative order
    }

    // selected ++ remaining must reconstruct the original sequence exactly
    // (every id accounted for exactly once, nothing dropped or duplicated),
    // checked across a few different batch_size values on the same input.
    {
        std::vector<tq::SampleId> ready = {5, 1, 4, 2, 3};
        for (std::size_t batch_size : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 100u}) {
            tq::FifoSampler sampler;
            auto [selected, remaining] = sampler.sample(ready, no_groups(ready.size()), batch_size);
            std::vector<tq::SampleId> reconstructed = selected;
            reconstructed.insert(reconstructed.end(), remaining.begin(), remaining.end());
            CHECK(reconstructed == ready);
        }
    }

    // Sampler state (or lack thereof): FifoSampler holds no internal state,
    // so the same instance called twice in a row with different inputs must
    // not leak anything between calls.
    {
        tq::FifoSampler sampler;
        auto [sel1, rem1] = sampler.sample({1, 2, 3}, no_groups(3), 2);
        CHECK((sel1 == std::vector<tq::SampleId>{1, 2}));
        auto [sel2, rem2] = sampler.sample({9, 8}, no_groups(2), 1);
        CHECK((sel2 == std::vector<tq::SampleId>{9}));
        CHECK((rem2 == std::vector<tq::SampleId>{8}));
    }

    // Called through the BaseSampler interface (virtual dispatch), not just
    // the concrete type -- proves the override is wired correctly.
    {
        tq::BaseSampler* base = new tq::FifoSampler();
        auto [selected, remaining] = base->sample({1, 2, 3, 4}, no_groups(4), 2);
        CHECK((selected == std::vector<tq::SampleId>{1, 2}));
        CHECK((remaining == std::vector<tq::SampleId>{3, 4}));
        delete base;
    }

    return tq::test::summary("FifoSamplerTest");
}
