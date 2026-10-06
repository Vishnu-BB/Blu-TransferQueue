#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

// One group BaseSampler::find_stranded() reports: real members found so
// far, and how long the oldest of them has been waiting. Track-and-expose
// only (see BaseSampler::find_stranded's doc) -- this is a report, not an
// action; nothing is cleared or modified by producing one.
struct StrandedGroup {
    std::string group_id;
    std::vector<SampleId> sample_ids; // every known member, sorted
    std::int64_t oldest_age_ms;
};

// Pluggable consumption-order logic. Mirrors upstream BaseSampler: given the
// currently-ready sample ids, pick which ones go into the next batch.
// Returns {selected, still_ready}.
//
// `ready_ids` is taken BY VALUE, not by const&: every real caller
// (Controller::select_and_consume) already owns a disposable, freshly
// built vector by the time it calls this (scan_data_status() returns a
// fresh vector, never a reference to internal state) and can std::move it
// in for free. A sampler that needs to reorder/sort its input (e.g.
// GRPOGroupNSampler) can then do so in place instead of making its own
// internal copy first -- this matters because select_and_consume() runs
// while holding Controller's single global lock (see docs/PHASE_7.md), so
// an avoidable O(ready_ids.size()) copy there serializes every other
// partition's concurrent requests behind it too, not just this one's.
//
// `group_ids` is parallel to `ready_ids` (same length, same order,
// group_ids[i] is ready_ids[i]'s real prompt-group identity, "" if never
// recorded) -- Controller::select_and_consume always resolves and supplies
// it, regardless of which sampler is in use, so a group-aware sampler (e.g.
// GRPOGroupNSampler) has access to the real grouping key instead of
// inferring it from id adjacency. FifoSampler ignores it; samplers that
// don't need grouping are free to do the same.
class BaseSampler {
public:
    virtual ~BaseSampler() = default;

    virtual std::pair<std::vector<SampleId>, std::vector<SampleId>>
    sample(std::vector<SampleId> ready_ids, std::vector<std::string> group_ids, std::size_t batch_size) = 0;

    // Track-and-expose only (confirmed decision, see
    // docs/GRPO_GROUP_N_SAMPLER.md): reports which groups among `ready_ids`
    // are incomplete AND have been waiting longer than `max_age_ms` (their
    // oldest member's `produced_at_ms` is more than `max_age_ms` before
    // `now_ms`). MUST NOT delete, clear, or otherwise mutate anything --
    // the caller (e.g. an ops script via a maintenance RPC) decides what,
    // if anything, to do with the report. `produced_at_ms` is parallel to
    // `ready_ids`/`group_ids` (same length/order).
    //
    // Default implementation: reports nothing. A sampler with no concept
    // of "group" (FifoSampler) has nothing that can be "stranded" --
    // overridden meaningfully by GRPOGroupNSampler.
    virtual std::vector<StrandedGroup> find_stranded(const std::vector<SampleId>& ready_ids,
                                                      const std::vector<std::string>& group_ids,
                                                      const std::vector<std::int64_t>& produced_at_ms,
                                                      std::int64_t now_ms, std::int64_t max_age_ms) const {
        (void)ready_ids;
        (void)group_ids;
        (void)produced_at_ms;
        (void)now_ms;
        (void)max_age_ms;
        return {};
    }
};

// Default: FIFO, no rank-awareness, no grouping.
class FifoSampler : public BaseSampler {
public:
    std::pair<std::vector<SampleId>, std::vector<SampleId>> sample(std::vector<SampleId> ready_ids,
                                                                     std::vector<std::string> group_ids,
                                                                     std::size_t batch_size) override;
};

}
