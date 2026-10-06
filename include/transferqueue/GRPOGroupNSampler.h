#pragma once

#include <cstddef>

#include "transferqueue/Sampler.h"

namespace tq {

// Group-based sampling without replacement for GRPO, where
// `n_samples_per_prompt` completions share one prompt and must be consumed
// together or not at all (never a partial group), since the group-relative
// advantage is only computable once every completion in the group is in
// hand.
//
// Groups by the real `group_id` parallel array BaseSampler::sample()
// receives (see Sampler.h), not by numeric id adjacency. This was changed
// from an earlier id-contiguity-based port of upstream's
// transfer_queue/sampler/grpo_group_n_sampler.py after an external review
// correctly identified a real, confirmed defect in that approach: see
// "Fixed: silent cross-group contamination" below. `group_id` for a sample
// is set via Controller::update_production_status's/
// DataPartitionStatus::update_production_status's optional `group_id`
// parameter -- it's the SAME string a writer already passes to
// GroupRouter::target_rank() for write-time shard routing (Phase 4), so a
// GRPO pipeline has exactly one group-identity string per prompt, used
// consistently for both routing and sampling.
//
// A sample with no recorded group_id ("") can never be grouped -- it
// simply stays in `still_ready` forever, same as a genuinely incomplete
// group. This is a deliberate fail-safe: refusing to guess is strictly
// better than the old behavior's risk of silently guessing wrong.
//
// Fixed: silent cross-group contamination. The old id-adjacency version
// had no way to distinguish "these ids are really one prompt's group" from
// "these ids just happen to be numerically adjacent." Two concurrent
// writers racing on id allocation (worker A gets ids {100,102}, worker B
// gets {101,103}) would produce sorted_ready = [100,101,102,103] --
// indistinguishable from one real group of 4, silently fusing two
// unrelated prompts' completions and corrupting the trainer's advantage
// computation with no error raised. Grouping by the real group_id instead
// makes this impossible: {100,102} and {101,103} bucket into their own
// groups regardless of how their ids interleave numerically. See
// docs/GRPO_GROUP_N_SAMPLER.md for the verification (a regression test
// reproduces the exact contamination scenario and confirms it no longer
// occurs).
//
// Tracked (not auto-evicted -- confirmed decision, see
// docs/GRPO_GROUP_N_SAMPLER.md): stranded samples / head-of-line blocking.
// If one member of a group never arrives, the rest sit in storage
// indefinitely with nothing here selecting them. find_stranded() (below)
// reports which incomplete groups have been waiting longer than a caller-
// given threshold -- a pure report, never an action. Deciding whether (and
// how) to actually clear a stranded group remains the caller's call.
//
// Also no longer applicable (a side effect of this fix, not separately
// addressed): the earlier caveat about PartitionIndexManager's id-reuse
// pool breaking numeric contiguity. Since grouping no longer depends on
// numeric contiguity at all, a recycled id landing inside a group's id
// range is harmless -- the group_id, not the id's value, is what's
// authoritative now.
//
// Deliberately NOT ported from upstream: per-(dp_rank, batch_index) result
// caching for deterministic replay across repeated calls with identical
// parameters. BaseSampler::sample() here has no dp_rank/batch_index
// concept -- no call site threads them through, and nothing in this
// codebase yet retries a read for "the same logical batch" (that only
// matters once multiple DP ranks independently call GET_META for shares of
// one global batch).
class GRPOGroupNSampler : public BaseSampler {
public:
    // Throws std::invalid_argument if n_samples_per_prompt is 0.
    explicit GRPOGroupNSampler(std::size_t n_samples_per_prompt = 1);

    // Throws std::invalid_argument if batch_size isn't a multiple of
    // n_samples_per_prompt, or if ready_ids.size() != group_ids.size().
    // Buckets ready ids by their real group_id (samples with an empty
    // group_id are never bucketed -- they can't be grouped), visits
    // buckets in sorted-group_id order for determinism, and accepts a
    // bucket as a complete group once it has at least n_samples_per_prompt
    // members (selecting the n_samples_per_prompt smallest ids from it,
    // sorted). Keeps scanning until it finds enough complete groups to
    // fill batch_size, or runs out -- in which case it returns nothing
    // selected at all (never a partial batch), with still_ready holding
    // every id, sorted. Returns {selected, still_ready}, both sorted; no
    // caller relies on original input order (every call site discards
    // still_ready).
    std::pair<std::vector<SampleId>, std::vector<SampleId>> sample(std::vector<SampleId> ready_ids,
                                                                     std::vector<std::string> group_ids,
                                                                     std::size_t batch_size) override;

    // Track-and-expose age-based reporting (see BaseSampler::find_stranded's
    // doc for the full contract -- this never deletes or mutates anything).
    // Buckets ready ids by group_id exactly like sample() does, PLUS one
    // extra bucket for group_id == "" (never recorded) -- an ungrouped id
    // can never complete a group by definition, so it's always eligible to
    // be reported, unlike a real group where eligibility depends on
    // n_samples_per_prompt. A bucket is reported iff it's incomplete (real
    // group with < n_samples_per_prompt members, or the "" bucket
    // unconditionally) AND its oldest member's produced_at_ms is more than
    // max_age_ms before now_ms. Throws std::invalid_argument if the three
    // input vectors aren't all the same length.
    std::vector<StrandedGroup> find_stranded(const std::vector<SampleId>& ready_ids,
                                              const std::vector<std::string>& group_ids,
                                              const std::vector<std::int64_t>& produced_at_ms, std::int64_t now_ms,
                                              std::int64_t max_age_ms) const override;

private:
    std::size_t n_samples_per_prompt_;
};

}
