#include "transferqueue/GRPOGroupNSampler.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tq {

GRPOGroupNSampler::GRPOGroupNSampler(std::size_t n_samples_per_prompt)
    : n_samples_per_prompt_(n_samples_per_prompt) {
    if (n_samples_per_prompt_ == 0) {
        throw std::invalid_argument("GRPOGroupNSampler: n_samples_per_prompt must be positive");
    }
}

std::pair<std::vector<SampleId>, std::vector<SampleId>> GRPOGroupNSampler::sample(std::vector<SampleId> ready_ids,
                                                                                   std::vector<std::string> group_ids,
                                                                                   std::size_t batch_size) {
    if (ready_ids.size() != group_ids.size()) {
        throw std::invalid_argument("GRPOGroupNSampler::sample: ready_ids and group_ids must be the same length");
    }
    if (batch_size % n_samples_per_prompt_ != 0) {
        throw std::invalid_argument("GRPOGroupNSampler::sample: batch_size (" + std::to_string(batch_size) +
                                     ") must be a multiple of n_samples_per_prompt (" +
                                     std::to_string(n_samples_per_prompt_) + ")");
    }

    std::size_t required_groups = batch_size / n_samples_per_prompt_;

    // Sort by SampleId via an index permutation (not sorting ready_ids
    // directly) so group_ids stays in sync without a second parallel sort
    // or an extra copy -- this is the one O(M log M) pass; everything
    // after it is O(M) or O(G log G) (G = distinct groups <= M). Running
    // this under Controller's global lock (see docs/PHASE_7.md) is why
    // avoiding redundant copies/sorts here actually matters.
    std::vector<std::size_t> order(ready_ids.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ready_ids[a] < ready_ids[b]; });

    // Bucket by real group_id (std::map -> deterministic, sorted-by-key
    // iteration order below). A sample with no recorded group_id ("") is
    // simply never bucketed -- it can never be grouped, so it can never be
    // selected; it always ends up in still_ready. sorted_ids is built in
    // the same pass, sorted by SampleId, used both for the
    // insufficient-groups return and for building still_ready.
    std::map<std::string, std::vector<SampleId>> buckets;
    std::vector<SampleId> sorted_ids;
    sorted_ids.reserve(ready_ids.size());
    for (std::size_t idx : order) {
        sorted_ids.push_back(ready_ids[idx]);
        if (!group_ids[idx].empty()) {
            buckets[group_ids[idx]].push_back(ready_ids[idx]);
        }
    }

    std::vector<SampleId> selected;
    std::unordered_set<SampleId> selected_set;
    std::size_t found_groups = 0;
    for (auto& [group_id, ids] : buckets) {
        (void)group_id;
        if (found_groups >= required_groups) {
            break;
        }
        if (ids.size() < n_samples_per_prompt_) {
            continue; // incomplete group -- stays ready, never selected
        }
        for (std::size_t k = 0; k < n_samples_per_prompt_; ++k) {
            selected.push_back(ids[k]);
            selected_set.insert(ids[k]);
        }
        ++found_groups;
    }

    if (found_groups < required_groups) {
        return {{}, std::move(sorted_ids)};
    }

    // selected is built in group-visit order (alphabetical by group_id),
    // not numeric id order -- groups found later can have smaller ids than
    // ones found earlier if their group_ids interleave. Sort it too so
    // both returned vectors are consistently in SampleId order; cheap,
    // since |selected| == batch_size, not M.
    std::sort(selected.begin(), selected.end());

    std::vector<SampleId> still_ready;
    still_ready.reserve(sorted_ids.size() - selected.size());
    for (SampleId id : sorted_ids) {
        if (selected_set.find(id) == selected_set.end()) {
            still_ready.push_back(id);
        }
    }

    return {selected, still_ready};
}

std::vector<StrandedGroup> GRPOGroupNSampler::find_stranded(const std::vector<SampleId>& ready_ids,
                                                             const std::vector<std::string>& group_ids,
                                                             const std::vector<std::int64_t>& produced_at_ms,
                                                             std::int64_t now_ms, std::int64_t max_age_ms) const {
    if (ready_ids.size() != group_ids.size() || ready_ids.size() != produced_at_ms.size()) {
        throw std::invalid_argument(
            "GRPOGroupNSampler::find_stranded: ready_ids, group_ids, and produced_at_ms must be the same length");
    }

    // Bucket every ready id by group_id, INCLUDING the "" (never recorded)
    // bucket -- an ungrouped id can never complete a group by definition,
    // so it's always worth surfacing if it's been sitting a while,
    // regardless of how many other ungrouped ids happen to be around it.
    std::map<std::string, std::vector<std::pair<SampleId, std::int64_t>>> buckets;
    for (std::size_t i = 0; i < ready_ids.size(); ++i) {
        buckets[group_ids[i]].emplace_back(ready_ids[i], produced_at_ms[i]);
    }

    std::vector<StrandedGroup> stranded;
    for (auto& [group_id, members] : buckets) {
        bool incomplete = group_id.empty() || members.size() < n_samples_per_prompt_;
        if (!incomplete) {
            continue; // a complete, selectable group -- not stranded, just waiting its turn
        }

        // Oldest member = smallest (earliest) produced_at_ms. -1 entries
        // (unknown timestamp -- shouldn't happen for a genuinely ready id,
        // since Controller always stamps this, but handled defensively)
        // are excluded from the search rather than treated as infinitely
        // old.
        std::int64_t oldest_produced_at = -1;
        for (const auto& [id, produced_at] : members) {
            (void)id;
            if (produced_at < 0) {
                continue;
            }
            if (oldest_produced_at < 0 || produced_at < oldest_produced_at) {
                oldest_produced_at = produced_at;
            }
        }
        if (oldest_produced_at < 0) {
            continue; // no known timestamp for anything in this bucket -- can't judge age
        }

        std::int64_t age_ms = now_ms - oldest_produced_at;
        if (age_ms < max_age_ms) {
            continue; // not old enough yet
        }

        std::vector<SampleId> sample_ids;
        sample_ids.reserve(members.size());
        for (const auto& [id, produced_at] : members) {
            (void)produced_at;
            sample_ids.push_back(id);
        }
        std::sort(sample_ids.begin(), sample_ids.end());

        stranded.push_back(StrandedGroup{group_id, std::move(sample_ids), age_ms});
    }

    return stranded;
}

}
