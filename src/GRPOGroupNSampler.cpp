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

    std::vector<std::size_t> order(ready_ids.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ready_ids[a] < ready_ids[b]; });

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
            continue; 
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

    std::map<std::string, std::vector<std::pair<SampleId, std::int64_t>>> buckets;
    for (std::size_t i = 0; i < ready_ids.size(); ++i) {
        buckets[group_ids[i]].emplace_back(ready_ids[i], produced_at_ms[i]);
    }

    std::vector<StrandedGroup> stranded;
    for (auto& [group_id, members] : buckets) {
        bool incomplete = group_id.empty() || members.size() < n_samples_per_prompt_;
        if (!incomplete) {
            continue; 
        }

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
            continue; 
        }

        std::int64_t age_ms = now_ms - oldest_produced_at;
        if (age_ms < max_age_ms) {
            continue; 
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
