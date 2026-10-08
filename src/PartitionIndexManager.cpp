#include "transferqueue/PartitionIndexManager.h"

#include <algorithm>
#include <stdexcept>

namespace tq {

std::vector<SampleId> PartitionIndexManager::allocate_indexes(const std::string& partition_id, std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("PartitionIndexManager::allocate_indexes: count must be > 0");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SampleId> indexes;
    indexes.reserve(count);

    std::size_t num_reuse = std::min(count, reusable_indexes_.size());
    if (num_reuse > 0) {
        indexes.insert(indexes.end(), reusable_indexes_.begin(), reusable_indexes_.begin() + num_reuse);
        reusable_indexes_.erase(reusable_indexes_.begin(), reusable_indexes_.begin() + num_reuse);
    }

    while (indexes.size() < count) {
        indexes.push_back(next_index_++);
    }

    auto& owned = partition_to_indexes_[partition_id];
    owned.insert(indexes.begin(), indexes.end());

    return indexes;
}

std::vector<SampleId> PartitionIndexManager::release_partition(const std::string& partition_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = partition_to_indexes_.find(partition_id);
    if (it == partition_to_indexes_.end()) {
        return {};
    }

    std::vector<SampleId> released(it->second.begin(), it->second.end());
    reusable_indexes_.insert(reusable_indexes_.end(), released.begin(), released.end());
    partition_to_indexes_.erase(it);
    return released;
}

void PartitionIndexManager::release_indexes(const std::string& partition_id,
                                             const std::vector<SampleId>& indexes_to_release) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = partition_to_indexes_.find(partition_id);
    if (it == partition_to_indexes_.end()) {
        return;
    }

    for (SampleId id : indexes_to_release) {
        if (it->second.find(id) == it->second.end()) {
            throw std::invalid_argument("PartitionIndexManager::release_indexes: index not owned by partition");
        }
    }

    std::unordered_set<SampleId> seen;
    std::vector<SampleId> unique_release;
    unique_release.reserve(indexes_to_release.size());
    for (SampleId id : indexes_to_release) {
        if (seen.insert(id).second) {
            unique_release.push_back(id);
        }
    }

    for (SampleId id : unique_release) {
        it->second.erase(id);
    }
    reusable_indexes_.insert(reusable_indexes_.end(), unique_release.begin(), unique_release.end());

    if (it->second.empty()) {
        partition_to_indexes_.erase(it);
    }
}

std::vector<SampleId> PartitionIndexManager::get_indexes_for_partition(const std::string& partition_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = partition_to_indexes_.find(partition_id);
    if (it == partition_to_indexes_.end()) {
        return {};
    }
    return std::vector<SampleId>(it->second.begin(), it->second.end());
}

}