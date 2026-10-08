#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

class PartitionIndexManager {
public:
    std::vector<SampleId> allocate_indexes(const std::string& partition_id, std::size_t count = 1);

    std::vector<SampleId> release_partition(const std::string& partition_id);

    void release_indexes(const std::string& partition_id, const std::vector<SampleId>& indexes_to_release);

    std::vector<SampleId> get_indexes_for_partition(const std::string& partition_id) const;

private:
    std::unordered_map<std::string, std::unordered_set<SampleId>> partition_to_indexes_;
    std::deque<SampleId> reusable_indexes_;
    SampleId next_index_ = 0;

    mutable std::mutex mutex_;
};

}
