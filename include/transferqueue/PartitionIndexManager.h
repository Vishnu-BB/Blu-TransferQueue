#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

// Allocates/releases global sample indexes per partition. Reuse pool (FIFO)
// is checked first; a monotonic counter mints new ids when it's empty.
// Mirrors upstream PartitionIndexManager.
//
// Thread-safe: one mutex guards all member access (same reasoning as
// DataPartitionStatus -- concurrent allocate/release calls are now real
// once TransferQueueServer processes requests on a thread pool).
class PartitionIndexManager {
public:
    std::vector<SampleId> allocate_indexes(const std::string& partition_id, std::size_t count = 1);

    // Releases all indexes owned by partition_id back to the reuse pool.
    std::vector<SampleId> release_partition(const std::string& partition_id);

    // Releases specific indexes; throws if any is not owned by partition_id.
    void release_indexes(const std::string& partition_id, const std::vector<SampleId>& indexes_to_release);

    std::vector<SampleId> get_indexes_for_partition(const std::string& partition_id) const;

private:
    std::unordered_map<std::string, std::unordered_set<SampleId>> partition_to_indexes_;
    // deque, not vector: allocate_indexes() erases a prefix range from the
    // front on every reuse-pool hit. A vector has to shift every remaining
    // element for that (O(pool size)); a deque only adjusts its front block
    // pointers (O(erased count)) -- same FIFO order, same external
    // behavior, strictly better complexity. (An earlier suggestion to fix
    // this by switching to LIFO order was declined back in Phase 0 because
    // it would have changed upstream's documented FIFO semantics; this
    // doesn't change any observable behavior at all.)
    std::deque<SampleId> reusable_indexes_;
    SampleId next_index_ = 0;

    mutable std::mutex mutex_;
};

}
