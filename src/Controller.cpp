#include "transferqueue/Controller.h"

#include <chrono>
#include <tuple>
#include <utility>

namespace {
constexpr std::size_t kPreAllocSampleNum = 1;

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
}

namespace tq {

bool TransferQueueController::ensure_partition_locked(const std::string& partition_id) {
    if (partitions_.find(partition_id) != partitions_.end()) {
        return false;
    }

    // Piecewise construction: DataPartitionStatus holds a std::mutex, so it's
    // non-movable -- it must be built in place inside the map node, not
    // constructed as a temporary and moved in.
    partitions_.emplace(std::piecewise_construct, std::forward_as_tuple(partition_id),
                         std::forward_as_tuple(partition_id));

    auto allocated = index_manager_.allocate_indexes(partition_id, kPreAllocSampleNum);
    partitions_.at(partition_id).register_pre_allocated_indexes(allocated);

    return true;
}

bool TransferQueueController::create_partition(const std::string& partition_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_partition_locked(partition_id);
}

void TransferQueueController::declare_schema(const std::string& partition_id,
                                              const std::unordered_map<std::string, FieldDtype>& schema) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_partition_locked(partition_id);
    find_partition(partition_id)->declare_schema(schema);
}

void TransferQueueController::validate_schema(const std::string& partition_id,
                                               const std::unordered_map<std::string, FieldDtype>& field_dtypes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    for (const auto& [field, dtype] : field_dtypes) {
        partition->validate_field(field, dtype);
    }
}

// Private, lock-free: every call site already holds mutex_.
DataPartitionStatus* TransferQueueController::find_partition(const std::string& partition_id) {
    auto it = partitions_.find(partition_id);
    return it == partitions_.end() ? nullptr : &it->second;
}

const DataPartitionStatus* TransferQueueController::find_partition(const std::string& partition_id) const {
    auto it = partitions_.find(partition_id);
    return it == partitions_.end() ? nullptr : &it->second;
}

bool TransferQueueController::update_production_status(const std::string& partition_id,
                                                         const std::vector<SampleId>& ids,
                                                         const std::vector<std::string>& fields,
                                                         std::optional<std::int32_t> shard_index,
                                                         std::optional<std::string> group_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return false;
    }
    partition->update_production_status(ids, fields, shard_index, current_version_, group_id);
    return true;
}

std::int32_t TransferQueueController::shard_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->shard_for_sample(id);
}

std::int64_t TransferQueueController::version_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->version_for_sample(id);
}

std::string TransferQueueController::group_id_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return "";
    }
    return partition->group_id_for_sample(id);
}

std::int64_t TransferQueueController::produced_at(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->produced_at(id);
}

std::int64_t TransferQueueController::advance_version() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ++current_version_;
}

std::int64_t TransferQueueController::current_version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_version_;
}

void TransferQueueController::register_shard(std::int32_t shard_index, const std::string& address) {
    std::lock_guard<std::mutex> lock(mutex_);
    shard_registry_[shard_index] = address;
}

std::optional<std::string> TransferQueueController::shard_address(std::int32_t shard_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shard_registry_.find(shard_index);
    if (it == shard_registry_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void TransferQueueController::mark_consumed(const std::string& partition_id, const std::string& task_name,
                                             const std::vector<SampleId>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    partition->mark_consumed(task_name, ids);
}

std::vector<SampleId> TransferQueueController::ready_indexes(const std::string& partition_id,
                                                               const std::vector<std::string>& fields,
                                                               const std::string& task_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    return partition->scan_data_status(fields, task_name);
}

std::vector<SampleId> TransferQueueController::select_and_consume(const std::string& partition_id,
                                                                    const std::vector<std::string>& fields,
                                                                    const std::string& task_name,
                                                                    BaseSampler& sampler, std::size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    auto ready = partition->scan_data_status(fields, task_name);

    // Resolve each ready id's real group_id so group-aware samplers (e.g.
    // GRPOGroupNSampler) can group by actual prompt identity instead of
    // guessing from id adjacency -- see docs/GRPO_GROUP_N_SAMPLER.md.
    std::vector<std::string> group_ids;
    group_ids.reserve(ready.size());
    for (SampleId id : ready) {
        group_ids.push_back(partition->group_id_for_sample(id));
    }

    // Moved, not copied: `ready` is a fresh vector nothing else references
    // (scan_data_status never returns a reference to internal state), and
    // this call runs under mutex_ -- avoiding the copy here matters since
    // the lock is global across every partition, not just this one. See
    // BaseSampler's by-value parameter note (Sampler.h) and docs/PHASE_7.md.
    auto [selected, remaining] = sampler.sample(std::move(ready), std::move(group_ids), batch_size);
    (void)remaining;
    if (!selected.empty()) {
        partition->mark_consumed(task_name, selected);
    }
    return selected;
}

std::vector<StrandedGroup> TransferQueueController::find_stranded_groups(const std::string& partition_id,
                                                                          const std::vector<std::string>& fields,
                                                                          const std::string& task_name,
                                                                          const BaseSampler& sampler,
                                                                          std::int64_t max_age_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    auto ready = partition->scan_data_status(fields, task_name);

    std::vector<std::string> group_ids;
    std::vector<std::int64_t> produced_at_ms;
    group_ids.reserve(ready.size());
    produced_at_ms.reserve(ready.size());
    for (SampleId id : ready) {
        group_ids.push_back(partition->group_id_for_sample(id));
        produced_at_ms.push_back(partition->produced_at(id));
    }

    // Read-only: unlike select_and_consume, nothing here is ever marked
    // consumed or cleared -- this is purely a report.
    return sampler.find_stranded(ready, group_ids, produced_at_ms, now_ms(), max_age_ms);
}

void TransferQueueController::reset_consumption(const std::string& partition_id,
                                                 const std::optional<std::string>& task_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    partition->reset_consumption(task_name);
}

TransferQueueController::ClearedSamples TransferQueueController::clear_partition(const std::string& partition_id,
                                                                                  bool clear_consumption) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    // The real set of sample ids this partition actually contains --
    // PartitionIndexManager only ever tracks the pre-allocated placeholder
    // it mints itself (see ensure_partition_locked); real sample ids are
    // chosen by callers and never registered with it, so
    // get_indexes_for_partition() can't answer "what does this partition
    // contain" (confirmed: returned stale/wrong ids here before this fix,
    // meaning clear_partition never actually cleared real data at all).
    auto owned_indexes = partition->all_sample_ids();

    // Shard assignments must be read before clear_data() -- it erases
    // sample_shard_ entries, so this is the last chance to know where each
    // id actually lived (needed for Phase 5's multi-shard fan-out).
    ClearedSamples result;
    result.sample_ids = owned_indexes;
    result.shard_indices.reserve(owned_indexes.size());
    for (SampleId id : owned_indexes) {
        result.shard_indices.push_back(partition->shard_for_sample(id));
    }

    partition->clear_data(owned_indexes, clear_consumption);
    index_manager_.release_partition(partition_id); // frees the pre-allocated placeholder for reuse
    partitions_.erase(partition_id);
    return result;
}

}
