#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "transferqueue/DataPartitionStatus.h"
#include "transferqueue/PartitionIndexManager.h"
#include "transferqueue/Sampler.h"

namespace tq {

class TransferQueueController {
public:
    bool create_partition(const std::string& partition_id);

    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    void validate_schema(const std::string& partition_id,
                          const std::unordered_map<std::string, FieldDtype>& field_dtypes) const;

    bool update_production_status(const std::string& partition_id, const std::vector<SampleId>& ids,
                                   const std::vector<std::string>& fields,
                                   std::optional<std::int32_t> shard_index = std::nullopt,
                                   std::optional<std::string> group_id = std::nullopt);

    std::int32_t shard_for_sample(const std::string& partition_id, SampleId id) const;

    std::int64_t version_for_sample(const std::string& partition_id, SampleId id) const;

    std::string group_id_for_sample(const std::string& partition_id, SampleId id) const;

    std::int64_t produced_at(const std::string& partition_id, SampleId id) const;

    std::int64_t advance_version();
    std::int64_t current_version() const;

    void register_shard(std::int32_t shard_index, const std::string& address);
    std::optional<std::string> shard_address(std::int32_t shard_index) const;

    void mark_consumed(const std::string& partition_id, const std::string& task_name,
                        const std::vector<SampleId>& ids);

    std::vector<SampleId> ready_indexes(const std::string& partition_id, const std::vector<std::string>& fields,
                                         const std::string& task_name) const;

    std::vector<SampleId> select_and_consume(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, BaseSampler& sampler,
                                              std::size_t batch_size);

    std::vector<StrandedGroup> find_stranded_groups(const std::string& partition_id,
                                                     const std::vector<std::string>& fields,
                                                     const std::string& task_name, const BaseSampler& sampler,
                                                     std::int64_t max_age_ms) const;

    void reset_consumption(const std::string& partition_id, const std::optional<std::string>& task_name = std::nullopt);

    struct ClearedSamples {
        std::vector<SampleId> sample_ids;
        std::vector<std::int32_t> shard_indices;
    };
    ClearedSamples clear_partition(const std::string& partition_id, bool clear_consumption = true);

private:
    DataPartitionStatus* find_partition(const std::string& partition_id);
    const DataPartitionStatus* find_partition(const std::string& partition_id) const;

    bool ensure_partition_locked(const std::string& partition_id);

    PartitionIndexManager index_manager_;
    std::unordered_map<std::string, DataPartitionStatus> partitions_;
    std::unordered_map<std::int32_t, std::string> shard_registry_;
    std::int64_t current_version_ = 0;

    mutable std::mutex mutex_;
};

}
