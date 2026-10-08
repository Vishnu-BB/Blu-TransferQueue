#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

class DataPartitionStatus {
public:
    explicit DataPartitionStatus(std::string partition_id) : partition_id_(std::move(partition_id)) {}

    void register_pre_allocated_indexes(const std::vector<SampleId>& indexes);

    void declare_schema(const std::unordered_map<std::string, FieldDtype>& schema);

    void validate_field(const std::string& field, FieldDtype dtype) const;

    void update_production_status(const std::vector<SampleId>& ids, const std::vector<std::string>& fields,
                                   std::optional<std::int32_t> shard_index = std::nullopt,
                                   std::optional<std::int64_t> policy_version = std::nullopt,
                                   std::optional<std::string> group_id = std::nullopt);

    std::int32_t shard_for_sample(SampleId id) const;

    std::int64_t version_for_sample(SampleId id) const;

    std::string group_id_for_sample(SampleId id) const;

    std::int64_t produced_at(SampleId id) const;

    void mark_consumed(const std::string& task_name, const std::vector<SampleId>& ids);
    bool has_consumed(const std::string& task_name, SampleId id) const;
    void reset_consumption(const std::optional<std::string>& task_name = std::nullopt);

    std::vector<SampleId> scan_data_status(const std::vector<std::string>& fields, const std::string& task_name) const;

    void clear_data(const std::vector<SampleId>& ids, bool clear_consumption = true);

    std::size_t total_samples_num() const;

    std::vector<SampleId> all_sample_ids() const;

private:
    // Lock-free: callers already hold mutex_.
    bool is_produced_locked(SampleId id, const std::vector<std::string>& fields) const;
    bool has_consumed_locked(const std::string& task_name, SampleId id) const;

    std::string partition_id_;
    std::unordered_set<SampleId> global_indexes_;
    std::unordered_set<SampleId> pre_allocated_indexes_;
    std::unordered_map<std::string, std::unordered_set<SampleId>> production_by_field_;
    std::unordered_map<std::string, std::unordered_set<SampleId>> consumption_by_task_;
    std::unordered_map<std::string, FieldDtype> declared_schema_;
    std::unordered_map<SampleId, std::int32_t> sample_shard_;
    std::unordered_map<SampleId, std::int64_t> sample_version_;
    std::unordered_map<SampleId, std::string> sample_group_id_;
    std::unordered_map<SampleId, std::int64_t> sample_produced_at_;

    mutable std::mutex mutex_;
};

}
