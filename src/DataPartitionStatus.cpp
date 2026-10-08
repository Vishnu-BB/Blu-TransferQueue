#include "transferqueue/DataPartitionStatus.h"

#include <chrono>
#include <stdexcept>

namespace tq {

namespace {
std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
}

void DataPartitionStatus::register_pre_allocated_indexes(const std::vector<SampleId>& indexes) {
    std::lock_guard<std::mutex> lock(mutex_);
    pre_allocated_indexes_.insert(indexes.begin(), indexes.end());
}

void DataPartitionStatus::declare_schema(const std::unordered_map<std::string, FieldDtype>& schema) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [field, dtype] : schema) {
        auto existing = declared_schema_.find(field);
        if (existing != declared_schema_.end() && existing->second != dtype) {
            throw std::invalid_argument("DataPartitionStatus::declare_schema: field '" + field +
                                         "' already declared with a different dtype in partition '" + partition_id_ +
                                         "'");
        }
        declared_schema_[field] = dtype;
    }
}

void DataPartitionStatus::validate_field(const std::string& field, FieldDtype dtype) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = declared_schema_.find(field);
    if (it == declared_schema_.end()) {
        return;
    }
    if (it->second != dtype) {
        throw std::invalid_argument("DataPartitionStatus::validate_field: field '" + field + "' in partition '" +
                                     partition_id_ + "' expected dtype " +
                                     std::to_string(static_cast<int>(it->second)) + " but got " +
                                     std::to_string(static_cast<int>(dtype)));
    }
}

void DataPartitionStatus::update_production_status(const std::vector<SampleId>& ids,
                                                     const std::vector<std::string>& fields,
                                                     std::optional<std::int32_t> shard_index,
                                                     std::optional<std::int64_t> policy_version,
                                                     std::optional<std::string> group_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& field : fields) {
        auto& produced = production_by_field_[field];
        produced.insert(ids.begin(), ids.end());
    }
    global_indexes_.insert(ids.begin(), ids.end());
    std::int64_t now = now_ms();
    for (SampleId id : ids) {
        sample_produced_at_.emplace(id, now);
    }
    if (shard_index.has_value()) {
        for (SampleId id : ids) {
            sample_shard_[id] = *shard_index;
        }
    }
    if (policy_version.has_value()) {
        for (SampleId id : ids) {
            sample_version_[id] = *policy_version;
        }
    }
    if (group_id.has_value()) {
        for (SampleId id : ids) {
            sample_group_id_[id] = *group_id;
        }
    }
}

std::int32_t DataPartitionStatus::shard_for_sample(SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sample_shard_.find(id);
    return it == sample_shard_.end() ? -1 : it->second;
}

std::int64_t DataPartitionStatus::version_for_sample(SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sample_version_.find(id);
    return it == sample_version_.end() ? -1 : it->second;
}

std::string DataPartitionStatus::group_id_for_sample(SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sample_group_id_.find(id);
    return it == sample_group_id_.end() ? "" : it->second;
}

std::int64_t DataPartitionStatus::produced_at(SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sample_produced_at_.find(id);
    return it == sample_produced_at_.end() ? -1 : it->second;
}

void DataPartitionStatus::mark_consumed(const std::string& task_name, const std::vector<SampleId>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& consumed = consumption_by_task_[task_name];
    consumed.insert(ids.begin(), ids.end());
}

bool DataPartitionStatus::has_consumed_locked(const std::string& task_name, SampleId id) const {
    auto it = consumption_by_task_.find(task_name);
    if (it == consumption_by_task_.end()) {
        return false;
    }
    return it->second.find(id) != it->second.end();
}

bool DataPartitionStatus::has_consumed(const std::string& task_name, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return has_consumed_locked(task_name, id);
}

void DataPartitionStatus::reset_consumption(const std::optional<std::string>& task_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (task_name) {
        auto it = consumption_by_task_.find(*task_name);
        if (it != consumption_by_task_.end()) {
            it->second.clear();
        }
    } else {
        for (auto& [name, consumed] : consumption_by_task_) {
            consumed.clear();
        }
    }
}

bool DataPartitionStatus::is_produced_locked(SampleId id, const std::vector<std::string>& fields) const {
    for (const auto& field : fields) {
        auto it = production_by_field_.find(field);
        if (it == production_by_field_.end() || it->second.find(id) == it->second.end()) {
            return false;
        }
    }
    return true;
}

std::vector<SampleId> DataPartitionStatus::scan_data_status(const std::vector<std::string>& fields,
                                                              const std::string& task_name) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // Unregistered field -> nothing can be ready for it.
    for (const auto& field : fields) {
        if (production_by_field_.find(field) == production_by_field_.end()) {
            return {};
        }
    }

    std::vector<SampleId> ready;
    for (SampleId id : global_indexes_) {
        if (!has_consumed_locked(task_name, id) && is_produced_locked(id, fields)) {
            ready.push_back(id);
        }
    }
    return ready;
}

void DataPartitionStatus::clear_data(const std::vector<SampleId>& ids, bool clear_consumption) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [field, produced] : production_by_field_) {
        for (SampleId id : ids) {
            produced.erase(id);
        }
    }
    if (clear_consumption) {
        for (auto& [task, consumed] : consumption_by_task_) {
            for (SampleId id : ids) {
                consumed.erase(id);
            }
        }
    }
    for (SampleId id : ids) {
        global_indexes_.erase(id);
        pre_allocated_indexes_.erase(id);
        sample_shard_.erase(id);
        sample_version_.erase(id);
        sample_group_id_.erase(id);
        sample_produced_at_.erase(id);
    }
}

std::size_t DataPartitionStatus::total_samples_num() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return global_indexes_.size();
}

std::vector<SampleId> DataPartitionStatus::all_sample_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SampleId> ids(global_indexes_.begin(), global_indexes_.end());
    for (SampleId id : pre_allocated_indexes_) {
        if (global_indexes_.find(id) == global_indexes_.end()) {
            ids.push_back(id);
        }
    }
    return ids;
}

}
