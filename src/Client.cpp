#include "transferqueue/Client.h"

namespace tq {

TransferQueueClient::TransferQueueClient(std::shared_ptr<TransferQueueController> controller,
                                          std::shared_ptr<StorageManager> storage,
                                          std::shared_ptr<BaseSampler> sampler)
    : controller_(std::move(controller)), storage_(std::move(storage)), sampler_(std::move(sampler)) {}

void TransferQueueClient::declare_schema(const std::string& partition_id,
                                          const std::unordered_map<std::string, FieldDtype>& schema) {
    controller_->declare_schema(partition_id, schema);
}

void TransferQueueClient::put(const std::string& partition_id, const std::vector<std::string>& fields,
                               const std::unordered_map<SampleId, Record>& data, const std::string& group_id) {
    controller_->create_partition(partition_id); 

    std::unordered_map<std::string, FieldDtype> field_dtypes;
    for (const auto& [id, record] : data) {
        (void)id;
        for (const auto& [field, tensor] : record) {
            field_dtypes[field] = to_field_dtype(tensor.dtype());
        }
    }
    controller_->validate_schema(partition_id, field_dtypes);

    BatchMeta meta;
    for (const auto& [id, record] : data) {
        (void)record;
        meta.sample_ids.push_back(id);
        meta.partition_ids.push_back(partition_id);
    }
    meta.fields = fields;
    storage_->put_data(meta, data);

    // Group samples by the exact subset of `fields` their own record
    // actually contains (not every sample need have every field -- see the
    // "gap" regression test below) so update_production_status is called
    // once per group instead of once per sample. Typically one group for
    // the whole batch -- collapsing N Controller-lock acquisitions into one
    // in that common case; see Server.cpp's PUT_DATA handler for the same
    // fix on the RPC path and why batching is behavior-preserving.
    std::optional<std::string> group_id_opt = group_id.empty() ? std::nullopt : std::optional<std::string>(group_id);
    std::unordered_map<std::string, std::pair<std::vector<std::string>, std::vector<SampleId>>> groups;
    for (const auto& [id, record] : data) {
        std::vector<std::string> produced_fields;
        produced_fields.reserve(fields.size());
        for (const auto& field : fields) {
            if (record.find(field) != record.end()) {
                produced_fields.push_back(field);
            }
        }
        std::string signature;
        for (const auto& f : produced_fields) {
            signature += f;
            signature += '\x1f';
        }
        auto& group = groups[signature];
        group.first = produced_fields;
        group.second.push_back(id);
    }
    for (auto& [signature, group] : groups) {
        (void)signature;
        controller_->update_production_status(partition_id, group.second, group.first, std::nullopt, group_id_opt);
    }
}

std::unordered_map<SampleId, Record> TransferQueueClient::get(const std::string& partition_id,
                                                                const std::vector<std::string>& fields,
                                                                const std::string& task_name,
                                                                std::size_t batch_size) {

    auto selected = controller_->select_and_consume(partition_id, fields, task_name, *sampler_, batch_size);

    BatchMeta meta;
    meta.sample_ids = selected;
    meta.partition_ids.assign(selected.size(), partition_id);
    meta.fields = fields;

    auto data = storage_->get_data(meta);
    return data;
}

}
