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
    controller_->create_partition(partition_id); // idempotent: no-op if it already exists

    // Throws before any write happens if a field conflicts with a declared
    // schema -- fail fast, no partial writes.
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

    // Only mark a field produced for a sample if that sample's own record
    // actually contains it -- marking every field in `fields` regardless
    // of what a given id's record actually held meant a sample missing a
    // claimed field could be reported ready, and a consumer's .at() on the
    // missing field would then throw downstream instead of failing here.
    // See docs/UNIT_TEST_FINDINGS.md.
    for (const auto& [id, record] : data) {
        std::vector<std::string> produced_fields;
        produced_fields.reserve(fields.size());
        for (const auto& field : fields) {
            if (record.find(field) != record.end()) {
                produced_fields.push_back(field);
            }
        }
        std::optional<std::string> group_id_opt = group_id.empty() ? std::nullopt : std::optional<std::string>(group_id);
        controller_->update_production_status(partition_id, {id}, produced_fields, std::nullopt, group_id_opt);
    }
}

std::unordered_map<SampleId, Record> TransferQueueClient::get(const std::string& partition_id,
                                                                const std::vector<std::string>& fields,
                                                                const std::string& task_name,
                                                                std::size_t batch_size) {
    // Atomic: see Controller::select_and_consume's doc comment -- matters
    // if multiple threads ever share one Client instance concurrently
    // against the same partition/task_name, not just the RPC servers'
    // thread pool.
    auto selected = controller_->select_and_consume(partition_id, fields, task_name, *sampler_, batch_size);

    BatchMeta meta;
    meta.sample_ids = selected;
    meta.partition_ids.assign(selected.size(), partition_id);
    meta.fields = fields;

    auto data = storage_->get_data(meta);
    return data;
}

}
