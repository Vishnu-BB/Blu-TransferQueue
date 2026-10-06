#include "transferqueue/Server.h"

namespace tq {

TransferQueueServer::TransferQueueServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                                          std::shared_ptr<StorageManager> storage,
                                          std::shared_ptr<BaseSampler> sampler)
    : RpcServerBase(std::move(server_id)),
      controller_(std::move(controller)),
      storage_(std::move(storage)),
      sampler_(std::move(sampler)) {}

Message TransferQueueServer::handle_request(const Message& request) {
    const auto& body = request.body;

    switch (request.request_type) {
        case RequestType::HANDSHAKE: {
            return Message::create(RequestType::HANDSHAKE_ACK, server_id(), {}, request.sender_id);
        }

        case RequestType::PUT_DATA: {
            auto data = extract_batch(request);

            controller_->create_partition(body.partition_id); // idempotent

            // Throws before any write happens if a field conflicts with a
            // declared schema -- fail fast, no partial writes. Mirrors
            // Client::put()'s same check for the in-process path.
            std::unordered_map<std::string, FieldDtype> field_dtypes;
            for (const auto& [id, record] : data) {
                (void)id;
                for (const auto& [field, tensor] : record) {
                    field_dtypes[field] = to_field_dtype(tensor.dtype());
                }
            }
            controller_->validate_schema(body.partition_id, field_dtypes);

            BatchMeta meta;
            for (const auto& [id, record] : data) {
                (void)record;
                meta.sample_ids.push_back(id);
                meta.partition_ids.push_back(body.partition_id);
            }
            meta.fields = body.fields;
            storage_->put_data(meta, data);

            // Only mark a field produced for a sample if that sample's own
            // record actually contains it -- see Client::put()'s identical
            // fix and docs/UNIT_TEST_FINDINGS.md.
            for (const auto& [id, record] : data) {
                std::vector<std::string> produced_fields;
                produced_fields.reserve(body.fields.size());
                for (const auto& field : body.fields) {
                    if (record.find(field) != record.end()) {
                        produced_fields.push_back(field);
                    }
                }
                std::optional<std::string> group_id_opt =
                    body.group_id.empty() ? std::nullopt : std::optional<std::string>(body.group_id);
                controller_->update_production_status(body.partition_id, {id}, produced_fields, std::nullopt,
                                                        group_id_opt);
            }

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::PUT_DATA_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::GET_META: {
            // Atomic: select_and_consume() does ready-scan, sample, and
            // mark-consumed under one Controller lock. The previous
            // three-separate-calls version had a real race under this
            // server's thread pool -- two concurrent GET_META requests
            // could both read the same ready set before either marked
            // anything consumed, and both receive the same sample(s). See
            // docs/PHASE_7.md.
            auto selected =
                controller_->select_and_consume(body.partition_id, body.fields, body.task_name, *sampler_,
                                                 body.batch_size);

            BatchMeta meta;
            meta.sample_ids = selected;
            meta.partition_ids.assign(selected.size(), body.partition_id);
            meta.fields = body.fields;
            auto data = storage_->get_data(meta);

            MessageBody response_body;
            response_body.success = true;
            response_body.sample_ids = selected;
            response_body.sample_versions.reserve(selected.size());
            for (SampleId id : selected) {
                response_body.sample_versions.push_back(controller_->version_for_sample(body.partition_id, id));
            }
            response_body.current_version = controller_->current_version();
            return make_batch_message(RequestType::GET_META_RESPONSE, server_id(), response_body, data,
                                       request.sender_id);
        }

        case RequestType::CLEAR_DATA: {
            BatchMeta meta;
            meta.sample_ids = body.sample_ids;
            meta.partition_ids.assign(body.sample_ids.size(), body.partition_id);
            storage_->clear_data(meta);

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::CLEAR_DATA_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::CLEAR_PARTITION: {
            auto cleared = controller_->clear_partition(body.partition_id, body.flag);

            // Controller only clears its own metadata -- it has no
            // StorageManager reference, so the actual bytes are only freed
            // here, using the ids it just handed back. This is the
            // colocated single-shard server, so shard_indices (Phase 5's
            // multi-shard fan-out info) doesn't matter here -- every id
            // goes to this one StorageManager regardless.
            if (!cleared.sample_ids.empty()) {
                BatchMeta clear_meta;
                clear_meta.sample_ids = cleared.sample_ids;
                clear_meta.partition_ids.assign(cleared.sample_ids.size(), body.partition_id);
                storage_->clear_data(clear_meta);
            }

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::CLEAR_PARTITION_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::RESET_CONSUMPTION: {
            controller_->reset_consumption(body.partition_id,
                                            body.task_name.empty() ? std::nullopt
                                                                    : std::optional<std::string>(body.task_name));

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::RESET_CONSUMPTION_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::ADVANCE_VERSION: {
            MessageBody response_body;
            response_body.success = true;
            response_body.current_version = controller_->advance_version();
            return Message::create(RequestType::ADVANCE_VERSION_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::FIND_STRANDED_GROUPS: {
            // Maintenance/observability only -- never consumes, clears, or
            // mutates anything. See docs/GRPO_GROUP_N_SAMPLER.md.
            auto stranded = controller_->find_stranded_groups(body.partition_id, body.fields, body.task_name,
                                                                *sampler_, body.max_age_ms);

            MessageBody response_body;
            response_body.success = true;
            for (const auto& group : stranded) {
                response_body.stranded_group_ids.push_back(group.group_id);
                response_body.stranded_group_sizes.push_back(static_cast<std::int32_t>(group.sample_ids.size()));
                response_body.stranded_group_ages_ms.push_back(group.oldest_age_ms);
                response_body.sample_ids.insert(response_body.sample_ids.end(), group.sample_ids.begin(),
                                                 group.sample_ids.end());
            }
            return Message::create(RequestType::FIND_STRANDED_GROUPS_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::DECLARE_SCHEMA: {
            std::unordered_map<std::string, FieldDtype> schema;
            for (std::size_t i = 0; i < body.fields.size() && i < body.field_dtypes.size(); ++i) {
                schema[body.fields[i]] = body.field_dtypes[i];
            }
            controller_->declare_schema(body.partition_id, schema);

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::DECLARE_SCHEMA_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        default: {
            MessageBody response_body;
            response_body.success = false;
            response_body.error_message = "no handler for request_type " + to_string(request.request_type);
            return Message::create(RequestType::REQUEST_ERROR, server_id(), response_body, request.sender_id);
        }
    }
}

}
