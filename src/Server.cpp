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

            // Group samples by the exact subset of body.fields their own
            // record actually contains (not every sample need have every
            // field -- see the "gap" regression test in ClientTest.cpp/
            // ServerTest.cpp) so update_production_status is called once
            // per group instead of once per sample. Typically one group for
            // the whole batch, since real callers produce the same fields
            // for every sample -- collapsing N Controller-lock acquisitions
            // (and N DataPartitionStatus-lock acquisitions underneath) into
            // one in that common case, same end state either way (see
            // DataPartitionStatus::update_production_status: it just
            // inserts every id into each field's produced-set, order- and
            // batching-independent).
            std::optional<std::string> group_id_opt =
                body.group_id.empty() ? std::nullopt : std::optional<std::string>(body.group_id);
            std::unordered_map<std::string, std::pair<std::vector<std::string>, std::vector<SampleId>>> groups;
            for (const auto& [id, record] : data) {
                std::vector<std::string> produced_fields;
                produced_fields.reserve(body.fields.size());
                for (const auto& field : body.fields) {
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
                controller_->update_production_status(body.partition_id, group.second, group.first, std::nullopt,
                                                        group_id_opt);
            }

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::PUT_DATA_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::GET_META: {
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
