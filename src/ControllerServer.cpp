#include "transferqueue/ControllerServer.h"

#include <stdexcept>
#include <unordered_map>

#include "transferqueue/RpcClient.h"

namespace tq {

ControllerServer::ControllerServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                                    std::shared_ptr<BaseSampler> sampler)
    : RpcServerBase(std::move(server_id)), controller_(std::move(controller)), sampler_(std::move(sampler)) {}

Message ControllerServer::handle_request(const Message& request) {
    const auto& body = request.body;

    switch (request.request_type) {
        case RequestType::HANDSHAKE: {
            // A plain client just checking connectivity leaves shard_index
            // at its -1 default; a StorageServer announcing itself sets
            // shard_index + address, which is the only thing that actually
            // registers it.
            if (body.shard_index >= 0 && !body.address.empty()) {
                controller_->register_shard(body.shard_index, body.address);
            }
            return Message::create(RequestType::HANDSHAKE_ACK, server_id(), {}, request.sender_id);
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

        case RequestType::NOTIFY_DATA_UPDATE: {
            controller_->create_partition(body.partition_id); // idempotent

            // Validated only now, after the StorageServer has already
            // durably written the data -- no two-phase-commit protocol to
            // reject the write up front. A mismatch here is a loud signal
            // to the writer (NOTIFY_DATA_UPDATE_ERROR), not a rollback.
            std::unordered_map<std::string, FieldDtype> field_dtypes;
            for (std::size_t i = 0; i < body.fields.size() && i < body.field_dtypes.size(); ++i) {
                field_dtypes[body.fields[i]] = body.field_dtypes[i];
            }
            try {
                controller_->validate_schema(body.partition_id, field_dtypes);
            } catch (const std::invalid_argument& e) {
                MessageBody response_body;
                response_body.success = false;
                response_body.error_message = e.what();
                return Message::create(RequestType::NOTIFY_DATA_UPDATE_ERROR, server_id(), response_body,
                                        request.sender_id);
            }

            std::optional<std::int32_t> shard_index =
                body.shard_index >= 0 ? std::optional<std::int32_t>(body.shard_index) : std::nullopt;
            std::optional<std::string> group_id =
                body.group_id.empty() ? std::nullopt : std::optional<std::string>(body.group_id);
            controller_->update_production_status(body.partition_id, body.sample_ids, body.fields, shard_index,
                                                    group_id);

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::NOTIFY_DATA_UPDATE_ACK, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::GET_SHARD_ADDRESS: {
            auto address = controller_->shard_address(body.shard_index);

            MessageBody response_body;
            response_body.shard_index = body.shard_index;
            response_body.success = address.has_value();
            if (address) {
                response_body.address = *address;
            } else {
                response_body.error_message = "shard_index not registered";
            }
            return Message::create(RequestType::GET_SHARD_ADDRESS_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::GET_META: {
            // Atomic: select_and_consume() does ready-scan, sample, and
            // mark-consumed under one Controller lock -- required once
            // multiple independent readers (e.g. several trainer ranks in
            // an N:M topology, see docs/PHASE_7.md) can issue concurrent
            // GET_META requests against the same partition/task_name. The
            // previous three-separate-calls version could let two
            // concurrent requests both read the same ready set before
            // either marked anything consumed, handing the same sample(s)
            // to both.
            auto selected =
                controller_->select_and_consume(body.partition_id, body.fields, body.task_name, *sampler_,
                                                 body.batch_size);

            // No StorageManager here -- this bundles each selected sample's
            // shard + every distinct shard's resolved address (per the
            // decision to bundle rather than force a separate
            // GET_SHARD_ADDRESS round trip), but the actual tensor data is
            // a second hop: the caller uses this to send GET_DATA straight
            // to each shard.
            MessageBody response_body;
            response_body.success = true;
            response_body.sample_ids = selected;
            response_body.sample_shard_indices.reserve(selected.size());

            std::unordered_map<std::int32_t, std::string> distinct_shards;
            for (SampleId id : selected) {
                std::int32_t shard = controller_->shard_for_sample(body.partition_id, id);
                response_body.sample_shard_indices.push_back(shard);
                if (shard >= 0 && distinct_shards.find(shard) == distinct_shards.end()) {
                    if (auto address = controller_->shard_address(shard)) {
                        distinct_shards.emplace(shard, *address);
                    }
                }
            }
            for (const auto& [shard, address] : distinct_shards) {
                response_body.shard_registry_indices.push_back(shard);
                response_body.shard_registry_addresses.push_back(address);
            }

            // Phase 6 (staleness/versioning): bundled the same way as the
            // shard registry above -- each selected sample's produced-at
            // version, plus the current version as of this read, so the
            // caller can compute staleness without a separate round trip.
            response_body.sample_versions.reserve(selected.size());
            for (SampleId id : selected) {
                response_body.sample_versions.push_back(controller_->version_for_sample(body.partition_id, id));
            }
            response_body.current_version = controller_->current_version();

            return Message::create(RequestType::GET_META_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::CLEAR_PARTITION: {
            auto cleared = controller_->clear_partition(body.partition_id, body.flag);

            // No local StorageManager to fall back on here (unlike the
            // colocated TransferQueueServer) -- every id's bytes live on a
            // remote StorageServer, so fan the clear out via a transient
            // RpcClient per shard actually used.
            std::unordered_map<std::int32_t, std::vector<SampleId>> ids_by_shard;
            for (std::size_t i = 0; i < cleared.sample_ids.size(); ++i) {
                std::int32_t shard = cleared.shard_indices[i];
                if (shard >= 0) {
                    ids_by_shard[shard].push_back(cleared.sample_ids[i]);
                }
            }
            // An unregistered/unreachable shard used to be silently skipped
            // here -- no error, success=true regardless, and the caller had
            // no way to know that shard's bytes were never actually freed
            // (Controller has already dropped its own metadata for those
            // ids by this point regardless, so there's no "undo"). Now
            // every shard is still attempted -- one shard's failure
            // (unresolved address, or the RPC itself throwing) no longer
            // aborts the rest of the fan-out -- and every shard that
            // couldn't be cleared is collected and surfaced to the caller.
            // See docs/UNIT_TEST_FINDINGS.md.
            std::vector<std::int32_t> unreachable_shards;
            for (const auto& [shard, ids] : ids_by_shard) {
                auto address = controller_->shard_address(shard);
                if (!address) {
                    unreachable_shards.push_back(shard);
                    continue;
                }
                try {
                    TransferQueueRpcClient shard_client(server_id(), *address);
                    shard_client.clear_data(body.partition_id, ids);
                } catch (const std::exception&) {
                    unreachable_shards.push_back(shard);
                }
            }

            MessageBody response_body;
            if (unreachable_shards.empty()) {
                response_body.success = true;
            } else {
                response_body.success = false;
                std::string message = "CLEAR_PARTITION: could not clear data on unregistered/unreachable shard(s): ";
                for (std::size_t i = 0; i < unreachable_shards.size(); ++i) {
                    if (i > 0) {
                        message += ", ";
                    }
                    message += std::to_string(unreachable_shards[i]);
                }
                response_body.error_message = message;
            }
            return Message::create(RequestType::CLEAR_PARTITION_RESPONSE, server_id(), response_body,
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

        case RequestType::RESET_CONSUMPTION: {
            controller_->reset_consumption(body.partition_id,
                                            body.task_name.empty() ? std::nullopt
                                                                    : std::optional<std::string>(body.task_name));

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::RESET_CONSUMPTION_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        default: {
            MessageBody response_body;
            response_body.success = false;
            response_body.error_message = "ControllerServer has no handler for request_type " +
                                           to_string(request.request_type);
            return Message::create(RequestType::REQUEST_ERROR, server_id(), response_body, request.sender_id);
        }
    }
}

}
