#include "transferqueue/ControllerServer.h"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "transferqueue/RpcClient.h"

namespace tq {

namespace {

// Sends CLEAR_DATA for each cleared sample to the shard holding it. Returns
// an error message naming every shard that couldn't be reached ("" if none).
std::string clear_on_shards(TransferQueueController& controller, const std::string& sender_id,
                            const std::string& partition_id,
                            const TransferQueueController::ClearedSamples& cleared) {
    std::unordered_map<std::int32_t, std::vector<SampleId>> ids_by_shard;
    for (std::size_t i = 0; i < cleared.sample_ids.size(); ++i) {
        std::int32_t shard = cleared.shard_indices[i];
        if (shard >= 0) {
            ids_by_shard[shard].push_back(cleared.sample_ids[i]);
        }
    }
    std::string unreachable;
    for (const auto& [shard, ids] : ids_by_shard) {
        auto address = controller.shard_address(shard);
        try {
            if (!address) {
                throw std::runtime_error("unregistered");
            }
            TransferQueueRpcClient shard_client(sender_id, *address);
            shard_client.clear_data(partition_id, ids);
        } catch (const std::exception&) {
            unreachable += (unreachable.empty() ? "" : ", ") + std::to_string(shard);
        }
    }
    return unreachable.empty() ? "" : "could not clear data on unregistered/unreachable shard(s): " + unreachable;
}

// Shard checkpoint/stats calls can move a whole shard's data; give them far
// longer than the default 5 s request timeout.
constexpr int kShardCheckpointTimeoutMs = 30 * 60 * 1000;
constexpr int kShardStatsTimeoutMs = 2000;

std::string shard_checkpoint_path(const std::string& dir, std::int32_t shard) {
    return (std::filesystem::path(dir) / ("shard-" + std::to_string(shard) + ".tqsh")).string();
}

std::string controller_checkpoint_path(const std::string& dir) {
    return (std::filesystem::path(dir) / "controller.tqck").string();
}

} // namespace

ControllerServer::~ControllerServer() {
    if (metrics_) {
        metrics_->stop();
    }
}

int ControllerServer::start_metrics(int port) {
    metrics_ = std::make_unique<MetricsHttpServer>([this] { return render_metrics(); });
    return metrics_->start(port);
}

std::string ControllerServer::render_metrics() {
    std::ostringstream out;
    auto value_str = [](const TagValue& v) -> std::string {
        if (auto* i = std::get_if<std::int64_t>(&v)) return std::to_string(*i);
        if (auto* d = std::get_if<double>(&v)) return std::to_string(*d);
        if (auto* b = std::get_if<bool>(&v)) return *b ? "1" : "0";
        return "";
    };
    // request_stats() keys are "requests:<TYPE>" / "request_seconds:<TYPE>".
    auto emit_requests = [&](const std::string& server, const Tag& stats) {
        for (const auto& [name, value] : stats) {
            auto colon = name.find(':');
            if (colon == std::string::npos) continue;
            std::string metric = name.substr(0, colon) == "requests" ? "tq_requests_total" : "tq_request_seconds_total";
            out << metric << "{server=\"" << server << "\",type=\"" << name.substr(colon + 1) << "\"} "
                << value_str(value) << "\n";
        }
    };

    out << "# TYPE tq_samples gauge\ntq_samples " << controller_->kv_sample_count() << "\n";
    out << "# TYPE tq_capacity_samples gauge\ntq_capacity_samples " << controller_->max_kv_samples() << "\n";
    out << "# TYPE tq_partition_samples gauge\n";
    for (const auto& [partition, count] : controller_->kv_partition_counts()) {
        out << "tq_partition_samples{partition=\"" << partition << "\"} " << count << "\n";
    }
    out << "# TYPE tq_policy_version gauge\ntq_policy_version " << controller_->current_version() << "\n";
    auto shards = controller_->shard_registry();
    out << "# TYPE tq_shards_registered gauge\ntq_shards_registered " << shards.size() << "\n";
    out << "# TYPE tq_requests_total counter\n# TYPE tq_request_seconds_total counter\n";
    emit_requests("controller", request_stats());

    std::ostringstream shard_gauges;
    for (const auto& [shard, address] : shards) {
        const std::string label = "{shard=\"" + std::to_string(shard) + "\"}";
        try {
            TransferQueueRpcClient client(server_id(), address, kShardStatsTimeoutMs);
            Tag stats = client.get_stats();
            shard_gauges << "tq_shard_up" << label << " 1\n";
            for (const char* gauge : {"bytes", "rows", "capacity_bytes"}) {
                if (auto it = stats.find(gauge); it != stats.end()) {
                    shard_gauges << "tq_shard_" << gauge << label << " " << value_str(it->second) << "\n";
                }
            }
            emit_requests("shard-" + std::to_string(shard), stats);
        } catch (const std::exception&) {
            shard_gauges << "tq_shard_up" << label << " 0\n";
        }
    }
    out << "# TYPE tq_shard_up gauge\n# TYPE tq_shard_bytes gauge\n# TYPE tq_shard_rows gauge\n"
           "# TYPE tq_shard_capacity_bytes gauge\n"
        << shard_gauges.str();
    return out.str();
}

ControllerServer::ControllerServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                                    std::shared_ptr<BaseSampler> sampler)
    : RpcServerBase(std::move(server_id)), controller_(std::move(controller)), sampler_(std::move(sampler)) {}

Message ControllerServer::handle_request(const Message& request) {
    const auto& body = request.body;

    switch (request.request_type) {
        case RequestType::HANDSHAKE: {
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
            controller_->create_partition(body.partition_id);

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
            auto selected =
                controller_->select_and_consume(body.partition_id, body.fields, body.task_name, *sampler_,
                                                 body.batch_size);

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

            response_body.sample_versions.reserve(selected.size());
            for (SampleId id : selected) {
                response_body.sample_versions.push_back(controller_->version_for_sample(body.partition_id, id));
            }
            response_body.current_version = controller_->current_version();

            return Message::create(RequestType::GET_META_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::CLEAR_PARTITION: {
            auto cleared = controller_->clear_partition(body.partition_id, body.flag);
            auto error = clear_on_shards(*controller_, server_id(), body.partition_id, cleared);

            MessageBody response_body;
            response_body.success = error.empty();
            if (!error.empty()) {
                response_body.error_message = "CLEAR_PARTITION: " + error;
            }
            return Message::create(RequestType::CLEAR_PARTITION_RESPONSE, server_id(), response_body,
                                    request.sender_id);
        }

        case RequestType::KV_RETRIEVE: {
            MessageBody response_body;
            response_body.sample_ids = controller_->kv_retrieve(body.partition_id, body.keys, body.flag);
            response_body.sample_shard_indices.reserve(response_body.sample_ids.size());
            for (SampleId id : response_body.sample_ids) {
                response_body.sample_shard_indices.push_back(
                    id == kMissingSample ? -1 : controller_->shard_for_sample(body.partition_id, id));
            }
            // Every registered shard, not just the ones these ids live on: a
            // writer needs the full map to place brand-new keys.
            for (const auto& [shard, address] : controller_->shard_registry()) {
                response_body.shard_registry_indices.push_back(shard);
                response_body.shard_registry_addresses.push_back(address);
            }
            response_body.success = true;
            return Message::create(RequestType::KV_RETRIEVE_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::KV_UPDATE: {
            controller_->create_partition(body.partition_id);
            if (!body.fields.empty()) {
                std::unordered_map<std::string, FieldDtype> field_dtypes;
                for (std::size_t i = 0; i < body.fields.size() && i < body.field_dtypes.size(); ++i) {
                    field_dtypes[body.fields[i]] = body.field_dtypes[i];
                }
                controller_->validate_schema(body.partition_id, field_dtypes);

                // One production-status update per shard (normally one or two per batch).
                std::unordered_map<std::int32_t, std::vector<SampleId>> ids_by_shard;
                for (std::size_t i = 0; i < body.sample_ids.size(); ++i) {
                    std::int32_t shard = i < body.sample_shard_indices.size() ? body.sample_shard_indices[i] : -1;
                    ids_by_shard[shard].push_back(body.sample_ids[i]);
                }
                for (const auto& [shard, ids] : ids_by_shard) {
                    controller_->update_production_status(
                        body.partition_id, ids, body.fields,
                        shard >= 0 ? std::optional<std::int32_t>(shard) : std::nullopt);
                }
            }
            if (!body.tags.empty()) {
                controller_->kv_merge_tags(body.partition_id, body.sample_ids, body.tags);
            }

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::KV_UPDATE_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::KV_LIST: {
            MessageBody response_body;
            for (auto& [key, tag] : controller_->kv_list(body.partition_id)) {
                response_body.keys.push_back(std::move(key));
                response_body.tags.push_back(std::move(tag));
            }
            response_body.success = true;
            return Message::create(RequestType::KV_LIST_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::SAVE_CHECKPOINT: {
            // body.address = checkpoint directory: controller metadata + one file per shard.
            std::filesystem::create_directories(body.address);
            controller_->save_kv_checkpoint(controller_checkpoint_path(body.address));
            for (const auto& [shard, address] : controller_->shard_registry()) {
                TransferQueueRpcClient(server_id(), address, kShardCheckpointTimeoutMs)
                    .save_checkpoint(shard_checkpoint_path(body.address, shard));
            }
            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::SAVE_CHECKPOINT_RESPONSE, server_id(), response_body,
                                   request.sender_id);
        }

        case RequestType::LOAD_CHECKPOINT: {
            // Shards first: if any data file is missing or bad, the controller's
            // metadata is left untouched. The checkpoint must come from a
            // deployment with the same shard indices.
            for (const auto& [shard, address] : controller_->shard_registry()) {
                const auto path = shard_checkpoint_path(body.address, shard);
                if (!std::filesystem::exists(path)) {
                    throw std::runtime_error("LOAD_CHECKPOINT: missing " + path +
                                             " (checkpoint taken with a different shard count?)");
                }
                TransferQueueRpcClient(server_id(), address, kShardCheckpointTimeoutMs).load_checkpoint(path);
            }
            controller_->load_kv_checkpoint(controller_checkpoint_path(body.address));
            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::LOAD_CHECKPOINT_RESPONSE, server_id(), response_body,
                                   request.sender_id);
        }

        case RequestType::GET_STATS: {
            MessageBody response_body;
            response_body.success = true;
            response_body.tags.push_back(request_stats());
            return Message::create(RequestType::GET_STATS_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::KV_CLEAR: {
            // Metadata first, so no reader can select these samples while
            // their data is being removed from the shards.
            auto cleared = controller_->kv_clear(body.partition_id, body.keys);
            auto error = clear_on_shards(*controller_, server_id(), body.partition_id, cleared);

            MessageBody response_body;
            response_body.success = error.empty();
            if (!error.empty()) {
                response_body.error_message = "KV_CLEAR: " + error;
            }
            return Message::create(RequestType::KV_CLEAR_RESPONSE, server_id(), response_body, request.sender_id);
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
