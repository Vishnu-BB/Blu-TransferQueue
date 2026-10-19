#include "transferqueue/KVClient.h"

#include <algorithm>
#include <stdexcept>

#include "transferqueue/GroupRouter.h"

namespace tq {

KVClient::KVClient(std::string client_id, std::string controller_address, int timeout_ms)
    : controller_address_(controller_address),
      client_id_(std::move(client_id)),
      timeout_ms_(timeout_ms),
      controller_(client_id_, controller_address, timeout_ms) {}

TransferQueueRpcClient& KVClient::shard_client(std::int32_t shard, const std::string& address) {
    auto& client = shards_[shard];
    if (!client) {
        client = std::make_unique<TransferQueueRpcClient>(client_id_, address, timeout_ms_);
    }
    return *client;
}

void KVClient::kv_batch_put(const std::string& partition_id, const std::vector<std::string>& keys,
                            const std::vector<Record>& rows, const std::vector<Tag>& tags) {
    if (!rows.empty() && rows.size() != keys.size()) {
        throw std::invalid_argument("KVClient::kv_batch_put: rows must be empty or one per key");
    }
    if (!tags.empty() && tags.size() != keys.size()) {
        throw std::invalid_argument("KVClient::kv_batch_put: tags must be empty or one per key");
    }
    if (keys.empty()) {
        return;
    }

    std::vector<std::string> fields;
    std::vector<FieldDtype> dtypes;
    if (!rows.empty()) {
        for (const auto& [field, tensor] : rows.front()) {
            fields.push_back(field);
            dtypes.push_back(to_field_dtype(tensor.dtype()));
        }
        for (const auto& row : rows) {
            if (row.size() != fields.size() ||
                !std::all_of(fields.begin(), fields.end(), [&](const auto& f) { return row.count(f) != 0; })) {
                throw std::invalid_argument("KVClient::kv_batch_put: every row must carry the same fields");
            }
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto meta = controller_.kv_retrieve(partition_id, keys, /*create=*/true);
    std::vector<std::int32_t> shards = meta.sample_shard_indices;

    if (!fields.empty()) {
        std::vector<std::int32_t> registered;
        for (const auto& [shard, address] : meta.shard_addresses) {
            registered.push_back(shard);
        }
        std::sort(registered.begin(), registered.end());
        if (registered.empty()) {
            throw std::runtime_error("KVClient::kv_batch_put: no storage shards registered with the controller");
        }
        GroupRouter router(static_cast<int>(registered.size()));

        std::unordered_map<std::int32_t, std::unordered_map<SampleId, Record>> by_shard;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (shards[i] < 0) {
                shards[i] = registered[router.target_rank(keys[i].substr(0, keys[i].find('_')))];
            }
            by_shard[shards[i]][meta.sample_ids[i]] = rows[i];
        }
        for (const auto& [shard, data] : by_shard) {
            auto address = meta.shard_addresses.find(shard);
            if (address == meta.shard_addresses.end()) {
                throw std::runtime_error("KVClient::kv_batch_put: shard " + std::to_string(shard) +
                                         " holds existing keys but is no longer registered");
            }
            shard_client(shard, address->second).put(partition_id, fields, data);
        }
    }

    controller_.kv_update(partition_id, meta.sample_ids, fields, dtypes, fields.empty() ? std::vector<std::int32_t>{} : shards,
                          tags);
}

std::unordered_map<std::string, Record> KVClient::kv_batch_get(const std::string& partition_id,
                                                               const std::vector<std::string>& keys,
                                                               const std::vector<std::string>& select_fields) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto meta = controller_.kv_retrieve(partition_id, keys, /*create=*/false);

    std::unordered_map<std::string, Record> result;
    std::unordered_map<std::int32_t, std::vector<SampleId>> ids_by_shard;
    std::unordered_map<SampleId, std::vector<std::string>> keys_by_id;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (meta.sample_ids[i] == kMissingSample) {
            continue;
        }
        result[keys[i]];  // known key: present even if nothing is stored yet
        if (meta.sample_shard_indices[i] < 0) {
            continue;
        }
        auto& key_list = keys_by_id[meta.sample_ids[i]];
        if (key_list.empty()) {
            ids_by_shard[meta.sample_shard_indices[i]].push_back(meta.sample_ids[i]);
        }
        key_list.push_back(keys[i]);
    }

    for (const auto& [shard, ids] : ids_by_shard) {
        auto address = meta.shard_addresses.find(shard);
        if (address == meta.shard_addresses.end()) {
            throw std::runtime_error("KVClient::kv_batch_get: shard " + std::to_string(shard) + " is not registered");
        }
        for (auto& [id, record] : shard_client(shard, address->second).get_data(partition_id, ids, select_fields)) {
            for (const auto& key : keys_by_id[id]) {
                result[key] = record;
            }
        }
    }
    return result;
}

std::vector<std::pair<std::string, Tag>> KVClient::kv_list(const std::string& partition_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return controller_.kv_list(partition_id);
}

void KVClient::kv_clear(const std::string& partition_id, const std::vector<std::string>& keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    controller_.kv_clear(partition_id, keys);
}

namespace {
constexpr int kCheckpointTimeoutMs = 60 * 60 * 1000;  // a save moves every shard's data
}

void KVClient::save_checkpoint(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    TransferQueueRpcClient(client_id_, controller_address_, kCheckpointTimeoutMs).save_checkpoint(dir);
}

void KVClient::load_checkpoint(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    TransferQueueRpcClient(client_id_, controller_address_, kCheckpointTimeoutMs).load_checkpoint(dir);
}

}
