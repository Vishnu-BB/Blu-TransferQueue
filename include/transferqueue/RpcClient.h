#pragma once

#include <memory>
#include <optional>
#include <string>

#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

namespace tq {

class TransferQueueRpcClient {
public:
    struct Impl;

    // timeout_ms bounds every call; on timeout the socket is rebuilt so a late
    // reply can't be mistaken for the next call's answer.
    TransferQueueRpcClient(std::string client_id, const std::string& server_address, int timeout_ms = 5000);
    ~TransferQueueRpcClient();

    TransferQueueRpcClient(const TransferQueueRpcClient&) = delete;
    TransferQueueRpcClient& operator=(const TransferQueueRpcClient&) = delete;

    void handshake();

    void announce_shard(std::int32_t shard_index, const std::string& address);

    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    void put(const std::string& partition_id, const std::vector<std::string>& fields,
             const std::unordered_map<SampleId, Record>& data, const std::string& group_id = "");

    std::unordered_map<SampleId, Record> get(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, std::size_t batch_size);

    std::unordered_map<SampleId, Record> get_data(const std::string& partition_id,
                                                   const std::vector<SampleId>& sample_ids,
                                                   const std::vector<std::string>& fields);

    void clear_data(const std::string& partition_id, const std::vector<SampleId>& sample_ids);
    void clear_partition(const std::string& partition_id, bool clear_consumption = true);
    void reset_consumption(const std::string& partition_id,
                            const std::optional<std::string>& task_name = std::nullopt);

    void notify_data_update(const std::string& partition_id, const std::vector<SampleId>& sample_ids,
                             const std::vector<std::string>& fields, const std::vector<FieldDtype>& field_dtypes,
                             std::int32_t shard_index, const std::string& group_id = "");

    std::optional<std::string> get_shard_address(std::int32_t shard_index);

    struct MetaResult {
        std::vector<SampleId> sample_ids;
        std::vector<std::int32_t> sample_shard_indices; 
        std::unordered_map<std::int32_t, std::string> shard_addresses;
        std::vector<std::int64_t> sample_versions; 
        std::int64_t current_version = -1;
    };
    MetaResult get_meta(const std::string& partition_id, const std::vector<std::string>& fields,
                         const std::string& task_name, std::size_t batch_size);

    std::int64_t advance_version();

    std::vector<StrandedGroup> find_stranded_groups(const std::string& partition_id,
                                                     const std::vector<std::string>& fields,
                                                     const std::string& task_name, std::int64_t max_age_ms);

    // ---- KV layer (ControllerServer) ----
    struct KvRetrieveResult {
        std::vector<SampleId> sample_ids;               // kMissingSample for unknown keys (create=false)
        std::vector<std::int32_t> sample_shard_indices; // -1 = no data written yet
        std::unordered_map<std::int32_t, std::string> shard_addresses; // every registered shard
    };
    KvRetrieveResult kv_retrieve(const std::string& partition_id, const std::vector<std::string>& keys, bool create);

    // Marks `fields` produced for sample_ids (on the given per-sample shards)
    // and/or merges per-sample tags. Either fields or tags may be empty.
    void kv_update(const std::string& partition_id, const std::vector<SampleId>& sample_ids,
                   const std::vector<std::string>& fields, const std::vector<FieldDtype>& field_dtypes,
                   const std::vector<std::int32_t>& sample_shard_indices, const std::vector<Tag>& tags);

    std::vector<std::pair<std::string, Tag>> kv_list(const std::string& partition_id);
    void kv_clear(const std::string& partition_id, const std::vector<std::string>& keys);

    // Controller: path is a checkpoint directory; it fans out to the shards and
    // restores all-or-nothing. StorageServer: path is that shard's file, and
    // load_checkpoint only stages it -- commit_load_checkpoint() swaps it in,
    // abort_load_checkpoint() drops it.
    void save_checkpoint(const std::string& path);
    void load_checkpoint(const std::string& path);
    void commit_load_checkpoint();
    void abort_load_checkpoint();

    Tag get_stats();  // the server's counters/gauges (see RpcServerBase::request_stats)

private:
    std::unique_ptr<Impl> impl_;
    std::string client_id_;
};

}
