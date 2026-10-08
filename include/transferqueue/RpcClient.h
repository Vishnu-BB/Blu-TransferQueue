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

    TransferQueueRpcClient(std::string client_id, const std::string& server_address);
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

private:
    std::unique_ptr<Impl> impl_;
    std::string client_id_;
};

}
