#pragma once

// KV interface over the split topology (one ControllerServer + N StorageServer
// shards) -- the C++ counterpart of upstream TransferQueue's kv_put/
// kv_batch_put/kv_batch_get/kv_list/kv_clear, which is all verl's v1 trainer
// uses.
//
//   put:  controller KV_RETRIEVE(create) -> PUT_DATA to each owning shard ->
//         controller KV_UPDATE (production status + merged tags). Data lands
//         before metadata, so no reader can see a field that isn't stored yet.
//   get:  controller KV_RETRIEVE -> GET_DATA from each shard holding the keys.
//
// New keys go to the shard picked by GroupRouter over the key's uid prefix
// (text before the first '_'), so every trajectory "{uid}_{session}_{i}" of a
// prompt sits on the same shard as its prompt row. A key keeps its shard for
// life: later partial-row writes (old_log_probs, advantages, ...) go wherever
// its first write went.

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "transferqueue/RpcClient.h"
#include "transferqueue/StorageManager.h"

namespace tq {

class KVClient {
public:
    KVClient(std::string client_id, std::string controller_address, int timeout_ms = 60000);

    // rows: one Record per key (all with the same field names), or empty for a
    // tags-only update. tags: one per key, or empty. Fields merge into the
    // existing row; tags merge into the existing tag.
    void kv_batch_put(const std::string& partition_id, const std::vector<std::string>& keys,
                      const std::vector<Record>& rows, const std::vector<Tag>& tags);

    // Unknown keys are omitted; a known key with no stored fields maps to an
    // empty Record. Empty select_fields = every stored field.
    std::unordered_map<std::string, Record> kv_batch_get(const std::string& partition_id,
                                                         const std::vector<std::string>& keys,
                                                         const std::vector<std::string>& select_fields);

    std::vector<std::pair<std::string, Tag>> kv_list(const std::string& partition_id);
    void kv_clear(const std::string& partition_id, const std::vector<std::string>& keys);

    // verl's tq.save_checkpoint / tq.load_checkpoint: the controller's KV
    // metadata plus every shard's rows, into/from directory `dir` (must be
    // reachable at the same path from every TQ node, e.g. a shared filesystem).
    // Load replaces the whole store and needs the same shard indices as the save.
    void save_checkpoint(const std::string& dir);
    void load_checkpoint(const std::string& dir);

private:
    std::string controller_address_;
    TransferQueueRpcClient& shard_client(std::int32_t shard, const std::string& address);

    // ponytail: one lock per client = one request in flight; give each thread
    // its own KVClient if a single client's throughput ever matters.
    std::mutex mutex_;
    std::string client_id_;
    int timeout_ms_;
    TransferQueueRpcClient controller_;
    std::unordered_map<std::int32_t, std::unique_ptr<TransferQueueRpcClient>> shards_;
};

}
