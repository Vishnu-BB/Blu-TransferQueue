#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "transferqueue/DataPartitionStatus.h"
#include "transferqueue/PartitionIndexManager.h"
#include "transferqueue/Sampler.h"

namespace tq {

class TransferQueueController {
public:
    // max_kv_samples: verl's transfer_queue.SimpleStorage.total_storage_size --
    // the most KV keys (prompt rows + trajectory rows, every partition) the
    // store holds at once. 0 = unlimited. kv_retrieve(create) refuses new keys
    // past it; clear consumed groups (kv_clear) to make room.
    explicit TransferQueueController(std::size_t max_kv_samples = 0) : max_kv_samples_(max_kv_samples) {}

    bool create_partition(const std::string& partition_id);

    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    void validate_schema(const std::string& partition_id,
                          const std::unordered_map<std::string, FieldDtype>& field_dtypes) const;

    bool update_production_status(const std::string& partition_id, const std::vector<SampleId>& ids,
                                   const std::vector<std::string>& fields,
                                   std::optional<std::int32_t> shard_index = std::nullopt,
                                   std::optional<std::string> group_id = std::nullopt);

    std::int32_t shard_for_sample(const std::string& partition_id, SampleId id) const;

    std::int64_t version_for_sample(const std::string& partition_id, SampleId id) const;

    std::string group_id_for_sample(const std::string& partition_id, SampleId id) const;

    std::int64_t produced_at(const std::string& partition_id, SampleId id) const;

    std::int64_t advance_version();
    std::int64_t current_version() const;

    void register_shard(std::int32_t shard_index, const std::string& address);
    std::optional<std::string> shard_address(std::int32_t shard_index) const;
    std::unordered_map<std::int32_t, std::string> shard_registry() const;

    void mark_consumed(const std::string& partition_id, const std::string& task_name,
                        const std::vector<SampleId>& ids);

    std::vector<SampleId> ready_indexes(const std::string& partition_id, const std::vector<std::string>& fields,
                                         const std::string& task_name) const;

    std::vector<SampleId> select_and_consume(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, BaseSampler& sampler,
                                              std::size_t batch_size);

    std::vector<StrandedGroup> find_stranded_groups(const std::string& partition_id,
                                                     const std::vector<std::string>& fields,
                                                     const std::string& task_name, const BaseSampler& sampler,
                                                     std::int64_t max_age_ms) const;

    void reset_consumption(const std::string& partition_id, const std::optional<std::string>& task_name = std::nullopt);

    struct ClearedSamples {
        std::vector<SampleId> sample_ids;
        std::vector<std::int32_t> shard_indices;
    };
    ClearedSamples clear_partition(const std::string& partition_id, bool clear_consumption = true);

    // Forgets individual samples (production, consumption, shard/version/tag,
    // key binding) without dropping the partition. Unknown ids are ignored.
    void clear_samples(const std::string& partition_id, const std::vector<SampleId>& ids);

    // ---- KV layer (verl's kv_* API). Keys are scoped per partition. ----
    // Maps keys to SampleIds. Unknown keys get a fresh id when `create` is true,
    // kMissingSample otherwise.
    std::vector<SampleId> kv_retrieve(const std::string& partition_id, const std::vector<std::string>& keys,
                                      bool create);
    void kv_merge_tags(const std::string& partition_id, const std::vector<SampleId>& ids, const std::vector<Tag>& tags);
    std::vector<std::pair<std::string, Tag>> kv_list(const std::string& partition_id) const;
    // Drops every trace of `keys` (production, consumption, tags, key binding)
    // and frees their ids; the caller still has to clear the returned samples'
    // data on their shards.
    ClearedSamples kv_clear(const std::string& partition_id, const std::vector<std::string>& keys);

    std::size_t kv_sample_count() const;  // live keys across every partition
    std::vector<std::pair<std::string, std::size_t>> kv_partition_counts() const;  // partition -> live keys

    // Checkpoint of the KV metadata (keys, ids, tags, shard, version, produced
    // fields per sample, plus the global version). Load replaces every
    // partition. Samples written through the non-KV put/get path are not
    // included. Callers pause writers around a save, as verl does at a step boundary.
    void save_kv_checkpoint(const std::string& path) const;
    void load_kv_checkpoint(const std::string& path);
    std::size_t max_kv_samples() const { return max_kv_samples_; }

private:
    DataPartitionStatus* find_partition(const std::string& partition_id);
    const DataPartitionStatus* find_partition(const std::string& partition_id) const;

    bool ensure_partition_locked(const std::string& partition_id);
    std::size_t kv_sample_count_locked() const;

    PartitionIndexManager index_manager_;
    std::unordered_map<std::string, DataPartitionStatus> partitions_;
    std::unordered_map<std::int32_t, std::string> shard_registry_;
    std::int64_t current_version_ = 0;
    std::size_t max_kv_samples_ = 0;

    mutable std::mutex mutex_;
};

}
