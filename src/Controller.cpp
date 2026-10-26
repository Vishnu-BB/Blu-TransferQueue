#include "transferqueue/Controller.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

namespace {
constexpr std::size_t kPreAllocSampleNum = 1;

std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
}

namespace tq {

bool TransferQueueController::ensure_partition_locked(const std::string& partition_id) {
    if (partitions_.find(partition_id) != partitions_.end()) {
        return false;
    }

    partitions_.emplace(std::piecewise_construct, std::forward_as_tuple(partition_id),
                         std::forward_as_tuple(partition_id));

    auto allocated = index_manager_.allocate_indexes(partition_id, kPreAllocSampleNum);
    partitions_.at(partition_id).register_pre_allocated_indexes(allocated);

    return true;
}

bool TransferQueueController::create_partition(const std::string& partition_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ensure_partition_locked(partition_id);
}

void TransferQueueController::declare_schema(const std::string& partition_id,
                                              const std::unordered_map<std::string, FieldDtype>& schema) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_partition_locked(partition_id);
    find_partition(partition_id)->declare_schema(schema);
}

void TransferQueueController::validate_schema(const std::string& partition_id,
                                               const std::unordered_map<std::string, FieldDtype>& field_dtypes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    for (const auto& [field, dtype] : field_dtypes) {
        partition->validate_field(field, dtype);
    }
}

DataPartitionStatus* TransferQueueController::find_partition(const std::string& partition_id) {
    auto it = partitions_.find(partition_id);
    return it == partitions_.end() ? nullptr : &it->second;
}

const DataPartitionStatus* TransferQueueController::find_partition(const std::string& partition_id) const {
    auto it = partitions_.find(partition_id);
    return it == partitions_.end() ? nullptr : &it->second;
}

bool TransferQueueController::update_production_status(const std::string& partition_id,
                                                         const std::vector<SampleId>& ids,
                                                         const std::vector<std::string>& fields,
                                                         std::optional<std::int32_t> shard_index,
                                                         std::optional<std::string> group_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return false;
    }
    partition->update_production_status(ids, fields, shard_index, current_version_, group_id);
    return true;
}

std::int32_t TransferQueueController::shard_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->shard_for_sample(id);
}

std::int64_t TransferQueueController::version_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->version_for_sample(id);
}

std::string TransferQueueController::group_id_for_sample(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return "";
    }
    return partition->group_id_for_sample(id);
}

std::int64_t TransferQueueController::produced_at(const std::string& partition_id, SampleId id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return -1;
    }
    return partition->produced_at(id);
}

std::int64_t TransferQueueController::advance_version() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ++current_version_;
}

std::int64_t TransferQueueController::current_version() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_version_;
}

void TransferQueueController::register_shard(std::int32_t shard_index, const std::string& address) {
    std::lock_guard<std::mutex> lock(mutex_);
    shard_registry_[shard_index] = address;
}

std::optional<std::string> TransferQueueController::shard_address(std::int32_t shard_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = shard_registry_.find(shard_index);
    if (it == shard_registry_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::unordered_map<std::int32_t, std::string> TransferQueueController::shard_registry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shard_registry_;
}

void TransferQueueController::mark_consumed(const std::string& partition_id, const std::string& task_name,
                                             const std::vector<SampleId>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    partition->mark_consumed(task_name, ids);
}

std::vector<SampleId> TransferQueueController::ready_indexes(const std::string& partition_id,
                                                               const std::vector<std::string>& fields,
                                                               const std::string& task_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    return partition->scan_data_status(fields, task_name);
}

std::vector<SampleId> TransferQueueController::select_and_consume(const std::string& partition_id,
                                                                    const std::vector<std::string>& fields,
                                                                    const std::string& task_name,
                                                                    BaseSampler& sampler, std::size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    auto ready = partition->scan_data_status(fields, task_name);

    std::vector<std::string> group_ids;
    group_ids.reserve(ready.size());
    for (SampleId id : ready) {
        group_ids.push_back(partition->group_id_for_sample(id));
    }

    auto [selected, remaining] = sampler.sample(std::move(ready), std::move(group_ids), batch_size);
    (void)remaining;
    if (!selected.empty()) {
        partition->mark_consumed(task_name, selected);
    }
    return selected;
}

std::vector<StrandedGroup> TransferQueueController::find_stranded_groups(const std::string& partition_id,
                                                                          const std::vector<std::string>& fields,
                                                                          const std::string& task_name,
                                                                          const BaseSampler& sampler,
                                                                          std::int64_t max_age_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    auto ready = partition->scan_data_status(fields, task_name);

    std::vector<std::string> group_ids;
    std::vector<std::int64_t> produced_at_ms;
    group_ids.reserve(ready.size());
    produced_at_ms.reserve(ready.size());
    for (SampleId id : ready) {
        group_ids.push_back(partition->group_id_for_sample(id));
        produced_at_ms.push_back(partition->produced_at(id));
    }

    return sampler.find_stranded(ready, group_ids, produced_at_ms, now_ms(), max_age_ms);
}

void TransferQueueController::reset_consumption(const std::string& partition_id,
                                                 const std::optional<std::string>& task_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return;
    }
    partition->reset_consumption(task_name);
}

TransferQueueController::ClearedSamples TransferQueueController::clear_partition(const std::string& partition_id,
                                                                                  bool clear_consumption) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    auto owned_indexes = partition->all_sample_ids();

    ClearedSamples result;
    result.sample_ids = owned_indexes;
    result.shard_indices.reserve(owned_indexes.size());
    for (SampleId id : owned_indexes) {
        result.shard_indices.push_back(partition->shard_for_sample(id));
    }

    partition->clear_data(owned_indexes, clear_consumption);
    index_manager_.release_partition(partition_id);
    partitions_.erase(partition_id);
    return result;
}

void TransferQueueController::clear_samples(const std::string& partition_id, const std::vector<SampleId>& ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (partition) {
        partition->clear_data(ids, /*clear_consumption=*/true);
    }
}

std::vector<SampleId> TransferQueueController::kv_retrieve(const std::string& partition_id,
                                                          const std::vector<std::string>& keys, bool create) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (create) {
        ensure_partition_locked(partition_id);
    }
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return std::vector<SampleId>(keys.size(), kMissingSample);
    }
    auto ids = partition->ids_for_keys(keys);
    if (!create) {
        return ids;
    }

    std::unordered_set<std::string> new_keys;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (ids[i] == kMissingSample) {
            new_keys.insert(keys[i]);
        }
    }
    std::size_t missing = new_keys.size();
    if (missing == 0) {
        return ids;
    }
    if (max_kv_samples_ > 0) {
        std::size_t live = kv_sample_count_locked();
        if (live + missing > max_kv_samples_) {
            throw std::runtime_error("TransferQueue full: " + std::to_string(live) + " samples held, " +
                                     std::to_string(missing) + " new key(s) would exceed total_storage_size=" +
                                     std::to_string(max_kv_samples_) + " -- clear consumed groups first");
        }
    }
    auto fresh = index_manager_.allocate_indexes(partition_id, missing);
    std::size_t next = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (ids[i] != kMissingSample) {
            continue;
        }
        // The same new key twice in one call must map to one id.
        auto dup = partition->ids_for_keys({keys[i]})[0];
        if (dup != kMissingSample) {
            ids[i] = dup;
            continue;
        }
        ids[i] = fresh[next++];
        partition->bind_key(keys[i], ids[i]);
    }
    if (next < fresh.size()) {
        index_manager_.release_indexes(partition_id, std::vector<SampleId>(fresh.begin() + next, fresh.end()));
    }
    return ids;
}

std::size_t TransferQueueController::kv_sample_count_locked() const {
    std::size_t total = 0;
    for (const auto& [id, partition] : partitions_) {
        (void)id;
        total += partition.key_count();
    }
    return total;
}

std::vector<std::pair<std::string, std::size_t>> TransferQueueController::kv_partition_counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<std::string, std::size_t>> out;
    for (const auto& [id, partition] : partitions_) {
        out.emplace_back(id, partition.key_count());
    }
    return out;
}

std::size_t TransferQueueController::kv_sample_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kv_sample_count_locked();
}

void TransferQueueController::kv_merge_tags(const std::string& partition_id, const std::vector<SampleId>& ids,
                                            const std::vector<Tag>& tags) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        throw std::invalid_argument("kv_merge_tags: unknown partition '" + partition_id + "'");
    }
    partition->merge_tags(ids, tags);
}

std::vector<std::pair<std::string, Tag>> TransferQueueController::kv_list(const std::string& partition_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto* partition = find_partition(partition_id);
    return partition ? partition->list_keys() : std::vector<std::pair<std::string, Tag>>{};
}

TransferQueueController::ClearedSamples TransferQueueController::kv_clear(const std::string& partition_id,
                                                                          const std::vector<std::string>& keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto* partition = find_partition(partition_id);
    if (!partition) {
        return {};
    }
    ClearedSamples result;
    for (SampleId id : partition->ids_for_keys(keys)) {
        if (id == kMissingSample) {
            continue;
        }
        result.sample_ids.push_back(id);
        result.shard_indices.push_back(partition->shard_for_sample(id));
    }
    if (result.sample_ids.empty()) {
        return result;
    }
    // ids_for_keys can repeat an id if a key is listed twice; release each once.
    std::vector<SampleId> unique_ids = result.sample_ids;
    std::sort(unique_ids.begin(), unique_ids.end());
    unique_ids.erase(std::unique(unique_ids.begin(), unique_ids.end()), unique_ids.end());
    partition->clear_data(unique_ids, /*clear_consumption=*/true);
    // The ids are NOT released here: their rows still sit on the storage shards until the caller
    // clears them (ControllerServer KV_CLEAR), then calls release_cleared(). Releasing first let a
    // concurrent kv_put reuse an id, write the new key's row into it, and have the pending shard
    // clear wipe that row (or merge into the old one) -- rows with missing or foreign fields.
    return result;
}

void TransferQueueController::release_cleared(const std::string& partition_id, std::vector<SampleId> ids) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (!ids.empty()) index_manager_.release_indexes(partition_id, ids);
}

namespace {
constexpr char kCheckpointMagic[4] = {'T', 'Q', 'C', 'K'};
constexpr std::uint32_t kCheckpointFormat = 2;  // 2: adds the shard index list

template <typename T>
void write_pod(std::ostream& os, const T& v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
template <typename T>
T read_pod(std::istream& is) {
    T v{};
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!is) throw std::runtime_error("load_kv_checkpoint: truncated checkpoint file");
    return v;
}
} // namespace

void TransferQueueController::save_kv_checkpoint(const std::string& path) const {
    // One record per (partition, produced-field set): prompt rows and
    // trajectory rows each collapse into a handful of records.
    std::vector<std::vector<std::uint8_t>> records;
    std::int64_t version = 0;
    std::vector<std::int32_t> shard_indices;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        version = current_version_;
        for (const auto& [shard, address] : shard_registry_) {
            (void)address;
            shard_indices.push_back(shard);
        }
        std::sort(shard_indices.begin(), shard_indices.end());
        for (const auto& [partition_id, partition] : partitions_) {
            std::map<std::vector<std::string>, MessageBody> groups;
            for (auto& snap : partition.kv_snapshot()) {
                auto& body = groups[snap.fields];
                body.partition_id = partition_id;
                body.fields = snap.fields;
                body.keys.push_back(std::move(snap.key));
                body.sample_ids.push_back(snap.id);
                body.tags.push_back(std::move(snap.tag));
                body.sample_shard_indices.push_back(snap.shard);
                body.sample_versions.push_back(snap.version);
            }
            for (auto& [fields, body] : groups) {
                (void)fields;
                records.push_back(Message::create(RequestType::KV_UPDATE, "checkpoint", std::move(body)).serialize());
            }
        }
    }

    // Write-then-rename so a crash mid-save never leaves a torn checkpoint.
    const std::string tmp = path + ".tmp";
    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("save_kv_checkpoint: cannot open " + tmp);
        os.write(kCheckpointMagic, 4);
        write_pod(os, kCheckpointFormat);
        write_pod(os, version);
        write_pod(os, static_cast<std::uint32_t>(shard_indices.size()));
        for (std::int32_t shard : shard_indices) write_pod(os, shard);
        write_pod(os, static_cast<std::uint32_t>(records.size()));
        for (const auto& rec : records) {
            write_pod(os, static_cast<std::uint64_t>(rec.size()));
            os.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(rec.size()));
        }
        if (!os.flush()) throw std::runtime_error("save_kv_checkpoint: write failed for " + tmp);
    }
    std::filesystem::rename(tmp, path);
}

TransferQueueController::KVCheckpoint TransferQueueController::parse_kv_checkpoint(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("load_kv_checkpoint: cannot open " + path);
    char magic[4];
    is.read(magic, 4);
    if (!is || std::memcmp(magic, kCheckpointMagic, 4) != 0) {
        throw std::runtime_error("load_kv_checkpoint: " + path + " is not a TransferQueue checkpoint");
    }
    if (read_pod<std::uint32_t>(is) != kCheckpointFormat) {
        throw std::runtime_error("load_kv_checkpoint: unsupported checkpoint format in " + path);
    }
    KVCheckpoint checkpoint;
    checkpoint.version = read_pod<std::int64_t>(is);
    const auto num_shards = read_pod<std::uint32_t>(is);
    for (std::uint32_t i = 0; i < num_shards; ++i) checkpoint.shard_indices.push_back(read_pod<std::int32_t>(is));
    const auto count = read_pod<std::uint32_t>(is);
    checkpoint.records.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto len = read_pod<std::uint64_t>(is);
        std::vector<std::uint8_t> bytes(len);
        is.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(len));
        if (!is) throw std::runtime_error("load_kv_checkpoint: truncated checkpoint file");
        checkpoint.records.push_back(Message::deserialize(bytes));
    }
    // Every shard a sample lives on must be one the checkpoint says it saved.
    const std::set<std::int32_t> saved(checkpoint.shard_indices.begin(), checkpoint.shard_indices.end());
    for (const auto& rec : checkpoint.records) {
        for (std::int32_t shard : rec.body.sample_shard_indices) {
            if (shard >= 0 && !saved.count(shard)) {
                throw std::runtime_error("load_kv_checkpoint: " + path + " references shard " + std::to_string(shard) +
                                         " that it did not save");
            }
        }
    }
    return checkpoint;
}

void TransferQueueController::apply_kv_checkpoint(KVCheckpoint checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [partition_id, partition] : partitions_) {
        (void)partition;
        index_manager_.release_partition(partition_id);
    }
    partitions_.clear();
    current_version_ = checkpoint.version;
    // Claim every restored id before any partition is created: creating one
    // pre-allocates an id, which must never be one we're about to restore.
    for (const auto& rec : checkpoint.records) {
        index_manager_.adopt_indexes(rec.body.partition_id, rec.body.sample_ids);
    }
    for (const auto& rec : checkpoint.records) {
        const auto& body = rec.body;
        ensure_partition_locked(body.partition_id);
        auto* partition = find_partition(body.partition_id);
        for (std::size_t i = 0; i < body.sample_ids.size(); ++i) {
            const SampleId id = body.sample_ids[i];
            partition->bind_key(body.keys[i], id);
            if (!body.fields.empty()) {
                partition->update_production_status(
                    {id}, body.fields,
                    body.sample_shard_indices[i] >= 0 ? std::optional<std::int32_t>(body.sample_shard_indices[i])
                                                      : std::nullopt,
                    body.sample_versions[i] >= 0 ? std::optional<std::int64_t>(body.sample_versions[i])
                                                 : std::nullopt);
            }
        }
        partition->merge_tags(body.sample_ids, body.tags);
    }
}

}
