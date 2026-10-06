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

// Control-plane orchestrator: owns one PartitionIndexManager (shared index
// space across partitions) and a partition_id -> DataPartitionStatus map,
// routing every call to the right partition. Mirrors upstream
// TransferQueueController, scoped down: no ZMQ server, metrics, checkpoint
// save/load, or KV retrieve -- those are separate, unscaffolded concerns.
//
// Thread-safe: one mutex held for the duration of every public method,
// not just the map lookup. This serializes all partitions behind one lock
// rather than letting independent partitions proceed concurrently --
// correct and simple, at a known throughput ceiling. The alternative
// (per-partition locking only) risks a dangling DataPartitionStatus*: a
// pointer handed out by find_partition() could be erased by a concurrent
// clear_partition() on another thread before its caller is done with it.
// Revisit with a proper reference-counted/erase-safe scheme if this lock
// becomes a measured bottleneck, not before.
class TransferQueueController {
public:
    bool create_partition(const std::string& partition_id);

    // Option B schema validation: ensures `partition_id` exists (creates it
    // if this is the very first call for it -- "upfront" doesn't require a
    // separate create_partition() call first), then declares `schema`.
    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    // Throws on a dtype mismatch against a declared schema. No-op if the
    // partition doesn't exist (matches every other method's convention
    // here) or if a field was never declared (gradual adoption).
    void validate_schema(const std::string& partition_id,
                          const std::unordered_map<std::string, FieldDtype>& field_dtypes) const;

    // `shard_index`: Phase 5's multi-shard case, recording which
    // StorageServer actually holds `ids` (arrives via a remote
    // NOTIFY_DATA_UPDATE). Omitted for the colocated case -- every
    // pre-Phase-5 caller (Client::put(), the colocated TransferQueueServer).
    //
    // Every call also stamps `ids` with current_version_ (Phase 6 staleness
    // tracking, see docs/PHASE_6.md) -- not a parameter here, since the
    // Controller is the single source of truth for "what version is it
    // right now," not the caller. `group_id`, if given, records which real
    // prompt-group `ids` belong to -- GRPOGroupNSampler's authoritative
    // grouping key (see docs/GRPO_GROUP_N_SAMPLER.md).
    bool update_production_status(const std::string& partition_id, const std::vector<SampleId>& ids,
                                   const std::vector<std::string>& fields,
                                   std::optional<std::int32_t> shard_index = std::nullopt,
                                   std::optional<std::string> group_id = std::nullopt);

    // -1 if the partition doesn't exist or the sample was never recorded
    // with a shard (colocated case).
    std::int32_t shard_for_sample(const std::string& partition_id, SampleId id) const;

    // -1 if the partition doesn't exist or the sample predates version
    // tracking (produced before Phase 6).
    std::int64_t version_for_sample(const std::string& partition_id, SampleId id) const;

    // "" if the partition doesn't exist or the sample was never recorded
    // with a group_id.
    std::string group_id_for_sample(const std::string& partition_id, SampleId id) const;

    // -1 if the partition doesn't exist or the sample is unknown.
    std::int64_t produced_at(const std::string& partition_id, SampleId id) const;

    // Phase 6 (staleness/versioning, see docs/PHASE_6.md): one global
    // policy-version counter, not per-partition -- a single GRPO policy is
    // shared across every partition. The trainer calls advance_version()
    // once after each weight sync; every update_production_status() call
    // stamps samples with whatever this was at that moment, so staleness
    // (current_version() - version_for_sample(...)) is always computable
    // later. Tracking only -- nothing here filters or rejects based on it;
    // see docs/PHASE_6.md for why enforcement was deliberately left out of
    // v1.
    std::int64_t advance_version();
    std::int64_t current_version() const;

    // Phase 5 shard registry: a StorageServer announces itself once (via
    // HANDSHAKE), readers/writers resolve its address from shard_index
    // afterward (via GET_SHARD_ADDRESS, or bundled into GET_META_RESPONSE
    // for reads -- see docs/PHASE_5.md). Global, not per-partition: one
    // shard can hold data for many different partitions.
    void register_shard(std::int32_t shard_index, const std::string& address);
    std::optional<std::string> shard_address(std::int32_t shard_index) const;

    void mark_consumed(const std::string& partition_id, const std::string& task_name,
                        const std::vector<SampleId>& ids);

    // Samples in `partition_id` with every field in `fields` produced and
    // not yet consumed by `task_name`.
    std::vector<SampleId> ready_indexes(const std::string& partition_id, const std::vector<std::string>& fields,
                                         const std::string& task_name) const;

    // Atomically reads ready_indexes(), samples from them via `sampler`,
    // and marks whatever was selected consumed -- all under one lock
    // acquisition. GET_META handlers (Server.cpp, ControllerServer.cpp)
    // must use this instead of calling ready_indexes()/mark_consumed()
    // separately: those three steps used to run as independent locked
    // calls, so two concurrent GET_META requests (real once a server
    // processes requests on a thread pool -- exactly what Phase 7's
    // multiple-trainer-rank topology does) could both read the same ready
    // set before either marked anything consumed, and both receive the
    // same sample(s). Found and fixed while building Phase 7's N:M test;
    // see docs/PHASE_7.md. Also resolves each ready id's group_id (one
    // lookup per id -- cheap relative to the sort every group-aware
    // sampler already does) and passes both parallel arrays to
    // `sampler.sample()`, so BaseSampler's own grouping logic (e.g.
    // GRPOGroupNSampler) has access to the real grouping key instead of
    // guessing from id adjacency. Empty if the partition doesn't exist.
    std::vector<SampleId> select_and_consume(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, BaseSampler& sampler,
                                              std::size_t batch_size);

    // Read-only counterpart to select_and_consume() for maintenance/
    // observability rather than the read path: resolves ready_indexes()
    // plus each one's group_id and produced_at, and asks `sampler` which
    // groups among them are stranded (see BaseSampler::find_stranded).
    // NEVER consumes, clears, or mutates anything -- track-and-expose only,
    // per the confirmed decision (docs/GRPO_GROUP_N_SAMPLER.md). Empty if
    // the partition doesn't exist.
    std::vector<StrandedGroup> find_stranded_groups(const std::string& partition_id,
                                                     const std::vector<std::string>& fields,
                                                     const std::string& task_name, const BaseSampler& sampler,
                                                     std::int64_t max_age_ms) const;

    void reset_consumption(const std::string& partition_id, const std::optional<std::string>& task_name = std::nullopt);

    // sample_ids: what was cleared. shard_indices: parallel array, which
    // shard each one lived on (-1 for the colocated case). The caller uses
    // these to also clear the corresponding storage entries -- Controller
    // has no StorageManager reference itself, so it can't do that part on
    // its own. For a single-shard/colocated server, every id goes to the
    // same StorageManager; for Phase 5's multi-shard case, ids can span
    // several shards and the caller (ControllerServer) groups by
    // shard_indices to fan the clear out to each one. Both empty if the
    // partition doesn't exist.
    struct ClearedSamples {
        std::vector<SampleId> sample_ids;
        std::vector<std::int32_t> shard_indices;
    };
    ClearedSamples clear_partition(const std::string& partition_id, bool clear_consumption = true);

private:
    DataPartitionStatus* find_partition(const std::string& partition_id);
    const DataPartitionStatus* find_partition(const std::string& partition_id) const;

    // Lock-free: caller already holds mutex_. Shared by create_partition()
    // and declare_schema() -- both need "create if it doesn't exist yet"
    // without calling back into a public method and deadlocking on the
    // same non-recursive mutex.
    bool ensure_partition_locked(const std::string& partition_id);

    PartitionIndexManager index_manager_;
    std::unordered_map<std::string, DataPartitionStatus> partitions_;
    std::unordered_map<std::int32_t, std::string> shard_registry_;
    std::int64_t current_version_ = 0;

    mutable std::mutex mutex_;
};

}
