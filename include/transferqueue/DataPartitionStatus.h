#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

// Production/consumption tracking for one partition. Mirrors upstream
// DataPartitionStatus, scoped down: no shape/nested-tensor metadata
// (upstream's FieldMeta tracks more than just dtype) since BatchMeta's full
// field_schema is still deferred, and no numpy-array capacity
// preallocation (a perf optimization, not a correctness requirement --
// hash sets don't need it). Dtype-only schema validation (Option B, agreed
// with the other writer teams) is in scope, since that's the concrete need
// that came up -- see declare_schema()/validate_field().
//
// Thread-safe: one mutex guards all member access. Needed once
// TransferQueueServer dispatches requests across a thread pool instead of
// processing them one at a time on a single thread (see Server.h) --
// concurrent PUT/GET/consumption calls for the same partition are now a
// real possibility, not just a hypothetical.
class DataPartitionStatus {
public:
    explicit DataPartitionStatus(std::string partition_id) : partition_id_(std::move(partition_id)) {}

    void register_pre_allocated_indexes(const std::vector<SampleId>& indexes);

    // Option B, upfront registration: declares this partition's field
    // dtypes once (typically by whoever produces the first sample).
    // Re-declaring an already-declared field with a *different* dtype
    // throws -- the schema itself must stay consistent even before any
    // data is written. Fields never declared here are simply never
    // validated (gradual adoption: a partition that never calls this
    // behaves exactly as before Option B existed).
    void declare_schema(const std::unordered_map<std::string, FieldDtype>& schema);

    // Throws if `field` has a declared dtype and `dtype` doesn't match it.
    // No-op if `field` was never declared.
    void validate_field(const std::string& field, FieldDtype dtype) const;

    // Marks `fields` as produced for `ids`. `shard_index`, if given, records
    // which shard (StorageServer) actually holds these samples -- Phase 5's
    // multi-shard case, where this call arrives via a remote
    // NOTIFY_DATA_UPDATE rather than a local, colocated write. Omitted
    // (nullopt) for the single-shard/colocated case -- matches every
    // existing caller (Client::put(), the colocated TransferQueueServer).
    // `policy_version`, if given, records which policy version was current
    // when these samples were produced (Phase 6 staleness tracking, see
    // docs/PHASE_6.md) -- always supplied by Controller (which stamps its
    // own current_version()), nullopt only for a direct/unit-test call.
    // `group_id`, if given, records which real prompt-group `ids` belong to
    // (e.g. "prompt-42") -- GRPOGroupNSampler's authoritative grouping key
    // (see docs/GRPO_GROUP_N_SAMPLER.md's "silent cross-group
    // contamination" fix). Omitted for partitions that don't use group-
    // aware sampling.
    void update_production_status(const std::vector<SampleId>& ids, const std::vector<std::string>& fields,
                                   std::optional<std::int32_t> shard_index = std::nullopt,
                                   std::optional<std::int64_t> policy_version = std::nullopt,
                                   std::optional<std::string> group_id = std::nullopt);

    // -1 if `id` was never recorded with a shard (colocated case, or
    // unknown id).
    std::int32_t shard_for_sample(SampleId id) const;

    // -1 if `id` was never recorded with a version (produced before Phase 6
    // tracking existed, or unknown id).
    std::int64_t version_for_sample(SampleId id) const;

    // "" if `id` was never recorded with a group_id -- a sample like this
    // can never be selected by GRPOGroupNSampler (see its class doc).
    std::string group_id_for_sample(SampleId id) const;

    // Wall-clock epoch-ms when `id` was FIRST produced (first-write-wins --
    // later update_production_status calls for an already-known id don't
    // move this), always stamped internally by update_production_status()
    // itself (no caller-supplied parameter, same reasoning as Phase 6's
    // policy_version stamping: this is "when did this call happen," which
    // only the callee can answer correctly). -1 if `id` is unknown. Backs
    // age-based stranded-group reporting -- see BaseSampler::find_stranded.
    std::int64_t produced_at(SampleId id) const;

    void mark_consumed(const std::string& task_name, const std::vector<SampleId>& ids);
    bool has_consumed(const std::string& task_name, SampleId id) const;
    void reset_consumption(const std::optional<std::string>& task_name = std::nullopt);

    // Sample is ready iff every field in `fields` is produced for it AND it
    // hasn't been consumed by `task_name` yet.
    std::vector<SampleId> scan_data_status(const std::vector<std::string>& fields, const std::string& task_name) const;

    // Removes samples from production/consumption tracking entirely.
    void clear_data(const std::vector<SampleId>& ids, bool clear_consumption = true);

    std::size_t total_samples_num() const;

    // Every sample id this partition has ever seen (global_indexes_, kept
    // current by update_production_status on every real write, union
    // pre_allocated_indexes_ for ones reserved but never produced). This is
    // the actual source of truth for "what does this partition contain" --
    // PartitionIndexManager only tracks the pre-allocated placeholder it
    // mints itself; real sample ids are chosen by callers and never flow
    // through it, so it can't answer this question.
    std::vector<SampleId> all_sample_ids() const;

private:
    // Lock-free: callers already hold mutex_.
    bool is_produced_locked(SampleId id, const std::vector<std::string>& fields) const;
    bool has_consumed_locked(const std::string& task_name, SampleId id) const;

    std::string partition_id_;
    std::unordered_set<SampleId> global_indexes_;
    std::unordered_set<SampleId> pre_allocated_indexes_;
    std::unordered_map<std::string, std::unordered_set<SampleId>> production_by_field_;
    std::unordered_map<std::string, std::unordered_set<SampleId>> consumption_by_task_;
    std::unordered_map<std::string, FieldDtype> declared_schema_;
    std::unordered_map<SampleId, std::int32_t> sample_shard_;
    std::unordered_map<SampleId, std::int64_t> sample_version_;
    std::unordered_map<SampleId, std::string> sample_group_id_;
    std::unordered_map<SampleId, std::int64_t> sample_produced_at_;

    mutable std::mutex mutex_;
};

}
