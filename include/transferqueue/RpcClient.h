#pragma once

#include <memory>
#include <optional>
#include <string>

#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

namespace tq {

// DEALER-socket RPC client: the network-calling counterpart to
// TransferQueueClient's in-process put()/get(). Same method shapes,
// different transport -- a rollout engine, agent loop, or training engine
// wiring into TransferQueue over the network uses this, not
// TransferQueueClient directly.
//
// NOT thread-safe: a single instance wraps one ZMQ DEALER socket, and ZMQ
// sockets must only ever be used from the thread that owns them -- calling
// any method concurrently from multiple threads on the SAME instance is
// undefined behavior (e.g. two threads' requests interleaving on the wire,
// or a thread reading the reply meant for another thread's call), not
// something this class detects or guards against. Give each thread its own
// instance, or serialize all access to a shared one externally (e.g. one
// mutex held around every call) -- this class does neither for you.
class TransferQueueRpcClient {
public:
    // Forward-declared here (not private) only so RpcClient.cpp's free
    // helper functions can reference the type by name -- the definition
    // itself stays in RpcClient.cpp, so callers outside it still can't do
    // anything with an incomplete type.
    struct Impl;

    TransferQueueRpcClient(std::string client_id, const std::string& server_address);
    ~TransferQueueRpcClient();

    TransferQueueRpcClient(const TransferQueueRpcClient&) = delete;
    TransferQueueRpcClient& operator=(const TransferQueueRpcClient&) = delete;

    void handshake();

    // Phase 5: a StorageServer announces itself to the ControllerServer
    // once at startup -- "I am shard_index at address" -- via HANDSHAKE
    // carrying those two fields. A plain handshake() (above) leaves them at
    // their defaults, so the ControllerServer's HANDSHAKE handler only
    // registers a shard when both are actually set.
    void announce_shard(std::int32_t shard_index, const std::string& address);

    // Option B, upfront registration -- see Controller::declare_schema.
    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    // `group_id`, if given, is recorded for every sample in `data` --
    // GRPOGroupNSampler's grouping key (see docs/GRPO_GROUP_N_SAMPLER.md).
    // Against the colocated TransferQueueServer (not a StorageServer --
    // see StorageServer.h, which has no Controller to record this
    // against), this reaches Controller directly from the PUT_DATA
    // handler.
    void put(const std::string& partition_id, const std::vector<std::string>& fields,
             const std::unordered_map<SampleId, Record>& data, const std::string& group_id = "");

    std::unordered_map<SampleId, Record> get(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, std::size_t batch_size);

    // Phase 5: direct GET_DATA against a StorageServer by sample id -- the
    // second hop after get_meta() resolves which shard(s) hold them.
    // Unlike get(), no sampling/consumption tracking happens here (a
    // StorageServer has no Controller to track that with); the caller
    // already knows exactly which ids it wants.
    std::unordered_map<SampleId, Record> get_data(const std::string& partition_id,
                                                   const std::vector<SampleId>& sample_ids,
                                                   const std::vector<std::string>& fields);

    void clear_data(const std::string& partition_id, const std::vector<SampleId>& sample_ids);
    void clear_partition(const std::string& partition_id, bool clear_consumption = true);
    void reset_consumption(const std::string& partition_id,
                            const std::optional<std::string>& task_name = std::nullopt);

    // Phase 5: told to a ControllerServer after a direct write to a
    // StorageServer (see StorageServer.h) -- the colocated TransferQueueServer
    // updates production status itself inside its own PUT_DATA handler, so
    // this is only ever needed against a split ControllerServer/StorageServer
    // deployment. field_dtypes is parallel to fields, matching
    // MessageBody's own convention. `group_id`, same meaning as put()'s --
    // this is the real write path for a split deployment, so this is where
    // group_id actually reaches the Controller in that topology.
    void notify_data_update(const std::string& partition_id, const std::vector<SampleId>& sample_ids,
                             const std::vector<std::string>& fields, const std::vector<FieldDtype>& field_dtypes,
                             std::int32_t shard_index, const std::string& group_id = "");

    // Phase 5: resolves a shard_index to its StorageServer address via a
    // ControllerServer. Mostly superseded by get_meta()'s bundled shard
    // addresses for reads -- this is for the write side, where a writer
    // needs an address before it has anything to read yet. nullopt if the
    // shard was never registered.
    std::optional<std::string> get_shard_address(std::int32_t shard_index);

    // Phase 5: GET_META against a ControllerServer (no StorageManager, so no
    // payload) -- which samples were selected, which shard each one lives
    // on, and the resolved address for every distinct shard referenced, so
    // the caller can fetch the actual data directly from the shard(s)
    // without a separate GET_SHARD_ADDRESS round trip per shard.
    // Phase 6 (staleness/versioning, see docs/PHASE_6.md): sample_versions
    // and current_version are bundled the same way shard info is --
    // tracking only, the caller decides what (if anything) to do about a
    // stale sample.
    struct MetaResult {
        std::vector<SampleId> sample_ids;
        std::vector<std::int32_t> sample_shard_indices; // parallel to sample_ids
        std::unordered_map<std::int32_t, std::string> shard_addresses;
        std::vector<std::int64_t> sample_versions; // parallel to sample_ids
        std::int64_t current_version = -1;
    };
    MetaResult get_meta(const std::string& partition_id, const std::vector<std::string>& fields,
                         const std::string& task_name, std::size_t batch_size);

    // Phase 6: trainer calls this once after each weight sync. Returns the
    // new current_version.
    std::int64_t advance_version();

    // Maintenance/observability only -- track-and-expose, see
    // docs/GRPO_GROUP_N_SAMPLER.md. Against a server whose sampler has no
    // concept of grouping (e.g. FifoSampler), always returns empty
    // (BaseSampler::find_stranded's default). NEVER clears or modifies
    // anything server-side; the caller decides what to do with the report.
    std::vector<StrandedGroup> find_stranded_groups(const std::string& partition_id,
                                                     const std::vector<std::string>& fields,
                                                     const std::string& task_name, std::int64_t max_age_ms);

private:
    std::unique_ptr<Impl> impl_;
    std::string client_id_;
};

}
