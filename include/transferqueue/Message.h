#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

// Request types this wire protocol actually carries. Scoped to exactly what
// Phase 0 implements (Controller/StorageManager), plus the handshake/error
// plumbing any client-server wiring needs. Mirrors upstream ZMQRequestType,
// minus KV/metrics/checkpoint (no backing feature yet) and CREATE_PARTITION
// (upstream doesn't have one either -- partition creation is implicit on
// first write, matching our Client::put()).
enum class RequestType {
    HANDSHAKE,
    HANDSHAKE_ACK,
    REQUEST_ERROR,

    PUT_DATA,
    PUT_DATA_RESPONSE,
    GET_DATA,
    GET_DATA_RESPONSE,
    CLEAR_DATA,
    CLEAR_DATA_RESPONSE,

    // Controller::update_production_status, triggered by a StorageManager
    // after it durably writes a PUT_DATA payload.
    NOTIFY_DATA_UPDATE,
    NOTIFY_DATA_UPDATE_ACK,
    NOTIFY_DATA_UPDATE_ERROR,

    // Controller::ready_indexes.
    GET_META,
    GET_META_RESPONSE,

    // Controller::clear_partition.
    CLEAR_PARTITION,
    CLEAR_PARTITION_RESPONSE,

    // Controller::reset_consumption.
    RESET_CONSUMPTION,
    RESET_CONSUMPTION_RESPONSE,

    // Controller::declare_schema (Option B schema validation, agreed with
    // the writer teams -- see docs/SCHEMA_VALIDATION.md).
    DECLARE_SCHEMA,
    DECLARE_SCHEMA_RESPONSE,

    // Controller::shard_address -- a shard (StorageServer) or a writer
    // resolving where to send data for a given shard_index. Phase 5
    // (multi-node/multi-shard): see docs/PHASE_5.md.
    GET_SHARD_ADDRESS,
    GET_SHARD_ADDRESS_RESPONSE,

    // Controller::advance_version -- the trainer calls this once after each
    // weight sync. Phase 6 (staleness/versioning): see docs/PHASE_6.md.
    ADVANCE_VERSION,
    ADVANCE_VERSION_RESPONSE,

    // Controller::find_stranded_groups -- a maintenance/observability call
    // (not part of the read path), reporting which GRPO groups have been
    // incomplete longer than a caller-given age threshold. Track-and-
    // expose only: never clears or modifies anything. See
    // docs/GRPO_GROUP_N_SAMPLER.md.
    FIND_STRANDED_GROUPS,
    FIND_STRANDED_GROUPS_RESPONSE,
};

std::string to_string(RequestType type);

// Control-plane fields (partition/sample/field addressing) are always
// serializable now. `payload` is an opaque byte blob for the actual field
// data carried by PUT_DATA / GET_DATA_RESPONSE -- real tensor (de)serialization
// is Phase 2 (retiring Record's std::any placeholder); this only proves the
// envelope round-trips correctly.
struct MessageBody {
    std::string partition_id;
    std::vector<SampleId> sample_ids;
    std::vector<std::string> fields;
    std::string task_name;
    std::uint32_t batch_size = 0; // GET_META request parameter

    // DECLARE_SCHEMA request parameter: parallel to `fields` (same
    // convention as sample_ids/partition_ids), field_dtypes[i] is the
    // declared dtype for fields[i].
    std::vector<FieldDtype> field_dtypes;

    // Generic boolean request parameter, meaning depends on request_type
    // (e.g. clear_consumption for CLEAR_PARTITION). Not used by every
    // request type; kept flat rather than a per-type struct since only one
    // request so far needs an extra bool.
    bool flag = false;

    // Response-only: whether the operation succeeded.
    bool success = true;
    std::string error_message;
    std::vector<std::uint8_t> payload;

    // Phase 5 (multi-node/multi-shard): HANDSHAKE (shard -> Controller,
    // "I am shard_index at address") and GET_SHARD_ADDRESS (either
    // direction) use these two directly.
    std::int32_t shard_index = -1;
    std::string address;

    // GET_META_RESPONSE / CLEAR_PARTITION_RESPONSE bundling (per the
    // decision to bundle rather than force a separate GET_SHARD_ADDRESS
    // round trip for reads): sample_shard_indices is parallel to
    // sample_ids, telling the caller which shard owns each one.
    // shard_registry_indices/_addresses is the resolved-address subset for
    // every distinct shard referenced above, so the caller doesn't need a
    // follow-up lookup for shards it hasn't already cached.
    std::vector<std::int32_t> sample_shard_indices;
    std::vector<std::int32_t> shard_registry_indices;
    std::vector<std::string> shard_registry_addresses;

    // Phase 6 (staleness/versioning, see docs/PHASE_6.md): GET_META_RESPONSE
    // bundles sample_versions (parallel to sample_ids, each sample's
    // produced-at version) and current_version (the version as of this
    // read), so a reader can compute staleness itself --
    // current_version - sample_versions[i]. ADVANCE_VERSION_RESPONSE also
    // uses current_version, for the new value after bumping. -1 is the "no
    // version recorded" sentinel, matching shard_index's convention.
    std::vector<std::int64_t> sample_versions;
    std::int64_t current_version = -1;

    // GRPOGroupNSampler's grouping key (see docs/GRPO_GROUP_N_SAMPLER.md):
    // PUT_DATA (colocated Server) and NOTIFY_DATA_UPDATE (split
    // ControllerServer) carry this so Controller::update_production_status
    // can record which real prompt-group the written ids belong to. Empty
    // string is the "no group_id" sentinel, matching shard_index's -1
    // convention (strings have no natural negative value).
    std::string group_id;

    // FIND_STRANDED_GROUPS request parameter: report groups whose oldest
    // member has been waiting at least this long.
    std::int64_t max_age_ms = 0;

    // FIND_STRANDED_GROUPS_RESPONSE: one entry per stranded group.
    // stranded_group_sizes is parallel to stranded_group_ids (how many
    // member ids that group has) and stranded_group_ages_ms is parallel
    // too (that group's oldest member's age in ms). The member ids
    // themselves are the flat concatenation of every stranded group's
    // members, in the same group order, reusing `sample_ids` above --
    // chunk it back into per-group lists using stranded_group_sizes.
    std::vector<std::string> stranded_group_ids;
    std::vector<std::int32_t> stranded_group_sizes;
    std::vector<std::int64_t> stranded_group_ages_ms;
};

// Mirrors upstream ZMQMessage.
struct Message {
    RequestType request_type;
    std::string sender_id;
    std::optional<std::string> receiver_id;
    std::string request_id;
    double timestamp;
    MessageBody body;

    static Message create(RequestType type, const std::string& sender_id, MessageBody body,
                           std::optional<std::string> receiver_id = std::nullopt);

    std::vector<std::uint8_t> serialize() const;
    static Message deserialize(const std::vector<std::uint8_t>& bytes);
};

}
