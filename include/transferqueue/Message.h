#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

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

    NOTIFY_DATA_UPDATE,
    NOTIFY_DATA_UPDATE_ACK,
    NOTIFY_DATA_UPDATE_ERROR,

    GET_META,
    GET_META_RESPONSE,

    CLEAR_PARTITION,
    CLEAR_PARTITION_RESPONSE,

    RESET_CONSUMPTION,
    RESET_CONSUMPTION_RESPONSE,

    DECLARE_SCHEMA,
    DECLARE_SCHEMA_RESPONSE,

    GET_SHARD_ADDRESS,
    GET_SHARD_ADDRESS_RESPONSE,

    ADVANCE_VERSION,
    ADVANCE_VERSION_RESPONSE,

    FIND_STRANDED_GROUPS,
    FIND_STRANDED_GROUPS_RESPONSE,
};

std::string to_string(RequestType type);

struct MessageBody {
    std::string partition_id;
    std::vector<SampleId> sample_ids;
    std::vector<std::string> fields;
    std::string task_name;
    std::uint32_t batch_size = 0; 
    std::vector<FieldDtype> field_dtypes;

    bool flag = false;

    bool success = true;
    std::string error_message;
    std::vector<std::uint8_t> payload;

    std::int32_t shard_index = -1;
    std::string address;

    std::vector<std::int32_t> sample_shard_indices;
    std::vector<std::int32_t> shard_registry_indices;
    std::vector<std::string> shard_registry_addresses;

    std::vector<std::int64_t> sample_versions;
    std::int64_t current_version = -1;

    std::string group_id;

    std::int64_t max_age_ms = 0;

    std::vector<std::string> stranded_group_ids;
    std::vector<std::int32_t> stranded_group_sizes;
    std::vector<std::int64_t> stranded_group_ages_ms;
};

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

    // Split-frame wire form: serialize_header() encodes everything except
    // body.payload (which can be large -- a whole batch's tensor data) as
    // its own small buffer, suitable for sending as one ZMQ frame while
    // body.payload goes out as a second, separate frame with no copy
    // merging the two together. The header alone still decodes correctly
    // via the regular deserialize() above (it encodes a zero-length payload
    // in that slot, exactly like a message that legitimately carries no
    // payload today) -- deserialize_split() just does that, then moves the
    // separately-received payload bytes into place. Used by
    // RpcClient.cpp/RpcServerBase.cpp (the real network path); serialize()/
    // deserialize() above remain the single-buffer form everything else
    // (tests, make_batch_message/extract_batch's in-memory use) continues
    // to use unchanged.
    std::vector<std::uint8_t> serialize_header() const;
    static Message deserialize_split(const std::vector<std::uint8_t>& header_bytes,
                                      std::vector<std::uint8_t> payload_bytes);
};

}
