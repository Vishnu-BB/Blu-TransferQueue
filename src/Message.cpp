#include "transferqueue/Message.h"

#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>

namespace tq {

std::string to_string(RequestType type) {
    switch (type) {
        case RequestType::HANDSHAKE: return "HANDSHAKE";
        case RequestType::HANDSHAKE_ACK: return "HANDSHAKE_ACK";
        case RequestType::REQUEST_ERROR: return "REQUEST_ERROR";
        case RequestType::PUT_DATA: return "PUT_DATA";
        case RequestType::PUT_DATA_RESPONSE: return "PUT_DATA_RESPONSE";
        case RequestType::GET_DATA: return "GET_DATA";
        case RequestType::GET_DATA_RESPONSE: return "GET_DATA_RESPONSE";
        case RequestType::CLEAR_DATA: return "CLEAR_DATA";
        case RequestType::CLEAR_DATA_RESPONSE: return "CLEAR_DATA_RESPONSE";
        case RequestType::NOTIFY_DATA_UPDATE: return "NOTIFY_DATA_UPDATE";
        case RequestType::NOTIFY_DATA_UPDATE_ACK: return "NOTIFY_DATA_UPDATE_ACK";
        case RequestType::NOTIFY_DATA_UPDATE_ERROR: return "NOTIFY_DATA_UPDATE_ERROR";
        case RequestType::GET_META: return "GET_META";
        case RequestType::GET_META_RESPONSE: return "GET_META_RESPONSE";
        case RequestType::CLEAR_PARTITION: return "CLEAR_PARTITION";
        case RequestType::CLEAR_PARTITION_RESPONSE: return "CLEAR_PARTITION_RESPONSE";
        case RequestType::RESET_CONSUMPTION: return "RESET_CONSUMPTION";
        case RequestType::RESET_CONSUMPTION_RESPONSE: return "RESET_CONSUMPTION_RESPONSE";
        case RequestType::DECLARE_SCHEMA: return "DECLARE_SCHEMA";
        case RequestType::DECLARE_SCHEMA_RESPONSE: return "DECLARE_SCHEMA_RESPONSE";
        case RequestType::GET_SHARD_ADDRESS: return "GET_SHARD_ADDRESS";
        case RequestType::GET_SHARD_ADDRESS_RESPONSE: return "GET_SHARD_ADDRESS_RESPONSE";
        case RequestType::ADVANCE_VERSION: return "ADVANCE_VERSION";
        case RequestType::ADVANCE_VERSION_RESPONSE: return "ADVANCE_VERSION_RESPONSE";
        case RequestType::FIND_STRANDED_GROUPS: return "FIND_STRANDED_GROUPS";
        case RequestType::FIND_STRANDED_GROUPS_RESPONSE: return "FIND_STRANDED_GROUPS_RESPONSE";
    }
    throw std::invalid_argument("to_string(RequestType): unknown value");
}

namespace {

// Minimal length-prefixed binary encoding. No external dependency (matches
// upstream's intent -- a compact wire format -- without pulling in msgpack
// for a protocol that, this phase, doesn't yet carry real tensor payloads).
void write_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
}

std::uint8_t read_u8(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    if (offset + 1 > size) {
        throw std::invalid_argument("Message::deserialize: truncated u8");
    }
    return data[offset++];
}

// Bounds-checked: an out-of-range byte used to be silently cast into
// RequestType anyway, constructing a Message with an undefined enum value
// that only failed later (and only indirectly) if something happened to
// call to_string() on it. Now it fails fast at the deserialize boundary
// instead of drifting downstream as a half-valid Message. See
// docs/UNIT_TEST_FINDINGS.md.
RequestType read_request_type(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint8_t value = read_u8(data, size, offset);
    if (value > static_cast<std::uint8_t>(RequestType::FIND_STRANDED_GROUPS_RESPONSE)) {
        throw std::invalid_argument("Message::deserialize: invalid request_type byte");
    }
    return static_cast<RequestType>(value);
}

std::uint32_t read_u32(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    if (offset + 4 > size) {
        throw std::invalid_argument("Message::deserialize: truncated u32");
    }
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= static_cast<std::uint32_t>(data[offset + i]) << (8 * i);
    }
    offset += 4;
    return v;
}

// Two's-complement bit pattern preserved through the uint32 round trip, so
// -1 (our "no shard" sentinel) round-trips correctly.
void write_i32(std::vector<std::uint8_t>& out, std::int32_t v) { write_u32(out, static_cast<std::uint32_t>(v)); }

std::int32_t read_i32(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    return static_cast<std::int32_t>(read_u32(data, size, offset));
}

void write_i32_list(std::vector<std::uint8_t>& out, const std::vector<std::int32_t>& items) {
    write_u32(out, static_cast<std::uint32_t>(items.size()));
    for (std::int32_t item : items) {
        write_i32(out, item);
    }
}

std::vector<std::int32_t> read_i32_list(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t count = read_u32(data, size, offset);
    std::vector<std::int32_t> items;
    items.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        items.push_back(read_i32(data, size, offset));
    }
    return items;
}

void write_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
}

std::uint64_t read_u64(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    if (offset + 8 > size) {
        throw std::invalid_argument("Message::deserialize: truncated u64");
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    return v;
}

// Two's-complement bit pattern preserved through the uint64 round trip, so
// -1 (our "no version recorded" sentinel) round-trips correctly.
void write_i64(std::vector<std::uint8_t>& out, std::int64_t v) { write_u64(out, static_cast<std::uint64_t>(v)); }

std::int64_t read_i64(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    return static_cast<std::int64_t>(read_u64(data, size, offset));
}

void write_i64_list(std::vector<std::uint8_t>& out, const std::vector<std::int64_t>& items) {
    write_u32(out, static_cast<std::uint32_t>(items.size()));
    for (std::int64_t item : items) {
        write_i64(out, item);
    }
}

std::vector<std::int64_t> read_i64_list(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t count = read_u32(data, size, offset);
    std::vector<std::int64_t> items;
    items.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        items.push_back(read_i64(data, size, offset));
    }
    return items;
}

void write_string(std::vector<std::uint8_t>& out, const std::string& s) {
    write_u32(out, static_cast<std::uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

std::string read_string(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t len = read_u32(data, size, offset);
    if (offset + len > size) {
        throw std::invalid_argument("Message::deserialize: truncated string");
    }
    std::string s(reinterpret_cast<const char*>(data + offset), len);
    offset += len;
    return s;
}

// SampleId is std::uint64_t (BatchMeta.h), so this is just write_u64/read_u64
// applied per element -- reuses them instead of hand-rolling the identical
// 8-byte loop again (was pure duplication, zero behavioral difference; see
// docs/UNIT_TEST_FINDINGS.md).
void write_sample_ids(std::vector<std::uint8_t>& out, const std::vector<SampleId>& ids) {
    write_u32(out, static_cast<std::uint32_t>(ids.size()));
    for (SampleId id : ids) {
        write_u64(out, id);
    }
}

std::vector<SampleId> read_sample_ids(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t count = read_u32(data, size, offset);
    std::vector<SampleId> ids;
    ids.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        ids.push_back(read_u64(data, size, offset));
    }
    return ids;
}

void write_string_list(std::vector<std::uint8_t>& out, const std::vector<std::string>& items) {
    write_u32(out, static_cast<std::uint32_t>(items.size()));
    for (const auto& item : items) {
        write_string(out, item);
    }
}

std::vector<std::string> read_string_list(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t count = read_u32(data, size, offset);
    std::vector<std::string> items;
    items.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        items.push_back(read_string(data, size, offset));
    }
    return items;
}

void write_field_dtypes(std::vector<std::uint8_t>& out, const std::vector<FieldDtype>& items) {
    write_u32(out, static_cast<std::uint32_t>(items.size()));
    for (FieldDtype item : items) {
        out.push_back(static_cast<std::uint8_t>(item));
    }
}

// Bounds-checked: an out-of-range byte used to be silently cast into
// FieldDtype anyway (undefined-enum-value territory), failing open rather
// than failing fast at the deserialize boundary. See docs/UNIT_TEST_FINDINGS.md.
FieldDtype read_field_dtype(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint8_t value = read_u8(data, size, offset);
    if (value > static_cast<std::uint8_t>(FieldDtype::Float4_e2m1_2x)) {
        throw std::invalid_argument("Message::deserialize: invalid FieldDtype byte");
    }
    return static_cast<FieldDtype>(value);
}

std::vector<FieldDtype> read_field_dtypes(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t count = read_u32(data, size, offset);
    std::vector<FieldDtype> items;
    items.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        items.push_back(read_field_dtype(data, size, offset));
    }
    return items;
}

void write_bytes(std::vector<std::uint8_t>& out, const std::vector<std::uint8_t>& bytes) {
    write_u32(out, static_cast<std::uint32_t>(bytes.size()));
    out.insert(out.end(), bytes.begin(), bytes.end());
}

std::vector<std::uint8_t> read_bytes(const std::uint8_t* data, std::size_t size, std::size_t& offset) {
    std::uint32_t len = read_u32(data, size, offset);
    if (offset + len > size) {
        throw std::invalid_argument("Message::deserialize: truncated payload");
    }
    std::vector<std::uint8_t> bytes(data + offset, data + offset + len);
    offset += len;
    return bytes;
}

std::string make_request_id() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    static const char* kHex = "0123456789abcdef";
    std::string id(8, '0');
    std::uint64_t v = rng();
    for (auto& c : id) {
        c = kHex[v & 0xF];
        v >>= 4;
    }
    return id;
}

} // namespace

Message Message::create(RequestType type, const std::string& sender_id, MessageBody body,
                         std::optional<std::string> receiver_id) {
    Message msg;
    msg.request_type = type;
    msg.sender_id = sender_id;
    msg.receiver_id = std::move(receiver_id);
    msg.request_id = make_request_id();
    msg.timestamp = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    msg.body = std::move(body);
    return msg;
}

std::vector<std::uint8_t> Message::serialize() const {
    std::vector<std::uint8_t> out;
    out.push_back(static_cast<std::uint8_t>(request_type));
    write_string(out, sender_id);
    out.push_back(receiver_id.has_value() ? 1 : 0);
    if (receiver_id.has_value()) {
        write_string(out, *receiver_id);
    }
    write_string(out, request_id);

    static_assert(sizeof(double) == 8, "Message::serialize assumes 8-byte double");
    std::uint64_t ts_bits;
    std::memcpy(&ts_bits, &timestamp, sizeof(ts_bits));
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(ts_bits >> (8 * i)));
    }

    write_string(out, body.partition_id);
    write_sample_ids(out, body.sample_ids);
    write_string_list(out, body.fields);
    write_field_dtypes(out, body.field_dtypes);
    write_string(out, body.task_name);
    write_u32(out, body.batch_size);
    out.push_back(body.flag ? 1 : 0);
    out.push_back(body.success ? 1 : 0);
    write_string(out, body.error_message);
    write_bytes(out, body.payload);
    write_i32(out, body.shard_index);
    write_string(out, body.address);
    write_i32_list(out, body.sample_shard_indices);
    write_i32_list(out, body.shard_registry_indices);
    write_string_list(out, body.shard_registry_addresses);
    write_i64_list(out, body.sample_versions);
    write_i64(out, body.current_version);
    write_string(out, body.group_id);
    write_i64(out, body.max_age_ms);
    write_string_list(out, body.stranded_group_ids);
    write_i32_list(out, body.stranded_group_sizes);
    write_i64_list(out, body.stranded_group_ages_ms);

    return out;
}

Message Message::deserialize(const std::vector<std::uint8_t>& bytes) {
    if (bytes.empty()) {
        throw std::invalid_argument("Message::deserialize: empty buffer");
    }

    const std::uint8_t* data = bytes.data();
    std::size_t size = bytes.size();
    std::size_t offset = 0;

    Message msg;
    msg.request_type = read_request_type(data, size, offset);
    msg.sender_id = read_string(data, size, offset);

    bool has_receiver = read_u8(data, size, offset) != 0;
    if (has_receiver) {
        msg.receiver_id = read_string(data, size, offset);
    } else {
        msg.receiver_id = std::nullopt;
    }

    msg.request_id = read_string(data, size, offset);

    if (offset + 8 > size) {
        throw std::invalid_argument("Message::deserialize: truncated timestamp");
    }
    std::uint64_t ts_bits = 0;
    for (int i = 0; i < 8; ++i) {
        ts_bits |= static_cast<std::uint64_t>(data[offset + i]) << (8 * i);
    }
    offset += 8;
    std::memcpy(&msg.timestamp, &ts_bits, sizeof(ts_bits));

    msg.body.partition_id = read_string(data, size, offset);
    msg.body.sample_ids = read_sample_ids(data, size, offset);
    msg.body.fields = read_string_list(data, size, offset);
    msg.body.field_dtypes = read_field_dtypes(data, size, offset);
    msg.body.task_name = read_string(data, size, offset);
    msg.body.batch_size = read_u32(data, size, offset);
    msg.body.flag = read_u8(data, size, offset) != 0;
    msg.body.success = read_u8(data, size, offset) != 0;
    msg.body.error_message = read_string(data, size, offset);
    msg.body.payload = read_bytes(data, size, offset);
    msg.body.shard_index = read_i32(data, size, offset);
    msg.body.address = read_string(data, size, offset);
    msg.body.sample_shard_indices = read_i32_list(data, size, offset);
    msg.body.shard_registry_indices = read_i32_list(data, size, offset);
    msg.body.shard_registry_addresses = read_string_list(data, size, offset);
    msg.body.sample_versions = read_i64_list(data, size, offset);
    msg.body.current_version = read_i64(data, size, offset);
    msg.body.group_id = read_string(data, size, offset);
    msg.body.max_age_ms = read_i64(data, size, offset);
    msg.body.stranded_group_ids = read_string_list(data, size, offset);
    msg.body.stranded_group_sizes = read_i32_list(data, size, offset);
    msg.body.stranded_group_ages_ms = read_i64_list(data, size, offset);

    return msg;
}

}
