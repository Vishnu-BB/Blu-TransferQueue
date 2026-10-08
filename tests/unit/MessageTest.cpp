// Unit tests for the wire protocol (include/transferqueue/Message.h,
// src/Message.cpp): Message/MessageBody/RequestType, serialize()/
// deserialize(), to_string(RequestType), Message::create.
//
// Zero dependencies -- build with `make unit_tests` or directly:
//   g++ -std=c++2a -Iinclude src/BatchMeta.cpp src/Message.cpp \
//       tests/unit/MessageTest.cpp -o /tmp/t && /tmp/t

#include "transferqueue/Message.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

#include "TestUtils.h"

namespace {

using tq::FieldDtype;
using tq::Message;
using tq::MessageBody;
using tq::RequestType;

// Every RequestType value declared in Message.h, kept in sync by hand (no
// reflection available) -- the to_string distinctness check below at least
// catches an internal duplicate/typo even if this list silently misses a
// newly-added enumerator.
const std::vector<RequestType> kAllRequestTypes = {
    RequestType::HANDSHAKE,
    RequestType::HANDSHAKE_ACK,
    RequestType::REQUEST_ERROR,
    RequestType::PUT_DATA,
    RequestType::PUT_DATA_RESPONSE,
    RequestType::GET_DATA,
    RequestType::GET_DATA_RESPONSE,
    RequestType::CLEAR_DATA,
    RequestType::CLEAR_DATA_RESPONSE,
    RequestType::NOTIFY_DATA_UPDATE,
    RequestType::NOTIFY_DATA_UPDATE_ACK,
    RequestType::NOTIFY_DATA_UPDATE_ERROR,
    RequestType::GET_META,
    RequestType::GET_META_RESPONSE,
    RequestType::CLEAR_PARTITION,
    RequestType::CLEAR_PARTITION_RESPONSE,
    RequestType::RESET_CONSUMPTION,
    RequestType::RESET_CONSUMPTION_RESPONSE,
    RequestType::DECLARE_SCHEMA,
    RequestType::DECLARE_SCHEMA_RESPONSE,
    RequestType::GET_SHARD_ADDRESS,
    RequestType::GET_SHARD_ADDRESS_RESPONSE,
    RequestType::ADVANCE_VERSION,
    RequestType::ADVANCE_VERSION_RESPONSE,
    RequestType::FIND_STRANDED_GROUPS,
    RequestType::FIND_STRANDED_GROUPS_RESPONSE,
};

// Every FieldDtype value declared in BatchMeta.h, same caveat as above.
const std::vector<FieldDtype> kAllFieldDtypes = {
    FieldDtype::Int8,      FieldDtype::Int16,      FieldDtype::Int32,        FieldDtype::Int64,
    FieldDtype::UInt8,     FieldDtype::UInt16,     FieldDtype::UInt32,       FieldDtype::UInt64,
    FieldDtype::Bfloat16,  FieldDtype::Float16,    FieldDtype::Float32,     FieldDtype::Float64,
    FieldDtype::Bool,      FieldDtype::Complex32,  FieldDtype::Complex64,    FieldDtype::Complex128,
    FieldDtype::Float4_e2m1, FieldDtype::Float4_e2m1_2x,
};

MessageBody make_fully_populated_body() {
    MessageBody body;
    body.partition_id = "rollout@shard-7";
    body.sample_ids = {0, 1, 2, UINT64_MAX - 1, UINT64_MAX};
    body.fields = {"prompt_ids", "response_ids", "log_probs", "reward"};
    body.field_dtypes = kAllFieldDtypes;
    body.task_name = "trainer-rank-3";
    body.batch_size = UINT32_MAX;
    body.flag = true;
    body.success = false;
    body.error_message = "simulated failure: shard unreachable";
    body.payload = {0xDE, 0xAD, 0x00, 0xBE, 0xEF, 0x00, 0x00, 0x01, 0xFF};
    body.shard_index = INT32_MAX;
    body.address = "tcp://10.0.0.5:5555";
    body.sample_shard_indices = {-1, 0, INT32_MAX, INT32_MIN};
    body.shard_registry_indices = {0, 1, INT32_MAX};
    body.shard_registry_addresses = {"tcp://a:1", "tcp://b:2", "tcp://c:3"};
    body.sample_versions = {-1, 0, INT64_MAX, INT64_MIN, 42};
    body.current_version = INT64_MAX;
    return body;
}

void check_body_equal(const MessageBody& a, const MessageBody& b) {
    CHECK(a.partition_id == b.partition_id);
    CHECK(a.sample_ids == b.sample_ids);
    CHECK(a.fields == b.fields);
    CHECK(a.field_dtypes == b.field_dtypes);
    CHECK(a.task_name == b.task_name);
    CHECK(a.batch_size == b.batch_size);
    CHECK(a.flag == b.flag);
    CHECK(a.success == b.success);
    CHECK(a.error_message == b.error_message);
    CHECK(a.payload == b.payload);
    CHECK(a.shard_index == b.shard_index);
    CHECK(a.address == b.address);
    CHECK(a.sample_shard_indices == b.sample_shard_indices);
    CHECK(a.shard_registry_indices == b.shard_registry_indices);
    CHECK(a.shard_registry_addresses == b.shard_registry_addresses);
    CHECK(a.sample_versions == b.sample_versions);
    CHECK(a.current_version == b.current_version);
}

} // namespace

int main() {
    // ---- 1. Exhaustive round-trip: every field populated at once, with
    // extreme/boundary values for every integer field (full 64-bit range
    // for sample_ids, every FieldDtype, INT32_MIN/MAX, INT64_MIN/MAX, the
    // -1 sentinels, UINT32_MAX for batch_size, an embedded-0x00 payload). ----
    {
        MessageBody body = make_fully_populated_body();
        auto msg = Message::create(RequestType::GET_META_RESPONSE, "sender-1", body, "receiver-1");
        auto bytes = msg.serialize();
        auto decoded = Message::deserialize(bytes);

        CHECK(decoded.request_type == RequestType::GET_META_RESPONSE);
        CHECK(decoded.sender_id == "sender-1");
        CHECK(decoded.receiver_id.has_value() && *decoded.receiver_id == "receiver-1");
        CHECK(decoded.request_id == msg.request_id);
        CHECK(decoded.timestamp == msg.timestamp);
        check_body_equal(decoded.body, body);

        // field_dtypes doesn't need to correspond 1:1 in length with fields
        // at the wire level -- Message doesn't cross-validate them (that's
        // Controller/DataPartitionStatus's job). Confirm it round-trips
        // with mismatched lengths without throwing.
        CHECK(body.fields.size() != body.field_dtypes.size());
    }

    // flag/success: both booleans independently, both states, in case a
    // copy-paste bug ever ties them to the same bit.
    {
        for (bool flag : {true, false}) {
            for (bool success : {true, false}) {
                MessageBody body;
                body.flag = flag;
                body.success = success;
                auto msg = Message::create(RequestType::CLEAR_PARTITION, "s", body);
                auto decoded = Message::deserialize(msg.serialize());
                CHECK(decoded.body.flag == flag);
                CHECK(decoded.body.success == success);
            }
        }
    }

    // ---- 2. Every RequestType value: round-trips, and to_string() gives a
    // non-empty, pairwise-distinct string for every single one. ----
    {
        std::set<std::string> names;
        for (RequestType type : kAllRequestTypes) {
            auto msg = Message::create(type, "s", MessageBody{});
            auto decoded = Message::deserialize(msg.serialize());
            CHECK(decoded.request_type == type);

            std::string name = tq::to_string(type);
            CHECK(!name.empty());
            names.insert(name);
        }
        CHECK(names.size() == kAllRequestTypes.size());
    }

    // ---- 3. receiver_id: nullopt round-trips as nullopt (not ""), a real
    // value round-trips as that value. ----
    {
        auto no_receiver = Message::create(RequestType::HANDSHAKE, "s", MessageBody{});
        CHECK(!no_receiver.receiver_id.has_value());
        auto decoded = Message::deserialize(no_receiver.serialize());
        CHECK(!decoded.receiver_id.has_value());

        auto with_receiver = Message::create(RequestType::HANDSHAKE, "s", MessageBody{}, "controller-1");
        auto decoded2 = Message::deserialize(with_receiver.serialize());
        CHECK(decoded2.receiver_id.has_value());
        CHECK(*decoded2.receiver_id == "controller-1");

        // Empty-string receiver_id is a real, present value -- must stay
        // distinct from nullopt after the round trip.
        auto empty_receiver = Message::create(RequestType::HANDSHAKE, "s", MessageBody{}, "");
        auto decoded3 = Message::deserialize(empty_receiver.serialize());
        CHECK(decoded3.receiver_id.has_value());
        CHECK(decoded3.receiver_id->empty());
    }

    // ---- 4. Empty strings for every string field must round-trip as
    // empty, not desync the fields that follow them on the wire. ----
    {
        MessageBody body;
        body.partition_id = "";
        body.task_name = "";
        body.error_message = "";
        body.address = "";
        // Non-empty fields around them to prove nothing after an empty
        // string got shifted/misread.
        body.sample_ids = {7};
        body.shard_index = 3;

        auto msg = Message::create(RequestType::GET_META, "", body);
        auto decoded = Message::deserialize(msg.serialize());
        CHECK(decoded.sender_id.empty());
        CHECK(decoded.body.partition_id.empty());
        CHECK(decoded.body.task_name.empty());
        CHECK(decoded.body.error_message.empty());
        CHECK(decoded.body.address.empty());
        CHECK((decoded.body.sample_ids == std::vector<tq::SampleId>{7}));
        CHECK(decoded.body.shard_index == 3);
    }

    // ---- 5. Empty vectors for every vector field must round-trip as
    // empty (size 0), with a sentinel field after them intact. ----
    {
        MessageBody body; // every vector field defaults to empty already
        body.current_version = 99; // sentinel placed after every vector field on the wire
        auto msg = Message::create(RequestType::GET_META_RESPONSE, "s", body);
        auto decoded = Message::deserialize(msg.serialize());
        CHECK(decoded.body.sample_ids.empty());
        CHECK(decoded.body.fields.empty());
        CHECK(decoded.body.field_dtypes.empty());
        CHECK(decoded.body.payload.empty());
        CHECK(decoded.body.sample_shard_indices.empty());
        CHECK(decoded.body.shard_registry_indices.empty());
        CHECK(decoded.body.shard_registry_addresses.empty());
        CHECK(decoded.body.sample_versions.empty());
        CHECK(decoded.body.current_version == 99);
    }

    // ---- 6. Truncation fuzz sweep: every prefix length of a fully
    // populated, valid message must either successfully decode the FULL
    // message (only at the exact full length) or throw
    // std::invalid_argument -- never anything else, never "succeed" with a
    // short/garbage result. ----
    {
        MessageBody body = make_fully_populated_body();
        auto msg = Message::create(RequestType::NOTIFY_DATA_UPDATE, "sender", body, "receiver");
        auto full = msg.serialize();
        CHECK(full.size() > 50); // sanity: this is actually a substantial message, worth truncating

        int ok_at_full_length_only = 0;
        for (std::size_t len = 0; len < full.size(); ++len) {
            std::vector<std::uint8_t> prefix(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(len));
            bool threw_invalid_argument = false;
            bool threw_other = false;
            bool succeeded = false;
            try {
                Message::deserialize(prefix);
                succeeded = true;
            } catch (const std::invalid_argument&) {
                threw_invalid_argument = true;
            } catch (...) {
                threw_other = true;
            }
            if (succeeded) ++ok_at_full_length_only;
            CHECK(!threw_other);
            CHECK(threw_invalid_argument || succeeded);
            // A truncated prefix must never silently "succeed" -- every
            // length strictly less than the full message is a genuine
            // truncation and must throw.
            CHECK(!succeeded);
        }
        // The full-length buffer itself must decode successfully (checked
        // elsewhere too, just confirming the sweep's own boundary).
        CHECK_NOTHROW(Message::deserialize(full));
        CHECK(ok_at_full_length_only == 0); // every length < full.size() failed, as asserted above
    }

    // ---- 7. FIXED (was a real robustness gap -- see
    // docs/UNIT_TEST_FINDINGS.md): deserialize() used to accept any
    // request_type byte unchecked, constructing a Message with an
    // out-of-range value that only failed later (and only indirectly) if
    // something called to_string() on it. It now validates the byte at the
    // deserialize boundary itself and throws immediately, before a
    // half-valid Message is ever constructed. ----
    {
        MessageBody body;
        auto msg = Message::create(RequestType::HANDSHAKE, "s", body);
        auto bytes = msg.serialize();
        CHECK(!bytes.empty());
        bytes[0] = 255; // far past the last real RequestType value
        CHECK_THROWS(Message::deserialize(bytes), std::invalid_argument);

        // A second, less extreme out-of-range value (immediately past the
        // last valid one) throws the same way.
        auto bytes2 = msg.serialize();
        bytes2[0] = static_cast<std::uint8_t>(kAllRequestTypes.size()); // one past the last valid value
        CHECK_THROWS(Message::deserialize(bytes2), std::invalid_argument);

        // The boundary itself: the actual last valid value must still
        // decode successfully (no off-by-one in the fix).
        auto bytes3 = msg.serialize();
        bytes3[0] = static_cast<std::uint8_t>(kAllRequestTypes.size() - 1);
        CHECK_NOTHROW(Message::deserialize(bytes3));
    }

    // ---- 7b. FIXED, same class of gap: an out-of-range FieldDtype byte
    // inside body.field_dtypes must also throw at deserialize(), not get
    // silently cast into an undefined enum value. ----
    {
        MessageBody body;
        body.fields = {"only_field"};
        body.field_dtypes = {FieldDtype::Float32}; // one real, valid byte to corrupt
        auto msg = Message::create(RequestType::DECLARE_SCHEMA, "s", body);
        auto bytes = msg.serialize();

        // Locate the FieldDtype byte by re-encoding a message with a
        // sentinel value and diffing, rather than hand-counting offsets
        // through every preceding field (partition_id, sample_ids, fields'
        // length-prefix+contents) -- robust to the wire layout shifting.
        MessageBody sentinel_body = body;
        sentinel_body.field_dtypes = {static_cast<FieldDtype>(0xAB)}; // not a real valid value, just a marker
        auto sentinel_msg = Message::create(RequestType::DECLARE_SCHEMA, "s", sentinel_body);
        sentinel_msg.request_id = msg.request_id; // keep everything else identical
        sentinel_msg.timestamp = msg.timestamp;
        auto sentinel_bytes = sentinel_msg.serialize();

        CHECK(bytes.size() == sentinel_bytes.size());
        std::size_t diff_offset = bytes.size();
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            if (bytes[i] != sentinel_bytes[i]) {
                diff_offset = i;
                break;
            }
        }
        CHECK(diff_offset < bytes.size()); // found the byte that encodes field_dtypes[0]

        auto corrupted = bytes;
        corrupted[diff_offset] = 255; // out of range for FieldDtype
        CHECK_THROWS(Message::deserialize(corrupted), std::invalid_argument);

        // The real, valid message must still decode cleanly (sanity check
        // that the diff-based offset-finding above actually found the
        // right byte, not a false positive elsewhere in the buffer).
        CHECK_NOTHROW(Message::deserialize(bytes));
    }

    // Deserializing a completely empty buffer must throw, independent of
    // the truncation sweep above (which never tests a message whose own
    // length is 0).
    { CHECK_THROWS(Message::deserialize({}), std::invalid_argument); }

    // ---- 8. Message::create: request_id uniqueness and a sane timestamp. ----
    {
        std::set<std::string> ids;
        double first_ts = -1;
        double last_ts = -1;
        constexpr int kN = 200;
        for (int i = 0; i < kN; ++i) {
            auto msg = Message::create(RequestType::HANDSHAKE, "s", MessageBody{});
            ids.insert(msg.request_id);
            if (i == 0) first_ts = msg.timestamp;
            last_ts = msg.timestamp;
            // A real wall-clock epoch-seconds timestamp for "now" (this
            // code was written after 2020-01-01, well before 2100-01-01) --
            // guards against a placeholder/zeroed/garbage value.
            CHECK(msg.timestamp > 1577836800.0);  // 2020-01-01
            CHECK(msg.timestamp < 4102444800.0);  // 2100-01-01
        }
        CHECK(ids.size() == static_cast<std::size_t>(kN)); // all pairwise distinct
        CHECK(last_ts >= first_ts); // non-decreasing across a tight loop
    }

    // ---- serialize_header()/deserialize_split(): the split wire form
    // RpcClient.cpp/RpcServerBase.cpp use to avoid an extra full-payload
    // copy (see docs/TransferQueue-Benchmark.md's Section 8/9). Confirms
    // three things nothing above exercises: the header alone decodes to an
    // EMPTY body.payload regardless of what the real payload was (it must
    // never leak into the header), the header is actually smaller than the
    // combined buffer when a real payload is present (otherwise there's no
    // point to any of this), and reassembling via deserialize_split()
    // produces a Message equivalent in every field to what the original
    // single-buffer serialize()/deserialize() round trip gives. ----
    {
        MessageBody body = make_fully_populated_body();
        CHECK(!body.payload.empty()); // fixture includes a real payload
        auto msg = Message::create(RequestType::GET_META_RESPONSE, "sender-1", body, "receiver-1");

        auto full_bytes = msg.serialize();
        auto header_bytes = msg.serialize_header();
        CHECK(header_bytes.size() < full_bytes.size());
        CHECK(full_bytes.size() - header_bytes.size() == body.payload.size());

        // Header alone, decoded via the ordinary deserialize(): payload
        // must be empty, not the real (non-empty) payload leaking through.
        auto header_only = Message::deserialize(header_bytes);
        CHECK(header_only.body.payload.empty());
        CHECK(header_only.request_type == RequestType::GET_META_RESPONSE);
        CHECK(header_only.sender_id == "sender-1");

        // Reassembled via deserialize_split(): equivalent to the original
        // single-buffer round trip in every field, including the payload
        // this time.
        auto reassembled = Message::deserialize_split(header_bytes, body.payload);
        CHECK(reassembled.request_type == msg.request_type);
        CHECK(reassembled.sender_id == msg.sender_id);
        CHECK(reassembled.receiver_id.has_value() && *reassembled.receiver_id == *msg.receiver_id);
        CHECK(reassembled.request_id == msg.request_id);
        CHECK(reassembled.timestamp == msg.timestamp);
        check_body_equal(reassembled.body, body);

        // An empty payload round-trips through the split form too (the
        // common case -- most request types never carry one at all).
        MessageBody empty_body;
        empty_body.partition_id = "p";
        auto empty_msg = Message::create(RequestType::HANDSHAKE, "s", empty_body);
        auto empty_header = empty_msg.serialize_header();
        auto empty_reassembled = Message::deserialize_split(empty_header, {});
        CHECK(empty_reassembled.body.payload.empty());
        CHECK(empty_reassembled.body.partition_id == "p");
    }

    return tq::test::summary("MessageTest");
}
