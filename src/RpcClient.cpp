#include "transferqueue/RpcClient.h"

#include <stdexcept>

#include <zmq.hpp>

#include "transferqueue/Message.h"

namespace tq {

struct TransferQueueRpcClient::Impl {
    zmq::context_t ctx;
    zmq::socket_t dealer{ctx, zmq::socket_type::dealer};
};

TransferQueueRpcClient::TransferQueueRpcClient(std::string client_id, const std::string& server_address)
    : impl_(std::make_unique<Impl>()), client_id_(std::move(client_id)) {
    impl_->dealer.set(zmq::sockopt::rcvtimeo, 5000);
    impl_->dealer.set(zmq::sockopt::linger, 0);
    impl_->dealer.connect(server_address);
}

TransferQueueRpcClient::~TransferQueueRpcClient() = default;

namespace {

// zmq::buffer()+send() always deep-copies its argument into a new ZMQ
// message before handing it to the kernel -- fine for the small header
// frame, wasteful for a payload that can be hundreds of MB (confirmed,
// measured cost: docs/TransferQueue-Benchmark.md's Section 8/9/11
// "RpcClient ZMQ send copy"). zmq_msg_init_data (the constructor below)
// instead lets ZMQ read directly out of `buffer`'s own memory, with a
// free_fn callback that releases it once ZMQ is actually done transmitting
// -- `buffer` is moved onto the heap so it outlives this call. Only worth
// it for non-trivial sizes; an empty/tiny buffer goes the normal route,
// since the heap allocation + callback indirection this needs would cost
// more than it saves.
void send_owned_buffer(zmq::socket_t& socket, std::vector<std::uint8_t> buffer, zmq::send_flags flags) {
    if (buffer.size() < 4096) {
        socket.send(zmq::buffer(buffer), flags);
        return;
    }
    auto* owner = new std::vector<std::uint8_t>(std::move(buffer));
    zmq::message_t msg(
        owner->data(), owner->size(),
        [](void*, void* hint) { delete static_cast<std::vector<std::uint8_t>*>(hint); }, owner);
    socket.send(msg, flags);
}

// Sent as two ZMQ frames (header, payload) instead of serialize()'s single
// combined buffer, so neither side pays the extra full-payload copy merging
// them together -- see Message.h's serialize_header()/deserialize_split()
// doc comment, and docs/TransferQueue-Benchmark.md's Section 8/9 bottleneck
// writeups ("Message::serialize/deserialize copying payload into/out of the
// envelope" cost). A payload-free request just sends an empty second frame
// -- negligible over loopback, and far simpler than a variable frame count.
// Takes `request` BY VALUE (every real caller passes a freshly-constructed
// temporary, see this file's ~14 call sites) so body.payload can be moved
// into send_owned_buffer() instead of copied.
Message call(TransferQueueRpcClient::Impl& impl, Message request) {
    auto header = request.serialize_header();
    send_owned_buffer(impl.dealer, std::move(header), zmq::send_flags::sndmore);
    send_owned_buffer(impl.dealer, std::move(request.body.payload), zmq::send_flags::none);

    zmq::message_t reply_header;
    if (!impl.dealer.recv(reply_header, zmq::recv_flags::none).has_value()) {
        throw std::runtime_error("TransferQueueRpcClient: timed out waiting for a reply");
    }
    zmq::message_t reply_payload;
    if (!impl.dealer.recv(reply_payload, zmq::recv_flags::none).has_value()) {
        throw std::runtime_error("TransferQueueRpcClient: timed out waiting for a reply payload");
    }

    std::vector<std::uint8_t> header_bytes(static_cast<const std::uint8_t*>(reply_header.data()),
                                            static_cast<const std::uint8_t*>(reply_header.data()) +
                                                reply_header.size());
    std::vector<std::uint8_t> payload_bytes(static_cast<const std::uint8_t*>(reply_payload.data()),
                                             static_cast<const std::uint8_t*>(reply_payload.data()) +
                                                 reply_payload.size());
    return Message::deserialize_split(header_bytes, std::move(payload_bytes));
}

void throw_if_error(const Message& response) {
    if (!response.body.success) {
        throw std::runtime_error("TransferQueueRpcClient: server reported failure: " + response.body.error_message);
    }
}

}

void TransferQueueRpcClient::handshake() {
    auto response = call(*impl_, Message::create(RequestType::HANDSHAKE, client_id_, {}));
    if (response.request_type != RequestType::HANDSHAKE_ACK) {
        throw std::runtime_error("TransferQueueRpcClient: handshake failed");
    }
}

void TransferQueueRpcClient::announce_shard(std::int32_t shard_index, const std::string& address) {
    MessageBody body;
    body.shard_index = shard_index;
    body.address = address;

    auto response = call(*impl_, Message::create(RequestType::HANDSHAKE, client_id_, body));
    if (response.request_type != RequestType::HANDSHAKE_ACK) {
        throw std::runtime_error("TransferQueueRpcClient: announce_shard failed");
    }
}

void TransferQueueRpcClient::declare_schema(const std::string& partition_id,
                                             const std::unordered_map<std::string, FieldDtype>& schema) {
    MessageBody body;
    body.partition_id = partition_id;
    for (const auto& [field, dtype] : schema) {
        body.fields.push_back(field);
        body.field_dtypes.push_back(dtype);
    }

    auto response = call(*impl_, Message::create(RequestType::DECLARE_SCHEMA, client_id_, body));
    throw_if_error(response);
}

void TransferQueueRpcClient::put(const std::string& partition_id, const std::vector<std::string>& fields,
                                  const std::unordered_map<SampleId, Record>& data, const std::string& group_id) {
    MessageBody body;
    body.partition_id = partition_id;
    body.fields = fields;
    body.group_id = group_id;

    auto response = call(*impl_, make_batch_message(RequestType::PUT_DATA, client_id_, body, data));
    throw_if_error(response);
}

std::unordered_map<SampleId, Record> TransferQueueRpcClient::get(const std::string& partition_id,
                                                                   const std::vector<std::string>& fields,
                                                                   const std::string& task_name,
                                                                   std::size_t batch_size) {
    MessageBody body;
    body.partition_id = partition_id;
    body.fields = fields;
    body.task_name = task_name;
    body.batch_size = static_cast<std::uint32_t>(batch_size);

    auto response = call(*impl_, Message::create(RequestType::GET_META, client_id_, body));
    throw_if_error(response);
    return extract_batch(std::move(response));
}

std::unordered_map<SampleId, Record> TransferQueueRpcClient::get_data(const std::string& partition_id,
                                                                        const std::vector<SampleId>& sample_ids,
                                                                        const std::vector<std::string>& fields) {
    MessageBody body;
    body.partition_id = partition_id;
    body.sample_ids = sample_ids;
    body.fields = fields;

    auto response = call(*impl_, Message::create(RequestType::GET_DATA, client_id_, body));
    throw_if_error(response);
    return extract_batch(std::move(response));
}

void TransferQueueRpcClient::clear_data(const std::string& partition_id, const std::vector<SampleId>& sample_ids) {
    MessageBody body;
    body.partition_id = partition_id;
    body.sample_ids = sample_ids;

    auto response = call(*impl_, Message::create(RequestType::CLEAR_DATA, client_id_, body));
    throw_if_error(response);
}

void TransferQueueRpcClient::clear_partition(const std::string& partition_id, bool clear_consumption) {
    MessageBody body;
    body.partition_id = partition_id;
    body.flag = clear_consumption;

    auto response = call(*impl_, Message::create(RequestType::CLEAR_PARTITION, client_id_, body));
    throw_if_error(response);
}

void TransferQueueRpcClient::reset_consumption(const std::string& partition_id,
                                                const std::optional<std::string>& task_name) {
    MessageBody body;
    body.partition_id = partition_id;
    body.task_name = task_name.value_or("");

    auto response = call(*impl_, Message::create(RequestType::RESET_CONSUMPTION, client_id_, body));
    throw_if_error(response);
}

void TransferQueueRpcClient::notify_data_update(const std::string& partition_id,
                                                 const std::vector<SampleId>& sample_ids,
                                                 const std::vector<std::string>& fields,
                                                 const std::vector<FieldDtype>& field_dtypes,
                                                 std::int32_t shard_index, const std::string& group_id) {
    MessageBody body;
    body.partition_id = partition_id;
    body.sample_ids = sample_ids;
    body.fields = fields;
    body.field_dtypes = field_dtypes;
    body.shard_index = shard_index;
    body.group_id = group_id;

    auto response = call(*impl_, Message::create(RequestType::NOTIFY_DATA_UPDATE, client_id_, body));
    if (response.request_type != RequestType::NOTIFY_DATA_UPDATE_ACK) {
        throw std::runtime_error("TransferQueueRpcClient: notify_data_update failed: " + response.body.error_message);
    }
}

std::optional<std::string> TransferQueueRpcClient::get_shard_address(std::int32_t shard_index) {
    MessageBody body;
    body.shard_index = shard_index;

    auto response = call(*impl_, Message::create(RequestType::GET_SHARD_ADDRESS, client_id_, body));
    if (!response.body.success) {
        return std::nullopt;
    }
    return response.body.address;
}

TransferQueueRpcClient::MetaResult TransferQueueRpcClient::get_meta(const std::string& partition_id,
                                                                     const std::vector<std::string>& fields,
                                                                     const std::string& task_name,
                                                                     std::size_t batch_size) {
    MessageBody body;
    body.partition_id = partition_id;
    body.fields = fields;
    body.task_name = task_name;
    body.batch_size = static_cast<std::uint32_t>(batch_size);

    auto response = call(*impl_, Message::create(RequestType::GET_META, client_id_, body));
    throw_if_error(response);

    MetaResult result;
    result.sample_ids = response.body.sample_ids;
    result.sample_shard_indices = response.body.sample_shard_indices;
    for (std::size_t i = 0; i < response.body.shard_registry_indices.size(); ++i) {
        result.shard_addresses[response.body.shard_registry_indices[i]] = response.body.shard_registry_addresses[i];
    }
    result.sample_versions = response.body.sample_versions;
    result.current_version = response.body.current_version;
    return result;
}

std::int64_t TransferQueueRpcClient::advance_version() {
    auto response = call(*impl_, Message::create(RequestType::ADVANCE_VERSION, client_id_, {}));
    throw_if_error(response);
    return response.body.current_version;
}

std::vector<StrandedGroup> TransferQueueRpcClient::find_stranded_groups(const std::string& partition_id,
                                                                         const std::vector<std::string>& fields,
                                                                         const std::string& task_name,
                                                                         std::int64_t max_age_ms) {
    MessageBody body;
    body.partition_id = partition_id;
    body.fields = fields;
    body.task_name = task_name;
    body.max_age_ms = max_age_ms;

    auto response = call(*impl_, Message::create(RequestType::FIND_STRANDED_GROUPS, client_id_, body));
    throw_if_error(response);

    std::vector<StrandedGroup> result;
    result.reserve(response.body.stranded_group_ids.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < response.body.stranded_group_ids.size(); ++i) {
        std::size_t count = static_cast<std::size_t>(response.body.stranded_group_sizes[i]);
        StrandedGroup group;
        group.group_id = response.body.stranded_group_ids[i];
        group.oldest_age_ms = response.body.stranded_group_ages_ms[i];
        group.sample_ids.assign(response.body.sample_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                                 response.body.sample_ids.begin() + static_cast<std::ptrdiff_t>(offset + count));
        offset += count;
        result.push_back(std::move(group));
    }
    return result;
}

}
