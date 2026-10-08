#include "transferqueue/StorageServer.h"

namespace tq {

StorageServer::StorageServer(std::string server_id, std::int32_t shard_index, std::shared_ptr<StorageManager> storage)
    : RpcServerBase(std::move(server_id)), shard_index_(shard_index), storage_(std::move(storage)) {}

Message StorageServer::handle_request(const Message& request) {
    const auto& body = request.body;

    switch (request.request_type) {
        case RequestType::HANDSHAKE: {
            return Message::create(RequestType::HANDSHAKE_ACK, server_id(), {}, request.sender_id);
        }

        case RequestType::PUT_DATA: {
            auto data = extract_batch(request);

            BatchMeta meta;
            meta.sample_ids.reserve(data.size());
            for (const auto& [id, record] : data) {
                (void)record;
                meta.sample_ids.push_back(id);
            }
            meta.partition_ids.assign(meta.sample_ids.size(), body.partition_id);
            meta.fields = body.fields;
            storage_->put_data(meta, data);

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::PUT_DATA_RESPONSE, server_id(), response_body, request.sender_id);
        }

        case RequestType::GET_DATA: {
            BatchMeta meta;
            meta.sample_ids = body.sample_ids;
            meta.partition_ids.assign(body.sample_ids.size(), body.partition_id);
            meta.fields = body.fields;
            auto data = storage_->get_data(meta);

            MessageBody response_body;
            response_body.success = true;
            response_body.sample_ids = body.sample_ids;
            return make_batch_message(RequestType::GET_DATA_RESPONSE, server_id(), response_body, data,
                                       request.sender_id);
        }

        case RequestType::CLEAR_DATA: {
            BatchMeta meta;
            meta.sample_ids = body.sample_ids;
            meta.partition_ids.assign(body.sample_ids.size(), body.partition_id);
            storage_->clear_data(meta);

            MessageBody response_body;
            response_body.success = true;
            return Message::create(RequestType::CLEAR_DATA_RESPONSE, server_id(), response_body, request.sender_id);
        }

        default: {
            MessageBody response_body;
            response_body.success = false;
            response_body.error_message = "StorageServer has no handler for request_type " +
                                           to_string(request.request_type);
            return Message::create(RequestType::REQUEST_ERROR, server_id(), response_body, request.sender_id);
        }
    }
}

}
