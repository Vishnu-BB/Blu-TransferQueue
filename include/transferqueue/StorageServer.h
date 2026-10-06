#pragma once

#include <memory>
#include <string>

#include "transferqueue/RpcServerBase.h"
#include "transferqueue/StorageManager.h"

namespace tq {

class StorageServer : public RpcServerBase {
public:
    StorageServer(std::string server_id, std::int32_t shard_index, std::shared_ptr<StorageManager> storage);

    std::int32_t shard_index() const { return shard_index_; }

private:
    Message handle_request(const Message& request) override;

    std::int32_t shard_index_;
    std::shared_ptr<StorageManager> storage_;
};

}