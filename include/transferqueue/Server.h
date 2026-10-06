#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/RpcServerBase.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

namespace tq {

class TransferQueueServer : public RpcServerBase {
public:
    TransferQueueServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                         std::shared_ptr<StorageManager> storage, std::shared_ptr<BaseSampler> sampler);

private:
    Message handle_request(const Message& request) override;

    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<StorageManager> storage_;
    std::shared_ptr<BaseSampler> sampler_;
};

}
