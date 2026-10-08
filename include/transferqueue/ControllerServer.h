#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/RpcServerBase.h"
#include "transferqueue/Sampler.h"

namespace tq {

class ControllerServer : public RpcServerBase {
public:
    ControllerServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                      std::shared_ptr<BaseSampler> sampler);

private:
    Message handle_request(const Message& request) override;

    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<BaseSampler> sampler_;
};

}
