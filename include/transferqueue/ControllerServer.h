#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/MetricsHttpServer.h"
#include "transferqueue/RpcServerBase.h"
#include "transferqueue/Sampler.h"

namespace tq {

class ControllerServer : public RpcServerBase {
public:
    ControllerServer(std::string server_id, std::shared_ptr<TransferQueueController> controller,
                      std::shared_ptr<BaseSampler> sampler);
    ~ControllerServer() override;

    // Serves Prometheus metrics on 0.0.0.0:port (0 = ephemeral) for the whole
    // deployment -- controller gauges plus every registered shard's, fetched
    // per scrape. Returns the bound port.
    int start_metrics(int port);
    std::string render_metrics();

private:
    Message handle_request(const Message& request) override;

    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<BaseSampler> sampler_;
    std::unique_ptr<MetricsHttpServer> metrics_;
};

}
