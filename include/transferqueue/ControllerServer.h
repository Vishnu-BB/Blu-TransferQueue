#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
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

    // Keeps a checkpoint consistent with concurrent traffic. Requests that change
    // metadata (KV_RETRIEVE with create, KV_UPDATE, KV_CLEAR, CLEAR_PARTITION,
    // NOTIFY_DATA_UPDATE) hold it shared; SAVE/LOAD_CHECKPOINT hold it exclusively
    // for their whole run, so the saved metadata and the saved shards describe
    // the same state -- e.g. a KV_CLEAR can't drop a key's data from a shard
    // after the key's metadata was already written to controller.tqck. Prefers
    // the exclusive side: once a save is waiting, new metadata changes queue
    // behind it instead of starving it. Reads never take it.
    class CheckpointGate {
    public:
        void lock_shared();
        void unlock_shared();
        void lock();
        void unlock();

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        int readers_ = 0;
        int writers_waiting_ = 0;
        bool writer_ = false;
    };
    CheckpointGate checkpoint_gate_;

    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<BaseSampler> sampler_;
    std::unique_ptr<MetricsHttpServer> metrics_;
};

}
