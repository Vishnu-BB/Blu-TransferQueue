#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/RpcServerBase.h"
#include "transferqueue/Sampler.h"

namespace tq {

// Phase 5's control-plane half of the Controller/StorageManager split (see
// docs/PHASE_5.md): holds a TransferQueueController and nothing else -- no
// StorageManager, so it never touches actual tensor data. Each
// StorageServer (shard) announces itself once via HANDSHAKE (shard_index +
// its own address); after that, GET_META_RESPONSE bundles the resolved
// shard address for every sample it returns, and CLEAR_PARTITION fans out
// CLEAR_DATA to every shard a cleared partition actually used (via a
// transient TransferQueueRpcClient per shard -- clear_partition is not a
// hot path, so a pooled/persistent connection isn't worth building yet).
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
