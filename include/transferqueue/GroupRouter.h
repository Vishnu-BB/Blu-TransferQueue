#pragma once

#include <string>

namespace tq {

// Write-time routing: given a GRPO group id (all rollouts sharing one
// prompt, needed together for group-relative advantage normalization),
// deterministically picks which rank's local TransferQueueServer should
// receive that group's samples. The caller connects an RpcClient to that
// rank's address and puts there -- routing happens before any data moves,
// not by filtering what's already stored (see docs/PHASE_4.md).
//
// Assumes rollout and trainer share the same rank count (1:1 mapping), so
// every read is local once routed -- no cross-rank fetch logic exists here.
// Revisit if/when rollout and trainer need independent parallelism degrees.
//
// Pure function of (group_id, world_size) -- every rank must compute the
// same answer without talking to each other, which is exactly what makes
// this safe to call independently on the producer and the (eventual) local
// reader.
class GroupRouter {
public:
    explicit GroupRouter(int world_size);

    int target_rank(const std::string& group_id) const;

private:
    int world_size_;
};

}
