// Needs MPI (mpic++, run via `mpirun -np N`) + Tensor-Implementations +
// cppzmq. Build with `make test_distributed` (NPROC=<n> to change rank
// count, default 2). No dist/communication/NCCL/DeviceMesh -- GroupRouter
// is plain hash % world_size, doesn't need any of that.
//
// Each rank runs its own local TransferQueueServer. GroupRouter decides,
// per group_id, which rank's server should own a sample -- proving
// write-time group-affinity routing actually works across real separate
// processes, not just the pure routing-function logic already covered in
// test_scaffold_smoke.cpp.
//
// Known simplification: ranks bind to a deterministic port
// (kBasePort + rank) on localhost rather than exchanging real addresses via
// MPI_Allgather -- correct for a same-node mpirun test; a real multi-node
// deployment needs actual address discovery (not built yet, see
// docs/PHASE_4.md).

#include <cassert>
#include <iostream>
#include <string>

#include <mpi.h>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/GroupRouter.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

constexpr int kBasePort = 57000;

Tensor make_scalar_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

std::string address_for_rank(int rank) { return "tcp://127.0.0.1:" + std::to_string(kBasePort + rank); }

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    tq::GroupRouter router(world_size);

    // Determinism: every rank must compute the same target for the same
    // group_id -- the whole scheme breaks otherwise, since a producer and a
    // (future) reader agree on where a group lives without talking first.
    for (const std::string& group_id : {"group-0", "group-1", "group-42"}) {
        int target = router.target_rank(group_id);
        assert(target >= 0 && target < world_size);
    }

    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();
    tq::TransferQueueServer server("rank-" + std::to_string(rank), controller, storage, sampler);
    server.start(address_for_rank(rank));

    MPI_Barrier(MPI_COMM_WORLD); // every rank's server must be up before anyone connects

    // Pick a group_id that actually routes off rank 0 when world_size > 1,
    // so the write provably crosses a process boundary rather than landing
    // on rank 0 by hash coincidence.
    std::string group_id = "grpo-group-0";
    if (world_size > 1) {
        for (int i = 0; i < 100; ++i) {
            std::string candidate = "grpo-group-" + std::to_string(i);
            if (router.target_rank(candidate) != 0) {
                group_id = candidate;
                break;
            }
        }
    }
    int owner = router.target_rank(group_id);

    if (rank == 0) {
        tq::TransferQueueRpcClient client("writer", address_for_rank(owner));
        tq::Record record;
        record["reward"] = make_scalar_f32(1.5f);
        client.put("rollout@0", {"reward"}, {{100, record}});
    }

    MPI_Barrier(MPI_COMM_WORLD); // the write must land before the owner reads it

    if (rank == owner) {
        tq::TransferQueueRpcClient client("reader", address_for_rank(owner));
        auto got = client.get("rollout@0", {"reward"}, "trainer", /*batch_size=*/10);
        assert(got.size() == 1);
        assert(static_cast<float*>(got.at(100).at("reward").data())[0] == 1.5f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    server.stop();

    if (rank == 0) {
        std::cout << "test_distributed_smoke: PASS (world_size=" << world_size << ", group '" << group_id
                  << "' routed to rank " << owner << ")\n";
    }

    MPI_Finalize();
    return 0;
}
