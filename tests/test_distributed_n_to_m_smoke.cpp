// Phase 7: a real N:M rollout:trainer topology, not just Phase 4's 1:1 one
// (test_distributed_smoke.cpp). Needs MPI (mpic++, `mpirun -np N`) +
// Tensor-Implementations + cppzmq. Build with `make test_distributed_n_to_m`
// (N_ROLLOUT/N_TRAINER/GROUPS_PER_ROLLOUT/N_SAMPLES_PER_PROMPT override the
// defaults).
//
// Rank layout, via MPI_Comm_split (color = role):
//   rank 0                        -- ControllerServer (metadata only)
//   ranks 1 .. N_ROLLOUT           -- one StorageServer each (a shard),
//                                     shard_index = rank - 1
//   ranks N_ROLLOUT+1 .. N_ROLLOUT+N_TRAINER -- pure trainer readers
//
// Each rollout rank "owns" (round-robin) a subset of prompt groups --
// group g's n_samples_per_prompt sample ids are a contiguous run
// [g*n, g*n+n), satisfying GRPOGroupNSampler's contract. For each group it
// owns, it picks the TARGET SHARD via GroupRouter(N_ROLLOUT) -- which may
// be a *different* rollout rank than the one producing it, proving writes
// genuinely cross process boundaries by routing decision, not just by
// construction -- writes directly to that shard's StorageServer, then
// NOTIFY_DATA_UPDATEs the ControllerServer.
//
// All trainer ranks share one task_name ("trainer") against the same
// ControllerServer, so Controller's per-task consumption tracking
// naturally splits the total batch across them -- each GET_META call
// (via GRPOGroupNSampler) returns one complete group or nothing, and no
// two trainer ranks ever receive the same group. That guarantee is only
// correct because of Controller::select_and_consume's atomicity (see
// docs/PHASE_7.md) -- a real race was found and fixed while building this
// test: concurrent GET_META calls under RpcServerBase's thread pool used
// to be able to both select the same ready group before either marked it
// consumed.
//
// Known simplification (same one Phase 4's test already made, for the same
// reason): ranks bind to a deterministic port (kBasePort + rank) on
// localhost rather than exchanging real addresses via MPI_Bcast/Allgather
// -- correct for a same-node mpirun test; real multi-node address
// discovery is still the Ray-launcher/HANDSHAKE mechanism described in
// docs/PHASE_5.md, not something this test needs to re-solve.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/GroupRouter.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

constexpr int kBasePort = 58000;

enum class Role { Controller, Rollout, Trainer };

std::string address_for_rank(int rank) { return "tcp://127.0.0.1:" + std::to_string(kBasePort + rank); }

Tensor make_scalar_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

float expected_reward(std::int64_t group_index, std::int64_t completion_index) {
    return static_cast<float>(group_index) * 100.0f + static_cast<float>(completion_index);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int n_rollout = argc > 1 ? std::atoi(argv[1]) : 2;
    int n_trainer = argc > 2 ? std::atoi(argv[2]) : 3;
    int groups_per_rollout = argc > 3 ? std::atoi(argv[3]) : 3;
    int n_samples_per_prompt = argc > 4 ? std::atoi(argv[4]) : 4;

    int expected_world_size = 1 + n_rollout + n_trainer;
    if (world_size != expected_world_size) {
        if (world_rank == 0) {
            std::cerr << "test_distributed_n_to_m_smoke: expected " << expected_world_size << " ranks (1 controller + "
                      << n_rollout << " rollout + " << n_trainer << " trainer), got " << world_size << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    Role role;
    int role_index = 0; // 0-based index within this rank's role group
    if (world_rank == 0) {
        role = Role::Controller;
    } else if (world_rank <= n_rollout) {
        role = Role::Rollout;
        role_index = world_rank - 1;
    } else {
        role = Role::Trainer;
        role_index = world_rank - 1 - n_rollout;
    }

    // MPI_Comm_split: one sub-communicator per role. Used here to prove the
    // split actually produced the right group sizes/ranks -- the real
    // cross-rank work below (writes, reads) goes over MPI_COMM_WORLD via
    // TCP/ZMQ, same as Phase 4's test; role_comm itself isn't needed as a
    // transport, just as the thing Phase 7 was asked to demonstrate.
    MPI_Comm role_comm;
    MPI_Comm_split(MPI_COMM_WORLD, static_cast<int>(role), world_rank, &role_comm);
    int role_comm_size = 0;
    int role_comm_rank = 0;
    MPI_Comm_size(role_comm, &role_comm_size);
    MPI_Comm_rank(role_comm, &role_comm_rank);
    switch (role) {
        case Role::Controller:
            assert(role_comm_size == 1);
            assert(role_comm_rank == 0);
            break;
        case Role::Rollout:
            assert(role_comm_size == n_rollout);
            assert(role_comm_rank == role_index);
            break;
        case Role::Trainer:
            assert(role_comm_size == n_trainer);
            assert(role_comm_rank == role_index);
            break;
    }

    const std::string partition = "rollout@n_to_m";
    const std::string task_name = "trainer";
    const std::string controller_address = address_for_rank(0);

    std::unique_ptr<tq::ControllerServer> controller_server;
    std::unique_ptr<tq::StorageServer> storage_server;

    if (role == Role::Controller) {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto sampler = std::make_shared<tq::GRPOGroupNSampler>(static_cast<std::size_t>(n_samples_per_prompt));
        controller_server = std::make_unique<tq::ControllerServer>("controller", controller, sampler);
        controller_server->start(controller_address);
    } else if (role == Role::Rollout) {
        auto storage = std::make_shared<tq::SimpleStorageManager>();
        storage_server =
            std::make_unique<tq::StorageServer>("shard-" + std::to_string(role_index), role_index, storage);
        storage_server->start(address_for_rank(world_rank));
    }

    MPI_Barrier(MPI_COMM_WORLD); // every server must be up before anyone connects

    if (role == Role::Rollout) {
        tq::TransferQueueRpcClient announcer("shard-" + std::to_string(role_index), controller_address);
        announcer.announce_shard(role_index, address_for_rank(world_rank));
    }

    MPI_Barrier(MPI_COMM_WORLD); // every shard must be registered before any write/read

    // ---- Rollout ranks: produce a round-robin subset of groups, route
    // each one to a shard via GroupRouter, write, then notify. ----
    int total_groups = n_rollout * groups_per_rollout;
    tq::GroupRouter router(n_rollout);

    if (role == Role::Rollout) {
        for (int group_index = 0; group_index < total_groups; ++group_index) {
            if (group_index % n_rollout != role_index) {
                continue; // not this rank's group to produce
            }
            std::string group_id = "group-" + std::to_string(group_index);
            int target_shard = router.target_rank(group_id);
            assert(target_shard >= 0 && target_shard < n_rollout);

            tq::SampleId base = static_cast<tq::SampleId>(group_index) * static_cast<tq::SampleId>(n_samples_per_prompt);
            std::unordered_map<tq::SampleId, tq::Record> batch;
            std::vector<tq::SampleId> ids;
            for (int c = 0; c < n_samples_per_prompt; ++c) {
                tq::SampleId id = base + static_cast<tq::SampleId>(c);
                tq::Record r;
                r["reward"] = make_scalar_f32(expected_reward(group_index, c));
                batch[id] = r;
                ids.push_back(id);
            }

            tq::TransferQueueRpcClient writer("rollout-" + std::to_string(role_index),
                                               address_for_rank(1 + target_shard));
            // group_id here is a no-op (StorageServer has no Controller to
            // record it against) but passed anyway, matching real caller
            // intent -- the actual grouping record happens via
            // notify_data_update() below, which does reach the Controller.
            writer.put(partition, {"reward"}, batch, group_id);

            tq::TransferQueueRpcClient notifier("rollout-" + std::to_string(role_index), controller_address);
            notifier.notify_data_update(partition, ids, {"reward"}, {tq::FieldDtype::Float32}, target_shard,
                                         group_id);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // every group must be fully written+notified before any trainer reads

    // ---- Trainer ranks: pull complete groups until none are left. Every
    // trainer shares task_name "trainer" against the same controller, so
    // Controller::select_and_consume's atomicity is what guarantees no two
    // trainer ranks ever receive the same group. ----
    int local_fetched = 0;
    int local_groups = 0;
    if (role == Role::Trainer) {
        tq::TransferQueueRpcClient reader("trainer-" + std::to_string(role_index), controller_address);
        while (true) {
            auto meta = reader.get_meta(partition, {"reward"}, task_name,
                                         static_cast<std::size_t>(n_samples_per_prompt));
            if (meta.sample_ids.empty()) {
                break;
            }
            assert(meta.sample_ids.size() == static_cast<std::size_t>(n_samples_per_prompt));
            ++local_groups;

            for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
                tq::SampleId id = meta.sample_ids[i];
                std::int32_t shard = meta.sample_shard_indices[i];
                assert(shard >= 0 && shard < n_rollout);
                const std::string& shard_address = meta.shard_addresses.at(shard);

                tq::TransferQueueRpcClient data_reader("trainer-" + std::to_string(role_index), shard_address);
                auto data = data_reader.get_data(partition, {id}, {"reward"});
                assert(data.count(id) == 1);

                std::int64_t group_index = static_cast<std::int64_t>(id) / n_samples_per_prompt;
                std::int64_t completion_index = static_cast<std::int64_t>(id) % n_samples_per_prompt;
                float got = static_cast<float*>(data.at(id).at("reward").data())[0];
                assert(got == expected_reward(group_index, completion_index));
            }
            local_fetched += static_cast<int>(meta.sample_ids.size());
        }
    }

    // Every rank contributes to one Allreduce (non-trainer ranks contribute
    // 0) -- the global total must equal exactly what was produced: no
    // sample lost, and (the thing the atomicity fix prevents) none
    // double-delivered to two different trainer ranks.
    int global_fetched = 0;
    MPI_Allreduce(&local_fetched, &global_fetched, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int global_groups = 0;
    MPI_Allreduce(&local_groups, &global_groups, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int expected_total_samples = total_groups * n_samples_per_prompt;
    assert(global_fetched == expected_total_samples);
    assert(global_groups == total_groups);

    MPI_Barrier(MPI_COMM_WORLD);
    if (role == Role::Rollout) {
        storage_server->stop();
    } else if (role == Role::Controller) {
        controller_server->stop();
    }
    MPI_Comm_free(&role_comm);

    if (world_rank == 0) {
        std::cout << "test_distributed_n_to_m_smoke: PASS (1 controller + " << n_rollout << " rollout + " << n_trainer
                  << " trainer ranks, " << total_groups << " groups x " << n_samples_per_prompt
                  << " samples = " << expected_total_samples << " samples, all delivered exactly once)\n";
    }

    MPI_Finalize();
    return 0;
}
