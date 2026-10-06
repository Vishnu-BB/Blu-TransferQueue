// Needs Tensor-Implementations (Record holds a real OwnTensor::Tensor) but
// not dist/communication/NCCL -- build with `make test_tensor`, no mpirun
// needed. Exercises StorageManager, Client, and the Message payload
// (de)serialization bridge with real tensors instead of Phase 0/1's
// std::any/opaque-bytes placeholders.
//
// Run with: ./test_tensor_smoke

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include "core/Tensor.h"
#include "transferqueue/Client.h"
#include "transferqueue/Controller.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

Tensor make_scalar_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

Tensor make_token_ids(const std::vector<std::int64_t>& ids) {
    Tensor t(Shape{{static_cast<std::int64_t>(ids.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        data[i] = ids[i];
    }
    return t;
}

} // namespace

int main() {
    // StorageManager: put then get round-trips a real tensor's bytes.
    {
        tq::SimpleStorageManager storage;
        tq::BatchMeta write_meta;
        write_meta.sample_ids = {1};
        write_meta.partition_ids = {"rollout@0"};
        write_meta.fields = {"reward"};
        storage.put_data(write_meta, {{1, {{"reward", make_scalar_f32(2.0f)}}}});

        tq::BatchMeta read_meta;
        read_meta.sample_ids = {1};
        read_meta.partition_ids = {"rollout@0"};
        auto rows = storage.get_data(read_meta);
        assert(rows.size() == 1);
        Tensor& got = rows.at(1).at("reward");
        assert(got.dtype() == Dtype::Float32);
        assert(got.numel() == 1);
        assert(static_cast<float*>(got.data())[0] == 2.0f);
    }

    // Client: full put -> get round trip with GRPO-shaped fields
    // (variable-length token ids + a scalar reward).
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto store = std::make_shared<tq::SimpleStorageManager>();
        auto fifo = std::make_shared<tq::FifoSampler>();
        tq::TransferQueueClient client(controller, store, fifo);

        tq::Record record;
        record["response_ids"] = make_token_ids({101, 202, 303});
        record["reward"] = make_scalar_f32(3.5f);
        client.put("rollout@0", {"response_ids", "reward"}, {{7, record}});

        auto got = client.get("rollout@0", {"response_ids", "reward"}, "trainer", /*batch_size=*/10);
        assert(got.size() == 1);
        Tensor& response_ids = got.at(7).at("response_ids");
        assert(response_ids.dtype() == Dtype::Int64);
        assert(response_ids.numel() == 3);
        assert(static_cast<std::int64_t*>(response_ids.data())[1] == 202);
        assert(static_cast<float*>(got.at(7).at("reward").data())[0] == 3.5f);

        // Consumed by "trainer" -- a second get for the same task sees nothing left.
        auto got_again = client.get("rollout@0", {"response_ids", "reward"}, "trainer", /*batch_size=*/10);
        assert(got_again.empty());
    }

    // Schema validation (Option B) through the real Client, with real
    // tensors: declare_schema() once, then put() with a matching dtype
    // succeeds, and put() with a conflicting dtype for an already-declared
    // field throws before anything is written.
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        auto store = std::make_shared<tq::SimpleStorageManager>();
        auto fifo = std::make_shared<tq::FifoSampler>();
        tq::TransferQueueClient client(controller, store, fifo);

        client.declare_schema("rollout@schema", {{"reward", tq::FieldDtype::Float32}});

        tq::Record good;
        good["reward"] = make_scalar_f32(1.5f);
        client.put("rollout@schema", {"reward"}, {{1, good}}); // matches -- no throw

        tq::Record bad;
        Tensor wrong_dtype(Shape{{1}}, Dtype::Float64);
        bad["reward"] = wrong_dtype;
        bool threw = false;
        try {
            client.put("rollout@schema", {"reward"}, {{2, bad}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        // The rejected write didn't partially land -- sample 2 was never
        // stored at all.
        auto got = client.get("rollout@schema", {"reward"}, "trainer", /*batch_size=*/10);
        assert(got.size() == 1);
        assert(got.count(2) == 0);
    }

    // serialize_batch/deserialize_batch: the Message::payload bridge --
    // reuses OwnTensor::save_tensor/load_tensor, round-trips shape+dtype+data.
    {
        std::unordered_map<tq::SampleId, tq::Record> data;
        data[1]["log_probs"] = make_token_ids({1, 2, 3}); // reusing as a stand-in int tensor
        data[2]["reward"] = make_scalar_f32(-1.25f);

        auto bytes = tq::serialize_batch(data);
        auto decoded = tq::deserialize_batch(bytes);

        assert(decoded.size() == 2);
        Tensor& lp = decoded.at(1).at("log_probs");
        assert(lp.numel() == 3);
        assert(static_cast<std::int64_t*>(lp.data())[2] == 3);
        assert(static_cast<float*>(decoded.at(2).at("reward").data())[0] == -1.25f);
    }

    // GPU-resident tensors (previously completely untested -- every prior
    // test here used CPU tensors only).
    {
        // StorageManager/Client never serialize a tensor on this path --
        // put_data/get_data just store and return the Tensor object itself
        // -- so the GPU device should survive untouched, unlike the wire
        // path below.
        tq::SimpleStorageManager storage;
        Tensor gpu_tensor = make_scalar_f32(9.5f).to_cuda(0);
        assert(gpu_tensor.is_cuda());

        tq::BatchMeta write_meta;
        write_meta.sample_ids = {1};
        write_meta.partition_ids = {"rollout@0"};
        write_meta.fields = {"reward"};
        storage.put_data(write_meta, {{1, {{"reward", gpu_tensor}}}});

        tq::BatchMeta read_meta;
        read_meta.sample_ids = {1};
        read_meta.partition_ids = {"rollout@0"};
        Tensor& got = storage.get_data(read_meta).at(1).at("reward");
        assert(got.is_cuda()); // device preserved -- no serialization happened
        assert(got.to_cpu().data() != nullptr);
        assert(static_cast<float*>(got.to_cpu().data())[0] == 9.5f);

        // serialize_batch/deserialize_batch DOES go through
        // OwnTensor::save_tensor/load_tensor (the Message::payload bridge,
        // Phase 2) -- save_tensor copies to host before writing, but
        // load_tensor always reconstructs on CPU regardless of the
        // original device (confirmed by reading Serialization.cpp directly,
        // not assumed). So a GPU tensor that round-trips through the wire
        // format comes back on CPU: values preserved, device not. This is
        // the path TransferQueueServer/RpcClient actually use (Phase 3) --
        // asserting it explicitly here so this behavior is documented by a
        // test, not just discoverable by reading someone else's source.
        std::unordered_map<tq::SampleId, tq::Record> gpu_data;
        gpu_data[1]["reward"] = gpu_tensor;
        auto bytes = tq::serialize_batch(gpu_data);
        auto decoded = tq::deserialize_batch(bytes);
        Tensor& round_tripped = decoded.at(1).at("reward");
        assert(!round_tripped.is_cuda());
        assert(round_tripped.is_cpu());
        assert(static_cast<float*>(round_tripped.data())[0] == 9.5f);
    }

    // Backpressure: put_data blocks once capacity_bytes is exceeded, and
    // unblocks once clear_data frees enough space. A 4-byte float32 scalar
    // is exactly 4 bytes; capacity 4 means "room for one such tensor."
    {
        tq::SimpleStorageManager storage(/*capacity_bytes=*/4);

        tq::BatchMeta meta;
        meta.sample_ids = {1};
        meta.partition_ids = {"rollout@0"};
        meta.fields = {"reward"};
        storage.put_data(meta, {{1, {{"reward", make_scalar_f32(1.0f)}}}});
        assert(storage.current_bytes() == 4);

        std::atomic<bool> second_put_done{false};
        std::thread writer([&] {
            tq::BatchMeta meta2;
            meta2.sample_ids = {2};
            meta2.partition_ids = {"rollout@0"};
            meta2.fields = {"reward"};
            // At capacity -- blocks until the main thread clears sample 1.
            storage.put_data(meta2, {{2, {{"reward", make_scalar_f32(2.0f)}}}});
            second_put_done = true;
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(!second_put_done.load()); // still blocked -- no space freed yet

        tq::BatchMeta clear_meta;
        clear_meta.sample_ids = {1};
        clear_meta.partition_ids = {"rollout@0"};
        storage.clear_data(clear_meta); // frees 4 bytes, wakes the writer

        writer.join();
        assert(second_put_done.load());
        assert(storage.current_bytes() == 4); // sample 1's 4 bytes freed, sample 2's 4 bytes taken
    }

    std::cout << "test_tensor_smoke: PASS\n";
    return 0;
}
