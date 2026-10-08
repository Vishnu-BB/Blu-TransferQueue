// Blu-TransferQueue benchmark harness -- NOT a correctness test (see
// tests/unit/ for those). Times a PUT->GET cycle for a configurable
// TensorDict shape (batch_size x seq_length float32 tensors, field_num
// fields/sample) and reports throughput in Gbps/GB/s, mirroring upstream
// TransferQueue's perftest.py so a CSV row from each can sit side by side.
// See docs/TransferQueue-Benchmark.md for the full methodology and the
// current scope limits (single-node, single-shard, CPU tensors only).
//
// Run with: ./build/benchmark [--path=rpc|in_process] [--batch_size=N]
//   [--seq_length=N] [--field_num=N] [--iterations=N] [--capacity_bytes=N]
//   [--output_csv=path]
//
// Defaults (batch_size=1024, seq_length=8192, field_num=9) match upstream's
// "small" CONFIG_MAP tier.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Client.h"
#include "transferqueue/Controller.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

struct Args {
    std::string path = "rpc"; // "rpc" is the one comparable to upstream
    std::size_t batch_size = 1024;
    std::size_t seq_length = 8192;
    std::size_t field_num = 9;
    int iterations = 6; // iteration 0 is warm-up, discarded (matches upstream)
    std::size_t capacity_bytes = 0; // 0 = unbounded, matches SimpleStorageManager's default
    std::string output_csv; // empty -> print to stdout
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq = arg.find('=');
        if (arg.rfind("--", 0) != 0 || eq == std::string::npos) continue;
        std::string key = arg.substr(2, eq - 2);
        std::string value = arg.substr(eq + 1);
        if (key == "path") args.path = value;
        else if (key == "batch_size") args.batch_size = std::stoull(value);
        else if (key == "seq_length") args.seq_length = std::stoull(value);
        else if (key == "field_num") args.field_num = std::stoull(value);
        else if (key == "iterations") args.iterations = std::stoi(value);
        else if (key == "capacity_bytes") args.capacity_bytes = std::stoull(value);
        else if (key == "output_csv") args.output_csv = value;
    }
    return args;
}

Tensor make_field_tensor(std::size_t seq_length) {
    Tensor t(Shape{{static_cast<std::int64_t>(seq_length)}}, Dtype::Float32);
    auto* data = static_cast<float*>(t.data());
    std::fill(data, data + seq_length, 1.0f);
    return t;
}

std::unordered_map<tq::SampleId, tq::Record> make_batch(std::size_t batch_size, std::size_t seq_length,
                                                          const std::vector<std::string>& fields) {
    std::unordered_map<tq::SampleId, tq::Record> data;
    data.reserve(batch_size);
    for (std::size_t i = 0; i < batch_size; ++i) {
        tq::Record record;
        for (const auto& field : fields) record[field] = make_field_tensor(seq_length);
        data[static_cast<tq::SampleId>(i)] = std::move(record);
    }
    return data;
}

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    bool use_rpc = (args.path == "rpc");

    std::vector<std::string> fields;
    for (std::size_t f = 0; f < args.field_num; ++f) fields.push_back("f" + std::to_string(f));
    std::unordered_map<std::string, tq::FieldDtype> schema;
    for (const auto& field : fields) schema[field] = tq::FieldDtype::Float32;

    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>(args.capacity_bytes);
    auto sampler = std::make_shared<tq::FifoSampler>();

    // RPC path owns these two for the process lifetime; in-process path
    // leaves them null and talks to controller/storage directly.
    std::unique_ptr<tq::TransferQueueServer> server;
    std::unique_ptr<tq::TransferQueueRpcClient> rpc_client;
    std::unique_ptr<tq::TransferQueueClient> in_process_client;

    std::function<void(const std::string&, const std::vector<std::string>&,
                        const std::unordered_map<tq::SampleId, tq::Record>&)>
        put_fn;
    std::function<std::unordered_map<tq::SampleId, tq::Record>(const std::string&, const std::vector<std::string>&,
                                                                 const std::string&, std::size_t)>
        get_fn;

    if (use_rpc) {
        server = std::make_unique<tq::TransferQueueServer>("bench-server", controller, storage, sampler);
        std::string address = server->start("tcp://127.0.0.1:0");
        rpc_client = std::make_unique<tq::TransferQueueRpcClient>("bench-client", address);
        rpc_client->declare_schema("bench", schema);
        put_fn = [&](const std::string& p, const std::vector<std::string>& f,
                      const std::unordered_map<tq::SampleId, tq::Record>& d) { rpc_client->put(p, f, d); };
        get_fn = [&](const std::string& p, const std::vector<std::string>& f, const std::string& t, std::size_t n) {
            return rpc_client->get(p, f, t, n);
        };
    } else {
        in_process_client = std::make_unique<tq::TransferQueueClient>(controller, storage, sampler);
        in_process_client->declare_schema("bench", schema);
        put_fn = [&](const std::string& p, const std::vector<std::string>& f,
                      const std::unordered_map<tq::SampleId, tq::Record>& d) { in_process_client->put(p, f, d); };
        get_fn = [&](const std::string& p, const std::vector<std::string>& f, const std::string& t, std::size_t n) {
            return in_process_client->get(p, f, t, n);
        };
    }

    double put_time_total = 0.0;
    double get_time_total = 0.0;
    int timed_iterations = 0;

    for (int iter = 0; iter < args.iterations; ++iter) {
        std::string partition = "bench@" + std::to_string(iter);
        auto data = make_batch(args.batch_size, args.seq_length, fields);

        auto t0 = std::chrono::steady_clock::now();
        put_fn(partition, fields, data);
        double put_time = seconds_since(t0);

        auto t1 = std::chrono::steady_clock::now();
        auto got = get_fn(partition, fields, "bench_task", args.batch_size);
        double get_time = seconds_since(t1);
        if (got.size() != args.batch_size) {
            std::cerr << "benchmark: expected " << args.batch_size << " samples ready, got " << got.size()
                      << " -- result below is not trustworthy\n";
        }

        // ponytail: untimed cleanup between iterations so memory doesn't grow
        // unbounded across --iterations runs; only the storage bytes are
        // freed (Controller's lightweight per-partition metadata is left to
        // accumulate -- fine at the iteration counts this harness is meant
        // for, revisit if --iterations needs to go into the thousands).
        if (use_rpc) {
            rpc_client->clear_partition(partition);
        } else {
            std::vector<tq::SampleId> ids;
            ids.reserve(args.batch_size);
            for (std::size_t i = 0; i < args.batch_size; ++i) ids.push_back(static_cast<tq::SampleId>(i));
            storage->clear_data(tq::BatchMeta(ids, std::vector<std::string>(args.batch_size, partition)));
        }

        if (iter == 0) continue; // warm-up, discarded -- matches upstream
        put_time_total += put_time;
        get_time_total += get_time;
        ++timed_iterations;
    }

    double avg_put_time = put_time_total / timed_iterations;
    double avg_get_time = get_time_total / timed_iterations;
    double total_bytes = static_cast<double>(args.field_num) * args.batch_size * args.seq_length * sizeof(float);

    // Upstream's perftest.py divides by 1024^3 (GiB), not 1e9 (decimal GB) --
    // confirmed directly against its create_test_case()/run_throughput_test()
    // (total_size_gb = bytes / (1024**3), then *8/put_time for "Gb/s"). Using
    // decimal GB here made every one of our numbers ~7.4% *lower* than a
    // like-for-like figure would be (1024^3 / 1e9 =~ 1.0737), understating
    // our own throughput relative to upstream's reported numbers -- not a
    // TransferQueue bug, a bug in this harness's own unit convention versus
    // the stated goal (docs/TransferQueue-Benchmark.md: "a CSV row from our
    // harness and a CSV row from perftest.py can sit side by side without
    // unit conversion"). Matching upstream's divisor exactly here, not just
    // documenting the discrepancy, since that's the whole point of this tool.
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    double put_gbit = (total_bytes * 8.0) / avg_put_time / kGiB;
    double get_gbit = (total_bytes * 8.0) / avg_get_time / kGiB;
    double total_gbit = (2.0 * total_bytes * 8.0) / (avg_put_time + avg_get_time) / kGiB;
    double put_gbyte = total_bytes / avg_put_time / kGiB;
    double get_gbyte = total_bytes / avg_get_time / kGiB;
    double total_gbyte = (2.0 * total_bytes) / (avg_put_time + avg_get_time) / kGiB;

    std::ostream* out = &std::cout;
    std::ofstream file;
    if (!args.output_csv.empty()) {
        file.open(args.output_csv);
        out = &file;
    }
    *out << "path,batch_size,seq_length,field_num,bytes,put_time,get_time,"
            "put_gbit_per_sec,get_gbit_per_sec,total_gbit_per_sec,"
            "put_gb_per_sec,get_gb_per_sec,total_gb_per_sec\n";
    *out << args.path << "," << args.batch_size << "," << args.seq_length << "," << args.field_num << ","
         << static_cast<std::uint64_t>(total_bytes) << "," << avg_put_time << "," << avg_get_time << "," << put_gbit
         << "," << get_gbit << "," << total_gbit << "," << put_gbyte << "," << get_gbyte << "," << total_gbyte
         << "\n";

    if (!args.output_csv.empty()) {
        std::cout << "wrote " << args.output_csv << " (" << timed_iterations << " timed iterations, path=" << args.path
                  << ")\n";
    }

    if (server) server->stop();
    return 0;
}
