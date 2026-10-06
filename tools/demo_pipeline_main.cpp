// Human-readable, end-to-end walkthrough of the TransferQueue pipeline using
// dummy GRPO rollout data -- NOT a correctness test (test_rpc_smoke.cpp
// already covers that). This prints every stage's input/output so a reader
// unfamiliar with the code can see exactly what moves through the system,
// in what shape, and when. Captured verbatim into docs/PIPELINE_WALKTHROUGH.md.
//
// Stands in for the real rollout engine (writer) and trainer (reader) --
// neither exists yet (see docs/PIPELINE_WALKTHROUGH.md's blockers section)
// -- using a real TransferQueueServer + TransferQueueRpcClient over an
// actual loopback socket, exactly like a real deployment would.
//
// Every "read" stage below uses get_meta() exactly once, never followed by
// a second get()/get_meta() for the same logical batch -- both map to the
// same GET_META request on the wire, and the server marks whatever the
// sampler selects as consumed on that same call. A second call for "the
// same" batch would actually sample again from whatever's left, not
// re-return the same selection. So each stage's "what the trainer received"
// section prints the already-known dummy values for the ids get_meta()
// reports as selected, rather than issuing a redundant round trip.
//
// Run with: ./build/demo_pipeline

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/GRPOGroupNSampler.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

struct Row {
    tq::SampleId id;
    std::string prompt_label;
    std::vector<std::int64_t> prompt_ids;
    std::vector<std::int64_t> response_ids;
    std::vector<float> log_probs;
    float reward;
};

Tensor make_i64_array(const std::vector<std::int64_t>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Int64);
    auto* data = static_cast<std::int64_t*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

Tensor make_f32_array(const std::vector<float>& values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Float32);
    auto* data = static_cast<float*>(t.data());
    for (std::size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return t;
}

std::string i64_list_str(const std::vector<std::int64_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(v[i]);
    }
    return out + "]";
}

std::string f32_list_str(const std::vector<float>& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

void print_row(const Row& row) {
    std::cout << "    sample " << row.id << " (prompt " << row.prompt_label << "): "
              << "prompt_ids=" << i64_list_str(row.prompt_ids) << "  "
              << "response_ids=" << i64_list_str(row.response_ids) << "  "
              << "log_probs=" << f32_list_str(row.log_probs) << "  "
              << "reward=" << std::fixed << std::setprecision(2) << row.reward << "\n";
}

void print_ready(tq::TransferQueueController& controller, const std::string& partition, const std::string& task) {
    auto ready = controller.ready_indexes(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task);
    std::cout << "  Controller::ready_indexes(task=" << task << ") = [";
    for (std::size_t i = 0; i < ready.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << ready[i];
    }
    std::cout << "]\n";
}

std::string id_list_str(const std::vector<tq::SampleId>& ids) {
    std::string out = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(ids[i]);
    }
    return out + "]";
}

std::string version_list_str(const std::vector<std::int64_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(v[i]);
    }
    return out + "]";
}

void header(const std::string& title) { std::cout << "\n=== " << title << " ===\n"; }

} // namespace

int main() {
    const std::string partition = "rollout@demo";
    const std::string task = "trainer";
    const std::size_t n_samples_per_prompt = 4;

    // ---- Dummy GRPO rollout data: 2 prompts, n_samples_per_prompt = 4 ----
    std::vector<Row> rows = {
        {0, "A", {11, 12, 13}, {101, 102, 103}, {-0.10f, -0.20f, -0.15f}, 0.10f},
        {1, "A", {11, 12, 13}, {104, 105, 106}, {-0.05f, -0.30f, -0.25f}, 0.40f},
        {2, "A", {11, 12, 13}, {107, 108, 109}, {-0.40f, -0.10f, -0.05f}, 0.20f},
        {3, "A", {11, 12, 13}, {110, 111, 112}, {-0.15f, -0.15f, -0.10f}, 0.90f},
        {4, "B", {21, 22, 23}, {201, 202, 203}, {-0.20f, -0.10f, -0.05f}, 0.30f},
        {5, "B", {21, 22, 23}, {204, 205, 206}, {-0.10f, -0.25f, -0.30f}, 0.60f},
        {6, "B", {21, 22, 23}, {207, 208, 209}, {-0.05f, -0.05f, -0.10f}, 0.00f},
        {7, "B", {21, 22, 23}, {210, 211, 212}, {-0.30f, -0.20f, -0.15f}, 0.80f},
    };

    header("STAGE 0: Dummy GRPO rollout data (input)");
    std::cout << "  partition = \"" << partition << "\", n_samples_per_prompt = " << n_samples_per_prompt << "\n\n";
    for (const auto& row : rows) print_row(row);

    // ---- Real server + client over an actual loopback socket ----
    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::GRPOGroupNSampler>(n_samples_per_prompt);
    tq::TransferQueueServer server("demo-server", controller, storage, sampler);
    std::string address = server.start("tcp://127.0.0.1:0");
    tq::TransferQueueRpcClient client("demo-client", address);

    client.declare_schema(partition, {{"prompt_ids", tq::FieldDtype::Int64},
                                       {"response_ids", tq::FieldDtype::Int64},
                                       {"log_probs", tq::FieldDtype::Float32},
                                       {"reward", tq::FieldDtype::Float32}});

    auto write_row = [&](const Row& row) {
        tq::Record r;
        r["prompt_ids"] = make_i64_array(row.prompt_ids);
        r["response_ids"] = make_i64_array(row.response_ids);
        r["log_probs"] = make_f32_array(row.log_probs);
        r["reward"] = make_f32_array({row.reward});
        // group_id is the real grouping key GRPOGroupNSampler now uses
        // (not id adjacency) -- the prompt label ("A"/"B") already
        // uniquely identifies the group here.
        client.put(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, {{row.id, r}}, row.prompt_label);
    };
    auto row_by_id = [&](tq::SampleId id) -> const Row& {
        for (const auto& row : rows) {
            if (row.id == id) return row;
        }
        throw std::out_of_range("unknown sample id");
    };

    header("STAGE 1: Rollout engine PUT_DATAs prompt A's full group (samples 0-3)");
    for (int i = 0; i <= 3; ++i) write_row(rows[i]);
    std::cout << "  wrote samples [0, 1, 2, 3]\n";
    print_ready(*controller, partition, task);

    header("STAGE 2: Rollout engine PUT_DATAs 3 of prompt B's 4 samples (4, 5, 6) -- sample 7 still in flight");
    for (int i = 4; i <= 6; ++i) write_row(rows[i]);
    std::cout << "  wrote samples [4, 5, 6]\n";
    print_ready(*controller, partition, task);

    header("STAGE 3: Trainer calls GET_META (batch_size=4) -- GRPOGroupNSampler groups ready ids by group_id");
    std::cout << "  group \"A\" = {0,1,2,3}, 4 members -> complete, selected\n";
    std::cout << "  group \"B\" = {4,5,6}, only 3 members -> incomplete, excluded, left ready\n";
    auto meta1 = client.get_meta(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task,
                                  n_samples_per_prompt);
    std::cout << "  GET_META_RESPONSE.sample_ids       = " << id_list_str(meta1.sample_ids) << "\n";
    std::cout << "  GET_META_RESPONSE.sample_versions  = " << version_list_str(meta1.sample_versions) << "\n";
    std::cout << "  GET_META_RESPONSE.current_version  = " << meta1.current_version << "\n";
    std::cout << "  Data the trainer now has for those ids (bundled in the same response on the colocated\n"
              << "  server, since it has both Controller and StorageManager):\n";
    for (auto id : meta1.sample_ids) print_row(row_by_id(id));

    header("STAGE 4: Trainer calls GET_META again (batch_size=4) -- prompt B still incomplete");
    print_ready(*controller, partition, task);
    auto meta2 = client.get_meta(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task,
                                  n_samples_per_prompt);
    std::cout << "  only {4,5,6} ready -- not a complete group of 4 -> GRPOGroupNSampler returns nothing\n";
    std::cout << "  GET_META_RESPONSE.sample_ids = " << id_list_str(meta2.sample_ids) << "  (empty, as expected)\n";

    header("STAGE 5: A weight sync happens -- trainer calls ADVANCE_VERSION");
    auto new_version = client.advance_version();
    std::cout << "  current_version is now " << new_version << "\n";

    header("STAGE 6: Rollout engine PUT_DATAs the last sample of prompt B (sample 7) -- produced AFTER the sync");
    write_row(rows[7]);
    print_ready(*controller, partition, task);

    header("STAGE 7: Trainer calls GET_META again -- prompt B is now complete");
    auto meta3 = client.get_meta(partition, {"prompt_ids", "response_ids", "log_probs", "reward"}, task,
                                  n_samples_per_prompt);
    std::cout << "  GET_META_RESPONSE.sample_ids       = " << id_list_str(meta3.sample_ids) << "\n";
    std::cout << "  GET_META_RESPONSE.sample_versions  = " << version_list_str(meta3.sample_versions) << "\n";
    std::cout << "  GET_META_RESPONSE.current_version  = " << meta3.current_version << "\n";
    std::cout << "  -> samples 4,5,6 were produced BEFORE the sync (version 0); sample 7 AFTER it (version 1).\n"
              << "     Same GRPO group, two different policy versions mixed in. TransferQueue tracks and\n"
              << "     exposes this (Phase 6); it never filters it out -- the trainer decides what to do.\n";
    for (auto id : meta3.sample_ids) print_row(row_by_id(id));

    header("STAGE 8: Trainer calls CLEAR_PARTITION once both groups are consumed");
    std::size_t bytes_before = storage->current_bytes();
    client.clear_partition(partition);
    std::size_t bytes_after = storage->current_bytes();
    std::cout << "  StorageManager bytes before clear: " << bytes_before << "\n";
    std::cout << "  StorageManager bytes after clear:  " << bytes_after << "\n";

    server.stop();
    std::cout << "\nDone.\n";
    return 0;
}
