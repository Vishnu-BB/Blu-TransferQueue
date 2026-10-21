// Checkpoint/restore tests for the KV store (KVClient::save_checkpoint / load_checkpoint ->
// ControllerServer SAVE/LOAD_CHECKPOINT -> TransferQueueController::save/load_kv_checkpoint +
// StorageServer -> SimpleStorageManager::save/load). Real ZMQ servers, no mocks.
// Build: `make unit_rpc_CheckpointTest`.
//
// Covers what KVClientTest's single round-trip case does not: realistic size, every dtype and
// shape compared byte-for-byte, saving over an existing checkpoint, empty stores, ids and the
// sample limit after restore, restoring onto a different shard count, corrupted files, and
// writes racing a save.

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/KVClient.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

#include "TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

namespace fs = std::filesystem;

std::mt19937_64 g_rng(12345);

Tensor random_tensor(Dtype dtype, std::vector<int64_t> dims) {
    Tensor t(Shape{dims}, dtype);
    auto* p = static_cast<std::uint8_t*>(t.data());
    for (size_t i = 0; i < t.nbytes(); ++i) p[i] = static_cast<std::uint8_t>(g_rng());
    if (dtype == Dtype::Bool) {
        for (size_t i = 0; i < t.nbytes(); ++i) p[i] &= 1;  // valid bools only
    }
    return t;
}

bool same_tensor(const Tensor& a, const Tensor& b) {
    if (a.dtype() != b.dtype() || a.ndim() != b.ndim() || a.nbytes() != b.nbytes()) return false;
    for (int64_t d = 0; d < a.ndim(); ++d) {
        if (a.shape().dims[d] != b.shape().dims[d]) return false;
    }
    return std::memcmp(a.data(), b.data(), a.nbytes()) == 0;
}

bool same_record(const tq::Record& a, const tq::Record& b) {
    if (a.size() != b.size()) return false;
    for (const auto& [field, t] : a) {
        auto it = b.find(field);
        if (it == b.end() || !same_tensor(t, it->second)) return false;
    }
    return true;
}

struct Cluster {
    std::shared_ptr<tq::TransferQueueController> controller;
    std::unique_ptr<tq::ControllerServer> controller_server;
    std::vector<std::shared_ptr<tq::SimpleStorageManager>> storages;
    std::vector<std::unique_ptr<tq::StorageServer>> shards;
    std::string address;

    explicit Cluster(int num_shards, std::size_t max_samples = 0)
        : controller(std::make_shared<tq::TransferQueueController>(max_samples)) {
        controller_server = std::make_unique<tq::ControllerServer>("ctrl", controller, std::make_shared<tq::FifoSampler>());
        address = controller_server->start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient announcer("announcer", address);
        for (int i = 0; i < num_shards; ++i) {
            storages.push_back(std::make_shared<tq::SimpleStorageManager>());
            shards.push_back(std::make_unique<tq::StorageServer>("shard-" + std::to_string(i), i, storages.back()));
            announcer.announce_shard(i, shards.back()->start("tcp://127.0.0.1:0"));
        }
    }
    ~Cluster() {
        for (auto& s : shards) s->stop();
        controller_server->stop();
    }
};

// Everything a store holds, as seen through the public KV API.
struct Snapshot {
    std::map<std::string, std::map<std::string, tq::Tag>> tags;      // partition -> key -> tag
    std::map<std::string, std::map<std::string, tq::Record>> rows;   // partition -> key -> fields
};

Snapshot snapshot(tq::KVClient& kv, const std::vector<std::string>& partitions) {
    Snapshot s;
    for (const auto& p : partitions) {
        std::vector<std::string> keys;
        for (auto& [key, tag] : kv.kv_list(p)) {
            keys.push_back(key);
            s.tags[p][key] = tag;
        }
        for (auto& [key, record] : kv.kv_batch_get(p, keys, {})) s.rows[p][key] = record;
    }
    return s;
}

bool same_snapshot(const Snapshot& a, const Snapshot& b) {
    if (a.tags != b.tags) return false;
    if (a.rows.size() != b.rows.size()) return false;
    for (const auto& [p, rows] : a.rows) {
        auto it = b.rows.find(p);
        if (it == b.rows.end() || it->second.size() != rows.size()) return false;
        for (const auto& [key, record] : rows) {
            auto r = it->second.find(key);
            if (r == it->second.end() || !same_record(record, r->second)) return false;
        }
    }
    return true;
}

const std::vector<std::string> kPartitions = {"train", "val", "extra"};

// verl-shaped content: prompt rows (tags + prompt fields) and n trajectories per prompt with a
// spread of dtypes/shapes, in three partitions. Returns the number of keys written.
int populate(tq::KVClient& kv, int groups_per_partition, int n) {
    int keys = 0;
    for (const auto& p : kPartitions) {
        for (int g = 0; g < groups_per_partition; ++g) {
            const std::string uid = p + "-u" + std::to_string(g);
            kv.kv_batch_put(p, {uid},
                            {{{"prompt_token_ids", random_tensor(Dtype::Int32, {7 + g % 5})},
                              {"ground_truth", tq::bytes_tensor("answer " + std::to_string(g))}}},
                            {{{"is_prompt", true},
                              {"status", std::string(g % 3 == 0 ? "running" : "finished")},
                              {"global_steps", std::int64_t{g % 9}},
                              {"score", 0.5 * g},
                              {"note", std::string("p") + std::to_string(g)}}});
            ++keys;
            std::vector<std::string> traj_keys;
            std::vector<tq::Record> rows;
            std::vector<tq::Tag> tags;
            for (int s = 0; s < n; ++s) {
                const int64_t len = 3 + (g + s) % 11;
                traj_keys.push_back(uid + "_" + std::to_string(s) + "_0");
                rows.push_back({{"responses", random_tensor(Dtype::Int64, {len})},
                                {"rollout_log_probs", random_tensor(Dtype::Float32, {len})},
                                {"hidden", random_tensor(Dtype::Bfloat16, {2, len})},
                                {"response_mask", random_tensor(Dtype::Bool, {len})},
                                {"rm_scores", random_tensor(Dtype::Float64, {1})},
                                {"ids16", random_tensor(Dtype::UInt16, {len})},
                                {"tiny", random_tensor(Dtype::Int8, {2, 2, 2})},
                                {"extra_fields", tq::bytes_tensor("{\"acc\": " + std::to_string(s) + "}")}});
                tags.push_back({{"is_prompt", false},
                                {"seq_len", std::int64_t{len}},
                                {"min_global_steps", std::int64_t{g % 9}}});
            }
            kv.kv_batch_put(p, traj_keys, rows, tags);
            keys += n;
        }
    }
    return keys;
}

std::string fresh_dir(const std::string& name) {
    const std::string dir = "build/ckpt_test_" + name;
    fs::remove_all(dir);
    return dir;
}

bool has_tmp_files(const std::string& dir) {
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() == ".tmp") return true;
    }
    return false;
}

} // namespace

int main() {
    // ---- 1. Realistic round trip: 3 partitions x 120 groups x (1 prompt + 4 trajectories), 8 dtypes,
    // 1-3D shapes, byte fields, every tag type -- restored into a brand-new deployment and compared
    // byte-for-byte through the public API. ----
    {
        const std::string dir = fresh_dir("roundtrip");
        Snapshot before;
        int written = 0;
        {
            Cluster a(2);
            tq::KVClient kv("a", a.address);
            written = populate(kv, 120, 4);
            for (int i = 0; i < 7; ++i) a.controller->advance_version();
            before = snapshot(kv, kPartitions);
            kv.save_checkpoint(dir);
            CHECK(fs::exists(dir + "/controller.tqck"));
            CHECK(fs::exists(dir + "/shard-0.tqsh") && fs::exists(dir + "/shard-1.tqsh"));
            CHECK(!has_tmp_files(dir));
            CHECK(fs::file_size(dir + "/shard-0.tqsh") > 0 && fs::file_size(dir + "/shard-1.tqsh") > 0);
        }
        Cluster b(2);
        tq::KVClient kv("b", b.address);
        kv.load_checkpoint(dir);
        const Snapshot after = snapshot(kv, kPartitions);
        CHECK(written == 3 * 120 * 5);
        CHECK(b.controller->kv_sample_count() == static_cast<std::size_t>(written));
        CHECK(b.controller->current_version() == 7);
        CHECK(same_snapshot(before, after));
        // Loading the same checkpoint again is idempotent.
        kv.load_checkpoint(dir);
        CHECK(same_snapshot(before, snapshot(kv, kPartitions)));
    }

    // ---- 2. The restored store is fully live: fresh ids never collide with restored ones, the
    // sample limit counts restored keys, partial-row writes and clears work. ----
    {
        const std::string dir = fresh_dir("live");
        Snapshot before;
        {
            Cluster a(2);
            tq::KVClient kv("a", a.address);
            populate(kv, 10, 2);  // 3 x 10 x 3 = 90 keys
            before = snapshot(kv, kPartitions);
            kv.save_checkpoint(dir);
        }
        Cluster b(2, /*max_samples=*/92);
        tq::KVClient kv("b", b.address);
        kv.load_checkpoint(dir);
        CHECK(b.controller->kv_sample_count() == 90);
        CHECK_NOTHROW(kv.kv_batch_put("train", {"new-a", "new-b"}, {{{"x", random_tensor(Dtype::Float32, {3})}},
                                                                    {{"x", random_tensor(Dtype::Float32, {3})}}},
                                      {}));
        CHECK_THROWS(kv.kv_batch_put("train", {"new-c"}, {}, {{}}), std::runtime_error);  // 93 > 92
        CHECK(same_snapshot(before, [&] {  // the new keys didn't overwrite any restored row
            Snapshot s = snapshot(kv, kPartitions);
            s.tags["train"].erase("new-a"); s.tags["train"].erase("new-b");
            s.rows["train"].erase("new-a"); s.rows["train"].erase("new-b");
            return s;
        }()));
        auto adv = random_tensor(Dtype::Float32, {1});
        kv.kv_batch_put("train", {"train-u1_0_0"}, {{{"advantages", adv}}}, {});
        auto row = kv.kv_batch_get("train", {"train-u1_0_0"}, {});
        CHECK(row["train-u1_0_0"].size() == 9 && same_tensor(row["train-u1_0_0"]["advantages"], adv));
        kv.kv_clear("train", {"train-u1", "train-u1_0_0", "train-u1_1_0"});
        CHECK(b.controller->kv_sample_count() == 89);
    }

    // ---- 3. Saving over an existing checkpoint replaces it; no .tmp files left behind. ----
    {
        const std::string dir = fresh_dir("overwrite");
        Cluster a(2);
        tq::KVClient kv("a", a.address);
        populate(kv, 5, 2);
        kv.save_checkpoint(dir);
        kv.kv_clear("train", {"train-u0", "train-u0_0_0", "train-u0_1_0"});
        kv.kv_batch_put("val", {"later"}, {}, {{{"is_prompt", true}}});
        const Snapshot second = snapshot(kv, kPartitions);
        kv.save_checkpoint(dir);
        CHECK(!has_tmp_files(dir));
        Cluster b(2);
        tq::KVClient kv2("b", b.address);
        kv2.load_checkpoint(dir);
        CHECK(same_snapshot(second, snapshot(kv2, kPartitions)));
    }

    // ---- 4. Empty stores: saving an empty store works, and loading it empties a populated one. ----
    {
        const std::string dir = fresh_dir("empty");
        {
            Cluster a(2);
            tq::KVClient kv("a", a.address);
            CHECK_NOTHROW(kv.save_checkpoint(dir));
        }
        Cluster b(2);
        tq::KVClient kv("b", b.address);
        populate(kv, 3, 2);
        kv.load_checkpoint(dir);
        CHECK(b.controller->kv_sample_count() == 0);
        for (const auto& p : kPartitions) CHECK(kv.kv_list(p).empty());
        CHECK(b.storages[0]->current_bytes() == 0 && b.storages[1]->current_bytes() == 0);
    }

    // ---- 5. Restoring onto a different shard count must be refused, never half-restored. ----
    {
        const std::string dir = fresh_dir("shards");
        {
            Cluster a(2);
            tq::KVClient kv("a", a.address);
            populate(kv, 20, 2);
            kv.save_checkpoint(dir);
        }
        {
            Cluster fewer(1);  // checkpoint has data on shard 1, deployment has only shard 0
            tq::KVClient kv("b", fewer.address);
            bool threw = false;
            try {
                kv.load_checkpoint(dir);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            CHECK(threw);  // FAILS today: loads shard 0 only and reports success
            if (!threw) {
                // Show the damage: keys listed in metadata whose data is unreachable.
                int unreachable = 0;
                for (const auto& p : kPartitions) {
                    std::vector<std::string> keys;
                    for (auto& [key, tag] : kv.kv_list(p)) keys.push_back(key);
                    try {
                        kv.kv_batch_get(p, keys, {});
                    } catch (const std::exception&) {
                        ++unreachable;
                    }
                }
                std::printf("  [shard mismatch] load into 1 shard reported success; %d/%zu partitions "
                            "unreadable afterwards\n", unreachable, kPartitions.size());
            }
        }
        {
            Cluster more(3);  // no shard-2 file: refused up front
            tq::KVClient kv("c", more.address);
            CHECK_THROWS(kv.load_checkpoint(dir), std::runtime_error);
        }
    }

    // ---- 6. A corrupted file must fail the load AND leave the store exactly as it was. ----
    for (const std::string victim : {"shard-0.tqsh", "shard-1.tqsh", "controller.tqck"}) {
        const std::string dir = fresh_dir("corrupt");
        {
            Cluster a(2);
            tq::KVClient kv("a", a.address);
            populate(kv, 15, 2);
            kv.save_checkpoint(dir);
        }
        fs::resize_file(dir + "/" + victim, fs::file_size(dir + "/" + victim) / 2);  // truncate

        Cluster b(2);
        tq::KVClient kv("b", b.address);
        // Enough distinct uids that BOTH shards hold live data before the failed load.
        for (int i = 0; i < 20; ++i) {
            kv.kv_batch_put("train", {"keep" + std::to_string(i)}, {{{"x", random_tensor(Dtype::Int64, {4})}}},
                            {{{"is_prompt", true}, {"status", std::string("pending")}}});
        }
        CHECK(b.storages[0]->current_bytes() > 0 && b.storages[1]->current_bytes() > 0);
        const Snapshot before = snapshot(kv, kPartitions);
        const std::size_t bytes0 = b.storages[0]->current_bytes(), bytes1 = b.storages[1]->current_bytes();
        CHECK_THROWS(kv.load_checkpoint(dir), std::runtime_error);
        const bool api_unchanged = same_snapshot(before, snapshot(kv, kPartitions));
        const bool shards_unchanged = b.storages[0]->current_bytes() == bytes0 && b.storages[1]->current_bytes() == bytes1;
        CHECK(api_unchanged);     // a failed load must not leave a half-restored store...
        CHECK(shards_unchanged);  // ...not even in shard storage the API can't currently reach
        if (!api_unchanged || !shards_unchanged) {
            std::printf("  [corrupt %s] load failed but the store was modified: api_unchanged=%d "
                        "shard bytes %zu->%zu, %zu->%zu\n", victim.c_str(), api_unchanged, bytes0,
                        b.storages[0]->current_bytes(), bytes1, b.storages[1]->current_bytes());
        }
    }

    // ---- 7. Writes racing a save: no crash, no torn file; the restored store is self-consistent
    // (every listed key with data can be read). ----
    {
        const std::string dir = fresh_dir("race");
        Cluster a(2);
        tq::KVClient seed("seed", a.address);
        populate(seed, 300, 4);  // big enough that each save takes a while
        std::atomic<bool> stop{false};
        std::atomic<int> puts{0}, clears{0};
        std::thread writer([&] {
            tq::KVClient w("writer", a.address);
            for (int i = 0; !stop; ++i) {
                const std::string key = "race-" + std::to_string(i);
                w.kv_batch_put("train", {key}, {{{"x", random_tensor(Dtype::Float32, {64})}}}, {{{"i", std::int64_t{i}}}});
                ++puts;
                if (i > 0) {  // clear every earlier key soon after writing it
                    w.kv_clear("train", {"race-" + std::to_string(i - 1)});
                    ++clears;
                }
            }
        });
        tq::KVClient saver("saver", a.address);
        for (int i = 0; i < 20; ++i) CHECK_NOTHROW(saver.save_checkpoint(dir));
        stop = true;
        writer.join();
        std::printf("  [race] %d puts and %d clears ran during 20 saves\n", puts.load(), clears.load());

        Cluster b(2);
        tq::KVClient kv("b", b.address);
        CHECK_NOTHROW(kv.load_checkpoint(dir));
        // The real inconsistency is metadata that says a field was PRODUCED while the restored shard
        // lacks it (the trainer would be handed a row without its tensors). A key the writer had only
        // created when the save ran -- no fields produced yet, no data -- is a consistent state, like
        // a tags-only prompt row; it is counted separately.
        int listed = 0, dangling = 0, created_only = 0;
        for (const auto& p : kPartitions) {
            std::vector<std::string> keys;
            for (auto& [key, tag] : kv.kv_list(p)) keys.push_back(key);
            listed += static_cast<int>(keys.size());
            const auto ids = b.controller->kv_retrieve(p, keys, /*create=*/false);
            const auto produced_x = b.controller->ready_indexes(p, {"x"}, "checkpoint-test");
            const std::set<tq::SampleId> has_x(produced_x.begin(), produced_x.end());
            auto rows = kv.kv_batch_get(p, keys, {});
            for (size_t k = 0; k < keys.size(); ++k) {
                if (keys[k].rfind("race-", 0) != 0) continue;
                const bool claims_x = has_x.count(ids[k]) != 0;
                if (claims_x && rows[keys[k]].count("x") == 0) ++dangling;
                if (!claims_x && rows[keys[k]].empty()) ++created_only;
            }
        }
        std::printf("  [race] restored %d keys: %d with produced fields missing their data, %d created-but-unwritten "
                    "(consistent)\n", listed, dangling, created_only);
        CHECK(dangling == 0);
    }

    return tq::test::summary("CheckpointTest");
}
