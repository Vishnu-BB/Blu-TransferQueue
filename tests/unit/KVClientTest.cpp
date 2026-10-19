// Unit tests for KVClient (include/transferqueue/KVClient.h, src/KVClient.cpp)
// against a real split deployment: one ControllerServer + two StorageServer
// shards over real ZMQ sockets, no mocks. Covers the verl v1 usage pattern:
// tags-only prompt rows, status updates that merge tags, trajectory rows
// written by the rollout and later extended by the trainer, kv_list polling,
// kv_clear eviction. Build: `make unit_rpc_KVClientTest`.

#include "transferqueue/KVClient.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/Tensor.h"
#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/RpcClient.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

#include "TestUtils.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;

namespace {

Tensor make_i64(std::vector<std::int64_t> values) {
    Tensor t(Shape{{static_cast<std::int64_t>(values.size())}}, Dtype::Int64);
    for (std::size_t i = 0; i < values.size(); ++i) static_cast<std::int64_t*>(t.data())[i] = values[i];
    return t;
}

Tensor make_f32(float value) {
    Tensor t(Shape{{1}}, Dtype::Float32);
    static_cast<float*>(t.data())[0] = value;
    return t;
}

std::int64_t i64_at(const Tensor& t, std::size_t i) { return static_cast<const std::int64_t*>(t.data())[i]; }
float f32_at(const Tensor& t) { return static_cast<const float*>(t.data())[0]; }

tq::Tag tag_of(tq::KVClient& kv, const std::string& partition, const std::string& key) {
    for (auto& [k, tag] : kv.kv_list(partition)) {
        if (k == key) return tag;
    }
    return {};
}

struct Cluster {
    std::shared_ptr<tq::TransferQueueController> controller = std::make_shared<tq::TransferQueueController>();
    tq::ControllerServer controller_server{"ctrl", controller, std::make_shared<tq::FifoSampler>()};
    std::shared_ptr<tq::SimpleStorageManager> storage0 = std::make_shared<tq::SimpleStorageManager>();
    std::shared_ptr<tq::SimpleStorageManager> storage1 = std::make_shared<tq::SimpleStorageManager>();
    tq::StorageServer shard0{"shard-0", 0, storage0};
    tq::StorageServer shard1{"shard-1", 1, storage1};
    std::string address;

    Cluster() {
        address = controller_server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient announcer("announcer", address);
        announcer.announce_shard(0, shard0.start("tcp://127.0.0.1:0"));
        announcer.announce_shard(1, shard1.start("tcp://127.0.0.1:0"));
    }
    ~Cluster() {
        shard0.stop();
        shard1.stop();
        controller_server.stop();
    }
};

} // namespace

int main() {
    // ---- verl v1 lifecycle: prompt row -> status updates -> trajectories -> trainer write-back ----
    {
        Cluster cluster;
        tq::KVClient driver("driver", cluster.address);
        tq::KVClient rollout("rollout", cluster.address);
        tq::KVClient trainer("trainer", cluster.address);

        // Driver registers prompt groups: tags only, no data.
        driver.kv_batch_put("train", {"u1", "u2"}, {},
                            {{{"is_prompt", true}, {"status", std::string("pending")}, {"global_steps", std::int64_t{0}}},
                             {{"is_prompt", true}, {"status", std::string("pending")}, {"global_steps", std::int64_t{0}}}});
        CHECK(driver.kv_list("train").size() == 2);

        // Rollout flips status from another client; other tag entries survive.
        rollout.kv_batch_put("train", {"u1"}, {}, {{{"status", std::string("running")}}});
        auto t1 = tag_of(driver, "train", "u1");
        CHECK(std::get<std::string>(t1.at("status")) == "running");
        CHECK(std::get<bool>(t1.at("is_prompt")));
        CHECK(std::get<std::int64_t>(t1.at("global_steps")) == 0);

        // Rollout writes two trajectories of u1 (data + tags).
        rollout.kv_batch_put("train", {"u1_0_0", "u1_1_0"},
                             {{{"responses", make_i64({5, 6, 7})}, {"rm_scores", make_f32(1.0f)}},
                              {{"responses", make_i64({8})}, {"rm_scores", make_f32(0.0f)}}},
                             {{{"status", std::string("success")}, {"response_len", std::int64_t{3}}},
                              {{"status", std::string("success")}, {"response_len", std::int64_t{1}}}});
        rollout.kv_batch_put("train", {"u1"}, {}, {{{"status", std::string("finished")}}});

        // Same-uid rows share a shard (group affinity) -- and it's a real shard.
        auto shard_a = cluster.controller->shard_for_sample(
            "train", cluster.controller->kv_retrieve("train", {"u1_0_0"}, false)[0]);
        auto shard_b = cluster.controller->shard_for_sample(
            "train", cluster.controller->kv_retrieve("train", {"u1_1_0"}, false)[0]);
        CHECK(shard_a == shard_b);
        CHECK(shard_a == 0 || shard_a == 1);

        // Trainer reads selected fields only.
        auto got = trainer.kv_batch_get("train", {"u1_0_0", "u1_1_0", "missing"}, {"responses"});
        CHECK(got.size() == 2);
        CHECK(got.count("missing") == 0);
        CHECK(got["u1_0_0"].size() == 1 && got["u1_0_0"].count("responses") == 1);
        CHECK(got["u1_0_0"]["responses"].numel() == 3 && i64_at(got["u1_0_0"]["responses"], 2) == 7);
        CHECK(i64_at(got["u1_1_0"]["responses"], 0) == 8);

        // Trainer writes a new column into the same rows (partial-row write).
        trainer.kv_batch_put("train", {"u1_0_0", "u1_1_0"},
                             {{{"advantages", make_f32(0.5f)}}, {{"advantages", make_f32(-0.5f)}}}, {});
        auto all = trainer.kv_batch_get("train", {"u1_0_0"}, {});
        CHECK(all["u1_0_0"].size() == 3);  // responses + rm_scores + advantages
        CHECK(f32_at(all["u1_0_0"]["advantages"]) == 0.5f);
        CHECK(f32_at(all["u1_0_0"]["rm_scores"]) == 1.0f);

        // Prompt row with no data: known key -> empty record.
        auto prompt_only = trainer.kv_batch_get("train", {"u2"}, {"responses"});
        CHECK(prompt_only.count("u2") == 1 && prompt_only["u2"].empty());

        // Eviction: clear the group; metadata and shard data both go.
        std::size_t bytes_before = cluster.storage0->current_bytes() + cluster.storage1->current_bytes();
        CHECK(bytes_before > 0);
        driver.kv_clear("train", {"u1", "u1_0_0", "u1_1_0"});
        CHECK(driver.kv_list("train").size() == 1);
        CHECK(trainer.kv_batch_get("train", {"u1_0_0"}, {}).empty());
        CHECK(cluster.storage0->current_bytes() + cluster.storage1->current_bytes() == 0);

        // Partitions are isolated: same key in "val" is a separate row.
        driver.kv_batch_put("val", {"u2"}, {}, {{{"status", std::string("pending")}}});
        CHECK(driver.kv_list("val").size() == 1);
        CHECK(tag_of(driver, "train", "u2").count("is_prompt") == 1);
        CHECK(tag_of(driver, "val", "u2").count("is_prompt") == 0);
    }

    // ---- argument validation ----
    {
        Cluster cluster;
        tq::KVClient kv("kv", cluster.address);
        CHECK_THROWS(kv.kv_batch_put("train", {"a", "b"}, {{{"x", make_f32(1)}}}, {}), std::invalid_argument);
        CHECK_THROWS(kv.kv_batch_put("train", {"a"}, {}, {{}, {}}), std::invalid_argument);
        CHECK_THROWS(kv.kv_batch_put("train", {"a", "b"}, {{{"x", make_f32(1)}}, {{"y", make_f32(1)}}}, {}),
                     std::invalid_argument);
        CHECK_NOTHROW(kv.kv_batch_put("train", {}, {}, {}));
        CHECK(kv.kv_list("nope").empty());
        CHECK(kv.kv_batch_get("nope", {"a"}, {}).empty());
        CHECK_NOTHROW(kv.kv_clear("nope", {"a"}));
    }

    // ---- no shards registered: tags-only works, a data write fails loudly ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        tq::ControllerServer server("ctrl", controller, std::make_shared<tq::FifoSampler>());
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::KVClient kv("kv", addr);
        CHECK_NOTHROW(kv.kv_batch_put("train", {"a"}, {}, {{{"status", std::string("pending")}}}));
        CHECK_THROWS(kv.kv_batch_put("train", {"a"}, {{{"x", make_f32(1)}}}, {}), std::runtime_error);
        server.stop();
    }

    // ---- client timeout doesn't desync later replies ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>();
        tq::ControllerServer server("ctrl", controller, std::make_shared<tq::FifoSampler>());
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::TransferQueueRpcClient client("c", addr, /*timeout_ms=*/2000);
        CHECK_NOTHROW(client.handshake());
        CHECK(client.kv_list("train").empty());
        client.kv_update("train", client.kv_retrieve("train", {"k"}, true).sample_ids, {}, {}, {},
                         {{{"n", std::int64_t{1}}}});
        auto listed = client.kv_list("train");
        CHECK(listed.size() == 1 && listed[0].first == "k");
        server.stop();
    }

    // ---- total_storage_size: new keys past the limit are refused; clearing frees room ----
    {
        auto controller = std::make_shared<tq::TransferQueueController>(/*max_kv_samples=*/3);
        tq::ControllerServer server("ctrl", controller, std::make_shared<tq::FifoSampler>());
        std::string addr = server.start("tcp://127.0.0.1:0");
        tq::KVClient kv("kv", addr);
        CHECK_NOTHROW(kv.kv_batch_put("train", {"a", "b"}, {}, {{}, {}}));
        CHECK_NOTHROW(kv.kv_batch_put("val", {"c"}, {}, {{}}));  // limit spans partitions
        CHECK_THROWS(kv.kv_batch_put("train", {"d"}, {}, {{}}), std::runtime_error);
        CHECK(controller->kv_sample_count() == 3);
        CHECK_NOTHROW(kv.kv_batch_put("train", {"a"}, {}, {{{"status", std::string("x")}}}));  // existing key: fine
        kv.kv_clear("train", {"a"});
        CHECK_NOTHROW(kv.kv_batch_put("train", {"d", "d"}, {}, {{}, {}}));  // duplicate counts once
        CHECK(controller->kv_sample_count() == 3);
        server.stop();
    }

    // ---- non-tensor fields ride as bytes tensors through a real shard ----
    {
        Cluster cluster;
        tq::KVClient kv("kv", cluster.address);
        const std::string extra = R"({"reward_extra_info": {"acc": 1.0}, "data_source": "gsm8k"})";
        kv.kv_batch_put("train", {"u_0_0", "u_1_0"},
                        {{{"extra_fields", tq::bytes_tensor(extra)}, {"rm_scores", make_f32(1.0f)}},
                         {{"extra_fields", tq::bytes_tensor("")}, {"rm_scores", make_f32(0.0f)}}},
                        {});
        auto got = kv.kv_batch_get("train", {"u_0_0", "u_1_0"}, {"extra_fields"});
        CHECK(tq::tensor_bytes(got["u_0_0"]["extra_fields"]) == extra);
        CHECK(tq::tensor_bytes(got["u_1_0"]["extra_fields"]).empty());
        CHECK_THROWS(tq::tensor_bytes(make_f32(1.0f)), std::invalid_argument);
    }

    // ---- checkpoint: save, wipe, restore into a FRESH deployment ----
    {
        const std::string dir = "build/kvclient_test_ckpt";
        std::filesystem::remove_all(dir);
        {
            Cluster cluster;
            tq::KVClient kv("kv", cluster.address);
            kv.kv_batch_put("train", {"u1", "u2"}, {},
                            {{{"is_prompt", true}, {"status", std::string("finished")}, {"global_steps", std::int64_t{4}}},
                             {{"is_prompt", true}, {"status", std::string("running")}}});
            kv.kv_batch_put("train", {"u1_0_0", "u1_1_0"},
                            {{{"responses", make_i64({1, 2})}, {"extra", tq::bytes_tensor("hi")}},
                             {{"responses", make_i64({3})}, {"extra", tq::bytes_tensor("yo")}}},
                            {{{"status", std::string("success")}}, {{"status", std::string("success")}}});
            kv.kv_batch_put("val", {"v1"}, {{{"responses", make_i64({9})}}}, {{{"is_prompt", false}}});
            cluster.controller->advance_version();
            cluster.controller->advance_version();
            kv.save_checkpoint(dir);
            CHECK(std::filesystem::exists(dir + "/controller.tqck"));
            CHECK(std::filesystem::exists(dir + "/shard-0.tqsh") && std::filesystem::exists(dir + "/shard-1.tqsh"));
        }
        Cluster fresh;  // brand-new controller + shards, empty
        tq::KVClient kv("kv2", fresh.address);
        kv.kv_batch_put("train", {"junk"}, {{{"responses", make_i64({0})}}}, {});  // replaced by the load
        kv.load_checkpoint(dir);

        CHECK(kv.kv_list("train").size() == 4);
        CHECK(kv.kv_list("val").size() == 1);
        CHECK(fresh.controller->current_version() == 2);
        auto t1 = tag_of(kv, "train", "u1");
        CHECK(std::get<std::string>(t1.at("status")) == "finished");
        CHECK(std::get<std::int64_t>(t1.at("global_steps")) == 4);
        auto rows = kv.kv_batch_get("train", {"u1_0_0", "u1_1_0", "junk"}, {});
        CHECK(rows.count("junk") == 0);
        CHECK(rows["u1_0_0"]["responses"].numel() == 2 && i64_at(rows["u1_0_0"]["responses"], 1) == 2);
        CHECK(tq::tensor_bytes(rows["u1_1_0"]["extra"]) == "yo");
        CHECK(i64_at(kv.kv_batch_get("val", {"v1"}, {"responses"})["v1"]["responses"], 0) == 9);

        // The restored store is fully live: new keys get fresh ids (no
        // collision with restored ones), partial-row writes land on the
        // restored row's shard, clear works.
        kv.kv_batch_put("train", {"u3"}, {{{"responses", make_i64({7})}}}, {});
        CHECK(i64_at(kv.kv_batch_get("train", {"u1_0_0"}, {"responses"})["u1_0_0"]["responses"], 0) == 1);
        kv.kv_batch_put("train", {"u1_0_0"}, {{{"advantages", make_f32(0.25f)}}}, {});
        CHECK(kv.kv_batch_get("train", {"u1_0_0"}, {})["u1_0_0"].size() == 3);
        kv.kv_clear("train", {"u1", "u1_0_0", "u1_1_0"});
        CHECK(kv.kv_list("train").size() == 2);

        CHECK_THROWS(kv.load_checkpoint("build/does_not_exist"), std::runtime_error);
        CHECK(kv.kv_list("train").size() == 2);  // a failed load changes nothing
        std::filesystem::remove_all(dir);
    }

    // ---- metrics: one Prometheus scrape covers the controller and every shard ----
    {
        Cluster cluster;
        tq::KVClient kv("kv", cluster.address);
        kv.kv_batch_put("train", {"u1_0_0", "u2_0_0"}, {{{"x", make_f32(1)}}, {{"x", make_f32(2)}}}, {});
        int port = cluster.controller_server.start_metrics(0);
        CHECK(port > 0);

        std::string body;
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            const char req[] = "GET /metrics HTTP/1.0\r\n\r\n";
            ::send(fd, req, sizeof(req) - 1, 0);
            char buf[4096];
            for (ssize_t n; (n = ::recv(fd, buf, sizeof(buf), 0)) > 0;) body.append(buf, static_cast<std::size_t>(n));
        }
        ::close(fd);
        auto has = [&](const std::string& s) { return body.find(s) != std::string::npos; };
        CHECK(has("HTTP/1.0 200 OK"));
        CHECK(has("tq_samples 2\n"));
        CHECK(has("tq_partition_samples{partition=\"train\"} 2\n"));
        CHECK(has("tq_shards_registered 2\n"));
        CHECK(has("tq_shard_up{shard=\"0\"} 1\n") && has("tq_shard_up{shard=\"1\"} 1\n"));
        CHECK(has("tq_shard_rows{shard="));
        CHECK(has("tq_requests_total{server=\"controller\",type=\"KV_RETRIEVE\"} "));
        CHECK(has("type=\"PUT_DATA\"} "));

        // A dead shard shows as down, the scrape still succeeds.
        cluster.shard1.stop();
        auto text = cluster.controller_server.render_metrics();
        CHECK(text.find("tq_shard_up{shard=\"1\"} 0\n") != std::string::npos);
    }

    return tq::test::summary("KVClientTest");
}
